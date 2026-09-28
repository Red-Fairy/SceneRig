"""Common utility functions for API clients, image encoding, and model response handling."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from lib.utils._api_keys import (
    CLAUDE_API_KEY,
    CLAUDE_BASE_URL,
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
)


def _log_usage(resp: Any, chat_args: dict) -> None:
    """Append this response's exact token usage to $GRASE_USAGE_LOG (JSONL), if set.

    The API key is shared company-wide, so the console can't attribute spend to a run;
    the response's own ``usage`` field can. The runner exports GRASE_USAGE_LOG to
    ``<scene>/usage.jsonl`` — every get_model_response caller (agent rounds AND the
    preprocess VLM calls) lands in one per-run ledger. Best-effort: usage accounting
    must never fail a request that already succeeded."""
    path = os.environ.get("GRASE_USAGE_LOG")
    if not path:
        return
    try:
        u = getattr(resp, "usage", None)
        row = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # Pipeline phase (C5): "preprocess" from the runner, then the stage name from
            # root.py at each stage entry; per-stage MCP children inherit it at spawn.
            # Makes the ledger answer "which stage costs what" instead of one flat total.
            "tag": os.environ.get("GRASE_USAGE_TAG"),
            "model": chat_args.get("model"),
            "effort": chat_args.get("reasoning_effort"),
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
        }
        cr = getattr(u, "cache_read_input_tokens", None)
        cw = getattr(u, "cache_creation_input_tokens", None)
        if cr is None:
            # Tool-less OpenAI calls return Chat Completions usage, not our shim.
            details = getattr(u, "prompt_tokens_details", None)
            cr = getattr(details, "cached_tokens", None)
            cw = getattr(details, "cache_write_tokens", None)
        if cr is not None:
            row["cache_read_input_tokens"] = cr
            row["cache_creation_input_tokens"] = cw
        response_id = getattr(resp, "id", None)
        if response_id:
            row["response_id"] = response_id
        for name in ("request_id", "reasoning_tokens"):
            value = getattr(resp if name == "request_id" else u, name, None)
            if value is not None:
                row[name] = value
        diagnostics = getattr(resp, "prompt_cache_diagnostics", None)
        if diagnostics is not None:
            row["prompt_cache_diagnostics"] = (
                diagnostics.model_dump(exclude_none=True)
                if hasattr(diagnostics, "model_dump")
                else diagnostics
            )
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("usage log failed: %s", exc)


# --------------------------------------------------------------------------- #
# T2.4 — native Anthropic Messages API + prompt caching (GRASE_NATIVE_API=1)   #
# --------------------------------------------------------------------------- #
# The pipeline's internal message/tool shapes stay OpenAI-formatted everywhere
# (memory JSONs, prompt_builder, generator/verifier); this layer translates at
# the single call boundary and wraps the response in duck-typed shims, so no
# caller changes. Rationale: the OpenAI-compat endpoint cannot set
# cache_control; caching requires the native API (audit 07-28 §13 / T2.4).

_NATIVE_CLIENT = None


def _native_client():
    global _NATIVE_CLIENT
    if _NATIVE_CLIENT is None:
        import anthropic

        from lib.utils._api_keys import CLAUDE_API_KEY

        # NOTE: CLAUDE_BASE_URL is the OpenAI-COMPAT base (".../v1/") — the native
        # SDK uses its own default base; only the key is shared.
        # max_retries=0: get_model_response owns every retry (same as the Responses
        # path), so a hang costs one MODEL_ATTEMPT_TIMEOUT_S, not SDK x wrapper attempts.
        _NATIVE_CLIENT = anthropic.Anthropic(api_key=CLAUDE_API_KEY, max_retries=0)
    return _NATIVE_CLIENT


def _data_url_to_image_block(url: str) -> dict:
    head, _, data = url.partition(",")
    media = head.removeprefix("data:").split(";")[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media, "data": data},
    }


def _parts_to_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            if part.get("text", "").strip():
                blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            blocks.append(_data_url_to_image_block(part["image_url"]["url"]))
    return blocks


_ANTHROPIC_BLOCK_TYPES = {"text", "thinking", "redacted_thinking", "tool_use"}
_RESPONSES_ITEM_TYPES = {"reasoning", "message", "function_call"}


def _openai_to_anthropic(chat_args: dict) -> dict:
    """chat.completions args -> messages.create kwargs (native shapes)."""
    system_blocks: list[dict] = []
    messages: list[dict] = []
    for msg in chat_args.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_blocks.extend(_parts_to_blocks(msg.get("content")))
            continue
        if role == "assistant":
            raw = msg.get("_raw_blocks")
            if raw and not all(b.get("type") in _ANTHROPIC_BLOCK_TYPES for b in raw):
                # Provider mismatch: these are OpenAI Responses items (reasoning /
                # message / function_call), not Anthropic blocks — replaying them
                # verbatim would 400. Fall through to the compat reconstruction.
                raw = None
            if raw:
                # replay the provider's own blocks verbatim — REQUIRED so thinking
                # blocks survive tool-use continuations on the same model. One
                # exception: the pipeline only executes tool_calls[0], so PARALLEL
                # tool_use blocks beyond the answered ids must be dropped — the
                # native API 400s on any tool_use without a matching tool_result
                # (seen live, stack batch mugs4 round 1).
                answered = {
                    tc.get("id") for tc in msg.get("tool_calls") or [] if tc.get("id")
                }
                raw = [
                    b
                    for b in raw
                    if b.get("type") != "tool_use" or b.get("id") in answered
                ]
                messages.append({"role": "assistant", "content": raw})
                continue
            blocks = _parts_to_blocks(msg.get("content"))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "input": args,
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": blocks or [{"type": "text", "text": "(no output)"}],
                }
            )
            continue
        if role == "tool":
            inner = _parts_to_blocks(msg.get("content")) or [
                {"type": "text", "text": "(no output)"}
            ]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id")
                            or msg.get("id")
                            or "",
                            "content": inner,
                        }
                    ],
                }
            )
            continue
        blocks = _parts_to_blocks(msg.get("content"))
        if blocks:
            messages.append({"role": "user", "content": blocks})

    tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object"}),
        }
        for t in chat_args.get("tools") or []
    ]

    if os.environ.get("GRASE_NATIVE_CACHE", "1") != "0":
        if system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        # Sliding breakpoint: the last block BEFORE the per-round volatile tail
        # (the "[Round Budget]" message changes every request; everything before
        # it is byte-stable between W1 cuts). The 20-block lookback then gives
        # incremental hits round over round.
        for m in reversed(messages):
            first = m["content"][0] if m.get("content") else {}
            if (
                m["role"] == "user"
                and first.get("type") == "text"
                and first.get("text", "").startswith(
                    ("[Round Budget]", "OBJECT STATE (")
                )
            ):
                continue
            m["content"][-1]["cache_control"] = {"type": "ephemeral"}
            break

    out: dict = {
        "model": chat_args.get("model"),
        "max_tokens": int(chat_args.get("max_tokens") or 32000),
        "messages": messages,
    }
    if system_blocks:
        out["system"] = system_blocks
    if tools:
        out["tools"] = tools
        tc = chat_args.get("tool_choice")
        if isinstance(tc, str) and tc in ("auto", "any", "none"):
            out["tool_choice"] = {"type": tc}
            if tc != "none":
                # the loop executes ONE call per round; parallel calls would leave
                # unanswered tool_use blocks in history (see replay filter above)
                out["tool_choice"]["disable_parallel_tool_use"] = True
    eff = chat_args.get("reasoning_effort")
    if eff:
        out["extra_body"] = {"output_config": {"effort": eff}}
    return out


class _ShimFunction:
    def __init__(self, name: str, arguments: str):
        self.name, self.arguments = name, arguments


class _ShimToolCall:
    def __init__(self, block: dict):
        self.id = block.get("id")
        self.type = "function"
        self.function = _ShimFunction(
            block.get("name", ""), json.dumps(block.get("input") or {})
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _ShimMessage:
    def __init__(self, resp):
        blocks = [
            {k: v for k, v in b.model_dump().items() if v is not None}
            for b in resp.content
        ]
        self.raw_content = blocks  # replayed verbatim on the next turn
        self.content = "\n".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        ).strip()
        tcs = [_ShimToolCall(b) for b in blocks if b.get("type") == "tool_use"]
        self.tool_calls = tcs or None


class _ShimUsage:
    def __init__(self, u):
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.prompt_tokens = (getattr(u, "input_tokens", 0) or 0) + cr + cw
        self.completion_tokens = getattr(u, "output_tokens", 0) or 0
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw


class _ShimChoice:
    def __init__(self, resp):
        self.message = _ShimMessage(resp)
        self.finish_reason = resp.stop_reason


class _ShimResponse:
    def __init__(self, resp):
        self.choices = [_ShimChoice(resp)]
        self.usage = _ShimUsage(resp.usage)


def _native_create(chat_args: dict):
    with _native_client().messages.stream(
        **_openai_to_anthropic(chat_args), timeout=MODEL_ATTEMPT_TIMEOUT_S
    ) as stream:
        _t0 = _last = time.time()
        _chunks = 0
        for _event in stream:
            if getattr(_event, "type", "") == "content_block_delta":
                _chunks += 1
            _now = time.time()
            if _now - _last >= 30:
                print(
                    f"[stream] generating... {_chunks} chunks, "
                    f"{int(_now - _t0)}s elapsed",
                    flush=True,
                )
                _last = _now
        resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        raise RuntimeError("native API returned stop_reason=refusal")
    return _ShimResponse(resp)


def _drop_nones(obj):
    """Recursively drop None-valued keys from item dumps. SDK model_dump()
    emits SDK-side fields as None (same lesson as the Anthropic shim's
    parsed_output) and the API rejects unknown/None fields on replay."""
    if isinstance(obj, dict):
        return {k: _drop_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_nones(v) for v in obj]
    return obj


def _parts_to_resp_content(content, text_type: str) -> list[dict]:
    """Chat-completions content (str or parts list) -> Responses content parts.
    ``text_type`` is "input_text" for user/system items, "output_text" for
    replayed assistant text (the API validates the direction)."""
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content.strip() else []
    parts = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            if part.get("text", "").strip():
                parts.append({"type": text_type, "text": part["text"]})
        elif part.get("type") == "image_url":
            parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
    return parts


def _supports_explicit_openai_cache(model: Any) -> bool:
    """Only GPT-5.6 and later support content-block cache breakpoints."""
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", str(model).lower())
    return bool(match and (int(match[1]), int(match[2] or 0)) >= (5, 6))


def _mark_responses_cache_breakpoints(items: list[dict]) -> bool:
    """Mark stable input turn endings without modifying replayed assistant items.

    The transient budget/state tail is never cached. Keep the system boundary
    and the last three input-turn boundaries (at most four writes). In explicit
    mode a previous write is only reusable if its boundary is still marked:
    moving just one breakpoint to the newest input would lose that lookup.
    Derive turn boundaries from history, so compaction/undo need no cache state.
    """
    stable_end = len(items)
    while stable_end:
        item = items[stable_end - 1]
        content = item.get("content") or []
        if not (
            item.get("role") == "user"
            and content
            and content[0].get("type") == "input_text"
            and content[0]
            .get("text", "")
            .startswith(("[Round Budget]", "OBJECT STATE ("))
        ):
            break
        stable_end -= 1

    system_boundary: int | None = None
    pending: int | None = None
    turn_boundaries: list[int] = []
    for index, item in enumerate(items[:stable_end]):
        role = item.get("role")
        if item.get("type") == "function_call_output":
            # Use the same content-block representation on EVERY replay, not
            # only while this particular output has a selected breakpoint.
            item["output"] = [{"type": "input_text", "text": item["output"]}]
            pending = index
        elif role in ("system", "developer", "user"):
            pending = index
            if role in ("system", "developer") and not turn_boundaries:
                system_boundary = index
        elif pending is not None:
            # Reasoning/message/function_call items start an assistant turn.
            turn_boundaries.append(pending)
            pending = None
    if pending is not None:
        turn_boundaries.append(pending)

    selected = set(turn_boundaries[-3:])
    if system_boundary is not None:
        selected.add(system_boundary)
    for index in selected:
        item = items[index]
        content = item.get("content") or item["output"]
        content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return bool(selected)


def _openai_to_responses(chat_args: dict) -> dict:
    """chat.completions args -> responses.create kwargs.

    Tool-result images never appear here: the agent loop keeps tool messages
    text-only and ships render images in a separate user message (see
    generator._update_memory). Cache-enabled tool outputs use input_text blocks
    so their stable endpoint can carry an explicit cache breakpoint.
    """
    items: list[dict] = []
    for msg in chat_args.get("messages", []):
        role = msg.get("role")
        if role == "assistant":
            raw = msg.get("_raw_blocks")
            if raw and all(b.get("type") in _RESPONSES_ITEM_TYPES for b in raw):
                # Replay the model's own output items (reasoning + message +
                # function_call) verbatim — required so a reasoning item stays
                # attached to its function_call across the tool loop. Drop
                # function_calls the loop never answered (it executes only
                # tool_calls[0]); an unanswered call_id would 400.
                answered = {
                    tc.get("id") for tc in msg.get("tool_calls") or [] if tc.get("id")
                }
                items.extend(
                    b
                    for b in raw
                    if b.get("type") != "function_call" or b.get("call_id") in answered
                )
                continue
            # Compat reconstruction (history from another provider/model): text
            # as an assistant message, tool calls as bare id-less function_call
            # items (an `id` would make the API demand the reasoning item that
            # produced it, which we don't have here).
            text = _parts_to_resp_content(msg.get("content"), "output_text")
            if text:
                items.append({"role": "assistant", "content": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue
        if role == "tool":
            texts = msg.get("content")
            if isinstance(texts, list):
                joined = "\n".join(
                    p.get("text", "") for p in texts if isinstance(p, dict)
                ).strip()
            else:
                joined = str(texts or "").strip()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or msg.get("id") or "",
                    "output": joined or "(no output)",
                }
            )
            continue
        # system / user (and anything else message-like)
        parts = _parts_to_resp_content(msg.get("content"), "input_text")
        if parts:
            items.append({"role": role, "content": parts})

    out: dict = {"model": chat_args.get("model"), "input": items}
    tools = [
        {
            "type": "function",
            "name": (t.get("function") or {}).get("name", ""),
            "description": (t.get("function") or {}).get("description", ""),
            "parameters": (t.get("function") or {}).get(
                "parameters", {"type": "object"}
            ),
            **(
                {"strict": t["function"]["strict"]}
                if "strict" in (t.get("function") or {})
                else {}
            ),
        }
        for t in chat_args.get("tools") or []
    ]
    if tools:
        out["tools"] = tools
        tc = chat_args.get("tool_choice")
        if isinstance(tc, str):
            out["tool_choice"] = tc
        if "parallel_tool_calls" in chat_args:
            out["parallel_tool_calls"] = chat_args["parallel_tool_calls"]
    eff = chat_args.get("reasoning_effort")
    if eff:
        out["reasoning"] = {"effort": eff}
    # encrypted_content makes the replayed reasoning items self-contained, so
    # the stateless full-history resend works even where responses aren't
    # stored server-side; harmless otherwise.
    out["include"] = ["reasoning.encrypted_content"]
    mot = chat_args.get("max_completion_tokens") or chat_args.get("max_tokens")
    out["max_output_tokens"] = int(mot) if mot else RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS
    if _supports_explicit_openai_cache(chat_args.get("model")):
        options = dict(chat_args.get("prompt_cache_options") or {})
        if os.environ.get(
            "GRASE_OPENAI_CACHE", "1"
        ) != "0" and _mark_responses_cache_breakpoints(items):
            options.update(mode="explicit", ttl="30m")
        if options:
            # SDK 2.43.0 preserves unknown nested fields but does not expose
            # prompt_cache_options as a keyword yet. No SDK upgrade is needed.
            out["extra_body"] = {"prompt_cache_options": options}
    return out


class _RespShimToolCall:
    def __init__(self, item: dict):
        # .id must be the CALL id ("call_..."): the tool message's tool_call_id
        # comes from here, and function_call_output pairs on call_id.
        self.id = item.get("call_id")
        self.type = "function"
        self.function = _ShimFunction(
            item.get("name", ""), item.get("arguments") or "{}"
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _RespShimMessage:
    def __init__(self, resp):
        items = [_drop_nones(i.model_dump()) for i in resp.output]
        self.truncated = getattr(resp, "status", None) == "incomplete"
        texts: list[str] = []
        tool_calls: list[_RespShimToolCall] = []
        for it in items:
            if it.get("type") == "message":
                for p in it.get("content") or []:
                    if p.get("type") == "output_text" and p.get("text"):
                        texts.append(p["text"])
                    elif p.get("type") == "refusal":
                        texts.append(p.get("refusal", ""))
            elif it.get("type") == "function_call" and not self.truncated:
                tool_calls.append(_RespShimToolCall(it))
        self.content = "\n".join(texts).strip()
        if self.truncated:
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", "?")
            self.raw_content = []  # never replay a truncated/runaway item
            self.content = self.content or f"[reply truncated: {reason}]"
        else:
            self.raw_content = items  # replayed verbatim on the next turn
        self.tool_calls = tool_calls or None


class _RespShimUsage:
    def __init__(self, u):
        # input_tokens already INCLUDES cached tokens (same inclusive semantics
        # the Anthropic shim reconstructs), so pass through and surface the
        # cached split for the ledger.
        self.prompt_tokens = getattr(u, "input_tokens", 0) or 0
        self.completion_tokens = getattr(u, "output_tokens", 0) or 0
        d = getattr(u, "input_tokens_details", None)
        self.cache_read_input_tokens = (getattr(d, "cached_tokens", 0) or 0) if d else 0
        self.cache_creation_input_tokens = getattr(d, "cache_write_tokens", None)
        output_details = getattr(u, "output_tokens_details", None)
        self.reasoning_tokens = getattr(output_details, "reasoning_tokens", None)


class _RespShimChoice:
    def __init__(self, resp):
        self.message = _RespShimMessage(resp)
        if self.message.truncated:
            self.finish_reason = "length"  # what the agent loops' cut-off branch reads
        else:
            self.finish_reason = (
                "tool_calls"
                if self.message.tool_calls
                else getattr(resp, "status", "stop")
            )


class _RespShimResponse:
    def __init__(self, resp, request_id=None):
        self.id = getattr(resp, "id", None)
        # Streamed final responses carry no _request_id; the caller passes the header.
        self.request_id = request_id or getattr(resp, "_request_id", None)
        self.prompt_cache_diagnostics = getattr(resp, "prompt_cache_diagnostics", None)
        self.choices = [_RespShimChoice(resp)]
        self.usage = _RespShimUsage(getattr(resp, "usage", None))


_RESPONSES_BUDGET_SECONDS = 3600.0
RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS = 16000
MODEL_ATTEMPT_TIMEOUT_S = 120.0
_RESPONSES_ATTEMPT_TIMEOUT_SECONDS = MODEL_ATTEMPT_TIMEOUT_S


def _request_event(trace: dict | None, event: str, **fields: Any) -> None:
    """Write operator-only timing, never prompt bodies, credentials or model memory."""
    if trace is None:
        return
    row = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "call_id": trace["call_id"],
        "pid": os.getpid(),
        "tag": os.environ.get("GRASE_USAGE_TAG"),
        "model": trace["model"],
        "effort": trace["effort"],
        "elapsed_s": round(time.monotonic() - trace["started"], 6),
        **fields,
    }
    # Malformed optional usage or a failed sink must never replay a successful call.
    try:
        line = json.dumps(row, allow_nan=False)
        # Override ML logging convention: use this standalone project's existing logger.
        logging.getLogger(__name__).info("Model request: %s", line)
        usage_path = os.environ.get("GRASE_USAGE_LOG")
        if usage_path:
            with Path(usage_path).with_suffix(".requests.jsonl").open("a") as stream:
                stream.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).warning("Request timing log failed: %s", exc)


def _responses_create(
    client: OpenAI,
    chat_args: dict,
    *,
    timeout: float = _RESPONSES_BUDGET_SECONDS,
    trace: dict | None = None,
    attempt: int = 1,
):
    prepared_at = time.monotonic()
    request = _openai_to_responses(chat_args)
    preparation_s = time.monotonic() - prepared_at
    remaining = min(_RESPONSES_ATTEMPT_TIMEOUT_SECONDS, timeout - preparation_s)
    if remaining <= 0:
        raise TimeoutError("Responses elapsed-time budget exhausted during preparation")
    _request_event(
        trace,
        "send",
        attempt=attempt,
        preparation_s=round(preparation_s, 6),
        timeout_s=remaining,
    )
    stream = client.with_options(max_retries=0).responses.create(
        **request, stream=True, timeout=remaining
    )
    headers = stream.response.headers
    request_id = headers.get("x-request-id")
    _request_event(
        trace,
        "headers",
        attempt=attempt,
        status_code=stream.response.status_code,
        request_id=request_id,
        provider_processing_ms=headers.get("openai-processing-ms"),
    )
    headers_at = time.monotonic()
    first_event_s = None
    events = 0
    final = None
    for event in stream:
        events += 1
        if first_event_s is None:
            first_event_s = round(time.monotonic() - headers_at, 3)
        if event.type in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            final = event
    if final is None:
        raise RuntimeError(
            f"Responses stream ended after {events} events without a final response"
        )
    _request_event(
        trace,
        "stream",
        attempt=attempt,
        events=events,
        first_event_s=first_event_s,
        terminal=final.type,
    )
    resp = final.response
    if final.type == "response.failed":
        err = getattr(resp, "error", None)
        raise RuntimeError(f"responses API failed: {getattr(err, 'message', err)}")
    if getattr(resp, "status", None) == "incomplete":
        reason = getattr(getattr(resp, "incomplete_details", None), "reason", "?")
        logging.getLogger(__name__).warning("responses API incomplete: %s", reason)
    return _RespShimResponse(resp, request_id=request_id)


def get_model_response(client: OpenAI, chat_args: dict, effort: str = "high") -> Any:
    ""
    provider = _provider_or_none(chat_args.get("model"))
    if (
        effort
        and "reasoning_effort" not in chat_args
        and provider in ("anthropic", "openai")
    ):
        chat_args = {**chat_args, "reasoning_effort": effort}
    if (
        provider == "openai"
        and chat_args.get("tools")
        and "parallel_tool_calls" not in chat_args
    ):
        # OpenAI fans out several tool calls per turn by default; the agent loops
        # execute exactly one per round, so force sequential. Gated on `tools`
        # because OpenAI rejects the param when no tools are sent — which is every
        # aux/preprocess VLM call. Lives here, not in the agents: provider branching
        # belongs with the other request shaping (see provider_of).
        chat_args = {**chat_args, "parallel_tool_calls": False}
    if (
        provider == "qwen"
        and "max_tokens" in chat_args
        and "extra_body" not in chat_args
    ):
        chat_args = {
            **chat_args,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    if provider == "openai" and "max_tokens" in chat_args:
        # OpenAI's reasoning models reject `max_tokens` on chat/completions and want
        # `max_completion_tokens`. Every other provider the pipeline talks to (incl.
        # Anthropic's and Gemini's OpenAI-compat endpoints) still takes `max_tokens`,
        # so translate only here rather than changing the ~8 aux call sites.
        chat_args = {**chat_args}
        chat_args["max_completion_tokens"] = chat_args.pop("max_tokens")
    last_error: Exception | None = None
    native = os.environ.get("GRASE_NATIVE_API", "0") == "1" and provider == "anthropic"
    # Tool-bearing OpenAI calls go through /v1/responses: gpt-5.6-family models
    # 400 on tools + reasoning at /v1/chat/completions, and 'reasoning_effort:
    # none' (the only chat-completions escape) would strip the agent stages of
    # thinking. Tool-less calls (all aux/preprocess VLM) stay on chat/completions.
    # GRASE_OPENAI_RESPONSES=0 reverts.
    use_responses = (
        provider == "openai"
        and bool(chat_args.get("tools"))
        and os.environ.get("GRASE_OPENAI_RESPONSES", "1") != "0"
    )
    trace = None
    if use_responses:
        trace = {
            "call_id": uuid.uuid4().hex,
            "started": time.monotonic(),
            "model": chat_args.get("model"),
            "effort": chat_args.get("reasoning_effort"),
        }
        _request_event(trace, "start", budget_s=_RESPONSES_BUDGET_SECONDS)
    overload_sleeps = [10, 30, 60, 120, 240]
    quick_sleeps = [10, 10]
    timeout_sleeps = [2, 5, 10, 20]
    attempt = 0
    while True:
        remaining = _RESPONSES_BUDGET_SECONDS
        if trace is not None:
            remaining -= time.monotonic() - trace["started"]
            if remaining <= 0:
                _request_event(
                    trace, "failed", reason="elapsed_budget", attempts=attempt
                )
                raise TimeoutError(
                    "Responses elapsed-time budget exhausted"
                ) from last_error
        try:
            if native:
                resp = _native_create(chat_args)
            elif use_responses:
                resp = _responses_create(
                    client,
                    chat_args,
                    timeout=remaining,
                    trace=trace,
                    attempt=attempt + 1,
                )
            else:
                resp = client.chat.completions.create(**chat_args)
            _log_usage(resp, chat_args)
            if trace is not None:
                _request_event(
                    trace,
                    "completed",
                    attempts=attempt + 1,
                    response_id=resp.id,
                    request_id=resp.request_id,
                    prompt_tokens=resp.usage.prompt_tokens,
                    cached_tokens=resp.usage.cache_read_input_tokens,
                    cache_write_tokens=resp.usage.cache_creation_input_tokens,
                    output_tokens=resp.usage.completion_tokens,
                    reasoning_tokens=resp.usage.reasoning_tokens,
                )
            return resp
        except Exception as exc:
            # Surface the real cause (e.g. 400 oversized image, 429 rate
            # limit) instead of swallowing it into a generic failure.
            logging.getLogger(__name__).error("Model request failed: %s", exc)
            last_error = exc
            _request_event(
                trace,
                "attempt_failed",
                attempt=attempt + 1,
                error_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            )
            msg = str(exc).lower()
            if (
                "529" in msg
                or "overloaded" in msg
                or "429" in msg
                or "rate limit" in msg
            ):
                sleeps = overload_sleeps
            elif (
                isinstance(exc, TimeoutError)
                or "timeout" in type(exc).__name__.lower()
                or "timed out" in msg
            ):
                sleeps = timeout_sleeps
            else:
                sleeps = quick_sleeps
            if attempt >= len(sleeps):
                _request_event(
                    trace, "failed", reason="retry_limit", attempts=attempt + 1
                )
                raise Exception(f"Failed to get model response: {last_error}") from exc
            if trace is not None:
                remaining = _RESPONSES_BUDGET_SECONDS - (
                    time.monotonic() - trace["started"]
                )
                if remaining <= sleeps[attempt]:
                    _request_event(
                        trace, "failed", reason="elapsed_budget", attempts=attempt + 1
                    )
                    raise TimeoutError(
                        "Responses elapsed-time budget exhausted"
                    ) from exc
            _request_event(
                trace, "retry", attempt=attempt + 1, backoff_s=sleeps[attempt]
            )
            time.sleep(sleeps[attempt])
            attempt += 1


_PROVIDERS = (
    # "fireworks" first: Fireworks model ids ("accounts/fireworks/models/kimi-k3")
    # embed other providers' family names (gpt-oss, qwen*, ...) further down the path.
    ("fireworks", "fireworks"),
    ("gpt", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("qwen", "qwen"),
)

_PROVIDER_CREDS = {
    "openai": (OPENAI_API_KEY, OPENAI_BASE_URL),
    "anthropic": (CLAUDE_API_KEY, CLAUDE_BASE_URL),
    "google": (GEMINI_API_KEY, GEMINI_BASE_URL),
    "qwen": (QWEN_API_KEY, QWEN_BASE_URL),
    "fireworks": (FIREWORKS_API_KEY, FIREWORKS_BASE_URL),
}


def provider_of(model_name: str) -> str:
    """Map a model name to its provider. Raises on an unrecognized name."""
    lowered = str(model_name).lower()
    for token, provider in _PROVIDERS:
        if token in lowered:
            return provider
    raise ValueError(f"Invalid model name: {model_name}")


def _provider_or_none(model_name: Any) -> str | None:
    """provider_of, but None instead of raising — for request-shaping decisions.

    An unrecognized model must not turn an otherwise-valid request into a crash;
    it just means we apply no provider-specific tweaks and send it as-is.
    """
    try:
        return provider_of(model_name)
    except ValueError:
        return None


def get_model_info(model_name: str) -> dict[str, str]:
    """Get API key and base URL for the specified model."""
    api_key, base_url = _PROVIDER_CREDS[provider_of(model_name)]
    return {"api_key": api_key, "base_url": base_url}


def build_client(
    model_name: str, api_key: str | None = None, base_url: str | None = None
) -> OpenAI:
    """Build an OpenAI client for the specified model.

    ``api_key``/``base_url`` override the provider defaults (the agents pass their
    config's resolved creds through here so CLI --api-key/--api-base-url still win).
    """
    info = get_model_info(model_name)
    # get_model_response owns retries; the SDK's own 2 retries would multiply a hang.
    kwargs: dict = {"max_retries": 0, "timeout": MODEL_ATTEMPT_TIMEOUT_S}
    if provider_of(model_name) == "fireworks":
        kwargs["timeout"] = 3600
    return OpenAI(
        api_key=api_key or info["api_key"],
        base_url=base_url or info["base_url"],
        **kwargs,
    )


# Long-edge cap shared by the pipeline's canonical input copy and the VLM encoding, so
# a raw 24MP phone photo never taxes GT-resolution renders (MoGE, register, texture...)
# or the base64 payload (model per-image size limits, e.g. Anthropic's 10 MB).
MAX_IMAGE_EDGE = 1536

AGENT_VLM_EDGE = int(os.environ.get("GRASE_VLM_EDGE", "768"))


def normalize_input_image(
    image_path: str, dst: str, max_edge: int = MAX_IMAGE_EDGE
) -> str:
    """Write an RGB PNG copy of ``image_path`` to ``dst`` with the long edge capped at
    ``max_edge`` (aspect preserved). The pipeline's canonical input: preprocessing points
    every downstream consumer (depth, masks, renders, VLM prompts) at this copy."""
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    image.save(dst, format="PNG")
    return dst


def display_path(path: Any, scene_root: Any = None) -> str:
    """Render a run artifact path for a PROMPT: relative to the run's scene root.

    No tool accepts a model-supplied path; ``resolve_display_path`` maps verifier
    artifact references back to the run directory.

    Falls back to the basename when the path lies outside ``scene_root`` (or no root is
    known), so a stray absolute path can never reach a prompt through here.
    """
    text = os.fspath(path) if path else ""
    if not text:
        return ""
    if scene_root:
        try:
            relative = os.path.relpath(
                os.path.abspath(text), os.path.abspath(scene_root)
            )
        except (OSError, ValueError):  # different drive / unresolvable
            return os.path.basename(text)
        if not relative.startswith(os.pardir):
            return relative
    return os.path.basename(text)


def resolve_display_path(path: Any, scene_root: Any = None) -> str:
    """Inverse of :func:`display_path` for a path the MODEL handed back.

    The verifier cites ``problem_images`` paths copied out of its prompt captions, and the
    generator then opens them. Accepts a scene-root-relative path (what prompts now show),
    an absolute path (in-flight runs started before the trim, and any caller that never
    relativized), or a bare basename. Returns the input unchanged when nothing resolves,
    so the caller's own existence check still owns the failure path.
    """
    text = os.fspath(path) if path else ""
    if not text or os.path.isabs(text):
        return text
    if scene_root:
        candidate = os.path.join(os.fspath(scene_root), text)
        if os.path.exists(candidate):
            return candidate
    return text


def get_image_base64(image_path: str, max_edge: int = MAX_IMAGE_EDGE) -> str:
    """Return a full data URL for the image, preserving original jpg/png format.

    ``max_edge`` caps the long side (never upscales). Default keeps the historical
    MAX_IMAGE_EDGE safety net for images that didn't enter through
    normalize_input_image; agent-loop callers pass AGENT_VLM_EDGE (768) so every
    round attachment lands at one size (audit §12)."""
    image = Image.open(image_path)
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge))  # in place, preserves aspect
    img_byte_array = io.BytesIO()
    ext = os.path.splitext(image_path)[1].lower()

    # Convert image to appropriate mode for saving
    if ext in [".jpg", ".jpeg"]:
        save_format = "JPEG"
        mime_subtype = "jpeg"
        # JPEG doesn't support transparency, convert RGBA to RGB
        if image.mode in ["RGBA", "LA", "P"]:
            # Convert P mode to RGB first, then handle RGBA
            if image.mode == "P":
                image = image.convert("RGBA")
            # Convert RGBA to RGB with white background
            if image.mode == "RGBA":
                # Create a white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[-1]
                )  # Use alpha channel as mask
                image = background
            elif image.mode == "LA":
                # Convert LA to RGB
                image = image.convert("RGB")
    elif ext == ".png":
        save_format = "PNG"
        mime_subtype = "png"
        # PNG supports transparency, but convert P mode to RGBA
        if image.mode == "P":
            image = image.convert("RGBA")
    else:
        # Fallback: keep original format if recognizable, else default to PNG
        save_format = image.format or "PNG"
        mime_subtype = (
            save_format.lower() if save_format.lower() in ["jpeg", "png"] else "png"
        )
        # Handle P mode for fallback cases
        if image.mode == "P":
            if save_format == "JPEG":
                image = image.convert("RGB")
            else:
                image = image.convert("RGBA")

    image.save(img_byte_array, format=save_format)
    img_byte_array.seek(0)
    base64enc_image = base64.b64encode(img_byte_array.read()).decode("utf-8")
    if base64enc_image.startswith("/9j/"):
        mime_subtype = "jpeg"
    elif base64enc_image.startswith("iVBOR"):
        mime_subtype = "png"
    elif base64enc_image.startswith("UklGR"):
        mime_subtype = "webp"
    return f"data:image/{mime_subtype};base64,{base64enc_image}"
