"""Generator Agent for code synthesis in the GRASE system.

The Generator Agent is responsible for iteratively generating and refining code
based on visual targets, using tool calls to execute and evaluate the generated code.
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Optional

from lib.agents.prompt_builder import (
    PromptBuilder,
    bounded_runtime_root_surface_edits,
    scene_graph_revision,
)
from lib.agents.scene_preseed import fetch_scene_info
from lib.agents.tool_client import ExternalToolClient
from lib.utils.common import (
    AGENT_VLM_EDGE,
    build_client,
    display_path,
    get_image_base64,
    get_model_response,
    resolve_display_path,
)


class GeneratorAgent:
    """Agent responsible for generating and refining code based on visual targets.

    The Generator Agent iteratively produces code, executes it, and refines based
    on feedback from the Verifier Agent. It uses MCP tools for code execution
    and scene manipulation.

    Attributes:
        config: Configuration dictionary containing model settings and paths.
        memory: Conversation memory for the agent.
        tool_client: Client for calling external MCP tools.
    """

    _POST_FLIP_FOLLOWUP_TYPE = "post_rotate_180_investigation"
    _POST_FLIP_ALLOWED_TOOLS = frozenset({"investigate_objects", "undo_last_step"})
    _EXECUTE_ARGUMENT_PROTOCOL_ERROR = "tool_argument_protocol_error"
    _EXECUTE_ARGUMENT_FORMAT_RETRY_LIMIT = 2

    def __init__(self, args: dict[str, Any]) -> None:
        """Initialize the Generator Agent.

        Args:
            args: Configuration dictionary with keys like 'model', 'api_key',
                  'generator_tools', 'max_rounds', etc.
        """
        self.config = args
        self.agent_name = self.config.get(
            "generator_agent_name", self.__class__.__name__
        )
        self.memory_filename = self.config.get(
            "generator_memory_filename", "generator_memory.json"
        )
        self.last_memory_path: Optional[str] = None
        self.last_tool_name: Optional[str] = None
        self.last_tool_arguments: dict[str, Any] = {}
        self.last_tool_response: Optional[dict[str, Any]] = None
        # Newest full-scene render this attempt produced (path + viewpoint); root hands it
        # to the paired verifier. See _remember_scene_render.
        self._last_scene_render: Optional[dict[str, Any]] = None
        self.memory: list[dict[str, Any]] = []
        self.init_plan: Optional[str] = None
        # A retained semantic 180-degree flip makes the object's pre-flip crops and
        # measurements stale.  The Blender backend issues a transaction-bound follow-up
        # record; this agent-side copy is the cross-server guard (notably, ``end`` lives
        # on a different MCP server).  Only matching structured backend resolution can
        # clear it -- a model merely naming the right tool is not enough.
        self._post_flip_required_followup: Optional[dict[str, Any]] = None
        self._post_flip_last_resolution: Optional[dict[str, Any]] = None

        # Provider-specific request shaping (OpenAI's parallel_tool_calls, reasoning
        # effort, max_completion_tokens) lives in common.get_model_response.

        # Initialize tool client
        self.tool_client = ExternalToolClient(
            self.config.get("generator_tools"), self.config
        )

        # Initialize LLM client (build_client carries per-provider settings, e.g.
        # the fireworks 3600s timeout; config creds still win when set).
        self.client = build_client(
            self.config["model"],
            api_key=self.config.get("api_key"),
            base_url=self.config.get("api_base_url"),
        )

        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("generator", "system")
        self.memory.extend(self.system_prompt)
        self._append_stage_retry_context()
        self._append_prior_plan_context()
        # Everything appended so far (system prompt + retry feedback + prior plan) is
        # the attempt's standing context — pin it past build_memory_append_only's
        # runaway-guard truncation.
        self.protected_head = len(self.memory)

    async def _preseed_scene_info(self) -> None:
        """CT5: fetch get_scene_info ONCE at stage entry and seed it into memory.

        Nearly every stage session opened with the model spending round 0 on
        get_scene_info ("I'll start by reading the current scene state") — a full LLM
        round-trip (~20-30 s + its window tokens) to fetch information the pipeline can
        hand it for a 0.5 s server-side tool call. The seed becomes part of the
        protected session-entry prefix, so neither composition's ledger nor the
        append-only runaway guard can discard it. Best-effort and flag-gated
        (GRASE_STAGE_PRESEED=0 reverts): a failed preseed must never block the stage —
        the model can still ask."""
        info = await fetch_scene_info(self.tool_client)
        if info is None:
            return
        self.memory.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "CURRENT SCENE STATE (auto-fetched at this attempt's entry; "
                            "bounded snapshot of the entry state). Do not call "
                            "get_scene_info merely to re-fetch this same baseline. The "
                            "scene may change after your edits; follow the stage workflow "
                            "if it explicitly requires a current post-edit diagnostic:\n"
                            + info
                        ),
                    }
                ],
            }
        )
        self.protected_head = len(self.memory)

    async def run(self) -> dict[str, Any]:
        """Run the generator agent loop to produce and refine code.

        Iteratively generates code, executes it via tools, and incorporates
        feedback from the verifier until completion or max rounds reached.
        Returns a lightweight artifact summary that the root agent can pass to
        downstream stages.
        """
        print(f"\n=== Running {self.agent_name} ===\n")

        # ``__new__``-constructed focused tests and a resumed run may not have passed
        # through __init__.  Do not reset an existing requirement: it must survive every
        # memory/window update until the backend resolves or cancels its exact token.
        if not hasattr(self, "_post_flip_required_followup"):
            self._post_flip_required_followup = None
        if not hasattr(self, "_post_flip_last_resolution"):
            self._post_flip_last_resolution = None
        # Production agents enter with no pending follow-up.  This condition keeps an
        # accidental/re-entrant run from making get_scene_info the forbidden first call
        # after a retained flip.
        if not self._post_flip_followup_pending():
            await self._preseed_scene_info()
        ended = False
        self._rules_all_pass = False
        self._edited_since_rules = False
        self._gate_input_revision = 0
        self._rules_revision: Optional[int] = None
        # Machine-readable evidence belongs to the same scene revision as the gate
        # verdict.  ``end`` replaces ``last_tool_response``, so without these fields the
        # verifier saw only a lossy prose/regex summary of the final rules check.  Keep
        # each backend record verbatim and mark whether its revision is still current.
        self._rule_evidence: Optional[dict[str, Any]] = None
        self._yaw_evidence: Optional[dict[str, Any]] = None
        self._relationship_evidence: Optional[list[dict[str, Any]]] = None
        self._constraint_results: Optional[list[dict[str, Any]]] = None
        self._rules_evidence_revision: Optional[int] = None
        # Residual yaw completion is separate from the rule verdict. A full ladder
        # pass can still require an ID-only backend correspondence before ``end``.
        self._advisory_requirement: Optional[dict[str, Any]] = None
        self._advisory_requirement_revision: Optional[int] = None
        self._yaw_resolution: Optional[dict[str, Any]] = None
        self._yaw_resolution_revision: Optional[int] = None
        self._advisory_protocol_error: Optional[str] = None
        # BEV freshness is bound to the same revision counter as the initializer rules gate.
        # This is advisory rather than a completion gate: after a scene edit, the transient
        # round reminder marks the prior image stale without rewriting append-only memory or
        # disturbing prompt-cache prefixes.
        self._last_bev_revision: Optional[int] = None
        max_rounds = int(self.config.get("max_rounds") or 1)
        terminal_end_grace_eligible = False
        self._terminal_end_grace_offered = False
        self._terminal_end_grace_accepted = False
        self._normal_rounds_exhausted = False
        # Malformed execute_and_evaluate arguments are rejected before execution, so
        # they may receive a small attempt-wide format allowance without spending a
        # semantic editing round.  This is intentionally separate from max_rounds and
        # absolutely capped: every other failure, and every malformed call after the
        # second allowance, consumes the normal round exactly as before.
        self._execute_argument_format_retries_used = 0
        # The outer request bound is the mechanical no-loop guarantee. ``i`` subtracts
        # only actually granted format credits; every other response advances the
        # normal-round index automatically.
        for request_index in range(
            max_rounds + self._EXECUTE_ARGUMENT_FORMAT_RETRY_LIMIT
        ):
            i = request_index - self._execute_argument_format_retries_used
            if i >= max_rounds:
                break
            print(f"=== Round {i} ===\n")

            # Prepare chat args
            print("Prepare chat args...")
            memory = self._memory_view()
            memory.append(self._round_budget_message(i, max_rounds))
            tool_configs = self._build_tool_configs()
            chat_args = {
                "model": self.config.get("model"),
                "messages": memory,
                "tools": tool_configs,
                "tool_choice": "auto",
            }

            # Generate response
            print("Generate response...")
            response = get_model_response(self.client, chat_args)
            message = response.choices[0].message

            # Handle tool call
            print("Handle tool call...")
            if not message.tool_calls:
                if message.content != "":
                    _raw = getattr(message, "raw_content", None)
                    self.memory.append(
                        {
                            **({"_raw_blocks": _raw} if _raw else {}),
                            "role": "assistant",
                            "content": message.content,
                        }
                    )
                else:
                    self.memory.append({"role": "assistant", "content": "No output"})
                _fr = getattr(response.choices[0], "finish_reason", None)
                self.memory.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was CUT OFF at the response token limit "
                            "before any tool call was emitted — everything you reasoned was "
                            "discarded and the round is spent. This round, keep reasoning to "
                            "a few sentences and go STRAIGHT to exactly one tool call."
                            if _fr in ("max_tokens", "length")
                            else "Every single output must contain a 'tool_call' field. Your previous message did not contain a 'tool_call' field. Please reconsider."
                        ),
                    }
                )
                self._save_memory()
                continue
            else:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                print(f"Call tool {tool_name}...")
                try:
                    tool_arguments = json.loads(tool_call.function.arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    print(f"Tool {tool_name}: arguments are not valid JSON ({exc})")
                    self.memory.append(
                        {
                            "role": "assistant",
                            "content": message.content
                            or f"[{tool_name} call with unparseable arguments]",
                        }
                    )
                    self.memory.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous {tool_name} call carried arguments that "
                                "were not valid JSON — most likely the reply was CUT OFF "
                                "at the output token limit. Nothing was executed and the "
                                "round is spent. Call the tool again with COMPLETE, "
                                "compact arguments."
                            ),
                        }
                    )
                    self._save_memory()
                    continue

                # A retained rotate_180 is a transaction boundary: before dispatching
                # ANY server's next tool, require a paired investigation containing all
                # flipped ids, or an immediate undo of that exact edit.  In particular,
                # this catches end/check/render calls which the Blender executor itself
                # cannot intercept because they may be routed elsewhere.
                followup_refusal = self._post_flip_tool_refusal(
                    tool_name, tool_arguments
                )
                if followup_refusal:
                    print(f"Refusing {tool_name}: post-flip investigation pending")
                    self._record_tool_refusal(message, tool_call, followup_refusal)
                    continue

                # A voluntary `end` means the stage claims completion. Enforce that
                # contract for every stage exposing check_rules_enforced: a missing,
                # failed, or stale gate bounces back in-loop. The normal round budget
                # is the backstop, so there is no escape hatch that eventually accepts
                # an invalid end.
                embedded_resolution_refusal = None
                if tool_name == "end":
                    embedded_resolution_refusal = (
                        await self._resolve_embedded_yaw_advisory(tool_arguments)
                    )
                refusal = (
                    embedded_resolution_refusal or self._refuse_end_reason()
                    if tool_name == "end"
                    else None
                )
                if refusal:
                    print("Refusing end: completion requirements not satisfied")
                    self.memory.append({
                        "role": "assistant", "content": message.content,
                        "tool_calls": [tool_call.model_dump()],
                    })  # fmt: skip
                    self.memory.append({
                        "role": "tool", "name": "end",
                        "tool_call_id": tool_call.id,
                        "content": [{"type": "text", "text": refusal}],
                    })  # fmt: skip
                    self._save_memory()
                    continue

                tool_response = await self.tool_client.call_tool(
                    tool_name, tool_arguments
                )
            execute_argument_format_error = self._is_execute_argument_format_error(
                tool_name, tool_response
            )
            tool_committed = not bool(tool_response.get("_tool_error", False))
            post_flip_error_invalidates_rules = self._post_flip_error_invalidates_rules(
                tool_name, tool_arguments, tool_response
            )
            # Consume this before _ensure_end_of_round_render, which intentionally
            # removes the private _tool_error marker while formatting model feedback.
            self._consume_post_flip_followup(tool_name, tool_arguments, tool_response)
            if (
                tool_name == "render_bev"
                and tool_committed
                and tool_response.get("image")
            ):
                self._last_bev_revision = self._gate_input_revision
            self._remember_tool_result(tool_name, tool_arguments, tool_response)
            tool_response = await self._ensure_end_of_round_render(
                tool_response, tool_name
            )
            self._remember_tool_result(tool_name, tool_arguments, tool_response)
            self._remember_scene_render(tool_name, tool_arguments, tool_response)

            # Update and save memory
            print("Update and save memory...")
            self._update_memory({"assistant": message, "user": tool_response})

            if (
                execute_argument_format_error
                and self._execute_argument_format_retries_used
                < self._EXECUTE_ARGUMENT_FORMAT_RETRY_LIMIT
            ):
                self._execute_argument_format_retries_used += 1
                self.memory.append(self._execute_argument_format_retry_message())
                self._save_memory()
                # The backend explicitly certified that nothing committed, so keep
                # the same normal-round index and do not touch gate/revision state.
                continue
            self._save_memory()

            # Track the rules gate for this round (before end-detection below).
            if tool_name == "check_rules_enforced":
                self._rules_all_pass = bool(tool_response.get("rules_all_pass", False))
                self._rules_revision = self._gate_input_revision
                self._capture_rules_evidence(tool_response)
                self._edited_since_rules = False
                terminal_end_grace_eligible = bool(
                    i == max_rounds - 1
                    and tool_committed
                    and self._rules_all_pass
                    and self._rules_passed_state() is True
                    and not self._post_flip_followup_pending()
                )
            elif tool_name == "resolve_yaw_advisory":
                self._capture_yaw_resolution(tool_response)
            elif tool_committed and self._tool_invalidates_rules(
                tool_name, tool_response
            ):
                previous_revision = self._gate_input_revision
                self._gate_input_revision += 1
                self._edited_since_rules = True
                # A rules-only mutation such as a coverage bypass does not stale a BEV. Carry
                # its revision forward instead of requiring an identical re-render.
                if (
                    not self._tool_mutates_scene(tool_name, tool_response)
                    and self._last_bev_revision == previous_revision
                ):
                    self._last_bev_revision = self._gate_input_revision

            if post_flip_error_invalidates_rules:
                # Fail closed even when the response was lost before the generator
                # learned the backend transaction token.  A later reconciliation or
                # exact undo still needs a fresh rules pass before end.
                self._gate_input_revision += 1
                self._edited_since_rules = True

            # Completion is a committed state transition, not merely an attempted
            # tool name. A transport/backend end error consumes this round and the
            # normal loop continues when budget remains.
            if tool_name == "end" and tool_committed:
                ended = True
                break

        # This flag records the normal-loop cutoff independently of whether the
        # optional end-only response rescues completion afterward.
        self._normal_rounds_exhausted = not ended
        if self._normal_rounds_exhausted and terminal_end_grace_eligible:
            ended = await self._run_terminal_end_grace()
            self._terminal_end_grace_accepted = ended

        # Completion and orchestration cutoff are distinct states. `end` is a voluntary
        # claim that the stage is complete. Exhausting the normal tool-call budget stops
        # the attempt unless its final call earned the single end-only grace above.
        self._hit_max_rounds = self._normal_rounds_exhausted and not ended
        self._completed_voluntarily = ended
        self._termination_reason = "agent_end" if ended else "budget_exhausted"
        if not ended:
            self._save_memory()
        await self._refresh_stale_rules_pass()

        print(f"\n=== Finish {self.agent_name} process ===\n")
        return self._build_result()

    def _append_prior_plan_context(self) -> None:
        """Refinement pass: show the PREVIOUS attempt's plan as material to revise. The agent
        re-plans first — reviewing the reference image, this plan, and the verifier feedback +
        flagged renders above — then calls initialize_plan with an UPDATED plan and refines the
        existing scene (the blend persists). The revised plan, not this stale one, is what it
        then follows."""
        plan = self.config.get("stage_prior_plan")
        if not plan:
            return
        self.memory.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a REFINEMENT pass. Below is the plan from your PREVIOUS attempt. Do NOT "
                            "just repeat it: review the reference image and the verifier feedback above (with "
                            "the flagged renders), then call initialize_plan with a REVISED plan that fixes "
                            "those issues — update it, do NOT start from scratch. Then EDIT the existing "
                            "surfaces in place with execute_and_evaluate so they match the revised plan — the "
                            "blend is preserved from the previous attempt and much of it is already correct. "
                            "Modify ONLY what the feedback flags; do NOT delete or re-create surfaces that "
                            "already pass (a from-scratch rebuild re-introduces construction bugs and throws "
                            "away verified state).\n\nPrevious attempt's plan:\n"
                            + str(plan)
                        ),
                    }
                ],
            }
        )

    def _append_stage_retry_context(self) -> None:
        """Append root-provided verifier feedback for a rejected stage retry."""
        if not self.config.get("stage_retry_feedback"):
            return
        feedback = self.config.get("stage_retry_feedback", {})
        retry_context = {
            "latest_feedback": feedback,
            "approval_checklist": self.config.get("stage_approval_checklist", []),
            "suggested_fixes": feedback.get("suggested_fixes", []),
            "problem_images": feedback.get("problem_images", []),
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Verifier feedback from the rejected stage attempt. Treat the pending approval_checklist as the todo list. "
                    "Use suggested_fixes as concrete implementation guidance, and inspect the problem images before editing. "
                    "In your next execution thought, explicitly state which checklist items and suggested fixes you are addressing, "
                    "and avoid regressing items marked done.\n"
                    + json.dumps(retry_context, indent=2, ensure_ascii=False)
                ),
            }
        ]
        for item in feedback.get("problem_images", []) or []:
            path = item.get("path") if isinstance(item, dict) else None
            # The verifier copies these out of its own prompt captions, which are now
            # scene-root-relative; map them back before touching the filesystem. Absolute
            # paths still resolve (in-flight runs, and any caller that never relativized).
            if path:
                path = resolve_display_path(path, self.config.get("scene_root"))
            if path and os.path.exists(path):
                try:
                    url = get_image_base64(path, max_edge=AGENT_VLM_EDGE)
                except Exception:
                    logging.warning(
                        "stage retry: skipping non-image problem path %s", path
                    )
                    continue
                content.append(
                    {
                        "type": "image_url",
                        # Retry-evidence crops/renders: agent-loop cap (768).
                        "image_url": {"url": url},
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Problem image loaded from local path: "
                            f"{display_path(path, self.config.get('scene_root'))}. "
                            f"Note: {item.get('note', '')}"
                        ),
                    }
                )
        self.memory.append({"role": "user", "content": content})

    def _is_execute_argument_format_error(
        self, tool_name: str, tool_response: Any
    ) -> bool:
        """Return whether a rejected execute call earns one format-only retry.

        All fields are required. In particular, a generic retryable transport error,
        a missing patch base, a SEARCH/REPLACE mismatch, or any response that cannot
        prove the scene stayed unmodified must consume the ordinary round budget.
        """
        return bool(
            tool_name == "execute_and_evaluate"
            and isinstance(tool_response, dict)
            and tool_response.get("_tool_error") is True
            and tool_response.get("error_code") == self._EXECUTE_ARGUMENT_PROTOCOL_ERROR
            and tool_response.get("expected_tool") == "execute_and_evaluate"
            and tool_response.get("retryable") is True
            and tool_response.get("scene_mutation") == "not_committed"
            and "required_followup" not in tool_response
        )

    def _execute_argument_format_retry_message(self) -> dict[str, Any]:
        """Return the bounded correction after a non-mutating protocol rejection."""
        used = self._execute_argument_format_retries_used
        remaining = max(self._EXECUTE_ARGUMENT_FORMAT_RETRY_LIMIT - used, 0)
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "[Execute Argument Format Retry]\n"
                        f"Format-only retry {used} of "
                        f"{self._EXECUTE_ARGUMENT_FORMAT_RETRY_LIMIT}; this rejected "
                        "call did not consume a normal editing round because the backend "
                        "certified scene_mutation=not_committed. Call "
                        "execute_and_evaluate again with Python in the actual top-level "
                        "code field. FULL-SCRIPT argument shape: "
                        '{"thought":"Run a complete Blender script.","code":"import '
                        'bpy\\nprint(len(bpy.context.scene.objects))","code_diff":""}. '
                        "Do not put "
                        'Python, JSON arguments, or <parameter name="code"> markup '
                        "inside thought. "
                        + (
                            f"{remaining} format-only retry allowance(s) remain."
                            if remaining
                            else "No format-only retry allowances remain; another malformed "
                            "call will consume a normal round."
                        )
                    ),
                }
            ],
        }

    def _round_budget_message(
        self,
        round_index: int,
        max_rounds: int,
        stage_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return a transient reminder of the remaining normal tool-call budget."""
        rounds_left = max(max_rounds - round_index, 0)
        rounds_after_this = max(max_rounds - round_index - 1, 0)
        if stage_name is None:
            stage_name = getattr(self, "config", {}).get("root_stage_name")
        stage_text = f" for the {stage_name} stage" if stage_name else ""
        bev_status = ""
        gate_status = ""
        post_flip_status = self._post_flip_round_status()
        late_flip_status = ""
        if (
            stage_name == "composition"
            and not self._post_flip_followup_pending()
            and rounds_after_this < 2
        ):
            late_flip_status = (
                " LATE rotate_180 BUDGET WARNING: fewer than 2 normal rounds remain "
                "after this one. Starting an optional rotate_180 now cannot complete "
                "both its mandatory next-round investigate_objects call and a fresh "
                "final check_rules_enforced call; the attempt will end "
                "budget_exhausted. Use rotate_180 now only if you accept that incomplete "
                "outcome. Otherwise finish the current work and run the final gate."
            )
        if stage_name == "initializer":
            gate_label, gate_detail = self._initializer_gate_budget_status()
            gate_status = f" Rules gate: {gate_label} — {gate_detail}"
            current = getattr(self, "_gate_input_revision", 0)
            captured = getattr(self, "_last_bev_revision", None)
            if captured is not None and captured != current:
                bev_status = (
                    f" The latest BEV was captured at scene revision {captured} and is stale "
                    f"after a later scene edit (current revision {current}); do not rely on it "
                    "as current evidence. Call a fresh BEV when you actually need that "
                    "diagnostic again."
                )
        has_rules_gate = "check_rules_enforced" in getattr(
            getattr(self, "tool_client", None), "tool_to_server", {}
        )
        final_gate_grace = ""
        if has_rules_gate:
            final_gate_grace = (
                " If the FINAL normal call itself is check_rules_enforced and it returns "
                "ALL RULES PASS for the current scene revision, exactly one terminal "
                "end-only response will follow. That response cannot edit, render, undo, "
                "bypass, or recheck; no failed, stale, missing, or earlier gate result "
                "earns it."
            )
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[Round Budget]\n"
                        f"This is normal tool-call round {round_index + 1} of {max_rounds}{stage_text}. "
                        f"You have {rounds_left} normal round(s) left including this one, "
                        f"and {rounds_after_this} normal round(s) after this. "
                        "The budget is a CEILING, not a target: the moment this stage's "
                        "requirements are met and further edits would only chase marginal "
                        "gains, call end — finishing early is the better outcome, and unused "
                        "rounds cost nothing. Call end ONLY when the completion conditions "
                        "in the stage system prompt are satisfied. On the last normal round, "
                        "call end if they are satisfied; otherwise make the highest-impact "
                        "permitted tool call. If the stage remains incomplete afterward, the "
                        "orchestrator stops the attempt and records budget_exhausted. Budget "
                        "exhaustion never waives a required gate."
                        + final_gate_grace
                        + gate_status
                        + bev_status
                        + post_flip_status
                        + late_flip_status
                    ),
                }
            ],
        }

    def _post_flip_round_status(self) -> str:
        pending = self._pending_post_flip_followup()
        if pending is None:
            return ""
        if pending.get("protocol_error"):
            return (
                " POST-FLIP FOLLOW-UP: BLOCKED by malformed backend transaction "
                f"metadata ({pending['protocol_error']}); the attempt must fail closed."
            )
        object_text = ", ".join(pending["object_ids"])
        return (
            " MANDATORY POST-FLIP FOLLOW-UP: a retained rotate_180 made prior "
            f"measurements for {object_text} stale. The next accepted tool must be "
            f"investigate_objects containing {object_text}, or undo_last_step for "
            f"transaction {pending['token']}; every other tool is blocked."
        )

    def _initializer_gate_budget_status(self) -> tuple[str, str]:
        """Return the initializer's model-visible gate freshness state.

        Keep the three labels operationally distinct: NOT_PASSED means there is no
        successful decision to reuse, while STALE means a real prior pass inspected an
        older scene revision. The latter tells the model why ``end`` will be refused.
        """
        if not bool(getattr(self, "_rules_all_pass", False)):
            return (
                "NOT_PASSED",
                "check_rules_enforced has never passed, or its latest result failed; "
                "end will be refused",
            )

        if hasattr(self, "_gate_input_revision"):
            current = getattr(self, "_gate_input_revision")
            checked = getattr(self, "_rules_revision", None)
            if checked == current:
                return (
                    "CURRENT",
                    f"ALL RULES PASS applies to current scene revision {current}",
                )
            return (
                "STALE",
                f"last ALL RULES PASS checked revision {checked}; current revision "
                f"{current}; rerun check_rules_enforced before end",
            )

        if not bool(getattr(self, "_edited_since_rules", False)):
            return "CURRENT", "ALL RULES PASS applies to the current scene"
        return (
            "STALE",
            "the scene was edited after ALL RULES PASS; rerun check_rules_enforced "
            "before end",
        )

    def _terminal_end_tool_configs(self) -> list[dict[str, Any]]:
        """Return exactly one advertised ``end`` schema for terminal grace."""
        for tool in self._build_tool_configs():
            if tool.get("function", {}).get("name") == "end":
                return [tool]
        return []

    def _terminal_end_grace_message(self) -> dict[str, Any]:
        current = getattr(self, "_gate_input_revision", 0)
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "[Terminal End-Only Grace]\n"
                        "The final normal-round check_rules_enforced call returned ALL "
                        f"RULES PASS for the current scene revision {current}. Normal "
                        "tool-call rounds are exhausted. This is your one terminal-only "
                        "response: the only available tool is end, and no scene edit, "
                        "render, undo, bypass, investigation, or additional gate check is "
                        "possible. If you voluntarily certify that the stage's visual and "
                        "prompt-defined completion conditions are satisfied, call exactly "
                        "one end tool now. If the current advisory_requirement is required, "
                        "include its ID-only yaw_advisory_resolution in that end call; the "
                        "backend validates it synchronously and refuses mismatch/stale/"
                        "unverified submissions. Otherwise emit no tool call; the attempt will "
                        "remain budget_exhausted."
                    ),
                }
            ],
        }

    def _record_terminal_grace_decline(self, message: Any, reason: str) -> None:
        """Persist an invalid/declined grace response without executing any tool."""
        calls = list(getattr(message, "tool_calls", None) or [])
        raw = getattr(message, "raw_content", None)
        entry: dict[str, Any] = {
            **({"_raw_blocks": raw} if raw else {}),
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
        }
        if calls:
            entry["tool_calls"] = [call.model_dump() for call in calls]
        self.memory.append(entry)
        # Pair every unexecuted tool call so the saved transcript remains valid if it
        # is inspected or replayed. This is bookkeeping only; no MCP tool is invoked.
        for call in calls:
            self.memory.append(
                {
                    "role": "tool",
                    "name": call.function.name,
                    "tool_call_id": call.id,
                    "content": [{"type": "text", "text": reason}],
                }
            )
        if not calls:
            self.memory.append({"role": "user", "content": reason})
        self._save_terminal_grace_memory()

    def _record_terminal_grace_error(self, reason: str) -> None:
        """Persist a fail-closed optional-grace error without aborting the stage."""
        self.memory.append(
            {
                "role": "user",
                "content": "[Terminal End-Only Grace Error]\n" + reason,
            }
        )
        self._save_terminal_grace_memory()

    def _save_terminal_grace_memory(self) -> None:
        """Best-effort audit persistence for the optional grace path."""
        try:
            self._save_memory()
        except Exception:  # noqa: BLE001 - optional grace must fail closed, not abort
            logging.exception("could not persist terminal end-only grace audit")

    async def _run_terminal_end_grace(self) -> bool:
        """Offer one non-mutating, end-only response after a final-round fresh pass.

        Returning True requires one voluntary ``end`` call. Any missing, extra, or
        hallucinated tool call is recorded but never executed, and there is no retry.
        """
        # A pending post-flip investigation is stronger than a historical gate pass.
        # Never even offer end-only grace while current-pose measurements are stale.
        if self._post_flip_followup_pending():
            return False
        end_tools = self._terminal_end_tool_configs()
        if len(end_tools) != 1 or self._rules_passed_state() is not True:
            return False

        print("=== Terminal end-only grace ===\n")
        self._terminal_end_grace_offered = True
        # Unlike the transient normal-round budget hint, this marker is part of the
        # durable transcript so audits can distinguish a rescued cutoff from an
        # ordinary in-budget end.
        self.memory.append(self._terminal_end_grace_message())
        memory = self._memory_view()
        chat_args = {
            "model": self.config.get("model"),
            "messages": memory,
            "tools": end_tools,
            # ``auto`` preserves the voluntary certification contract: forcing a tool
            # choice would turn a machine gate pass into an implicit agent approval.
            "tool_choice": "auto",
        }
        try:
            response = get_model_response(self.client, chat_args)
            message = response.choices[0].message
        except Exception as exc:  # noqa: BLE001 - optional grace fails closed
            logging.warning("terminal end-only grace model request failed: %s", exc)
            self._record_terminal_grace_error(
                f"Model response failed ({exc}); the attempt remains budget_exhausted."
            )
            return False
        calls = list(getattr(message, "tool_calls", None) or [])
        if len(calls) != 1 or calls[0].function.name != "end":
            self._record_terminal_grace_decline(
                message,
                "Terminal grace declined: exactly one end call was required; no tool "
                "was executed and the attempt remains budget_exhausted.",
            )
            return False

        try:
            tool_arguments = json.loads(calls[0].function.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            self._record_terminal_grace_decline(
                message,
                "Terminal grace declined: the end call arguments were invalid; no tool "
                "was executed and the attempt remains budget_exhausted.",
            )
            return False
        if not isinstance(tool_arguments, dict):
            self._record_terminal_grace_decline(
                message,
                "Terminal grace declined: end arguments must be an object; no tool was "
                "executed and the attempt remains budget_exhausted.",
            )
            return False

        embedded_refusal = await self._resolve_embedded_yaw_advisory(tool_arguments)
        refusal = embedded_refusal or self._refuse_end_reason()
        if refusal:
            self._record_terminal_grace_decline(message, refusal)
            return False

        try:
            tool_response = await self.tool_client.call_tool("end", tool_arguments)
        except Exception as exc:  # noqa: BLE001 - optional grace fails closed
            logging.warning("terminal end-only grace end call failed: %s", exc)
            self._record_terminal_grace_decline(
                message,
                "Terminal grace end call failed ("
                + str(exc)
                + "); the attempt remains budget_exhausted.",
            )
            return False
        end_committed = not bool(tool_response.get("_tool_error", False))
        self._remember_tool_result("end", tool_arguments, tool_response)
        tool_response = await self._ensure_end_of_round_render(tool_response, "end")
        self._remember_tool_result("end", tool_arguments, tool_response)
        self._update_memory({"assistant": message, "user": tool_response})
        self._save_terminal_grace_memory()
        return end_committed

    def _memory_view(self) -> list[dict[str, Any]]:
        """The messages actually sent to the model. With ``memory_window`` set
        (composition only — root.py nulls it for every other stage) old rounds collapse
        to a text ledger and only the last M rounds keep their images. Every other stage
        sends its memory verbatim: append-only, no truncation, no image aging."""
        window = self.config.get("memory_window")
        if window:
            return self.prompt_builder.build_memory_windowed(
                self.memory, int(window), self.protected_head
            )
        return self.prompt_builder.build_memory_append_only(
            self.memory, self.protected_head
        )

    def _build_tool_configs(self) -> list[dict[str, Any]]:
        """Build tool configs for the generator."""
        configs = [x for v in self.tool_client.tool_configs.values() for x in v]
        pending = self._pending_post_flip_followup()
        if not pending or pending.get("protocol_error"):
            return configs

        # Make the model-facing contract match the hard dispatcher guard.  This is an
        # ergonomic aid, not the enforcement boundary: models can hallucinate an
        # unadvertised tool, and end is hosted by another MCP server.  If a mixed-version
        # deployment is missing both schemas, retain the full list so the API request is
        # still valid; every forbidden call remains blocked below.
        narrowed = [
            config
            for config in configs
            if config.get("function", {}).get("name") in self._POST_FLIP_ALLOWED_TOOLS
        ]
        return narrowed or configs

    def _pending_post_flip_followup(self) -> Optional[dict[str, Any]]:
        pending = getattr(self, "_post_flip_required_followup", None)
        return pending if isinstance(pending, dict) else None

    def _post_flip_followup_pending(self) -> bool:
        return self._pending_post_flip_followup() is not None

    @staticmethod
    def _followup_object_ids(value: Any) -> Optional[list[str]]:
        """Return a normalized non-empty, duplicate-free object-id list."""
        if not isinstance(value, list) or not value:
            return None
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return None
        normalized = [item.strip() for item in value]
        if len(set(normalized)) != len(normalized):
            return None
        return normalized

    def _normalize_required_followup(
        self, record: Any
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Validate the backend's retained-flip transaction contract.

        Invalid requirement data must not be treated like an absent requirement: the
        scene may already contain a committed semantic flip, so silently ignoring a
        malformed token would let the agent certify stale measurements.
        """
        if not isinstance(record, dict):
            return None, "required_followup is not an object"
        if record.get("type") != self._POST_FLIP_FOLLOWUP_TYPE:
            return None, "required_followup has an unknown type"
        object_ids = self._followup_object_ids(record.get("object_ids"))
        if object_ids is None:
            return None, "required_followup.object_ids must be unique non-empty strings"
        token = record.get("token")
        if not isinstance(token, str) or not token.strip():
            return None, "required_followup.token must be a non-empty string"
        allowed_tools = record.get("allowed_tools")
        if (
            not isinstance(allowed_tools, list)
            or any(not isinstance(name, str) for name in allowed_tools)
            or len(allowed_tools) != len(set(allowed_tools))
            or set(allowed_tools) != self._POST_FLIP_ALLOWED_TOOLS
        ):
            return (
                None,
                "required_followup.allowed_tools must contain exactly "
                "investigate_objects and undo_last_step",
            )
        return (
            {
                "type": self._POST_FLIP_FOLLOWUP_TYPE,
                "object_ids": object_ids,
                "token": token.strip(),
                "allowed_tools": sorted(self._POST_FLIP_ALLOWED_TOOLS),
            },
            None,
        )

    def _normalize_resolved_followup(
        self, record: Any
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not isinstance(record, dict):
            return None, "resolved_followup is not an object"
        if record.get("type") != self._POST_FLIP_FOLLOWUP_TYPE:
            return None, "resolved_followup has an unknown type"
        object_ids = self._followup_object_ids(record.get("object_ids"))
        if object_ids is None:
            return None, "resolved_followup.object_ids must be unique non-empty strings"
        token = record.get("token")
        if not isinstance(token, str) or not token.strip():
            return None, "resolved_followup.token must be a non-empty string"
        status = record.get("status")
        if status not in {"satisfied", "cancelled_by_undo"}:
            return None, "resolved_followup.status is invalid"
        return (
            {
                "type": self._POST_FLIP_FOLLOWUP_TYPE,
                "object_ids": object_ids,
                "token": token.strip(),
                "status": status,
            },
            None,
        )

    @staticmethod
    def _append_tool_text(response: dict[str, Any], text: str) -> None:
        current = response.get("text")
        if isinstance(current, list):
            current.append(text)
        elif current is None:
            response["text"] = [text]
        else:
            response["text"] = [str(current), text]

    def _set_malformed_post_flip_requirement(
        self,
        reason: str,
        tool_arguments: Any,
    ) -> None:
        # Preserve the requested id when possible for an actionable artifact, but do
        # not synthesize a transaction token.  With no trustworthy token, no later
        # response can prove that it resolved this exact edit; the attempt fails closed.
        object_id = (
            tool_arguments.get("object") if isinstance(tool_arguments, dict) else None
        )
        self._post_flip_required_followup = {
            "type": self._POST_FLIP_FOLLOWUP_TYPE,
            "object_ids": [object_id] if isinstance(object_id, str) else [],
            "token": None,
            "allowed_tools": [],
            "protocol_error": reason,
        }

    def _consume_post_flip_followup(
        self,
        tool_name: str,
        tool_arguments: Any,
        tool_response: Any,
    ) -> None:
        """Advance the post-flip state only from matching backend evidence."""
        if not isinstance(tool_response, dict):
            return

        pending_before = self._pending_post_flip_followup()
        is_error = bool(tool_response.get("_tool_error", False))

        # A backend pending record is authoritative even when the current tool call
        # itself returned an error.  This is how a stateless/direct caller resyncs
        # after a retained flip whose original success response was lost.
        if "required_followup" in tool_response:
            normalized, error = self._normalize_required_followup(
                tool_response.get("required_followup")
            )
            if error:
                self._set_malformed_post_flip_requirement(error, tool_arguments)
                self._append_tool_text(
                    tool_response,
                    "POST-FLIP FOLLOW-UP PROTOCOL ERROR: "
                    + error
                    + ". The committed scene cannot be certified because its exact "
                    "re-investigation transaction is unknown.",
                )
                return
            if pending_before is not None:
                same_requirement = bool(
                    not pending_before.get("protocol_error")
                    and normalized["type"] == pending_before.get("type")
                    and normalized["token"] == pending_before.get("token")
                    and set(normalized["object_ids"])
                    == set(pending_before.get("object_ids") or [])
                )
                if same_requirement:
                    self._append_tool_text(
                        tool_response,
                        "The mandatory post-rotate_180 follow-up remains pending; "
                        "the backend repeated its matching transaction identity.",
                    )
                    return
                self._append_tool_text(
                    tool_response,
                    "POST-FLIP FOLLOW-UP PROTOCOL ERROR: the backend issued a second "
                    "requirement while an earlier one was still pending; the earlier "
                    "transaction remains authoritative.",
                )
                return
            self._post_flip_required_followup = normalized
            self._append_tool_text(
                tool_response,
                "MANDATORY NEXT CALL: investigate_objects must include "
                + ", ".join(normalized["object_ids"])
                + "; the only alternative is undo_last_step for this exact rotate_180. "
                "No other tool may run first.",
            )
            return

        errored_rotate = bool(
            is_error
            and tool_name == "move"
            and isinstance(tool_arguments, dict)
            and tool_arguments.get("aspect") == "rotate_180"
        )
        if errored_rotate:
            mutation_state = tool_response.get("scene_mutation")
            if mutation_state == "not_committed":
                # The backend returned normally and explicitly proved that the
                # attempted flip never survived (including transactional rollback).
                return
            reason = (
                "errored rotate_180 reported a committed scene without a valid "
                "required_followup"
                if mutation_state == "committed"
                else "errored rotate_180 has unknown commit state and no valid "
                "required_followup"
            )
            self._set_malformed_post_flip_requirement(reason, tool_arguments)
            self._append_tool_text(
                tool_response,
                "POST-FLIP FOLLOW-UP PROTOCOL ERROR: "
                + reason
                + ". The prior rules pass is stale and end is blocked; this attempt "
                "must fail closed unless the backend transaction is reconciled.",
            )
            return

        if pending_before is None:
            return

        # A transport/tool failure never resolves the requirement, even if a malformed
        # server happens to attach resolution-looking metadata to the error payload.
        if is_error:
            self._append_tool_text(
                tool_response,
                "The mandatory post-rotate_180 follow-up remains pending because this "
                "tool call failed.",
            )
            return

        resolution_raw = tool_response.get("resolved_followup")
        resolution, error = self._normalize_resolved_followup(resolution_raw)
        expected_status = {
            "investigate_objects": "satisfied",
            "undo_last_step": "cancelled_by_undo",
        }.get(tool_name)
        identities_match = bool(
            resolution is not None
            and not pending_before.get("protocol_error")
            and resolution["type"] == pending_before.get("type")
            and resolution["token"] == pending_before.get("token")
            and set(resolution["object_ids"])
            == set(pending_before.get("object_ids") or [])
            and len(resolution["object_ids"])
            == len(pending_before.get("object_ids") or [])
            and resolution["status"] == expected_status
        )
        if identities_match:
            self._post_flip_last_resolution = deepcopy(resolution)
            self._post_flip_required_followup = None
            return

        detail = error or (
            "resolved_followup did not match the pending type, object ids, token, "
            "or expected tool outcome"
        )
        self._append_tool_text(
            tool_response,
            "POST-FLIP FOLLOW-UP NOT CLEARED: "
            + detail
            + ". Retry the required investigation or undo; all other tools remain "
            "blocked.",
        )

    @staticmethod
    def _post_flip_error_invalidates_rules(
        tool_name: str, tool_arguments: Any, tool_response: Any
    ) -> bool:
        """Whether an error proves or may hide a committed retained flip.

        A normal error is non-mutating.  ``rotate_180`` is different: its scene
        commit precedes response construction, so a transport/formatting failure is
        ambiguous unless the backend explicitly proves no commit.  A structured
        required-followup from any blocked Blender call also proves that such a flip
        exists and must stale a pre-flip rules pass.
        """
        if not isinstance(tool_response, dict) or not tool_response.get("_tool_error"):
            return False
        if "required_followup" in tool_response:
            return True
        if tool_response.get("scene_mutation") == "committed":
            return True
        return bool(
            tool_name == "move"
            and isinstance(tool_arguments, dict)
            and tool_arguments.get("aspect") == "rotate_180"
            and tool_response.get("scene_mutation") != "not_committed"
        )

    def _post_flip_tool_refusal(
        self, tool_name: str, tool_arguments: Any
    ) -> Optional[str]:
        """Return a pre-dispatch refusal while a retained flip is unmeasured."""
        pending = self._pending_post_flip_followup()
        if pending is None:
            return None
        protocol_error = pending.get("protocol_error")
        if protocol_error:
            return (
                "Tool refused: a retained rotate_180 produced malformed mandatory "
                f"follow-up metadata ({protocol_error}). No subsequent call can be "
                "accepted without a transaction identity; this attempt must fail closed."
            )

        required = list(pending["object_ids"])
        required_text = ", ".join(required)
        if tool_name == "undo_last_step":
            return None
        if tool_name == "investigate_objects":
            objects = (
                tool_arguments.get("objects")
                if isinstance(tool_arguments, dict)
                else None
            )
            provided = (
                {item for item in objects if isinstance(item, str)}
                if isinstance(objects, list)
                else set()
            )
            missing = [object_id for object_id in required if object_id not in provided]
            if not missing:
                return None
            return (
                "Tool refused: the first call after retained rotate_180 must be "
                "investigate_objects containing every flipped object. Missing: "
                + ", ".join(missing)
                + f". Required transaction token: {pending['token']}. The pending "
                "requirement was not consumed."
            )
        return (
            f"Tool {tool_name} refused: retained rotate_180 for {required_text} made "
            "all pre-flip measurements stale. The next accepted call must be "
            f"investigate_objects containing {required_text}, or undo_last_step for "
            f"transaction {pending['token']}. No move, render, execute, rules check, "
            "scene-info call, or end may occur first."
        )

    def _record_tool_refusal(self, message: Any, tool_call: Any, reason: str) -> None:
        """Append a paired transcript entry for a tool rejected before dispatch."""
        entry: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [tool_call.model_dump()],
        }
        raw = getattr(message, "raw_content", None)
        if raw:
            entry["_raw_blocks"] = raw
        self.memory.append(entry)
        self.memory.append(
            {
                "role": "tool",
                "name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "content": [{"type": "text", "text": reason}],
            }
        )
        self._save_memory()

    def _remember_tool_result(
        self,
        tool_name: Optional[str],
        tool_arguments: Optional[dict[str, Any]],
        tool_response: Optional[dict[str, Any]],
    ) -> None:
        """Remember the latest tool result for root-level verifier gating."""
        self.last_tool_name = tool_name
        self.last_tool_arguments = tool_arguments or {}
        self.last_tool_response = tool_response

    # Tools whose image[0] is a FULL-SCENE render of the current state. image[1], when
    # present, is the paired reference / pseudo-GT — never the render. Every OTHER
    # image-returning tool (investigate_objects, move, render_bev, initialize_viewpoint)
    # returns crops or alternate-camera views, which must never stand in for "the current
    # scene render" a verifier compares against the target photo.
    _FULL_SCENE_RENDER_TOOLS = frozenset(
        {
            "execute_and_evaluate",
            "build_root_surface",
            "remove_root_surface",
            "render_current_scene",
        }
    )

    def _remember_scene_render(
        self,
        tool_name: Optional[str],
        tool_arguments: Optional[dict[str, Any]],
        tool_response: Optional[dict[str, Any]],
    ) -> None:
        """Record the newest full-scene render this attempt produced, with its viewpoint.

        Root hands this explicit artifact to the paired verifier instead of discovering
        renders by filesystem glob."""
        if tool_name not in self._FULL_SCENE_RENDER_TOOLS or not tool_response:
            return
        if tool_response.get("image_kind") == "relocation_crop":
            return  # settle crops windowed on the moved object, not the whole scene
        images = tool_response.get("image") or []
        if not images:
            return
        args = tool_arguments or {}
        self._last_scene_render = {
            "path": images[0],
            "azimuth": float(args.get("azimuth") or 0.0),
            "elevation": float(args.get("elevation") or 0.0),
            "tool": tool_name,
        }

    def _update_memory(self, message: dict[str, Any]) -> None:
        """Update the conversation memory with the new assistant and tool messages.

        Args:
            message: Dictionary containing 'assistant' (the model response) and
                     'user' (the tool response with text, images, and optional verifier result).
        """
        # Add tool calling
        assistant_content = message["assistant"].content
        assistant_tool_calls = message["assistant"].tool_calls[0].model_dump()
        entry = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [assistant_tool_calls],
        }
        raw = getattr(message["assistant"], "raw_content", None)
        if raw:
            # T2.4 native API: the provider's own content blocks (incl. thinking),
            # replayed VERBATIM by the request translator — required for tool-use
            # continuations on the same model. Absent on the OpenAI-compat path.
            entry["_raw_blocks"] = raw
        self.memory.append(entry)

        # Add tool response
        tool_call_id = message["assistant"].tool_calls[0].id
        tool_call_name = message["assistant"].tool_calls[0].function.name
        tool_response = []
        user_response = []

        if "image" in message["user"]:
            response_texts = message["user"].get("text", [])
            response_images = message["user"].get("image", [])
            for idx, image in enumerate(response_images):
                text = (
                    response_texts[idx]
                    if idx < len(response_texts)
                    else f"Render image {idx}"
                )
                user_response.append({"type": "text", "text": text})
                user_response.append(
                    {
                        "type": "image_url",
                        # Tool-returned round image: agent-loop cap (768).
                        "image_url": {
                            "url": get_image_base64(image, max_edge=AGENT_VLM_EDGE)
                        },
                    }
                )
                user_response.append(
                    {
                        "type": "text",
                        "text": (
                            "Image loaded from local path: "
                            f"{display_path(image, self.config.get('scene_root'))}"
                        ),
                    }
                )
            for text in response_texts[len(response_images) :]:
                tool_response.append({"type": "text", "text": text})
        else:
            for text in message["user"]["text"]:
                tool_response.append({"type": "text", "text": text})
        if "verifier_result" in message["user"]:
            tool_response.append(
                {
                    "type": "text",
                    "text": "The following information is what the verifier agent returns to you: (1) Visual difference analysis between the current scene and the target scene (2) Suggested code modifications to follow.",
                }
            )
            for text in message["user"]["verifier_result"]["text"]:
                tool_response.append({"type": "text", "text": text})
        if "image" in message["user"]:
            n_images = len(message["user"].get("image") or [])
            tool_response.append(
                {
                    "type": "text",
                    "text": self._image_feedback_instruction(
                        n_images,
                        tool_name=self.last_tool_name,
                        image_kind=message["user"].get("image_kind"),
                    ),
                }
            )

        # The backend keeps structured gate evidence out of ordinary prose, but a
        # required yaw handoff is different: the model must copy the exact opaque
        # token and issued candidate ids into its next ID-only call.  Keep this
        # presentation local to generator memory.  Adding the token to the backend's
        # ``text`` would also put it in ``last_tool_response`` and leak it through the
        # otherwise-public verifier handoff.
        if (
            tool_call_name == "check_rules_enforced"
            and message["user"].get("rules_all_pass") is True
        ):
            requirement = message["user"].get("advisory_requirement")
            if isinstance(requirement, dict) and requirement.get("state") == "required":
                from lib.tools.geometry.yaw_advisory import (
                    normalize_advisory_requirement,
                )

                try:
                    bounded_requirement = normalize_advisory_requirement(requirement)
                except (TypeError, ValueError):
                    bounded_requirement = None
                if bounded_requirement is not None:
                    tool_response.append(
                        {
                            "type": "text",
                            "text": (
                                "CURRENT REQUIRED advisory_requirement (copy these "
                                "exact issued identifiers and opaque token; submit only "
                                "the ID-only resolver fields):\n"
                                + json.dumps(
                                    bounded_requirement,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=True,
                                )
                            ),
                        }
                    )

        self.memory.append(
            {
                "role": "tool",
                "content": tool_response,
                "name": tool_call_name,
                "tool_call_id": tool_call_id,
            }
        )
        if user_response:
            self.memory.append({"role": "user", "content": user_response})

        # Add initial plan. Guarded: a FAILED initialize_plan call returns a bare
        # {"text": [...]} dict (tool_client failure shapes / in-tool error) with no
        # "plan" key — skip the bookkeeping and let the model see the error text and
        # retry, instead of a KeyError killing the whole run.
        if tool_call_name == "initialize_plan" and message["user"].get("plan"):
            self.init_plan = "\n".join(message["user"]["plan"])
            self.memory.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your current plan (follow it):\n" + self.init_plan,
                        }
                    ],
                }
            )

    def _image_feedback_instruction(
        self,
        n_images: int = 2,
        tool_name: Optional[str] = None,
        image_kind: Optional[str] = None,
    ) -> str:
        ""
        from lib.prompts.static_scene.scopes import (
            LIGHTING_SCOPE_SHORT,
            TEXTURE_SCOPE_SHORT,
            composition_capability_enabled,
            effective_composition_scope_short,
            effective_initializer_scope_short,
        )

        stage = self.config.get("root_stage_name")
        harness_profile = self.config.get("harness_profile")
        harness_profile_manifest = self.config.get("harness_profile_manifest")
        composition_mesh_edit = composition_capability_enabled(
            harness_profile, harness_profile_manifest, "composition_mesh_edit"
        )
        scope = {
            "initializer": effective_initializer_scope_short(
                harness_profile, harness_profile_manifest
            ),
            "texture": TEXTURE_SCOPE_SHORT,
            "composition": effective_composition_scope_short(
                harness_profile, harness_profile_manifest
            ),
            "lighting": LIGHTING_SCOPE_SHORT,
        }.get(stage)
        if tool_name == "render_bev" and scope:
            return (
                "The next message contains a TOP-DOWN BEV captured from the scene at this "
                "moment (read-only — nothing was edited, so this is NOT a signal to undo). "
                "A later scene edit makes this historical BEV stale; call a fresh BEV when "
                "you actually need that diagnostic again. It is CAMERA-ALIGNED: "
                "image-UP = deeper into the scene (-Y), image-RIGHT = the camera's right (-X), "
                "so left/right here match left/right in the target photo. Use it ONLY for layout "
                "facts the perspective render hides: which SIDE of the table each wall sits on "
                "relative to the camera dot+arrow, the table footprint's yaw and extent, "
                "gaps/overlaps between surfaces, and whether the gray objects sit inside the "
                "table outline. Do NOT judge colors, materials, or lighting here (surfaces are "
                "flat id-tints), and do NOT compare it pixel-wise to the target photo (different "
                "projection). Fix any wrong-side or mis-yawed surface with execute_and_evaluate, "
                f"within this stage's scope ({scope})."
            )
        # Terse per-call pointers only — the crop format (IMAGE 1 = render / IMAGE 2 =
        # reference photo, boxed by id, locked (0,0) camera, how to compare / spot a
        # 180-reversal) is documented ONCE in the composition system prompt, so it is
        # not repeated on every move/investigate (it dominated the live window).
        if tool_name == "move" and scope:
            # A retained rotate_180 arms the mandatory post-flip investigation; the
            # generic "re-investigate if the change was large" reads as OPTIONAL right
            # under the MANDATORY NEXT CALL line, so that response gets its own pointer.
            if getattr(self, "_post_flip_required_followup", None):
                return (
                    "Post-move render CROP follows (reference camera) — context only. "
                    "It is NOT a substitute for the mandatory paired investigation: per "
                    "the line above, your next call MUST be investigate_objects "
                    "including the flipped object, or undo_last_step of this exact "
                    "rotate_180."
                )
            return (
                "Post-move render CROP follows (reference camera; compare against your "
                "latest investigate's photo crop; re-investigate if the change was large)."
            )
        if tool_name == "investigate_objects" and scope:
            return (
                "Two crops follow: IMAGE 1 = your render, IMAGE 2 = the reference photo "
                "(same window, each object boxed by id)."
            )
        if image_kind == "relocation_crop" and scope:
            return (
                "Two post-edit CROPs follow (locked reference camera, NOT the "
                "(azimuth, elevation) you passed): IMAGE 1 = your edited render, "
                "IMAGE 2 = the reference photo, same window, each moved object boxed by "
                "id in both. The objects your edit relocated were physics-settled. The "
                "settle note is a HINT — decide for yourself from the crops, and "
                "undo_last_step only if they show the object wrong or toppled."
            )
        if image_kind == "deduped_pair" and scope:
            return (
                "The next message contains ONE image: your scene render for the reference "
                "view. Its GROUND-TRUTH pair is the target photo attached in the FIRST "
                f"message of this conversation (unchanged). Compare them and judge ONLY within "
                f"this stage's scope ({scope}). If your last edit made this stage's result "
                "worse or regressed a done checklist item, call undo_last_step before a "
                "different in-scope edit."
            )
        if tool_name == "check_rules_enforced":
            # Its images are per-surface coverage-mismatch DIAGNOSTICS (one per failing
            # surface, count varies) — the generic branches below would caption two of
            # them as "render + reference", which they are not.
            return (
                "The images that follow are COVERAGE-MISMATCH DIAGNOSTICS, one per FAILING "
                "surface named in the text above (each shows that surface's reference-view "
                "footprint vs the photo's). They are NOT render/reference pairs. Use them to "
                "correct those surfaces' size/position/yaw so their coverage matches the "
                "reference frame, then re-run check_rules_enforced."
            )
        if scope and n_images < 2:
            return (
                "The next message contains ONE image from this tool (no paired reference) — inspect "
                f"it directly against the target image and judge ONLY within this stage's scope ({scope}). "
                "If your last edit made this stage's result worse or regressed a done checklist item, "
                "call undo_last_step before a different in-scope edit."
            )
        if stage == "composition":
            action = (
                "Use the returned evidence for an in-scope pose correction or the single "
                "target-evidenced mesh-repair route; if your last edit worsened the match "
                "or regressed a done checklist item, call undo_last_step before a different "
                "in-scope edit."
                if composition_mesh_edit
                else "Move/scale/rotate objects so your render matches the reference for this "
                "view; if your last edit worsened the match or regressed a done checklist item, "
                "call undo_last_step before a different layout edit."
            )
            return (
                "The next messages contain YOUR scene render AND the matching reference for the "
                "chosen (azimuth, elevation) view (the REAL target photo at (0,0); a pseudo-GT at a "
                f"novel view). Compare them and judge ONLY within this stage's scope ({scope}). "
                + action
            )
        if stage == "lighting":
            # This stage has no (azimuth, elevation) argument, so its reference is ALWAYS the
            # real photo — naming the pseudo-GT alternative here would re-offer a viewpoint
            # the tools no longer take. Same text otherwise.
            return (
                "The next user message contains TWO images: your scene render AND the REAL "
                f"target photo for that view. Compare them and judge ONLY within this stage's scope ({scope}). "
                "If your last edit made this stage's result worse or regressed a done checklist item, "
                "call undo_last_step before trying a different in-scope edit."
            )
        if scope:
            return (
                "The next user message contains TWO images: your scene render from the chosen view "
                "AND its reference (the REAL target photo at the reference view (0,0), or a pseudo-GT "
                f"at a novel view). Compare them and judge ONLY within this stage's scope ({scope}). "
                "If your last edit made this stage's result worse or regressed a done checklist item, "
                "call undo_last_step before trying a different in-scope edit."
            )
        return (
            "The next user message contains the normal render from this tool call. Inspect it directly against the target "
            "image before deciding the next code change, including stage-relevant visible differences."
        )

    async def _ensure_end_of_round_render(
        self, tool_response: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """Attach a normal render to each generator round when Blender is available."""
        if tool_response.pop("_tool_error", False):
            # a FAILED call gets no end-of-round render: appending two images to
            # an error (e.g. unknown investigate ids) buried the error text and
            # read as if the call half-worked — the agent should just re-call.
            return tool_response
        non_scene_tools = {
            "initialize_plan",
            "get_scene_info",
            "check_rules_enforced",
            # move: its response already carries the measured result (IoU/score
            # delta, settle tilt) from its own isolated renders — a full-scene
            # render + reference pair after EVERY move cost minutes per scene and
            # 2 images/round of context. Visual checks are on-demand
            # (investigate_objects / render_current_scene).
            "move",
        }
        if (
            tool_name == "end"
            or tool_name in non_scene_tools
            or "image" in tool_response
            # A successful-looking investigate with missing/mismatched structured
            # resolution is still fail-closed. Do not sneak an automatic render call
            # past the global pending-follow-up guard.
            or self._post_flip_followup_pending()
            or "render_current_scene" not in self.tool_client.tool_to_server
        ):
            return tool_response

        print("Render current scene for end-of-round generator feedback...")
        render_response = await self.tool_client.call_tool("render_current_scene", {})
        if "image" in render_response:
            original_text = tool_response.get("text", [])
            tool_response["text"] = render_response.get("text", []) + original_text
            tool_response["image"] = render_response["image"]
            if "image_kind" in render_response:
                # Carry the render's payload tag (e.g. CT3's "deduped_pair") into the
                # merged response — dropping it made the note composer caption a
                # deduped PAIR as "no paired reference" (fix6 composition msg 16).
                tool_response["image_kind"] = render_response["image_kind"]
            # Attribute it to the render, not to the host tool (whose name is not in
            # _FULL_SCENE_RENDER_TOOLS): a bare call is always the locked source view.
            self._remember_scene_render("render_current_scene", {}, render_response)
        elif "text" in render_response:
            tool_response.setdefault("text", [])
            tool_response["text"].extend(render_response["text"])
        return tool_response

    def _save_memory(self) -> None:
        """Save the conversation memory to a JSON file in the output directory."""
        output_file = self.config.get("output_dir") + "/" + self.memory_filename
        self.last_memory_path = output_file
        with open(output_file, "w") as f:
            json.dump(self.memory, f, indent=4, ensure_ascii=False)

    def _build_result(self) -> dict[str, Any]:
        """Return a summary artifact for root-stage handoff."""
        # ``moge_dir/scene_graph.json`` is the live graph: preprocessing creates
        # revision zero, and build_root_surface/remove_root_surface transactionally
        # revise that same artifact. Prefer it over an attempt-local diagnostic copy so
        # retries and downstream stages cannot receive a stale root inventory.
        md = self.config.get("moge_dir")
        active_graph_path = os.path.join(md, "scene_graph.json") if md else None
        if active_graph_path and os.path.exists(active_graph_path):
            scene_graph_json = active_graph_path
        else:
            scene_graph_json = self._optional_artifact_path("scene_graph.json")
        scene_graph = self._load_json_artifact(scene_graph_json)
        graph_is_current = bool(
            isinstance(scene_graph, dict) and "error" not in scene_graph
        )
        graph_revision = scene_graph_revision(scene_graph)
        runtime_root_edits = bounded_runtime_root_surface_edits(scene_graph)
        return {
            "agent_name": self.agent_name,
            "memory_file": self.last_memory_path
            or os.path.join(self.config.get("output_dir", ""), self.memory_filename),
            "output_dir": self.config.get("output_dir"),
            "agent_output_dir": self.config.get(
                "agent_output_dir", self.config.get("output_dir")
            ),
            "attempt_output_dir": self.config.get(
                "attempt_output_dir", self.config.get("output_dir")
            ),
            "attempt_idx": self.config.get("attempt_idx"),
            "stage_dir": self.config.get("stage_dir"),
            "init_plan": self.init_plan,
            "scene_graph_json": scene_graph_json,
            "scene_graph": scene_graph,
            "scene_graph_revision": graph_revision,
            # Bounded active-root provenance, bound to the graph snapshot above. It
            # survives a later rules check/end and is safe to serialize to the verifier.
            "runtime_root_surface_edits": runtime_root_edits,
            "runtime_root_surface_edits_revision": graph_revision,
            "runtime_root_surface_edits_current": graph_is_current,
            "last_tool_name": self.last_tool_name,
            "last_tool_arguments": self.last_tool_arguments,
            "last_tool_response": self.last_tool_response,
            # {path, azimuth, elevation, tool} of the newest full-scene render, or None.
            # Root's verifier hand-off prefers this over globbing renders/.
            "last_scene_render": getattr(self, "_last_scene_render", None),
            # Procedural cutoff: true only when normal rounds expired and the optional
            # terminal response did not complete the stage. Root uses this for rejection.
            "hit_max_rounds": bool(getattr(self, "_hit_max_rounds", False)),
            # Audit fields keep a grace-rescued cutoff distinguishable from an ordinary
            # in-budget end without changing root's existing hit_max_rounds semantics.
            "normal_rounds_exhausted": bool(
                getattr(self, "_normal_rounds_exhausted", False)
            ),
            "terminal_end_grace_offered": bool(
                getattr(self, "_terminal_end_grace_offered", False)
            ),
            "terminal_end_grace_accepted": bool(
                getattr(self, "_terminal_end_grace_accepted", False)
            ),
            "post_flip_required_followup": deepcopy(
                getattr(self, "_post_flip_required_followup", None)
            ),
            "post_flip_last_resolution": deepcopy(
                getattr(self, "_post_flip_last_resolution", None)
            ),
            # Keep voluntary success distinct from an orchestrator cutoff. The latter
            # never implies that `end` was called or that a stage gate passed.
            "completed_voluntarily": bool(
                getattr(self, "_completed_voluntarily", False)
            ),
            "termination_reason": getattr(
                self, "_termination_reason", "budget_exhausted"
            ),
            # Whether check_rules_enforced last reported ALL PASS with no edit since (a valid gate).
            # None for stages without the tool (texture/lighting); root gates only where it applies.
            "rules_passed": self._rules_passed_state(),
            "gate_input_revision": getattr(self, "_gate_input_revision", 0),
            "rules_revision": getattr(self, "_rules_revision", None),
            # Verbatim backend evidence from the latest gate call.  The revision/current
            # fields below prevent a stale record from being presented to the verifier
            # as current, while retaining that record in the attempt artifact for audits.
            "rule_evidence": deepcopy(getattr(self, "_rule_evidence", None)),
            "yaw_evidence": deepcopy(getattr(self, "_yaw_evidence", None)),
            "relationship_evidence": deepcopy(
                getattr(self, "_relationship_evidence", None)
            ),
            "constraint_results": deepcopy(getattr(self, "_constraint_results", None)),
            "advisory_requirement": deepcopy(
                getattr(self, "_advisory_requirement", None)
            ),
            "advisory_requirement_revision": getattr(
                self, "_advisory_requirement_revision", None
            ),
            "advisory_requirement_current": (
                self._current_advisory_requirement() is not None
            ),
            "yaw_resolution": deepcopy(getattr(self, "_yaw_resolution", None)),
            "yaw_resolution_revision": getattr(self, "_yaw_resolution_revision", None),
            "yaw_resolution_current": self._current_yaw_resolution() is not None,
            "completion_ready": self._completion_ready_state(),
            "rules_evidence_revision": getattr(self, "_rules_evidence_revision", None),
            "rules_evidence_current": self._rules_evidence_is_current(),
            "last_bev_revision": getattr(self, "_last_bev_revision", None),
            # the gate's yaw advisory (weak/absent anchor): the verifier's note
            # arms its visual yaw check with it (None when the gate measured yaw)
            "yaw_note": getattr(self, "_yaw_note", None),
        }

    async def _refresh_stale_rules_pass(self) -> None:
        """Gate PASSED but the agent squeezed in more edits before ending (e.g. a final
        material recolor, 0704_pm_real8210 attempt 1) — instead of guessing whether the edit
        was benign, RE-MEASURE: one headless gate call so ``rules_passed`` reflects the FINAL
        scene. A clean re-pass avoids a false procedural rejection; a real regression is
        caught with evidence (the rules-gap note's "last result FAILED" becomes literally
        true). Deliberately fires ONLY on a stale PASS — auto-running when the gate never ran
        (or last failed) would let agents skip the gate and free-ride on the recheck.
        A recheck error explicitly invalidates the old pass; unavailable evidence is
        never equivalent to a current PASS."""
        # This deterministic backend recheck is still a tool call.  Making it while a
        # flip follow-up is pending would violate the same immediate-next-call contract
        # enforced against model-selected tools, and could incorrectly certify stale
        # investigation coverage after budget exhaustion.
        if self._post_flip_followup_pending():
            return
        if not (
            getattr(self, "_rules_all_pass", False)
            and getattr(self, "_edited_since_rules", False)
        ):
            return
        if "check_rules_enforced" not in self.tool_client.tool_to_server:
            return
        print(
            "Rules gate passed earlier but edits followed — re-running the gate once..."
        )
        try:
            resp = await self.tool_client.call_tool("check_rules_enforced", {})
        except Exception as e:  # noqa: BLE001 - record a failed gate, keep artifact
            print(f"[generator] stale-pass recheck failed ({e}); invalidating old pass")
            self._rules_all_pass = False
            self._rules_revision = None
            self._rules_evidence_revision = None
            self._edited_since_rules = True
            return
        self._rules_all_pass = bool(resp.get("rules_all_pass", False))
        self._rules_revision = getattr(self, "_gate_input_revision", 0)
        self._capture_rules_evidence(resp)
        self._edited_since_rules = False
        print(f"[generator] stale-pass recheck: rules_all_pass={self._rules_all_pass}")

    @staticmethod
    def _yaw_note_from_evidence(evidence: Any) -> Optional[str]:
        ""
        if not isinstance(evidence, dict):
            return None
        applicability = evidence.get("applicability") or {}
        verdict = evidence.get("verdict") or {}
        selection = evidence.get("selection") or {}
        app_status = str(applicability.get("status") or "").lower()
        verdict_status = str(verdict.get("status") or "").lower()
        enforcement = str(selection.get("enforcement") or "").lower()
        if app_status in {"not_applicable", "not-applicable"}:
            return None
        needs_visual = (
            app_status == "unknown"
            or verdict_status in {"advisory", "unverified", "unknown"}
            or enforcement == "advisory"
        )
        if not needs_visual:
            return None
        reason = (
            verdict.get("reason")
            or selection.get("reason")
            or applicability.get("reason")
            or "structured yaw evidence is not strong enough for hard enforcement"
        )
        label = (
            "ADVISORY"
            if enforcement == "advisory"
            else (verdict_status.upper() or "UNVERIFIED")
        )
        return f"yaw {label} — {reason}"

    def _capture_rules_evidence(self, response: Any) -> None:
        """Capture one ``check_rules_enforced`` result at its inspected revision.

        Contract accepted from the Blender backend:

        * ``rule_evidence``: optional overall/level evidence mapping;
        * ``yaw_evidence``: schema-v1 structured applicability/anchor/verdict mapping;
        * ``relationship_evidence``: immutable source-relationship verdict rows;
        * ``constraint_results``: generated-scene fulfillment of compiled constraints.

        Missing fields clear the corresponding prior record.  This matters for the
        ordered ladder: a new STRUCTURE failure must not accidentally retain CONTACT
        evidence from an older check.  Deep copies prevent later response decoration
        from mutating the audit record.
        """
        response = response if isinstance(response, dict) else {}
        rule = response.get("rule_evidence")
        yaw = response.get("yaw_evidence")
        relationships = response.get("relationship_evidence")
        constraints = response.get("constraint_results")
        advisory = response.get("advisory_requirement")

        self._rule_evidence = deepcopy(rule) if isinstance(rule, dict) else None
        self._yaw_evidence = deepcopy(yaw) if isinstance(yaw, dict) else None
        self._relationship_evidence = (
            deepcopy(relationships) if isinstance(relationships, list) else None
        )
        self._constraint_results = (
            deepcopy(constraints) if isinstance(constraints, list) else None
        )
        self._rules_evidence_revision = getattr(self, "_gate_input_revision", 0)

        # Every gate check supersedes the previous request/resolution. Requirements
        # are emitted only after all three initializer levels pass; a missing or
        # malformed requirement on a fresh pass is retained as a protocol error so
        # ``end`` fails closed instead of guessing from yaw prose.
        self._advisory_requirement = None
        self._advisory_requirement_revision = None
        self._yaw_resolution = None
        self._yaw_resolution_revision = None
        self._advisory_protocol_error = None
        if advisory is not None:
            from lib.tools.geometry.yaw_advisory import (
                normalize_advisory_requirement,
            )

            try:
                self._advisory_requirement = normalize_advisory_requirement(advisory)
                self._advisory_requirement_revision = getattr(
                    self, "_gate_input_revision", 0
                )
            except (TypeError, ValueError) as exc:
                self._advisory_protocol_error = f"invalid advisory_requirement: {exc}"

        # Prefer the backend's explicit compatibility note, then derive it from the
        # schema.  This removes root's dependency on the executor's historical regex.
        self._yaw_note = response.get("yaw_note") or self._yaw_note_from_evidence(
            self._yaw_evidence
        )

    def _rules_evidence_is_current(self) -> bool:
        revision = getattr(self, "_rules_evidence_revision", None)
        current = getattr(self, "_gate_input_revision", 0)
        has_evidence = any(
            getattr(self, attr, None) is not None
            for attr in (
                "_rule_evidence",
                "_yaw_evidence",
                "_relationship_evidence",
                "_constraint_results",
                "_advisory_requirement",
            )
        )
        return has_evidence and revision is not None and revision == current

    def _current_rules_evidence(self, kind: str) -> Any:
        """Return structured evidence only when it describes the current scene."""
        if not self._rules_evidence_is_current():
            return None
        attr = {
            "rule": "_rule_evidence",
            "yaw": "_yaw_evidence",
            "relationships": "_relationship_evidence",
            "constraints": "_constraint_results",
            "constraint_results": "_constraint_results",
        }[kind]
        return deepcopy(getattr(self, attr, None))

    def _tool_invalidates_rules(
        self, tool_name: str, tool_response: Optional[dict[str, Any]]
    ) -> bool:
        """Whether a successful call changed a rules-gate input.

        Tool servers declare effects during initialization. The small fallback set
        keeps older/custom Blender servers safe until they publish metadata too.
        Failed/rejected calls do not advance the revision because they did not commit.
        """
        if not isinstance(tool_response, dict) or tool_response.get("_tool_error"):
            return False
        effects = getattr(self.tool_client, "tool_effects", {}).get(tool_name, {})
        if effects.get("mutates_scene") or effects.get("invalidates_rules"):
            return True
        return tool_name in {
            "execute_and_evaluate",
            "build_root_surface",
            "remove_root_surface",
            "move",
            "undo_last_step",
            "bypass",
        }

    def _tool_mutates_scene(
        self, tool_name: str, tool_response: Optional[dict[str, Any]]
    ) -> bool:
        """Whether a successful tool call changed Blender scene state (not just gate state)."""
        if not isinstance(tool_response, dict) or tool_response.get("_tool_error"):
            return False
        effects = getattr(self.tool_client, "tool_effects", {}).get(tool_name, {})
        if "mutates_scene" in effects:
            return bool(effects["mutates_scene"])
        return tool_name in {
            "execute_and_evaluate",
            "build_root_surface",
            "remove_root_surface",
            "move",
            "undo_last_step",
        }

    async def _resolve_embedded_yaw_advisory(
        self, end_arguments: dict[str, Any]
    ) -> Optional[str]:
        """Resolve an embedded submission only for a current required yaw advisory.

        Null and unused submissions are removed before dispatch to generator-base;
        the ordinary completion gate still enforces required or missing evidence.
        The model cannot use this shortcut to smuggle measurements or a conclusion:
        the strict backend validates the same ID schema as a standalone call.
        """
        if not isinstance(end_arguments, dict):
            return "end refused: end arguments must be an object."
        submission = end_arguments.pop("yaw_advisory_resolution", None)
        if submission is None:
            return None
        requirement = self._current_advisory_requirement()
        if requirement is None or requirement.get("state") != "required":
            return None
        if "resolve_yaw_advisory" not in getattr(
            self.tool_client, "tool_to_server", {}
        ):
            return (
                "end refused: this initializer does not expose the required "
                "resolve_yaw_advisory backend tool."
            )
        if not isinstance(submission, dict):
            return (
                "end refused: yaw_advisory_resolution must be an ID-only object "
                "copied from the current advisory request."
            )
        try:
            response = await self.tool_client.call_tool(
                "resolve_yaw_advisory", submission
            )
        except Exception as exc:  # noqa: BLE001 - optional shortcut fails closed
            return f"end refused: yaw advisory resolution transport failed ({exc})."
        self._capture_yaw_resolution(response)
        resolution = self._current_yaw_resolution()
        status = str((resolution or {}).get("status") or "unverified")
        if status in {"verified_match", "not_applicable"}:
            return None
        measurements = (resolution or {}).get("measurements") or {}
        delta = measurements.get("max_delta_degrees_mod_90")
        suffix = (
            f" (backend max delta {float(delta):.1f}deg)" if delta is not None else ""
        )
        return (
            f"end refused: embedded yaw advisory resolution returned {status}{suffix}. "
            "For a mismatch, correct the support and rerun the full rules gate; for "
            "stale/unverified evidence, use only a complete current candidate basis."
        )

    def _current_advisory_requirement(self) -> Optional[dict[str, Any]]:
        revision = getattr(self, "_advisory_requirement_revision", None)
        current = getattr(self, "_gate_input_revision", 0)
        requirement = getattr(self, "_advisory_requirement", None)
        if isinstance(requirement, dict) and revision == current:
            return deepcopy(requirement)
        return None

    def _current_yaw_resolution(self) -> Optional[dict[str, Any]]:
        revision = getattr(self, "_yaw_resolution_revision", None)
        current = getattr(self, "_gate_input_revision", 0)
        resolution = getattr(self, "_yaw_resolution", None)
        requirement = self._current_advisory_requirement()
        if not isinstance(resolution, dict) or revision != current:
            return None
        if requirement and requirement.get("state") != "not_required":
            if resolution.get("advisory_id") != requirement.get("advisory_id"):
                return None
        return deepcopy(resolution)

    def _completion_ready_state(self) -> Optional[bool]:
        """Whether every backend-owned completion gate is current and satisfied."""
        rules_state = self._rules_passed_state()
        if rules_state is not True:
            return rules_state
        if "resolve_yaw_advisory" not in getattr(
            self.tool_client, "tool_to_server", {}
        ):
            return True
        requirement = self._current_advisory_requirement()
        if requirement is None:
            return False
        if requirement.get("state") in {"manual_review", "not_required"}:
            return True
        return (self._current_yaw_resolution() or {}).get("status") == "verified_match"

    def _capture_yaw_resolution(self, response: Any) -> None:
        from lib.tools.geometry.yaw_advisory import normalize_yaw_resolution

        raw = response.get("yaw_resolution") if isinstance(response, dict) else None
        try:
            resolution = normalize_yaw_resolution(raw)
        except (TypeError, ValueError) as exc:
            self._yaw_resolution = None
            self._yaw_resolution_revision = None
            self._advisory_protocol_error = f"invalid yaw_resolution: {exc}"
            return
        requirement = self._current_advisory_requirement()
        if (
            requirement
            and requirement.get("state") != "not_required"
            and resolution.get("advisory_id") != requirement.get("advisory_id")
        ):
            self._yaw_resolution = None
            self._yaw_resolution_revision = None
            self._advisory_protocol_error = (
                "yaw_resolution advisory_id does not match current requirement"
            )
            return
        self._yaw_resolution = resolution
        self._yaw_resolution_revision = getattr(self, "_gate_input_revision", 0)
        self._advisory_protocol_error = None

    def _refuse_end_reason(self) -> Optional[str]:
        """Bounce a voluntary `end` while a stage's rules gate is not current.

        ``None`` from _rules_passed_state means this stage has no rules tool, so its
        ordinary prompt-defined completion criteria govern. The finite normal-round
        budget bounds repeated invalid end attempts; accepting one after an arbitrary
        refusal count would contradict the hard gate.
        """
        rules_state = self._rules_passed_state()
        if rules_state is False:
            return (
                "end refused: check_rules_enforced has not PASSED on the current scene "
                "state (it was never run, its last result FAILED, or the scene was edited "
                "after it passed). Run check_rules_enforced now, fix what it lists "
                "(investigation coverage / penetration), and call end only after ALL "
                "RULES PASS."
            )
        # No rules tool means no residual-yaw protocol (texture/lighting).
        if rules_state is None:
            return None
        has_advisory_tool = "resolve_yaw_advisory" in getattr(
            getattr(self, "tool_client", None), "tool_to_server", {}
        )
        if not has_advisory_tool:
            return None
        requirement = self._current_advisory_requirement()
        if requirement is None:
            detail = getattr(self, "_advisory_protocol_error", None)
            return (
                "end refused: the current initializer rules pass did not provide a "
                "valid advisory_requirement"
                + (f" ({detail})" if detail else "")
                + ". Rerun check_rules_enforced; completion cannot infer this state "
                "from prose."
            )
        state = requirement.get("state")
        if state in {"manual_review", "not_required"}:
            return None
        resolution = self._current_yaw_resolution()
        status = str((resolution or {}).get("status") or "unresolved")
        if status == "verified_match":
            return None
        measurements = (resolution or {}).get("measurements") or {}
        delta = measurements.get("max_delta_degrees_mod_90")
        delta_note = (
            f"; backend max edge delta={float(delta):.1f}deg"
            if delta is not None
            else ""
        )
        return (
            "end refused: the current residual yaw advisory is machine-resolvable "
            f"but its status is {status}{delta_note}. Call resolve_yaw_advisory with "
            "only the current issued candidate IDs/token. A verified mismatch requires "
            "a support correction and a fresh full rules pass."
        )

    def _rules_passed_state(self) -> Optional[bool]:
        """True iff the check_rules_enforced gate last passed AND nothing was edited after it.
        None when this stage does not have the tool (so root won't gate on it)."""
        has_tool = "check_rules_enforced" in getattr(
            self.tool_client, "tool_to_server", {}
        )
        if not has_tool:
            return None
        if self._post_flip_followup_pending():
            return False
        if hasattr(self, "_gate_input_revision"):
            current = getattr(self, "_gate_input_revision")
            checked = getattr(self, "_rules_revision", None)
            return bool(getattr(self, "_rules_all_pass", False)) and checked == current
        # Compatibility for focused tests and custom subclasses instantiated without
        # running __init__. Production agents always use revision binding above.
        return bool(getattr(self, "_rules_all_pass", False)) and not getattr(
            self, "_edited_since_rules", False
        )

    def _optional_artifact_path(self, filename: str) -> Optional[str]:
        """Return an artifact path if it exists in this agent's artifact directories."""
        candidate_dirs = [
            self.config.get("attempt_output_dir"),
            self.config.get("output_dir"),
            self.config.get("agent_output_dir"),
        ]
        seen = set()
        for output_dir in candidate_dirs:
            if not output_dir or output_dir in seen:
                continue
            seen.add(output_dir)
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                return path
        return None

    def _load_json_artifact(self, path: Optional[str]) -> Optional[dict[str, Any]]:
        """Load a JSON artifact into the stage summary when available."""
        if not path:
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load JSON artifact {path}: {e}"}

    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()
