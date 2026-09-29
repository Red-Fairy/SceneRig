#!/usr/bin/env python3
"""Serve a live SceneRig job dashboard from an output directory."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

STAGE_ORDER = [
    "initializer",
    "texture",
    "lighting",
    "composition",
]
AGENT_STAGE_NAMES = {
    "InitializerPlannerAgent": "initializer",
    "InitializerPlannerVerifierAgent": "initializer",
    "TextureAgent": "texture",
    "TextureVerifierAgent": "texture",
    "CompositionAgent": "composition",
    "CompositionVerifierAgent": "composition",
    "LightingAgent": "lighting",
    "LightingVerifierAgent": "lighting",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MEMORY_SUFFIX = "_memory.json"
MAX_TEXT = 14000
MAX_FIELD_TEXT = 10000
MAX_MEMORY_TEXT = 40000
DATA_IMAGE_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
PATH_IMAGE_RE = re.compile(r"(/[^ \n\t'\"<>]+\.(?:png|jpg|jpeg|webp))", re.IGNORECASE)


@dataclass
class DashboardConfig:
    output_dir: Path
    repo_root: Path
    target_dir: Optional[Path]
    host: str
    port: int


def safe_read_text(path: Path, limit: int = 200_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > limit:
                f.seek(max(0, size - limit))
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def safe_read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def truncate_text(text: str, limit: int = MAX_TEXT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n[truncated]"


def file_url(path: Path) -> str:
    return "/file?path=" + quote(str(path))


def path_payload(path: Optional[Path], root: Path) -> Optional[dict[str, Any]]:
    if not path:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "path": str(path),
        "name": path.name,
        "relative": str(path.relative_to(root))
        if is_relative_to(path, root)
        else str(path),
        "mtime": stat.st_mtime,
        "mtime_label": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        ),
        "size": stat.st_size,
        "url": file_url(path),
    }


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def iter_files(root: Path, suffixes: Optional[Iterable[str]] = None) -> Iterable[Path]:
    if not root.exists():
        return
    normalized = {s.lower() for s in suffixes} if suffixes else None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            path = Path(dirpath) / filename
            if normalized is None or path.suffix.lower() in normalized:
                yield path


def is_verifier_multiview_render(path: Path) -> bool:
    """Skip verifier initialize_viewpoint contact-sheet frames 1.png..4.png."""
    parts = path.parts
    if "investigator" not in parts or "renders" not in parts:
        return False
    return path.stem in {"1", "2", "3", "4"}


def newest_file(paths: Iterable[Path]) -> Optional[Path]:
    newest: Optional[Path] = None
    newest_mtime = -1.0
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest = path
            newest_mtime = mtime
    return newest


def is_task_dir(path: Path) -> bool:
    return (path / "task.log").exists() or (path / "blender_file.blend").exists()


def task_dirs_in_run(run_dir: Path) -> list[Path]:
    if is_task_dir(run_dir):
        return [run_dir]
    if not run_dir.exists():
        return []
    return [
        child
        for child in sorted(run_dir.iterdir())
        if child.is_dir() and is_task_dir(child)
    ]


def run_entry_name(run_dir: Path, task_dir: Optional[Path], output_dir: Path) -> str:
    if task_dir is None:
        if is_relative_to(run_dir, output_dir):
            return str(run_dir.relative_to(output_dir)) or run_dir.name
        return run_dir.name
    if run_dir == task_dir:
        return task_dir.name
    if is_relative_to(task_dir, output_dir):
        return str(task_dir.relative_to(output_dir))
    return f"{run_dir.name}/{task_dir.name}"


def run_entries(
    run_dir: Path, task_dirs: list[Path], output_dir: Path
) -> list[dict[str, Any]]:
    if not task_dirs:
        return [
            {
                "run_dir": run_dir,
                "task_dirs": [],
                "name": run_entry_name(run_dir, None, output_dir),
                "parent_name": run_dir.name,
                "scene_name": "",
            }
        ]
    return [
        {
            "run_dir": run_dir,
            "task_dirs": [task_dir],
            "name": run_entry_name(run_dir, task_dir, output_dir),
            "parent_name": run_dir.name if run_dir != task_dir else "",
            "scene_name": task_dir.name,
        }
        for task_dir in task_dirs
    ]


def natural_sort_parts(text: str) -> list[Any]:
    parts: list[Any] = []
    for part in re.split(r"(\d+)", text.lower()):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return parts


def run_name_sort_key(run: dict[str, Any]) -> tuple:
    parent = str(run.get("parent_name") or run.get("name") or "")
    scene = str(run.get("scene_name") or "")
    # Timestamp-like run folders are shown newest first, but still by folder name
    # rather than activity time so the sidebar does not jump while jobs update.
    timestamp_match = re.fullmatch(r"(\d{8})_(\d{6})", parent)
    if timestamp_match:
        date_part, time_part = timestamp_match.groups()
        return (0, -int(date_part), -int(time_part), natural_sort_parts(scene))
    return (1, natural_sort_parts(parent), natural_sort_parts(scene))


def discover_runs(output_dir: Path) -> list[dict[str, Any]]:
    """Discover monitorable runs below the selected directory.

    Supported inputs:
    - a single task dir: output/<run>/<task>
    - a single run dir: output/<run>, split into one entry per task
    - a run root: output, where new timestamp dirs appear over time
    """
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        return []

    if is_task_dir(output_dir):
        return run_entries(output_dir, [output_dir], output_dir)

    direct_tasks = task_dirs_in_run(output_dir)
    if direct_tasks:
        return run_entries(output_dir, direct_tasks, output_dir)

    runs: list[dict[str, Any]] = []
    for child in sorted(output_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        task_dirs = task_dirs_in_run(child)
        if task_dirs or (child / "args.json").exists():
            runs.extend(run_entries(child, task_dirs, output_dir))
    return runs


def stage_from_agent(agent_name: str) -> str:
    if agent_name in AGENT_STAGE_NAMES:
        return AGENT_STAGE_NAMES[agent_name]
    lowered = agent_name.lower()
    for stage in STAGE_ORDER:
        if stage in lowered:
            return stage
    return "unknown"


def parse_attempt(path: Path, task_dir: Path) -> Optional[dict[str, Any]]:
    parts = path.parts
    if "stages" not in parts:
        return None
    try:
        idx = parts.index("stages")
        stage_index = parts[idx + 1]
        agent_name = parts[idx + 2]
        attempt_name = parts[idx + 3]
    except IndexError:
        return None
    if not attempt_name.startswith("attempt_"):
        return None

    role = "verifier" if "Verifier" in agent_name else "generator"
    memory_path = newest_file(p for p in path.glob(f"*{MEMORY_SUFFIX}") if p.is_file())
    memory = safe_read_json(memory_path) if memory_path else None
    latest = summarize_memory(memory or [])
    render_paths = []
    if role == "generator":
        render_paths = sorted(
            (
                render_path
                for render_path in iter_files(path, IMAGE_SUFFIXES)
                if not is_verifier_multiview_render(render_path)
            ),
            key=lambda p: (p.stat().st_mtime if p.exists() else 0, str(p)),
        )
    render = render_paths[-1] if render_paths else None
    script = (
        newest_file(p for p in (path / "scripts").glob("*.py"))
        if (path / "scripts").exists()
        else None
    )

    try:
        mtime = max(
            [path.stat().st_mtime]
            + [
                p.stat().st_mtime
                for p in [memory_path, render, script]
                if p is not None
            ]
        )
    except OSError:
        mtime = 0.0

    verifier_result = latest.get("verifier_result")
    approved = None
    if verifier_result:
        approved = verifier_result.get("approved")
    elif "Verifier" in agent_name:
        end_text = latest.get("last_tool_text", "")
        match = re.search(r"Approved:\s*(True|False)", end_text)
        if match:
            approved = match.group(1) == "True"

    return {
        "stage_index": stage_index,
        "stage": stage_from_agent(agent_name),
        "agent": agent_name,
        "role": role,
        "attempt": attempt_name.replace("attempt_", ""),
        "path": str(path),
        "relative": str(path.relative_to(task_dir)),
        "mtime": mtime,
        "mtime_label": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        if mtime
        else "",
        "memory": path_payload(memory_path, task_dir),
        "render": path_payload(render, task_dir),
        "renders": [
            payload
            for payload in (
                path_payload(render_path, task_dir) for render_path in render_paths
            )
            if payload
        ],
        "script": path_payload(script, task_dir),
        "latest": latest,
        "approved": approved,
    }


def summarize_memory(memory: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(memory, list):
        return {}

    latest_assistant = None
    latest_tool = None
    latest_user = None
    latest_image_path = None
    for item in memory:
        role = item.get("role")
        if role == "assistant":
            latest_assistant = item
        elif role == "tool":
            latest_tool = item
        elif role == "user":
            latest_user = item
            found = image_path_from_content(item.get("content"))
            if found:
                latest_image_path = found

    tool_call = None
    assistant_text = ""
    if latest_assistant:
        assistant_text = content_to_text(latest_assistant.get("content"))
        calls = latest_assistant.get("tool_calls") or []
        if calls:
            call = calls[0]
            function = call.get("function", {})
            arguments = function.get("arguments", "")
            parsed_args = parse_json_string(arguments)
            tool_call = {
                "name": function.get("name", ""),
                "arguments": compact_value(
                    parsed_args if parsed_args is not None else arguments
                ),
                "raw_arguments": truncate_text(arguments, MAX_FIELD_TEXT),
            }

    last_tool_text = content_to_text(latest_tool.get("content")) if latest_tool else ""
    current_vlm_input = find_latest_vlm_input(memory, latest_assistant)
    verifier_result = parse_verifier_result(last_tool_text)

    return {
        "message_count": len(memory),
        "assistant_text": truncate_text(assistant_text),
        "tool_call": tool_call,
        "last_tool_name": latest_tool.get("name") if latest_tool else "",
        "last_tool_text": truncate_text(last_tool_text),
        "current_vlm_input": truncate_text(current_vlm_input),
        "latest_user_text": truncate_text(
            content_to_text(latest_user.get("content")) if latest_user else ""
        ),
        "latest_image_path_hint": latest_image_path,
        "verifier_result": verifier_result,
    }


def parse_json_string(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def compact_value(value: Any, text_limit: int = MAX_FIELD_TEXT, depth: int = 0) -> Any:
    if depth > 5:
        return "[nested value truncated]"
    if isinstance(value, str):
        return truncate_text(value, text_limit)
    if isinstance(value, list):
        kept = [compact_value(item, text_limit, depth + 1) for item in value[:30]]
        if len(value) > 30:
            kept.append(f"[{len(value) - 30} more items truncated]")
        return kept
    if isinstance(value, dict):
        return {
            str(key): compact_value(val, text_limit, depth + 1)
            for key, val in list(value.items())[:60]
        }
    return value


def parse_verifier_result(text: str) -> Optional[dict[str, Any]]:
    if not text or "Approved:" not in text:
        return None
    match = re.search(r"Approved:\s*(True|False)", text)
    return {
        "approved": match.group(1) == "True" if match else None,
        "text": truncate_text(text, 3000),
    }


def image_path_from_content(content: Any) -> Optional[str]:
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        marker = "Image loaded from local path:"
        if isinstance(text, str) and marker in text:
            return text.split(marker, 1)[1].strip()
    return None


def find_latest_vlm_input(
    memory: list[dict[str, Any]], latest_assistant: Optional[dict[str, Any]]
) -> str:
    if latest_assistant and latest_assistant in memory:
        search = memory[: memory.index(latest_assistant)]
    else:
        search = memory
    chunks: list[str] = []
    for item in reversed(search):
        role = item.get("role")
        if role not in {"user", "tool"}:
            continue
        text = content_to_text(item.get("content"))
        if text:
            chunks.append(f"{role}: {text}")
        if len("\n\n".join(chunks)) > 3500:
            break
    return "\n\n".join(reversed(chunks))


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    chunks.append("[image]")
                else:
                    chunks.append(str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def dashboard_allowed_roots(config: DashboardConfig) -> list[Path]:
    return [config.output_dir.resolve(), config.repo_root.resolve() / "data"]


def resolve_dashboard_file(raw_path: str, config: DashboardConfig) -> Optional[Path]:
    path = Path(unquote(raw_path)).resolve()
    if any(is_relative_to(path, root) for root in dashboard_allowed_roots(config)):
        return path
    return None


def memory_image_url(
    value: str, memory_path: Path, config: DashboardConfig
) -> Optional[str]:
    value = value.strip()
    if not value:
        return None
    if value.startswith("data:image/"):
        return value
    if value.startswith("/file?"):
        return value
    if value.startswith("file://"):
        value = value[7:]
    if value.startswith(("http://", "https://")):
        return value
    candidates = []
    if value.startswith("/"):
        candidates.append(Path(value))
    else:
        candidates.append(memory_path.parent / value)
        candidates.append(config.repo_root / value)
        candidates.append(config.output_dir / value)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            resolved.suffix.lower() in IMAGE_SUFFIXES
            and resolved.exists()
            and any(
                is_relative_to(resolved, root)
                for root in dashboard_allowed_roots(config)
            )
        ):
            return file_url(resolved)
    return None


def compact_memory_text(text: str) -> str:
    text = DATA_IMAGE_RE.sub("[embedded image]", text)
    text = PATH_IMAGE_RE.sub("[image]", text)
    return truncate_text(text, MAX_MEMORY_TEXT)


def text_memory_parts(
    text: str, memory_path: Path, config: DashboardConfig,
    include_path_images: bool = True,
) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = [{"type": "text", "text": compact_memory_text(text)}]
    seen = set()
    for match in DATA_IMAGE_RE.finditer(text):
        url = match.group(0)
        if url not in seen:
            parts.append({"type": "image", "url": url})
            seen.add(url)
    # Avoid resolving path captions when the same message embeds its image.
    if include_path_images:
        for match in PATH_IMAGE_RE.finditer(text):
            image = memory_image_url(match.group(1), memory_path, config)
            if image and image not in seen:
                parts.append({"type": "image", "url": image})
                seen.add(image)
    return parts


def memory_parts(
    content: Any, memory_path: Path, config: DashboardConfig
) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    if isinstance(content, str):
        image = memory_image_url(content, memory_path, config)
        if image:
            parts.append({"type": "image", "url": image})
        else:
            parts.extend(text_memory_parts(content, memory_path, config))
        return parts
    if isinstance(content, list):
        # Path captions for embedded images would otherwise duplicate the preview.
        has_embedded = any(
            isinstance(i, dict) and i.get("type") == "image_url" for i in content
        )
        for item in content:
            if isinstance(item, str):
                parts.extend(memory_parts(item, memory_path, config))
                continue
            if not isinstance(item, dict):
                parts.append(
                    {
                        "type": "text",
                        "text": compact_memory_text(
                            json.dumps(item, ensure_ascii=False)
                        ),
                    }
                )
                continue
            item_type = item.get("type", "")
            if item_type == "image_url":
                raw_url = (
                    item.get("image_url", {}).get("url")
                    if isinstance(item.get("image_url"), dict)
                    else item.get("image_url")
                )
                image = memory_image_url(str(raw_url or ""), memory_path, config)
                if image:
                    parts.append({"type": "image", "url": image})
                continue
            if item_type in {"text", "input_text", "output_text"}:
                parts.extend(
                    text_memory_parts(
                        str(item.get("text", "")), memory_path, config,
                        include_path_images=not has_embedded,
                    )
                )
                continue
            text = json.dumps(item, ensure_ascii=False, indent=2)
            parts.extend(text_memory_parts(text, memory_path, config))
        return parts
    if content is not None:
        parts.append(
            {
                "type": "text",
                "text": compact_memory_text(
                    json.dumps(content, ensure_ascii=False, indent=2)
                ),
            }
        )
    return parts


def collect_memory(raw_path: str, config: DashboardConfig) -> dict[str, Any]:
    path = resolve_dashboard_file(raw_path, config)
    if (
        not path
        or not path.exists()
        or not path.is_file()
        or not path.name.endswith(MEMORY_SUFFIX)
    ):
        raise FileNotFoundError("Memory file not found")
    memory = safe_read_json(path)
    if not isinstance(memory, list):
        memory = []
    messages = []
    for idx, item in enumerate(memory):
        if not isinstance(item, dict):
            item = {"role": "unknown", "content": item}
        role = str(item.get("role") or item.get("type") or "unknown")
        parts = memory_parts(item.get("content"), path, config)
        tool_calls = item.get("tool_calls") or []
        if tool_calls:
            parts.extend(
                text_memory_parts(
                    json.dumps(tool_calls, ensure_ascii=False, indent=2), path, config
                )
            )
        messages.append(
            {
                "index": idx + 1,
                "role": role,
                "name": str(item.get("name") or ""),
                "parts": parts or [{"type": "text", "text": ""}],
            }
        )
    return {
        "path": str(path),
        "name": path.name,
        "relative": str(path.relative_to(config.output_dir))
        if is_relative_to(path, config.output_dir)
        else str(path),
        "message_count": len(messages),
        "messages": messages,
    }


def collect_attempts(task_dir: Path) -> list[dict[str, Any]]:
    attempts = []
    stages_dir = task_dir / "stages"
    if not stages_dir.exists():
        return attempts
    for attempt_dir in sorted(stages_dir.glob("*/*/attempt_*")):
        if attempt_dir.is_dir():
            attempt = parse_attempt(attempt_dir, task_dir)
            if attempt:
                attempts.append(attempt)
    attempts.sort(key=lambda item: item.get("mtime", 0), reverse=True)
    return attempts


def parse_log(task_dir: Path) -> dict[str, Any]:
    log_path = task_dir / "task.log"
    text = safe_read_text(log_path)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-80:])
    status = "waiting"
    if not log_path.exists():
        status = "waiting for task.log"
    elif re.search(
        r"completed successfully|Root scene pipeline finished|Static scene task .* completed",
        text,
    ):
        status = "complete"
    elif re.search(r"failed|Traceback|RuntimeError|Error:", text, flags=re.IGNORECASE):
        status = "error"
    elif lines:
        age = time.time() - log_path.stat().st_mtime
        status = "active" if age < 180 else "idle"

    active_stage = None
    active_attempt = None
    for line in reversed(lines):
        match = re.search(
            r"Running generator attempt (\d+)/(\d+) for stage ([a-z_]+)", line
        )
        if match:
            active_attempt = {
                "current": int(match.group(1)),
                "max": int(match.group(2)),
            }
            active_stage = match.group(3)
            break
        # The initializer uses distinct log messages from the remaining stages.
        match = re.search(r"Running initializer attempt (\d+)/(\d+)", line)
        if match:
            active_attempt = {
                "current": int(match.group(1)),
                "max": int(match.group(2)),
            }
            active_stage = "initializer"
            break
        if re.search(r"Starting initializer .*loop", line):
            active_stage = "initializer"
            break
        match = re.search(r"Starting stage ([a-z_]+)", line)
        if match:
            active_stage = match.group(1)
            break

    return {
        "path": path_payload(log_path, task_dir),
        "status": status,
        "tail": tail,
        "active_stage": active_stage,
        "active_attempt": active_attempt,
    }


# Preprocess sub-steps, LAST-completed-first (marker file/dir -> label). task.log
# only appears when the agent pipeline launches, so preprocess must be detected
# from the artifacts it writes progressively.
_PRE_STEPS = [
    # pseudo_gt/ is NOT a marker: it is created early by a parallel task
    ("physics/pose_changes.json", "physics settle done (finalizing)"),
    ("physics", "physics settle"),
    ("meshes", "SAM3D meshes"),
    ("placement.json", "placement + graph"),
    ("masks/masks.json", "segmentation"),
    ("moge/moge.json", "depth"),
]
_PRE_PROBES = [
    "input.png", "stage_timings.json", "moge", "moge/moge.json", "masks",
    "masks/masks.json", "placement.json", "meshes", "physics",
    "physics/settle_server.log", "pseudo_gt",
]  # fmt: skip


def preprocess_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """Live preprocess detection for a task with NO task.log yet: the runner
    creates task.log only when it launches the agent pipeline, so the log-based
    status read 'waiting' for the whole ~10-minute preprocess. Returns None when
    preprocess artifacts are absent (a genuinely queued run)."""
    if (task_dir / "task.log").exists():
        return None
    manifest = safe_read_json(task_dir / "preprocess_progress.json")
    if isinstance(manifest, dict) and manifest.get("version") == 1:
        manifest_state = str(manifest.get("state") or "running")
        completed = manifest.get("completed_steps") or []
        skipped = manifest.get("skipped_steps") or []
        total = manifest.get("total_steps") or []
        active_steps = manifest.get("active_steps") or []
        label = " + ".join(str(step) for step in active_steps) or (
            manifest_state if manifest_state in {"complete", "failed"} else "finalizing"
        )
        updated = float(manifest.get("updated_at") or 0.0)
        fresh = updated > 0 and time.time() - updated < 600
        accounted = len(set(completed) | set(skipped))
        return {
            "active": manifest_state == "running" and fresh,
            "label": label,
            "frac": min(1.0, accounted / len(total)) if total else 0.0,
            "source": "manifest",
            "manifest": manifest,
        }
    newest = 0.0
    for rel in _PRE_PROBES:
        try:
            newest = max(newest, (task_dir / rel).stat().st_mtime)
        except OSError:
            continue
    if newest == 0.0:
        return None
    label, done = "starting (depth + segmentation)", 0
    for i, (marker, step) in enumerate(_PRE_STEPS):
        if (task_dir / marker).exists():
            label, done = step, len(_PRE_STEPS) - i
            break
    # mesh generation / settle can be minutes-quiet between file writes
    active = (time.time() - newest) < 600
    return {
        "active": active,
        "label": label,
        "frac": done / (len(_PRE_STEPS) + 1),
    }


def estimate_progress(
    attempts: list[dict[str, Any]], log_info: dict[str, Any]
) -> dict[str, Any]:
    stage = log_info.get("active_stage")
    latest = attempts[0] if attempts else None
    if not stage and latest:
        stage = latest.get("stage")
    if not stage:
        return {"percent": 0, "label": "Waiting for first stage"}

    # Each stage gets an equal slot, with partial credit for active agents.
    slot = 1 / len(STAGE_ORDER)
    stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    base = stage_idx * slot
    within = 0.0
    if latest and latest.get("stage") == stage:
        within = slot * (0.48 if latest.get("role") == "generator" else 0.78)
        if latest.get("approved") is True:
            within = slot
    percent = max(1, min(99, round((base + within) * 100)))
    if log_info.get("status") == "complete":
        percent = 100
    return {"percent": percent, "label": f"{stage} ({percent}%)"}


def run_mtime(run_dir: Path, task_dirs: list[Path]) -> float:
    candidates = [run_dir / "args.json"]
    candidates.extend(task_dir / "task.log" for task_dir in task_dirs)
    candidates.extend(task_dir / "blender_file.blend" for task_dir in task_dirs)
    newest = 0.0
    for path in candidates:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def collect_task(
    task_dir: Path,
    run_dir: Path,
    output_dir: Path,
    target_dir: Optional[Path],
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    attempts = collect_attempts(task_dir)
    log_info = parse_log(task_dir)
    latest_attempt = attempts[0] if attempts else None
    latest_render = newest_file(iter_files(task_dir / "final" / "renders", IMAGE_SUFFIXES))
    if latest_render is None:
        latest_render = newest_file(iter_files(task_dir, IMAGE_SUFFIXES))
    target = None
    if target_dir is not None:
        target = newest_file(iter_files(target_dir / task_dir.name, IMAGE_SUFFIXES))
    if target is None:
        run_input = task_dir / "input.png"
        target = run_input if run_input.is_file() else None
    scene_graph = newest_file(
        p for p in task_dir.glob("stages/*/InitializerPlannerAgent/scene_graph.json")
    )
    progress = estimate_progress(attempts, log_info)
    status = log_info.get("status", "unknown")
    if latest_attempt and status == "waiting":
        status = "active"
    active_stage_override = None
    if status.startswith("waiting"):
        pre = preprocess_state(task_dir)
        if pre:
            status = "active" if pre["active"] else "idle"
            active_stage_override = "preprocess"
            # preprocess occupies the first ~15% of the bar, by completed sub-step
            pct = max(1, round(pre["frac"] * 15))
            progress = {"percent": pct, "label": f"preprocess: {pre['label']}"}

    return {
        "name": task_dir.name
        if run_dir == task_dir
        else f"{run_dir.name}/{task_dir.name}",
        "task_name": task_dir.name,
        "run_name": run_name or run_dir.name,
        "run_path": str(run_dir),
        "path": str(task_dir),
        "relative": str(task_dir.relative_to(output_dir))
        if is_relative_to(task_dir, output_dir)
        else str(task_dir),
        "status": status,
        "active_stage": active_stage_override
        or log_info.get("active_stage")
        or (latest_attempt or {}).get("stage"),
        "active_attempt": log_info.get("active_attempt"),
        "progress": progress,
        "latest_attempt": latest_attempt,
        "attempts": attempts,
        "latest_render": path_payload(latest_render, task_dir),
        "target_image": path_payload(target, task_dir) if target else None,
        "scene_graph": path_payload(scene_graph, task_dir),
        "log": log_info,
    }


def aggregate_run_status(tasks: list[dict[str, Any]], has_args: bool) -> str:
    statuses = {task.get("status", "unknown") for task in tasks}
    if "active" in statuses:
        return "active"
    if "error" in statuses:
        return "error"
    if tasks and statuses == {"complete"}:
        return "complete"
    if any(status.startswith("waiting") for status in statuses) or (
        has_args and not tasks
    ):
        return "waiting"
    if "idle" in statuses:
        return "idle"
    return "unknown"


def collect_state(
    config: DashboardConfig, tracked_runs: Optional[set[str]] = None
) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    runs = discover_runs(output_dir)
    runs.sort(key=run_name_sort_key)
    tasks = []
    run_payloads = []
    for run in runs:
        run_dir = run["run_dir"].resolve()
        task_dirs = [task_dir.resolve() for task_dir in run["task_dirs"]]
        run_name = run.get("name") or run_dir.name
        parent_name = run.get("parent_name") or run_dir.name
        scene_name = run.get("scene_name") or ""
        args_path = run_dir / "args.json"
        tracked = tracked_runs is None or run_name in tracked_runs
        run_tasks = []
        if tracked:
            run_tasks = [
                collect_task(
                    task_dir,
                    run_dir,
                    output_dir,
                    config.target_dir.resolve() if config.target_dir is not None else None,
                    run_name,
                )
                for task_dir in task_dirs
            ]
        run_payloads.append(
            {
                "name": run_name,
                "parent_name": parent_name,
                "scene_name": scene_name,
                "path": str(run_dir),
                "relative": run_name,
                "mtime": run_mtime(run_dir, task_dirs),
                "mtime_label": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(run_mtime(run_dir, task_dirs))
                )
                if run_mtime(run_dir, task_dirs)
                else "",
                "status": aggregate_run_status(run_tasks, args_path.exists())
                if tracked
                else "off",
                "tracked": tracked,
                "args": safe_read_json(args_path)
                if tracked and args_path.exists()
                else None,
                "task_count": len(task_dirs),
            }
        )
        tasks.extend(run_tasks)
    tasks.sort(
        key=lambda task: (
            (task.get("latest_attempt") or {}).get("mtime")
            or (task.get("log", {}).get("path") or {}).get("mtime")
            or 0
        ),
        reverse=True,
    )
    args_path = output_dir / "args.json"
    return {
        "output_dir": str(output_dir),
        "generated_at": time.time(),
        "generated_label": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": safe_read_json(args_path) if args_path.exists() else None,
        "run_count": len(runs),
        "runs": run_payloads,
        "tasks": tasks,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(render_index(self.config))
            return
        if parsed.path == "/memory":
            self.send_html(render_memory_page())
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            tracked_values = query.get("tracked")
            tracked_runs = None
            if tracked_values is not None:
                tracked_runs = {
                    name
                    for value in tracked_values
                    for name in value.split("\n")
                    if name
                }
            state = collect_state(self.config, tracked_runs)
            self.send_json(state)
            return
        if parsed.path == "/api/memory":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            try:
                self.send_json(collect_memory(raw_path, self.config))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Memory file not found")
            return
        if parsed.path == "/file":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            self.send_file(raw_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, raw_path: str) -> None:
        path = resolve_dashboard_file(raw_path, self.config)
        if not path:
            self.send_error(HTTPStatus.FORBIDDEN, "Path is outside dashboard roots")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read file")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def render_index(config: DashboardConfig) -> str:
    output_dir = html.escape(str(config.output_dir.resolve()))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SceneRig Live Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #64717f;
      --line: #d7dde4;
      --panel: #ffffff;
      --bg: #f6f8fb;
      --accent: #276ef1;
      --ok: #18865b;
      --warn: #a26105;
      --bad: #bf2c2c;
      --track: #e8edf3;
      --code: #101820;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; max-width: 100%; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 700; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; word-break: break-all; }}
    .statusbar {{ display: flex; gap: 10px; align-items: center; color: var(--muted); font-size: 13px; }}
    .dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--ok); display: inline-block; }}
    main {{
      width: 100%;
      max-width: 100vw;
      min-height: calc(100vh - 78px);
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
      align-items: start;
      overflow-x: hidden;
    }}
    .sidebar {{
      position: sticky;
      top: 78px;
      max-height: calc(100vh - 78px);
      border-right: 1px solid var(--line);
      background: #fff;
      overflow: auto;
      padding: 14px;
    }}
    .side-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: #344253;
      font-size: 13px;
      font-weight: 700;
      margin: 0 0 10px;
    }}
    .run-list {{ display: grid; gap: 8px; }}
    .track-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .track-actions button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--accent);
      min-height: 32px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 650;
    }}
    .track-actions button:hover {{ border-color: #a9c7ff; background: #f5f9ff; }}
    .run-group {{
      display: grid;
      gap: 6px;
      padding: 2px 0 6px;
      border-bottom: 1px solid #eef1f5;
    }}
    .run-group:last-child {{ border-bottom: 0; }}
    .run-group-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: #526174;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 2px 2px;
      overflow-wrap: anywhere;
    }}
    .run-button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      color: var(--ink);
      padding: 10px;
      text-align: left;
      cursor: pointer;
      display: grid;
      gap: 7px;
      font: inherit;
    }}
    .run-button.child {{ margin-left: 10px; width: calc(100% - 10px); }}
    .run-button:hover {{ border-color: #a9c7ff; background: #f5f9ff; }}
    .run-button.selected {{ border-color: var(--accent); background: #eff5ff; box-shadow: inset 3px 0 0 var(--accent); }}
    .run-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .run-title {{ display: flex; align-items: center; min-width: 0; gap: 8px; }}
    .run-title input {{ flex: 0 0 auto; width: 16px; height: 16px; accent-color: var(--accent); }}
    .run-name {{ font-size: 14px; font-weight: 700; overflow-wrap: anywhere; }}
    .run-meta {{ color: var(--muted); font-size: 12px; }}
    .content {{
      min-width: 0;
      overflow: visible;
      padding: 18px 20px 30px;
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .run-summary {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      display: grid;
      gap: 6px;
    }}
    .run-summary h2 {{ margin: 0; font-size: 18px; color: var(--ink); }}
    .job {{
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .job-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(180px, 1.3fr) auto;
      gap: 16px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    .job-title {{ font-size: 18px; font-weight: 700; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      background: #fff;
      text-transform: capitalize;
    }}
    .badge.active {{ color: #084a9b; border-color: #a9c7ff; background: #eff5ff; }}
    .badge.complete {{ color: var(--ok); border-color: #a8dec8; background: #effaf5; }}
    .badge.error {{ color: var(--bad); border-color: #f0b4b4; background: #fff1f1; }}
    .badge.waiting {{ color: var(--warn); border-color: #f3d49f; background: #fff8eb; }}
    .badge.idle {{ color: var(--muted); border-color: var(--line); background: #f8fafc; }}
    .progress-wrap {{ display: grid; gap: 7px; }}
    .progress-label {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 13px; }}
    .progress {{ height: 12px; border-radius: 999px; background: var(--track); overflow: hidden; }}
    .progress > div {{ height: 100%; width: 0%; background: linear-gradient(90deg, #276ef1, #12a594); transition: width .25s ease; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      padding: 16px;
      min-width: 0;
    }}
    .job-body {{ display: grid; gap: 16px; padding: 16px; min-width: 0; }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      min-width: 0;
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
      gap: 16px;
      min-width: 0;
    }}
    .panel {{ min-width: 0; overflow-wrap: anywhere; }}
    .status-panel {{ grid-column: 1 / -1; }}
    section {{ min-width: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 14px; color: #344253; }}
    .media-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      color: #344253;
      font-size: 13px;
      font-weight: 700;
    }}
    .media {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fbfd;
      overflow: hidden;
      min-height: 320px;
      display: grid;
      place-items: center;
    }}
    .media img {{ display: block; width: 100%; height: auto; max-height: 560px; object-fit: contain; }}
    .timeline {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px 10px;
      margin-top: 10px;
    }}
    .timeline input[type="range"] {{
      width: 100%;
      min-width: 0;
      accent-color: var(--accent);
      grid-column: 1;
      grid-row: 1;
    }}
    .timeline button {{
      grid-row: 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--accent);
      min-height: 32px;
      padding: 4px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
    }}
    .timeline button[data-action="freeze"] {{ grid-column: 2; }}
    .timeline button[data-action="live"] {{ grid-column: 3; }}
    .timeline button:hover {{ border-color: #a9c7ff; background: #f5f9ff; }}
    .timeline-readout {{
      grid-column: 1 / -1;
      grid-row: 2;
      color: #344253;
      font-size: 12px;
      font-weight: 650;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .timeline-label {{ grid-column: 1 / -1; grid-row: 3; color: var(--muted); font-size: 12px; }}
    .empty {{ color: var(--muted); font-size: 13px; padding: 18px; text-align: center; }}
    .kv {{ display: grid; grid-template-columns: 112px 1fr; gap: 7px 12px; font-size: 13px; margin-bottom: 12px; }}
    .k {{ color: var(--muted); }}
    .v {{ min-width: 0; word-break: break-word; }}
    pre {{
      max-width: 100%;
      margin: 0;
      padding: 12px;
      min-height: 130px;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--code);
      color: #eef4fb;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .attempts {{
      display: grid;
      gap: 8px;
      max-height: 520px;
      overflow: auto;
      padding-right: 2px;
    }}
    .attempt {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
      display: grid;
      gap: 6px;
      font-size: 13px;
    }}
    .attempt-top {{ display: flex; gap: 8px; align-items: center; justify-content: space-between; }}
    .attempt-top > * {{ min-width: 0; }}
    .attempt-name {{ font-weight: 650; }}
    .links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .links a {{ color: var(--accent); text-decoration: none; font-size: 12px; }}
    .two-up {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    @media (max-width: 1100px) {{
      main {{ height: auto; grid-template-columns: 1fr; overflow: visible; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--line); max-height: 280px; }}
      .content {{ overflow: visible; padding: 16px; }}
      .job-head, .grid, .compare-grid, .details-grid {{ grid-template-columns: 1fr; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .two-up {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SceneRig Live Dashboard</h1>
      <div class="sub">{output_dir}</div>
    </div>
    <div class="statusbar"><span class="dot"></span><span id="refresh">Connecting...</span></div>
  </header>
  <main>
    <aside class="sidebar">
      <div class="side-title"><span>Runs</span><span id="run-count">0</span></div>
      <div class="track-actions">
        <button type="button" id="track-all">Enable all</button>
        <button type="button" id="track-none">Disable all</button>
      </div>
      <div id="run-list" class="run-list"></div>
    </aside>
    <div id="app" class="content"></div>
  </main>
  <script>
    const app = document.getElementById('app');
    const runList = document.getElementById('run-list');
    const runCount = document.getElementById('run-count');
    const trackAll = document.getElementById('track-all');
    const trackNone = document.getElementById('track-none');
    const refresh = document.getElementById('refresh');
    let selectedRun = localStorage.getItem('viga:selectedRun') || '';
    let trackedRuns = null;
    const storedTrackedRuns = localStorage.getItem('viga:trackedRuns');
    if (storedTrackedRuns !== null) {{
      try {{
        trackedRuns = new Set(JSON.parse(storedTrackedRuns));
      }} catch (_) {{
        trackedRuns = null;
      }}
    }}
    let lastState = null;
    let lastRenderSignature = '';
    let scrubbing = false;
    const renderPins = new Map();

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
    }}
    function displayText(value) {{
      return String(value ?? '').replace(/\\\\n/g, '\\n').replace(/\\\\t/g, '  ');
    }}
    function escText(value) {{
      return esc(displayText(value));
    }}
    function percent(value) {{ return Math.max(0, Math.min(100, Number(value || 0))); }}
    function link(file, label) {{
      if (!file) return '';
      const href = String(file.name || '').endsWith('_memory.json')
        ? `/memory?path=${{encodeURIComponent(file.path)}}`
        : file.url;
      return `<a href="${{esc(href)}}" target="_blank" rel="noreferrer">${{esc(label || file.name)}}</a>`;
    }}
    function formatArgs(args) {{
      if (!args) return 'No active tool call yet.';
      return formatValue(args);
    }}
    function formatValue(value, depth = 0) {{
      const pad = '  '.repeat(depth);
      const childPad = '  '.repeat(depth + 1);
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return displayText(value);
      if (typeof value !== 'object') return String(value);
      if (Array.isArray(value)) {{
        return value.map((item, idx) => `${{pad}}[${{idx}}] ${{formatValue(item, depth + 1)}}`).join('\\n');
      }}
      return Object.entries(value).map(([key, val]) => {{
        if (val && typeof val === 'object') {{
          return `${{pad}}${{key}}:\\n${{formatValue(val, depth + 1)}}`;
        }}
        const formatted = formatValue(val, depth + 1);
        return formatted.includes('\\n') ? `${{pad}}${{key}}:\\n${{childPad}}${{formatted.replace(/\\n/g, `\\n${{childPad}}`)}}` : `${{pad}}${{key}}: ${{formatted}}`;
      }}).join('\\n');
    }}
    function fileUrl(file) {{
      if (!file) return '';
      return `${{file.url}}&v=${{encodeURIComponent(file.mtime || 0)}}`;
    }}
    function image(file, attrs = '') {{
      if (!file) return '<div class="empty">No render image has been written yet.</div>';
      return `<img ${{attrs}} src="${{esc(fileUrl(file))}}" alt="${{esc(file.name)}}">`;
    }}
    function targetImage(file) {{
      if (!file) return '<div class="empty">No target image found for this task.</div>';
      return image(file);
    }}
    function statusBadge(status) {{
      return `<span class="badge ${{esc(status || '')}}">${{esc(status || 'unknown')}}</span>`;
    }}
    function isTracked(runName) {{
      return trackedRuns === null || trackedRuns.has(runName);
    }}
    function saveTrackedRuns() {{
      if (trackedRuns === null) {{
        localStorage.removeItem('viga:trackedRuns');
      }} else {{
        localStorage.setItem('viga:trackedRuns', JSON.stringify([...trackedRuns].sort()));
      }}
    }}
    function trackedStateUrl() {{
      if (trackedRuns === null) return '/api/state';
      const params = new URLSearchParams();
      params.set('tracked', [...trackedRuns].join('\\n'));
      return `/api/state?${{params.toString()}}`;
    }}
    function runButton(run, grouped = false) {{
      const label = grouped && run.scene_name ? run.scene_name : run.name;
      const meta = grouped && run.scene_name ? `${{run.name}} · ` : '';
      const checked = isTracked(run.name) ? 'checked' : '';
      return `
        <button class="run-button ${{grouped && run.scene_name ? 'child' : ''}} ${{run.name === selectedRun ? 'selected' : ''}}" data-run="${{esc(run.name)}}">
          <span class="run-row"><span class="run-title"><input type="checkbox" data-action="track-run" data-run="${{esc(run.name)}}" ${{checked}}><span class="run-name">${{esc(label)}}</span></span>${{statusBadge(run.status)}}</span>
          <span class="run-meta">${{esc(meta)}}${{esc(run.task_count)}} job(s) · ${{esc(run.mtime_label || 'no activity yet')}}</span>
        </button>
      `;
    }}
    function naturalCompare(a, b) {{
      return String(a || '').localeCompare(String(b || ''), undefined, {{ numeric: true, sensitivity: 'base' }});
    }}
    function runCompare(a, b) {{
      const parentA = a.parent_name || a.name || '';
      const parentB = b.parent_name || b.name || '';
      const tsA = parentA.match(/^(\\d{{8}})_(\\d{{6}})$/);
      const tsB = parentB.match(/^(\\d{{8}})_(\\d{{6}})$/);
      if (tsA && tsB && parentA !== parentB) return naturalCompare(parentB, parentA);
      if (tsA && !tsB) return -1;
      if (!tsA && tsB) return 1;
      const parentCmp = naturalCompare(parentA, parentB);
      if (parentCmp) return parentCmp;
      return naturalCompare(a.scene_name || '', b.scene_name || '');
    }}
    function renderRuns(runs) {{
      runs = [...runs].sort(runCompare);
      runCount.textContent = String(runs.length);
      if (!runs.length) {{
        runList.innerHTML = '<div class="empty">No runs found.</div>';
      }} else {{
        const groups = [];
        const byParent = new Map();
        runs.forEach(run => {{
          const parent = run.parent_name || '';
          if (!parent || !run.scene_name) {{
            groups.push({{ parent: '', runs: [run] }});
            return;
          }}
          if (!byParent.has(parent)) {{
            const group = {{ parent, runs: [] }};
            byParent.set(parent, group);
            groups.push(group);
          }}
          byParent.get(parent).runs.push(run);
        }});
        runList.innerHTML = groups.map(group => {{
          if (!group.parent) return group.runs.map(run => runButton(run)).join('');
          return `
            <section class="run-group">
              <div class="run-group-title"><span>${{esc(group.parent)}}</span><span>${{esc(group.runs.length)}} scene(s)</span></div>
              ${{group.runs.map(run => runButton(run, true)).join('')}}
            </section>
          `;
        }}).join('');
      }}
      runList.querySelectorAll('.run-button').forEach(button => {{
        button.addEventListener('click', () => {{
          selectedRun = button.dataset.run || '';
          localStorage.setItem('viga:selectedRun', selectedRun);
          renderState(lastState, true);
        }});
      }});
      runList.querySelectorAll('input[data-action="track-run"]').forEach(input => {{
        input.addEventListener('click', event => event.stopPropagation());
        input.addEventListener('change', () => {{
          const runName = input.dataset.run || '';
          if (trackedRuns === null) {{
            trackedRuns = new Set((lastState?.runs || []).map(run => run.name));
          }}
          if (input.checked) trackedRuns.add(runName);
          else trackedRuns.delete(runName);
          saveTrackedRuns();
          lastRenderSignature = '';
          load(true);
        }});
      }});
    }}
    function findJob(jobKey) {{
      return (lastState?.tasks || []).find(task => task.path === jobKey) || null;
    }}
    function buildTimeline(job) {{
      return (job?.attempts || [])
        .flatMap(a => {{
          const renders = (a.renders && a.renders.length) ? a.renders : (a.render ? [a.render] : []);
          return renders.map((render, renderIndex) => ({{
            ...a,
            render,
            render_index: renderIndex + 1,
            frame_label: render.relative?.split('/renders/')[1] || render.name || `render ${{renderIndex + 1}}`,
          }}));
        }})
        .sort((a, b) => Number(a.render?.mtime || a.mtime || 0) - Number(b.render?.mtime || b.mtime || 0));
    }}
    function timelineSelection(job) {{
      const timeline = buildTimeline(job);
      const liveIndex = Math.max(0, timeline.length - 1);
      const pinned = renderPins.has(job.path) ? Number(renderPins.get(job.path)) : null;
      const timelineIndex = pinned === null ? liveIndex : Math.max(0, Math.min(liveIndex, pinned));
      const selectedFrame = timeline[timelineIndex] || null;
      const previewRender = selectedFrame?.render || job.latest_render;
      const previewMode = pinned === null ? 'Live' : `Pinned frame ${{timelineIndex + 1}}`;
      const renderStep = selectedFrame?.frame_label || selectedFrame?.render?.relative?.split('/renders/')[1] || selectedFrame?.render?.name || '';
      const readout = selectedFrame
        ? `${{selectedFrame.stage}} / ${{selectedFrame.role}} · attempt ${{selectedFrame.attempt}} · render ${{selectedFrame.render_index || timelineIndex + 1}} · ${{renderStep || ''}}`
        : 'No frame';
      const timelineLabel = selectedFrame
        ? `${{previewMode}} · ${{selectedFrame.stage}}/${{selectedFrame.role}} attempt ${{selectedFrame.attempt}} · ${{selectedFrame.render?.mtime_label || selectedFrame.mtime_label || ''}}`
        : 'No render timeline yet.';
      const detailsFrame = pinned === null ? null : selectedFrame;
      return {{ timeline, timelineIndex, selectedFrame, detailsFrame, previewRender, pinned, timelineLabel, readout }};
    }}
    function attemptDetails(job, frame = null) {{
      const attempt = frame || job.latest_attempt || {{}};
      const info = attempt.latest || {{}};
      const call = info.tool_call || null;
      return {{
        stage: attempt.stage || job.active_stage || 'waiting',
        run: job.run_name || '',
        agent: attempt.agent || 'none yet',
        attempt: attempt.attempt || 'none',
        tool: call?.name || info.last_tool_name || 'none',
        updated: attempt.mtime_label || job.log?.path?.mtime_label || '',
        vlmInput: info.current_vlm_input || 'Waiting for saved model context.',
        vlmOutput: call ? formatArgs(call.arguments) : (info.assistant_text || 'No VLM output recorded yet.'),
        status: info.last_tool_text || job.log?.tail || 'No status yet.',
      }};
    }}
    function updateAttemptDetails(card, job, frame = null) {{
      const details = attemptDetails(job, frame);
      const setText = (selector, value) => {{
        const el = card?.querySelector(selector);
        if (el) el.textContent = displayText(value);
      }};
      setText('[data-field="stage"]', details.stage);
      setText('[data-field="run"]', details.run);
      setText('[data-field="agent"]', details.agent);
      setText('[data-field="attempt"]', details.attempt);
      setText('[data-field="tool"]', details.tool);
      setText('[data-field="updated"]', details.updated);
      setText('[data-field="vlm-input"]', details.vlmInput);
      setText('[data-field="vlm-output"]', details.vlmOutput);
      setText('[data-field="status"]', details.status);
    }}
    function applyTimelineSelection(card, job, selection) {{
      const input = card?.querySelector('[data-action="timeline"]');
      const img = card?.querySelector('[data-role="render-image"]');
      const label = card?.querySelector('.timeline-label');
      const readout = card?.querySelector('.timeline-readout');
      const liveButton = card?.querySelector('[data-action="live"]');
      const freezeButton = card?.querySelector('[data-action="freeze"]');
      if (input) input.value = selection.timelineIndex;
      if (img && selection.previewRender) {{
        const nextUrl = fileUrl(selection.previewRender);
        if (img.getAttribute('src') !== nextUrl) {{
          img.setAttribute('src', nextUrl);
          img.setAttribute('alt', selection.previewRender.name || 'render');
        }}
      }}
      if (label) label.textContent = selection.timelineLabel;
      if (readout) readout.textContent = selection.readout;
      updateAttemptDetails(card, job, selection.detailsFrame);
      if (liveButton) liveButton.disabled = selection.pinned === null;
      if (freezeButton) freezeButton.disabled = selection.timeline.length === 0;
    }}
    function updateTimelinePreview(input) {{
      const jobKey = input.dataset.job || '';
      const job = findJob(jobKey);
      if (!job) return;
      renderPins.set(jobKey, Number(input.value || 0));
      const selection = timelineSelection(job);
      const card = input.closest('.job');
      applyTimelineSelection(card, job, selection);
    }}
    function renderJob(job) {{
      const p = percent(job.progress?.percent);
      const jobKey = job.path;
      const {{ timeline, timelineIndex, previewRender, pinned, timelineLabel, readout, detailsFrame }} = timelineSelection(job);
      const details = attemptDetails(job, detailsFrame);
      const attemptRows = (job.attempts || []).map(a => `
        <div class="attempt">
          <div class="attempt-top">
            <span class="attempt-name">${{esc(a.stage)}} / ${{esc(a.role)}} attempt ${{esc(a.attempt)}}</span>
            <span class="badge ${{a.approved === true ? 'complete' : a.approved === false ? 'error' : ''}}">${{a.approved === true ? 'approved' : a.approved === false ? 'rejected' : esc(a.mtime_label || '')}}</span>
          </div>
          <div class="k">${{esc(a.agent)}} · stage index ${{esc(a.stage_index)}}</div>
          <div class="links">${{link(a.memory, 'memory')}} ${{link(a.render, 'render')}} ${{link(a.script, 'script')}}</div>
        </div>
      `).join('');
      return `
        <article class="job" data-job="${{esc(jobKey)}}">
          <div class="job-head">
            <div>
              <div class="job-title">${{esc(job.name)}}</div>
              <div class="sub">${{esc(job.path)}}</div>
            </div>
            <div class="progress-wrap">
              <div class="progress-label"><span>${{esc(job.progress?.label || 'waiting')}}</span><span>${{p}}%</span></div>
              <div class="progress"><div style="width: ${{p}}%"></div></div>
            </div>
            <span class="badge ${{esc(job.status)}}">${{esc(job.status)}}</span>
          </div>
          <div class="job-body">
            <section class="compare-grid">
              <div class="panel">
                <div class="media-title"><span>Target Image</span>${{link(job.target_image, 'open')}}</div>
                <div class="media">${{targetImage(job.target_image)}}</div>
              </div>
              <div class="panel">
                <div class="media-title"><span>Selected Render</span>${{previewRender ? link(previewRender, 'open') : ''}}</div>
                <div class="media">${{image(previewRender, 'data-role="render-image"')}}</div>
                <div class="timeline">
                  <button type="button" data-action="freeze" data-job="${{esc(jobKey)}}" ${{timeline.length ? '' : 'disabled'}}>Freeze</button>
                  <button type="button" data-action="live" data-job="${{esc(jobKey)}}" ${{pinned === null ? 'disabled' : ''}}>Live</button>
                  <input type="range" min="0" max="${{Math.max(0, timeline.length - 1)}}" value="${{timelineIndex}}" data-action="timeline" data-job="${{esc(jobKey)}}" ${{timeline.length <= 1 ? 'disabled' : ''}}>
                  <span class="timeline-readout">${{esc(readout)}}</span>
                  <div class="timeline-label">${{esc(timelineLabel)}}</div>
                </div>
              </div>
            </section>
            <div class="links">${{link(job.scene_graph, 'scene graph')}} ${{link(job.log?.path, 'task log')}}</div>
            <section class="details-grid">
              <div class="panel">
              <h2>Live State</h2>
              <div class="kv">
                <div class="k">Stage</div><div class="v" data-field="stage">${{esc(details.stage)}}</div>
                <div class="k">Run</div><div class="v" data-field="run">${{esc(details.run)}}</div>
                <div class="k">Agent</div><div class="v" data-field="agent">${{esc(details.agent)}}</div>
                <div class="k">Attempt</div><div class="v" data-field="attempt">${{esc(details.attempt)}}</div>
                <div class="k">Tool</div><div class="v" data-field="tool">${{esc(details.tool)}}</div>
                <div class="k">Updated</div><div class="v" data-field="updated">${{esc(details.updated)}}</div>
              </div>
              </div>
              <div class="panel">
              <h2>VLM Input Now</h2>
              <pre data-field="vlm-input">${{escText(details.vlmInput)}}</pre>
              <h2 style="margin-top: 12px;">VLM Output</h2>
              <pre data-field="vlm-output">${{escText(details.vlmOutput)}}</pre>
              </div>
              <div class="panel status-panel">
              <h2>Current Status</h2>
              <pre data-field="status">${{escText(details.status)}}</pre>
              <h2 style="margin-top: 12px;">Attempts</h2>
              <div class="attempts">${{attemptRows || '<div class="empty">No stage attempts have been created yet.</div>'}}</div>
              </div>
            </section>
          </div>
        </article>
      `;
    }}
    function stateSignature(state) {{
      const runs = (state?.runs || []).map(run => [run.name, run.status, run.task_count, run.mtime]);
      const tasks = (state?.tasks || [])
        .filter(task => !selectedRun || task.run_name === selectedRun)
        .map(task => [
          task.path,
          task.status,
          task.active_stage,
          task.progress?.percent,
          task.latest_render?.mtime,
          task.log?.path?.mtime,
          task.log?.tail,
          task.latest_attempt?.path,
          task.latest_attempt?.mtime,
          task.latest_attempt?.latest?.assistant_text,
          task.latest_attempt?.latest?.last_tool_text,
          task.latest_attempt?.latest?.last_tool_name,
          task.latest_attempt?.latest?.current_vlm_input,
          JSON.stringify(task.latest_attempt?.latest?.tool_call || null),
          (task.attempts || []).map(a => [
            a.path,
            a.mtime,
            a.approved,
            a.latest?.assistant_text,
            a.latest?.last_tool_text,
            a.latest?.last_tool_name,
            a.latest?.current_vlm_input,
            JSON.stringify(a.latest?.tool_call || null),
            (a.renders || []).map(r => r.path + ':' + r.mtime)
          ])
        ]);
      return JSON.stringify([selectedRun, runs, tasks]);
    }}
    function capturePaneScrolls() {{
      const scrolls = {{}};
      app.querySelectorAll('.job').forEach(card => {{
        const jobKey = card.dataset.job || '';
        card.querySelectorAll('pre[data-field]').forEach(pre => {{
          scrolls[`${{jobKey}}::${{pre.dataset.field}}`] = {{
            top: pre.scrollTop,
            left: pre.scrollLeft,
            atBottom: Math.abs((pre.scrollHeight - pre.clientHeight) - pre.scrollTop) < 4,
          }};
        }});
      }});
      return scrolls;
    }}
    function restorePaneScrolls(scrolls) {{
      app.querySelectorAll('.job').forEach(card => {{
        const jobKey = card.dataset.job || '';
        card.querySelectorAll('pre[data-field]').forEach(pre => {{
          const saved = scrolls[`${{jobKey}}::${{pre.dataset.field}}`];
          if (!saved) return;
          pre.scrollLeft = saved.left;
          pre.scrollTop = saved.atBottom ? pre.scrollHeight : saved.top;
        }});
      }});
    }}
    function renderState(state, force = false) {{
      if (!state) return;
        const runs = state.runs || [];
        if ((!selectedRun || !runs.some(run => run.name === selectedRun)) && runs.length) {{
          selectedRun = runs[0].name;
          localStorage.setItem('viga:selectedRun', selectedRun);
          force = true;
        }}
        const signature = stateSignature(state);
        if (!force && signature === lastRenderSignature) {{
          refresh.textContent = `Updated ${{state.generated_label}} · ${{state.run_count || 0}} run(s) · ${{state.tasks.length}} job(s) · no file changes`;
          return;
        }}
        const paneScrolls = capturePaneScrolls();
        lastRenderSignature = signature;
        renderRuns(runs);
        const selected = runs.find(run => run.name === selectedRun) || null;
        const visibleTasks = (state.tasks || []).filter(task => !selectedRun || task.run_name === selectedRun);
        const trackedLabel = trackedRuns === null ? 'all tracked' : `${{trackedRuns.size}} tracked`;
        refresh.textContent = `Updated ${{state.generated_label}} · ${{state.run_count || 0}} run(s) · ${{trackedLabel}} · ${{state.tasks.length}} loaded job(s)`;
        app.innerHTML = selected
          ? `<section class="run-summary"><h2>${{esc(selected.name)}}</h2><div class="sub">${{esc(selected.path)}}</div><div class="links">${{statusBadge(selected.status)}} <span class="badge">${{esc(selected.task_count)}} job(s)</span></div></section>` + (selected.tracked === false ? '<div class="job"><div class="empty">Tracking is disabled for this run.</div></div>' : (visibleTasks.length ? visibleTasks.map(renderJob).join('') : '<div class="job"><div class="empty">This run has no task folders yet.</div></div>'))
          : '<div class="job"><div class="empty">No run folders found under the selected output directory.</div></div>';
        restorePaneScrolls(paneScrolls);
        app.querySelectorAll('[data-action="timeline"]').forEach(input => {{
          input.addEventListener('pointerdown', () => {{
            scrubbing = true;
          }});
          input.addEventListener('pointerup', () => {{
            scrubbing = false;
          }});
          input.addEventListener('pointercancel', () => {{
            scrubbing = false;
          }});
          input.addEventListener('input', () => {{
            updateTimelinePreview(input);
          }});
          input.addEventListener('change', () => {{
            scrubbing = false;
            updateTimelinePreview(input);
          }});
        }});
        app.querySelectorAll('[data-action="live"]').forEach(button => {{
          button.addEventListener('click', () => {{
            const jobKey = button.dataset.job || '';
            renderPins.delete(jobKey);
            const job = findJob(jobKey);
            const card = button.closest('.job');
            if (job && card) {{
              const selection = timelineSelection(job);
              applyTimelineSelection(card, job, selection);
            }} else {{
              renderState(lastState, true);
            }}
          }});
        }});
        app.querySelectorAll('[data-action="freeze"]').forEach(button => {{
          button.addEventListener('click', () => {{
            const jobKey = button.dataset.job || '';
            const job = findJob(jobKey);
            const card = button.closest('.job');
            if (job && card) {{
              const currentSelection = timelineSelection(job);
              renderPins.set(jobKey, currentSelection.timelineIndex);
              const selection = timelineSelection(job);
              applyTimelineSelection(card, job, selection);
            }} else {{
              renderState(lastState, true);
            }}
          }});
        }});
    }}
    if (trackAll) {{
      trackAll.addEventListener('click', () => {{
        trackedRuns = null;
        saveTrackedRuns();
        lastRenderSignature = '';
        load(true);
      }});
    }}
    if (trackNone) {{
      trackNone.addEventListener('click', () => {{
        trackedRuns = new Set();
        saveTrackedRuns();
        lastRenderSignature = '';
        load(true);
      }});
    }}
    async function load(force = false) {{
      try {{
        const response = await fetch(trackedStateUrl(), {{ cache: 'no-store' }});
        lastState = await response.json();
        if (!scrubbing) renderState(lastState, force);
      }} catch (error) {{
        refresh.textContent = `Dashboard error: ${{error}}`;
      }}
    }}
    load();
    setInterval(load, 2500);
  </script>
</body>
</html>"""


def render_memory_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SceneRig Memory Chat</title>
  <style>
    :root {
      --bg: #eef2f6;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667282;
      --line: #d9e1ea;
      --user: #e7f0ff;
      --assistant: #ffffff;
      --system: #f3f5f7;
      --accent: #245fd6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 62px;
      padding: 12px 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.94);
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
    }
    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }
    a.button, button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--accent);
      min-height: 34px;
      padding: 7px 11px;
      text-decoration: none;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }
    main {
      width: min(1120px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }
    .empty {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      padding: 24px;
      text-align: center;
    }
    .message {
      display: grid;
      grid-template-columns: 104px minmax(0, 1fr);
      gap: 12px;
      margin: 12px 0;
      align-items: start;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      text-align: right;
      padding-top: 8px;
      overflow-wrap: anywhere;
    }
    .role {
      color: var(--ink);
      font-weight: 750;
      text-transform: capitalize;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--assistant);
      padding: 12px;
      min-width: 0;
      overflow: hidden;
    }
    .message.user .bubble { background: var(--user); }
    .message.system .bubble, .message.tool .bubble { background: var(--system); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .part + .part { margin-top: 10px; }
    .image-frame {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      overflow: hidden;
      display: inline-grid;
      max-width: 100%;
    }
    .image-frame img {
      display: block;
      max-width: min(720px, 100%);
      max-height: 520px;
      width: auto;
      height: auto;
      object-fit: contain;
    }
    @media (max-width: 720px) {
      header { align-items: flex-start; flex-direction: column; }
      .actions { width: 100%; }
      .message { grid-template-columns: 1fr; }
      .meta { text-align: left; padding-top: 0; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="title">Memory Chat</h1>
      <div class="sub" id="subtitle">Loading memory...</div>
    </div>
    <div class="actions">
      <a class="button" id="raw-link" href="#" target="_blank" rel="noreferrer">Raw JSON</a>
      <a class="button" href="/">Dashboard</a>
    </div>
  </header>
  <main id="chat"><div class="empty">Loading memory...</div></main>
  <script>
    const params = new URLSearchParams(location.search);
    const memoryPath = params.get('path') || '';
    const chat = document.getElementById('chat');
    const title = document.getElementById('title');
    const subtitle = document.getElementById('subtitle');
    const rawLink = document.getElementById('raw-link');

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }
    function displayText(value) {
      return String(value ?? '').replace(/\\\\n/g, '\\n').replace(/\\\\t/g, '  ');
    }
    function partHtml(part) {
      if (part.type === 'image') {
        return `<div class="part image-frame"><img src="${esc(part.url)}" alt="memory image" loading="lazy"></div>`;
      }
      return `<pre class="part">${esc(displayText(part.text || ''))}</pre>`;
    }
    function messageHtml(message) {
      const role = String(message.role || 'unknown').toLowerCase();
      const cls = ['user', 'assistant', 'system', 'tool'].includes(role) ? role : 'assistant';
      const name = message.name ? `<div>${esc(message.name)}</div>` : '';
      return `
        <article class="message ${cls}">
          <aside class="meta">
            <div class="role">${esc(role)}</div>
            <div>#${esc(message.index)}</div>
            ${name}
          </aside>
          <section class="bubble">${(message.parts || []).map(partHtml).join('')}</section>
        </article>
      `;
    }
    async function loadMemory() {
      if (!memoryPath) {
        chat.innerHTML = '<div class="empty">No memory path was provided.</div>';
        subtitle.textContent = '';
        rawLink.style.display = 'none';
        return;
      }
      rawLink.href = `/file?path=${encodeURIComponent(memoryPath)}`;
      try {
        const response = await fetch(`/api/memory?path=${encodeURIComponent(memoryPath)}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const payload = await response.json();
        title.textContent = payload.name || 'Memory Chat';
        subtitle.textContent = `${payload.relative || payload.path} · ${payload.message_count || 0} messages`;
        chat.innerHTML = payload.messages?.length
          ? payload.messages.map(messageHtml).join('')
          : '<div class="empty">This memory file has no messages.</div>';
      } catch (error) {
        chat.innerHTML = `<div class="empty">Could not load memory: ${esc(error)}</div>`;
      }
    }
    loadMemory();
  </script>
</body>
</html>"""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SceneRig live dashboard")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="SceneRig run output directory or a single task output directory.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help=(
            "Optional directory containing per-task target image folders. When omitted, "
            "each run's input.png is used."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    target_dir = Path(args.target_dir) if args.target_dir else None
    if target_dir is not None and not target_dir.is_absolute():
        target_dir = repo_root / target_dir
    config = DashboardConfig(
        output_dir=output_dir.resolve(),
        repo_root=repo_root,
        target_dir=target_dir.resolve() if target_dir is not None else None,
        host=args.host,
        port=args.port,
    )
    DashboardHandler.config = config
    server = ThreadingHTTPServer((config.host, config.port), DashboardHandler)
    url = f"http://{config.host}:{config.port}"
    print(f"Serving SceneRig dashboard at {url}")
    print(f"Watching output directory: {config.output_dir}")
    if config.target_dir is not None:
        print(f"Using target directory: {config.target_dir}")
    else:
        print("Using each run's input.png as its target image")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
