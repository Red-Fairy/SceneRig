"""Run discovery, progress, and conversation parsing for the local demo."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, unquote


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
class RunDataConfig:
    output_dir: Path
    repo_root: Path


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


def allowed_roots(config: RunDataConfig) -> list[Path]:
    return [config.output_dir.resolve(), config.repo_root.resolve() / "data"]


def resolve_file(raw_path: str, config: RunDataConfig) -> Optional[Path]:
    path = Path(unquote(raw_path)).resolve()
    if any(is_relative_to(path, root) for root in allowed_roots(config)):
        return path
    return None


def memory_image_url(
    value: str, memory_path: Path, config: RunDataConfig
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
                for root in allowed_roots(config)
            )
        ):
            return file_url(resolved)
    return None


def compact_memory_text(text: str) -> str:
    text = DATA_IMAGE_RE.sub("[embedded image]", text)
    text = PATH_IMAGE_RE.sub("[image]", text)
    return truncate_text(text, MAX_MEMORY_TEXT)


def text_memory_parts(
    text: str, memory_path: Path, config: RunDataConfig,
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
    content: Any, memory_path: Path, config: RunDataConfig
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


def collect_memory(raw_path: str, config: RunDataConfig) -> dict[str, Any]:
    path = resolve_file(raw_path, config)
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
