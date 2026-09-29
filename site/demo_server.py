#!/usr/bin/env python3
"""Detailed, read-only SceneRig run viewer."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "site"))
sys.path.insert(0, str(REPO_ROOT))  # so `lib.*` helpers (surface_relations, ...) import

import run_data  # noqa: E402
OUTPUT_ROOT = REPO_ROOT / "output"
OUTPUT_BASE = REPO_ROOT / "output"
# The demo auto-indexes every output/static_scene* dir as its own sidebar root, keyed by its
# output/-relative path ("static_scene_0721_0731/<run>/scene") so run_dir_for tells archives
# apart from live static_scene runs (keyed "<run>/scene"). BLACKLIST: dirs to ignore entirely
# (not indexed, not served). Add a name here to hide it.
ARCHIVE_SKIP = {
    "static_scene_before_0710",
    "static_scene_0711_0720",
    "static_scene_discarded",  # superseded attempts; hidden from every viewer
}
# static_scene itself is not an "archive" root — it's surfaced separately as "default" (with
# its own keying). Excluded from the archive glob, but NOT from the blacklist / file serving.
_LIVE_ROOT_NAME = "static_scene"

# View-only / single-root mode, set from the CLI in main() (defaults keep the full demo).
# VIEW_ONLY hides the launch/upload panel and 403s the launch/queue POST endpoints; ONLY_ROOT
# (a root LABEL like "benchmark_final", or None) restricts the sidebar to that one archive root
# and drops the live "default" runs. Used by the hosted grase-demo-view service.
VIEW_ONLY = True
ONLY_ROOT: Optional[str] = None


def scene_archive_dirs() -> list[Path]:
    """Live glob of every output/static_scene* dir to surface as its own root, minus the live
    root and the blacklist. Re-globbed on EVERY scan, so a newly-cut folder is auto-indexed
    with no server restart. Newest name first (the UI re-sorts by recency anyway)."""
    return []


def is_scene_file(path: Path) -> bool:
    """True if `path` is under an indexable output/static_scene* dir, so its images serve
    without a static ALLOWED_ROOTS entry (static_scene included; blacklisted dirs excluded)."""
    try:
        rel = path.resolve().relative_to(OUTPUT_BASE.resolve())
    except ValueError:
        return False
    return bool(rel.parts)


def root_label(name: str) -> str:
    """output dir name -> the sidebar's top-level key: static_scene -> "default",
    static_scene_benchmark -> "benchmark", static_scene_0721_0731 -> "0721_0731"."""
    return name[len("static_scene_") :] if name != "static_scene" else "default"
DEMO_RUNS_DIR = REPO_ROOT / "site" / "demo_runs"
TASK_NAME = "scene"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

# Generator/verifier tool sets. Object reconstruction is mandatory preprocessing;
# agent-stage tools only edit and inspect the reconstructed scene.
GENERATOR_TOOLS = (
    "lib/tools/blender/exec.py,lib/tools/generator_base.py,"
    "lib/tools/initialize_plan.py"
)
VERIFIER_TOOLS = "lib/tools/blender/investigator.py,lib/tools/verifier_base.py"

# data/ is allowed so external runs' target images (e.g. data/eval_set/...) render.
ALLOWED_ROOTS = [
    OUTPUT_ROOT.resolve(),
    DEMO_RUNS_DIR.resolve(),
    (REPO_ROOT / "data").resolve(),
]  # archive-root files are served via is_scene_file() (dynamic, no restart needed)

# GPUs the demo may use. Default 0-3 (shared pods leave 4-7 for interactive use);
# the dedicated grase-server pod launches with SCENERIG_DEMO_GPUS=0,...,7 (all 8).
DEMO_GPUS = [int(g) for g in os.environ.get("SCENERIG_DEMO_GPUS", "0,1,2,3").split(",")]
# SAM3D preprocessing is heavy: cap concurrency to one run per configured pool GPU and
# assign the lowest-numbered free GPU first. Excess requests wait in a FIFO queue.
RUN_GPUS = DEMO_GPUS
RUN_MAX = len(RUN_GPUS)

_run_counter = 0
_run_counter_lock = threading.Lock()


@dataclass(kw_only=True, slots=True)
class RunProc:
    run_id: str
    popen: subprocess.Popen
    started: float


_procs: dict[str, RunProc] = {}
_procs_lock = threading.Lock()

# Pipeline admission control.
_active_runs: dict[str, int] = {}  # run_id -> gpu currently held
_run_queue: list[dict[str, Any]] = []  # FIFO of waiting {run_id, dataset_dir, label}
_queue_lock = threading.Lock()

# Interactive 3D scene: export the run's current .blend -> GLB (cached by the
# blend's mtime), served to an in-browser viewer. Only the run being viewed
# triggers an export (run_status is polled per selected run), so this stays cheap.
BLENDER_BIN = os.environ.get(
    "SCENERIG_BLENDER_COMMAND",
    str(REPO_ROOT / "lib" / "utils" / "third_party" / "blender-4.5" / "blender"),
)


def object_iou(item: dict[str, Any]) -> Optional[float]:
    value = (item.get("final") or {}).get("iou")
    if isinstance(value, (int, float)):
        return float(value)
    trace = item.get("trace") or []
    if trace:
        value = trace[-1].get("iou", trace[-1].get("score"))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def scene_mean_iou(_task_dir: Path) -> None:
    return None
_glb_jobs: dict[str, dict[str, Any]] = {}  # run_id -> {"mtime": int, "status": str}
_glb_lock = threading.Lock()
_glb_sem = threading.Semaphore(2)  # cap concurrent blender exports


def current_blend(task_dir: Path) -> Optional[Path]:
    """The .blend representing the run's current state."""
    final = task_dir / "final" / "final.blend"
    if final.exists():
        return final
    accum = task_dir / "blender_file.blend"
    return accum if accum.exists() else None


def _export_glb(run_id: str, blend: Path, glb: Path) -> None:
    with _glb_sem:
        glb.parent.mkdir(parents=True, exist_ok=True)
        # Temp must end in .glb (Blender appends .glb otherwise) and must NOT match
        # the scene_*.glb glob (so in-progress files aren't served).
        tmp = glb.with_name("_building_" + glb.name)
        # bake_export.py bakes procedural materials -> image textures, then
        # exports GLB (glTF can't represent procedural shader networks).
        bake_script = str(REPO_ROOT / "site" / "bake_export.py")
        ok = False
        try:
            r = subprocess.run(
                [
                    BLENDER_BIN,
                    "--background",
                    str(blend),
                    "--python",
                    bake_script,
                    "--",
                    str(tmp),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REPO_ROOT),
            )
            ok = r.returncode == 0 and tmp.exists()
            if ok:
                os.replace(tmp, glb)
        except Exception:  # noqa: BLE001 - export failure -> mark error, viewer falls back
            ok = False
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        with _glb_lock:
            job = _glb_jobs.get(run_id)
            if job:
                job["status"] = "done" if ok else "error"


def scene_state(run_id: str, task_dir: Path) -> dict[str, Any]:
    """GLB url for the current blend; kicks off a background export if stale."""
    blend = current_blend(task_dir)
    if not blend:
        return {"glb_url": None, "building": False}
    try:
        mtime = int(blend.stat().st_mtime)
    except OSError:
        return {"glb_url": None, "building": False}
    glb = task_dir / "web" / f"scene_{mtime}.glb"
    if glb.exists():
        return {"glb_url": rel_file_url(glb), "building": False}
    # Most recent already-exported scene, shown while a newer one builds.
    web = task_dir / "web"
    prev = sorted(web.glob("scene_*.glb")) if web.exists() else []
    prev_url = rel_file_url(prev[-1]) if prev else None
    start = False
    with _glb_lock:
        job = _glb_jobs.get(run_id)
        same = bool(job and job.get("mtime") == mtime)
        if same and job["status"] == "building":
            return {"glb_url": prev_url, "building": True}
        if same and job["status"] == "error":
            return {"glb_url": prev_url, "building": False, "error": True}
        _glb_jobs[run_id] = {"mtime": mtime, "status": "building"}  # new/changed blend
        start = True
    if start:
        threading.Thread(
            target=_export_glb, args=(run_id, blend, glb), daemon=True
        ).start()
    return {"glb_url": prev_url, "building": True}


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return text[:40] or "scene"


def new_run_id(label: str) -> str:
    global _run_counter
    with _run_counter_lock:
        _run_counter += 1
        seq = _run_counter
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # `seq` makes the id unique even for same-second, same-filename uploads.
    return f"{stamp}_{seq:03d}_{slugify(label)}"


def run_dir_for(run_id: str) -> Path:
    candidate = Path(unquote(run_id))
    if candidate.is_absolute():
        if is_relative_to(candidate, OUTPUT_ROOT):
            return candidate.resolve()
        return OUTPUT_ROOT / ".invalid-run"
    # External (manually-run) runs are keyed by their task path relative to
    # OUTPUT_ROOT (contains "/"); demo runs are keyed by a bare id -> demo_<id>/scene.
    if "/" in run_id:
        # Archive runs carry their output/-relative prefix (first segment == a static_scene*
        # dir); resolve against the live filesystem so newly-added roots work without restart.
        first = run_id.split("/", 1)[0]
        if first.startswith("static_scene") and (OUTPUT_BASE / first).is_dir():
            return OUTPUT_BASE / run_id
        return OUTPUT_ROOT / run_id
    return OUTPUT_ROOT / f"demo_{run_id}" / TASK_NAME


def build_command(run_id: str, dataset_dir: Path, gpu: str) -> list[str]:
    return [
        PYTHON,
        "lib/runners/static_scene.py",
        "--dataset-path",
        str(dataset_dir),
        "--task",
        TASK_NAME,
        "--model",
        "claude-opus-5",
        "--max-workers",
        "1",
        "--max-stage-attempts",
        "3",
        "--verifier-max-rounds",
        "5",
        "--blender-command",
        "lib/utils/third_party/blender-4.5/blender",
        "--blender-script",
        "data/static_scene/generator_script.py",
        "--generator-tools",
        GENERATOR_TOOLS,
        "--verifier-tools",
        VERIFIER_TOOLS,
        "--output-dir",
        f"output/static_scene/demo_{run_id}",
        "--test-id",
        f"demo_{run_id}",
        "--gpu-devices",
        gpu,
    ]


# Anthropic rejects images whose base64 exceeds 10 MB and downscales anything
# over ~1568 px on the long edge anyway, so normalize uploads to a real,
# bounded PNG. (Without this, a phone photo re-encoded to PNG by the pipeline's
# get_image_base64 became a 25 MB request -> 400 -> "Failed to get model response".)
MAX_IMAGE_EDGE = 1536


def normalize_image(image_bytes: bytes, dst: Path) -> tuple[int, int]:
    """Write a bounded, true-PNG copy of the upload. Returns final (w, h)."""
    im = Image.open(io.BytesIO(image_bytes))
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, MAX_IMAGE_EDGE / max(w, h))
    if scale < 1.0:
        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    im.save(dst, format="PNG")
    return im.size


def _update_run_manifest(run_id: str, **updates: Any) -> None:
    """Persist restart-safe scheduling metadata without parsing launch logs."""
    path = DEMO_RUNS_DIR / run_id / "run.json"
    data = run_data.safe_read_json(path) or {}
    data.update({"run_id": run_id, "workload": "sam3d_reconstruction", **updates})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# Downscaled previews for the "Input image" pane. Source targets (esp. external
# runs' target.jpg) can be full-res phone photos; the pane only needs a small
# preview, so serve a bounded JPEG instead of shipping the whole file.
THUMB_EDGE = 768
_thumb_cache: dict[tuple[str, float, int], bytes] = {}


def thumb_bytes(path: Path, max_edge: int) -> bytes:
    """Bounded JPEG preview of a raster image (cached by path/mtime/size)."""
    key = (str(path), path.stat().st_mtime, max_edge)
    hit = _thumb_cache.get(key)
    if hit is not None:
        return hit
    im = Image.open(path)
    im = im.convert("RGB")
    im.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    if len(_thumb_cache) > 256:  # simple bound
        _thumb_cache.clear()
    _thumb_cache[key] = data
    return data


def _spawn_run(run_id: str, dataset_dir: Path, gpu: str) -> None:
    """Actually start the pipeline subprocess on the given GPU."""
    cmd = build_command(run_id, dataset_dir, gpu)
    opts = run_data.safe_read_json(DEMO_RUNS_DIR / run_id / "options.json") or {}
    if opts.get("generative_resegment"):
        cmd.append("--generative-resegment")
    if opts.get("room_mode"):
        cmd.append("--room-mode")
    if opts.get("same_size_categories"):
        cmd += ["--same-size-categories", *opts["same_size_categories"]]
    run_dir_for(run_id).mkdir(parents=True, exist_ok=True)
    # SAM3D reconstruction is conditioned on the scene's MoGE-2 point map (the pipeline
    # default). Pin it ON explicitly so a demo server started in a shell that had
    # SCENERIG_SAM3D_POINTMAP=0 still uses MoGE-2 depth; recorded in launch.log.
    env = {**os.environ, "SCENERIG_SAM3D_POINTMAP": "1"}
    launch_log = open(DEMO_RUNS_DIR / run_id / "launch.log", "w")
    launch_log.write(
        "CUDA gpu: "
        + gpu
        + "\nSCENERIG_SAM3D_POINTMAP=1\n"
        + shlex.join(cmd)
        + "\n\n"
    )
    launch_log.flush()
    popen = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    with _procs_lock:
        _procs[run_id] = RunProc(run_id=run_id, popen=popen, started=time.time())
    _update_run_manifest(
        run_id, state="running", gpu=int(gpu), pid=popen.pid, started=time.time()
    )


def launch_run(
    image_bytes: bytes, filename: str, options: Optional[dict] = None
) -> str:
    label = Path(filename or "scene").stem
    run_id = new_run_id(label)
    dataset_dir = DEMO_RUNS_DIR / run_id / "dataset"
    task_dir = dataset_dir / TASK_NAME
    task_dir.mkdir(parents=True, exist_ok=True)
    # Persist per-run pipeline options next to the dataset so queued runs (and
    # runs promoted after a server restart) launch with them; _spawn_run reads it.
    if any((options or {}).values()):
        (DEMO_RUNS_DIR / run_id / "options.json").write_text(json.dumps(options))
    normalize_image(image_bytes, task_dir / "target.png")
    # Create the output dir up front so queued (not-yet-started) runs are still
    # discoverable by list_runs (which globs output/static_scene/demo_*).
    run_dir_for(run_id).mkdir(parents=True, exist_ok=True)
    _update_run_manifest(run_id, state="queued", label=label, created=time.time())

    # Every run is a full SAM3D reconstruction: each loads a ~32 GiB Molmo model,
    # so admit at most one per configured pool GPU (lowest free first) and FIFO-queue
    # the rest. (This
    # is the only mode — colocating two runs on a GPU OOMs the pointing model.)
    with _queue_lock:
        free = [g for g in RUN_GPUS if g not in _active_runs.values()]
        if free:
            gpu = min(free)
            _active_runs[run_id] = gpu
            _spawn_run(run_id, dataset_dir, str(gpu))
        else:
            _run_queue.append(
                {"run_id": run_id, "dataset_dir": dataset_dir, "label": label}
            )
    return run_id


def _adopt_orphan_runs(alive: set[str]) -> None:
    """Restore GPU reservations from explicit manifests after a server restart."""
    for rid in alive:
        if rid in _active_runs:
            continue
        metadata = run_data.safe_read_json(DEMO_RUNS_DIR / rid / "run.json") or {}
        if metadata.get("workload") != "sam3d_reconstruction":
            continue
        gpu = metadata.get("gpu")
        if gpu in RUN_GPUS:
            _active_runs[rid] = gpu


def restore_scheduling_state() -> None:
    """Rehydrate live reservations and the persisted FIFO queue after restart."""
    alive = alive_run_ids()
    queued: list[tuple[float, str, dict[str, Any]]] = []
    try:
        manifest_paths = list(DEMO_RUNS_DIR.glob("*/run.json"))
    except OSError:
        manifest_paths = []
    for path in manifest_paths:
        metadata = run_data.safe_read_json(path) or {}
        if metadata.get("workload") != "sam3d_reconstruction":
            continue
        rid = str(metadata.get("run_id") or path.parent.name)
        if metadata.get("state") != "queued" or rid in alive:
            continue
        dataset_dir = path.parent / "dataset"
        if not (dataset_dir / TASK_NAME / "target.png").exists():
            continue
        item = {
            "run_id": rid,
            "dataset_dir": dataset_dir,
            "label": metadata.get("label") or rid,
        }
        queued.append((float(metadata.get("created") or 0), rid, item))

    with _queue_lock:
        _adopt_orphan_runs(alive)
        known = {q["run_id"] for q in _run_queue} | set(_active_runs)
        for _created, rid, item in sorted(queued, key=lambda row: (row[0], row[1])):
            if rid not in known:
                _run_queue.append(item)
                known.add(rid)


def reap_and_promote() -> None:
    """Free GPUs from finished runs and start the next queued ones."""
    alive = alive_run_ids()
    with _queue_lock:
        _adopt_orphan_runs(alive)
        for rid, _gpu in list(_active_runs.items()):
            with _procs_lock:
                proc = _procs.get(rid)
            is_alive = rid in alive if proc is None else proc.popen.poll() is None
            if not is_alive:
                del _active_runs[rid]
                updates: dict[str, Any] = {"state": "finished", "finished": time.time()}
                if proc is not None:
                    returncode = proc.popen.poll()
                    if returncode is not None:
                        updates["returncode"] = returncode
                _update_run_manifest(rid, **updates)
        free = sorted(g for g in RUN_GPUS if g not in _active_runs.values())
        while _run_queue and free:
            gpu = free.pop(0)
            item = _run_queue.pop(0)
            _active_runs[item["run_id"]] = gpu
            _spawn_run(item["run_id"], item["dataset_dir"], str(gpu))


def _descendant_pids(pid: int) -> list[int]:
    found: list[int] = []
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            out = subprocess.check_output(["pgrep", "-P", str(p)], text=True)
        except Exception:  # noqa: BLE001 - no children / pgrep error
            continue
        for tok in out.split():
            c = int(tok)
            found.append(c)
            stack.append(c)
    return found


def stop_run(run_id: str) -> dict[str, Any]:
    """Stop a run: drop it from the queue if waiting, else kill its whole
    process tree (runner -> main.py -> MCP tool servers -> blender/sam), and
    drop a marker so the UI shows it as 'stopped'."""
    with _queue_lock:
        n0 = len(_run_queue)
        _run_queue[:] = [q for q in _run_queue if q["run_id"] != run_id]
        _active_runs.pop(run_id, None)
        dequeued = len(_run_queue) < n0

    # Find the run's processes by its unique output-dir token, plus descendants.
    try:
        roots = [
            int(t)
            for t in subprocess.check_output(
                ["pgrep", "-f", f"demo_{run_id}"], text=True
            ).split()
        ]
    except Exception:  # noqa: BLE001 - nothing running
        roots = []
    pids: set[int] = set(roots)
    for r in roots:
        pids.update(_descendant_pids(r))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        if sig is signal.SIGTERM and pids:
            time.sleep(2)
    with _procs_lock:
        rp = _procs.get(run_id)
    if rp:
        try:
            rp.popen.kill()
        except Exception:  # noqa: BLE001
            pass

    task_dir = run_dir_for(run_id)
    try:
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "stopped.marker").write_text("stopped by user\n")
    except OSError:
        pass
    _update_run_manifest(run_id, state="stopped")
    return {"stopped": True, "killed": len(pids), "dequeued": dequeued}


def queue_snapshot() -> dict[str, Any]:
    with _queue_lock:
        return {
            "capacity": RUN_MAX,
            "gpus": list(RUN_GPUS),
            "active": len(_active_runs),
            "queued": len(_run_queue),
            "order": [q["run_id"] for q in _run_queue],
        }


def _reaper_loop() -> None:
    while True:
        try:
            reap_and_promote()
        except Exception:  # noqa: BLE001 - never let the reaper thread die
            pass
        time.sleep(3)


# --------------------------------------------------------------------------- #
# Status                                                                       #
# --------------------------------------------------------------------------- #
def rel_file_url(path: Path) -> str:
    return "file?path=" + quote(str(path.resolve()))


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def latest_stage_renders(task_dir: Path) -> list[dict[str, Any]]:
    """One newest generator render per stage, in pipeline order."""
    out: list[dict[str, Any]] = []
    for stage in run_data.STAGE_ORDER:
        best: Optional[Path] = None
        best_mtime = -1.0
        for agent_dir in (task_dir / "stages").glob("*/*"):
            if run_data.stage_from_agent(agent_dir.name) != stage:
                continue
            if "Verifier" in agent_dir.name:
                continue
            for render in run_data.iter_files(agent_dir, run_data.IMAGE_SUFFIXES):
                if run_data.is_verifier_multiview_render(render):
                    continue
                try:
                    mtime = render.stat().st_mtime
                except OSError:
                    continue
                if mtime > best_mtime:
                    best, best_mtime = render, mtime
        if best is not None:
            out.append({"stage": stage, "url": rel_file_url(best)})
    return out


def alive_run_ids() -> set[str]:
    """run_ids whose pipeline process is currently alive (restart-robust).

    One pgrep scan; reads the demo_<id>/scene token from either the preprocessing
    runner or its main.py child.  Looking only for main.py loses the GPU reservation
    during the long SAM3D phase after a demo-server restart.
    """
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "lib/runners/static_scene.py|main.py"], text=True
        )
    except Exception:  # noqa: BLE001 - pgrep missing / no matches
        return set()
    ids: set[str] = set()
    for line in out.splitlines():
        # The runner receives .../demo_<id>; main.py receives
        # .../demo_<id>/scene.  Both own the same admission slot.
        m = re.search(r"demo_([^/ ]+)(?:/scene)?(?:[ /]|$)", line)
        if m:
            ids.add(m.group(1))
    return ids


# A run whose process we can't see is "active" while its task.log moved within this window
# (sized to span a verifier's render + VLM call — see the note in compute_status), else "idle".
_ACTIVE_STALE_SEC = 600


def compute_status(
    task_dir: Path, run_id: str, alive: set[str], queued: Optional[set[str]] = None
) -> str:
    """Status from real liveness, not log-text scraping.

    queued: waiting for a SAM3D GPU slot (not started yet). complete:
    final.blend exists. error: a fatal crash (non-empty fatal_error.log or a
    nonzero process exit) and no final output -- NOT a caught, per-object error
    logged mid-run. active: process alive or the log moved recently. Otherwise
    idle/waiting.
    """
    result_manifest = run_data.safe_read_json(
        task_dir / "final" / "pipeline_result.json"
    ) or {}
    if result_manifest.get("pipeline_complete") is False:
        return "error"
    if (task_dir / "final" / "final.blend").exists():
        return "complete"
    if (task_dir / "stopped.marker").exists():
        return "stopped"
    if queued is not None and run_id in queued:
        return "queued"
    fatal = task_dir / "fatal_error.log"
    with _procs_lock:
        proc = _procs.get(run_id)
    run_manifest = run_data.safe_read_json(
        DEMO_RUNS_DIR / run_id / "run.json"
    ) or {}
    persisted_returncode = run_manifest.get("returncode")
    exited_nonzero = (
        proc is not None and proc.popen.poll() not in (None, 0)
    ) or (
        isinstance(persisted_returncode, int) and persisted_returncode != 0
    )
    try:
        fatal_logged = fatal.exists() and fatal.stat().st_size > 0
    except OSError:
        fatal_logged = False
    if exited_nonzero or fatal_logged:
        return "error"
    if run_id in alive or (proc is not None and proc.popen.poll() is None):
        return "active"
    log = task_dir / "task.log"
    try:
        # task.log mtime is the ONLY liveness signal for runs whose process this server
        # can't see (external `run_e2e.sh` runs, and anything on another pod sharing
        # /fsx — pgrep can't reach it). A verifier does a render + one big multimodal VLM
        # call with no log writes in between; measured legit silences reach ~380 s, so a
        # 120 s window flipped mid-verifier runs to "idle". 600 s covers a slow/retried
        # verifier call with margin; a truly dead run still settles to "idle" after it.
        if log.exists() and (time.time() - log.stat().st_mtime) < _ACTIVE_STALE_SEC:
            return "active"
        if log.exists():
            return "idle"
    except OSError:
        pass
    # No task.log yet != waiting: the runner creates task.log only when the AGENT
    # pipeline launches, so a run sat on "waiting" for its whole ~10-min preprocess.
    # Preprocess liveness comes from the artifacts it writes progressively.
    pre = run_data.preprocess_state(task_dir)
    if pre:
        return "active" if pre["active"] else "idle"
    return "waiting"


def external_target_image(task_dir: Path) -> Optional[Path]:
    """Best-effort: read --target-image-path from the run's task.log."""
    try:
        head = (task_dir / "task.log").read_text(errors="replace")[:2000]
    except OSError:
        return None
    try:
        tokens = shlex.split(head.splitlines()[0])
    except (ValueError, IndexError):
        return None
    value: Optional[str] = None
    for i, token in enumerate(tokens):
        if token == "--target-image-path" and i + 1 < len(tokens):
            value = tokens[i + 1]
            break
        if token.startswith("--target-image-path="):
            value = token.split("=", 1)[1]
            break
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if p.exists() and (any(is_relative_to(p, r) for r in ALLOWED_ROOTS) or is_scene_file(p)):
        return p
    return None


def _quat_to_euler_deg(q: list[float]) -> list[float]:
    """SAM3D stores rotation as a (w, x, y, z) quaternion; convert to XYZ Euler degrees
    for a human-readable pose readout (matches the register panel's euler display)."""
    import math

    w, x, y, z = (float(v) for v in q[:4])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [round(math.degrees(a), 1) for a in (roll, pitch, yaw)]


def _fmt_pose(p: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Format a stored pose ({translation, rotation-quat, scale}) as human-readable
    uniform scale + camera-space translation (m) + XYZ Euler degrees. None -> None."""
    if not p:
        return None
    scale = p.get("scale") or [1.0]
    quat = p.get("rotation") or [1.0, 0.0, 0.0, 0.0]
    return {
        "scale": round(float(scale[0]), 3),
        "translate": [round(float(v), 3) for v in (p.get("translation") or [0, 0, 0])],
        "euler": _quat_to_euler_deg(quat),
    }


def _icp_pose(meshes_dir: Path, slug: str) -> Optional[dict[str, Any]]:
    """The pose SAM3D estimates per object, from the ``{slug}_info.json`` sidecar.
    ``pre``/``post`` bracket the shape-ICP step (None on runs before this was recorded);
    ``final`` is the fully optimized pose. ``applied`` = whether ICP was accepted."""
    info = run_data.safe_read_json(meshes_dir / f"{slug}_info.json")
    if not info or "scale" not in info:
        return None
    return {
        "pre": _fmt_pose(info.get("icp_pre")),
        "post": _fmt_pose(info.get("icp_post")),
        "final": _fmt_pose(info),
        "applied": info.get("icp_applied"),
    }


def _resolve_mask_img(masks_dir: Path, stored: Optional[str]) -> Optional[Path]:
    """Resolve a masks image by BASENAME within THIS run's masks_dir first, then fall back to
    the stored repo-relative/absolute path. masks.json / generative_resegment.json store paths
    that hardcode the run's ORIGINAL location (output/static_scene/<name>/...), so they go stale
    the moment a run is archived (mv to static_scene_<batch>/) or renamed — which silently blanked
    the detected-instances overlays and the occlusion crops for every moved run. None if neither
    location has the file."""
    if not stored:
        return None
    cand = masks_dir / os.path.basename(stored)
    if cand.exists():
        return cand
    p = (REPO_ROOT / stored) if not os.path.isabs(stored) else Path(stored)
    return p if p.exists() else None


def preprocess_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """Phase-1 (MoGE + masking) artifacts for the run, or None if not a MoGE run."""
    import math

    moge_dir, masks_dir = task_dir / "moge", task_dir / "masks"
    if not moge_dir.exists() and not masks_dir.exists():
        return None
    state: dict[str, Any] = {}
    state["progress"] = run_data.safe_read_json(
        task_dir / "preprocess_progress.json"
    )
    depth = moge_dir / "depth_viz.png"
    if depth.exists():
        state["moge_depth_url"] = rel_file_url(depth)
    comp = masks_dir / "composite.png"
    if comp.exists():
        state["masks_composite_url"] = rel_file_url(comp)
    mj = run_data.safe_read_json(moge_dir / "moge.json") or {}
    # Input image aspect ratio (w/h) so discarded-object marker tiles (which have no
    # overlay image to imply a shape) can match the input instead of a forced square.
    iw, ih = mj.get("image_width"), mj.get("image_height")
    if iw and ih:
        state["input_aspect"] = round(float(iw) / float(ih), 4)
    if mj.get("fov_x_deg"):
        fx = float(mj["fov_x_deg"])
        state["camera"] = {
            "fov_x": round(fx, 1),
            "fov_y": round(float(mj["fov_y_deg"]), 1) if mj.get("fov_y_deg") else None,
            "lens": round(18.0 / math.tan(math.radians(fx) / 2.0), 1),
            "width": mj.get("image_width"),
            "height": mj.get("image_height"),
        }
    masks = run_data.safe_read_json(masks_dir / "masks.json") or {}
    meshes_dir = task_dir / "meshes"
    # Occlusion verdicts (generative_resegment): occluder edges + vital flags per
    # instance — used to badge objects whose occlusion WAS detected but whose
    # recovery fell back (no redetect record), which otherwise look untouched.
    reseg = run_data.safe_read_json(masks_dir / "generative_resegment.json") or {}
    occl_of: dict = {}
    for e in reseg.get("edges", []):
        occl_of.setdefault(e.get("occluded"), []).append(e.get("occluder"))
    reseg_vitals = reseg.get("vitals") or {}

    def _tint_overlay(img_path, mask_path, out: Path, color) -> Path:
        """Mask tinted on an image, cached by the MASK's mtime."""
        if not (out.exists() and out.stat().st_mtime >= os.path.getmtime(mask_path)):
            import numpy as _np
            from PIL import Image as _I

            img = _np.asarray(_I.open(img_path).convert("RGB")).astype("float32")
            m = _np.load(mask_path) > 0
            if m.shape != img.shape[:2]:
                m = (
                    _np.asarray(
                        _I.fromarray(m.astype("uint8") * 255).resize(
                            (img.shape[1], img.shape[0])
                        )
                    )
                    > 127
                )
            img[m] = 0.55 * img[m] + 0.45 * _np.array(color, dtype="float32")
            _I.fromarray(img.clip(0, 255).astype("uint8")).save(out)
        return out

    def _redetect_view(r: dict) -> dict | None:
        # Re-directed instance (generative resegment): pair the TRUE original mask
        # (from mask_path — no pass ever rewrites that npy) with the re-detected
        # mask tinted on the EDITED image. The cached overlay_path png is NOT used
        # for the original panel: tier3_component_merge regenerates it from the
        # post-merge ACTIVE mask (0807 misc_IMG_8226: the "original mask" panel
        # showed the merged amodal union covering the monitor's base plate, byte-
        # identical to the redetect panel).
        rd = r.get("redetect") or {}
        mp, ep = rd.get("mask_path"), rd.get("edited_image")
        if not (mp and ep and os.path.exists(mp) and os.path.exists(ep)):
            return None
        try:
            out = _tint_overlay(
                ep, mp, Path(ep).parent / (Path(mp).stem + "_overlay.png"),
                [60.0, 220.0, 60.0],
            )
        except Exception:  # noqa: BLE001 - a broken overlay must not hide the run
            return None
        view = {
            "edited_overlay_url": rel_file_url(out),
            "vital_part": rd.get("vital_part") or "",
            "removed": rd.get("removed") or [],
        }
        mp0 = r.get("mask_path")
        mp0 = (
            str(REPO_ROOT / mp0)
            if mp0 and not os.path.isabs(mp0)
            else mp0
        )
        ref = masks_dir.parent / "input.png"
        if mp0 and os.path.exists(mp0) and ref.exists():
            try:
                out0 = _tint_overlay(
                    ref, mp0, Path(mp0).parent / (Path(mp0).stem + "_origview.png"),
                    [220.0, 70.0, 60.0],
                )
                view["original_overlay_url"] = rel_file_url(out0)
            except Exception:  # noqa: BLE001 - fall back to the active overlay
                pass
        rid = f"{r.get('category')}#{r.get('instance')}"
        view["merged_in"] = [
            m0.get("merged")
            for m0 in masks.get("merge_log") or []
            if m0.get("kept") == rid and m0.get("merged")
        ]
        return view

    insts = []
    for r in masks.get("instances", []):
        ovp = _resolve_mask_img(masks_dir, r.get("overlay_path"))
        cat, inst = r.get("category") or "", r.get("instance")
        slug = re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_") + f"_{inst}"
        rv = reseg_vitals.get(f"{cat}#{inst}") or {}
        rd_view = _redetect_view(r)
        if rd_view is not None:  # carry the VLM occlusion estimate onto the redetect record
            vf = rv.get("visible_fraction")
            rd_view["hidden_pct"] = None if vf is None else round((1 - vf) * 100)
            rd_view["vital_missing"] = bool(rv.get("vital_part_missing"))
            if not rd_view.get("vital_part"):
                rd_view["vital_part"] = rv.get("vital_part") or ""
        insts.append(
            {
                "category": r.get("category"),
                "instance": r.get("instance"),
                "description": r.get("description"),
                "kind": r.get("kind"),
                "tier": r.get("tier"),
                "support": r.get("support"),
                "score": round(r["score"], 2) if r.get("score") is not None else None,
                "depth": round(r["depth"], 2) if r.get("depth") is not None else None,
                "overlay_url": rel_file_url(ovp) if ovp else None,
                "icp": _icp_pose(meshes_dir, slug),
                "redetect": rd_view,
                "occlusion": {
                    "occluders": occl_of.get(f"{cat}#{inst}", []),
                    "vital_part": (reseg_vitals.get(f"{cat}#{inst}") or {}).get(
                        "vital_part", ""
                    ),
                    "severity": (reseg_vitals.get(f"{cat}#{inst}") or {}).get(
                        "severity"
                    ),
                    "hidden_pct": None
                    if (reseg_vitals.get(f"{cat}#{inst}") or {}).get(
                        "visible_fraction"
                    )
                    is None
                    else round(
                        (
                            1
                            - (reseg_vitals[f"{cat}#{inst}"])["visible_fraction"]
                        )
                        * 100
                    ),
                }
                if (
                    f"{cat}#{inst}" in occl_of
                    or (reseg_vitals.get(f"{cat}#{inst}") or {}).get(
                        "vital_part_missing"
                    )
                )
                else None,
            }
        )
    state["instances"] = insts
    state["surfaces_skipped"] = masks.get("surfaces_skipped", [])
    state["merge_log"] = masks.get("merge_log", [])
    # Scene routing: raw router verdict (scene_kind), the effective form + track
    # (routing.form / routing.mode), and the coverage revert when it fired — the only
    # two things that can actually override the router (room-track-off demotion and
    # the >=30%-covered work-surface revert).
    state["routing"] = masks.get("routing") or {}
    state["scene_kind"] = masks.get("scene_kind")
    state["work_surface_reverted"] = masks.get("work_surface_reverted")
    # Discarded objects, shown as markers in the per-object mask gallery: items that got no mask
    # (unsegmented / pruned off the main support) + items the proposer-verifier removed pre-mask.
    dropped = []
    for d in masks.get("dropped", []):
        if isinstance(d, dict):
            d = dict(d)
            ovp = _resolve_mask_img(masks_dir, d.get("overlay_path"))  # geometric drops keep their mask
            d["overlay_url"] = rel_file_url(ovp) if ovp else None
        dropped.append(d)
    review = masks.get("proposal_review") or {}
    # The verifier's drops are already recorded in masks["dropped"] (reason
    # verifier_off_support) by current pipelines -- only synthesize an entry for ids
    # missing from it (older runs), else every verifier drop shows twice.
    seen = {
        f"{d.get('category')}#{d.get('instance')}".lower()
        if isinstance(d, dict)
        else str(d).lower()
        for d in dropped
    }
    for d in review.get("drop") or []:
        if str(d).strip().lower() in seen:
            continue
        dropped.append(
            {"id": str(d), "reason": "verifier removed (off the main support)"}
        )
    state["dropped"] = dropped
    state["proposal_review"] = masks.get(
        "proposal_review"
    )  # the 1-pass verifier's feedback
    # Merge process: every pair the over-segmentation VLM judged, with the zoomed close-ups
    # it saw and its decision — for inspection of why a pair did/didn't merge.
    decisions = []
    for d in masks.get("merge_decisions", []):
        clean = _resolve_mask_img(masks_dir, d.get("clean"))
        marked = _resolve_mask_img(masks_dir, d.get("marked"))
        decisions.append(
            {
                "a": d.get("a"),
                "b": d.get("b"),
                "desc_a": d.get("desc_a"),
                "desc_b": d.get("desc_b"),
                "merge": bool(d.get("merge")),
                "keep": d.get("keep"),
                "clean_url": rel_file_url(clean) if clean else None,
                "marked_url": rel_file_url(marked) if marked else None,
            }
        )
    state["merge_decisions"] = decisions
    # Occlusion process (generative_resegment): every touching pair the pairwise VLM
    # judged (with its crops), hierarchy-dropped edges, vital-part verdicts, and the
    # final redetect set — the full decision trail behind the instance badges.
    gr = run_data.safe_read_json(masks_dir / "generative_resegment.json") or {}
    if gr:
        occ_pairs = []
        for e in gr.get("pairs", []):
            clean = _resolve_mask_img(masks_dir, e.get("clean_crop"))
            marked = _resolve_mask_img(masks_dir, e.get("marked_crop"))
            occ_pairs.append(
                {
                    "a": e.get("a"),
                    "b": e.get("b"),
                    "occluder": e.get("occluder"),
                    "occluded": e.get("occluded"),
                    "reason": e.get("reason"),
                    "clean_url": rel_file_url(clean) if clean else None,
                    "marked_url": rel_file_url(marked) if marked else None,
                }
            )
        vitals = []
        for rid, v in (gr.get("vitals") or {}).items():
            if not v.get("vital_part_missing"):
                continue
            mk = _resolve_mask_img(masks_dir, v.get("marked_crop"))
            vitals.append(
                {
                    "rid": rid,
                    "vital_part": v.get("vital_part"),
                    "reason": v.get("reason"),
                    "marked_url": rel_file_url(mk) if mk else None,
                }
            )
        state["occlusion"] = {
            "pairs": occ_pairs,
            # crop severity + hidden ratio per instance (badges + "~N% hidden")
            "severities": {
                rid: {
                    "severity": v.get("severity"),
                    "hidden_pct": None
                    if v.get("visible_fraction") is None
                    else round((1 - v["visible_fraction"]) * 100),
                }
                for rid, v in (gr.get("vitals") or {}).items()
                if v.get("severity")
            },
            "dropped_edges": [
                {"occluder": e.get("occluder"), "occluded": e.get("occluded")}
                for e in gr.get("dropped_edges", [])
            ],
            "vitals": vitals,
            "redetected": gr.get("redetected", []),
            "border": [k for k, v in (gr.get("border") or {}).items() if v],
        }
    # Canonical root: the VLM-chosen ground reference (floor/table) pinned to z=0.
    grav = mj.get("gravity") or {}
    if grav.get("root"):
        state["root"] = {
            "root": grav.get("root"),
            "surface": grav.get("surface"),
            "z_root": round(float(grav["z_root"]), 3)
            if grav.get("z_root") is not None
            else None,
        }
    croot = masks_dir / "canonical_root.png"
    if croot.exists():
        state["root"] = {**state.get("root", {}), "image_url": rel_file_url(croot)}
    placement = run_data.safe_read_json(task_dir / "placement.json") or {}
    state["placement_count"] = len(placement.get("objects", []))
    # Legacy flip-detection payload retained for API compatibility. The bundled UI no
    # longer renders this panel, but external consumers may still read state["flips"].
    meshes_dir = task_dir / "meshes"
    flips = []
    for fr in run_data.safe_read_json(meshes_dir / "flips.json") or []:
        tgt = meshes_dir / fr.get("target", "")
        cands = []
        for c in fr.get("candidates", []):
            p = meshes_dir / c.get("file", "")
            cands.append(
                {"tag": c.get("tag"), "url": rel_file_url(p) if p.exists() else None}
            )
        flips.append(
            {
                "id": fr.get("id"),
                "chosen_tag": fr.get("chosen_tag"),
                "chosen_idx": fr.get("chosen_idx"),
                "reason": fr.get("reason", ""),
                "target_url": rel_file_url(tgt) if tgt.exists() else None,
                "candidates": cands,
            }
        )
    state["flips"] = flips
    return state


def scene_graph_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """The preprocessing scene graph as a support tree (root surfaces -> objects), the
    main-support FORM, and the proposer's root-surface RELATIONSHIPS (resolved to readable
    surface names + their meaning) — the data the agent initializer builds the room from."""
    graph = run_data.safe_read_json(task_dir / "scene_graph.json")
    if not graph or not graph.get("nodes"):
        return None
    byid = {n.get("id"): n for n in graph["nodes"]}
    nodes = []
    for n in graph["nodes"]:
        nodes.append(
            {
                "id": n.get("id"),
                "category": n.get("category"),
                "kind": n.get("kind"),
                "support": n.get("support") or n.get("parent"),
                "form": n.get("form"),  # tabletop|table on the main support only
            }
        )

    def _name(i: Optional[str]) -> str:  # readable surface name for the relationship
        m = byid.get(i, {})
        return (m.get("description") or "").strip() or m.get("category") or (i or "?")

    from lib.tools.geometry.surface_relations import RELATIONSHIP_MEANINGS

    rels = []
    for r in graph.get("relationships", []) or []:
        t = r.get("type")
        rels.append(
            {
                "type": t,
                "a": r.get("a"),
                "b": r.get("b"),
                "a_name": _name(r.get("a")),
                "b_name": _name(r.get("b")),
                "meaning": RELATIONSHIP_MEANINGS.get(t, ""),
            }
        )
    # The two geometric relationship gates (preprocess backstops): against upgrades
    # + impossible-corner drops, persisted by preprocess into masks.json.
    masks_data = run_data.safe_read_json(task_dir / "masks" / "masks.json") or {}
    backstop = masks_data.get("rel_backstop") or {}
    # Same-size groups.  The final placement rows are the authority for which groups
    # actually survived mask/placement/mesh validation and were normalized.  The
    # versioned masks registry contributes source/display/minimum provenance without
    # allowing a merely proposed or inactive group to look scale-locked.
    registry_groups = {
        str(group.get("group_id")): group
        for group in (masks_data.get("same_size_resolution") or {}).get("groups", [])
        if isinstance(group, dict) and group.get("group_id")
    }
    placement = run_data.safe_read_json(task_dir / "placement.json") or {}
    groups: dict[str, dict[str, Any]] = {}
    for o in placement.get("objects", []) or []:
        if not o.get("same_size"):
            continue
        cat = o.get("category") or "?"
        group_id = str(o.get("same_size_group_id") or f"legacy:{cat}")
        registered = registry_groups.get(group_id) or {}
        g = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "category": registered.get("canonical_category") or cat,
                "display_name": registered.get("display_name")
                or o.get("same_size_group_display_name")
                or o.get("same_size_display_name")
                or cat,
                "source": registered.get("source")
                or o.get("same_size_group_source")
                or o.get("same_size_source")
                or "legacy",
                "minimum_members": int(
                    registered.get("minimum_members")
                    or o.get("same_size_group_minimum_members")
                    or o.get("same_size_minimum_members")
                    or 2
                ),
                "members": [],
                "size": None,
            },
        )
        g["members"].append(f"{cat}#{o.get('instance')}")
        if g["size"] is None and o.get("size"):
            g["size"] = [round(float(v), 3) for v in o["size"]]
    # A valid current placement should contain only active groups.  Keep the explicit
    # comparison for legacy/debug artifacts so the UI never implies a lock below the
    # source-specific threshold (manual=2, automatic=3).
    for g in groups.values():
        g["unified"] = len(g["members"]) >= g["minimum_members"]
    same_size = sorted(
        groups.values(), key=lambda g: (g["display_name"], g["group_id"])
    )
    return {
        "nodes": nodes,
        "relationships": rels,
        "backstop": backstop,
        "same_size": same_size,
    }


# Pose-panel severity thresholds. NOT new numbers — these mirror the pipeline's own
# caps so the panel flags exactly what a stage would call a failure:
#   physics.CAPSIZE_DEG / CAPSIZE_DISP_MM      (the preprocess drop test)
#   composition_physics.CERT_TILT_CAP_DEG / CERT_DXY_CAP_M   (the certify free sim)
_CAPSIZE_DEG, _CAPSIZE_DISP_MM = 45.0, 50.0
_CERT_TILT_DEG, _CERT_DXY_MM = 10.0, 50.0


def pose_severity(ch: dict, dv: dict, cv: dict) -> Optional[str]:
    """"red" | "amber" | None for one object's pose record.

    Replaces the old chip, which fired on ``delivered.capsized`` — delivered pose vs the
    MoGE placement. In a monocular reconstruction the whole scene grounds by hundreds of
    mm, so that threshold is crossed by construction: it painted 142 of 701 objects
    (20%) red across the 0808 benchmark while MISSING 5 objects that actually failed a
    stage, and in misc_still_life it flagged all six rollables while leaving jug_0
    unflagged at 962 mm — further than anything it did flag.

    * red   — a stage MEASURED this object failing, so it is actionable: the preprocess
              drop test capsized it, or the certify free sim toppled it / moved it past
              the certify caps. BOTH stages use the same rollable split
              ``physics.capsized`` uses — rollables by DISPLACEMENT, everything else by
              TILT — because "a ROLLABLE object rolls benignly, so tilt is meaningless
              for it". Applying the tilt cap to rollables at certify (the first cut of
              this function) painted 19 extra objects red for rotating in place: a
              marker at 20.9 deg that had moved 2.9 mm, a pen at 57.5 deg / 6.3 mm. That
              is the same misjudgement whose cost is recorded in composition_physics
              (a lying mic read as "capsize 118-180 deg" and burned 5-11 rejected moves
              per run on abc3).
    * amber — delivered sits past the capsize TILT vs the placement while both per-stage
              checks read clean. This is the cross-stage blind spot ``delivered`` was
              added for (0725_snapdown2's vase: 82 deg at preprocess certify, then
              0.0 deg / 0.1 mm at composition certify, ``fell`` false — invisible
              everywhere). Kept as its own tier so it cannot be confused with a
              stage failure.
    * None  — large delivered DISPLACEMENT alone. That is grounding: a photo-derived
              placement dropped onto real support. Expected, not a defect (96 of the 110
              objects the old rule painted red).

    Legacy runs (pre-2026-07-31) carry no ``delivered``; they degrade to the drop-test
    rule rather than raising.
    """
    rollable = bool(ch.get("rollable"))
    drift = float(ch.get("disp_raw_mm") or 0.0)
    tilt = abs(float(ch.get("tilt_deg") or 0.0))
    drop_failed = drift > _CAPSIZE_DISP_MM if rollable else tilt > _CAPSIZE_DEG

    cert_drift = float(cv.get("dxy") or 0.0) * 1000.0
    cert_tilt = abs(float(cv.get("tilt_deg") or 0.0))
    # ``toppled`` is certify's own verdict and is already rollable-aware, so it
    # overrides for either class.
    cert_failed = bool(cv.get("toppled")) or (
        cert_drift > _CERT_DXY_MM if rollable else cert_tilt > _CERT_TILT_DEG
    )

    if drop_failed or cert_failed:
        return "red"
    if abs(float(dv.get("tilt_deg") or 0.0)) > _CAPSIZE_DEG:
        return "amber"
    return None


def pose_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """Physics-settle + grounding pose changes per object (pitch/roll tilt, rest z-shift).

    Three DIFFERENT verdicts live in pose_changes.json and they answer different
    questions (see physics.write_pose_changes / composition_physics.delivered_vs_placed):

    * ``fell``      — the settle ladder accepted a fallen/flipped rest at preprocess.
                      An accepted flip-to-stable-face lands here and is CORRECT, so it
                      is reported apart from the rest (this panel used to paint the
                      whole set red as "cannot stand, >45deg", which was neither the
                      predicate nor the truth).
    * ``delivered`` — placed pose vs DELIVERED pose. The one that says whether the file
                      being shipped has an object on its face.
    * ``composition_certify`` — drift of the final free sim only. Small by construction
                      once an object capsized earlier, so it is shown as a per-object
                      row, never as the headline.

    Pre-2026-07-31 runs wrote ``toppled`` instead of ``fell`` and no ``delivered``."""
    data = run_data.safe_read_json(task_dir / "physics" / "pose_changes.json")
    if not data or not data.get("objects"):
        return None
    delivered = data.get("delivered") or {}
    cert = data.get("composition_certify") or {}
    objs = []
    for name, ch in data["objects"].items():
        label = name[4:] if name.startswith("obj_") else name
        dv, cv = delivered.get(name) or {}, cert.get(name) or {}
        objs.append(
            {
                "name": label,
                "tilt_deg": round(float(ch.get("tilt_deg", 0.0)), 1),
                # legacy runs: `toppled` was this same flag under the old name
                "fell": bool(ch.get("fell", ch.get("toppled", False))),
                "flip_accepted": bool(ch.get("flip_accepted")),
                # placed -> delivered (absent on runs that predate the measure)
                "delivered": (
                    {
                        "tilt_deg": round(float(dv.get("tilt_deg", 0.0)), 1),
                        "dxy_mm": round(float(dv.get("dxy", 0.0)) * 1000.0, 1),
                        "capsized": bool(dv.get("capsized")),
                    }
                    if dv
                    else None
                ),
                # drift of the delivered-scene free sim alone — BOTH components: the
                # tilt alone cannot say whether an object fell over (a fruit rolling
                # 262 mm off a plate barely reads as tilt until it lands)
                "cert_tilt_deg": (
                    round(float(cv["tilt_deg"]), 1) if cv.get("tilt_deg") is not None
                    else None
                ),  # fmt: skip
                "cert_dxy_mm": (
                    round(float(cv["dxy"]) * 1000.0, 1) if cv.get("dxy") is not None
                    else None
                ),  # fmt: skip
                "cert_toppled": bool(cv.get("toppled")),
                # "red" | "amber" | None — see pose_severity for why this is NOT
                # delivered.capsized any more
                "severity": pose_severity(ch, dv, cv),
                "rest_mm": round(float(ch.get("rest_dz", 0.0)) * 1000.0, 1),
                # raw | pristine (rescued) | stabilized | rolled_back | raw_capped
                "chosen": ch.get("chosen"),
                "tilt_raw": (
                    round(float(ch["tilt_raw"]), 1)
                    if ch.get("tilt_raw") is not None
                    else None
                ),
                # rollables are disp-gated (rolling in place is benign): surface
                # the raw drift and, for the roll-back rung, how far the settled
                # rotation was translated back to the placed xy
                "rollable": bool(ch.get("rollable")),
                "disp_raw_mm": (
                    round(float(ch["disp_raw_mm"]), 1)
                    if ch.get("disp_raw_mm") is not None
                    else None
                ),
                "rollback_mm": (
                    round(float(ch["rollback_xy_mm"]), 1)
                    if ch.get("rollback_xy_mm") is not None
                    else None
                ),
            }
        )
    # Severity first, then the larger per-stage magnitude. The old key was delivered
    # tilt, which floats grounding noise to the top of the list — the objects a reader
    # can act on are the ones a STAGE flagged.
    _rank = {"red": 2, "amber": 1}

    def _sort_key(o):
        cert = max(abs(o.get("cert_tilt_deg") or 0.0), (o.get("cert_dxy_mm") or 0.0) / 5.0)
        drop = max(abs(o["tilt_deg"]), (o.get("disp_raw_mm") or 0.0) / 5.0)
        return (-_rank.get(o["severity"], 0), -max(cert, drop))

    objs.sort(key=_sort_key)
    return {
        "objects": objs,
        "n_red": sum(1 for o in objs if o["severity"] == "red"),
        "n_amber": sum(1 for o in objs if o["severity"] == "amber"),
        "table": data.get("table"),
        "table_shift_mm": round(float(data.get("table_shift", 0.0)) * 1000.0, 1),
        "has_delivered": bool(delivered),
        # False = the certify sim hit its step cap mid-motion: the baked poses are a
        # snapshot, not a rest. Was stdout-only until 2026-07-31.
        "certify_converged": data.get("composition_certify_converged", True),
    }


def pose_match_state(task_dir: Path) -> Optional[list[dict[str, Any]]]:
    """Per-object point-cloud ICP result (post-settle): rms before/after, the applied
    [x, y, yaw, scale] correction, and whether it was accepted."""
    data = run_data.safe_read_json(task_dir / "pose_match.json")
    if not isinstance(data, list) or not data:
        return None
    out = []
    for v in data:
        t = v.get("t") or [0, 0, 0]
        rb, ra = v.get("rms_before"), v.get("rms_after")
        out.append(
            {
                "obj": v.get("obj"),
                "n_pts": v.get("n_pts"),
                "rms_before_mm": None if rb is None else round(rb * 1000, 1),
                "rms_after_mm": None if ra is None else round(ra * 1000, 1),
                "improve_pct": None
                if not rb or ra is None
                else round(100 * (1 - ra / rb)),
                "dx_mm": round(t[0] * 1000, 1),
                "dy_mm": round(t[1] * 1000, 1),
                "yaw_deg": None if v.get("yaw_deg") is None else round(v["yaw_deg"], 1),
                "scale": None if v.get("scale") is None else round(v["scale"], 3),
                "accepted": bool(v.get("accepted")),
                "skipped": v.get("skipped"),
            }
        )
    out.sort(key=lambda o: -(o["improve_pct"] or 0))
    return out


# ---- Weighted progress: historical stage durations -> a bar that moves during ----
# ---- preprocess (substage detection via on-disk artifacts).                   ----
_HIST_WEIGHTS: Optional[dict] = None
# Fallbacks measured over the 0709 batches (seconds).
_DEFAULT_STAGE_W = {
    "preprocess": 1460.0, "initializer": 400.0, "texture": 355.0,
    "composition": 126.0, "composition_certify": 200.0, "lighting": 118.0, "export": 11.0,
}
_DEFAULT_STEP_W = {
    "depth": 71.0, "segmentation": 229.0, "resegment": 283.0, "canonicalize": 1.0,
    "pseudo_gt": 262.0, "placement+graph": 3.0, "meshes": 395.0, "settle": 207.0,
    "vlm_physics": 120.0, "camera_lock": 3.0, "pose_match": 8.0,
}


def stage_weights() -> dict:
    """Mean per-stage / per-preprocess-step wall seconds over every run that
    recorded stage_timings.json (computed once per server process)."""
    global _HIST_WEIGHTS
    if _HIST_WEIGHTS is not None:
        return _HIST_WEIGHTS
    import glob as _glob

    acc: dict[str, list[float]] = {}
    step_acc: dict[str, list[float]] = {}
    for f in _glob.glob(str(OUTPUT_ROOT / "*" / "*" / "stage_timings.json")):
        t = run_data.safe_read_json(Path(f)) or {}
        for k, v in t.items():
            if k == "preprocess_steps" and isinstance(v, dict):
                for sk, sv in v.items():
                    step_acc.setdefault(sk, []).append(float(sv))
            elif isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
    stage_w = dict(_DEFAULT_STAGE_W)
    for k, vals in acc.items():
        if vals:
            stage_w[k] = sum(vals) / len(vals)
    step_w = dict(_DEFAULT_STEP_W)
    for k, vals in step_acc.items():
        if vals:
            step_w[k] = sum(vals) / len(vals)
    _HIST_WEIGHTS = {"stages": stage_w, "steps": step_w}
    return _HIST_WEIGHTS


# Preprocess substeps in pipeline order -> the artifact that marks each COMPLETE.
_PRE_MARKERS = [
    ("depth", "moge/points.npy"),
    ("segmentation", "masks/masks.json"),
    ("resegment", "masks/generative_resegment.json"),
    ("pseudo_gt", "pseudo_gt/cameras.json"),
    ("placement+graph", "scene_graph.json"),
    ("settle", "physics/pose_changes.json"),
    ("pose_match", "pose_match.json"),
]
# composition_certify (the final whole-scene physics certify settle) is a real, often
# multi-minute stage between composition and export; include it so progress reflects it
# instead of sitting at 99% for its whole duration.
_STAGE_SEQ = ["preprocess", "initializer", "texture", "lighting", "composition", "composition_certify", "export"]


def weighted_progress(task_dir: Path, log_info: dict) -> Optional[dict[str, Any]]:
    """Overall percent weighted by historical stage durations; during preprocess the
    substage is detected from on-disk artifacts (the old estimator sat at 0% for the
    entire ~25-minute preprocess)."""
    try:
        w = stage_weights()
        stage_w, step_w = w["stages"], w["steps"]
        run_t = run_data.safe_read_json(task_dir / "stage_timings.json") or {}
        total = sum(stage_w.get(k, 0.0) for k in _STAGE_SEQ)
        done = sum(stage_w[k] for k in run_t if k in stage_w and k != "preprocess_steps"
                   and k != "preprocess")
        if "preprocess" in run_t:
            done += stage_w["preprocess"]
            # an AGENT stage is active: credit half its historical weight
            active = log_info.get("active_stage")
            label = active or "finalizing"
            key = "initializer" if active == "initializer" else active
            if key in stage_w and key not in run_t:
                done += 0.5 * stage_w[key]
        else:
            manifest = run_data.safe_read_json(
                task_dir / "preprocess_progress.json"
            )
            if isinstance(manifest, dict) and manifest.get("version") == 1:
                manifest_state = str(manifest.get("state") or "running")
                total_steps = manifest.get("total_steps") or []
                completed = set(manifest.get("completed_steps") or [])
                skipped = set(manifest.get("skipped_steps") or [])
                active_steps = manifest.get("active_steps") or []
                pre_total = sum(step_w.get(k, 1.0) for k in total_steps)
                pre_done = sum(
                    step_w.get(k, 1.0)
                    for k in total_steps
                    if k in completed or k in skipped
                )
                # Give an active long step partial visual credit while keeping the
                # manifest (not directory creation) authoritative for its label.
                if manifest_state == "running":
                    pre_done += 0.5 * sum(
                        step_w.get(k, 1.0)
                        for k in active_steps
                        if k in total_steps and k not in completed
                    )
                elif manifest_state == "complete":
                    pre_done = pre_total
                frac = min(1.0, pre_done / pre_total) if pre_total else 0.0
                done += frac * stage_w["preprocess"]
                label = "preprocess · " + (
                    " + ".join(str(k) for k in active_steps)
                    or (
                        manifest_state
                        if manifest_state in {"complete", "failed"}
                        else "finalizing"
                    )
                )
                pct = max(1, min(99, round(100 * done / total))) if total else 1
                return {"percent": pct, "label": f"{label} ({pct}%)"}

            # Older runs: infer completed preprocess substeps from artifacts.
            # resegment is OPTIONAL (--generative-resegment, off in the demo): its
            # marker never appears on a disabled run, so the first-missing-marker scan
            # would sit on "resegment" for the whole rest of preprocess (0725
            # tableverse read as stuck while it was actually settling). Drop the
            # marker (and its weight) unless the run actually enabled it.
            markers = _PRE_MARKERS
            run_args = run_data.safe_read_json(task_dir.parent / "args.json") or {}
            if not run_args.get("generative_resegment"):
                markers = [m for m in markers if m[0] != "resegment"]
            pre_total = sum(step_w.get(k, 0.0) for k, _ in markers) + step_w.get(
                "meshes", 0.0
            )
            pre_done, nxt = 0.0, markers[0][0]
            for k, marker in markers:
                if (task_dir / marker).exists():
                    pre_done += step_w.get(k, 0.0)
                    if k == "placement+graph":  # meshes come between graph and settle
                        if any((task_dir / "meshes").glob("*.glb")):
                            pre_done += step_w.get("meshes", 0.0)
                else:
                    nxt = k
                    break
            else:
                nxt = "camera lock"
            frac = pre_done / pre_total if pre_total else 0.0
            done += frac * stage_w["preprocess"]
            label = f"preprocess \u00b7 {nxt}"
        pct = max(1, min(99, round(100 * done / total))) if total else 1
        return {"percent": pct, "label": f"{label} ({pct}%)"}
    except Exception:  # noqa: BLE001 - progress is cosmetic; fall back to the old one
        return None


def _reg_physics(ph: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a trace round's physics-settle block to flat display fields. `cum_tilt_deg`
    may be a scalar or a per-member dict; collapse to the worst (max) tilt. Missing -> Nones
    (the move had no settle, e.g. a pure render-compare step)."""
    if not ph:
        return {"tilt": None, "lift_mm": None, "accepted": None}
    tilt = ph.get("cum_tilt_deg")
    if isinstance(tilt, dict):
        tilt = max(tilt.values(), default=0.0) if tilt else None
    return {
        "tilt": round(float(tilt), 1) if tilt is not None else None,
        "lift_mm": ph.get("lift_mm"),
        "accepted": ph.get("accepted"),
    }


def register_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """Per-object pose-refinement log written by the composition stage's PoseSession:
    IoU before/after, the line-search trace, and the reference vs refined-render crops.
    None if the run has no `register/register.json` yet. (The directory name predates the
    2026-07-10 merge of the standalone register stage into composition.)"""
    reg_dir = task_dir / "register"
    data = run_data.safe_read_json(reg_dir / "register.json")
    if not data or not data.get("objects"):
        return None
    in_progress = data.get("in_progress")
    objs, befores, afters, improved = [], [], [], 0
    for o in data["objects"]:
        oid = o.get("id", "?")
        cat, _, inst = oid.partition("#")
        slug = re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_") + "_" + (inst or "0")
        ref, ren = reg_dir / f"obj_{slug}_ref.png", reg_dir / f"obj_{slug}.png"
        tr = o.get("trace", [])
        refined = bool(tr)  # the composition agent investigated + moved this object
        # IoU before/after the pose-refinement loop. `final` holds the current committed
        # score (the measured baseline for objects never refined); the trace holds the
        # per-round trajectory. Prefer `final.iou` for `after`, fall back to the last round.
        cur = object_iou(o)  # final.iou -> last trace round; shared with the galleries
        if refined:
            before = float(tr[0].get("iou", tr[0].get("score", 0.0)))
            after = float(cur if cur is not None else before)
        else:
            # Never refined: only the baseline IoU exists (in `final`) — show it as-is
            # rather than a misleading 0.000 from the empty trace.
            before = after = float(cur) if cur is not None else 0.0
        active = oid == in_progress  # this object is still aligning
        if not active:  # summary stats over committed only
            befores.append(before)
            afters.append(after)
            improved += 1 if after > before + 1e-6 else 0
        pose = o.get("pose", {})
        tpose = [round(x, 3) for x in pose.get("translate", [0, 0, 0])]
        epose = [round(x, 2) for x in pose.get("euler", [0, 0, 0])]
        spose = round(float(pose.get("scale", 1)), 3)
        moved = any(tpose) or any(epose) or abs(spose - 1.0) > 1e-6
        objs.append(
            {
                "id": oid,
                "active": active,
                "before": round(before, 3),
                "after": round(after, 3),
                "delta": round(after - before, 3),
                "refined": refined,  # investigated + moved (vs. placement kept as-is)
                "moved": moved,  # agent issued an explicit move vs. physics-settle only
                "pose": {"translate": tpose, "euler": epose, "scale": spose},
                "trace": [
                    {
                        "round": t.get("round"),
                        "axis": t.get("axis"),
                        "iou": round(float(t.get("iou", t.get("score", 0.0))), 4),
                        "gain": round(float(t["gain"]), 4) if "gain" in t else None,
                        # Physics-settle outcome of the move (current pipeline): net tilt
                        # and lift of the object's support group, and whether it was kept.
                        **_reg_physics(t.get("physics")),
                        "render_url": rel_file_url(reg_dir / t["render"])
                        if t.get("render") and (reg_dir / t["render"]).exists()
                        else None,
                    }
                    for t in tr
                ],
                "ref_url": rel_file_url(ref) if ref.exists() else None,
                "render_url": rel_file_url(ren) if ren.exists() else None,
            }
        )
    # investigate composites (merged pipeline): the LEFT render | RIGHT tinted-photo
    # crops the composition agent actually saw, in call order.
    investigations = []
    for f in sorted(
        reg_dir.glob("investigate_[0-9]*.png"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
    ):
        if f.stem.endswith(("_render", "_scene", "_photo")):
            continue  # the agent's per-crop files; the panel shows the composite
        investigations.append(
            {"n": int(re.sub(r"\D", "", f.stem) or 0), "url": rel_file_url(f)}
        )
    n = len(befores)
    return {
        "count": len(objs),
        "improved": improved,
        "in_progress": in_progress,
        "mean_before": round(sum(befores) / n, 3) if n else 0.0,
        "mean_after": round(sum(afters) / n, 3) if n else 0.0,
        "investigations": investigations,
        "objects": objs,
    }


_STATE_VIZ = [
    ("placement", "raw placement (SAM3D/MoGE) — may float / interpenetrate"),
    ("icp", "after settle + ICP — photo-aligned, collisions allowed"),
    ("simulated", "after joint physics settle — simulation-ready"),
]


def state_viz_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """White-background object-set snapshots at the three preprocess pose milestones.
    None until the first snapshot exists (older runs never rendered them)."""
    viz = task_dir / "physics" / "viz"
    states = [
        {"tag": tag, "caption": cap, "url": rel_file_url(viz / f"state_{tag}.png")}
        for tag, cap in _STATE_VIZ
        if (viz / f"state_{tag}.png").exists()
    ]
    return {"states": states} if states else None


def pseudo_gt_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """The pseudo-GT novel views built in preprocessing: per (azimuth, elevation) orbit, the
    raw novel-view render and the GPT-Image-2 completion. Trust metadata says whether the
    composition agent may compare against that completion; failed or excessive-drift
    artifacts remain visible here but are withheld from the agent. The render is SHARP
    feed-forward 3DGS (``render_raw.png``, current backend),
    falling back to the legacy MoGE point-cloud splat (``render.png``). None if the run
    built no pseudo-GT set."""
    pgt_dir = task_dir / "pseudo_gt"
    data = run_data.safe_read_json(pgt_dir / "cameras.json")
    if not data or not data.get("views"):
        return None
    views = []
    for v in data["views"]:
        vd = pgt_dir / v.get("tag", "")
        completed = vd / "completed.png"
        # Inpaint source render: SHARP render first -> point-cloud splat -> n/a.
        render, render_kind = None, None
        for name, kind in (("render_raw.png", "sharp"), ("render.png", "splat")):
            if (vd / name).exists():
                render, render_kind = vd / name, kind
                break
        mask = next(
            (vd / n for n in ("hole_mask_vis.png", "hole_mask.png") if (vd / n).exists()),
            None,
        )
        views.append(
            {
                "az": v.get("az"),
                "el": v.get("el"),
                "tag": v.get("tag", ""),
                "completed_url": rel_file_url(completed)
                if completed.exists()
                else None,
                "render_url": rel_file_url(render) if render else None,
                "render_kind": render_kind,  # "sharp" | "splat" | None
                "mask_url": rel_file_url(mask) if mask else None,
                # Legacy camera files predate the trust field; preserve their prior
                # behavior. New preprocess runs explicitly mark drift/failure cases
                # false, and the composition tool withholds those references.
                "trusted": v.get("pseudo_gt_trusted", True) is not False,
                "untrusted_reason": v.get("pseudo_gt_untrusted_reason"),
                "keep_drift": v.get("keep_drift"),
            }
        )
    return {
        "count": len(views),
        "trusted_count": sum(v["trusted"] for v in views),
        "views": views,
    }


def run_status(run_id: str) -> dict[str, Any]:
    task_dir = run_dir_for(run_id)
    external = "/" in run_id
    # The MoGE preprocess copies the source image here; prefer it (always under an
    # allowed root, so it serves even when the dataset lives outside the repo).
    saved_input = task_dir / "input.png"
    if saved_input.exists():
        dataset_target = saved_input
    elif external:
        dataset_target = external_target_image(task_dir)
    else:
        dataset_target = DEMO_RUNS_DIR / run_id / "dataset" / TASK_NAME / "target.png"

    if not task_dir.exists() and not (dataset_target and dataset_target.exists()):
        return {"run_id": run_id, "status": "not_found"}

    log_info = run_data.parse_log(task_dir)
    attempts = run_data.collect_attempts(task_dir)
    progress = weighted_progress(task_dir, log_info) or run_data.estimate_progress(
        attempts, log_info
    )

    final_dir = task_dir / "final"
    final_blend = final_dir / "final.blend"
    final_render = final_dir / "renders" / "final_render.png"
    manifest = run_data.safe_read_json(final_dir / "pipeline_result.json")

    queue = queue_snapshot()
    queued_set = set(queue["order"])
    status = compute_status(task_dir, run_id, alive_run_ids(), queued_set)
    if status == "complete":
        progress = {"percent": 100, "label": "complete"}
    queue_position = queue["order"].index(run_id) + 1 if run_id in queued_set else None
    if status == "queued":
        progress = {
            "percent": 0,
            "label": f"queued — position {queue_position} of {queue['queued']}",
        }

    with _procs_lock:
        proc = _procs.get(run_id)

    latest_render = run_data.newest_file(
        r
        for r in run_data.iter_files(task_dir, run_data.IMAGE_SUFFIXES)
        if not run_data.is_verifier_multiview_render(r)
    )

    started = proc.started if proc else None
    stage = log_info.get("active_stage")
    if stage is None and status in ("active", "idle"):
        if run_data.preprocess_state(task_dir):
            stage = "preprocess"
    return {
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "progress": progress,
        "elapsed": round(time.time() - started) if started else None,
        "target_url": rel_file_url(dataset_target)
        if (dataset_target and dataset_target.exists())
        else None,
        "latest_render_url": rel_file_url(latest_render) if latest_render else None,
        "stage_renders": latest_stage_renders(task_dir),
        "final_render_url": rel_file_url(final_render)
        if final_render.exists()
        else None,
        "final_blend_url": rel_file_url(final_blend) if final_blend.exists() else None,
        "scene_name": (manifest or {})
        .get("stage_artifacts", {})
        .get("0", {})
        .get("initializer", {})
        .get("scene_graph", {})
        .get("scene_graph", {})
        .get("scene", {})
        .get("name")
        if manifest
        else None,
        "log_tail": log_info.get("tail", ""),
        "queue_position": queue_position,
        "queue": queue,
        "scene": scene_state(run_id, task_dir),
        "preprocess": preprocess_state(task_dir),
        "scene_graph": scene_graph_state(task_dir),
        "pose": pose_state(task_dir),
        "pose_match": pose_match_state(task_dir),
        "register": register_state(task_dir),
        "state_viz": state_viz_state(task_dir),
        "pseudo_gt": pseudo_gt_state(task_dir),
        # Per-stage wall-clock seconds: {preprocess, initializer, texture, lighting,
        # composition, composition_certify, export}. None until stage_timings.json exists.
        "timings": run_data.safe_read_json(task_dir / "stage_timings.json"),
    }


# /api/runs is polled every few seconds by every open tab. The sidebar only needs
# label/status/mtime per run — NOT the per-stage info from parse_log, which
# dominated the scan (~3.6s) and made the whole listing ~13s on fsx, so the
# sidebar sat on its placeholder until the first scan returned. We now skip
# parse_log here (the detail view's /api/run still parses it on click), list
# *every* run, and cache the result behind a lock so overlapping polls don't each
# rebuild it.
# A full scan of ~1k+ run dirs across every archive root takes ~60-90 s on cold Lustre, so
# refresh sparingly — with stale-while-revalidate the list is served instantly from cache
# regardless, and a background rebuild every ~5 min keeps Lustre from being hammered
# continuously (a 30 s TTL made the scan run back-to-back and spiked pod I/O).
RUNS_CACHE_TTL = 300.0
_runs_cache: dict[str, Any] = {"t": 0.0, "data": None}
_refresh_lock = threading.Lock()
_runs_refreshing = False


def list_runs() -> list[dict[str, Any]]:
    # Stale-while-revalidate. The scan walks ~1k+ run dirs across every archive root and
    # on cold Lustre exceeds the TTL; the old code rebuilt WHILE HOLDING the lock, so once
    # the cache went stale EVERY /api/runs blocked on the multi-second scan and the sidebar
    # never loaded. Now: once built, always return the cached list immediately; when it goes
    # stale, kick off ONE background rebuild and keep serving the stale copy until it lands.
    data = _runs_cache["data"]
    if data is None:
        # No cache yet (fresh server): kick the build off in the BACKGROUND and return an
        # empty list now. A cold scan of ~1k+ run dirs on Lustre takes tens of seconds;
        # building it synchronously here blocked the very first /api/runs (and every poll
        # queued behind it) so the sidebar never appeared. The 5 s poll picks up the runs
        # once the background build lands.
        _refresh_runs_async()
        return []
    if time.time() - _runs_cache["t"] >= RUNS_CACHE_TTL:
        _refresh_runs_async()  # stale: refresh in the background, serve the stale copy now
    return data


def _refresh_runs_async() -> None:
    """(Re)build the runs cache in a background thread (at most one in flight)."""
    global _runs_refreshing
    with _refresh_lock:
        if _runs_refreshing:
            return
        _runs_refreshing = True

    def _worker() -> None:
        global _runs_refreshing
        try:
            runs = _build_runs()
            _runs_cache["data"] = runs
            _runs_cache["t"] = time.time()
        finally:
            _runs_refreshing = False

    threading.Thread(target=_worker, daemon=True).start()


def _build_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    alive = alive_run_ids()
    queued_set = set(queue_snapshot()["order"])
    for entry in run_data.discover_runs(OUTPUT_ROOT):
        for task_dir in entry["task_dirs"]:
            task_dir = task_dir.resolve()
            run_id = str(task_dir)
            mtime = 0.0
            for path in (task_dir / "task.log", task_dir):
                try:
                    mtime = max(mtime, path.stat().st_mtime)
                except OSError:
                    pass
            runs.append(
                {
                    "run_id": run_id,
                    "label": entry["name"],
                    "status": compute_status(task_dir, run_id, alive, queued_set),
                    "stage": None,
                    "mtime": mtime,
                    "external": True,
                    "root": OUTPUT_ROOT.name,
                    "iou": None,
                }
            )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def resolve_file(raw_path: str) -> Optional[Path]:
    path = Path(unquote(raw_path)).resolve()
    if any(is_relative_to(path, root) for root in ALLOWED_ROOTS) or is_scene_file(path):
        return path
    return None


# --------------------------------------------------------------------------- #
# Agent memory (per generator / verifier)                                      #
# --------------------------------------------------------------------------- #
DATA_CONFIG = run_data.RunDataConfig(
    output_dir=REPO_ROOT / "output",
    repo_root=REPO_ROOT,
)


def list_agents(run_id: str) -> list[dict[str, Any]]:
    """List every generator/verifier memory in a run, in pipeline order."""
    task_dir = run_dir_for(run_id)
    stages = task_dir / "stages"
    agents: list[dict[str, Any]] = []
    for mem in stages.glob("*/*/attempt_*/*_memory.json"):
        rel = mem.relative_to(stages).parts  # (idx, Agent, attempt_n, file)
        if len(rel) < 4:
            continue
        stage_index, agent, attempt = rel[0], rel[1], rel[2]
        agents.append(
            {
                "stage_index": stage_index,
                "stage": run_data.stage_from_agent(agent),
                "agent": agent,
                "role": "verifier" if "Verifier" in agent else "generator",
                "attempt": attempt.replace("attempt_", ""),
                "messages": run_data.safe_read_json(mem) and None,  # cheap presence
                "mem": str(mem.resolve()),
            }
        )

    def sort_key(a: dict[str, Any]) -> tuple:
        try:
            idx = int(a["stage_index"])
        except ValueError:
            idx = 0
        stage_rank = (
            run_data.STAGE_ORDER.index(a["stage"])
            if a["stage"] in run_data.STAGE_ORDER
            else 99
        )
        try:
            att = int(a["attempt"])
        except ValueError:
            att = 0
        # Interleave by attempt so each stage reads generator#1 -> verifier#1 ->
        # generator#2 -> ... (the actual round order), not all generators then all
        # verifiers. Generator sorts before its paired verifier within the same attempt.
        return (idx, stage_rank, att, 0 if a["role"] == "generator" else 1)

    agents.sort(key=sort_key)
    for a in agents:
        a.pop("messages", None)
    return agents


def _cache_inline_image(url: str, mem_path: Path) -> str:
    """Spill a base64 `data:` image to disk and return a `file?path=` URL for it.

    The agent's own conversation stores render/reference images inline, so a memory
    arrives with them as `data:image/png;base64,...`. Passing those through made a single
    agent's payload 17.6 MB — of which 17.59 MB was image text and ~2 KB was the reasoning
    anyone opens the panel to read. The browser then had to parse all of it before
    rendering a character. Spilled once (~44 ms for 10 images) and cached; afterwards the
    <img loading="lazy"> tags fetch only what scrolls into view, in parallel and cacheable.
    """
    try:
        head, b64 = url.split(",", 1)
        ext = "png" if "png" in head else ("jpg" if "jpeg" in head or "jpg" in head else "bin")
        raw = base64.b64decode(b64)
    except Exception:  # noqa: BLE001 - a malformed data URL must not break the panel
        return url
    cache = mem_path.parent / "_inline"
    dest = cache / f"{hashlib.sha1(raw).hexdigest()[:16]}.{ext}"
    if not dest.exists():
        try:
            cache.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        except OSError:
            return url  # read-only output tree: fall back to the inline blob
    return "file?path=" + quote(str(dest.resolve()))


def _split_out_thought(text: str) -> Optional[tuple[str, str]]:
    """Split a tool_calls JSON dump into (thought, everything-else), or None.

    `thought` is a REQUIRED parameter of execute_and_evaluate — "think step by step about
    the current scene and reason about what code to write next" — so it is the one
    reasoning field present on every edit (per-round assistant prose is optional and
    absent on ~68% of turns). collect_memory appends the whole tool_calls array as one
    pretty-printed blob, which buried that reasoning inside ~1.9 KB of Blender code.
    """
    try:
        calls = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not (isinstance(calls, list) and calls and isinstance(calls[0], dict)):
        return None
    if "function" not in calls[0]:
        return None
    thoughts = []
    for call in calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            continue
        if isinstance(args, dict) and str(args.get("thought", "")).strip():
            thoughts.append(str(args.pop("thought")).strip())
            fn["arguments"] = json.dumps(args, ensure_ascii=False, indent=2)
    if not thoughts:
        return None
    return "\n\n".join(thoughts), json.dumps(calls, ensure_ascii=False, indent=2)


def load_memory(raw_path: str) -> dict[str, Any]:
    """Parse one agent memory file; render images, mark them as <image>."""
    data = run_data.collect_memory(raw_path, DATA_CONFIG)
    mem_path = Path(unquote(raw_path)).resolve()
    for msg in data.get("messages", []):
        parts = []
        for part in msg.get("parts", []):
            if part.get("type") == "image":
                url = part.get("url", "")
                if url.startswith("data:"):  # inline blob -> disk, fetched lazily
                    url = _cache_inline_image(url, mem_path)
                elif url.startswith("/file?"):  # make relative for the proxy
                    url = "file?" + url[len("/file?") :]
                parts.append({"type": "image", "url": url})
            else:
                text = (
                    part.get("text", "")
                    .replace("[embedded image]", "<image>")
                    .replace("[image]", "<image>")
                )
                # Reasoning first, code after — the panel exists to be read.
                split = _split_out_thought(text)
                if split:
                    thought, rest = split
                    parts.append({"type": "text", "text": thought, "kind": "thought"})
                    parts.append({"type": "text", "text": rest, "kind": "tool_calls"})
                else:
                    parts.append({"type": "text", "text": text})
        msg["parts"] = parts
    return data


# --------------------------------------------------------------------------- #
# HTTP handler                                                                 #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # -- helpers --
    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routing --
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        # Routes are matched with endswith() so the app works whether or not the
        # `/<cluster>/<service>/` proxy prefix is stripped before forwarding.
        route = parsed.path
        if route.endswith("/api/runs"):
            self._json({"runs": list_runs(), "queue": queue_snapshot()})
            return
        # run_id may contain "/" (external runs), so prefer ?id=; keep the path
        # form as a fallback for demo ids.
        if route.endswith("/api/run") or re.search(r"/api/run/[^/]+$", route):
            qid = parse_qs(parsed.query).get("id", [""])[0]
            if not qid:
                m = re.search(r"/api/run/([^/]+)$", route)
                qid = unquote(m.group(1)) if m else ""
            self._json(run_status(qid))
            return
        if route.endswith("/api/agents") or re.search(r"/api/agents/[^/]+$", route):
            qid = parse_qs(parsed.query).get("id", [""])[0]
            if not qid:
                m = re.search(r"/api/agents/([^/]+)$", route)
                qid = unquote(m.group(1)) if m else ""
            self._json({"agents": list_agents(qid)})
            return
        if route.endswith("/api/memory"):
            raw = parse_qs(parsed.query).get("path", [""])[0]
            try:
                self._json(load_memory(raw))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Memory not found")
            return
        if route.endswith("/static/model-viewer.min.js"):
            self._serve_static(
                REPO_ROOT / "site" / "static" / "model-viewer.min.js",
                "text/javascript",
            )
            return
        if route.endswith("/file"):
            q = parse_qs(parsed.query)
            try:
                thumb = int(q.get("thumb", ["0"])[0])
            except ValueError:
                thumb = 0
            self._serve_file(q.get("path", [""])[0], thumb)
            return
        if route.endswith("/healthz"):
            self._json({"ok": True})
            return
        # Index is the fallback for any directory-style path (root or proxied).
        if route in ("", "/") or route.endswith("/") or route.endswith("/index.html"):
            self._html(index_html())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Read-only viewer")

    def _serve_file(self, raw_path: str, thumb: int = 0) -> None:
        path = resolve_file(raw_path)
        if not path:
            self.send_error(HTTPStatus.FORBIDDEN, "Outside allowed roots")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        # Serve a bounded JPEG preview when ?thumb=<edge> is requested for a raster
        # image (the "Input image" pane); fall back to the raw file on any failure.
        if thumb and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            try:
                data = thumb_bytes(path, thumb)
                ctype = "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "max-age=3600")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:  # noqa: BLE001 - fall back to the full file
                pass
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Read failed")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        if path.suffix.lower() == ".blend":
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path: Path, ctype: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=86400")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>
// This page uses relative API URLs (api/runs, file?…) so it works behind the
// /<cluster>/<service>/ proxy — but only with a trailing slash. Without one the
// browser resolves api/runs against /<cluster>/ and every call 404s (sidebar
// shows no runs). Redirect to the trailing-slash form before anything fetches.
if (location.pathname && !location.pathname.endsWith('/'))
  location.replace(location.pathname + '/' + location.search + location.hash);
</script>
<title>Geometric-Rich Agentic Scene Reconstruction -- Image to Blender Scene</title>
<style>
  :root{--ink:#1f2933;--muted:#64717f;--line:#d7dde4;--bg:#f6f8fb;--panel:#fff;
        --accent:#276ef1;--ok:#18865b;--bad:#bf2c2c;--warn:#a26105;--track:#e8edf3;}
  *{box-sizing:border-box;}
  body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       background:var(--bg);color:var(--ink);}
  header{display:flex;align-items:baseline;gap:14px;padding:16px 24px;background:#fff;
         border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;}
  header h1{margin:0;font-size:19px;}
  header h1 .brand{font-weight:800;letter-spacing:.5px;color:var(--accent);margin-right:8px;}
  header .sub{color:var(--muted);font-size:13px;}
  main{display:grid;grid-template-columns:minmax(300px,360px) minmax(0,1fr);gap:0;}
  .side{border-right:1px solid var(--line);background:#fff;padding:18px;min-height:calc(100vh - 58px);}
  .content{padding:20px 24px 48px;display:grid;gap:18px;align-content:start;min-width:0;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;}
  h2{font-size:14px;margin:0 0 12px;color:#344253;}
  /* per-stage wall-clock time shown next to a panel name */
  .ptime{margin-left:auto;color:var(--muted);font:normal 12px ui-monospace,monospace;}
  .meta .total{margin-left:auto;color:var(--muted);font:12px ui-monospace,monospace;}
  /* nested preprocessing sub-panels (scene graph / physics / pseudo-GT) */
  .card.subpanel{margin-top:14px;background:#fbfcfe;box-shadow:none;}
  .card.subpanel>h2.toggle,.card.subpanel>h2{font-size:13px;color:#3b4a5a;}
  .card.collapsible>h2.toggle{cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px;margin:0;}
  .card.collapsible.collapsed>h2.toggle{margin:0;}
  .card.collapsible .caret{flex:0 0 auto;width:0;height:0;border-left:6px solid var(--muted);
    border-top:5px solid transparent;border-bottom:5px solid transparent;transition:transform .15s;}
  .card.collapsible:not(.collapsed) .caret{transform:rotate(90deg);}
  .card.collapsible .cardbody{margin-top:12px;}
  .card.collapsed>.cardbody{display:none;}
  label{font-size:13px;color:var(--muted);display:block;margin-bottom:6px;}
  .drop{border:2px dashed var(--line);border-radius:10px;padding:18px;text-align:center;
        cursor:pointer;color:var(--muted);font-size:13px;background:#fbfcfe;}
  .drop.hover{border-color:var(--accent);background:#f0f6ff;}
  .drop img{max-width:100%;max-height:200px;border-radius:6px;margin-top:6px;}
  select,button{font:inherit;}
  select{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;margin-bottom:12px;background:#fff;}
  button.primary{width:100%;background:var(--accent);color:#fff;border:0;border-radius:8px;
                 padding:11px;font-weight:650;cursor:pointer;}
  button.primary:disabled{opacity:.5;cursor:not-allowed;}
  .runs{display:grid;gap:7px;margin-top:8px;max-height:52vh;overflow-x:hidden;overflow-y:auto;}
  /* Two-line row: name (truncates) on top, status + time below. The status pill sits on
     its own left-aligned line so it is always visible without horizontal scrolling, no
     matter how long the run name is. */
  #run-search{width:100%;box-sizing:border-box;padding:6px 9px;margin-bottom:6px;border:1px solid var(--line);
    border-radius:8px;font-size:13px;}
  #run-tabs{display:flex;gap:6px;margin-bottom:8px;}
  .rvtab{flex:1;padding:4px 8px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;
    font-size:12px;color:var(--muted);}
  .rvtab.sel{border-color:var(--accent);color:var(--accent);font-weight:600;background:#f5f9ff;}
  .run.rgroup .rn{font-family:ui-monospace,monospace;letter-spacing:.02em;}
  .run.rgroup .rn::before{content:"\1F4C5\00a0 ";}  /* calendar glyph before the date group */
  .runhead{padding:6px 8px;margin-bottom:4px;border-bottom:1px solid #2a2a2a;color:var(--muted);font-size:12px;}
  .runhead b{color:#cde;}
  .run .iou{color:#9c9;font:11px ui-monospace,monospace;margin-left:6px;white-space:nowrap;}
  .dim{color:#777;}
  .run-back{padding:6px 8px;margin-bottom:6px;cursor:pointer;color:var(--accent);font-size:12px;font-weight:600;}
  .run-back:hover{text-decoration:underline;}
  .run{border:1px solid var(--line);border-radius:8px;padding:8px 10px;cursor:pointer;font-size:13px;
       display:flex;flex-direction:column;gap:5px;min-width:0;background:#fbfcfe;}
  .run:hover{border-color:#a9c7ff;background:#f5f9ff;}
  .run.sel{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent);}
  .run .rl{display:flex;align-items:center;gap:6px;min-width:0;}
  .run .rn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;font-weight:600;color:var(--ink);}
  .run .rmeta{display:flex;align-items:center;gap:8px;min-width:0;}
  .run .rtime{margin-left:auto;color:var(--muted);font:11px ui-monospace,monospace;white-space:nowrap;}
  .badge{flex:0 0 auto;font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);
         color:var(--muted);text-transform:capitalize;white-space:nowrap;}
  .badge.complete{color:var(--ok);border-color:#a8dec8;background:#effaf5;}
  .badge.active{color:#084a9b;border-color:#a9c7ff;background:#eff5ff;}
  .badge.error{color:var(--bad);border-color:#f0b4b4;background:#fff1f1;}
  .badge.waiting,.badge.idle{color:var(--warn);border-color:#f3d49f;background:#fff8eb;}
  .badge.queued{color:#7a4ea0;border-color:#d7bdee;background:#f7f0ff;}
  .badge.stopped{color:#6b7280;border-color:#d7dde4;background:#f1f3f5;}
  .stopbtn{border:1px solid #f0b4b4;background:#fff1f1;color:#bf2c2c;border-radius:8px;
           padding:3px 11px;font:inherit;font-size:12px;font-weight:650;cursor:pointer;}
  .stopbtn:hover{background:#ffe3e3;border-color:#e89a9a;}
  /* Queued-run callout (shown in a run's detail view) */
  .queuebar{display:flex;align-items:center;gap:10px;background:#fff8eb;border:1px solid #f3d49f;
            border-radius:10px;padding:11px 14px;font-size:13px;color:#7a5b12;line-height:1.45;}
  /* SAM3D GPU panel (sidebar) */
  .qpanel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px 14px;margin-bottom:14px;}
  .qhead{display:flex;align-items:center;gap:8px;margin-bottom:11px;}
  .qhead .dot{width:9px;height:9px;border-radius:50%;background:linear-gradient(135deg,#276ef1,#12a594);
              box-shadow:0 0 0 3px rgba(39,110,241,.12);}
  .qhead .ttl{font-size:13px;font-weight:700;color:var(--ink);}
  .qhead .cnt{margin-left:auto;font-size:11px;font-weight:700;border-radius:999px;padding:2px 9px;
              color:var(--muted);background:#eef2f7;border:1px solid var(--line);}
  .qhead .cnt.on{color:#fff;background:linear-gradient(135deg,#276ef1,#12a594);border-color:transparent;}
  .qgrid{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;}
  .qslot{height:34px;border-radius:8px;display:grid;place-items:center;font-size:13px;font-weight:700;
         border:1px solid var(--line);background:#f6f8fb;color:#aab4c0;transition:all .25s ease;}
  .qslot.busy{background:linear-gradient(135deg,#276ef1,#12a594);color:#fff;border-color:transparent;
              box-shadow:0 3px 9px rgba(39,110,241,.28);}
  .qfoot{margin-top:11px;font-size:12px;display:flex;align-items:center;gap:7px;font-weight:600;}
  .qfoot.idle{color:var(--ok);}
  .qfoot.full,.qfoot.wait{color:var(--warn);}
  .qfoot .pill{background:#fff8eb;border:1px solid #f3d49f;border-radius:999px;padding:1px 9px;
               font-weight:700;color:#a26105;}
  .progress{height:10px;border-radius:999px;background:var(--track);overflow:hidden;margin:6px 0 2px;}
  .progress>div{height:100%;width:0;background:linear-gradient(90deg,#276ef1,#12a594);transition:width .3s;}
  .cmp{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  .cmp.split{display:flex;gap:0;align-items:stretch;}
  .cmp.split>.pane{flex:1 1 0;min-width:150px;overflow:hidden;}
  .cmp.split>.pane:first-child{padding-right:10px;}
  .cmp.split>.pane:last-child{padding-left:10px;}
  .splitter{flex:0 0 8px;align-self:stretch;cursor:col-resize;border-radius:5px;
    background:var(--line);position:relative;transition:background .15s;}
  .splitter:hover,.splitter.drag{background:#7aa2c8;}
  .splitter::after{content:"";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    width:2px;height:28px;border-left:2px dotted #fff;border-right:2px dotted #fff;opacity:.7;}
  .media{border:1px solid var(--line);border-radius:8px;background:#f9fbfd;min-height:240px;
         display:grid;place-items:center;overflow:hidden;}
  .media img{width:100%;height:auto;max-height:480px;object-fit:contain;display:block;}
  .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;}
  #d-pre-inst{max-height:62vh;overflow-y:auto;padding-right:4px;}
  /* Detected-instances group headings + a double-width grid for re-detected pairs so the
     original|edited images each render at the same size as a normal single-image tile. */
  .inst-h{font:600 12px ui-sans-serif,system-ui;color:var(--ink);margin:16px 0 8px;}
  .inst-h:first-child{margin-top:0;}
  .gallery.redet{grid-template-columns:repeat(auto-fill,minmax(312px,1fr));}
  .gallery figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;}
  .gallery img{width:100%;display:block;}
  .gallery figcaption{font-size:12px;color:var(--muted);padding:5px 8px;text-transform:capitalize;overflow-wrap:anywhere;white-space:normal;}
  .reggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;}
  .pgtgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;}
  .pgtcard{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff;}
  .pgttitle{font:12px ui-monospace,monospace;margin-bottom:8px;}
  .pgtimgs{display:flex;align-items:center;gap:6px;}
  .pgtimgs figure{margin:0;text-align:center;flex:0 0 auto;}
  .pgtimgs img,.pgtwait{width:128px;height:96px;object-fit:contain;background:var(--bg);border:1px solid var(--line);border-radius:4px;display:block;}
  .pgtimgs figure:last-child img{border-color:var(--ok);box-shadow:0 0 0 2px color-mix(in srgb,var(--ok) 30%,transparent);}
  .pgtimgs figcaption{font-size:10px;color:var(--muted);margin-top:2px;}
  .pgtwait{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px;}
  .pgtarrow{color:var(--muted);font-size:16px;}
  .tier{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;font-weight:600;}
  .tier.t1{background:#e6f4ea;color:#137333;} .tier.t2{background:#fef7e0;color:#b06000;}
  .tier.t3{background:#fce8e6;color:#a50e0e;} .tier.t4{background:#f1f3f4;color:#5f6368;}
  figure.discarded{opacity:0.85;}
  /* Marker tile shape comes from an inline `aspect-ratio` (the input image's) when known;
     min-height is just the fallback so a marker with unknown aspect isn't a giant square. */
  figure.discarded .discard-mark{display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:4px;min-height:90px;background:repeating-linear-gradient(45deg,#fbecec,#fbecec 8px,#f7e0e0 8px,#f7e0e0 16px);
    border:1px dashed #d99;border-radius:6px;color:#a50e0e;font-size:30px;}
  figure.discarded .discard-mark span{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;}
  .vrow{padding:3px 0;line-height:1.5;}
  .vadd,.vdrop,.vfix{display:inline-block;min-width:128px;padding:1px 7px;border-radius:8px;font-weight:600;font-size:11px;}
  .vadd{background:#e6f4ea;color:#137333;} .vdrop{background:#fce8e6;color:#a50e0e;} .vfix{background:#fef7e0;color:#b06000;}
  .round{border:1px solid var(--line);border-radius:6px;padding:6px 10px;margin:4px 0;background:#fff;}
  .round .ok{color:var(--ok);font-weight:600;} .round .bad{color:var(--bad);font-weight:600;}
  .rootbox{display:flex;gap:14px;align-items:flex-start;border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fafbff;}
  .rootbox img{max-width:260px;border-radius:6px;border:1px solid var(--line);}
  .sgtree{font:12px ui-monospace,monospace;line-height:1.7;}
  .sgnode{padding:1px 0;} .sgkind{display:inline-block;min-width:74px;color:var(--muted);}
  .sgsurf{color:#137333;font-weight:600;} .sgobj{color:var(--ink);}
  .sgform{display:inline-block;margin-left:6px;padding:0 6px;border-radius:8px;font:600 10px ui-sans-serif,system-ui;
    background:#e6f4ea;color:#137333;border:1px solid #b7e1c4;vertical-align:1px;}
  .sgrels{margin-top:14px;border-top:1px solid var(--line);padding-top:10px;}
  .sgrelhdr{font:600 12px ui-sans-serif,system-ui;color:var(--ink);margin-bottom:8px;}
  .sgrel{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;padding:6px 0;border-bottom:1px dashed var(--line);}
  .sgrel:last-child{border-bottom:0;}
  .sgreltype{display:inline-block;min-width:96px;padding:1px 8px;border-radius:8px;font:600 11px ui-monospace,monospace;
    background:#eef2f7;color:#3b4a5a;border:1px solid var(--line);text-align:center;}
  .sgrelpair{font:12px ui-sans-serif,system-ui;color:var(--ink);}
  .sgrelid{color:var(--muted);font:11px ui-monospace,monospace;}
  .sgrelarrow{color:var(--muted);padding:0 2px;}
  .sgrelmean{flex-basis:100%;font:11px ui-sans-serif,system-ui;color:var(--muted);padding-left:104px;}
  .posegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;}
  .posecard{border:1px solid var(--line);border-radius:8px;padding:9px 12px;background:#fff;}
  .posecard .pn{font-size:13px;font-weight:600;margin-bottom:6px;}
  .poserow{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;margin:2px 0;}
  .poserow .k{color:var(--muted);}
  .regcard{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff;}
  .regcard .rid{font-size:13px;margin-bottom:6px;}
  .regbar{display:flex;align-items:center;gap:6px;margin:3px 0;}
  .regbar .l{width:42px;color:var(--muted);font-size:11px;}
  .regbar .t{flex:1;height:10px;background:var(--track);border-radius:5px;overflow:hidden;}
  .regbar .t>div{height:100%;}
  .regbar .v{width:42px;text-align:right;font:11px ui-monospace,monospace;}
  .regstrip{display:flex;align-items:flex-start;gap:8px;margin-top:8px;}
  .regfilm{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;flex:1;}
  .regframe{margin:0;text-align:center;flex:0 0 auto;}
  .regframe img,.regwait{width:92px;height:92px;object-fit:contain;background:var(--bg);border:1px solid var(--line);border-radius:4px;display:block;}
  .regframe.target img{border-color:var(--accent);}
  .svizgrid{display:flex;gap:8px;align-items:center;overflow-x:auto;}
  .svizframe{margin:0;text-align:center;flex:1 1 0;min-width:220px;}
  .svizframe img{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px;display:block;}
  .svizframe figcaption{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.4;}
  .svizarrow{font-size:20px;color:var(--muted);flex:0 0 auto;}
  /* Investigation composites: a responsive wrapping grid (was a single overflowing
     row) so the wide scene|photo crops tile neatly instead of piling up. */
  .reginv{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-bottom:14px;}
  .reginvcard{margin:0;}
  .reginvcard img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:zoom-in;}
  .reginvcard figcaption{font-size:10px;color:var(--muted);margin-top:3px;text-align:center;}
  .regframe.last img{border-color:var(--ok);box-shadow:0 0 0 2px color-mix(in srgb,var(--ok) 30%,transparent);}
  .regframe figcaption{font-size:10px;color:var(--muted);margin-top:2px;line-height:1.2;}
  .regarrow{align-self:center;color:var(--muted);font-size:18px;padding:0 2px;}
  .regwait{display:flex;align-items:center;justify-content:center;}
  .regcard.active{border-color:var(--accent);box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 25%,transparent);}
  .reglive{color:var(--accent);font:12px ui-monospace,monospace;}
  table.regtrace{border-collapse:collapse;font:11px ui-monospace,monospace;width:100%;margin-top:8px;}
  table.regtrace th,table.regtrace td{border:1px solid var(--line);padding:2px 6px;text-align:right;}
  table.regtrace th{color:var(--muted);font-weight:500;}
  .empty{color:var(--muted);font-size:13px;padding:24px;text-align:center;}
  pre{background:#101820;color:#eef4fb;border-radius:8px;padding:12px;max-height:240px;overflow:auto;
      font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;}
  .meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:10px;}
  a.dl{display:inline-block;background:var(--ok);color:#fff;text-decoration:none;padding:9px 14px;
       border-radius:8px;font-weight:650;font-size:13px;}
  .stagegrp{margin-bottom:10px;}
  .stagegrp h3{font-size:12px;color:#526174;margin:0 0 6px;text-transform:capitalize;}
  .agents{display:flex;flex-wrap:wrap;gap:7px;}
  .agent-item{border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;font-size:12px;
              background:#fbfcfe;display:flex;gap:7px;align-items:center;}
  .agent-item:hover{border-color:#a9c7ff;background:#f5f9ff;}
  .agent-item .role{font-size:10px;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--muted);}
  .agent-item .role.generator{color:#084a9b;border-color:#a9c7ff;background:#eff5ff;}
  .agent-item .role.verifier{color:#7a4ea0;border-color:#d7bdee;background:#f7f0ff;}
  .modal{position:fixed;inset:0;background:rgba(15,24,32,.55);display:none;z-index:50;
         padding:28px;overflow:auto;}
  .modal.open{display:block;}
  .modal-box{max-width:900px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;
             box-shadow:0 20px 60px rgba(0,0,0,.3);}
  .modal-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 18px;
              border-bottom:1px solid var(--line);position:sticky;top:0;background:#fff;}
  .modal-head h2{margin:0;font-size:15px;}
  .modal-head button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit;}
  .modal-body{padding:16px 18px;display:grid;gap:12px;}
  .msg{border:1px solid var(--line);border-radius:10px;overflow:hidden;}
  .msg-role{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:6px 12px;
            background:#f3f6fb;color:#526174;border-bottom:1px solid var(--line);}
  .msg.system .msg-role{background:#fff8eb;color:#a26105;}
  .msg.assistant .msg-role{background:#eff5ff;color:#084a9b;}
  .msg.tool .msg-role{background:#eef7f1;color:#18865b;}
  .msg.user .msg-role{background:#f7f0ff;color:#7a4ea0;}
  .msg-body{padding:10px 12px;display:grid;gap:8px;}
  .msg-body pre{margin:0;background:#0f1820;color:#eef4fb;}
  /* the agent's `thought` for this edit — the reason the panel is worth opening, so it
     reads as prose on a light ground rather than as another dark code block */
  .thought{border-left:3px solid #084a9b;background:#f5f9ff;border-radius:0 6px 6px 0;padding:8px 10px;}
  .thought-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
                 color:#084a9b;margin-bottom:4px;}
  .thought pre{background:none;color:#1a2733;white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.5;}
  .calls summary{cursor:pointer;font-size:11px;font-weight:700;text-transform:uppercase;
                 letter-spacing:.04em;color:var(--muted);padding:4px 0;}
  .calls[open] summary{margin-bottom:6px;}
  .msg-imgs{display:flex;flex-wrap:wrap;gap:8px;}
  .msg-imgs figure{margin:0;}
  .msg-imgs img{max-height:150px;max-width:220px;border:1px solid var(--line);border-radius:6px;display:block;}
  .msg-imgs figcaption{font-size:11px;color:var(--muted);text-align:center;margin-top:2px;}
  /* 3D scene viewer */
  .media-title{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;
               color:#344253;font-size:13px;font-weight:700;}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;}
  .seg button{border:0;background:#fff;color:var(--muted);font:inherit;font-size:12px;font-weight:650;
              padding:4px 11px;cursor:pointer;}
  .seg button.on{background:var(--accent);color:#fff;}
  #scenehost{position:relative;min-height:420px;}
  model-viewer{width:100%;height:460px;background:#eef2f7;border-radius:8px;--poster-color:transparent;}
  .scenenote{position:absolute;top:10px;left:10px;background:rgba(16,24,32,.72);color:#fff;font-size:11px;
             font-weight:650;padding:4px 9px;border-radius:999px;display:none;align-items:center;gap:6px;}
  .scenenote.show{display:inline-flex;}
  .scenenote .spin{width:9px;height:9px;border-radius:50%;border:2px solid #ffffff55;border-top-color:#fff;
                   animation:spin .8s linear infinite;}
  @keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
<header>
  <h1><span class="brand">SceneRig</span>Geometric-Rich Agentic Scene Reconstruction</h1>
  <span class="sub">Single image &rarr; Blender scene (staged inverse graphics)</span>
</header>
<main>
  <aside class="side">
    <!--LAUNCH_PANEL_START-->
    <div class="card" style="margin-bottom:16px;">
      <h2>New reconstruction</h2>
      <div id="drop" class="drop">Click or drop an image here</div>
      <input id="file" type="file" accept="image/*" hidden>
      <div style="height:12px;"></div>
      <details style="margin-bottom:12px;">
        <summary style="cursor:pointer;font-size:12px;color:var(--muted);">Advanced options</summary>
        <label style="display:block;font-size:12px;margin-top:8px;"><input id="opt-reseg" type="checkbox"> Occlusion recovery (resegment, ~+5 min)</label>
        <label style="display:block;font-size:12px;margin-top:6px;"><input id="opt-room" type="checkbox"> Room mode (VLM routes room vs tabletop)</label>
        <label style="display:block;font-size:12px;margin-top:6px;">Same-size manual allow-list (optional; 2+ objects)
          <input id="opt-samesize" type="text" placeholder="e.g. banana, block" style="width:100%;margin-top:2px;">
          <span style="color:var(--muted)">Leave empty for automatic duplicate discovery (3+ manufactured objects).</span>
        </label>
      </details>
      <button id="go" class="primary" disabled>Reconstruct scene</button>
      <div id="msg" style="font-size:12px;color:var(--muted);margin-top:8px;"></div>
    </div>
    <div id="queuebar"></div>
    <!--LAUNCH_PANEL_END-->
    <div class="card">
      <h2>Runs</h2>
      <input id="run-search" type="search" placeholder="Search experiments&hellip;" autocomplete="off">
      <div id="run-tabs">
        <button type="button" class="rvtab sel" data-view="date">By date</button>
        <button type="button" class="rvtab" data-view="all">All</button>
      </div>
      <div id="runs" class="runs"><div class="empty">Loading runs&hellip;</div></div>
    </div>
  </aside>
  <section id="content" class="content">
    <div class="card"><div class="empty">Select a run on the left.</div></div>
  </section>
</main>
<div id="modal" class="modal">
  <div class="modal-box">
    <div class="modal-head"><h2 id="modalTitle">Agent memory</h2><button id="modalClose">Close</button></div>
    <div id="modalBody" class="modal-body"></div>
  </div>
</div>
<script type="module" src="static/model-viewer.min.js"></script>
<script>
const $=id=>document.getElementById(id);
window.__VIEW_ONLY__ = false;  // flipped to true server-side when serving the view-only page
let selected=null, pendingB64=null, pendingName=null, pollTimer=null;
// Runs sidebar view state: `date` (3-level drill-down split('_')[0] -> [1] -> runs, default)
// or `all` (flat). `runRoot`/`runL1`/`runL2` are the drilled-into keys (output dir, then the
// per-root levels -- see runK0/runKEval/runK1/runK2); `runSearch` is a name filter.
let allRuns=[], runView='date', runRoot=null, runL1=null, runL2=null, runSearch='';
let builtRun=null, curGlb=null, viewMode='3d';

function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}

// ---- upload (launch panel; absent on the view-only page, so guard on its nodes) ----
if($('drop')&&$('go')){
  const drop=$('drop'), fileInput=$('file');
  drop.onclick=()=>fileInput.click();
  drop.ondragover=e=>{e.preventDefault();drop.classList.add('hover');};
  drop.ondragleave=()=>drop.classList.remove('hover');
  drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hover');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0]);};
  fileInput.onchange=()=>{if(fileInput.files[0])loadFile(fileInput.files[0]);};
  function loadFile(f){
    pendingName=f.name;
    const r=new FileReader();
    r.onload=()=>{pendingB64=r.result;drop.innerHTML='<img src="'+r.result+'"><div>'+esc(f.name)+'</div>';$('go').disabled=false;};
    r.readAsDataURL(f);
  }
  $('go').onclick=async()=>{
    if(!pendingB64)return;
    $('go').disabled=true;$('msg').textContent='Launching pipeline...';
    try{
      const resp=await fetch('api/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({image_b64:pendingB64,filename:pendingName,
          options:{generative_resegment:$('opt-reseg').checked,room_mode:$('opt-room').checked,
                   same_size_categories:$('opt-samesize').value}})});
      const d=await resp.json();
      if(d.error){$('msg').textContent='Error: '+d.error;$('go').disabled=false;return;}
      $('msg').textContent='Started run '+d.run_id;
      selected=d.run_id;
      await refreshRuns();
      poll();
    }catch(e){$('msg').textContent='Error: '+e;$('go').disabled=false;}
  };
}

// ---- queue banner ----
function renderQueue(q){
  const bar=$('queuebar'); if(!bar)return;
  if(!q){bar.innerHTML='';return;}
  const cap=q.capacity||6, active=q.active||0, waiting=q.queued||0;
  const gpus=q.gpus||[...Array(cap).keys()];
  let slots='';
  for(let i=0;i<cap;i++){
    const busy=i<active, g=gpus[i];
    slots+=`<div class="qslot ${busy?'busy':''}" title="GPU ${g} — ${busy?'busy':'free'}">${g}</div>`;
  }
  let foot;
  if(waiting>0) foot=`<div class="qfoot wait"><span class="pill">${waiting}</span> ${waiting===1?'run':'runs'} waiting in queue</div>`;
  else if(active>=cap) foot=`<div class="qfoot full">All GPUs busy &middot; next request will queue</div>`;
  else foot=`<div class="qfoot idle">${cap-active} GPU${cap-active===1?'':'s'} free &middot; ready</div>`;
  bar.innerHTML=`
    <div class="qpanel">
      <div class="qhead">
        <span class="dot"></span>
        <span class="ttl">SAM3D GPUs</span>
        <span class="cnt ${active>0?'on':''}">${active}/${cap} busy</span>
      </div>
      <div class="qgrid">${slots}</div>
      ${foot}
    </div>`;
}

// ---- runs list ----
// Compact "time since" from an epoch-seconds mtime ('' if missing).
function agoTime(mtime){
  if(!mtime)return '';
  const s=Math.floor(Date.now()/1000-mtime);
  if(s<45)return 'just now';
  const m=Math.round(s/60); if(m<60)return m+'m ago';
  const h=Math.round(m/60); if(h<24)return h+'h ago';
  return Math.round(h/24)+'d ago';
}
async function refreshRuns(){
  try{
    const d=await(await fetch('api/runs')).json();
    renderQueue(d.queue);
    allRuns=d.runs||[];
    renderRuns();
  }catch(e){}
}
// Drill-down: L0 = the output dir the run lives in ("default" = output/static_scene,
// "benchmark" = static_scene_benchmark, "0721_0731" = static_scene_0721_0731, ...).
//   non-eval roots: two more levels from the run name, as before -- L1 = split('_')[0]
//     (experiment date by convention), L2 = split('_')[1] (experiment group), then the runs.
//   curated roots (eval / benchmark, and any benchmark_<date> e.g. static_scene_benchmark_0807):
//     ONE more level, the curated group (same six as grase-viewer-simple-eval), then the runs.
//     eval/benchmark[_0806] names are flat <group>_<scene> (eval flattened 08-03, eval_0806 08-06);
//     A defensive runKEval strip still accepts an old <date>_benchmark_ prefix. Without it, such
//     a name would drill L1=split('_')[0] / L2=split('_')[1] and put every scene in its own L2 group.
// Demo runs (web-form uploads, external:false) live in the live root, so they bucket under
// "default" and keep their own L1 "demo" / single L2 "misc" so they don't fragment by id.
function runK0(r){ return String(r.root||'default'); }
function runKEval(r){
  // Single-group key for EVERY non-default root: the run name's "<task>" = split('_')[0].
  // Names are flat "<task>_<scene>" (benchmark_0807 was flattened on 08-07, its
  // "<date>_benchmark_" prefix trimmed in the filesystem). The strip stays as a defensive
  // guard in case a future dated benchmark ships with that prefix.
  const n=String(r.label||r.run_id||'').replace(/^\d{4}_benchmark_/, '');
  if(n.startsWith('airoa_moma_')) return 'moma';  // else split('_')[0] reads as "airoa"
  return n.split('_')[0]||'misc';
}
function runK1(r){ if(!r.external) return 'demo'; return String(r.label||r.run_id||'').split('_')[0]||'misc'; }
function runK2(r){ if(!r.external) return 'misc'; return String(r.label||r.run_id||'').split('_')[1]||'misc'; }
// Mean registration IoU over a set of runs (scenes without one are simply not counted).
function meanIoU(items){
  const v=items.map(r=>r.iou).filter(x=>typeof x==='number');
  return v.length ? {mean:v.reduce((a,b)=>a+b,0)/v.length, n:v.length} : null;
}
function iouChip(x){
  return typeof x==='number' ? ` <span class="iou">IoU ${x.toFixed(3)}</span>` : '';
}
// "12 scenes &middot; mean IoU 0.778 (11)" — the count in parens appears only when some
// scene in the set has no IoU, so a partial average can't be misread as covering everything.
function iouSummary(items){
  const m=meanIoU(items);
  if(!m) return '';
  const part = m.n<items.length ? ` <span class="dim">(${m.n}/${items.length})</span>` : '';
  return ` &middot; mean IoU <b>${m.mean.toFixed(3)}</b>${part}`;
}
function runRowHTML(r){
  return `<div class="run ${r.run_id===selected?'sel':''}" data-id="${esc(r.run_id)}">
      <div class="rl">${r.external?'<span class="role">ext</span>':''}<span class="rn">${esc(r.label||r.run_id)}</span></div>
      <div class="rmeta"><span class="badge ${esc(r.status)}">${esc(r.status)}</span>${iouChip(r.iou)}<span class="rtime">${esc(agoTime(r.mtime))}</span></div>
    </div>`;
}
function wireRunRows(){
  $('runs').querySelectorAll('.run[data-id]').forEach(el=>el.onclick=()=>{selected=el.dataset.id;poll();renderRuns();});
}
// A clickable list of groups (key -> runs), newest-group-first, with count + recency.
function groupListHTML(items, keyfn){
  const groups={};
  items.forEach(r=>{const k=keyfn(r);(groups[k]=groups[k]||[]).push(r);});
  const keys=Object.keys(groups).sort((a,b)=>
    Math.max(...groups[b].map(r=>r.mtime||0))-Math.max(...groups[a].map(r=>r.mtime||0)));
  return keys.map(k=>{
    const g=groups[k], newest=Math.max(...g.map(r=>r.mtime||0));
    return `<div class="run rgroup" data-key="${esc(k)}">
      <div class="rl"><span class="rn">${esc(k)}</span></div>
      <div class="rmeta"><span class="rtime">${g.length} run${g.length>1?'s':''}${iouSummary(g)} &middot; ${esc(agoTime(newest))}</span></div>
    </div>`;
  }).join('');
}
// Header over the current listing: how many scenes it covers and their mean IoU. Scoped to
// what is on screen, so it reads "all scenes" at the top level and narrows as you drill in.
function runsHeadHTML(items, scope){
  return `<div class="runhead">${esc(scope)} &middot; ${items.length} scene${items.length===1?'':'s'}</div>`;
}
function renderRuns(){
  const box=$('runs'); const q=runSearch.trim().toLowerCase();
  if(!allRuns.length){box.innerHTML='<div class="empty">No runs yet.</div>';return;}
  // Search: global flat filter over every run (ignores the drill-down).
  if(q){
    const hits=allRuns.filter(r=>String(r.label||r.run_id||'').toLowerCase().includes(q));
    box.innerHTML=hits.length?runsHeadHTML(hits,`search "${q}"`)+hits.map(runRowHTML).join(''):'<div class="empty">No matches.</div>';
    wireRunRows(); return;
  }
  if(runView==='all'){ box.innerHTML=runsHeadHTML(allRuns,'all scenes')+allRuns.map(runRowHTML).join(''); wireRunRows(); return; }
  // Level 0: which output dir.
  if(runRoot===null){
    box.innerHTML=runsHeadHTML(allRuns,'all roots')+groupListHTML(allRuns, runK0);
    box.querySelectorAll('.rgroup').forEach(el=>el.onclick=()=>{runRoot=el.dataset.key;runL1=null;runL2=null;renderRuns();});
    return;
  }
  const inRoot=allRuns.filter(r=>runK0(r)===runRoot);
  const backHTML=(txt,n)=>`<div class="run-back" id="run-back">&larr; ${esc(txt)} (${n})</div>`;
  const wireBack=fn=>{$('run-back').onclick=fn;};
  // Every non-default root: ONE grouping level (runKEval = split('_')[0] "task"), then the
  // runs -- same shape benchmark/eval used, now applied to all archives. Only "default"
  // (the live static_scene root) keeps the date -> group two-level drill below.
  if(runRoot!=='default'){
    if(runL1===null){
      box.innerHTML=backHTML(runRoot,inRoot.length)+runsHeadHTML(inRoot,runRoot)+groupListHTML(inRoot, runKEval);
      wireBack(()=>{runRoot=null;renderRuns();});
      box.querySelectorAll('.rgroup').forEach(el=>el.onclick=()=>{runL1=el.dataset.key;renderRuns();});
      return;
    }
    const ge=inRoot.filter(r=>runKEval(r)===runL1);
    box.innerHTML=backHTML(`${runRoot} / ${runL1}`,ge.length)+runsHeadHTML(ge,`${runRoot} / ${runL1}`)+ge.map(runRowHTML).join('');
    wireBack(()=>{runL1=null;renderRuns();});
    wireRunRows();
    return;
  }
  // non-eval: Level 1 = split('_')[0] groups within the chosen root.
  if(runL1===null){
    box.innerHTML=backHTML(runRoot,inRoot.length)+runsHeadHTML(inRoot,runRoot)+groupListHTML(inRoot, runK1);
    wireBack(()=>{runRoot=null;renderRuns();});
    box.querySelectorAll('.rgroup').forEach(el=>el.onclick=()=>{runL1=el.dataset.key;runL2=null;renderRuns();});
    return;
  }
  const inL1=inRoot.filter(r=>runK1(r)===runL1);
  // Level 2: split('_')[1] groups within the chosen L1.
  if(runL2===null){
    box.innerHTML=backHTML(`${runRoot} / ${runL1}`,inL1.length)+runsHeadHTML(inL1,`${runRoot} / ${runL1}`)+groupListHTML(inL1, runK2);
    wireBack(()=>{runL1=null;renderRuns();});
    box.querySelectorAll('.rgroup').forEach(el=>el.onclick=()=>{runL2=el.dataset.key;renderRuns();});
    return;
  }
  // Level 3: the runs within the chosen root / L1 / L2.
  const g=inL1.filter(r=>runK2(r)===runL2);
  box.innerHTML=backHTML(`${runRoot} / ${runL1} / ${runL2}`,g.length)+runsHeadHTML(g,`${runRoot} / ${runL1} / ${runL2}`)+g.map(runRowHTML).join('');
  wireBack(()=>{runL2=null;renderRuns();});
  wireRunRows();
}

// ---- detail ----
function pct(p){return Math.max(0,Math.min(100,Number(p||0)));}
function media(url,alt){return url?`<img src="${esc(url)}" alt="${esc(alt)}">`:'<div class="empty">No render yet.</div>';}
// seconds -> compact "Hh Mm Ss" (drops leading zero units). '' for null/undefined.
function fmtTime(s){if(s==null||isNaN(s))return '';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
  if(h)return `${h}h ${m}m`; if(m)return `${m}m ${sec}s`; return `${sec}s`;}
function stageTime(t,k){return (t&&t[k]!=null)?fmtTime(t[k]):'';}

// Build the static skeleton once per run (stable ids); poll() only updates
// fields in place so the <model-viewer> (and the user's camera) survive polls.
function buildSkeleton(){
  $('content').innerHTML=`
    <div class="card">
      <div class="meta" id="d-meta"></div>
      <div id="d-queued" style="display:none;"></div>
      <div id="d-prog">
        <div class="progress"><div id="d-bar" style="width:0%"></div></div>
        <div id="d-proglabel" style="font-size:12px;color:var(--muted);"></div>
      </div>
      <div id="d-blend" style="margin-top:14px;"></div>
    </div>
    <div class="card">
      <div class="cmp split" id="cmp-main">
        <div class="pane" id="pane-left"><h2>Input image</h2><div class="media"><img id="d-target" alt="target"></div></div>
        <div class="splitter" id="cmp-splitter" title="Drag to resize"></div>
        <div class="pane" id="pane-right">
          <div class="media-title"><span>Reconstructed scene</span>
            <span class="seg"><button id="seg3d">3D scene</button><button id="segimg">Render</button></span>
          </div>
          <div class="media" id="scenehost">
            <model-viewer id="viewer3d" camera-controls
              interaction-prompt="none" shadow-intensity="0.6" exposure="1.0"
              min-camera-orbit="auto auto 0.005m" max-camera-orbit="auto auto 100m"
              min-field-of-view="2deg" max-field-of-view="55deg"
              environment-image="neutral" tone-mapping="neutral"></model-viewer>
            <img id="viewerImg" style="display:none;width:100%;max-height:460px;object-fit:contain;">
            <div id="scenenote" class="scenenote"><span class="spin"></span><span id="scenenote-t">Building 3D scene…</span></div>
          </div>
        </div>
      </div>
    </div>
    <div class="card" id="d-pre-card" style="display:none;">
      <h2>Preprocessing<span class="ptime" id="d-pre-time"></span></h2>
      <div id="d-pre-summary" style="font-size:12px;color:var(--muted);margin-bottom:12px;"></div>
      <div class="cmp">
        <div><div class="media-title"><span>depth</span></div><div class="media"><img id="d-pre-depth" alt="depth"></div></div>
        <div><div class="media-title"><span>Per-object masks</span></div><div class="media"><img id="d-pre-masks" alt="masks composite"></div></div>
      </div>
      <div id="d-pre-mode" style="font-size:12px;margin:16px 0 0;"></div>
      <h3 style="margin:8px 0 8px;font-size:13px;color:var(--ink);">Canonical root</h3>
      <div id="d-pre-root" style="font-size:12px;"></div>
      <h3 style="margin:16px 0 8px;font-size:13px;color:var(--ink);">Proposer verifier &mdash; 1-pass audit</h3>
      <div id="d-pre-verify" style="font-size:12px;"></div>
      <!-- Preprocessing sub-panels (collapsed by default; images lazy-load on expand) -->
      <div class="card subpanel" id="d-pre-inst-card">
        <h2>Detected instances (metric depth)</h2>
        <div id="d-pre-inst"></div>
      </div>
      <div class="card subpanel" id="d-pre-merge-card">
        <h2>Merge process &mdash; over-segmentation VLM pair decisions</h2>
        <div id="d-pre-merge" style="font-size:12px;"></div>
      </div>
      <div class="card subpanel" id="d-pre-occl-card">
        <h2>Occlusion &mdash; pairwise VLM edges, vital parts, redetects</h2>
        <div id="d-pre-occl" style="font-size:12px;"></div>
      </div>
      <div class="card subpanel" id="d-sg-card" style="display:none;">
        <h2>Scene graph &mdash; support hierarchy</h2>
        <div id="d-sg-summary" style="font-size:12px;color:var(--muted);margin-bottom:10px;"></div>
        <div id="d-sg-tree" class="sgtree"></div>
        <div id="d-sg-rels" class="sgrels" style="display:none;"></div>
      </div>
      <div class="card subpanel" id="d-pm-card" style="display:none;">
        <h2>Point-cloud pose match &mdash; per-object ICP before/after</h2>
        <div id="d-pm" style="font-size:12px;"></div>
      </div>
      <div class="card subpanel" id="d-pose-card" style="display:none;">
        <h2>Physics drop test &mdash; settle tilt &amp; stability</h2>
        <div id="d-pose-summary" style="font-size:12px;color:var(--muted);margin-bottom:12px;"></div>
        <div class="posegrid" id="d-pose-box"></div>
      </div>
      <div class="card subpanel" id="d-pgt-card" style="display:none;">
        <h2>Pseudo-GT novel views &mdash; SHARP render + GPT-Image-2</h2>
        <div id="d-pgt-summary" style="font-size:12px;color:var(--muted);margin-bottom:12px;"></div>
        <div class="pgtgrid" id="d-pgt-box"></div>
      </div>
    </div>
    <div class="card" id="d-sviz-card" style="display:none;">
      <h2>Object poses &mdash; SAM3D placement &rarr; aligned &amp; physically settled</h2>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px;">The object set alone
        (no background surfaces) through the reference camera at the preprocess
        milestones: raw SAM3D placement, then after photo alignment + physics settling.</div>
      <div class="svizgrid" id="d-sviz-box"></div>
    </div>
    <div class="card" id="d-reg-card" style="display:none;">
      <h2>Composition &mdash; pose refinement (investigate / move)<span class="ptime" id="d-reg-time"></span></h2>
      <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">The composition agent
        investigates objects (rendered scene vs. reference photo crops) and requests moves;
        the backend applies each move, physics-settles the object's support group, and keeps
        it only if it stays stable and improves IoU. The per-object trace below shows each
        round's IoU, gain, and settle outcome (tilt / lift / kept).</div>
      <div id="d-reg-summary" style="font-size:12px;color:var(--muted);margin-bottom:12px;"></div>
      <div class="reginv" id="d-reg-inv"></div>
      <div class="reggrid" id="d-reg-box"></div>
    </div>
    <div class="card" id="d-gallery-card" style="display:none;"><h2>Stage progression</h2><div class="gallery" id="d-gallery"></div></div>
    <div class="card" id="d-time-card"><h2>Timing</h2>
      <div id="d-timing" style="font-size:12px;"><div class="empty">No timings yet.</div></div>
    </div>
    <div class="card" id="d-agents-card"><h2>Agent memory (generator / verifier)</h2>
      <div style="font-size:12px;color:var(--muted);margin-bottom:10px;">
        Click an agent to inspect the exact conversation the model saw &mdash; prompts, tool calls,
        results, and image inputs (shown as <code>&lt;image&gt;</code> thumbnails).</div>
      <div id="agentbox"><div class="empty">No agent memory yet.</div></div>
    </div>
    <div class="card" id="d-log-card"><h2>Log</h2><pre id="d-log"></pre></div>`;
  setupCollapsibles();
  $('seg3d').onclick=()=>setView('3d');
  $('segimg').onclick=()=>setView('render');
  setupSplitter();
}

let splitPct=localStorage.getItem('cmpSplit')||'50';
let splitDrag=false;
function applySplit(){const L=$('pane-left'); if(L) L.style.flex='0 0 '+splitPct+'%';}
function setupSplitter(){
  const sp=$('cmp-splitter'); if(!sp)return;
  applySplit();
  sp.onmousedown=sp.ontouchstart=e=>{splitDrag=true;sp.classList.add('drag');
    document.body.style.userSelect='none';e.preventDefault();};
  sp.ondblclick=()=>{splitPct='50';applySplit();localStorage.setItem('cmpSplit',splitPct);};
}
// window listeners bound once (survive per-run skeleton rebuilds)
function _splitMove(e){
  if(!splitDrag)return;
  const cmp=$('cmp-main'); if(!cmp)return;
  const r=cmp.getBoundingClientRect();
  const x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;
  splitPct=Math.max(15,Math.min(85,x/r.width*100)).toFixed(1);
  applySplit();
}
function _splitUp(){
  if(!splitDrag)return;
  splitDrag=false; const sp=$('cmp-splitter'); if(sp)sp.classList.remove('drag');
  document.body.style.userSelect=''; localStorage.setItem('cmpSplit',splitPct);
}
window.addEventListener('mousemove',_splitMove);
window.addEventListener('touchmove',_splitMove,{passive:false});
window.addEventListener('mouseup',_splitUp);
window.addEventListener('touchend',_splitUp);

function setView(m){
  viewMode=m;
  $('seg3d').classList.toggle('on',m==='3d');
  $('segimg').classList.toggle('on',m==='render');
  $('viewer3d').style.display=(m==='3d')?'block':'none';
  $('viewerImg').style.display=(m==='render')?'block':'none';
}

function updateScene(d){
  const scene=d.scene||{};
  const renderUrl=d.final_render_url||d.latest_render_url;
  if(renderUrl){const im=$('viewerImg'); if(im.getAttribute('src')!==renderUrl) im.src=renderUrl;}
  // swap the GLB only when it actually changes (preserves camera).
  if(scene.glb_url && scene.glb_url!==curGlb){ $('viewer3d').src=scene.glb_url; curGlb=scene.glb_url; }
  const note=$('scenenote');
  if(viewMode==='3d' && !curGlb){
    note.classList.add('show'); $('scenenote-t').textContent= scene.building?'Building 3D scene…':'No scene yet';
  } else if(viewMode==='3d' && scene.building){
    note.classList.add('show'); $('scenenote-t').textContent='Updating 3D scene…';
  } else { note.classList.remove('show'); }
}

function updatePreprocess(pp){
  const card=$('d-pre-card'); if(!card) return;
  if(!pp){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-pre-card')) return;   // lazy: populate only when expanded
  const cam=pp.camera||{};
  const bits=[`<b>${pp.placement_count||0}</b> objects placed`];
  if(cam.lens) bits.push(`camera <b>${esc(cam.lens)}mm</b> (${esc(cam.fov_x)}&deg;&times;${esc(cam.fov_y)}&deg;, ${esc(cam.width)}&times;${esc(cam.height)})`);
  if(pp.surfaces_skipped&&pp.surfaces_skipped.length) bits.push(`surfaces as primitives: ${pp.surfaces_skipped.map(esc).join(', ')}`);
  if(pp.merge_log&&pp.merge_log.length) bits.push(`merges: ${pp.merge_log.length}`);
  $('d-pre-summary').innerHTML=bits.join(' &nbsp;&middot;&nbsp; ');
  if(pp.moge_depth_url){const e=$('d-pre-depth'); if(e.getAttribute('src')!==pp.moge_depth_url)e.src=pp.moge_depth_url;}
  if(pp.masks_composite_url){const e=$('d-pre-masks'); if(e.getAttribute('src')!==pp.masks_composite_url)e.src=pp.masks_composite_url;}
  // Proposer verifier: the 1-pass audit's feedback (added surfaces / dropped objects / fixed rels)
  const vEl=$('d-pre-verify'); if(vEl){
    const rv=pp.proposal_review;
    const relStr=rs=>(rs&&rs.length)?rs.map(r=>`${esc(r.type)}(${esc(r.a)},${esc(r.b)})`).join(', '):'none';
    if(!rv){ vEl.innerHTML=`<span style="color:var(--muted)">verifier off (no review recorded)</span>`; }
    else if(rv.ran===false){ vEl.innerHTML=`<span class="bad">verifier did not run</span>${rv.error?` <span style="color:var(--muted)">(${esc(rv.error)})</span>`:''}`; }
    else {
      const add=rv.add_surfaces||[], drop=rv.drop||[], items=[];
      add.forEach(s=>items.push(`<div class="vrow"><span class="vadd">+ added surface</span> <b>${esc(s.category||s)}</b>${s.description?` <span style="color:var(--muted)">${esc(s.description)}</span>`:''}</div>`));
      drop.forEach(d=>items.push(`<div class="vrow"><span class="vdrop">&minus; dropped object</span> <b>${esc(d)}</b> <span style="color:var(--muted)">(off the main support)</span></div>`));
      if(rv.changed_rels){
        const diff=(rv.rels_before!==undefined&&rv.rels_after!==undefined)
          ? `<span style="color:var(--muted)">${relStr(rv.rels_before)}</span> &rarr; <b>${relStr(rv.rels_after)}</b>`
          : `<b>corrected</b>`;
        items.push(`<div class="vrow"><span class="vfix">~ fixed relationships</span> ${diff}</div>`);
      }
      vEl.innerHTML=items.length
        ? `<div style="color:var(--muted);margin-bottom:6px;">audited the proposal &middot; ${items.length} correction(s)${rv.hints?` &middot; <span title="${esc(rv.hints)}">geometry hint given</span>`:''}</div>`+items.join('')
        : `<span class="ok">&check; audited &mdash; no corrections needed</span>${rv.hints?` <span style="color:var(--muted)">(geometry hint was given)</span>`:''}`;
    }
  }
  // Scene routing. The router went 3-WAY on 2026-07-31: scene_kind is the RAW verdict
  // (room | closeup:table | closeup:tabletop) and routing.form is the EFFECTIVE flag
  // every downstream consumer reads (preprocess.scene_form); routing.mode is only the
  // 2-way track name. The old code compared scene_kind against mode, so once the router
  // went 3-way EVERY closeup scene tripped the mismatch branch and claimed an override
  // ("router saw closeup:tabletop, but the room track is off") -- 124/124 runs across
  // 0801-0802. The verdict is only ever overridden two ways: a ROOM verdict demoted
  // because the room track is off, or the work-surface coverage revert.
  const rtg=pp.routing||{}, modeEl=$('d-pre-mode');
  const kind=pp.scene_kind||'';
  // legacy runs (pre-routing.form) carry only the 2-way scene_kind
  const form=rtg.form||(rtg.mode==='room'?'room':kind);
  const rev=pp.work_surface_reverted;
  const FORM_HELP={
    'room':'floor is the ground; the work surface is an ordinary object standing on it',
    'closeup:table':'the work surface is the stage and something holds it up — the build must include a base/legs (rules gate requires >15cm below the top)',
    'closeup:tabletop':'nothing visible beneath the work surface — a bare slab is the correct build',
  };
  if(modeEl){
    if(form){
      const isRoom=form==='room';
      const pill=`<span class="tier" title="${esc(FORM_HELP[form]||'')}" style="background:${isRoom?'#e6f4ea':'#eef2f7'};color:${isRoom?'#137333':'#3b4a5a'};font-size:11px;">${esc(form)}</span>`;
      let note='';
      if(rev){
        const cov=Math.round((rev.coverage||0)*100), th=Math.round((rev.threshold||0.3)*100);
        note=` <span style="color:var(--muted)">&mdash; router saw &ldquo;${esc(rev.router_scene_kind||kind)}&rdquo;; work surface ${cov}% covered (&ge; ${th}%) &rarr; reverted to closeup</span>`;
      } else if(kind==='room'&&!isRoom){
        note=` <span style="color:var(--muted)">&mdash; router saw &ldquo;room&rdquo;, demoted because the room track is off</span>`;
      }
      modeEl.innerHTML=`<b>Scene mode:</b> ${pill}${note}`;
    } else modeEl.innerHTML='';
  }
  // VLM-chosen canonical root surface (pinned to z=0) -- plain text, like the proposer verifier
  const rt=pp.root, rootEl=$('d-pre-root');
  if(rt&&rt.root){
    rootEl.innerHTML=`${esc(rt.surface||'?')} <span style="color:var(--muted)">(${esc(rt.root)})</span>`
      +`${rt.z_root!=null?` &middot; plane z<sub>root</sub> ${esc(rt.z_root)}m &rarr; 0`:''}`
      +` <span style="color:var(--muted)">&mdash; VLM ground reference, pinned to z=0</span>`;
  } else rootEl.innerHTML=`<span style="color:var(--muted)">no root recorded</span>`;
  // Merge process: every pair the over-segmentation VLM judged + the zoomed close-up it saw.
  const mdec=pp.merge_decisions||[], mEl=$('d-pre-merge');
  if(!mdec.length){ mEl.innerHTML=`<span style="color:var(--muted)">no pairs evaluated</span>`; }
  else {
    mEl.innerHTML=mdec.map(d=>{
      const yes=d.merge;
      const verdict=yes?`<span class="ok">&check; MERGED</span>`:`<span class="bad">kept separate</span>`;
      const keep=(yes&&d.keep)?` <span style="color:var(--muted)">(kept name: ${esc(d.keep)})</span>`:'';
      const imgs=[d.clean_url?`<figure><img src="${esc(d.clean_url)}"><figcaption>close-up</figcaption></figure>`:'',
                  d.marked_url?`<figure><img src="${esc(d.marked_url)}"><figcaption>A red / B blue</figcaption></figure>`:''].join('');
      return `<div style="margin-bottom:12px;border-left:3px solid ${yes?'#137333':'#999'};padding-left:8px;">`
        +`<div><b>${esc(d.a)}</b> <span style="color:#c00">(A)</span> &harr; <b>${esc(d.b)}</b> <span style="color:#06c">(B)</span> &mdash; ${verdict}${keep}</div>`
        +`<div class="gallery">${imgs}</div></div>`;
    }).join('');
  }
  // Occlusion process: pairwise edges (with the crops the VLM saw), hierarchy drops,
  // vital-part verdicts, and the redetect outcome.
  const occ=pp.occlusion, oEl=$('d-pre-occl');
  if(oEl){
    if(!occ||(!occ.pairs.length&&!occ.vitals.length)){
      oEl.innerHTML=`<span style="color:var(--muted)">no touching pairs / vital parts detected</span>`;
    } else if(!collapsed('d-pre-occl-card')) {
      const redet=new Set(occ.redetected||[]);
      const sev=occ.severities||{};
      const pairRows=occ.pairs.map(e=>{
        const hit=!!e.occluder;
        const verdict=hit?`<span class="bad">${esc(e.occluder)} covers ${esc(e.occluded)}</span>`
                         :`<span class="ok">adjacent only</span>`;
        const sv=sev[e.occluded]||{};
        const pct=(sv.hidden_pct!=null)?` ~${sv.hidden_pct}% hidden`:'';
        const out=hit?(redet.has(e.occluded)?` <span class="tier t3">REDETECTED${esc(pct)}</span>`
                      :(sv.severity==='slight'?` <span class="tier" style="background:#777;color:#fff;">slight${esc(pct)} &mdash; skipped</span>`
                                              :` <span class="tier" style="background:#b26a00;color:#fff;">${sv.severity?esc(sv.severity)+esc(pct)+' &mdash; ':''}recovery fell back</span>`)):'';
        const imgs=[e.clean_url?`<figure><img src="${esc(e.clean_url)}"><figcaption>close-up</figcaption></figure>`:'',
                    e.marked_url?`<figure><img src="${esc(e.marked_url)}"><figcaption>A red / B blue</figcaption></figure>`:''].join('');
        return `<div style="margin-bottom:12px;border-left:3px solid ${hit?'#b26a00':'#999'};padding-left:8px;">`
          +`<div><b>${esc(e.a)}</b> <span style="color:#c00">(A)</span> &harr; <b>${esc(e.b)}</b> <span style="color:#06c">(B)</span> &mdash; ${verdict}${out}`
          +`${e.reason?`<br><span style="color:var(--muted)">${esc(e.reason)}</span>`:''}</div>`
          +`<div class="gallery">${imgs}</div></div>`;
      }).join('');
      const dropRows=(occ.dropped_edges||[]).map(e=>
        `<div style="color:var(--muted);margin-bottom:4px;">&#10005; <s>${esc(e.occluder)} covers ${esc(e.occluded)}</s> &mdash; hierarchy gate (a support cannot occlude what rests on it)</div>`
      ).join('');
      const vitalRows=(occ.vitals||[]).map(v=>{
        const out=redet.has(v.rid)?` <span class="tier t3">REDETECTED</span>`
                                  :` <span class="tier" style="background:#b26a00;color:#fff;">recovery fell back</span>`;
        return `<div style="margin-bottom:10px;border-left:3px solid #06c;padding-left:8px;">`
          +`<div><b>${esc(v.rid)}</b> &mdash; vital part missing: <b>${esc(v.vital_part)}</b>${out}`
          +`${v.reason?`<br><span style="color:var(--muted)">${esc(v.reason)}</span>`:''}</div>`
          +(v.marked_url?`<div class="gallery"><figure><img src="${esc(v.marked_url)}"><figcaption>instance (red)</figcaption></figure></div>`:'')
          +`</div>`;
      }).join('');
      const border=(occ.border||[]).length?`<div style="color:var(--muted);margin-top:6px;">frame-cut (border): ${occ.border.map(esc).join(', ')}</div>`:'';
      oEl.innerHTML=pairRows+dropRows+(vitalRows?`<h3 style="margin:10px 0 6px;font-size:12px;">Vital parts</h3>`+vitalRows:'')+border;
    }
  }
  const inst=pp.instances||[], drops=pp.dropped||[], box=$('d-pre-inst');
  const TIER={1:'tier 1 · text',2:'tier 2 · crop+text',3:'tier 3 · point+tracker',4:'dropped'};
  // rebuild when the set OR the redetect state changes (no flicker on plain polls)
  const instSig=inst.length+'|'+drops.length+'|'+inst.filter(i=>i.redetect).length+'|'+inst.filter(i=>i.occlusion).length;
  if(box.dataset.sig!==instSig){
    box.dataset.sig=instSig;
    const buildFig=(it)=>{
      const tb=it.tier?`<span class="tier t${esc(it.tier)}">${esc(TIER[it.tier]||('tier '+it.tier))}</span> `:'';
      const p=it.icp;
      const poseLine=(q)=>`scale ${esc(q.scale)} &middot; t (${q.translate.map(esc).join(', ')}) &middot; rot° (${q.euler.map(esc).join(', ')})`;
      let icp='';
      if(p){
        if(p.pre&&p.post){  // before/after the shape-ICP step (camera space)
          const tag=p.applied===false?' <span style="color:#c00">(ICP rejected — no change)</span>':(p.applied===true?' <span style="color:#137333">(ICP applied)</span>':'');
          icp=`<br><span style="color:var(--muted)" title="pose entering shape-ICP (after manual alignment)">before ICP &middot; ${poseLine(p.pre)}</span>`
            +`<br><span style="color:var(--muted)" title="pose right after shape-ICP (before render-compare)">after ICP &nbsp;&middot; ${poseLine(p.post)}${tag}</span>`;
        } else if(p.final){  // older run: only the final optimized pose was recorded
          icp=`<br><span style="color:var(--muted)" title="SAM3D final layout pose (camera space)">pose &middot; ${poseLine(p.final)}</span>`;
        }
      }
      const rd=it.redetect;
      let pic=it.overlay_url?`<img src="${esc(it.overlay_url)}">`:'';
      let rdcap='';
      if(rd&&rd.edited_overlay_url){  // re-directed: TRUE original mask | re-detected (active) mask on the edited image
        const origUrl=rd.original_overlay_url||it.overlay_url;  // overlay_path is ACTIVE-state (t3 merge rewrites it)
        const mergedNote=(rd.merged_in&&rd.merged_in.length)?` · +${rd.merged_in.map(esc).join(', ')} merged in`:``;
        pic=`<div style="display:flex;gap:2px">`
          +`<div style="flex:1;min-width:0">${origUrl?`<img src="${esc(origUrl)}">`:''}<div style="font-size:10px;color:var(--muted);text-align:center">reference · original mask</div></div>`
          +`<div style="flex:1;min-width:0"><img src="${esc(rd.edited_overlay_url)}"><div style="font-size:10px;color:var(--muted);text-align:center">edited · re-detected mask${mergedNote}</div></div></div>`;
        const rbits=[];
        if(rd.hidden_pct!=null) rbits.push(`~${esc(rd.hidden_pct)}% occluded <span title="VLM-predicted occlusion ratio">(VLM)</span>`);
        if(rd.vital_missing||rd.vital_part) rbits.push(`vital part missing${rd.vital_part?`: ${esc(rd.vital_part)}`:''}`);
        if(rd.removed&&rd.removed.length) rbits.push(`removed ${rd.removed.map(esc).join(', ')}`);
        rdcap=`<br><span class="tier t3">REDETECTED</span> <span style="color:var(--muted)">${rbits.join(' &middot; ')}</span>`;
      } else if(it.occlusion){  // detected occluded but NOT redetected: deliberate skip (slight) or failed recovery
        const oc=it.occlusion;
        const skipped=oc.severity==='slight'&&!oc.vital_part;
        const pct=(oc.hidden_pct!=null)?` ~${oc.hidden_pct}% hidden`:'';
        const badge=skipped?`<span class="tier" style="background:#777;color:#fff;">OCCLUDED (slight${esc(pct)}) &mdash; not redetected</span>`
                           :`<span class="tier" style="background:#b26a00;color:#fff;">OCCLUDED${oc.severity?` (${esc(oc.severity)}${esc(pct)})`:''} &mdash; recovery fell back</span>`;
        rdcap=`<br>${badge} <span style="color:var(--muted)">${oc.occluders.length?`by ${oc.occluders.map(esc).join(', ')}`:''}${oc.vital_part?` &middot; vital: ${esc(oc.vital_part)}`:''}</span>`;
      }
      return `<figure>${pic}`
      +`<figcaption>${tb}<b>${esc(it.category)} #${esc(it.instance)}</b>${it.depth!=null?` &middot; ${esc(it.depth)}m`:''}${it.score!=null?` &middot; score ${esc(it.score)}`:''}${it.support?`<br><span style="color:var(--muted)">on ${esc(it.support)}</span>`:''}${it.description?`<br><span style="color:var(--muted)">${esc(it.description)}</span>`:''}${rdcap}${icp}</figcaption></figure>`;
    };
    // Split the detected instances into three groups: re-detected (occlusion recovery, shows the
    // original|edited pair -> double-width card), occluded-but-not-re-detected (slight skip OR
    // heavy recovery-fell-back), and the rest.
    const isRedet=(it)=>!!(it.redetect&&it.redetect.edited_overlay_url);
    const isOccl=(it)=>!isRedet(it)&&!!it.occlusion;
    const normalFigs=inst.filter(it=>!isRedet(it)&&!isOccl(it)).map(buildFig).join('');
    const occlFigs=inst.filter(isOccl).map(buildFig).join('');
    const redetFigs=inst.filter(isRedet).map(buildFig).join('');
    // Discarded objects: a marker tile (or, for geometric drops that kept their mask, the rejected
    // overlay) so they're visible, with the reason why.
    const gone=drops.map(d=>{
      const name=(d.category!=null)?`${esc(d.category)}${d.instance!=null?' #'+esc(d.instance):''}`
                                   :esc(d.id||(typeof d==='string'?d:JSON.stringify(d)));
      const reason=esc(d.reason||'discarded');
      const markAR=pp.input_aspect?`aspect-ratio:${esc(pp.input_aspect)};`:'';
      const mark=d.overlay_url
        ? `<div style="position:relative"><img src="${esc(d.overlay_url)}" style="opacity:.55">`
          +`<div class="discard-mark" style="position:absolute;inset:0">&#10005;<span>discarded</span></div></div>`
        : `<div class="discard-mark" style="${markAR}">&#10005;<span>discarded</span></div>`;
      return `<figure class="discarded">${mark}`
        +`<figcaption><span class="tier t4">discarded</span> <b>${name}</b>`
        +`<br><span style="color:var(--muted)">${reason}${d.description?': '+esc(d.description):''}</span></figcaption></figure>`;
    }).join('');
    const sec=(title,body,cls)=>body?`<h3 class="inst-h">${title}</h3><div class="gallery${cls||''}">${body}</div>`:'';
    box.innerHTML=
        sec('Detected',normalFigs)
      + sec('Occluded &mdash; not re-detected',occlFigs)
      + sec('Re-detected (occlusion recovery)',redetFigs,' redet')
      + sec('Discarded',gone);
  }
}

function updateSceneGraph(sg){
  const card=$('d-sg-card'); if(!card) return;
  if(!sg||!sg.nodes||!sg.nodes.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-sg-card')) return;
  const nodes=sg.nodes, byId={}; nodes.forEach(n=>byId[n.id]=n);
  const surfaces=nodes.filter(n=>n.kind==='root_surface');
  const kids=id=>nodes.filter(n=>n.kind!=='root_surface'&&n.support===id);
  const objSurf=nodes.filter(n=>n.kind!=='root_surface'&&(!n.support||!byId[n.support]));
  const row=(n,d)=>{const cls=n.kind==='root_surface'?'sgsurf':'sgobj';
    const formb=n.form?` <span class="sgform" title="main support form (legs/base merged when 'table')">${esc(n.form)}</span>`:'';
    return `<div class="sgnode" style="padding-left:${d*22}px"><span class="sgkind">${esc(n.kind==='root_surface'?'surface':'object')}</span>`
      +`<span class="${cls}">${esc(n.id)}</span>${formb}</div>`+kids(n.id).map(c=>row(c,d+1)).join('');};
  let html=surfaces.map(s=>row(s,0)).join('');
  if(objSurf.length) html+=`<div class="sgnode" style="color:var(--muted)">(unparented)</div>`+objSurf.map(o=>row(o,1)).join('');
  const rels=sg.relationships||[];
  // Same-size objects: the pipeline resized these to one shared box and FROZE their
  // scale, so composition cannot scale-edit them (the SIZE-LOCK NOTE). Shown as its own
  // section because a locked object is otherwise indistinguishable from a mis-sized one.
  const ss=sg.same_size||[];
  if(ss.length){
    html+=`<div class="sgrelhdr" style="margin-top:10px">Same-size objects &mdash; resized to one shared box; scale LOCKED for composition</div>`
      +ss.map(g=>{
        const box=g.size?` <span class="sgrelid">shared box ${g.size.map(v=>v.toFixed(3)).join(' &times; ')} m</span>`:'';
        const note=g.unified?'':` <span class="sgrelid">(below ${g.minimum_members||2}-member threshold &mdash; not unified)</span>`;
        const source=g.source==='user_explicit'?'manual':'automatic';
        const label=g.display_name||g.category;
        return `<div class="sgnode"><span class="sgkind">same-size</span>`
          +`<span class="sgobj">${esc(label)}</span> <span class="sgrelid">&times;${g.members.length} (${source})</span>`
          +`${box}${note}<div class="sgrelmean">${g.members.map(esc).join(', ')}</div></div>`;
      }).join('');
  }
  const ssCount=ss.reduce((a,g)=>a+g.members.length,0);
  $('d-sg-summary').innerHTML=`<b>${surfaces.length}</b> root surface(s) &middot; <b>${nodes.length-surfaces.length}</b> object(s) &middot; <b>${rels.length}</b> relationship(s)`
    +(ssCount?` &middot; <b>${ssCount}</b> same-size (scale locked)`:'');
  $('d-sg-tree').innerHTML=html;
  const relBox=$('d-sg-rels');
  if(relBox){
    if(rels.length){
      relBox.style.display='block';
      const bs=sg.backstop||{};
      const gateRows=[
        ...(bs.against_upgraded||[]).map(u=>`<div class="sgrel"><span class="sgreltype" style="background:#137333;color:#fff;">against gate</span><span class="sgrelpair">upgraded: ${esc(u)}</span></div>`),
        ...(bs.corners_dropped||[]).map(c=>`<div class="sgrel"><span class="sgreltype" style="background:#b26a00;color:#fff;">corner gate</span><span class="sgrelpair"><s>${esc(c)}</s></span></div>`),
      ].join('');
      relBox.innerHTML=`<div class="sgrelhdr">Surface relationships &mdash; how the agent builds the room (proposer-detected; geometric gates applied)</div>`
        +rels.map(r=>`<div class="sgrel"><span class="sgreltype">${esc(r.type)}</span>`
          +`<span class="sgrelpair">${esc(r.a_name)} <span class="sgrelid">(${esc(r.a)})</span> <span class="sgrelarrow">&harr;</span> ${esc(r.b_name)} <span class="sgrelid">(${esc(r.b)})</span></span>`
          +(r.meaning?`<div class="sgrelmean">${esc(r.meaning)}</div>`:'')+`</div>`).join('')
        +gateRows;
    } else { relBox.style.display='none'; relBox.innerHTML=''; }
  }
}

function updatePoseMatch(pm){
  const card=$('d-pm-card'); if(!card) return;
  if(!pm||!pm.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-pm-card')) return;
  const rows=pm.map(v=>{
    const badge=v.skipped?`<span class="tier" style="background:#999;color:#fff;">skipped: ${esc(v.skipped)}</span>`
              :(v.accepted?`<span class="tier t1">applied</span>`
                          :`<span class="tier" style="background:#b26a00;color:#fff;">rejected</span>`);
    const imp=(v.improve_pct!=null)?`<div style="background:#e5efe5;border-radius:3px;height:8px;max-width:120px;"><div style="background:#137333;height:8px;border-radius:3px;width:${Math.max(2,Math.min(100,v.improve_pct))}%"></div></div>`:'';
    return `<tr>
      <td style="padding:3px 10px 3px 4px;"><b>${esc(v.obj)}</b></td>
      <td style="padding:3px 10px;">${v.rms_before_mm!=null?esc(v.rms_before_mm)+' &rarr; '+esc(v.rms_after_mm)+' mm':''}</td>
      <td style="padding:3px 10px;">${imp}${v.improve_pct!=null?`<span style="color:var(--muted)">-${esc(v.improve_pct)}%</span>`:''}</td>
      <td style="padding:3px 10px;color:var(--muted)">${v.dx_mm!=null?`&Delta;(${esc(v.dx_mm)}, ${esc(v.dy_mm)})mm`:''} ${v.yaw_deg!=null?`yaw ${esc(v.yaw_deg)}&deg;`:''} ${v.scale!=null?`&times;${esc(v.scale)}`:''}</td>
      <td style="padding:3px 4px;">${badge} <span style="color:var(--muted)">${v.n_pts?esc(v.n_pts)+' pts':''}</span></td>
    </tr>`;
  }).join('');
  $('d-pm').innerHTML=`<table style="border-collapse:collapse;width:100%;">
    <tr style="color:var(--muted);text-align:left;"><th style="padding:2px 4px;">object</th><th style="padding:2px 10px;">cloud rms</th><th style="padding:2px 10px;">improvement</th><th style="padding:2px 10px;">correction [x,y,yaw,scale]</th><th style="padding:2px 4px;"></th></tr>
    ${rows}</table>`;
}

function updatePose(po){
  const card=$('d-pose-card'); if(!card) return;
  if(!po||!po.objects||!po.objects.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-pose-card')) return;
  // three DIFFERENT verdicts, never merged into one count (see pose_state)
  const nred=po.n_red!=null?po.n_red:po.objects.filter(o=>o.severity==='red').length;
  const namb=po.n_amber!=null?po.n_amber:po.objects.filter(o=>o.severity==='amber').length;
  const nflip=po.objects.filter(o=>o.fell&&o.flip_accepted).length;
  const nfell=po.objects.filter(o=>o.fell&&!o.flip_accepted).length;
  const nres=po.objects.filter(o=>o.chosen==='pristine').length;
  const nroll=po.objects.filter(o=>o.rollable).length;
  const nrb=po.objects.filter(o=>o.chosen==='rolled_back').length;
  $('d-pose-summary').innerHTML=`root <b>${esc(po.table||'—')}</b> top pinned to z=0`
    +`${po.table_shift_mm?` (shift ${esc(po.table_shift_mm)}mm)`:''} &nbsp;&middot;&nbsp; `
    +`<b>${po.objects.length}</b> objects drop-tested`
    +(nroll?` &middot; <b>${nroll}</b> rollable (gated on drift, not tilt)`:``)
    +(nres?` &middot; <span style="color:#2e9e5b;font-weight:600">${nres} rescued by pristine</span>`:``)
    +(nrb?` &middot; <span style="color:#2e7fd9;font-weight:600">${nrb} rolled back to placed xy</span>`:``)
    +(nflip?` &middot; <span style="color:#c98a1b;font-weight:600">${nflip} flipped to stable face</span>`:``)
    +(nfell?` &middot; <span style="color:#c98a1b;font-weight:600">${nfell} accepted fallen rest</span>`:``)
    +(nred?` &middot; <span style="color:#e0483a;font-weight:600">${nred} flagged by a stage</span>`
          :` &middot; <span style="color:#2e9e5b;font-weight:600">no stage flagged an object</span>`)
    +(namb?` &middot; <span style="color:#c98a1b;font-weight:600">${namb} shipped rotated (stages clean)</span>`:``)
    +(po.has_delivered?``:` &middot; <span style="color:var(--muted)">no delivered-vs-placed measure (pre-07-31 run)</span>`)
    +(po.certify_converged===false
        ?` &middot; <span style="color:#e0483a;font-weight:600">&#9888; certify hit its step cap — baked poses are a snapshot, not a rest</span>`:``);
  $('d-pose-box').innerHTML=po.objects.map(o=>{
    const resc=o.chosen==='pristine';
    const rb=o.chosen==='rolled_back';
    // severity, NOT delivered.capsized: the latter fires on grounding (see pose_severity)
    const red=o.severity==='red', amber=o.severity==='amber';
    const chip=o.rollable?' <span style="color:var(--muted);font-size:10px;border:1px solid var(--muted);border-radius:3px;padding:0 3px;vertical-align:1px;">ROLLABLE</span>':'';
    return `<div class="posecard"${red?' style="outline:2px solid #e0483a"':(amber?' style="outline:2px solid #c98a1b"':(resc?' style="outline:2px solid #2e9e5b"':(rb?' style="outline:2px solid #2e7fd9"':'')))}>`
    +`<div class="pn">${esc(o.name)}${chip}${red?' <span style="color:#e0483a;font-weight:700">&#9888; FLAGGED BY A STAGE</span>'
        :(amber?' <span style="color:#c98a1b;font-weight:700">&#9888; SHIPPED ROTATED (stages clean)</span>'
        :(o.fell?(o.flip_accepted?' <span style="color:#c98a1b;font-weight:700">&#8635; FLIPPED TO STABLE FACE</span>'
                                 :' <span style="color:#c98a1b;font-weight:700">ACCEPTED FALLEN REST</span>')
        :(resc?' <span style="color:#2e9e5b;font-weight:700">&#10003; RESCUED</span>'
        :(rb?' <span style="color:#2e7fd9;font-weight:700">&#8617; ROLLED BACK</span>':''))))}</div>`
    +(o.cert_tilt_deg!=null?`<div class="poserow"><span class="k">certify free sim (joint, all dynamic)</span><span${red?' style="color:#e0483a;font-weight:600"':''}>${esc(o.cert_tilt_deg)}&deg;${o.cert_dxy_mm!=null?` &middot; ${esc(o.cert_dxy_mm)} mm`:''}</span></div>`:``)
    +`<div class="poserow"><span class="k">preprocess drop-test, per-object${o.rollable?' (tilt not gated)':' (gate 45&deg;)'}</span><span>${esc(o.tilt_deg)}&deg;</span></div>`
    +(o.rollable&&o.disp_raw_mm!=null?`<div class="poserow"><span class="k">&nbsp;&nbsp;&#8627; drift (gate 50mm)</span><span>${esc(o.disp_raw_mm)} mm</span></div>`:``)
    +(o.delivered?`<div class="poserow"><span class="k">delivered vs MoGE placement (whole pipeline)</span><span${amber?' style="color:#c98a1b;font-weight:600"':' style="color:var(--muted)"'}>${esc(o.delivered.tilt_deg)}&deg; &middot; ${esc(o.delivered.dxy_mm)} mm</span></div>`:``)
    +(rb&&o.rollback_mm!=null?`<div class="poserow"><span class="k">rolled back</span><span style="color:#2e7fd9">${esc(o.rollback_mm)} mm &rarr; placed xy, settled rotation kept</span></div>`:``)
    +(resc&&o.tilt_raw!=null?`<div class="poserow"><span class="k">raw (ditched)</span><span style="color:var(--muted)">${esc(o.tilt_raw)}&deg; &rarr; pristine</span></div>`:``)
    +`<div class="poserow"><span class="k">preprocess re-drop z-shift</span><span>${o.rest_mm>=0?'+':''}${esc(o.rest_mm)} mm</span></div>`
    +`</div>`;}).join('');
}

function updatePseudoGt(pg){
  const card=$('d-pgt-card'); if(!card) return;
  if(!pg||!pg.views||!pg.views.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-pgt-card')) return;
  const trusted=pg.trusted_count==null?pg.count:pg.trusted_count;
  $('d-pgt-summary').innerHTML=`<b>${pg.count}</b> views, <b>${trusted}</b> trusted &mdash; each orbit camera gets a SHARP feed-forward 3DGS render of the source view (falling back to a MoGE point-cloud splat), and GPT-Image-2 polishes/completes it. Only trusted source/pseudo-GT references are returned to the composition agent; drift or generation failures are shown here but withheld from it.`;
  const fig=(u,lbl)=>u
    ? `<figure><img src="${esc(u)}" loading="lazy"><figcaption>${lbl}</figcaption></figure>`
    : `<figure><div class="pgtwait">n/a</div><figcaption>${lbl}</figcaption></figure>`;
  const renderLbl=(k)=>k==='sharp'?'SHARP render':(k==='splat'?'point-cloud splat (holes)':'render (n/a)');
  $('d-pgt-box').innerHTML=pg.views.map(v=>{
    const ref=(v.az===0&&v.el===0);
    const trust=v.trusted!==false;
    const reason=v.untrusted_reason==='generation_failed'?'generation failed':(v.untrusted_reason==='excessive_observed_content_drift'?'excessive observed-content drift':(v.untrusted_reason||'quality check failed'));
    const imgs=ref
      ? fig(v.completed_url,'source view')
      : fig(v.render_url,renderLbl(v.render_kind))+`<div class="pgtarrow">&rarr;</div>`+fig(v.completed_url,'GPT-completed');
    return `<div class="pgtcard"><div class="pgttitle">azimuth ${v.az}&deg; &middot; elevation ${v.el}&deg;`
      +`${ref?' <span style="color:var(--muted)">(source reference)</span>':(trust?' <span style="color:var(--ok)">(trusted soft target)</span>':` <span style="color:var(--bad)">(withheld: ${esc(reason)})</span>`)}</div>`
      +`<div class="pgtimgs">${imgs}</div></div>`;
  }).join('');
}

function updateStateViz(sv){
  const card=$('d-sviz-card'); if(!card) return;
  if(!sv||!sv.states||!sv.states.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-sviz-card')) return;
  $('d-sviz-box').innerHTML=sv.states.map((s,i)=>
    (i?'<div class="svizarrow">&rarr;</div>':'')
    +`<figure class="svizframe"><img src="${esc(s.url)}" loading="lazy">`
    +`<figcaption><b>${esc(s.tag)}</b><br>${esc(s.caption)}</figcaption></figure>`).join('');
}

function openImg(url){ if(url) window.open(url,'_blank'); }
function updateRegister(rg){
  const card=$('d-reg-card'); if(!card) return;
  if(!rg||!rg.objects||!rg.objects.length){ card.style.display='none'; return; }
  card.style.display='block';
  if(collapsed('d-reg-card')) return;
  const moves=rg.objects.reduce((a,o)=>a+o.trace.length,0);
  const kept=rg.objects.reduce((a,o)=>a+o.trace.filter(t=>t.accepted!==false).length,0);
  $('d-reg-summary').innerHTML=`<b>${rg.count}</b> objects &nbsp;&middot;&nbsp; `
    +(rg.investigations&&rg.investigations.length?`<b>${rg.investigations.length}</b> investigations &nbsp;&middot;&nbsp; `:'')
    +(moves?`<b>${moves}</b> moves (${kept} kept) &nbsp;&middot;&nbsp; `:'')
    +`mean IoU <b>${rg.mean_before}</b> &rarr; <b>${rg.mean_after}</b> &nbsp;&middot;&nbsp; `
    +`<span style="color:var(--ok)">${rg.improved} improved</span>`
    +(rg.in_progress?` &nbsp;&middot;&nbsp; <span class="spin"></span> refining <b>${esc(rg.in_progress)}</b>…`:'');
  // Investigation composites (rendered scene | reference photo) the agent saw, in call order.
  const inv=$('d-reg-inv');
  if(inv) inv.innerHTML=(rg.investigations||[]).map(v=>
    `<figure class="reginvcard"><img src="${esc(v.url)}" loading="lazy" onclick="openImg('${esc(v.url)}')">`
    +`<figcaption>investigate #${v.n} &mdash; rendered scene | reference photo</figcaption></figure>`).join('');
  const bar=(lbl,v,c)=>`<div class="regbar"><span class="l">${lbl}</span>`
    +`<div class="t"><div style="width:${(Math.max(0,Math.min(1,v))*100).toFixed(1)}%;background:${c}"></div></div>`
    +`<span class="v">${v.toFixed(3)}</span></div>`;
  const fmtc=v=>v==null?'':(''+v);       // compact cell (blank for null)
  const figwait=lbl=>`<figure class="regframe"><div class="regwait"><span class="spin"></span></div><figcaption>${lbl}</figcaption></figure>`;
  $('d-reg-box').innerHTML=rg.objects.map(o=>{
    const col=o.delta>=0?'var(--ok)':'var(--bad)';
    // Per-round trace: IoU + gain + the physics-settle outcome (tilt / lift / kept).
    const tr=o.trace.map(t=>{
      const ok=t.accepted===false?'<span style="color:var(--bad)">✗</span>'
              :t.accepted===true?'<span style="color:var(--ok)">✓</span>':'';
      return `<tr><td>${fmtc(t.round)}</td><td>${esc(t.axis||'—')}</td>`
        +`<td>${t.iou.toFixed(3)}</td><td>${t.gain!=null?(t.gain>=0?'+':'')+t.gain.toFixed(3):''}</td>`
        +`<td>${t.tilt!=null?t.tilt.toFixed(1)+'°':''}</td><td>${t.lift_mm!=null?fmtc(t.lift_mm)+'mm':''}</td>`
        +`<td>${ok}</td></tr>`;
    }).join('');
    // Legacy runs stored per-step renders + a target crop; show the filmstrip only then.
    const fr=o.trace.filter(t=>t.render_url);
    const frames=fr.map((t,i)=>{
      const last=!o.active&&i===fr.length-1;
      const lbl=(t.round===0?'start':'r'+(t.round??'')+(t.axis?' · '+esc(t.axis):''));
      return `<figure class="regframe${last?' last':''}"><img src="${esc(t.render_url)}" loading="lazy">`
        +`<figcaption>${lbl}<br>IoU ${t.iou.toFixed(3)}</figcaption></figure>`;
    }).join('');
    const hasStrip=o.ref_url||fr.length||o.active;
    const strip=hasStrip?`<div class="regstrip">`
      +(o.ref_url?`<figure class="regframe target"><img src="${esc(o.ref_url)}" loading="lazy"><figcaption>target</figcaption></figure><div class="regarrow">&rarr;</div>`:'')
      +`<div class="regfilm">${frames||(o.active?figwait('rendering…'):'')}</div></div>`:'';
    const p=o.pose;
    // Never-refined objects (agent kept the placement) have no trajectory — show their
    // absolute IoU, not a "+0.000" delta, and skip the before/after bars.
    const head=o.active
      ? `<span class="reglive"><span class="spin"></span> aligning…</span>`
      : (!o.refined
        ? `<span style="font:12px ui-monospace,monospace">IoU ${o.after.toFixed(3)}</span> <span style="color:var(--muted);font-size:11px">not refined</span>`
        : `<span style="color:${col};font:12px ui-monospace,monospace">IoU ${o.delta>=0?'+':''}${o.delta.toFixed(3)}</span>`);
    const bars=(!o.refined&&!o.active)
      ? bar('IoU',o.after,'var(--muted)')
      : bar('before',o.before,'var(--muted)')+bar(o.active?'current':'after',o.after,col);
    // Pose footer only for legacy runs that recorded a committed pose; the current
    // composition path leaves it identity (moves live in the per-round trace above).
    const posef=o.moved
      ? `<div style="color:var(--muted);font:10px ui-monospace,monospace;margin-top:6px;">`
        +`move · t[${p.translate.join(', ')}] · e[${p.euler.join(', ')}] · s ${p.scale}</div>`
      : '';
    return `<div class="regcard${o.active?' active':''}">`
      +`<div class="rid"><b>${esc(o.id)}</b> ${head}</div>`
      +bars
      +strip
      +(o.trace.length?`<table class="regtrace"><thead><tr><th>r</th><th>axis</th><th>IoU</th><th>gain</th><th>tilt</th><th>lift</th><th>kept</th></tr></thead>`
        +`<tbody>${tr}</tbody></table>`:'')
      +posef
      +`</div>`;
  }).join('');
}

// ---- collapsible visualization blocks (default closed, lazily populated) ----
const COLLAPSIBLE=['d-pre-card','d-pre-inst-card','d-pre-merge-card','d-pre-occl-card','d-time-card','d-sg-card','d-pm-card','d-pose-card','d-sviz-card','d-reg-card','d-pgt-card','d-gallery-card','d-agents-card','d-log-card'];
let openBlocks=new Set();   // persists across polls + run switches; empty = all closed
let lastDetail=null;
function collapsed(id){const c=$(id);return !!c&&c.classList.contains('collapsed');}
function isOpen(id){const c=$(id);return !!c&&!c.classList.contains('collapsed');}
function setupCollapsibles(){
  for(const id of COLLAPSIBLE){
    const card=$(id); if(!card||card.dataset.cinit) continue;
    card.dataset.cinit='1';
    card.classList.add('collapsible');
    if(!openBlocks.has(id)) card.classList.add('collapsed');
    const h2=card.querySelector('h2'); if(!h2) continue;
    const body=document.createElement('div'); body.className='cardbody';
    while(h2.nextSibling) body.appendChild(h2.nextSibling);   // move content under the toggle
    card.appendChild(body);
    h2.classList.add('toggle');
    h2.insertAdjacentHTML('afterbegin','<span class="caret"></span>');
    h2.addEventListener('click',()=>toggleBlock(id));
  }
}
function toggleBlock(id){
  const card=$(id); if(!card) return;
  if(card.classList.toggle('collapsed')) openBlocks.delete(id);   // now collapsed
  else { openBlocks.add(id); lazyLoadBlock(id); }                 // opened -> populate now
}
function lazyLoadBlock(id){
  const d=lastDetail; if(!d) return;
  ({
    'd-pre-card':()=>updatePreprocess(d.preprocess),
    'd-sg-card':()=>updateSceneGraph(d.scene_graph),
    'd-pose-card':()=>updatePose(d.pose),
    'd-pm-card':()=>updatePoseMatch(d.pose_match),
    'd-reg-card':()=>updateRegister(d.register),
    'd-sviz-card':()=>updateStateViz(d.state_viz),
    'd-pgt-card':()=>updatePseudoGt(d.pseudo_gt),
    'd-gallery-card':()=>updateGallery(d),
    'd-pre-occl-card':()=>updatePreprocess(d.preprocess),
    'd-time-card':()=>updateTiming(d.timings),
    'd-agents-card':()=>loadAgents(),
    'd-log-card':()=>{const e=$('d-log'); if(e) e.textContent=d.log_tail||'(waiting for log...)';},
  }[id]||(()=>{}))();
}
function updateGallery(d){
  const card=$('d-gallery-card'); if(!card) return;
  const gallery=(d.stage_renders||[]).map(s=>{const t=stageTime(d.timings,s.stage);
    return `<figure><img src="${esc(s.url)}" loading="lazy"><figcaption>${esc(s.stage)}${t?` <span style="color:var(--muted)">&middot; ${t}</span>`:''}</figcaption></figure>`;}).join('');
  card.style.display=gallery?'block':'none';
  if(collapsed('d-gallery-card')) return;
  $('d-gallery').innerHTML=gallery;
}

function updateTiming(t){
  const box=$('d-timing'); if(!box) return;
  if(!t||!Object.keys(t).length){ box.innerHTML='<div class="empty">No timings yet.</div>'; return; }
  const steps=t.preprocess_steps||{}, par=t.preprocess_parallel||{};
  const entries=Object.entries(steps);
  // The background pseudo-GT build spawns right after canonicalize and joins at the
  // end: render it as a THIRD column cell row-spanning exactly the steps it overlaps,
  // with a vertical bar marking the span. Its wall time never joins the totals.
  // Each background step carries its spawn anchor ('after') and how many following
  // sequential rows it overlaps ('span'): render as a third-column cell row-spanning
  // that range, vertical bar marking the extent (Gantt-style beside the column).
  const cellFor=(k,v)=>
    `<div style="white-space:nowrap;"><b>&#8741; ${esc(k)}</b></div>`
    +`<div>${fmtTime(v.secs)} in bg</div>`
    +`<div>${fmtTime(v.hidden)} hidden${v.wait?`, ${fmtTime(v.wait)} wait`:''}</div>`
    +(v.ok===false?'<div style="color:#c00;font-weight:600;">FAILED</div>':'');
  const anchored={}, unanchored=[];
  for(const [k,v] of Object.entries(par)){
    const ai=entries.findIndex(([sk])=>sk===(v.after||'canonicalize'));
    if(ai>=0&&(v.span||0)>0) anchored[ai+1]=(anchored[ai+1]||[]).concat([[k,v]]);
    else unanchored.push([k,v]);
  }
  const row=(k,v,sub,extra)=>`<tr${sub?' style="color:var(--muted)"':''}><td style="padding:2px 14px 2px ${sub?22:6}px;">${k}</td><td style="text-align:right;padding:2px 6px;">${fmtTime(v)}</td>${extra||''}</tr>`;
  let html='<table style="border-collapse:collapse;">';
  if(t.preprocess!=null) html+=row('preprocess',t.preprocess,false);
  entries.forEach(([k,v],i)=>{
    let extra='';
    for(const [pk,pv] of (anchored[i]||[])){
      const span=Math.min(pv.span,entries.length-i);
      extra+=`<td rowspan="${span}" style="border-left:3px solid #b26a00;padding:2px 10px;vertical-align:middle;color:var(--muted);font-size:11px;line-height:1.5;">${cellFor(pk,pv)}</td>`;
    }
    html+=row(k==='pseudo_gt_wait'?'pseudo_gt (join wait)':esc(k),v,true,extra);
  });
  for(const [pk,pv] of unanchored)  // older runs / unknown anchor: plain row fallback
    html+=`<tr style="color:var(--muted)"><td style="padding:2px 14px 2px 22px;">&#8741; ${esc(pk)}</td><td style="text-align:right;padding:2px 6px;">${fmtTime(pv.secs)}</td><td style="border-left:3px solid #b26a00;padding:2px 10px;font-size:11px;">${cellFor(pk,pv)}</td></tr>`;
  // composition_certify = the final whole-scene physics certification settle (runs after
  // the composition agent, before export); label it clearly so its ~min cost is visible.
  const STAGE_LABEL={composition_certify:'composition &middot; certify (physics)'};
  for(const k of ['initializer','texture','lighting','composition','composition_certify','export'])
    if(t[k]!=null) html+=row(STAGE_LABEL[k]||esc(k),t[k],false);
  const skip=new Set(['preprocess_steps','preprocess_parallel']);
  const tot=Object.entries(t).reduce((a,[k,v])=>a+(skip.has(k)?0:(Number(v)||0)),0);
  html+=`<tr style="border-top:1px solid var(--line);font-weight:600;"><td style="padding:3px 14px 2px 6px;">total</td><td style="text-align:right;padding:3px 6px;">${fmtTime(tot)}</td></tr>`;
  box.innerHTML=html+'</table>';
}

function renderDetail(d){
  if(d.status==='not_found'){$('content').innerHTML='<div class="card"><div class="empty">Run not found.</div></div>';builtRun=null;return;}
  lastDetail=d;
  if(builtRun!==d.run_id){ buildSkeleton(); builtRun=d.run_id; curGlb=null; setView(viewMode); if(isOpen('d-agents-card')) loadAgents(); }

  const elapsed=d.elapsed!=null?Math.floor(d.elapsed/60)+'m '+(d.elapsed%60)+'s':'';
  const stoppable=(d.status==='active'||d.status==='queued')&&!String(d.run_id).includes('/')&&!window.__VIEW_ONLY__;
  const totalS=Object.entries(d.timings||{}).reduce((a,[k,v])=>a+(k==='preprocess_steps'?0:(Number(v)||0)),0);
  $('d-meta').innerHTML=`<span><b>${esc(d.run_id)}</b></span>
    <span class="badge ${esc(d.status)}">${esc(d.status)}</span>
    ${elapsed?`<span>elapsed: ${esc(elapsed)}</span>`:''}
    ${d.scene_name?`<span>scene: ${esc(d.scene_name)}</span>`:''}
    ${stoppable?'<button id="d-stop" class="stopbtn">&#9632; Stop</button>':''}
    ${totalS?`<span class="total" title="sum of per-stage wall-clock timings">total ${fmtTime(totalS)}</span>`:''}`;
  const sb=$('d-stop'); if(sb) sb.onclick=stopRun;

  if(d.status==='queued'){
    $('d-prog').style.display='none';
    const q=$('d-queued'); q.style.display='block';
    q.innerHTML=`<div class="queuebar" style="margin:6px 0 0;"><span>⏳ Waiting for a SAM3D GPU &mdash; <b>position ${esc(d.queue_position)}</b> of ${esc(d.queue&&d.queue.queued)} in queue (${esc(d.queue&&d.queue.active)}/${esc(d.queue&&d.queue.capacity)} GPUs busy). Starts automatically when a slot frees.</span></div>`;
  } else {
    $('d-queued').style.display='none'; $('d-prog').style.display='block';
    $('d-bar').style.width=pct(d.progress&&d.progress.percent)+'%';
    $('d-proglabel').textContent=(d.progress&&d.progress.label)||'';
  }

  $('d-blend').innerHTML=d.final_blend_url?`<a class="dl" href="${esc(d.final_blend_url)}">&#11015; Download final.blend</a>`:'';
  if(d.target_url){const t=$('d-target'); const tu=d.target_url+'&thumb=768'; if(t.getAttribute('src')!==tu) t.src=tu;}
  updateScene(d);
  updatePreprocess(d.preprocess);
  updateSceneGraph(d.scene_graph);
  updatePose(d.pose);
  updatePoseMatch(d.pose_match);
  updateStateViz(d.state_viz);
  updateRegister(d.register);
  updatePseudoGt(d.pseudo_gt);

  updateGallery(d);
  if(isOpen('d-time-card')) updateTiming(d.timings);
  // per-stage time next to panel names (preprocess covers its sub-panels; the pose
  // panel is the composition stage's investigate/move loop, so it shows composition's time)
  const setT=(id,k)=>{const e=$(id); if(e) e.textContent=stageTime(d.timings,k);};
  setT('d-pre-time','preprocess');
  setT('d-reg-time','composition');
  if(isOpen('d-log-card')) $('d-log').textContent=d.log_tail||'(waiting for log...)';
}

// ---- agent memory ----
const ROLE_ORDER=['initializer','texture','lighting','composition'];
async function loadAgents(){
  if(!selected)return;
  const box=$('agentbox'); if(!box)return;
  try{
    const d=await(await fetch('api/agents?id='+encodeURIComponent(selected))).json();
    const agents=d.agents||[];
    if(!agents.length){box.innerHTML='<div class="empty">No agent memory yet.</div>';return;}
    const groups={};
    agents.forEach(a=>{(groups[a.stage]=groups[a.stage]||[]).push(a);});
    const order=Object.keys(groups).sort((a,b)=>(ROLE_ORDER.indexOf(a)+1||99)-(ROLE_ORDER.indexOf(b)+1||99));
    box.innerHTML=order.map(stage=>`
      <div class="stagegrp">
        <h3>stage ${esc(groups[stage][0].stage_index)} &middot; ${esc(stage)}</h3>
        <div class="agents">${groups[stage].map(a=>`
          <div class="agent-item" data-mem="${esc(a.mem)}" data-label="${esc(a.agent+' &middot; attempt '+a.attempt)}">
            <span class="role ${esc(a.role)}">${esc(a.role)}</span>
            <span>${esc(a.agent.replace(/Agent$/,''))} #${esc(a.attempt)}</span>
          </div>`).join('')}</div>
      </div>`).join('');
    box.querySelectorAll('.agent-item').forEach(el=>el.onclick=()=>openMemory(el.dataset.mem,el.dataset.label));
  }catch(e){}
}

function renderMessage(m){
  const parts=m.parts||[];
  const imgs=parts.filter(p=>p.type==='image');
  // `thought` is the agent's reasoning for this edit (required on every
  // execute_and_evaluate); the tool_calls blob after it is mostly Blender code, so it is
  // collapsed. Anything without a `kind` is ordinary prose and renders as before.
  const thought=parts.filter(p=>p.kind==='thought').map(p=>p.text).join('\n\n').trim();
  const calls=parts.filter(p=>p.kind==='tool_calls').map(p=>p.text).join('\n').trim();
  const plain=parts.filter(p=>p.type==='text'&&!p.kind).map(p=>p.text).join('\n').trim();
  const imgHtml=imgs.length?`<div class="msg-imgs">${imgs.map(p=>
    `<figure><img src="${esc(p.url)}" loading="lazy"><figcaption>&lt;image&gt;</figcaption></figure>`).join('')}</div>`:'';
  const plainHtml=plain?`<pre>${esc(plain)}</pre>`:'';
  const thoughtHtml=thought?`<div class="thought"><div class="thought-label">reasoning</div><pre>${esc(thought)}</pre></div>`:'';
  const callsHtml=calls?`<details class="calls"><summary>tool call &amp; code</summary><pre>${esc(calls)}</pre></details>`:'';
  const any=plain||thought||calls||imgs.length;
  return `<div class="msg ${esc(m.role)}">
    <div class="msg-role">${esc(m.role)}${m.name?' &middot; '+esc(m.name):''}</div>
    <div class="msg-body">${plainHtml}${thoughtHtml}${callsHtml}${imgHtml}${any?'':'<div class="empty">(empty)</div>'}</div></div>`;
}
async function openMemory(memPath,label){
  $('modalTitle').innerHTML=label||'Agent memory';
  $('modalBody').innerHTML='<div class="empty">Loading...</div>';
  $('modal').classList.add('open');
  try{
    const d=await(await fetch('api/memory?path='+encodeURIComponent(memPath))).json();
    const msgs=d.messages||[];
    $('modalBody').innerHTML=msgs.length?msgs.map(renderMessage).join(''):'<div class="empty">No messages.</div>';
  }catch(e){$('modalBody').innerHTML='<div class="empty">Failed to load memory.</div>';}
}
function closeMemory(){$('modal').classList.remove('open');}

async function stopRun(){
  if(!selected)return;
  if(!confirm('Stop this run? The pipeline process will be killed and cannot resume.'))return;
  const sb=$('d-stop'); if(sb){sb.disabled=true;sb.textContent='Stopping…';}
  try{
    await fetch('api/stop',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:selected})});
  }catch(e){}
  poll(); refreshRuns();
}

async function poll(){
  if(pollTimer)clearTimeout(pollTimer);
  if(!selected)return;
  try{
    const d=await(await fetch('api/run?id='+encodeURIComponent(selected))).json();
    renderDetail(d);
    refreshRuns();
    // keep polling while the run is live OR a 3D export is still building.
    const building=d.scene&&d.scene.building;
    if((d.status!=='complete'&&d.status!=='error')||building){pollTimer=setTimeout(poll,3000);}
  }catch(e){pollTimer=setTimeout(poll,5000);}
}

$('modalClose').onclick=closeMemory;
$('modal').onclick=e=>{if(e.target===$('modal'))closeMemory();};
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMemory();});
// Runs sidebar controls: search filters live (no refetch); tabs switch view + reset drill-down.
$('run-search').addEventListener('input',e=>{runSearch=e.target.value;renderRuns();});
$('run-tabs').querySelectorAll('.rvtab').forEach(b=>b.onclick=()=>{
  runView=b.dataset.view; runRoot=null; runL1=null; runL2=null;
  $('run-tabs').querySelectorAll('.rvtab').forEach(x=>x.classList.toggle('sel',x===b));
  renderRuns();
});
refreshRuns();
setInterval(()=>{if(!selected)refreshRuns();},5000);
</script>
</body>
</html>
"""

_INDEX_VIEW_HTML: Optional[str] = None


def index_html() -> str:
    """The page for the current mode. In view-only mode the launch/upload panel and GPU queue
    bar are stripped out (the launch-only JS self-guards on the now-absent nodes) and the
    window.__VIEW_ONLY__ flag is flipped on so the run-detail Stop button is suppressed."""
    global _INDEX_VIEW_HTML
    if not VIEW_ONLY:
        return INDEX_HTML
    if _INDEX_VIEW_HTML is None:
        html = re.sub(
            r"<!--LAUNCH_PANEL_START-->.*?<!--LAUNCH_PANEL_END-->", "", INDEX_HTML, flags=re.S
        )
        html = html.replace("window.__VIEW_ONLY__ = false;", "window.__VIEW_ONLY__ = true;")
        _INDEX_VIEW_HTML = html
    return _INDEX_VIEW_HTML


def main() -> None:
    parser = argparse.ArgumentParser(description="SceneRig detailed run viewer")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global VIEW_ONLY, OUTPUT_ROOT, ALLOWED_ROOTS
    VIEW_ONLY = True
    OUTPUT_ROOT = Path(args.output_dir).expanduser().resolve()
    ALLOWED_ROOTS = [OUTPUT_ROOT, (REPO_ROOT / "data").resolve()]
    DATA_CONFIG.output_dir = OUTPUT_ROOT

    _refresh_runs_async()  # prewarm the runs cache so the first visitor isn't waiting on it

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SceneRig demo serving on http://{args.host}:{args.port}", flush=True)
    print(f"Watching {OUTPUT_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
