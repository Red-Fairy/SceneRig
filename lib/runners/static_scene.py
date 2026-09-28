#!/usr/bin/env python3
"""Single-image SceneRig runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from lib.agents.harness_profile import DEFAULT_HARNESS_PROFILE, HARNESS_PROFILE_NAMES
from lib.agents.tool_defaults import (
    GENERATOR_TOOLS_DEFAULT,
    VERIFIER_TOOLS_DEFAULT,
    split_tool_scripts,
)
from lib.utils.cli import positive_int

PREPROCESS_MANIFEST_VERSION = 1
_PREPROCESS_REQUIRED_FILES = (
    "input.png",
    "masks/masks.json",
    "scene_graph.json",
    "main_support_yaw_observation.json",
    "initializer_constraints.json",
    "placement.json",
    "physics/pose_changes.json",
    "physics/preprocess_pose_changes.json",
    "blender_file.blend",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_empty_blend(blender_command: str, created_blender_file: str) -> None:
    program = (
        "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); "
        f"bpy.ops.wm.save_mainfile(filepath={created_blender_file!r})"
    )
    subprocess.run(
        [
            blender_command,
            "--background",
            "--factory-startup",
            "--python-expr",
            program,
        ],
        check=True,
    )


def _write_preprocess_manifest(scene_dir: Path, source_image: Path) -> Path:
    pose_changes = scene_dir / "physics/pose_changes.json"
    pose_snapshot = scene_dir / "physics/preprocess_pose_changes.json"
    if pose_changes.is_file() and not pose_snapshot.exists():
        shutil.copy2(pose_changes, pose_snapshot)

    missing = [rel for rel in _PREPROCESS_REQUIRED_FILES if not (scene_dir / rel).is_file()]
    if missing:
        raise RuntimeError(
            "preprocessing did not produce required artifacts: " + ", ".join(missing)
        )

    artifacts: list[str] = []
    for root_name in ("masks", "meshes", "moge", "physics", "pseudo_gt"):
        root = scene_dir / root_name
        if root.exists():
            artifacts.extend(
                str(path.relative_to(scene_dir))
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and path.name not in {"settle_server.log", "isaac.port"}
                and not path.name.startswith(".")
            )
    artifacts.extend(_PREPROCESS_REQUIRED_FILES)

    manifest = {
        "schema_version": PREPROCESS_MANIFEST_VERSION,
        "source_image": {
            "path": str(source_image.resolve()),
            "sha256": _sha256(scene_dir / "input.png"),
            "original_sha256": _sha256(source_image),
        },
        "artifacts": sorted(set(artifacts)),
    }
    path = scene_dir / "preprocess_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _validate_scene_inventory(scene_dir: Path, *, gpt6_harness: bool = False) -> None:
    from lib.tools.geometry.inventory_contract import validate_scene_artifacts

    validate_scene_artifacts(
        scene_dir,
        allow_runtime_additions=True,
        allow_runtime_inventory=gpt6_harness,
        validate_runtime_physics=gpt6_harness,
    )


def _stage_depth_inputs(
    scene_dir: Path,
    *,
    depth: Optional[str],
    camera: Optional[str],
) -> tuple[Optional[str], Optional[float]]:
    """Stage optional metric depth/camera inputs and return GT-depth options."""

    if camera and not depth:
        raise ValueError("--camera is supported when paired with --depth")
    if not depth:
        return None, None
    if depth == "auto":
        return "auto", None

    depth_path = Path(depth).expanduser()
    if not depth_path.is_file():
        raise FileNotFoundError(f"--depth does not exist: {depth_path}")
    if not camera:
        return str(depth_path), None

    camera_path = Path(camera).expanduser()
    if not camera_path.is_file():
        raise FileNotFoundError(f"--camera does not exist: {camera_path}")
    staged = scene_dir / "provided_geometry"
    staged.mkdir(parents=True, exist_ok=True)
    staged_depth = staged / "depth.npy"
    staged_intrinsics = staged / "intrinsics.json"
    shutil.copy2(depth_path, staged_depth)
    shutil.copy2(camera_path, staged_intrinsics)
    return str(staged_depth), None


def _task_config(image: str, output_dir: str, task: str = "scene") -> dict:
    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"target image not found: {path}")
    return {
        "task_name": task,
        "target_image_path": str(path.resolve()),
        "output_dir": str(Path(output_dir) / task),
        "init_code_path": "",
        "init_image_path": "",
    }


def _build_main_command(
    task_config: dict,
    args: argparse.Namespace,
    scene_dir: Path,
    blender_file: Path,
) -> list[str]:
    harness = "gpt6_v1" if args.gpt6_harness else DEFAULT_HARNESS_PROFILE
    cmd = [
        sys.executable,
        "main.py",
        "--model",
        args.model,
        "--harness-profile",
        harness,
        "--preprocess-model",
        args.preprocess_model or args.model,
        "--max-stage-attempts",
        str(args.max_stage_attempts),
        "--target-image-path",
        str(scene_dir / "input.png"),
        "--moge-dir",
        str(scene_dir),
        "--output-dir",
        task_config["output_dir"],
        "--task-name",
        task_config["task_name"],
        "--generator-tools",
        args.generator_tools,
        "--verifier-tools",
        args.verifier_tools,
        "--blender-command",
        args.blender_command,
        "--blender-file",
        str(blender_file),
        "--blender-save",
        str(blender_file),
        "--blender-script",
        args.blender_script,
        "--init-code-path",
        task_config["init_code_path"],
        "--init-image-path",
        task_config["init_image_path"],
        "--clear-memory",
        "--verifier-max-rounds",
        str(args.verifier_max_rounds),
    ]
    if args.gpu_devices:
        cmd.extend(["--gpu-devices", args.gpu_devices])
    if args.ignore_objects:
        cmd.extend(["--ignore_objects", *args.ignore_objects])
    return cmd


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    secret_flags = {"--api-key", "--token", "--access-token"}
    for i, token in enumerate(redacted):
        if token in secret_flags and i + 1 < len(redacted):
            redacted[i + 1] = "<redacted>"
        for flag in secret_flags:
            if token.startswith(flag + "="):
                redacted[i] = flag + "=<redacted>"
    return redacted


def _run_and_tee_output(cmd: list[str], log_file) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return subprocess.CompletedProcess(cmd, proc.wait())


def run_task(task_config: dict, args: argparse.Namespace) -> tuple[str, bool, Optional[str]]:
    task_name = task_config["task_name"]
    scene_dir = Path(task_config["output_dir"])
    scene_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GRASE_USAGE_LOG"] = str(scene_dir / "usage.jsonl")
    os.environ["GRASE_USAGE_TAG"] = "preprocess"
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    blender_file = scene_dir / "blender_file.blend"
    if Path(args.blender_file).exists():
        shutil.copy2(args.blender_file, blender_file)
    else:
        _create_empty_blend(args.blender_command, str(blender_file))

    try:
        gt_depth, gt_fov_x_deg = _stage_depth_inputs(
            scene_dir, depth=args.depth, camera=args.camera
        )
        if gt_depth == "auto":
            candidate = Path(task_config["target_image_path"]).parent / "depth.npy"
            if not candidate.is_file():
                raise FileNotFoundError(f"--depth auto expected {candidate}")
            gt_depth = str(candidate)

        print(f"[preprocess] reconstructing {task_config['target_image_path']}")
        from lib.tools.geometry.preprocess import preprocess_scene

        info = preprocess_scene(
            task_config["target_image_path"],
            str(scene_dir),
            blender_cmd=args.blender_command,
            blend_path=str(blender_file),
            model=args.preprocess_model or args.model,
            ignore_objects=args.ignore_objects,
            gt_depth=gt_depth,
            gt_fov_x_deg=gt_fov_x_deg,
            vlm_physics=True,
            vlm_physics_jobs=args.vlm_physics_batch,
        )
        if not info.get("objects"):
            raise RuntimeError("preprocessing produced zero object instances")
        _validate_scene_inventory(scene_dir, gpt6_harness=args.gpt6_harness)
        manifest = _write_preprocess_manifest(
            scene_dir, Path(task_config["target_image_path"])
        )
        print(f"[preprocess] manifest: {manifest}")

        if args.preprocess_only:
            print(f"PREPROCESS_ONLY_OK {scene_dir}")
            return task_name, True, None

        cmd = _build_main_command(task_config, args, scene_dir, blender_file)
        log_path = scene_dir / "task.log"
        with log_path.open("w") as log_file:
            log_file.write("Command: " + shlex.join(_redact_command(cmd)) + "\n\n")
            log_file.flush()
            result = _run_and_tee_output(cmd, log_file)
        if result.returncode == 0:
            return task_name, True, None
        return (
            task_name,
            False,
            f"agent process failed with return code {result.returncode}; see {log_path}",
        )
    except Exception as exc:  # noqa: BLE001
        fatal = scene_dir / "fatal_error.log"
        fatal.write_text(f"{type(exc).__name__}: {exc}\n")
        return task_name, False, str(exc)
    finally:
        try:
            from lib.tools.geometry.physics import shutdown_shared_settle_server

            shutdown_shared_settle_server(scene_dir / "physics")
        except Exception:
            pass


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.harness_profile != DEFAULT_HARNESS_PROFILE:
        parser.error("use --gpt6-harness instead of --harness-profile")
    if args.gpt6_harness and "gpt6_v1" not in HARNESS_PROFILE_NAMES:
        parser.error("this checkout does not expose the GPT-6 harness")
    for option, value in (
        ("--generator-tools", args.generator_tools),
        ("--verifier-tools", args.verifier_tools),
    ):
        try:
            scripts = split_tool_scripts(value)
        except ValueError as exc:
            parser.error(f"{option}: {exc}")
        for script in scripts:
            if not Path(script).is_file():
                parser.error(f"{option} script not found: {script}")
    if not Path(args.blender_script).is_file():
        parser.error(f"--blender-script not found: {args.blender_script}")
    blender_path = Path(str(args.blender_command))
    if shutil.which(str(args.blender_command)) is None and not (
        blender_path.is_file() and os.access(blender_path, os.X_OK)
    ):
        parser.error(f"--blender-command is not executable/found: {args.blender_command}")
    if args.camera and not args.depth:
        parser.error("--camera currently requires --depth")
    if args.depth and args.depth != "auto" and not Path(args.depth).expanduser().is_file():
        parser.error(f"--depth does not exist: {args.depth}")
    if args.camera and not Path(args.camera).expanduser().is_file():
        parser.error(f"--camera does not exist: {args.camera}")


def main() -> int:
    os.environ["GRASE_RUN_OWNER_PID"] = str(os.getpid())
    parser = argparse.ArgumentParser(description="SceneRig single-image reconstruction")
    parser.add_argument("--image", "--target-image", dest="image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="scene", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gpu-devices",
        "--gpu",
        dest="gpu_devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES"),
    )
    parser.add_argument("--model", default="claude-opus-5", help=argparse.SUPPRESS)
    parser.add_argument("--preprocess-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpt6-harness", action="store_true")
    parser.add_argument("--harness-profile", default=DEFAULT_HARNESS_PROFILE, help=argparse.SUPPRESS)
    parser.add_argument(
        "--depth",
        default=None,
        help="Metric Z-depth .npy, or 'auto' for depth.npy next to the image",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Intrinsics JSON {fx,fy,cx,cy,w,h}; requires --depth",
    )
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument(
        "--verifier-max-rounds",
        type=positive_int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-stage-attempts",
        type=positive_int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vlm-physics-batch",
        type=positive_int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ignore_objects",
        nargs="*",
        default=["robot arm", "robotic arm"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blender-command",
        default="lib/utils/third_party/blender-4.5/blender",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blender-file",
        default="data/static_scene/empty_scene.blend",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blender-script",
        default="data/static_scene/generator_script.py",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--generator-tools", default=GENERATOR_TOOLS_DEFAULT, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-tools", default=VERIFIER_TOOLS_DEFAULT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    _validate_args(parser, args)

    task = _task_config(args.image, args.output_dir, args.task)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
    )

    start = time.time()
    name, ok, error = run_task(task, args)
    elapsed = time.time() - start
    print("\nSceneRig run complete:")
    print(f"  Task: {name}")
    print(f"  Successful: {1 if ok else 0}")
    print(f"  Failed: {0 if ok else 1}")
    print(f"  Execution time: {elapsed:.2f} seconds")
    if not ok:
        print(f"  Error: {error}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
