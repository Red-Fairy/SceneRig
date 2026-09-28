"""Separate GRASE Blender-to-Isaac conversion with its own result contract.

This command never edits the core Blender ``pipeline_result.json``. It writes
``scene/isaac/conversion_result.json`` and exits:

* 0: dynamic_verified
* 2: unstable
* 3: stable_with_kinematic_fallback (converted, but not dynamically ready)
* 1: conversion_failed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.tools.geometry.inventory_contract import validate_scene_artifacts
from lib.utils.provenance import git_state

ISAAC_PY = Path("/fsx/rundongluo/isaac/venv/bin/python")


def _json_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text())
        return len(payload) if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _uses_gpt6_conversion_contract(core: dict) -> bool:
    """A gpt6_v1 run is recognized by its declared profile name, whatever its capabilities."""

    declared = core.get("harness_profile")
    name = declared.get("name") if isinstance(declared, dict) else declared
    return name == "gpt6_v1"


def _validate_gpt6_conversion_inputs(exp: Path, core: dict) -> None:
    """Check the semantic scene-artifact contract of a GPT-6 run before USD conversion."""

    if not _uses_gpt6_conversion_contract(core):
        return
    validate_scene_artifacts(
        exp / "scene",
        allow_runtime_additions=True,
        allow_runtime_inventory=True,
        validate_runtime_physics=True,
    )


def run_step(name: str, command: list[str], timeout: int) -> dict:
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "OMNI_KIT_ALLOW_ROOT": "1",
            },
        )
        return {
            "name": name,
            "returncode": result.returncode,
            "duration_seconds": round(time.time() - started, 2),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "returncode": 124,
            "duration_seconds": round(time.time() - started, 2),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "error": f"timed out after {timeout}s",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a completed GRASE Blender artifact to verified Isaac USD"
    )
    parser.add_argument("exp_dir")
    parser.add_argument("--isaac-python", default=str(ISAAC_PY))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--package-usdz", action="store_true")
    args = parser.parse_args()

    exp = Path(args.exp_dir)
    if not exp.is_absolute():
        exp = REPO_ROOT / exp
    exp = exp.resolve()
    blend = exp / "scene/final/final.blend"
    core_manifest = exp / "scene/final/pipeline_result.json"
    isaac_dir = exp / "scene/isaac"
    isaac_dir.mkdir(parents=True, exist_ok=True)
    result_path = isaac_dir / "conversion_result.json"
    result = {
        "schema_version": 2,
        "conversion_status": "running",
        "simulation_status": "not_verified",
        "simulation_ready": False,
        "core_blender_result_unchanged": False,
        "source": {"blend": str(blend), "pipeline_manifest": str(core_manifest)},
        "outputs": {},
        "steps": [],
        "warnings": [],
        "provenance": {
            "converter_code": git_state(REPO_ROOT),
            "runtime_versions": {"isaac_sim": None, "physx": None},
        },
    }
    original_core = core_manifest.read_bytes() if core_manifest.is_file() else None

    exit_code = 1
    try:
        if not blend.is_file():
            raise FileNotFoundError(f"Blender artifact not found: {blend}")
        if not core_manifest.is_file():
            raise FileNotFoundError(f"core Blender manifest not found: {core_manifest}")
        core = json.loads(core_manifest.read_text())
        if core.get("artifact_status") not in (None, "valid") or not core.get(
            "pipeline_complete"
        ):
            raise RuntimeError("core manifest does not identify a valid Blender artifact")
        for key, output in (core.get("outputs") or {}).items():
            path = output.get("path") if isinstance(output, dict) else None
            if not path or not Path(path).exists():
                raise RuntimeError(f"core manifest output {key!r} is missing: {path}")
        _validate_gpt6_conversion_inputs(exp, core)

        # A previous attempt's files must never qualify or contaminate the current
        # conversion. Keep the directory itself so the terminal result can always be
        # written, but clear every generated conversion product first.
        for name in (
            "scene.usd",
            "scene_visual.usdc",
            "scene_collision.usdc",
            "scene_collision.json",
            "collision_export_request.json",
            "scene.usdz",
            "visual_meshes.npz",
            "object_identity.json",
            "verify_report.json",
            "physics_overrides.json",
            "physics_vlm.json",
            "collision",
        ):
            stale = isaac_dir / name
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
            elif stale.exists() or stale.is_symlink():
                stale.unlink()
        verify_path = isaac_dir / "verify_report.json"

        conversion = run_step(
            "blend_to_isaac",
            [
                sys.executable,
                str(REPO_ROOT / "isaac/blend_to_isaac.py"),
                str(exp),
                "--skip-verify",
                "--isaac-python",
                args.isaac_python,
            ],
            args.timeout,
        )
        result["steps"].append(conversion)
        if conversion["returncode"] != 0:
            raise RuntimeError("Blender-to-USD conversion failed")

        stabilize = run_step(
            "auto_stabilize_and_verify",
            [
                args.isaac_python,
                str(REPO_ROOT / "isaac/isaac_auto_stabilize.py"),
                str(exp),
                "--isaac-python",
                args.isaac_python,
            ],
            args.timeout,
        )
        result["steps"].append(stabilize)
        if stabilize["returncode"] not in (0, 2, 3):
            raise RuntimeError("Isaac stabilization/verification process failed")
        if not verify_path.is_file():
            raise RuntimeError("Isaac verification produced no report")
        verify = json.loads(verify_path.read_text())
        simulation_status = verify.get("simulation_status", "unstable")
        allowed_statuses = {
            "dynamic_verified",
            "stable_with_kinematic_fallback",
            "unstable",
        }
        if simulation_status not in allowed_statuses:
            raise RuntimeError(
                f"Isaac verification returned an unknown status: {simulation_status}"
            )
        if simulation_status == "dynamic_verified" and not (
            verify.get("pass") is True and verify.get("dynamic_ready") is True
        ):
            raise RuntimeError("Isaac dynamic-verification fields are inconsistent")
        if simulation_status == "stable_with_kinematic_fallback" and not (
            verify.get("physics_pass") is True and verify.get("kinematic_bodies")
        ):
            raise RuntimeError("Isaac kinematic-fallback fields are inconsistent")
        result["conversion_status"] = "completed"
        result["simulation_status"] = simulation_status
        result["simulation_ready"] = simulation_status == "dynamic_verified"
        result["verify_summary"] = {
            key: verify.get(key)
            for key in (
                "pass",
                "physics_pass",
                "dynamic_ready",
                "missing_bodies",
                "unexpected_bodies",
                "kinematic_bodies",
                "identity_error",
                "thresholds",
            )
        }
        result["provenance"]["runtime_versions"] = verify.get(
            "runtime_versions"
        ) or {"isaac_sim": None, "physx": None}
        if simulation_status == "dynamic_verified":
            exit_code = 0
        elif simulation_status == "stable_with_kinematic_fallback":
            result["warnings"].append(
                {
                    "code": "kinematic_fallback",
                    "message": "USD is stable but required bodies are kinematic; it is not dynamically ready",
                }
            )
            exit_code = 3
        else:
            result["warnings"].append(
                {
                    "code": "unstable",
                    "message": "USD did not pass dynamic settle verification",
                }
            )
            exit_code = 2

        if args.package_usdz and exit_code in (0, 3):
            package = run_step(
                "package_usdz",
                [
                    args.isaac_python,
                    "-c",
                    "from pxr import UsdUtils; import sys; "
                    "UsdUtils.CreateNewUsdzPackage(sys.argv[1], sys.argv[2])",
                    str(isaac_dir / "scene.usd"),
                    str(isaac_dir / "scene.usdz"),
                ],
                args.timeout,
            )
            result["steps"].append(package)
            if package["returncode"] != 0:
                result["warnings"].append(
                    {"code": "package_failed", "message": "optional USDZ packaging failed"}
                )
    except Exception as exc:  # noqa: BLE001 - terminal status must always be written
        result["conversion_status"] = "conversion_failed"
        result["simulation_status"] = "not_verified"
        result["simulation_ready"] = False
        result["error"] = str(exc)
        exit_code = 1
    finally:
        for name in (
            "scene.usd",
            "scene_visual.usdc",
            "scene_collision.usdc",
            "scene_collision.json",
            "collision_export_request.json",
            "scene.usdz",
            "object_identity.json",
            "verify_report.json",
        ):
            path = isaac_dir / name
            if path.is_file():
                result["outputs"][name] = {"path": str(path)}
        collision = isaac_dir / "collision"
        result["outputs"]["collision"] = {
            "path": str(collision),
            "files": sorted(
                str(item.relative_to(collision))
                for item in (collision.rglob("*") if collision.is_dir() else ())
                if item.is_file()
            ),
        }
        physics_sources = {}
        for name in ("physics_vlm.json", "physics_overrides.json"):
            path = isaac_dir / name
            physics_sources[name] = {
                "path": str(path),
                "object_count": _json_count(path),
                "selected": path.is_file(),
            }
        result["provenance"]["physics_sources"] = physics_sources
        result["provenance"]["physics_source_summary"] = {
            "vlm_available": physics_sources["physics_vlm.json"]["selected"],
            "override_available": physics_sources["physics_overrides.json"]["selected"],
            "precedence": "per-object override > VLM estimate > heuristic default",
        }
        result["core_blender_result_unchanged"] = (
            original_core is not None and core_manifest.read_bytes() == original_core
        )
        if not result["core_blender_result_unchanged"]:
            result["warnings"].append(
                {
                    "code": "core_manifest_changed",
                    "message": "core Blender manifest changed during Isaac conversion",
                }
            )
        result["result_path"] = str(result_path)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(
        f"ISAAC_CONVERSION_{result['conversion_status'].upper()} "
        f"simulation_status={result['simulation_status']} result={result_path}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
