#!/usr/bin/env python3
"""Lightweight SceneRig run evaluator.

This intentionally avoids rendering or visualization. It checks the artifacts that
the public single-image pipeline is expected to produce and prints a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PREPROCESS_FILES = (
    "input.png",
    "masks/masks.json",
    "scene_graph.json",
    "placement.json",
    "physics/preprocess_pose_changes.json",
    "blender_file.blend",
)


def _scene_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "scene").is_dir():
        return path / "scene"
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"_json_error": True}
    return data if isinstance(data, dict) else {"_non_object_json": True}


def _count_instances(masks: dict[str, Any]) -> dict[str, int]:
    instances = masks.get("instances") or []
    objects = [row for row in instances if row.get("kind", "object") == "object"]
    roots = [row for row in instances if row.get("kind") == "root"]
    redetected = [row for row in objects if row.get("redetect")]
    return {
        "instances": len(instances),
        "objects": len(objects),
        "root_surfaces": len(roots),
        "generative_resegments": len(redetected),
    }


def evaluate(path: Path) -> dict[str, Any]:
    scene = _scene_dir(path)
    masks = _read_json(scene / "masks/masks.json")
    placement = _read_json(scene / "placement.json")
    final = _read_json(scene / "final/pipeline_result.json")
    manifest = _read_json(scene / "preprocess_manifest.json")

    missing = [
        rel for rel in REQUIRED_PREPROCESS_FILES if not (scene / rel).is_file()
    ]
    final_blend = scene / "final/final.blend"
    if not final_blend.is_file():
        final_blend = scene / "blender_file.blend"

    placement_rows = placement.get("objects") or placement.get("instances") or []
    return {
        "scene_dir": str(scene),
        "preprocess_complete": not missing,
        "missing_preprocess_artifacts": missing,
        "mask_counts": _count_instances(masks),
        "placed_objects": len(placement_rows) if isinstance(placement_rows, list) else 0,
        "final_result": {
            "exists": bool(final),
            "pipeline_complete": final.get("pipeline_complete"),
            "artifact_status": final.get("artifact_status"),
            "export_status": final.get("export_status"),
            "final_blend": str(final_blend) if final_blend.is_file() else None,
        },
        "preprocess_manifest_artifacts": len(manifest.get("artifacts") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a SceneRig output directory")
    parser.add_argument("run_dir", help="Run output directory, or its scene/ subdirectory")
    args = parser.parse_args()
    summary = evaluate(Path(args.run_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["preprocess_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
