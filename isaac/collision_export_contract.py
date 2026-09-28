"""Bind a fresh Blender collision-source export to finalized scene identities."""

from __future__ import annotations

import json
from pathlib import Path

from lib.tools.geometry.runtime_object_repair import sha256_file


def build_collision_export_request(exp: Path) -> dict:
    """Authenticate authored roots; never infer authorization from Blender names.

    Baseline exports retain their ordinary evaluated meshes. GPT-6 exports reuse
    the same current-revision binding authority as composition certification.
    """
    # Optional GPT-6 boundary dependencies are loaded only by the export driver.
    from isaac.convert_scene import (
        _uses_gpt6_conversion_contract,
        _validate_gpt6_conversion_inputs,
    )

    scene = exp.resolve() / "scene"
    placement_path = scene / "placement.json"
    placement = json.loads(placement_path.read_text())
    names = [row["mesh_name"] for row in placement["objects"]]
    if any(not isinstance(name, str) or not name.startswith("obj_") for name in names):
        raise ValueError(f"Malformed export object names: {names!r}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Export requires unique nonempty object names: {names!r}")
    core_path = scene / "final/pipeline_result.json"
    core = json.loads(core_path.read_text()) if core_path.is_file() else {}
    authored_ids = {}
    if _uses_gpt6_conversion_contract(core):
        _validate_gpt6_conversion_inputs(exp.resolve(), core)
        # Optional GPT-6 authority; do not duplicate its revision/transaction rules.
        from lib.tools.geometry.composition_physics import _gpt6_certification_contract

        _, authored_ids, _ = _gpt6_certification_contract(scene, names)
    elif (scene / "runtime_objects/inventory.json").exists() or any(
        row.get("procedural_capture") for row in placement["objects"]
    ):
        raise ValueError("Authored export requires a finalized canonical GPT-6 profile")
    blend = scene / "final/final.blend"
    digest = sha256_file(blend)
    declared = ((core.get("outputs") or {}).get("final_blend") or {}).get("sha256")
    if declared is not None and digest != declared:
        raise ValueError("Collision export Blend differs from its finalized hash")
    return {
        "schema_version": 1,
        "policy": "current_pose_authored_union_v1",
        "blend_sha256": digest,
        "placement_sha256": sha256_file(placement_path),
        "expected_objects": sorted(names),
        "authored_empty_ids": authored_ids,
    }
