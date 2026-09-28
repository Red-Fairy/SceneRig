"""Focused checks for the immutable-source runtime-object inventory overlay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lib.tools.geometry.inventory_contract import (
    InventoryContractError,
    resolved_inventory,
    validate_scene_artifacts,
    validate_scene_graph_inventory,
)
from lib.tools.geometry.runtime_object_repair import (
    RuntimeObjectRepairError,
    build_runtime_object_record,
    new_runtime_inventory,
    runtime_inventory_tombstones,
    sha256_file,
    stage_runtime_mesh_revision,
    stage_runtime_object_batch,
    validate_runtime_inventory_schema,
    write_runtime_inventory,
)


def _physical_material(status: str = "estimated", material: str = "ceramic") -> dict:
    return {
        "status": status,
        "provenance": {"source": "physics_vlm"},
        "material": material,
        "mass_kg": 0.4,
        "mass_range_kg": [0.2, 0.8],
        "friction": 0.5,
        "density_kgm3": 1000.0,
        "density_gate": "ok" if status == "estimated" else "fallback",
    }


def _source_scene(tmp_path: Path) -> tuple[dict, dict, dict]:
    masks_dir = tmp_path / "masks"
    meshes_dir = tmp_path / "meshes"
    masks_dir.mkdir(parents=True)
    meshes_dir.mkdir(parents=True)
    np.save(masks_dir / "table.npy", np.ones((4, 4), dtype=bool))
    np.save(masks_dir / "mug.npy", np.ones((4, 4), dtype=bool))
    (meshes_dir / "mug.glb").write_bytes(b"source mug glTF")

    masks = {
        "instances": [
            {
                "category": "table",
                "instance": 0,
                "kind": "root_surface",
                "support": None,
                "mask_path": "masks/table.npy",
            },
            {
                "category": "mug",
                "instance": 0,
                "kind": "object",
                "support": "table#0",
                "mask_path": "masks/mug.npy",
            },
        ],
        "unmasked_root_surfaces": [],
        "relationships": [],
    }
    graph = {
        "main_support_id": "table#0",
        "nodes": [
            {
                "id": "table#0",
                "category": "table",
                "kind": "root_surface",
                "support": None,
                "children": ["mug#0"],
                "mask_path": "masks/table.npy",
            },
            {
                "id": "mug#0",
                "category": "mug",
                "kind": "object",
                "support": "table#0",
                "parent": "table#0",
                "children": [],
                "mask_path": "masks/mug.npy",
            },
        ],
        "roots": ["table#0"],
        "relationships": [],
    }
    placement = {
        "objects": [
            {
                "category": "mug",
                "instance": 0,
                "mesh_name": "obj_mug_0",
                "mesh_glb": "meshes/mug.glb",
            }
        ]
    }
    (masks_dir / "masks.json").write_text(json.dumps(masks))
    (tmp_path / "scene_graph.json").write_text(json.dumps(graph))
    (tmp_path / "placement.json").write_text(json.dumps(placement))
    return masks, graph, placement


def _runtime_record(
    tmp_path: Path,
    *,
    transaction_id: int = 7,
    mesh_name: str = "obj_bowl_1",
    mesh_bytes: bytes = b"runtime bowl glTF",
) -> dict:
    object_dir = tmp_path / "runtime_objects" / "bowl_1"
    object_dir.mkdir(parents=True, exist_ok=True)
    np.save(object_dir / "mask.npy", np.ones((3, 5), dtype=bool))
    mesh = object_dir / "mesh-r001.glb"
    mesh.write_bytes(mesh_bytes)
    mask = object_dir / "mask.npy"
    mask_path = "runtime_objects/bowl_1/mask.npy"
    mesh_path = "runtime_objects/bowl_1/mesh-r001.glb"
    graph_node = {
        "id": "bowl#1",
        "category": "bowl",
        "kind": "object",
        "support": "table#0",
        "parent": "table#0",
        "children": [],
        "mask_path": mask_path,
        "runtime_added": True,
        "runtime_transaction_id": transaction_id,
    }
    placement = {
        "category": "bowl",
        "instance": 1,
        "mesh_name": mesh_name,
        "mesh_glb": mesh_path,
        "runtime_added": True,
        "runtime_transaction_id": transaction_id,
    }
    return build_runtime_object_record(
        transaction_id=transaction_id,
        category="bowl",
        instance=1,
        mask_path=mask_path,
        evidence={"reason": "large visible bowl is absent", "target_region": [0, 1]},
        graph_node=graph_node,
        placement=placement,
        mesh_sha256=sha256_file(mesh),
        mask_sha256=sha256_file(mask),
        physical_material=_physical_material(),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "reconstruction_glb"},
            "sha256": sha256_file(mesh),
        },
    )


def _maskless_record(
    tmp_path: Path,
    *,
    object_id: str = "bowl#1",
    support: str = "table#0",
    transaction_id: int = 7,
    mesh_name: str | None = None,
) -> dict:
    category, raw_instance = object_id.rsplit("#", 1)
    instance = int(raw_instance)
    resolved_mesh_name = mesh_name or f"obj_{category}_{instance}"
    object_dir = tmp_path / "runtime_objects" / f"{category}_{instance}"
    object_dir.mkdir(parents=True, exist_ok=True)
    mesh = object_dir / "mesh-r001.glb"
    mesh.write_bytes(f"authored {object_id} glTF".encode())
    mesh_path = f"runtime_objects/{category}_{instance}/mesh-r001.glb"
    graph_node = {
        "id": object_id,
        "category": category,
        "kind": "object",
        "support": support,
        "parent": support,
        "children": [],
        "mask_path": None,
        "runtime_added": True,
        "runtime_transaction_id": transaction_id,
    }
    placement = {
        "category": category,
        "instance": instance,
        "mesh_name": resolved_mesh_name,
        "mesh_glb": mesh_path,
        "runtime_added": True,
        "runtime_transaction_id": transaction_id,
    }
    return build_runtime_object_record(
        transaction_id=transaction_id,
        category=category,
        instance=instance,
        mask_path=None,
        mask_sha256=None,
        mask_policy="excluded",
        evidence={"source": "declared_initializer_addition"},
        graph_node=graph_node,
        placement=placement,
        mesh_sha256=sha256_file(mesh),
        physical_material=_physical_material(),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "authored_glb"},
            "sha256": sha256_file(mesh),
        },
    )


def _committed_runtime_scene(
    tmp_path: Path,
    *,
    legacy_tracked_mask: bool = False,
) -> tuple[dict, dict, dict, dict]:
    masks, graph, placement = _source_scene(tmp_path)
    masks_before = (tmp_path / "masks" / "masks.json").read_bytes()
    if legacy_tracked_mask:
        # Historical tracked inventories remain readable; new authored additions
        # go through the maskless batch API below.
        record = _runtime_record(tmp_path)
        overlay = new_runtime_inventory(tmp_path)
        overlay["objects"].append(record)
        graph["nodes"][0]["children"].append(record["id"])
        graph["nodes"].append(copy.deepcopy(record["graph_node"]))
        placement["objects"].append(copy.deepcopy(record["placement"]))
        staged = {
            "scene_graph": graph,
            "placement": placement,
            "runtime_inventory": overlay,
        }
    else:
        record = _maskless_record(tmp_path)
        staged = stage_runtime_object_batch(
            tmp_path,
            graph=graph,
            placement=placement,
            transaction_id=7,
            additions=[record],
            removed_object_ids=[],
        )
    assert (tmp_path / "masks" / "masks.json").read_bytes() == masks_before
    (tmp_path / "scene_graph.json").write_text(json.dumps(staged["scene_graph"]))
    (tmp_path / "placement.json").write_text(json.dumps(staged["placement"]))
    write_runtime_inventory(tmp_path, staged["runtime_inventory"])
    validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)
    return (
        masks,
        staged["scene_graph"],
        staged["placement"],
        staged["runtime_inventory"],
    )


def _write_physics_contract(tmp_path: Path, placement: dict, overlay: dict) -> None:
    physics_dir = tmp_path / "physics"
    physics_dir.mkdir(exist_ok=True)
    runtime = overlay["objects"][0]["physical_material"]
    fields = {
        key: runtime[key]
        for key in (
            "material",
            "mass_kg",
            "mass_range_kg",
            "friction",
            "density_kgm3",
            "density_gate",
        )
    }
    fields["mass_source"] = "runtime_test_estimate"
    names = [row["mesh_name"] for row in placement["objects"]]
    runtime_names = {
        str(record["placement"]["mesh_name"]) for record in overlay["objects"]
    }
    (physics_dir / "physics_vlm.json").write_text(json.dumps({"obj_bowl_1": fields}))
    (physics_dir / "physics_estimate_manifest.json").write_text(
        json.dumps(
            {
                "expected_count": len(names),
                "objects": {
                    name: {
                        "status": "cached"
                        if name in runtime_names
                        else "failed_fallback"
                    }
                    for name in names
                },
            }
        )
    )
    identity = [
        [1.0 if row == column else 0.0 for column in range(4)] for row in range(4)
    ]
    (physics_dir / "blend_base.json").write_text(
        json.dumps({name: identity for name in names})
    )
    (physics_dir / "pose_changes.json").write_text(
        json.dumps({"objects": {name: {"physics_overrides": None} for name in names}})
    )


def _write_source_physics_contract(tmp_path: Path, placement: dict) -> None:
    """Write a complete translator-consumable contract without a runtime overlay."""

    physics_dir = tmp_path / "physics"
    physics_dir.mkdir(exist_ok=True)
    names = [row["mesh_name"] for row in placement["objects"]]
    estimates = {
        name: {
            "material": "ceramic",
            "mass_kg": 0.4,
            "mass_range_kg": [0.2, 0.8],
            "mass_source": "test_estimate",
            "friction": 0.5,
            "density_kgm3": 1000.0,
            "density_gate": "ok",
        }
        for name in names
    }
    (physics_dir / "physics_vlm.json").write_text(json.dumps(estimates))
    (physics_dir / "physics_estimate_manifest.json").write_text(
        json.dumps(
            {
                "expected_count": len(names),
                "objects": {name: {"status": "cached"} for name in names},
            }
        )
    )
    identity = [
        [1.0 if row == column else 0.0 for column in range(4)] for row in range(4)
    ]
    (physics_dir / "blend_base.json").write_text(
        json.dumps({name: identity for name in names})
    )
    (physics_dir / "pose_changes.json").write_text(
        json.dumps({"objects": {name: {"physics_overrides": None} for name in names}})
    )


def test_legacy_scene_needs_no_runtime_overlay(tmp_path: Path) -> None:
    masks, graph, _ = _source_scene(tmp_path)

    report = validate_scene_artifacts(tmp_path)

    assert report["runtime_inventory_present"] is False
    assert report["runtime_object_ids"] == []
    assert validate_scene_graph_inventory(masks, graph)["runtime_root_ids"] == []


def test_legacy_runtime_root_surface_remains_allowed_without_overlay(
    tmp_path: Path,
) -> None:
    masks, graph, _ = _source_scene(tmp_path)
    graph["nodes"].append(
        {
            "id": "wall#1",
            "category": "wall",
            "kind": "root_surface",
            "support": None,
            "parent": None,
            "children": [],
            "runtime_added": True,
        }
    )
    graph["roots"].append("wall#1")

    report = validate_scene_graph_inventory(masks, graph)

    assert report["runtime_root_ids"] == ["wall#1"]
    assert report["runtime_object_ids"] == []
    assert "wall#1" in report["resolved_node_ids"]


def test_runtime_object_without_overlay_remains_forbidden(tmp_path: Path) -> None:
    masks, graph, _ = _source_scene(tmp_path)
    record = _runtime_record(tmp_path)
    graph["nodes"][0]["children"].append("bowl#1")
    graph["nodes"].append(record["graph_node"])

    with pytest.raises(
        InventoryContractError,
        match="runtime-added object bowl#1 has no committed runtime overlay record",
    ):
        validate_scene_graph_inventory(masks, graph)


@pytest.mark.parametrize("legacy_tracked_mask", [False, True])
def test_committed_overlay_resolves_without_mutating_source_masks(
    tmp_path: Path,
    legacy_tracked_mask: bool,
) -> None:
    masks, graph, _, overlay = _committed_runtime_scene(
        tmp_path, legacy_tracked_mask=legacy_tracked_mask
    )

    report = validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)
    resolved = resolved_inventory(masks, overlay)

    assert report["runtime_inventory_present"] is True
    assert report["runtime_object_ids"] == ["bowl#1"]
    assert report["object_mesh_names"] == {
        "bowl#1": "obj_bowl_1",
        "mug#0": "obj_mug_0",
    }
    assert resolved["bowl#1"]["source_section"] == "runtime_objects"
    assert (
        next(node for node in graph["nodes"] if node["id"] == "bowl#1")["parent"]
        == "table#0"
    )
    source_payload = json.loads((tmp_path / "masks" / "masks.json").read_text())
    assert [row["category"] for row in source_payload["instances"]] == ["table", "mug"]


def test_maskless_authored_record_stages_without_synthetic_mask(
    tmp_path: Path,
) -> None:
    masks, graph, placement = _source_scene(tmp_path)
    source_masks_before = (tmp_path / "masks" / "masks.json").read_bytes()
    record = _maskless_record(tmp_path)

    staged = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=7,
        additions=[record],
        removed_object_ids=[],
    )

    active = staged["runtime_inventory"]["objects"][0]
    assert active["mask_policy"] == "excluded"
    assert active["mask_path"] is None
    assert active["scoring_mask_path"] is None
    assert active["graph_node"]["mask_path"] is None
    assert "mask_path" not in active["placement"]
    assert (
        resolved_inventory(masks, staged["runtime_inventory"])["bowl#1"]["mask_path"]
        is None
    )
    assert (tmp_path / "masks" / "masks.json").read_bytes() == source_masks_before


def test_excluded_record_rejects_any_hidden_mask_binding(tmp_path: Path) -> None:
    _source_scene(tmp_path)
    record = _maskless_record(tmp_path)
    overlay = new_runtime_inventory(tmp_path)
    overlay["objects"].append(record)
    record["mask_path"] = "masks/mug.npy"
    record["mask_sha256"] = sha256_file(tmp_path / "masks" / "mug.npy")
    record["graph_node"]["mask_path"] = record["mask_path"]

    with pytest.raises(
        RuntimeObjectRepairError,
        match="excluded mask policy requires mask_path=null",
    ):
        validate_runtime_inventory_schema(overlay)


def test_atomic_maskless_batch_adds_support_and_child_in_any_order(
    tmp_path: Path,
) -> None:
    _, graph, placement = _source_scene(tmp_path)
    stand = _maskless_record(tmp_path, object_id="stand#0")
    cup = _maskless_record(tmp_path, object_id="cup#0", support="stand#0")

    staged = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=7,
        additions=[cup, stand],
        removed_object_ids=[],
    )

    nodes = {row["id"]: row for row in staged["scene_graph"]["nodes"]}
    assert nodes["stand#0"]["children"] == ["cup#0"]
    assert nodes["cup#0"]["parent"] == "stand#0"
    assert {row["id"] for row in staged["runtime_inventory"]["objects"]} == {
        "cup#0",
        "stand#0",
    }


def test_failed_batch_is_pure_and_tombstoned_identity_cannot_resurrect(
    tmp_path: Path,
) -> None:
    _, graph, placement = _source_scene(tmp_path)
    graph_before = copy.deepcopy(graph)
    placement_before = copy.deepcopy(placement)
    orphan = _maskless_record(tmp_path, object_id="cup#0", support="ghost#0")
    with pytest.raises(RuntimeObjectRepairError, match="missing parent 'ghost#0'"):
        stage_runtime_object_batch(
            tmp_path,
            graph=graph,
            placement=placement,
            transaction_id=8,
            additions=[orphan],
            removed_object_ids=[],
        )
    assert graph == graph_before
    assert placement == placement_before

    removed = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=9,
        additions=[],
        removed_object_ids=["mug#0"],
    )
    replacement = _maskless_record(
        tmp_path, object_id="mug#0", transaction_id=10, mesh_name="obj_mug_0"
    )
    with pytest.raises(RuntimeObjectRepairError, match="cannot be reused"):
        stage_runtime_object_batch(
            tmp_path,
            graph=removed["scene_graph"],
            placement=removed["placement"],
            runtime_inventory=removed["runtime_inventory"],
            transaction_id=10,
            additions=[replacement],
            removed_object_ids=[],
        )


def test_runtime_record_requires_committed_transaction_evidence_and_materials(
    tmp_path: Path,
) -> None:
    _source_scene(tmp_path)
    record = _runtime_record(tmp_path)
    overlay = new_runtime_inventory(tmp_path)
    overlay["objects"].append(record)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["transaction_id"] = None
    with pytest.raises(RuntimeObjectRepairError, match="transaction_id"):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["evidence"] = {}
    with pytest.raises(RuntimeObjectRepairError, match="evidence"):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["physical_material"].pop("provenance")
    with pytest.raises(RuntimeObjectRepairError, match="physical_material"):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["physical_material"].pop("mass_kg")
    with pytest.raises(RuntimeObjectRepairError, match="mass_kg"):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["visual_material"].pop("sha256")
    with pytest.raises(RuntimeObjectRepairError, match="visual_material"):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0]["visual_material"]["sha256"] = "f" * 64
    with pytest.raises(
        RuntimeObjectRepairError,
        match="visual_material sha256 does not match the current committed mesh",
    ):
        validate_runtime_inventory_schema(broken)

    broken = copy.deepcopy(overlay)
    broken["objects"][0].pop("mask_sha256")
    with pytest.raises(RuntimeObjectRepairError, match="mask_sha256"):
        validate_runtime_inventory_schema(broken)


def test_runtime_record_accepts_explicit_physics_fallback(tmp_path: Path) -> None:
    _source_scene(tmp_path)
    record = _runtime_record(tmp_path)
    record["physical_material"] = _physical_material("fallback", "plastic")
    overlay = new_runtime_inventory(tmp_path)
    overlay["objects"].append(record)

    assert list(validate_runtime_inventory_schema(overlay)) == ["bowl#1"]


def test_runtime_overlay_requires_live_graph_and_placement_bindings(
    tmp_path: Path,
) -> None:
    _, graph, placement, _ = _committed_runtime_scene(tmp_path)
    placement["objects"] = [
        row for row in placement["objects"] if row["category"] != "bowl"
    ]
    (tmp_path / "placement.json").write_text(json.dumps(placement))
    with pytest.raises(InventoryContractError, match="missing placement rows: bowl#1"):
        validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)

    missing_graph_scene = tmp_path / "missing_graph"
    _, graph, placement = _source_scene(missing_graph_scene)
    record = _runtime_record(missing_graph_scene)
    overlay = new_runtime_inventory(missing_graph_scene)
    overlay["objects"].append(record)
    write_runtime_inventory(missing_graph_scene, overlay)
    placement["objects"].append(record["placement"])
    (missing_graph_scene / "placement.json").write_text(json.dumps(placement))
    with pytest.raises(
        InventoryContractError, match="committed runtime objects missing graph nodes"
    ):
        validate_scene_artifacts(missing_graph_scene, allow_runtime_inventory=True)


def test_resolved_materialization_rejects_duplicate_mesh_name(tmp_path: Path) -> None:
    masks, graph, placement = _source_scene(tmp_path)
    record = _runtime_record(tmp_path, mesh_name="obj_mug_0")
    overlay = new_runtime_inventory(tmp_path)
    overlay["objects"].append(record)
    graph["nodes"][0]["children"].append("bowl#1")
    graph["nodes"].append(record["graph_node"])
    placement["objects"].append(record["placement"])
    (tmp_path / "scene_graph.json").write_text(json.dumps(graph))
    (tmp_path / "placement.json").write_text(json.dumps(placement))
    write_runtime_inventory(tmp_path, overlay)

    with pytest.raises(InventoryContractError, match="mesh_name 'obj_mug_0' is shared"):
        validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)
    assert "bowl#1" in resolved_inventory(masks, overlay)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty", "mesh_glb is empty"),
        ("outside", "outside backend-owned runtime_objects"),
        ("extension", "current mesh is not a GLB"),
    ],
)
def test_current_mesh_revision_is_nonempty_owned_glb(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _, _, _, overlay = _committed_runtime_scene(tmp_path)
    record = overlay["objects"][0]
    revision = record["mesh_revisions"][0]
    mesh = tmp_path / revision["mesh_glb"]
    if mutation == "empty":
        mesh.write_bytes(b"")
        revision["sha256"] = sha256_file(mesh)
    elif mutation == "outside":
        external = tmp_path / "external.glb"
        external.write_bytes(b"external")
        revision["mesh_glb"] = str(external)
        revision["sha256"] = sha256_file(external)
        record["placement"]["mesh_glb"] = str(external)
        placement = json.loads((tmp_path / "placement.json").read_text())
        placement["objects"][-1]["mesh_glb"] = str(external)
        (tmp_path / "placement.json").write_text(json.dumps(placement))
    else:
        wrong_type = mesh.with_suffix(".obj")
        wrong_type.write_bytes(b"not really an obj")
        revision["mesh_glb"] = str(wrong_type)
        revision["sha256"] = sha256_file(wrong_type)
        record["placement"]["mesh_glb"] = str(wrong_type)
        placement = json.loads((tmp_path / "placement.json").read_text())
        placement["objects"][-1]["mesh_glb"] = str(wrong_type)
        (tmp_path / "placement.json").write_text(json.dumps(placement))
    # Keep material provenance internally bound while isolating the artifact failure
    # each parameterized case intends to exercise.
    record["visual_material"]["sha256"] = revision["sha256"]
    write_runtime_inventory(tmp_path, overlay)

    with pytest.raises(InventoryContractError, match=message):
        validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)


def test_runtime_material_is_bound_to_canonical_usd_physics(tmp_path: Path) -> None:
    _, _, placement, overlay = _committed_runtime_scene(tmp_path)
    _write_physics_contract(tmp_path, placement, overlay)

    validate_scene_artifacts(tmp_path, validate_runtime_physics=True)

    physics_path = tmp_path / "physics" / "physics_vlm.json"
    physics = json.loads(physics_path.read_text())
    physics["obj_bowl_1"]["friction"] = 0.1
    physics_path.write_text(json.dumps(physics))
    with pytest.raises(
        InventoryContractError, match="physical_material.friction differs"
    ):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


def test_no_overlay_still_validates_canonical_usd_physics_identity(
    tmp_path: Path,
) -> None:
    _, _, placement = _source_scene(tmp_path)
    _write_source_physics_contract(tmp_path, placement)

    validate_scene_artifacts(tmp_path, validate_runtime_physics=True)

    physics_path = tmp_path / "physics" / "physics_vlm.json"
    physics = json.loads(physics_path.read_text())
    physics["obj_ghost_9"] = copy.deepcopy(physics["obj_mug_0"])
    physics_path.write_text(json.dumps(physics))
    with pytest.raises(
        InventoryContractError,
        match="physics estimates contain stale placement objects: obj_ghost_9",
    ):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


@pytest.mark.parametrize(
    ("relative", "payload", "message"),
    [
        ("pose_changes.json", "{", "physics pose changes is unreadable"),
        ("blend_base.json", "[]", "physics Blend base is not a JSON object"),
    ],
)
def test_canonical_usd_physics_rejects_malformed_required_json(
    tmp_path: Path, relative: str, payload: str, message: str
) -> None:
    _, _, placement = _source_scene(tmp_path)
    _write_source_physics_contract(tmp_path, placement)
    (tmp_path / "physics" / relative).write_text(payload)

    with pytest.raises(InventoryContractError, match=message):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


@pytest.mark.parametrize("relative", ["pose_changes.json", "blend_base.json"])
def test_canonical_usd_physics_rejects_stale_pose_identity(
    tmp_path: Path, relative: str
) -> None:
    _, _, placement = _source_scene(tmp_path)
    _write_source_physics_contract(tmp_path, placement)
    path = tmp_path / "physics" / relative
    payload = json.loads(path.read_text())
    records = payload["objects"] if relative == "pose_changes.json" else payload
    records["obj_ghost_9"] = records.pop("obj_mug_0")
    path.write_text(json.dumps(payload))

    with pytest.raises(InventoryContractError, match="stale objects: obj_ghost_9"):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


def test_runtime_addition_need_not_have_preprocess_pose_history(tmp_path: Path) -> None:
    _, _, placement, overlay = _committed_runtime_scene(tmp_path)
    _write_physics_contract(tmp_path, placement, overlay)
    pose_path = tmp_path / "physics" / "pose_changes.json"
    pose = json.loads(pose_path.read_text())
    pose["objects"].pop("obj_bowl_1")
    pose_path.write_text(json.dumps(pose))

    validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


def test_canonical_usd_physics_requires_translator_fields(tmp_path: Path) -> None:
    _, _, placement = _source_scene(tmp_path)
    _write_source_physics_contract(tmp_path, placement)
    physics_path = tmp_path / "physics" / "physics_vlm.json"
    physics = json.loads(physics_path.read_text())
    physics["obj_mug_0"].pop("mass_source")
    physics_path.write_text(json.dumps(physics))

    with pytest.raises(InventoryContractError, match="no nonempty mass_source"):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


def test_runtime_physics_requires_manifest_claimed_canonical_estimate(
    tmp_path: Path,
) -> None:
    _, _, placement, overlay = _committed_runtime_scene(tmp_path)
    _write_physics_contract(tmp_path, placement, overlay)
    manifest_path = tmp_path / "physics" / "physics_estimate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["objects"]["obj_mug_0"]["status"] = "cached"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        InventoryContractError,
        match="marks obj_mug_0 as cached but physics_vlm has no canonical object entry",
    ):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


def test_runtime_physics_rejects_stale_canonical_estimate(tmp_path: Path) -> None:
    _, _, placement, overlay = _committed_runtime_scene(tmp_path)
    _write_physics_contract(tmp_path, placement, overlay)
    physics_path = tmp_path / "physics" / "physics_vlm.json"
    estimates = json.loads(physics_path.read_text())
    estimates["obj_removed_0"] = copy.deepcopy(estimates["obj_bowl_1"])
    physics_path.write_text(json.dumps(estimates))

    with pytest.raises(
        InventoryContractError,
        match="runtime physics estimates contain stale placement objects: obj_removed_0",
    ):
        validate_scene_artifacts(tmp_path, validate_runtime_physics=True)


@pytest.mark.parametrize("legacy_tracked_mask", [False, True])
def test_mesh_revision_staging_updates_overlay_and_placement_only_in_copy(
    tmp_path: Path,
    legacy_tracked_mask: bool,
) -> None:
    _, graph, placement, overlay = _committed_runtime_scene(
        tmp_path, legacy_tracked_mask=legacy_tracked_mask
    )
    replacement = tmp_path / "runtime_objects" / "bowl_1" / "mesh-r002.glb"
    replacement.write_bytes(b"corrected bowl glTF")
    placement_before = copy.deepcopy(placement)
    overlay_before = copy.deepcopy(overlay)

    staged = stage_runtime_mesh_revision(
        tmp_path,
        graph=graph,
        placement=placement,
        object_id="bowl#1",
        transaction_id=8,
        mesh_glb="runtime_objects/bowl_1/mesh-r002.glb",
        mesh_sha256=sha256_file(replacement),
        runtime_inventory=overlay,
        physical_material=_physical_material(),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "corrected_glb"},
            "sha256": sha256_file(replacement),
        },
    )

    assert placement == placement_before
    assert overlay == overlay_before
    revised = staged["runtime_inventory"]["objects"][0]
    assert revised["current_mesh_revision"] == 2
    assert [row["status"] for row in revised["mesh_revisions"]] == [
        "superseded",
        "committed",
    ]
    assert staged["placement"]["objects"][-1]["mesh_glb"].endswith("mesh-r002.glb")


def test_mesh_revision_requires_refreshed_visual_material_digest(
    tmp_path: Path,
) -> None:
    _, graph, placement, overlay = _committed_runtime_scene(tmp_path)
    replacement = tmp_path / "runtime_objects" / "bowl_1" / "mesh-r002.glb"
    replacement.write_bytes(b"corrected bowl glTF")

    with pytest.raises(
        RuntimeObjectRepairError,
        match="visual_material sha256 does not match the current committed mesh",
    ):
        stage_runtime_mesh_revision(
            tmp_path,
            graph=graph,
            placement=placement,
            object_id="bowl#1",
            transaction_id=8,
            mesh_glb="runtime_objects/bowl_1/mesh-r002.glb",
            mesh_sha256=sha256_file(replacement),
            runtime_inventory=overlay,
            physical_material=_physical_material(),
        )


def test_first_source_mesh_revision_materializes_overlay_without_replacing_identity(
    tmp_path: Path,
) -> None:
    masks, graph, placement = _source_scene(tmp_path)
    source_masks_before = (tmp_path / "masks" / "masks.json").read_bytes()
    replacement_dir = tmp_path / "runtime_objects" / "mug_0"
    replacement_dir.mkdir(parents=True)
    replacement = replacement_dir / "mesh-r001.glb"
    replacement.write_bytes(b"corrected source mug glTF")

    staged = stage_runtime_mesh_revision(
        tmp_path,
        graph=graph,
        placement=placement,
        object_id="mug#0",
        transaction_id=9,
        mesh_glb="runtime_objects/mug_0/mesh-r001.glb",
        mesh_sha256=sha256_file(replacement),
        replacement_placement={
            **placement["objects"][0],
            "mask_path": str((tmp_path / "masks" / "mug.npy").resolve()),
        },
        evidence={"reason": "source mesh has a missing handle"},
        physical_material=_physical_material(),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "corrected_glb"},
            "sha256": sha256_file(replacement),
        },
    )

    record = staged["runtime_inventory"]["objects"][0]
    assert record["origin"] == "source_revised"
    assert record["mask_path"] == "masks/mug.npy"
    assert record["scoring_mask_path"] == "masks/mug.npy"
    assert record["placement"]["mask_path"] == "masks/mug.npy"
    assert record["mesh_revisions"][0]["provenance"] == {"source": "preprocess"}
    assert record["mesh_revisions"][0]["transaction_id"] is None
    assert record["mesh_revisions"][1]["status"] == "committed"
    mug_node = next(
        node for node in staged["scene_graph"]["nodes"] if node["id"] == "mug#0"
    )
    assert mug_node.get("runtime_added") is not True
    assert (tmp_path / "masks" / "masks.json").read_bytes() == source_masks_before

    (tmp_path / "scene_graph.json").write_text(json.dumps(staged["scene_graph"]))
    (tmp_path / "placement.json").write_text(json.dumps(staged["placement"]))
    write_runtime_inventory(tmp_path, staged["runtime_inventory"])
    report = validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)
    assert report["runtime_added_object_ids"] == []
    assert report["runtime_revised_object_ids"] == ["mug#0"]
    resolved = resolved_inventory(masks, staged["runtime_inventory"])
    assert resolved["mug#0"]["source_section"] == "instances"
    assert resolved["mug#0"]["mesh_glb"].endswith("mesh-r001.glb")


def test_direct_source_mesh_revision_excludes_stale_source_mask(
    tmp_path: Path,
) -> None:
    masks, graph, placement = _source_scene(tmp_path)
    replacement_dir = tmp_path / "runtime_objects" / "mug_0"
    replacement_dir.mkdir(parents=True)
    replacement = replacement_dir / "mesh-r001.glb"
    replacement.write_bytes(b"directly authored mug")

    staged = stage_runtime_mesh_revision(
        tmp_path,
        graph=graph,
        placement=placement,
        object_id="mug#0",
        transaction_id=9,
        mesh_glb="runtime_objects/mug_0/mesh-r001.glb",
        mesh_sha256=sha256_file(replacement),
        evidence={"reason": "direct primitive reconstruction"},
        physical_material=_physical_material(),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "authored_glb"},
            "sha256": sha256_file(replacement),
        },
        mask_policy="excluded",
    )

    record = staged["runtime_inventory"]["objects"][0]
    assert record["mask_policy"] == "excluded"
    assert record["mask_path"] is None
    assert record["source_mask_path"] == "masks/mug.npy"
    assert record["source_mask_sha256"] == sha256_file(tmp_path / "masks" / "mug.npy")
    assert (
        resolved_inventory(masks, staged["runtime_inventory"])["mug#0"]["mask_path"]
        is None
    )


@pytest.mark.parametrize("legacy_tracked_mask", [False, True])
def test_runtime_object_removal_is_pure_and_preserves_artifacts_for_undo(
    tmp_path: Path,
    legacy_tracked_mask: bool,
) -> None:
    _, graph, placement, overlay = _committed_runtime_scene(
        tmp_path, legacy_tracked_mask=legacy_tracked_mask
    )
    graph_before = copy.deepcopy(graph)
    placement_before = copy.deepcopy(placement)
    overlay_before = copy.deepcopy(overlay)
    mesh = tmp_path / overlay["objects"][0]["placement"]["mesh_glb"]

    staged = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=10,
        additions=[],
        removed_object_ids=["bowl#1"],
        removal_evidence={"bowl#1": {"reason": "addition duplicated a source object"}},
        runtime_inventory=overlay,
    )

    assert graph == graph_before
    assert placement == placement_before
    assert overlay == overlay_before
    assert staged["runtime_inventory"]["objects"] == []
    tombstone = staged["runtime_inventory"]["tombstones"][0]
    assert tombstone["origin"] == "runtime_removed"
    assert tombstone["prior_runtime_record"] == overlay["objects"][0]
    assert [row["category"] for row in staged["placement"]["objects"]] == ["mug"]
    assert staged["scene_graph"]["nodes"][0]["children"] == ["mug#0"]
    assert mesh.is_file()


def test_source_revised_removal_persists_exact_reversible_tombstone(
    tmp_path: Path,
) -> None:
    _, graph, placement = _source_scene(tmp_path)
    replacement_dir = tmp_path / "runtime_objects" / "mug_0"
    replacement_dir.mkdir(parents=True)
    replacement = replacement_dir / "mesh.glb"
    replacement.write_bytes(b"revised mug")
    staged = stage_runtime_mesh_revision(
        tmp_path,
        graph=graph,
        placement=placement,
        object_id="mug#0",
        transaction_id=11,
        mesh_glb="runtime_objects/mug_0/mesh.glb",
        mesh_sha256=sha256_file(replacement),
        evidence={"reason": "mesh silhouette is wrong"},
        physical_material=_physical_material("fallback", "plastic"),
        visual_material={
            "status": "embedded",
            "provenance": {"source": "preserved"},
            "sha256": sha256_file(replacement),
        },
    )

    old_node = copy.deepcopy(staged["scene_graph"]["nodes"][1])
    old_placement = copy.deepcopy(staged["placement"]["objects"][0])
    old_record = copy.deepcopy(staged["runtime_inventory"]["objects"][0])
    removed = stage_runtime_object_batch(
        tmp_path,
        graph=staged["scene_graph"],
        placement=staged["placement"],
        transaction_id=12,
        additions=[],
        removed_object_ids=["mug#0"],
        removal_evidence={"mug#0": {"reason": "not actually a mug"}},
        runtime_inventory=staged["runtime_inventory"],
    )

    assert removed["runtime_inventory"]["objects"] == []
    tombstone = runtime_inventory_tombstones(removed["runtime_inventory"])["mug#0"]
    assert tombstone["origin"] == "source_removed"
    assert tombstone["graph_node"] == old_node
    assert tombstone["placement"] == old_placement
    assert tombstone["prior_runtime_record"] == old_record
    assert "mug#0" not in {row["id"] for row in removed["scene_graph"]["nodes"]}
    assert "mug#0" not in resolved_inventory(
        json.loads((tmp_path / "masks" / "masks.json").read_text()),
        removed["runtime_inventory"],
    )


def test_source_replacement_is_maskless_and_preserves_revision_history(
    tmp_path: Path,
) -> None:
    _, graph, placement = _source_scene(tmp_path)
    replacement = _maskless_record(
        tmp_path,
        object_id="mug#0",
        transaction_id=13,
        mesh_name="obj_mug_0",
    )

    staged = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=13,
        additions=[replacement],
        removed_object_ids=["mug#0"],
    )

    record = staged["runtime_inventory"]["objects"][0]
    assert record["id"] == "mug#0"
    assert record["origin"] == "source_revised"
    assert record["mask_policy"] == "excluded"
    assert record["mask_path"] is None
    assert record["source_mask_path"] == "masks/mug.npy"
    assert record["current_mesh_revision"] == 2
    assert [row["status"] for row in record["mesh_revisions"]] == [
        "superseded",
        "committed",
    ]
    assert staged["runtime_inventory"]["tombstones"] == []
    node = next(row for row in staged["scene_graph"]["nodes"] if row["id"] == "mug#0")
    assert node["mask_path"] is None
    assert node.get("runtime_added") is not True


def test_pure_source_removal_is_tombstoned_and_source_masks_stay_immutable(
    tmp_path: Path,
) -> None:
    masks, graph, placement = _source_scene(tmp_path)
    source_relationship = {
        "relationship_id": "mug-beside-table",
        "a": "mug#0",
        "b": "table#0",
        "relation": "beside",
        "enforcement": "hard",
    }
    masks["relationships"].append(source_relationship)
    graph["relationships"].append(copy.deepcopy(source_relationship))
    (tmp_path / "masks" / "masks.json").write_text(json.dumps(masks))
    source_masks_before = (tmp_path / "masks" / "masks.json").read_bytes()

    staged = stage_runtime_object_batch(
        tmp_path,
        graph=graph,
        placement=placement,
        transaction_id=14,
        additions=[],
        removed_object_ids=["mug#0"],
        removal_evidence={"mug#0": {"reason": "declared duplicate"}},
    )

    assert staged["runtime_inventory"]["objects"] == []
    tombstone = staged["runtime_inventory"]["tombstones"][0]
    assert tombstone["id"] == "mug#0"
    assert tombstone["origin"] == "source_removed"
    assert tombstone["placement"] == placement["objects"][0]
    assert staged["scene_graph"]["relationships"] == []
    assert resolved_inventory(masks, staged["runtime_inventory"]) == {
        "table#0": resolved_inventory(masks)["table#0"]
    }
    assert (tmp_path / "masks" / "masks.json").read_bytes() == source_masks_before
    (tmp_path / "scene_graph.json").write_text(json.dumps(staged["scene_graph"]))
    (tmp_path / "placement.json").write_text(json.dumps(staged["placement"]))
    write_runtime_inventory(tmp_path, staged["runtime_inventory"])
    report = validate_scene_artifacts(tmp_path, allow_runtime_inventory=True)
    assert report["source_node_ids"] == ["table#0"]
    assert report["removed_source_node_ids"] == ["mug#0"]
    assert report["tombstoned_object_ids"] == ["mug#0"]


def test_allow_runtime_additions_false_rejects_overlay_object(tmp_path: Path) -> None:
    _committed_runtime_scene(tmp_path)

    with pytest.raises(
        InventoryContractError, match="runtime-added graph nodes are not allowed"
    ):
        validate_scene_artifacts(
            tmp_path,
            allow_runtime_additions=False,
            allow_runtime_inventory=True,
        )


def test_baseline_safe_default_rejects_runtime_overlay(tmp_path: Path) -> None:
    _committed_runtime_scene(tmp_path)

    with pytest.raises(
        InventoryContractError, match="runtime object inventory is not allowed"
    ):
        validate_scene_artifacts(tmp_path)
