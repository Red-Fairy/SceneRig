"""Focused checks for crash-safe GPT-6 initializer object transactions."""

import hashlib
import importlib.util
import json
import multiprocessing
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Developer/test checkouts intentionally omit ignored local credentials.
if (
    "lib.utils._api_keys" not in sys.modules
    and importlib.util.find_spec("lib.utils._api_keys") is None
):
    api_keys = types.ModuleType("lib.utils._api_keys")
    for provider in ("CLAUDE", "FIREWORKS", "GEMINI", "OPENAI", "QWEN"):
        setattr(api_keys, f"{provider}_API_KEY", "")
        setattr(api_keys, f"{provider}_BASE_URL", "")
    sys.modules["lib.utils._api_keys"] = api_keys

from lib.tools.blender.exec import (
    Executor,
)


def _hold_initializer_scene_lock(scene_dir, acquired, release):
    """Process target proving the scene lock is inter-process, not thread-local."""
    executor = Executor.__new__(Executor)
    executor.moge_dir = scene_dir
    fd = executor._acquire_initializer_scene_lock()
    acquired.set()
    try:
        if not release.wait(10):
            raise RuntimeError("test did not release initializer scene lock")
    finally:
        executor._release_initializer_scene_lock(fd)


def _bare_executor(tmp_path, *, load_ledger: bool = False) -> Executor:
    scene = tmp_path / "scene"
    scene.mkdir(exist_ok=True)
    live = tmp_path / "live.blend"
    if not live.exists():
        live.write_bytes(b"blend-before")
    ledger_path = tmp_path / "ledger.json"

    executor = Executor.__new__(Executor)
    executor.root_stage_name = "initializer"
    executor.harness_profile = "gpt6_v1"
    executor.harness_profile_manifest = {
        "capabilities": {
            "mutation_journal": True,
            "initializer_code_transactions": True,
        }
    }
    executor.moge_dir = str(scene)
    executor.blender_save = str(live)
    executor.initializer_ledger_path = str(ledger_path)
    executor._initializer_ledger = (
        executor._load_initializer_ledger()
        if load_ledger
        else executor._empty_initializer_ledger()
    )
    executor._pipeline_object_names = ["obj_mug_0"]
    executor._n2id_cache = None
    executor.attempt_idx = 1
    executor.base_state = "already-captured"
    executor.edit_history = []
    executor._ledger_history = []
    executor._graph_history = []
    executor._edit_meta = []
    executor._initializer_constraint_artifact = None
    return executor


def _world_semantics(**overrides) -> dict:
    detail = {
        "schema_version": 1,
        "world_coordinate_decimals": 5,
        "part_count": 1,
        "mesh_count": 1,
        "vertex_count": 8,
        "edge_count": 12,
        "polygon_count": 6,
        "geometry_sha256": "1" * 64,
        "topology_sha256": "2" * 64,
        "hierarchy_sha256": "a" * 64,
        "material_sha256": "3" * 64,
        "modifier_sha256": "4" * 64,
        "visibility_sha256": "5" * 64,
    }
    detail.update(overrides)
    raw = json.dumps(detail, sort_keys=True, separators=(",", ":"))
    detail["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return detail


def _integrity_signature(
    local_digest: str,
    *,
    content_digest: str | None = None,
    matrix_x: float = 0.0,
    world_semantics: dict | None = None,
) -> dict:
    return {
        "name": "object",
        "matrix": [
            [1.0, 0.0, 0.0, matrix_x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "scale": [1.0, 1.0, 1.0],
        "parent": None,
        "hide_render": False,
        "hide_viewport": False,
        "integrity_sha256": local_digest * 64,
        **(
            {"content_sha256": content_digest * 64}
            if content_digest is not None
            else {}
        ),
        **({"world_semantics": world_semantics} if world_semantics is not None else {}),
    }


def test_typed_validation_accepts_origin_recenter_with_same_world_semantics(tmp_path):
    executor = _bare_executor(tmp_path)
    semantic = _world_semantics()
    pre = {
        "object_integrity": {
            "obj_bottle_0": _integrity_signature(
                "a", matrix_x=0.0, world_semantics=semantic
            )
        }
    }
    post = {
        "object_integrity": {
            # PoseSession moved the origin/local vertices, not the rendered geometry.
            "obj_bottle_0": _integrity_signature(
                "b", matrix_x=0.77, world_semantics=semantic
            ),
            "obj_monitor_1": _integrity_signature(
                "c", matrix_x=1.2, world_semantics=_world_semantics()
            ),
        }
    }

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_monitor_1"] = _integrity_signature("c")
    touched = executor._validate_typed_initializer_result(
        pre,
        post,
        target="obj_monitor_1",
        physics_reports=[{"name": "obj_monitor_1", "members": ["obj_monitor_1"]}],
    )

    assert touched["obj_bottle_0"]["representation_only"] is True
    assert touched["obj_bottle_0"]["before_matrix"][0][3] == 0.0
    assert touched["obj_bottle_0"]["after_matrix"][0][3] == 0.77
    assert "representation_only" not in touched["obj_monitor_1"]


def test_typed_validation_accepts_real_sub_micrometre_rounding_jitter(
    tmp_path, monkeypatch
):
    executor = _bare_executor(tmp_path)
    before_semantics = _world_semantics()
    after_semantics = _world_semantics(geometry_sha256="6" * 64)
    before = _integrity_signature(
        "a",
        content_digest="d",
        matrix_x=0.07025700807571411,
        world_semantics=before_semantics,
    )
    after = _integrity_signature(
        # Real toast-standard collateral delta: 0.35762786865234375 micrometres.
        "c",
        content_digest="d",
        matrix_x=0.07025736570358276,
        world_semantics=after_semantics,
    )
    pre = {"object_integrity": {"obj_bread_2": before}}
    post = {
        "object_integrity": {
            "obj_bread_2": after,
            "obj_bread_3": _integrity_signature(
                "b", matrix_x=0.4, world_semantics=_world_semantics()
            ),
        }
    }
    pre_blend = tmp_path / "transaction-before.blend"
    post_blend = tmp_path / "post-settle.blend"
    proof_before = {
        "obj_bread_2": {
            "part_names": ["obj_bread_2"],
            "vertex_counts": np.asarray([2], dtype=np.int64),
            "vertices": np.asarray(
                [[0.07025700807571411, 0.0, 0.0], [0.1, 0.2, 0.3]],
                dtype=np.float64,
            ),
        }
    }
    proof_after = {
        "obj_bread_2": {
            "part_names": ["obj_bread_2"],
            "vertex_counts": np.asarray([2], dtype=np.int64),
            "vertices": np.asarray(
                [[0.07025736570358276, 0.0, 0.0], [0.1, 0.2, 0.3]],
                dtype=np.float64,
            ),
        }
    }
    calls = []

    def prove(before_path, after_path, names):
        calls.append((before_path, after_path, names))
        return Executor._compare_typed_world_vertex_records(
            proof_before, proof_after, names
        )

    monkeypatch.setattr(executor, "_typed_world_vertex_equivalence", prove)

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_bread_3"] = _integrity_signature("c")
    touched = executor._validate_typed_initializer_result(
        pre,
        post,
        target="obj_bread_3",
        pre_blend_path=pre_blend,
        post_blend_path=post_blend,
    )

    assert calls == [(pre_blend, post_blend, ["obj_bread_2"])]
    assert touched["obj_bread_2"]["numerical_jitter"] is True
    assert "representation_only" not in touched["obj_bread_2"]
    assert Executor._matrix_error(
        touched["obj_bread_2"]["before_matrix"],
        touched["obj_bread_2"]["after_matrix"],
    ) == pytest.approx(3.5762786865234375e-7)
    assert touched["obj_bread_2"]["world_vertex_equivalence"] == {
        "schema_version": 1,
        "method": "trusted_corresponding_world_vertices",
        "max_abs_error_m": pytest.approx(3.5762786865234375e-7),
        "tolerance_m": 2e-5,
        "vertex_count": 2,
        "part_count": 1,
    }


def test_typed_world_vertex_proof_accepts_boundary_and_rejects_above_tolerance():
    before = {
        "obj_bread_2": {
            "part_names": ["obj_bread_2"],
            "vertex_counts": np.asarray([1]),
            "vertices": np.asarray([[0.0, 0.0, 0.0]]),
        }
    }
    at_boundary = {
        "obj_bread_2": {
            "part_names": ["obj_bread_2"],
            "vertex_counts": np.asarray([1]),
            "vertices": np.asarray([[2e-5, 0.0, 0.0]]),
        }
    }
    above = {
        "obj_bread_2": {
            "part_names": ["obj_bread_2"],
            "vertex_counts": np.asarray([1]),
            "vertices": np.asarray([[2.0001e-5, 0.0, 0.0]]),
        }
    }

    accepted = Executor._compare_typed_world_vertex_records(
        before, at_boundary, ["obj_bread_2"]
    )
    assert accepted["obj_bread_2"]["max_abs_error_m"] == pytest.approx(2e-5)
    with pytest.raises(RuntimeError, match="world-vertex error.*exceeds"):
        Executor._compare_typed_world_vertex_records(before, above, ["obj_bread_2"])


def test_typed_validation_rejects_hierarchy_change_before_world_probe(
    tmp_path, monkeypatch
):
    executor = _bare_executor(tmp_path)
    pre = {
        "object_integrity": {
            "obj_bread_2": _integrity_signature(
                "a",
                content_digest="d",
                matrix_x=0.07,
                world_semantics=_world_semantics(),
            )
        }
    }
    post = {
        "object_integrity": {
            "obj_bread_2": _integrity_signature(
                "c",
                content_digest="e",
                matrix_x=0.07000036,
                world_semantics=_world_semantics(
                    geometry_sha256="6" * 64,
                    hierarchy_sha256="b" * 64,
                ),
            ),
            "obj_bread_3": _integrity_signature(
                "b", matrix_x=0.4, world_semantics=_world_semantics()
            ),
        }
    }
    monkeypatch.setattr(
        executor,
        "_typed_world_vertex_equivalence",
        lambda *_args, **_kwargs: pytest.fail("world proof must not run"),
    )

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_bread_3"] = _integrity_signature("c")
    with pytest.raises(RuntimeError, match="unexpectedly changed obj_bread_2"):
        executor._validate_typed_initializer_result(
            pre,
            post,
            target="obj_bread_3",
            pre_blend_path=tmp_path / "before.blend",
            post_blend_path=tmp_path / "after.blend",
        )


@pytest.mark.parametrize(
    ("semantic_change", "changed_value"),
    [
        ("geometry_sha256", "6" * 64),  # actual world movement/vertex change
        ("topology_sha256", "7" * 64),  # mesh connectivity/UV change
        ("hierarchy_sha256", "a" * 63 + "b"),
        ("material_sha256", "8" * 64),
        ("modifier_sha256", "b" * 64),
        ("visibility_sha256", "9" * 64),
    ],
)
def test_typed_validation_rejects_non_target_world_semantic_changes(
    tmp_path, semantic_change, changed_value
):
    executor = _bare_executor(tmp_path)
    pre = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature("a", world_semantics=_world_semantics()),
            "obj_bottle_0": _integrity_signature(
                "b", world_semantics=_world_semantics()
            ),
        }
    }
    post = {
        "object_integrity": {
            "obj_mug_0": pre["object_integrity"]["obj_mug_0"],
            "obj_bottle_0": _integrity_signature(
                "c",
                matrix_x=0.02,
                world_semantics=_world_semantics(**{semantic_change: changed_value}),
            ),
        }
    }

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_mug_0"] = _integrity_signature("c")
    with pytest.raises(RuntimeError, match="unexpectedly changed obj_bottle_0"):
        executor._validate_typed_initializer_result(
            pre,
            post,
            target="obj_mug_0",
            physics_reports=[{"name": "obj_mug_0", "members": ["obj_mug_0"]}],
        )


def test_typed_world_semantics_falls_back_to_exact_rule_when_missing(tmp_path):
    executor = _bare_executor(tmp_path)
    pre = {"object_integrity": {"obj_mug_0": _integrity_signature("a")}}
    post = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature("b"),
            "obj_plate_0": _integrity_signature("c"),
        }
    }

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_plate_0"] = _integrity_signature("c")
    with pytest.raises(RuntimeError, match="unexpectedly changed obj_mug_0"):
        executor._validate_typed_initializer_result(
            pre,
            post,
            target="obj_plate_0",
        )


def test_typed_world_semantics_rejects_forged_aggregate_digest(tmp_path):
    executor = _bare_executor(tmp_path)
    semantic = _world_semantics()
    semantic["sha256"] = "f" * 64
    pre = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature("a", world_semantics=semantic)
        }
    }
    post = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature(
                "b", matrix_x=0.5, world_semantics=semantic
            ),
            "obj_plate_0": _integrity_signature(
                "c", world_semantics=_world_semantics()
            ),
        }
    }

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_plate_0"] = _integrity_signature("c")
    with pytest.raises(RuntimeError, match="unexpectedly changed obj_mug_0"):
        executor._validate_typed_initializer_result(
            pre,
            post,
            target="obj_plate_0",
        )


def test_representation_exception_is_not_enabled_for_baseline_profile(tmp_path):
    executor = _bare_executor(tmp_path)
    executor.harness_profile = "baseline"
    semantic = _world_semantics()
    pre = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature("a", world_semantics=semantic)
        }
    }
    post = {
        "object_integrity": {
            "obj_mug_0": _integrity_signature(
                "b", matrix_x=0.5, world_semantics=semantic
            ),
            "obj_plate_0": _integrity_signature(
                "c", world_semantics=_world_semantics()
            ),
        }
    }

    # Exercise the surviving composition replacement path: the target already exists,
    # while this test's representation/integrity assertion concerns a bystander.
    pre["object_integrity"]["obj_plate_0"] = _integrity_signature("c")
    with pytest.raises(RuntimeError, match="unexpectedly changed obj_mug_0"):
        executor._validate_typed_initializer_result(
            pre,
            post,
            target="obj_plate_0",
        )


def test_world_semantics_subprocess_capability_is_gpt6_initializer_only(
    tmp_path, monkeypatch
):
    executor = _bare_executor(tmp_path)
    executor.blender_command = "blender"
    executor.blender_file = str(tmp_path / "live.blend")
    executor.blender_script = str(tmp_path / "wrapper.py")
    executor.target_image_path = None
    executor.gpu_devices = None
    executor.render_engine = (
        "BLENDER_EEVEE_NEXT"  # 4.2+ id; "BLENDER_EEVEE" is invalid on 4.5
    )
    executor.script_path = tmp_path
    executor.render_path = tmp_path
    executor._registered_root_categories_for_wrapper = lambda: {}
    executor._initializer_floor_helper_allowed = lambda: False
    executor._runtime_root_bindings_for_wrapper = lambda: {}
    captured = []

    def fake_run(*_args, **kwargs):
        captured.append(kwargs["env"])
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setenv("GRASE_TYPED_WORLD_SEMANTICS", "1")
    monkeypatch.setattr("lib.tools.blender.exec.subprocess.run", fake_run)

    executor.harness_profile = "baseline"
    executor._execute_blender(str(tmp_path / "probe.py"))
    assert "GRASE_TYPED_WORLD_SEMANTICS" not in captured[-1]

    executor.harness_profile = "gpt6_v1"
    executor._execute_blender(str(tmp_path / "probe.py"))
    assert captured[-1]["GRASE_TYPED_WORLD_SEMANTICS"] == "1"

    executor.root_stage_name = "composition"
    executor._execute_blender(str(tmp_path / "probe.py"))
    assert "GRASE_TYPED_WORLD_SEMANTICS" not in captured[-1]


def _write_canonical_source_state(tmp_path):
    scene = tmp_path / "scene"
    masks = scene / "masks"
    meshes = scene / "meshes"
    physics = scene / "physics"
    masks.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    physics.mkdir(parents=True, exist_ok=True)
    (masks / "mug.npy").write_bytes(b"source-mask")
    (masks / "masks.json").write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "category": "mug",
                        "instance": 0,
                        "mask_path": "masks/mug.npy",
                    }
                ]
            }
        )
    )
    (meshes / "mug.glb").write_bytes(b"source-mesh")
    graph = {
        "nodes": [
            {
                "id": "mug#0",
                "category": "mug",
                "kind": "object",
                "parent": "table#0",
                "support": "table#0",
                "children": [],
                "mask_path": "masks/mug.npy",
            }
        ]
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
    (scene / "scene_graph.json").write_text(json.dumps(graph))
    (scene / "placement.json").write_text(json.dumps(placement))
    (physics / "physics_vlm.json").write_text(
        json.dumps({"obj_mug_0": {"material": "ceramic", "mass_kg": 0.4}})
    )
    (physics / "physics_estimate_manifest.json").write_text(
        json.dumps({"objects": {"obj_mug_0": {"status": "estimated"}}})
    )
    (physics / "blend_base.json").write_text(
        json.dumps({"obj_mug_0": [[1, 0, 0, 0]] * 4})
    )
    (physics / "pose_changes.json").write_text(
        json.dumps(
            {
                "objects": {"obj_mug_0": {"settle_total": [[1, 0, 0, 0]] * 4}},
                "composition_certify": {"obj_mug_0": {"drift_m": 0.0}},
                "composition_certify_converged": True,
                "delivered": {"obj_mug_0": {"capsized": False}},
                "certify_run_id": "old-run",
            }
        )
    )
    return graph, placement


def test_recovery_hook_is_explicitly_profile_gated(tmp_path):
    executor = _bare_executor(tmp_path)
    executor.harness_profile = "baseline"

    assert executor._typed_initializer_recovery_enabled() is False


def test_source_mask_artifacts_are_protected_only_by_gpt6_profile(tmp_path):
    executor = _bare_executor(tmp_path)
    executor.harness_profile_manifest["capabilities"]["runtime_object_inventory"] = True
    masks_dir = tmp_path / "scene" / "masks"
    masks_dir.mkdir()
    relative_mask = masks_dir / "relative.npy"
    absolute_mask = tmp_path / "absolute.npy"
    relative_mask.write_bytes(b"relative-mask")
    absolute_mask.write_bytes(b"absolute-mask")
    manifest = masks_dir / "masks.json"
    manifest.write_text(
        json.dumps(
            {
                "objects": [
                    {"mask_path": "masks/relative.npy"},
                    {"nested": {"mask_path": str(absolute_mask)}},
                ]
            }
        )
    )

    protected = executor._protected_artifact_snapshots()

    assert manifest in protected
    assert relative_mask.resolve() in protected
    assert absolute_mask.resolve() in protected

    executor.harness_profile = "baseline"
    baseline = executor._protected_artifact_snapshots()
    assert manifest not in baseline
    assert relative_mask.resolve() not in baseline
    assert absolute_mask.resolve() not in baseline


def test_begin_closes_requested_event_when_ledger_persist_fails(tmp_path):
    executor = _bare_executor(tmp_path)
    events = []
    executor._append_mutation_event = events.append

    def fail_persist():
        raise OSError("disk unavailable")

    executor._persist_initializer_ledger = fail_persist
    with pytest.raises(RuntimeError) as raised:
        executor._begin_mutation("add_object", {"reason": "missing mug"})

    assert [event["status"] for event in events] == ["requested", "error"]
    assert events[-1]["details"]["scene_mutation"] == "not_started"
    assert raised.value.audit_recorded is True
    assert executor._initializer_ledger == executor._empty_initializer_ledger()


def test_transaction_id_is_not_reused_after_ledger_write_failure(tmp_path):
    executor = _bare_executor(tmp_path)

    def fail_persist():
        raise OSError("disk unavailable")

    executor._persist_initializer_ledger = fail_persist
    with pytest.raises(RuntimeError):
        executor._begin_mutation("add_object", {"reason": "missing mug"})
    del executor._persist_initializer_ledger

    txid = executor._begin_mutation("add_object", {"reason": "missing plate"})

    assert txid == 2
    assert executor._initializer_ledger["next_transaction_id"] == 3


@pytest.mark.parametrize("request_already_in_ledger", [False, True])
def test_markerless_request_is_closed_once_without_touching_scene(
    tmp_path, request_already_in_ledger
):
    executor = _bare_executor(tmp_path)
    live_before = (tmp_path / "live.blend").read_bytes()
    request = {
        "transaction_id": 1,
        "kind": "add_object",
        "status": "requested",
        "request": {"reason": "missing mug"},
    }
    executor._append_mutation_event(request)
    if request_already_in_ledger:
        executor._initializer_ledger["events"].append(
            {
                "id": 1,
                "kind": "add_object",
                "status": "requested",
                "request": {"reason": "missing mug"},
            }
        )
        executor._initializer_ledger["next_transaction_id"] = 2
        executor._persist_initializer_ledger()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()
    journal_after_first = restarted._mutation_journal_path().read_bytes()
    ledger_after_first = (tmp_path / "ledger.json").read_bytes()
    restarted._recover_incomplete_runtime_mutations()

    journal = [json.loads(line) for line in journal_after_first.decode().splitlines()]
    assert [event["status"] for event in journal] == ["requested", "error"]
    assert journal[-1]["details"]["scene_mutation"] == "not_started"
    assert [
        event["status"]
        for event in restarted._initializer_ledger["events"]
        if event.get("id") == 1
    ] == ["requested", "error"]
    assert restarted._initializer_ledger["next_transaction_id"] == 2
    assert restarted._mutation_journal_path().read_bytes() == journal_after_first
    assert (tmp_path / "ledger.json").read_bytes() == ledger_after_first
    assert (tmp_path / "live.blend").read_bytes() == live_before


def test_pending_recovery_blocks_a_later_mutation_before_audit(tmp_path):
    executor = _bare_executor(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=json.loads(json.dumps(executor._initializer_ledger)),
    )
    journal_before = executor._mutation_journal_path().read_bytes()
    ledger_before = (tmp_path / "ledger.json").read_bytes()
    live_before = (tmp_path / "live.blend").read_bytes()

    with pytest.raises(RuntimeError, match="still requires recovery"):
        executor._begin_mutation("add_object", {"reason": "missing plate"})

    assert executor._mutation_journal_path().read_bytes() == journal_before
    assert (tmp_path / "ledger.json").read_bytes() == ledger_before
    assert (tmp_path / "live.blend").read_bytes() == live_before


def test_older_pending_transaction_is_not_restored_across_a_newer_id(tmp_path):
    executor = _bare_executor(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=json.loads(json.dumps(executor._initializer_ledger)),
    )
    (tmp_path / "live.blend").write_bytes(b"newer-scene-state")
    executor._append_mutation_event(
        {
            "transaction_id": txid + 1,
            "kind": "add_object",
            "status": "error",
            "details": {"scene_mutation": "not_started"},
        }
    )

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"newer-scene-state"
    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "recovery_needed"


def test_prepared_transaction_is_restored_on_next_startup(tmp_path):
    executor = _bare_executor(tmp_path)
    placement = tmp_path / "scene" / "placement.json"
    placement.write_text('{"objects": [{"mesh_name": "obj_mug_0"}]}')
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=ledger_after_request,
    )

    (tmp_path / "live.blend").write_bytes(b"partially-mutated-blend")
    placement.write_text('{"objects": [{"mesh_name": "obj_bad"}]}')
    (tmp_path / "scene" / "scene_graph.json").write_text('{"nodes": []}')
    asset = tmp_path / "scene" / "runtime_objects" / "assets" / f"tx_{txid}_mug"
    asset.mkdir(parents=True)
    (asset / "mesh.glb").write_bytes(b"orphan")

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._pipeline_object_names = ["obj_bad"]
    restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"
    assert placement.read_text() == '{"objects": [{"mesh_name": "obj_mug_0"}]}'
    assert not (tmp_path / "scene" / "scene_graph.json").exists()
    assert not asset.exists()
    assert restarted._pipeline_object_names == ["obj_mug_0"]
    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "rolled_back"
    assert marker["payloads_pruned"] is True
    assert not (restarted._runtime_transaction_dir(txid) / "before.blend").exists()
    assert not (restarted._runtime_transaction_dir(txid) / "snapshots").exists()
    assert restarted._initializer_ledger["events"][-1]["status"] == "rolled_back"


@pytest.mark.parametrize(
    "kind", ["execute_and_evaluate_objects", "remove_runtime_object"]
)
def test_initializer_startup_recovers_before_rejecting_empty_placement(tmp_path, kind):
    executor = _bare_executor(tmp_path)
    placement = tmp_path / "scene" / "placement.json"
    placement.write_text('{"objects": [{"mesh_name": "obj_mug_0"}]}')
    txid = int(executor._begin_mutation(kind, {"reason": "duplicate"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind=kind,
        ledger_after_request=ledger_after_request,
    )
    placement.write_text('{"objects": []}')
    (tmp_path / "live.blend").write_bytes(b"partially-removed-blend")
    baseline = tmp_path / "entry.blend"
    baseline.write_bytes(b"blend-before")

    restarted = Executor(
        blender_command="blender",
        blender_file=str(tmp_path / "live.blend"),
        blender_script=str(tmp_path / "wrapper.py"),
        script_save=str(tmp_path / "scripts"),
        render_save=str(tmp_path / "renders"),
        blender_save=str(tmp_path / "live.blend"),
        moge_dir=str(tmp_path / "scene"),
        root_stage_name="initializer",
        stage_dir=str(tmp_path / "stage"),
        initializer_baseline_blend=str(baseline),
        initializer_ledger_path=str(tmp_path / "ledger.json"),
        harness_profile="gpt6_v1",
        harness_profile_manifest=executor.harness_profile_manifest,
    )

    assert restarted._pipeline_object_names == ["obj_mug_0"]
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"
    assert json.loads(placement.read_text())["objects"][0]["mesh_name"] == "obj_mug_0"


def test_committed_journal_wins_over_unfinalized_marker(tmp_path):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=ledger_after_request,
    )
    (tmp_path / "live.blend").write_bytes(b"committed-blend")
    artifact_manifest = executor._typed_initializer_artifact_manifest(
        txid=txid,
        kind="edit_object_pose",
        state="committed",
        object_id="mug#0",
        target_mesh="obj_mug_0",
        touched_objects={},
        strict=True,
    )
    executor._initializer_ledger["active_transactions"].append(
        {
            "id": txid,
            "kind": "edit_object_pose",
            "status": "committed",
            "artifact_manifest": artifact_manifest,
        }
    )
    executor._persist_initializer_ledger()
    executor._append_mutation_event(
        {
            "transaction_id": txid,
            "kind": "edit_object_pose",
            "status": "committed",
            "details": {"artifact_manifest": artifact_manifest},
        }
    )

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"committed-blend"
    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "committed"
    assert marker["payloads_pruned"] is True
    assert not (restarted._runtime_transaction_dir(txid) / "before.blend").exists()
    assert any(
        event.get("id") == txid and event.get("status") == "committed"
        for event in restarted._initializer_ledger["events"]
    )


@pytest.mark.parametrize("corruption", ["live_blend", "ledger_manifest"])
def test_unfinalized_committed_marker_rejects_unbound_current_state(
    tmp_path, corruption
):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=ledger_after_request,
    )
    (tmp_path / "live.blend").write_bytes(b"committed-blend")
    artifact_manifest = executor._typed_initializer_artifact_manifest(
        txid=txid,
        kind="edit_object_pose",
        state="committed",
        object_id="mug#0",
        target_mesh="obj_mug_0",
        touched_objects={},
        strict=True,
    )
    executor._initializer_ledger["active_transactions"].append(
        {
            "id": txid,
            "kind": "edit_object_pose",
            "status": "committed",
            "artifact_manifest": json.loads(json.dumps(artifact_manifest)),
        }
    )
    executor._persist_initializer_ledger()
    executor._append_mutation_event(
        {
            "transaction_id": txid,
            "kind": "edit_object_pose",
            "status": "committed",
            "details": {"artifact_manifest": artifact_manifest},
        }
    )
    if corruption == "live_blend":
        (tmp_path / "live.blend").write_bytes(b"corrupt-after-commit")
    else:
        executor._initializer_ledger["active_transactions"][0]["artifact_manifest"][
            "manifest_sha256"
        ] = "0" * 64
        executor._persist_initializer_ledger()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "recovery_needed"
    assert (restarted._runtime_transaction_dir(txid) / "before.blend").is_file()


def test_typed_failure_preserves_physics_rejection_reports(tmp_path):
    executor = _bare_executor(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    live_before, graph_before, artifacts, names = (
        executor._runtime_transaction_snapshots(
            txid,
            kind="edit_object_pose",
            ledger_after_request=ledger_after_request,
        )
    )
    error = RuntimeError("physics rejected the pose")
    error.reports = [{"name": "obj_mug_0", "converged": False}]
    (tmp_path / "live.blend").write_bytes(b"bad-physics-result")

    result = executor._typed_initializer_failure(
        txid=txid,
        kind="edit_object_pose",
        error=error,
        live_before=live_before,
        graph_before=graph_before,
        artifact_before=artifacts,
        pipeline_names_before=names,
        ledger_after_request=ledger_after_request,
    )

    assert result["output"]["physics_reports"] == error.reports
    journal = [
        json.loads(line)
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    assert journal[-1]["details"]["physics_reports"] == error.reports
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "rolled_back"
    assert marker["terminal_details"]["physics_reports"] == error.reports


@pytest.mark.parametrize("corrupt_role", ["blend", "placement"])
def test_startup_recovery_rejects_corrupt_snapshot_payload(tmp_path, corrupt_role):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(executor._begin_mutation("edit_object_pose", {"reason": "wrong"}))
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=ledger_after_request,
    )
    tx_dir = executor._runtime_transaction_dir(txid)
    if corrupt_role == "blend":
        (tx_dir / "before.blend").write_bytes(b"corrupt-blend")
    else:
        manifest = json.loads((tx_dir / "snapshot_manifest.json").read_text())
        placement = next(
            row
            for row in manifest["runtime_artifacts"]
            if row["relative_path"] == "placement.json"
        )
        (tx_dir / placement["snapshot_path"]).write_bytes(b"corrupt-placement")
    (tmp_path / "live.blend").write_bytes(b"partially-mutated")

    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        executor._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"partially-mutated"
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "recovery_needed"
    assert marker["terminal_details"]["artifact_manifest"]["state"] == "error"


def test_committed_manifest_matches_ledger_journal_marker_and_delivered_pose(
    tmp_path, monkeypatch
):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(
        executor._begin_mutation("execute_and_evaluate_objects", {"reason": "wrong"})
    )
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    _, _, artifact_before, _ = executor._runtime_transaction_snapshots(
        txid,
        kind="execute_and_evaluate_objects",
        ledger_after_request=ledger_after_request,
    )
    before_matrix = [[1.0, 0.0, 0.0, 0.0]] * 4
    after_matrix = [[1.0, 0.0, 0.0, 0.1]] * 4
    post = {
        "object_integrity": {"obj_mug_0": {"matrix": after_matrix, "vertices": 100}}
    }
    executor._read_penetration_data = lambda **_kwargs: post
    executor._push_edit_snapshot = lambda *_args, **_kwargs: None
    # Presentation normalization is deliberately exercised after the durable commit:
    # even a malformed renderer response must return the committed transaction rather
    # than escaping to the typed wrapper's pre-decision rollback path.
    executor._render_novel_view = lambda *_args: {
        "status": "success",
        "output": None,
    }
    import lib.tools.geometry.inventory_contract as inventory_contract

    monkeypatch.setattr(
        inventory_contract, "validate_scene_artifacts", lambda *_args, **_kwargs: {}
    )
    result = executor._commit_typed_initializer_mutation(
        txid=txid,
        declarations={"added_names": {}, "removed_names": {}, "removed_objects": []},
        reason="obviously wrong pose",
        pre={
            "object_integrity": {
                "obj_mug_0": {"matrix": before_matrix, "vertices": 100}
            }
        },
        reports=[{"name": "obj_mug_0", "converged": True}],
        artifact_before=artifact_before,
    )

    assert result["status"] == "success"
    transaction = result["output"]["transaction"]
    manifest = transaction["artifact_manifest"]
    journal = [
        json.loads(line)
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert (
        executor._initializer_ledger["active_transactions"][-1]["artifact_manifest"]
        == manifest
    )
    assert journal[-1]["details"]["artifact_manifest"] == manifest
    assert marker["terminal_details"]["artifact_manifest"] == manifest
    assert (
        journal[-1]["details"]["objects"]["obj_mug_0"]["after_matrix"] == after_matrix
    )
    assert manifest["targets"][0]["current_mesh"]["sha256"]
    assert manifest["targets"][0]["mask"]["sha256"]
    assert manifest["targets"][0]["scoring_mask"]["sha256"]
    assert all(
        row["sha256"]
        for role, row in manifest["artifacts"].items()
        if role != "runtime_inventory"
    )


@pytest.mark.parametrize(
    "failure_point", ["committed_journal_ledger", "terminal_marker"]
)
def test_commit_boundary_failure_preserves_post_state_until_startup_rollforward(
    tmp_path, monkeypatch, failure_point
):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(
        executor._begin_mutation("execute_and_evaluate_objects", {"reason": "wrong"})
    )
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    _, _, artifact_before, _ = executor._runtime_transaction_snapshots(
        txid,
        kind="execute_and_evaluate_objects",
        ledger_after_request=ledger_after_request,
    )
    committed_blend = b"post-simulation-committed-blend"
    (tmp_path / "live.blend").write_bytes(committed_blend)
    before_matrix = [[1.0, 0.0, 0.0, 0.0]] * 4
    after_matrix = [[1.0, 0.0, 0.0, 0.1]] * 4
    executor._read_penetration_data = lambda **_kwargs: {
        "object_integrity": {"obj_mug_0": {"matrix": after_matrix, "vertices": 100}}
    }
    executor._push_edit_snapshot = lambda *_args, **_kwargs: None
    executor._render_novel_view = lambda *_args: pytest.fail(
        "presentation rendering must not run while commit recovery is pending"
    )
    import lib.tools.geometry.inventory_contract as inventory_contract

    monkeypatch.setattr(
        inventory_contract, "validate_scene_artifacts", lambda *_args, **_kwargs: {}
    )
    if failure_point == "committed_journal_ledger":
        persist_ledger = executor._persist_initializer_ledger
        persist_calls = 0

        def fail_committed_event_ledger():
            nonlocal persist_calls
            persist_calls += 1
            if persist_calls == 2:
                raise OSError("injected committed-event ledger failure")
            return persist_ledger()

        executor._persist_initializer_ledger = fail_committed_event_ledger
    else:
        set_recovery_state = executor._set_runtime_recovery_state

        def fail_terminal_marker(transaction_id, state, **kwargs):
            if state == "committed":
                raise OSError("injected terminal marker failure")
            return set_recovery_state(transaction_id, state, **kwargs)

        executor._set_runtime_recovery_state = fail_terminal_marker

    result = executor._commit_typed_initializer_mutation(
        txid=txid,
        declarations={"added_names": {}, "removed_names": {}, "removed_objects": []},
        reason="obviously wrong pose",
        pre={
            "object_integrity": {
                "obj_mug_0": {"matrix": before_matrix, "vertices": 100}
            }
        },
        reports=[{"name": "obj_mug_0", "converged": True}],
        artifact_before=artifact_before,
    )

    journal_before_recovery = [
        json.loads(line)
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert result["status"] == "error"
    assert result["output"]["scene_mutation"] == "commit_recovery_pending"
    assert result["output"]["retryable"] is False
    assert [event["status"] for event in journal_before_recovery] == [
        "requested",
        "committed",
    ]
    assert marker["state"] == "recovery_needed"
    assert (tmp_path / "live.blend").read_bytes() == committed_blend
    assert len(executor._initializer_ledger["active_transactions"]) == 1
    persisted_before_recovery = json.loads((tmp_path / "ledger.json").read_text())
    if failure_point == "committed_journal_ledger":
        assert not any(
            event.get("id") == txid and event.get("status") == "committed"
            for event in persisted_before_recovery["events"]
        )

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()
    journal_after_first = restarted._mutation_journal_path().read_bytes()
    ledger_after_first = (tmp_path / "ledger.json").read_bytes()
    restarted._recover_incomplete_runtime_mutations()

    terminal_marker = json.loads(
        restarted._runtime_recovery_marker_path(txid).read_text()
    )
    committed_ledger_events = [
        event
        for event in restarted._initializer_ledger["events"]
        if event.get("id") == txid and event.get("status") == "committed"
    ]
    assert terminal_marker["state"] == "committed"
    assert terminal_marker["payloads_pruned"] is True
    assert len(committed_ledger_events) == 1
    assert restarted._mutation_journal_path().read_bytes() == journal_after_first
    assert (tmp_path / "ledger.json").read_bytes() == ledger_after_first
    assert (tmp_path / "live.blend").read_bytes() == committed_blend


def test_precommit_failure_remains_eligible_for_exact_rollback(tmp_path, monkeypatch):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    txid = int(
        executor._begin_mutation("execute_and_evaluate_objects", {"reason": "wrong"})
    )
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    live_before, graph_before, artifact_before, names_before = (
        executor._runtime_transaction_snapshots(
            txid,
            kind="execute_and_evaluate_objects",
            ledger_after_request=ledger_after_request,
        )
    )
    (tmp_path / "live.blend").write_bytes(b"uncommitted-post-simulation-blend")
    executor._read_penetration_data = lambda **_kwargs: {
        "object_integrity": {
            "obj_mug_0": {
                "matrix": [[1.0, 0.0, 0.0, 0.1]] * 4,
                "vertices": 100,
            }
        }
    }
    executor._push_edit_snapshot = lambda *_args, **_kwargs: None

    def fail_precommit_sync(_txid):
        raise OSError("injected precommit durability failure")

    executor._durably_sync_typed_initializer_state = fail_precommit_sync
    import lib.tools.geometry.inventory_contract as inventory_contract

    monkeypatch.setattr(
        inventory_contract, "validate_scene_artifacts", lambda *_args, **_kwargs: {}
    )

    with pytest.raises(OSError, match="precommit durability failure") as raised:
        executor._commit_typed_initializer_mutation(
            txid=txid,
            declarations={
                "added_names": {},
                "removed_names": {},
                "removed_objects": [],
            },
            reason="obviously wrong pose",
            pre={
                "object_integrity": {
                    "obj_mug_0": {
                        "matrix": [[1.0, 0.0, 0.0, 0.0]] * 4,
                        "vertices": 100,
                    }
                }
            },
            reports=[{"name": "obj_mug_0", "converged": True}],
            artifact_before=artifact_before,
        )

    # The declared initializer transaction uses this same pre-decision rollback boundary.
    executor._restore_scene_graph_bytes = lambda payload, **_kwargs: (
        (tmp_path / "scene" / "scene_graph.json").write_bytes(payload)
        if payload is not None
        else None
    )
    result = executor._typed_initializer_failure(
        txid=txid,
        kind="execute_and_evaluate_objects",
        error=raised.value,
        live_before=live_before,
        graph_before=graph_before,
        artifact_before=artifact_before,
        pipeline_names_before=names_before,
        ledger_after_request=ledger_after_request,
        object_id="mug#0",
        target_mesh="obj_mug_0",
    )

    statuses = [
        json.loads(line)["status"]
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    assert result["output"]["scene_mutation"] == "not_committed"
    assert statuses == ["requested", "rolled_back"]
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"


def test_runtime_removal_withdraws_all_physics_and_preserves_target_manifest(
    tmp_path, monkeypatch
):
    executor = _bare_executor(tmp_path)
    graph, placement = _write_canonical_source_state(tmp_path)
    mesh_path = tmp_path / "scene" / "meshes" / "mug.glb"
    mask_path = tmp_path / "scene" / "masks" / "mug.npy"
    record = {
        "id": "mug#0",
        "category": "mug",
        "instance": 0,
        "origin": "runtime_added",
        "graph_node": graph["nodes"][0],
        "placement": placement["objects"][0],
        "mask_path": "masks/mug.npy",
        "scoring_mask_path": "masks/mug.npy",
        "mask_sha256": executor._typed_initializer_file_sha256(mask_path),
        "scoring_mask_sha256": executor._typed_initializer_file_sha256(mask_path),
        "current_mesh_revision": 1,
        "mesh_revisions": [
            {
                "revision": 1,
                "mesh_glb": "meshes/mug.glb",
                "sha256": executor._typed_initializer_file_sha256(mesh_path),
            }
        ],
        "physical_material": {"material": "ceramic", "mass_kg": 0.4},
        "visual_material": {"status": "embedded", "sha256": "a" * 64},
    }
    inventory = {"objects": [record]}
    inventory_path = tmp_path / "scene" / "runtime_objects" / "inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(json.dumps(inventory))
    txid = int(
        executor._begin_mutation(
            "execute_and_evaluate_objects", {"object": "mug#0", "reason": "duplicate"}
        )
    )
    ledger_after_request = json.loads(json.dumps(executor._initializer_ledger))
    _, _, artifact_before, _ = executor._runtime_transaction_snapshots(
        txid,
        kind="execute_and_evaluate_objects",
        ledger_after_request=ledger_after_request,
    )
    withdrawn = executor._typed_initializer_target_manifest(
        object_id="mug#0",
        target_mesh="obj_mug_0",
        graph=graph,
        placement=placement,
        runtime_inventory=inventory,
        inventory_record=record,
        expected_present=True,
    )
    change = executor._remove_physics_identity("obj_mug_0", transaction_id=txid)

    physics = tmp_path / "scene" / "physics"
    assert withdrawn["inventory_record"] == record
    assert withdrawn["physical_material"]["material"] == "ceramic"
    assert withdrawn["physics_records"]["pose_changes"] is not None
    assert withdrawn["current_mesh"]["sha256"] == record["mesh_revisions"][0]["sha256"]
    assert "obj_mug_0" not in json.loads((physics / "physics_vlm.json").read_text())
    assert (
        "obj_mug_0"
        not in json.loads((physics / "physics_estimate_manifest.json").read_text())[
            "objects"
        ]
    )
    assert "obj_mug_0" not in json.loads((physics / "blend_base.json").read_text())
    pose = json.loads((physics / "pose_changes.json").read_text())
    assert "obj_mug_0" not in pose["objects"]
    assert not {
        "composition_certify",
        "composition_certify_converged",
        "delivered",
        "certify_run_id",
    } & set(pose)
    assert pose["runtime_object_removals"][-1]["transaction_id"] == txid
    assert change["removed_records"]["pose_changes"]["value"]

    (tmp_path / "scene" / "scene_graph.json").write_text(json.dumps({"nodes": []}))
    (tmp_path / "scene" / "placement.json").write_text(json.dumps({"objects": []}))
    inventory_path.write_text(json.dumps({"objects": []}))
    (tmp_path / "live.blend").write_bytes(b"blend-without-mug")
    executor._pipeline_object_names = []
    executor._read_penetration_data = lambda **_kwargs: {"object_integrity": {}}
    executor._push_edit_snapshot = lambda *_args, **_kwargs: None
    executor._render_novel_view = lambda *_args: {
        "status": "success",
        "output": {"text": []},
    }
    import lib.tools.geometry.inventory_contract as inventory_contract

    monkeypatch.setattr(
        inventory_contract, "validate_scene_artifacts", lambda *_args, **_kwargs: {}
    )
    result = executor._commit_typed_initializer_mutation(
        txid=txid,
        declarations={
            "added_names": {},
            "removed_names": {"mug#0": "obj_mug_0"},
            "removed_objects": ["mug#0"],
        },
        reason="duplicate",
        pre={
            "object_integrity": {
                "obj_mug_0": {
                    "matrix": [[1.0, 0.0, 0.0, 0.0]] * 4,
                    "vertices": 100,
                }
            }
        },
        reports=[],
        artifact_before=artifact_before,
        physics_state_change=change,
    )
    terminal_manifest = result["output"]["transaction"]["artifact_manifest"]
    assert terminal_manifest["targets"][0]["present"] is False
    assert terminal_manifest["targets"][0]["object_id"] == "mug#0"
    transaction = result["output"]["transaction"]
    assert transaction["objects"]["obj_mug_0"]["before_signature"]["vertices"] == 100
    assert transaction["physics_state_change"] == change


def _prepare_typed_pose_undo(tmp_path, *, kind="edit_object_pose"):
    executor = _bare_executor(tmp_path)
    _write_canonical_source_state(tmp_path)
    graph_path = tmp_path / "scene" / "scene_graph.json"

    def restore_graph(payload, *, durable=False):
        if payload is None:
            graph_path.unlink(missing_ok=True)
        else:
            writer = (
                executor._durable_atomic_write_bytes
                if durable
                else executor._atomic_write_bytes
            )
            writer(graph_path, payload)

    executor._restore_scene_graph_bytes = restore_graph
    prior_blend = tmp_path / "prior.blend"
    prior_blend.write_bytes(b"blend-before")
    after_blend = tmp_path / "after.blend"
    after_blend.write_bytes(b"blend-after")
    (tmp_path / "live.blend").write_bytes(b"blend-after")
    txid = 3
    requested = {
        "id": txid,
        "kind": kind,
        "status": "requested",
    }
    transaction = {
        "id": txid,
        "kind": kind,
        "status": "committed",
        "target": "obj_mug_0",
        "object_id": "mug#0",
    }
    objects = {
        "obj_mug_0": {
            "before_signature": {**_integrity_signature("a"), "name": "obj_mug_0"},
            "after_signature": {
                **_integrity_signature("b", matrix_x=0.2),
                "name": "obj_mug_0",
            },
        }
    }
    transaction["objects"] = objects
    is_batch = kind == "execute_and_evaluate_objects"
    if is_batch:
        transaction["declarations"] = {
            "added_names": {},
            "removed_names": {},
            "removed_objects": [],
        }
        transaction["authored_objects"] = objects
    artifact_manifest = executor._typed_initializer_artifact_manifest(
        txid=txid,
        kind=kind,
        state="committed",
        object_id=None if is_batch else "mug#0",
        target_mesh=None if is_batch else "obj_mug_0",
        touched_objects=objects,
        authored_objects=objects if is_batch else None,
        target_states=[
            {"object_id": "mug#0", "mesh_name": "obj_mug_0", "present": True}
        ]
        if is_batch
        else None,
        strict=True,
    )
    transaction["artifact_manifest"] = artifact_manifest
    commit_event = {
        "kind": kind,
        "status": "committed",
        "details": {
            key: transaction[key]
            for key in (
                "artifact_manifest",
                "objects",
                "authored_objects",
                "declarations",
            )
            if key in transaction
        },
    }
    executor._initializer_ledger = {
        "schema_version": 1,
        "next_transaction_id": 4,
        "events": [requested, {"id": txid, **commit_event}],
        "active_transactions": [transaction],
        "runtime_root_surfaces": [],
    }
    executor._persist_initializer_ledger()
    executor._append_mutation_event({"transaction_id": txid, **commit_event})
    executor._ledger_base = {
        "schema_version": 1,
        "next_transaction_id": 4,
        "events": [requested],
        "active_transactions": [],
        "runtime_root_surfaces": [],
    }
    executor.base_state = str(prior_blend)
    executor.edit_history = [str(after_blend)]
    executor._ledger_history = [json.loads(json.dumps(executor._initializer_ledger))]
    graph_bytes = (tmp_path / "scene" / "scene_graph.json").read_bytes()
    executor._graph_base = graph_bytes
    executor._graph_history = [graph_bytes]
    executor._edit_meta = [
        {
            "kind": kind,
            "transaction_id": txid,
            "object_id": "mug#0",
            "artifact_before": executor._snapshot_runtime_artifacts(),
        }
    ]
    marker = executor._runtime_recovery_marker_path(txid)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": txid,
                "kind": kind,
                "state": "committed",
                "terminal_details": {"artifact_manifest": artifact_manifest},
            }
        )
    )
    executor._pose_dirty = False
    executor._armed = set()
    executor._post_flip_investigation_required = None
    return executor, txid


def _prepare_durable_undo_decision(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)

    def fail_audit(_event):
        raise OSError("audit disk unavailable")

    executor._append_mutation_event = fail_audit
    ok, warning = executor._undo_latest_edit(record_undo_event=True)
    assert ok is True
    assert "durably committed" in warning
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_commit_decided"
    return executor, txid, marker


def test_durable_undo_decision_digest_corruption_preserves_ledger_bytes(tmp_path):
    executor, txid, marker = _prepare_durable_undo_decision(tmp_path)
    marker["terminal_details"]["decision_sha256"] = "0" * 64
    executor._durable_atomic_write_json(
        executor._runtime_recovery_marker_path(txid), marker
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_before = ledger_path.read_bytes()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    assert ledger_path.read_bytes() == ledger_before
    failed_marker = json.loads(
        restarted._runtime_recovery_marker_path(txid).read_text()
    )
    assert failed_marker["state"] == "undo_commit_decided"
    assert failed_marker["recovery_error"]["phase"] == "startup_recovery"


def test_wrong_but_structurally_valid_post_undo_ledger_is_rejected_without_rewrite(
    tmp_path,
):
    executor, txid, marker = _prepare_durable_undo_decision(tmp_path)
    decision = marker["terminal_details"]
    decision["post_undo_ledger"]["events"][0]["tampered"] = True
    decision_body = {
        key: value for key, value in decision.items() if key != "decision_sha256"
    }
    decision["decision_sha256"] = executor._typed_initializer_json_sha256(decision_body)
    executor._durable_atomic_write_json(
        executor._runtime_recovery_marker_path(txid), marker
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_before = ledger_path.read_bytes()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    assert ledger_path.read_bytes() == ledger_before
    failed_marker = json.loads(
        restarted._runtime_recovery_marker_path(txid).read_text()
    )
    assert failed_marker["state"] == "undo_commit_decided"
    assert failed_marker["recovery_error"]["phase"] == "startup_recovery"


def test_exact_and_conflicting_undone_ledger_rows_fail_closed(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    ok, error = executor._undo_latest_edit(record_undo_event=True)
    assert (ok, error) == (True, "")

    exact = next(
        row
        for row in executor._initializer_ledger["events"]
        if row.get("id") == txid and row.get("status") == "undone"
    )
    conflicting = json.loads(json.dumps(exact))
    conflicting["details"]["scene_mutation"] = "conflicting-rewind"
    executor._initializer_ledger["events"].append(conflicting)
    executor._persist_initializer_ledger()
    ledger_path = tmp_path / "ledger.json"
    ledger_before = ledger_path.read_bytes()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    assert ledger_path.read_bytes() == ledger_before


@pytest.mark.parametrize("missing_peer", ["journal", "ledger"])
def test_terminal_undone_marker_requires_exact_journal_and_ledger_parity(
    tmp_path, missing_peer
):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    ok, error = executor._undo_latest_edit(record_undo_event=True)
    assert (ok, error) == (True, "")

    if missing_peer == "journal":
        journal_path = executor._mutation_journal_path()
        rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
        rows = [
            row
            for row in rows
            if not (
                int(row.get("transaction_id") or -1) == txid
                and row.get("status") == "undone"
            )
        ]
        journal_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    else:
        executor._initializer_ledger["events"] = [
            row
            for row in executor._initializer_ledger["events"]
            if not (row.get("id") == txid and row.get("status") == "undone")
        ]
        executor._persist_initializer_ledger()
    ledger_path = tmp_path / "ledger.json"
    ledger_before = ledger_path.read_bytes()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    assert ledger_path.read_bytes() == ledger_before


def test_typed_undo_persists_manifest_to_ledger_journal_and_recovery(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)

    ok, error = executor._undo_latest_edit(record_undo_event=True)

    assert (ok, error) == (True, "")
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"
    journal = [
        json.loads(line)
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    ledger_event = executor._initializer_ledger["events"][-1]
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert journal[-1]["status"] == "undone"
    assert ledger_event["status"] == "undone"
    assert marker["state"] == "undone"
    assert (
        journal[-1]["details"]["artifact_manifest"]
        == ledger_event["details"]["artifact_manifest"]
    )
    assert (
        marker["terminal_details"]["artifact_manifest"]
        == ledger_event["details"]["artifact_manifest"]
    )
    assert not executor.edit_history


def test_undo_snapshot_failure_keeps_committed_marker_and_scene(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    durable_write = executor._durable_atomic_write_json

    def fail_snapshot_manifest(path, payload):
        if path.name == "snapshot_manifest.json":
            raise OSError("snapshot disk unavailable")
        return durable_write(path, payload)

    executor._durable_atomic_write_json = fail_snapshot_manifest
    ok, error = executor._undo_latest_edit(record_undo_event=True)

    assert ok is False
    assert "crash-safe typed initializer undo" in error
    assert (tmp_path / "live.blend").read_bytes() == b"blend-after"
    assert executor._initializer_ledger["active_transactions"][0]["id"] == txid
    assert executor.edit_history == [str(tmp_path / "after.blend")]
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "committed"
    assert not (executor._runtime_transaction_dir(txid) / "before.blend").exists()


def test_typed_undo_audit_failure_leaves_durable_decision_for_rollforward(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)

    def fail_audit(_event):
        raise OSError("audit disk unavailable")

    executor._append_mutation_event = fail_audit
    ok, error = executor._undo_latest_edit(record_undo_event=True)

    assert ok is True
    assert "durably committed" in error
    assert "restart the initializer" in error
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"
    assert not executor._initializer_ledger["active_transactions"]
    assert not executor.edit_history
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_commit_decided"

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()

    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undone"
    assert marker["terminal_details"]["startup_rollforward"] is True
    journal = [
        json.loads(line)
        for line in restarted._mutation_journal_path().read_text().splitlines()
    ]
    assert sum(event.get("status") == "undone" for event in journal) == 1
    assert (
        sum(
            event.get("id") == txid and event.get("status") == "undone"
            for event in restarted._initializer_ledger["events"]
        )
        == 1
    )


def test_typed_undo_ledger_failure_after_journal_rolls_forward_once(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    persist = executor._persist_initializer_ledger
    persist_calls = 0

    def fail_second_persist():
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise OSError("ledger disk unavailable")
        return persist()

    executor._persist_initializer_ledger = fail_second_persist
    ok, warning = executor._undo_latest_edit(record_undo_event=True)

    assert ok is True
    assert "bookkeeping remains recovery-pending" in warning
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_commit_decided"
    journal = [
        json.loads(line)
        for line in executor._mutation_journal_path().read_text().splitlines()
    ]
    assert sum(event.get("status") == "undone" for event in journal) == 1

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()
    journal_after = restarted._mutation_journal_path().read_bytes()
    ledger_after = (tmp_path / "ledger.json").read_bytes()
    marker_after = restarted._runtime_recovery_marker_path(txid).read_bytes()
    restarted._recover_incomplete_runtime_mutations()

    assert restarted._mutation_journal_path().read_bytes() == journal_after
    assert (tmp_path / "ledger.json").read_bytes() == ledger_after
    assert restarted._runtime_recovery_marker_path(txid).read_bytes() == marker_after
    assert (
        sum(
            event.get("id") == txid and event.get("status") == "undone"
            for event in restarted._initializer_ledger["events"]
        )
        == 1
    )


def test_corrupt_undone_state_never_rolls_back_a_durable_undo_decision(tmp_path):
    executor, txid = _prepare_typed_pose_undo(tmp_path)

    def fail_audit(_event):
        raise OSError("audit disk unavailable")

    executor._append_mutation_event = fail_audit
    ok, _warning = executor._undo_latest_edit(record_undo_event=True)
    assert ok is True
    (tmp_path / "live.blend").write_bytes(b"corrupt-undone-state")

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted._recover_incomplete_runtime_mutations()

    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_commit_decided"
    assert marker["terminal_details"]["undo_event"]["status"] == "undone"
    assert marker["recovery_error"]["phase"] == "startup_recovery"
    assert (tmp_path / "live.blend").read_bytes() == b"corrupt-undone-state"

    restarted_again = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="cannot recover initializer transaction"):
        restarted_again._recover_incomplete_runtime_mutations()
    assert (tmp_path / "live.blend").read_bytes() == b"corrupt-undone-state"


@pytest.mark.parametrize("kind", ["edit_object_pose", "execute_and_evaluate_objects"])
def test_interrupted_typed_undo_restores_last_committed_state_on_startup(
    tmp_path, kind
):
    executor, txid = _prepare_typed_pose_undo(tmp_path, kind=kind)
    committed_ledger = json.loads(json.dumps(executor._initializer_ledger))
    executor._runtime_transaction_snapshots(
        txid,
        kind=kind,
        ledger_after_request=committed_ledger,
        purpose="undo",
    )

    (tmp_path / "live.blend").write_bytes(b"partially-undone-blend")
    scene = tmp_path / "scene"
    (scene / "scene_graph.json").write_text('{"nodes": []}')
    (scene / "placement.json").write_text('{"objects": []}')
    executor._initializer_ledger = json.loads(json.dumps(executor._ledger_base))
    executor._persist_initializer_ledger()

    restarted = _bare_executor(tmp_path, load_ledger=True)

    def restore_graph(payload, *, durable=False):
        graph_path = scene / "scene_graph.json"
        if payload is None:
            graph_path.unlink(missing_ok=True)
        else:
            restarted._atomic_write_bytes(graph_path, payload)

    restarted._restore_scene_graph_bytes = restore_graph
    restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"blend-after"
    assert (
        json.loads((scene / "scene_graph.json").read_text())["nodes"][0]["id"]
        == "mug#0"
    )
    assert (
        json.loads((scene / "placement.json").read_text())["objects"][0]["mesh_name"]
        == "obj_mug_0"
    )
    assert restarted._initializer_ledger["active_transactions"][0]["id"] == txid
    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "committed"
    assert marker["terminal_details"]["startup_recovered_undo"] is True
    assert not (restarted._runtime_transaction_dir(txid) / "before.blend").exists()
    journal = [
        json.loads(line)
        for line in restarted._mutation_journal_path().read_text().splitlines()
    ]
    assert [row["status"] for row in journal] == ["committed"]
    # The downstream reporting reader must accept the recovered durable commit.
    from lib.tools.geometry.initializer_pose_policy import initializer_pose_exclusions

    canonical_ledger = scene / "stages/0/initializer_object_transactions/ledger.json"
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)
    canonical_ledger.write_bytes(Path(restarted.initializer_ledger_path).read_bytes())
    assert set(initializer_pose_exclusions(scene, ["obj_mug_0"])) == {"obj_mug_0"}
    before = restarted._mutation_journal_path().read_bytes()
    restarted._recover_incomplete_runtime_mutations()
    assert restarted._mutation_journal_path().read_bytes() == before
    assert set(initializer_pose_exclusions(scene, ["obj_mug_0"])) == {"obj_mug_0"}


@pytest.mark.parametrize("kind", ["edit_object_pose", "execute_and_evaluate_objects"])
def test_failed_undo_preserves_commit_authorization(tmp_path, monkeypatch, kind):
    from lib.tools.geometry.initializer_pose_policy import initializer_pose_exclusions

    executor, txid = _prepare_typed_pose_undo(tmp_path, kind=kind)
    manifest = executor._typed_initializer_artifact_manifest

    def fail_before_undo_decision(**kwargs):
        if kwargs["state"] == "undone":
            raise RuntimeError("injected failure before undo decision")
        return manifest(**kwargs)

    monkeypatch.setattr(
        executor, "_typed_initializer_artifact_manifest", fail_before_undo_decision
    )
    ok, message = executor._undo_latest_edit(record_undo_event=True)
    assert not ok and "injected failure" in message
    assert (tmp_path / "live.blend").read_bytes() == b"blend-after"
    journal = executor._initializer_mutation_journal_events()
    assert [row["status"] for row in journal] == ["committed", "error"]
    executor._recover_incomplete_runtime_mutations()
    ledger = (
        Path(executor.moge_dir) / "stages/0/initializer_object_transactions/ledger.json"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(Path(executor.initializer_ledger_path).read_bytes())
    assert set(initializer_pose_exclusions(Path(executor.moge_dir), ["obj_mug_0"])) == {
        "obj_mug_0"
    }


def test_historical_startup_annotation_keeps_original_authorization(tmp_path):
    from lib.tools.geometry.initializer_pose_policy import initializer_pose_exclusions

    executor, txid = _prepare_typed_pose_undo(tmp_path)
    manifest = executor._initializer_ledger["active_transactions"][0][
        "artifact_manifest"
    ]
    event = {
        "transaction_id": txid,
        "kind": "edit_object_pose",
        "status": "committed",
        "details": {
            "phase": "startup_undo_recovery",
            "scene_mutation": "interrupted_undo_abandoned",
            "artifact_manifest": manifest,
        },
    }
    executor._append_mutation_event(event)
    executor._reconcile_recovered_terminal_event(event)
    executor._recover_incomplete_runtime_mutations()
    ledger = (
        Path(executor.moge_dir) / "stages/0/initializer_object_transactions/ledger.json"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(Path(executor.initializer_ledger_path).read_bytes())
    assert set(initializer_pose_exclusions(Path(executor.moge_dir), ["obj_mug_0"])) == {
        "obj_mug_0"
    }


def test_durable_undo_decision_finishes_interrupted_marker_on_startup(
    tmp_path,
):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    set_recovery_state = executor._set_runtime_recovery_state

    def crash_before_undo_marker(txid_arg, state, *, details=None):
        if state == "undone":
            raise SystemExit("simulated process death")
        return set_recovery_state(txid_arg, state, details=details)

    executor._set_runtime_recovery_state = crash_before_undo_marker
    with pytest.raises(SystemExit, match="simulated process death"):
        executor._undo_latest_edit(record_undo_event=True)

    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_commit_decided"
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()

    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undone"
    assert marker["terminal_details"]["startup_rollforward"] is True
    assert not restarted._initializer_ledger["active_transactions"]
    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"


def test_undo_decision_delegate_failure_keeps_undo_prepared_for_startup_rollback(
    tmp_path,
):
    """A pre-publication failure cannot make the rewind appear committed."""
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    set_recovery_state = executor._set_runtime_recovery_state

    def fail_before_decision(txid_arg, state, *, details=None):
        if state == "undo_commit_decided":
            raise OSError("decision publication delegate unavailable")
        return set_recovery_state(txid_arg, state, details=details)

    executor._set_runtime_recovery_state = fail_before_decision
    ok, warning = executor._undo_latest_edit(record_undo_event=True)

    assert ok is False
    assert "uncertain commit-point publication" in warning
    assert executor.edit_history == [str(tmp_path / "after.blend")]
    marker = json.loads(executor._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "undo_prepared"

    restarted = _bare_executor(tmp_path, load_ledger=True)

    def restore_graph(payload, *, durable=False):
        graph_path = tmp_path / "scene" / "scene_graph.json"
        if payload is None:
            graph_path.unlink(missing_ok=True)
        else:
            writer = (
                restarted._durable_atomic_write_bytes
                if durable
                else restarted._atomic_write_bytes
            )
            writer(graph_path, payload)

    restarted._restore_scene_graph_bytes = restore_graph
    restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"blend-after"
    assert restarted._initializer_ledger["active_transactions"][0]["id"] == txid
    marker = json.loads(restarted._runtime_recovery_marker_path(txid).read_text())
    assert marker["state"] == "committed"


def test_undo_decision_directory_fsync_failure_rolls_surviving_decision_forward_once(
    tmp_path, monkeypatch
):
    """After-replace fsync failure is ambiguous; a surviving decision wins on restart."""
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    tx_dir = executor._runtime_transaction_dir(txid)
    fsync_directory = Executor._fsync_directory
    failed_once = False

    def fail_after_decision_replace(path):
        nonlocal failed_once
        marker_path = tx_dir / "recovery.json"
        if (
            not failed_once
            and Path(path) == tx_dir
            and marker_path.is_file()
            and json.loads(marker_path.read_text()).get("state")
            == "undo_commit_decided"
        ):
            failed_once = True
            raise OSError("tx-directory fsync unavailable after replace")
        return fsync_directory(path)

    monkeypatch.setattr(
        Executor, "_fsync_directory", staticmethod(fail_after_decision_replace)
    )
    ok, warning = executor._undo_latest_edit(record_undo_event=True)

    assert failed_once is True
    assert ok is False
    assert "uncertain commit-point publication" in warning
    assert executor.edit_history == [str(tmp_path / "after.blend")]
    marker = json.loads((tx_dir / "recovery.json").read_text())
    assert marker["state"] == "undo_commit_decided"

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()

    assert (tmp_path / "live.blend").read_bytes() == b"blend-before"
    journal = [
        json.loads(line)
        for line in restarted._mutation_journal_path().read_text().splitlines()
    ]
    assert sum(event.get("status") == "undone" for event in journal) == 1
    assert (
        sum(
            event.get("id") == txid and event.get("status") == "undone"
            for event in restarted._initializer_ledger["events"]
        )
        == 1
    )
    assert json.loads((tx_dir / "recovery.json").read_text())["state"] == "undone"


def test_transaction_directory_parent_barrier_precedes_snapshot_scene_mutation(
    tmp_path, monkeypatch
):
    """The tx dentry is durable before snapshots can publish a recovery marker."""
    executor = _bare_executor(tmp_path)
    txid = 1
    tx_dir = executor._runtime_transaction_dir(txid)
    transactions_root = tx_dir.parent
    fsync_directory = Executor._fsync_directory
    barriers = []

    def record_barrier(path):
        barriers.append(Path(path))
        return fsync_directory(path)

    monkeypatch.setattr(Executor, "_fsync_directory", staticmethod(record_barrier))
    executor._runtime_transaction_snapshots(
        txid,
        kind="edit_object_pose",
        ledger_after_request=executor._empty_initializer_ledger(),
    )

    assert transactions_root in barriers
    assert (tx_dir / "recovery.json").is_file()

    failing_root = tmp_path / "barrier-failure"
    failing_root.mkdir()
    failing = _bare_executor(failing_root)
    failing_tx_dir = failing._runtime_transaction_dir(txid)
    failing_transactions_root = failing_tx_dir.parent
    live_before = (failing_root / "live.blend").read_bytes()

    def fail_transaction_parent_barrier(path):
        if Path(path) == failing_transactions_root:
            raise OSError("transactions parent fsync unavailable")
        return fsync_directory(path)

    monkeypatch.setattr(
        Executor,
        "_fsync_directory",
        staticmethod(fail_transaction_parent_barrier),
    )
    with pytest.raises(OSError, match="transactions parent fsync unavailable"):
        failing._runtime_transaction_snapshots(
            txid,
            kind="edit_object_pose",
            ledger_after_request=failing._empty_initializer_ledger(),
        )

    assert (failing_root / "live.blend").read_bytes() == live_before
    assert not (failing_tx_dir / "recovery.json").exists()
    assert not (failing_root / "scene" / "scene_graph.json").exists()


def _authored_code_test_state(tmp_path):
    executor = _bare_executor(tmp_path)
    live = Path(executor.blender_save)
    live.write_bytes(b"blend-before")
    recovery = tmp_path / "typed-before.blend"
    recovery.write_bytes(b"blend-before")
    executor.blender_file = str(live)
    return executor, live, recovery


def test_initializer_authored_code_promotes_only_candidate_and_clears_capability(
    tmp_path,
):
    executor, live, recovery = _authored_code_test_state(tmp_path)
    code = "import bpy\nbpy.context.scene['authored'] = True\n"
    observed = {}

    def execute_candidate(script_path, render_path):
        candidate = Path(executor.blender_save)
        observed["candidate"] = candidate
        observed["allow"] = dict(executor._composition_mesh_transaction)
        observed["script"] = Path(script_path).read_text()
        assert render_path == "__norender__"
        assert Path(executor.blender_file) == candidate
        assert candidate.name == "authored_candidate.blend"
        assert candidate.read_bytes() == b"blend-before"
        assert live.read_bytes() == b"blend-before"
        assert recovery.read_bytes() == b"blend-before"
        candidate.write_bytes(b"blend-after")
        return True, [], "accepted", ""

    executor._execute_blender = execute_candidate
    result = executor._run_initializer_authored_code(
        code=code,
        kind="composition_edit_object_mesh",
        transaction_id=7,
        object_id="plate#0",
        mesh_name="obj_plate_0",
        live_before=recovery,
    )

    assert live.read_bytes() == b"blend-after"
    assert recovery.read_bytes() == b"blend-before"
    assert observed["allow"] == {
        "transaction_id": 7,
        "added_objects": ["obj_plate_0"],
        "removed_objects": ["obj_plate_0"],
    }
    assert observed["script"].startswith(
        "# Backend-owned procedural object identity. Do not reassign.\n"
        'TARGET_OBJECT_NAME = "obj_plate_0"\n'
        'TARGET_OBJECT_ID = "plate#0"\n'
        "TRANSACTION_ID = 7\n\n"
    )
    assert (
        f"exec(compile({code!r}, '<authored-code>', 'exec'), globals())"
        in observed["script"]
    )
    assert result["code_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    assert (
        result["candidate_blend_sha256"] == hashlib.sha256(b"blend-after").hexdigest()
    )
    assert not observed["candidate"].exists()
    assert not hasattr(executor, "_composition_mesh_transaction")
    assert executor.blender_file == str(live)
    assert executor.blender_save == str(live)


@pytest.mark.parametrize("tamper_target", ["protected_artifact", "live_blend"])
def test_initializer_authored_code_rejects_and_restores_out_of_candidate_mutation(
    tmp_path, tamper_target
):
    executor, live, recovery = _authored_code_test_state(tmp_path)
    graph = Path(executor.moge_dir) / "scene_graph.json"
    graph.write_bytes(b'{"safe": true}')

    def execute_and_tamper(_script_path, _render_path):
        Path(executor.blender_save).write_bytes(b"candidate-after")
        if tamper_target == "protected_artifact":
            graph.write_bytes(b'{"attacker": true}')
        else:
            live.write_bytes(b"attacker-live")
        return True, [], "", ""

    executor._execute_blender = execute_and_tamper

    with pytest.raises(
        RuntimeError,
        match="authored object code modified backend-owned transaction state",
    ):
        executor._run_initializer_authored_code(
            code="import bpy\n",
            kind="composition_edit_object_mesh",
            transaction_id=8,
            object_id="mug#0",
            mesh_name="obj_mug_0",
            live_before=recovery,
        )

    assert graph.read_bytes() == b'{"safe": true}'
    assert live.read_bytes() == b"blend-before"
    assert recovery.read_bytes() == b"blend-before"
    assert not hasattr(executor, "_composition_mesh_transaction")
    assert executor.blender_file == str(live)
    assert executor.blender_save == str(live)


def test_wrong_kind_marker_does_not_hide_markerless_request(tmp_path):
    executor = _bare_executor(tmp_path)
    request = {
        "transaction_id": 1,
        "kind": "edit_object_pose",
        "status": "requested",
        "request": {"reason": "wrong pose"},
    }
    other_terminal = {
        "transaction_id": 1,
        "kind": "add_object",
        "status": "rolled_back",
        "details": {"artifact_manifest": {"sentinel": "other-kind"}},
    }
    for event in (request, other_terminal):
        executor._append_mutation_event(event)
        executor._reconcile_recovered_terminal_event(event)

    marker_path = executor._runtime_recovery_marker_path(1)
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": 1,
                "kind": "add_object",
                "state": "rolled_back",
                "terminal_details": {"artifact_manifest": {"sentinel": "other-kind"}},
            }
        )
    )

    restarted = _bare_executor(tmp_path, load_ledger=True)
    restarted._recover_incomplete_runtime_mutations()

    journal = restarted._initializer_mutation_journal_events()
    pose_statuses = [
        event["status"]
        for event in journal
        if event["transaction_id"] == 1 and event.get("kind") == "edit_object_pose"
    ]
    assert pose_statuses == ["requested", "error"]
    assert [
        event["status"]
        for event in restarted._initializer_ledger["events"]
        if event.get("id") == 1 and event.get("kind") == "edit_object_pose"
    ] == ["requested", "error"]


def test_markerless_committed_event_cannot_be_hidden_by_later_error(tmp_path):
    executor = _bare_executor(tmp_path)
    executor._persist_initializer_ledger()
    for status in ("requested", "committed", "error"):
        event = {
            "transaction_id": 1,
            "kind": "edit_object_pose",
            "status": status,
        }
        if status == "requested":
            event["request"] = {"reason": "wrong pose"}
        else:
            event["details"] = {"scene_mutation": status}
        executor._append_mutation_event(event)
    journal_path = executor._mutation_journal_path()
    ledger_path = tmp_path / "ledger.json"
    journal_before = journal_path.read_bytes()
    ledger_before = ledger_path.read_bytes()
    live_before = (tmp_path / "live.blend").read_bytes()

    restarted = _bare_executor(tmp_path, load_ledger=True)
    with pytest.raises(RuntimeError, match="committed/undone event"):
        restarted._recover_incomplete_runtime_mutations()

    assert journal_path.read_bytes() == journal_before
    assert ledger_path.read_bytes() == ledger_before
    assert (tmp_path / "live.blend").read_bytes() == live_before


def test_torn_final_journal_suffix_is_repaired(tmp_path):
    executor = _bare_executor(tmp_path)
    executor._append_mutation_event(
        {
            "transaction_id": 1,
            "kind": "edit_object_pose",
            "status": "requested",
        }
    )
    journal_path = executor._mutation_journal_path()
    durable_prefix = journal_path.read_bytes()
    journal_path.write_bytes(durable_prefix + b'{"schema_version": 1')

    events = executor._initializer_mutation_journal_events()

    assert len(events) == 1
    assert events[0]["status"] == "requested"
    assert journal_path.read_bytes() == durable_prefix


@pytest.mark.parametrize("stage", [None, "initializer"])
def test_initializer_recovery_and_guard_skip_composition_marker(tmp_path, stage):
    executor, txid = _prepare_typed_pose_undo(tmp_path)
    marker_path = executor._runtime_recovery_marker_path(txid)
    marker = json.loads(marker_path.read_text())
    if stage:
        marker["stage"] = stage
    marker_path.write_text(json.dumps(marker))
    foreign = executor._runtime_recovery_marker_path(txid + 1)
    foreign.parent.mkdir()
    foreign.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": txid + 1,
                "stage": "composition",
                "kind": "composition_edit_object_mesh",
                "state": "committed",
            }
        )
    )
    executor._append_mutation_event(
        {
            "transaction_id": txid + 1,
            "stage": "composition",
            "kind": "composition_edit_object_mesh",
            "status": "committed",
        }
    )
    foreign_before = foreign.read_bytes()
    executor._recover_incomplete_runtime_mutations()
    executor._assert_no_incomplete_runtime_mutation()
    assert foreign.read_bytes() == foreign_before
    assert [
        event["transaction_id"]
        for event in executor._initializer_mutation_journal_events()
    ] == [txid]


@pytest.mark.parametrize(
    "stage,kind",
    [
        ("initializer", "composition_edit_object_mesh"),
        ("composition", "edit_object_pose"),
        ("initializer", "unknown_kind"),
    ],
)
def test_initializer_recovery_does_not_ignore_misbound_kind(tmp_path, stage, kind):
    executor = _bare_executor(tmp_path)
    marker = executor._runtime_recovery_marker_path(1)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": 1,
                "stage": stage,
                "kind": kind,
                "state": "committed",
            }
        )
    )
    for operation in (
        executor._recover_incomplete_runtime_mutations,
        executor._assert_no_incomplete_runtime_mutation,
    ):
        with pytest.raises(RuntimeError, match="invalid initializer recovery marker"):
            operation()


def test_complete_malformed_middle_journal_frame_fails_without_repair(tmp_path):
    executor = _bare_executor(tmp_path)
    executor._append_mutation_event(
        {
            "transaction_id": 1,
            "kind": "edit_object_pose",
            "status": "requested",
        }
    )
    journal_path = executor._mutation_journal_path()
    valid_frame = journal_path.read_bytes()
    malformed = valid_frame + b"{not-json}\n" + valid_frame
    journal_path.write_bytes(malformed)

    with pytest.raises(RuntimeError, match="mutation journal line 2 is invalid"):
        executor._initializer_mutation_journal_events()

    assert journal_path.read_bytes() == malformed


def test_initializer_scene_lock_serializes_processes(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    context = multiprocessing.get_context("fork")
    first_acquired = context.Event()
    first_release = context.Event()
    second_acquired = context.Event()
    second_release = context.Event()
    second_release.set()
    first = context.Process(
        target=_hold_initializer_scene_lock,
        args=(str(scene), first_acquired, first_release),
    )
    second = context.Process(
        target=_hold_initializer_scene_lock,
        args=(str(scene), second_acquired, second_release),
    )

    first.start()
    try:
        assert first_acquired.wait(5)
        second.start()
        assert not second_acquired.wait(0.25)
        first_release.set()
        assert second_acquired.wait(5)
    finally:
        first_release.set()
        second_release.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_composition_startup_accepts_a_committed_mesh_marker_after_later_edits(
    tmp_path,
):
    """2026-09-15 (v5accept robolab_clutter_shelf): the final-settle repair round starts a
    SECOND composition session; a committed composition mesh marker must not be re-verified
    against artifacts that later layout edits and the final certify legitimately rewrote."""
    executor = _bare_executor(tmp_path)
    executor.root_stage_name = "composition"
    executor.harness_profile_manifest["capabilities"].update(
        composition_mesh_edit=True, strict_post_edit_physics=True
    )
    marker = executor._runtime_recovery_marker_path(7)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": 7,
                "stage": "composition",
                "kind": "composition_edit_object_mesh",
                "state": "committed",
                "payloads_pruned": True,
            }
        )
    )
    executor._append_mutation_event(
        {
            "transaction_id": 7,
            "stage": "composition",
            "kind": "composition_edit_object_mesh",
            "status": "committed",
            # hashes that no longer match anything on disk — the scene moved on
            "details": {
                "artifact_manifest": {
                    "schema_version": 1,
                    "artifacts": {"scene_graph.json": {"sha256": "0" * 64}},
                }
            },
        }
    )
    executor._recover_incomplete_composition_mesh_mutations()  # must not raise
    assert json.loads(marker.read_text())["state"] == "committed"
