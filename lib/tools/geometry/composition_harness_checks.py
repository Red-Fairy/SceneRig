"""Focused tests for the opt-in GPT-6 composition harness."""

import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

# Developer/test checkouts intentionally omit local credentials/launcher paths. The
# Executor import below only needs their module contracts, never their contents.
if (
    "lib.utils._api_keys" not in sys.modules
    and importlib.util.find_spec("lib.utils._api_keys") is None
):
    api_keys = types.ModuleType("lib.utils._api_keys")
    for provider in ("CLAUDE", "FIREWORKS", "GEMINI", "OPENAI", "QWEN"):
        setattr(api_keys, f"{provider}_API_KEY", "")
        setattr(api_keys, f"{provider}_BASE_URL", "")
    sys.modules["lib.utils._api_keys"] = api_keys
if (
    "lib.utils._path" not in sys.modules
    and importlib.util.find_spec("lib.utils._path") is None
):
    local_paths = types.ModuleType("lib.utils._path")
    local_paths.path_to_cmd = {}
    sys.modules["lib.utils._path"] = local_paths

import lib.tools.blender.exec as blender_exec_module
import lib.tools.geometry.register as register_module
from lib.agents.harness_profile import resolve_harness_profile
from lib.agents.prompt_builder import build_object_state_table
from lib.prompts.static_scene.generators.composition import (
    composition_generator_system,
    static_scene_composition_generator_system,
)
from lib.tools.blender.exec import Executor
from lib.tools.geometry.composition_physics import (
    CompositionPhysics,
    PhysicsSettlementRejected,
    _strict_settlement_rejection,
)
from lib.tools.geometry.register import (
    PoseSession,
    _merge_runtime_pose_inventory,
    _profile_capability_enabled,
)


def _commit(**overrides):
    result = {
        "members": ["obj_mug_0"],
        "totals": {"obj_mug_0": np.eye(4)},
        "blend_deltas": {"obj_mug_0": np.eye(4)},
        "cum_tilt_deg": {"obj_mug_0": 0.0},
        "lift_mm": 0.0,
        "capsized": [],
        "penetration": None,
        "penetrations": {},
        "converged": True,
    }
    result.update(overrides)
    return result


def test_composition_prompt_addendum_is_profile_and_capability_gated():
    manifest = {"capabilities": {"composition_direct_pose_edit": True}}
    assert composition_generator_system() == static_scene_composition_generator_system
    assert (
        composition_generator_system("baseline", manifest)
        == static_scene_composition_generator_system
    )
    disabled = {"capabilities": {"composition_direct_pose_edit": False}}
    assert (
        composition_generator_system("gpt6_v1", disabled)
        == static_scene_composition_generator_system
    )

    prompt = composition_generator_system("gpt6_v1", manifest)
    assert "edit_object_poses" in prompt
    assert "[Direct Pose Editing]" in prompt
    assert "correct pose with move() FIRST" in prompt
    assert "Each object accepts at most two committed typed pose" in prompt
    assert "simulator's initial condition" in prompt
    assert "POST-SIMULATION" in prompt


def test_gpt6_default_profile_prompt_is_layout_first_without_the_typed_tool():
    from lib.agents.harness_profile import resolve_harness_profile

    prompt = composition_generator_system("gpt6_v1", resolve_harness_profile("gpt6_v1"))
    assert "[Layout Editing]" in prompt
    assert "execute_and_evaluate leaves it armed" in prompt
    assert "physics-settled before the render" in prompt
    assert "re-cooks its collider" in prompt
    assert "[Investigated Mesh Repair]" in prompt  # mesh route stays, gated
    assert "edit_object_poses" not in prompt
    assert "[Direct Pose Editing]" not in prompt
    assert "move() FIRST" not in prompt
    assert "re-layout" not in prompt  # composition corrects, it does not redesign


def test_profile_capability_never_enables_baseline():
    manifest = {"capabilities": {"strict_post_edit_physics": True}}
    assert not _profile_capability_enabled(
        "baseline", manifest, "strict_post_edit_physics"
    )
    assert _profile_capability_enabled("gpt6_v1", manifest, "strict_post_edit_physics")
    assert not _profile_capability_enabled(
        "gpt6_v1",
        {"capabilities": {"strict_post_edit_physics": False}},
        "strict_post_edit_physics",
    )


def test_direct_yaw_feedback_routes_to_typed_tool_only_for_gpt6():
    executor = Executor.__new__(Executor)
    executor._size_locked_ids = lambda: set()
    weak_yaw = {
        "render_px": 100,
        "mask_px": 100,
        "overlap": 0.5,
        "shift_px": None,
        "yaw_deg": 30.0,
        "yaw_aniso": 1.0,
        "flagged": False,
    }

    executor.harness_profile = "baseline"
    executor.harness_profile_manifest = {
        "capabilities": {"composition_direct_pose_edit": True}
    }
    baseline = "\n".join(
        executor._object_hint_lines(
            "mug#0", None, weak_yaw, 0.4, emit_orientation_check=False
        )
    )
    assert "execute_and_evaluate to yaw" in baseline
    assert "edit_object_poses" not in baseline

    executor.harness_profile = "gpt6_v1"
    routed = "\n".join(
        executor._object_hint_lines(
            "mug#0", None, weak_yaw, 0.4, emit_orientation_check=False
        )
    )
    assert "edit_object_poses" in routed
    assert "physics simulation follows" in routed
    assert "execute_and_evaluate to yaw" not in routed


def _pose_normalization_executor():
    executor = Executor.__new__(Executor)
    executor._name2id = lambda: {"obj_fork_0": "fork#0"}
    session = SimpleNamespace(
        objects=lambda: ["fork#0"],
        ctx={"fork#0": {"mesh_name": "obj_fork_0"}},
    )
    return executor, session


def test_composition_direct_pose_undo_records_terminal_audit(tmp_path):
    before = tmp_path / "before.blend"
    live = tmp_path / "live.blend"
    history = tmp_path / "history.blend"
    before.write_bytes(b"before")
    live.write_bytes(b"after")
    history.write_bytes(b"after")
    executor = Executor.__new__(Executor)
    executor.root_stage_name = "composition"
    executor.blender_save = str(live)
    executor.base_state = str(before)
    executor.edit_history = [str(history)]
    executor._edit_meta = [
        {
            "kind": "edit_object_poses",
            "transaction_id": "composition-test-tx",
            "object_id": "fork#0,knife#0",
            "edit_token": "edit_object_poses:1:3",
        }
    ]
    executor._ledger_history = []
    executor._graph_history = []
    executor._graph_base = None
    executor._read_scene_graph_bytes = lambda: None
    executor._post_flip_investigation_required = None
    executor._pose_dirty = False
    executor._armed = {"fork#0"}
    events = []
    executor._record_mutation_status = lambda txid, kind, status, details=None: (
        events.append((txid, kind, status, details))
    )

    ok, warning = executor._undo_latest_edit(record_undo_event=True)

    assert ok is True
    assert warning == ""
    assert live.read_bytes() == b"before"
    assert executor.edit_history == []
    assert events == [
        (
            "composition-test-tx",
            "edit_object_poses",
            "undone",
            {
                "scene_mutation": "restored_pre_transaction_state",
                "object_ids": ["fork#0", "knife#0"],
                "edit_token": "edit_object_poses:1:3",
            },
        )
    ]


@pytest.mark.parametrize(
    ("euler_deg", "quaternion_wxyz"),
    [
        ([0, 0, 180], [0, 0, 0, 1]),
        # Rounded Blender XYZ Euler conversion for a nontrivial attitude.
        ([10, -20, 30], [0.94371438, 0.12767944, -0.14487812, 0.26853582]),
        # Live random-toast pair: six-digit serialization differs by ~0.187 deg.
        ([18.37, -25.3, -79.6], [0.762572, -0.017184, -0.266158, -0.589356]),
        # Quaternion sign is not an attitude difference.
        ([0, 0, 180], [0, 0, 0, -1]),
    ],
)
def test_pose_normalization_canonicalizes_equivalent_dual_rotation_to_euler(
    euler_deg, quaternion_wxyz
):
    executor, session = _pose_normalization_executor()

    normalized = executor._normalize_typed_pose_edits(
        [
            {
                "object": "fork#0",
                "mode": "delta",
                "rotation_euler_deg": euler_deg,
                "rotation_quaternion_wxyz": quaternion_wxyz,
            }
        ],
        session,
    )

    assert normalized[0]["rotation_euler_deg"] == [float(value) for value in euler_deg]
    assert "rotation_quaternion_wxyz" not in normalized[0]


def test_pose_normalization_preserves_quaternion_only_requests():
    executor, session = _pose_normalization_executor()

    normalized = executor._normalize_typed_pose_edits(
        [
            {
                "object": "fork#0",
                "mode": "absolute",
                "rotation_quaternion_wxyz": [0.9238795, 0, 0, 0.3826834],
            }
        ],
        session,
    )

    assert normalized[0]["rotation_quaternion_wxyz"] == [
        0.9238795,
        0.0,
        0.0,
        0.3826834,
    ]
    assert "rotation_euler_deg" not in normalized[0]


def test_pose_normalization_rejects_conflicting_dual_rotation():
    executor, session = _pose_normalization_executor()

    with pytest.raises(ValueError, match="conflicting Euler and quaternion"):
        executor._normalize_typed_pose_edits(
            [
                {
                    "object": "fork#0",
                    "mode": "delta",
                    "rotation_euler_deg": [0, 0, 180],
                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                }
            ],
            session,
        )


def test_pose_normalization_rejects_one_degree_vs_fifteen_degree_rotation():
    executor, session = _pose_normalization_executor()

    with pytest.raises(ValueError, match="conflicting Euler and quaternion"):
        executor._normalize_typed_pose_edits(
            [
                {
                    "object": "fork#0",
                    "mode": "delta",
                    "rotation_euler_deg": [0, 0, 1],
                    "rotation_quaternion_wxyz": [
                        0.9914448614,
                        0,
                        0,
                        0.1305261922,
                    ],
                }
            ],
            session,
        )


def test_pose_normalization_rejects_unrepresentable_quaternion_magnitude():
    executor, session = _pose_normalization_executor()

    with pytest.raises(ValueError, match="quaternion cannot be zero"):
        executor._normalize_typed_pose_edits(
            [
                {
                    "object": "fork#0",
                    "mode": "delta",
                    "rotation_quaternion_wxyz": [1e308, 1e308, 1e308, 1e308],
                }
            ],
            session,
        )


def test_initialize_tool_menus_preserve_baseline_and_only_extend_gpt6(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class StubExecutor:
        def __init__(self, **kwargs):
            self.harness_profile = kwargs["harness_profile"]
            self.harness_profile_manifest = kwargs["harness_profile_manifest"]

        def _has_capability(self, capability: str) -> bool:
            return _profile_capability_enabled(
                self.harness_profile,
                self.harness_profile_manifest,
                capability,
            )

    monkeypatch.setattr(blender_exec_module, "Executor", StubExecutor)
    baseline_expected = {
        "texture": [
            "execute_and_evaluate",
            "render_current_scene",
            "get_scene_info",
            "undo_last_step",
        ],
        "lighting": [
            "execute_and_evaluate",
            "render_current_scene",
            "get_scene_info",
            "undo_last_step",
        ],
        "composition": [
            "investigate_objects",
            "move",
            "execute_and_evaluate",
            "render_current_scene",
            "render_bev",
            "get_scene_info",
            "check_rules_enforced",
            "undo_last_step",
        ],
        "initializer": [
            "execute_and_evaluate",
            "render_current_scene",
            "render_bev",
            "get_scene_info",
            "check_rules_enforced",
            "resolve_yaw_advisory",
            "build_root_surface",
            "remove_root_surface",
            "nudge_object",
            "bypass",
            "undo_last_step",
        ],
    }

    def menu(stage: str, profile: str) -> tuple[list[str], set[str]]:
        result = blender_exec_module.initialize(
            {
                "blender_command": "blender",
                "blender_file": "scene.blend",
                "blender_script": "wrapper.py",
                "output_dir": str(tmp_path / stage / profile),
                "root_stage_name": stage,
                "harness_profile": profile,
                "harness_profile_manifest": resolve_harness_profile(profile),
            }
        )
        assert result["status"] == "success"
        output = result["output"]
        names = [tool["function"]["name"] for tool in output["tool_configs"]]
        return names, set(output["tool_effects"])

    for stage, expected in baseline_expected.items():
        names, effects = menu(stage, "baseline")
        assert names == expected
        assert not {
            "add_object",
            "remove_runtime_object",
            "edit_object_mesh",
            "edit_object_pose",
            "edit_object_poses",
        } & (set(names) | effects)

    composition_names, composition_effects = menu("composition", "gpt6_v1")
    # gpt6_v1 v4 (2026-09-14): the typed pose tool is withdrawn. Composition pose edits
    # go through move() (measured) or execute_and_evaluate (free code, physics-settled
    # after every call); edit_object_mesh replaces a whole object.
    assert composition_names == [
        "investigate_objects",
        "edit_object_mesh",
        "move",
        *baseline_expected["composition"][2:],
    ]
    assert "edit_object_poses" not in composition_names
    assert "edit_object_mesh" in composition_effects
    assert "edit_object_poses" not in composition_effects

    initializer_names, initializer_effects = menu("initializer", "gpt6_v1")
    # GPT-6 initializer object changes are declarations on its specialized
    # execute_and_evaluate schema, not four independent menu entries; 2026-09-15 owner:
    # nudge_object is withdrawn from this profile (every pose change is a transaction).
    assert initializer_names == [
        n for n in baseline_expected["initializer"] if n != "nudge_object"
    ]
    assert "execute_and_evaluate" in initializer_effects

    for stage in ("texture", "lighting"):
        assert menu(stage, "gpt6_v1")[0] == baseline_expected[stage]


def test_runtime_inventory_adds_committed_pose_records(tmp_path):
    runtime_dir = tmp_path / "runtime_objects"
    runtime_dir.mkdir()
    mask = runtime_dir / "mug.npy"
    np.save(mask, np.ones((2, 2), dtype=bool))
    inventory = runtime_dir / "inventory.json"
    inventory.write_text(
        """{
          "objects": [
            {
              "id": "mug#0",
              "category": "mug",
              "instance": 0,
              "origin": "runtime_added",
              "status": "committed",
              "mask_path": "runtime_objects/mug.npy",
              "graph_node": {
                "id": "mug#0", "kind": "object", "parent": "table#0",
                "children": [], "runtime_added": true
              },
              "placement": {
                "category": "mug", "instance": 0, "mesh_name": "obj_mug_0",
                "center": [0, 0, 0], "size": [1, 1, 1], "depth": 1,
                "runtime_added": true
              }
            }
          ]
        }"""
    )
    graph = {
        "nodes": [
            {
                "id": "table#0",
                "kind": "root_surface",
                "parent": None,
                "children": [],
            }
        ],
        "roots": ["table#0"],
    }
    masks = {}
    placement = {}

    _merge_runtime_pose_inventory(graph, masks, placement, inventory)

    assert set(masks) == {("mug", 0)}
    assert masks[("mug", 0)]["mask_path"] == str(mask)
    assert placement[("mug", 0)]["mesh_name"] == "obj_mug_0"
    assert "mug#0" in graph["nodes"][0]["children"]
    assert {node["id"] for node in graph["nodes"]} == {"table#0", "mug#0"}


def test_source_revision_keeps_source_mask_and_overlays_placement(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask = masks_dir / "mug.npy"
    np.save(mask, np.ones((2, 2), dtype=bool))
    runtime_dir = tmp_path / "runtime_objects"
    runtime_dir.mkdir()
    inventory = runtime_dir / "inventory.json"
    inventory.write_text(
        """{
          "objects": [{
            "id": "mug#0", "category": "mug", "instance": 0,
            "origin": "source_revised", "status": "committed",
            "mask_path": "masks/mug.npy",
            "graph_node": {
              "id": "mug#0", "kind": "object", "parent": "table#0",
              "children": []
            },
            "placement": {
              "category": "mug", "instance": 0, "mesh_name": "obj_mug_0",
              "mesh_glb": "runtime_objects/mug-r2.glb", "center": [1, 2, 3],
              "size": [1, 1, 1], "depth": 2
            }
          }]
        }"""
    )
    graph = {
        "nodes": [
            {"id": "table#0", "kind": "root_surface", "children": ["mug#0"]},
            {
                "id": "mug#0",
                "kind": "object",
                "parent": "table#0",
                "children": [],
            },
        ],
        "roots": ["table#0"],
    }
    source_mask = {
        "category": "mug",
        "instance": 0,
        "mask_path": str(mask),
    }
    masks = {("mug", 0): source_mask}
    placement = {
        ("mug", 0): {
            "category": "mug",
            "instance": 0,
            "mesh_name": "obj_mug_0",
            "mesh_glb": "meshes/mug-r1.glb",
        }
    }

    _merge_runtime_pose_inventory(graph, masks, placement, inventory)

    assert masks[("mug", 0)] is source_mask
    assert placement[("mug", 0)]["mesh_glb"] == "runtime_objects/mug-r2.glb"
    assert len(graph["nodes"]) == 2


def test_runtime_mesh_revision_invalidates_geometry_dependent_pose_state(tmp_path):
    physics_dir = tmp_path / "physics"
    physics_dir.mkdir()
    previous_target = {
        "chosen": "stabilized",
        "physics_overrides": {
            "friction": 1.7,
            "angular_damping": 8.0,
            "flatten_base_mm": 2.0,
        },
        "tilt_deg": 18.0,
        "settle_total": [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.03],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    other_record = {
        "chosen": "pristine",
        "physics_overrides": {"friction": 0.9},
        "settle_total": np.eye(4).tolist(),
    }
    pose_path = physics_dir / "pose_changes.json"
    pose_path.write_text(
        json.dumps(
            {
                "run_id": "run/task",
                "mode": "incremental",
                "objects": {
                    "obj_mug_0": previous_target,
                    "obj_plate_0": other_record,
                },
                "composition_certify": {"obj_mug_0": {"tilt_deg": 0.2}},
                "composition_certify_converged": True,
                "delivered": {"obj_mug_0": {"capsized": False}},
                "certify_run_id": "run/task",
            }
        )
    )
    executor = Executor.__new__(Executor)
    executor.moge_dir = str(tmp_path)

    event = executor._invalidate_runtime_mesh_pose_state(
        "obj_mug_0", transaction_id=17, mesh_sha256="a" * 64
    )

    saved = json.loads(pose_path.read_text())
    active = saved["objects"]["obj_mug_0"]
    assert set(active) == {"settle_total", "runtime_mesh_revision"}
    assert active["settle_total"] == np.eye(4).tolist()
    assert active["runtime_mesh_revision"] == {
        "transaction_id": 17,
        "mesh_sha256": "a" * 64,
        "collider_state": "fresh_rebuild_required",
        "physics_override_state": "invalidated",
    }
    assert saved["objects"]["obj_plate_0"] == other_record
    assert saved["run_id"] == "run/task"
    assert saved["mode"] == "incremental"
    assert all(
        key not in saved
        for key in (
            "composition_certify",
            "composition_certify_converged",
            "delivered",
            "certify_run_id",
        )
    )
    assert event["invalidated_object_fields"] == sorted(previous_target)
    assert event["invalidated_late_blocks"] == [
        "composition_certify",
        "composition_certify_converged",
        "delivered",
        "certify_run_id",
    ]
    assert len(event["superseded_record_sha256"]) == 64
    assert saved["runtime_mesh_invalidations"] == [
        {key: value for key, value in event.items() if key != "pose_changes_present"}
    ]


@pytest.mark.parametrize("payload", [b"{broken", b"[]"])
def test_runtime_material_force_refresh_preserves_invalid_existing_file(
    tmp_path, payload
):
    physics_dir = tmp_path / "physics"
    physics_dir.mkdir()
    estimates_path = physics_dir / "physics_vlm.json"
    estimates_path.write_bytes(payload)
    executor = Executor.__new__(Executor)
    executor.moge_dir = str(tmp_path)

    with pytest.raises(RuntimeError, match="cannot refresh runtime material"):
        executor._estimate_runtime_physics_material(
            {"mesh_name": "obj_mug_0"},
            physical_material_hint=None,
            transaction_id=18,
            force_refresh=True,
        )

    assert estimates_path.read_bytes() == payload


def test_runtime_material_estimator_logs_never_reach_mcp_stdout(tmp_path, capsys):
    physics_dir = tmp_path / "physics"
    physics_dir.mkdir()
    (tmp_path / "placement.json").write_text(
        json.dumps({"objects": [{"mesh_name": "obj_mug_0"}]})
    )
    (physics_dir / "physics_vlm.json").write_text(
        json.dumps(
            {
                "obj_mug_0": {
                    "material": "ceramic",
                    "mass_kg": 0.4,
                    "mass_range_kg": [0.3, 0.5],
                    "mass_source": "estimated",
                    "friction": 0.45,
                    "density_kgm3": 500.0,
                    "density_gate": "ok",
                }
            }
        )
    )
    executor = Executor.__new__(Executor)
    executor.moge_dir = str(tmp_path)
    executor.physics_model = "gpt-6-astra"

    result = executor._estimate_runtime_physics_material(
        {"mesh_name": "obj_mug_0"},
        physical_material_hint=None,
        transaction_id=19,
        force_refresh=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PHYSICS_ESTIMATE_OK" in captured.err
    assert result["status"] == "estimated"
    assert result["material"] == "ceramic"


@pytest.mark.parametrize(
    ("commit", "intent", "expected"),
    [
        (_commit(converged=False), None, "did not converge"),
        (
            _commit(
                penetration={
                    "body": "obj_mug_0",
                    "other": "obj_table_0",
                    "after_mm": 14.0,
                }
            ),
            None,
            "inside obj_table_0",
        ),
        (
            _commit(
                capsized=[{"name": "obj_mug_0", "rollable": False, "tilt_deg": 72.0}]
            ),
            "preserve",
            "intended preserve",
        ),
    ],
)
def test_strict_settlement_rejection_covers_hard_failures(commit, intent, expected):
    assert expected in _strict_settlement_rejection("obj_mug_0", commit, intent)


def test_side_mode_allows_target_attitude_but_not_toppled_bystander():
    target = _commit(
        capsized=[{"name": "obj_mug_0", "rollable": False, "tilt_deg": 90.0}]
    )
    assert _strict_settlement_rejection("obj_mug_0", target, "side") is None

    bystander = _commit(
        capsized=[{"name": "obj_spoon_0", "rollable": False, "tilt_deg": 90.0}]
    )
    assert "obj_spoon_0" in _strict_settlement_rejection("obj_mug_0", bystander, "side")


def _settle_fixture(strict: bool, *, converged: bool):
    physics = CompositionPhysics.__new__(CompositionPhysics)
    physics._names = ["obj_mug_0"]
    physics._synced = {"obj_mug_0": np.eye(4)}
    physics.parents = {}
    requested = np.eye(4)
    requested[0, 3] = 0.1
    settled = requested.copy()
    matrices = iter(
        [{"obj_mug_0": requested}, {"obj_mug_0": settled}]
        if strict
        else [{"obj_mug_0": requested}]
    )
    physics._absorb_geom_edits = lambda session: []
    physics._blend_matrices = lambda session: next(matrices)
    physics.commit_move = lambda session, name: _commit(converged=converged)
    accepted = []
    rejected = []
    physics.accept = lambda session, commit: accepted.append(commit)
    physics.reject = lambda session, commit: rejected.append(commit)
    session = SimpleNamespace(
        harness_profile="gpt6_v1" if strict else "baseline",
        strict_post_edit_physics=strict,
        ctx={"mug#0": {"mesh_name": "obj_mug_0"}},
    )
    return physics, session, accepted, rejected


def test_strict_direct_edit_rejects_nonconvergence_before_accept():
    physics, session, accepted, rejected = _settle_fixture(True, converged=False)
    with pytest.raises(PhysicsSettlementRejected, match="did not converge"):
        physics.settle_edited(session)
    assert not accepted
    assert len(rejected) == 1


def test_baseline_direct_edit_retains_reviewable_nonconverged_result():
    physics, session, accepted, rejected = _settle_fixture(False, converged=False)
    reports = physics.settle_edited(session)
    assert len(reports) == 1
    assert reports[0]["converged"] is False
    assert len(accepted) == 1
    assert not rejected


@pytest.mark.parametrize(("strict", "applied"), [(False, True), (True, False)])
def test_rotate_180_kinematic_fallback_is_disabled_only_in_strict_profile(
    monkeypatch, tmp_path, strict, applied
):
    monkeypatch.setattr(register_module.np, "load", lambda path: np.ones((2, 2)))
    monkeypatch.setattr(register_module, "_isolate_iou", lambda client, names: None)
    monkeypatch.setattr(
        register_module,
        "_score_pose",
        lambda *args, **kwargs: {"score": 1.0, "iou": 1.0},
    )
    session = PoseSession.__new__(PoseSession)
    session.harness_profile = "gpt6_v1" if strict else "baseline"
    session.strict_post_edit_physics = strict
    session.physics = None
    session.client = object()
    session.work = tmp_path
    session.ctx = {
        "mug#0": {
            "mesh_name": "obj_mug_0",
            "mask_path": str(tmp_path / "mask.npy"),
            "visible_score": ["obj_mug_0"],
            "center_y": 0.0,
            "depth": 1.0,
        }
    }
    session.pose = {
        "mug#0": {
            "translate": [0.0, 0.0, 0.0],
            "euler": [0.0, 0.0, 0.0],
            "scale": 1.0,
        }
    }
    session.cur = {"mug#0": None}
    session.trace = {"mug#0": []}
    session.rounds = 0
    session.score = lambda object_id: {"score": 1.0, "iou": 1.0}
    session._dump = lambda: None
    session._invalidate_hints = lambda object_id, axis: None
    session._descendant_meshes = lambda name: []

    result = session.flip_180("mug#0")

    assert result["applied"] is applied
    assert bool(result["physics"]) is strict
    if strict:
        assert result["physics"]["failure_kind"] == "infrastructure_error"
        assert result["physics"]["recovery_succeeded"] is True
    assert session.pose["mug#0"]["euler"][2] == pytest.approx(np.pi if applied else 0.0)


@pytest.mark.parametrize(("strict", "applied"), [(False, True), (True, False)])
def test_move_kinematic_fallback_is_disabled_only_in_strict_profile(
    monkeypatch, tmp_path, strict, applied
):
    pose = {
        "translate": [0.0, 0.0, 0.0],
        "euler": [0.0, 0.0, 0.0],
        "scale": 1.0,
    }
    candidate = {**pose, "translate": [0.1, 0.0, 0.0], "_tag": "test"}
    monkeypatch.setattr(register_module.np, "load", lambda path: np.ones((2, 2)))
    monkeypatch.setattr(register_module, "_isolate_iou", lambda client, names: None)
    monkeypatch.setattr(
        register_module, "_candidates", lambda axis, current, size: [candidate]
    )
    monkeypatch.setattr(
        register_module, "_fine_candidates", lambda axis, current, size: []
    )

    def score_pose(client, name, current, *args, **kwargs):
        score = abs(float(current["translate"][0])) * 10.0
        return {"score": score, "iou": score}

    monkeypatch.setattr(register_module, "_score_pose", score_pose)
    session = PoseSession.__new__(PoseSession)
    session.harness_profile = "gpt6_v1" if strict else "baseline"
    session.strict_post_edit_physics = strict
    session.physics = None
    session.client = object()
    session.work = tmp_path
    session.ctx = {
        "mug#0": {
            "mesh_name": "obj_mug_0",
            "mask_path": str(tmp_path / "mask.npy"),
            "visible_score": ["obj_mug_0"],
            "center_y": 0.0,
            "depth": 1.0,
            "size": 1.0,
        }
    }
    session.pose = {"mug#0": pose}
    session.cur = {"mug#0": {"score": 0.0, "iou": 0.0}}
    session.trace = {"mug#0": []}
    session.rounds = 0
    session.score = lambda object_id: {"score": 0.0, "iou": 0.0}
    session._dump = lambda: None
    session._invalidate_hints = lambda object_id, axis: None

    result = session.optimize_axis("mug#0", "x-axis")

    assert result["applied"] is applied
    assert bool(result["physics"]) is strict
    assert session.pose["mug#0"]["translate"][0] == pytest.approx(
        0.1 if applied else 0.0
    )


class _FlipSession:
    def __init__(self, result, *, save_error: Exception | None = None):
        self.result = result
        self.save_error = save_error
        self.save_calls = 0
        self.rebuild_calls = 0
        self.closed = False

    def objects(self):
        return ["mug#0"]

    def flip_180(self, object_id):
        assert object_id == "mug#0"
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def save(self):
        self.save_calls += 1
        if self.save_error is not None:
            raise self.save_error

    def rebuild(self):
        self.rebuild_calls += 1

    def close(self):
        self.closed = True


def _flip_executor(tmp_path, result, *, strict=True, save_error=None):
    before = tmp_path / "before.blend"
    live = tmp_path / "live.blend"
    before.write_bytes(b"before")
    live.write_bytes(b"before")
    session = _FlipSession(result, save_error=save_error)
    executor = Executor.__new__(Executor)
    executor.root_stage_name = "composition"
    executor.harness_profile = "gpt6_v1" if strict else "baseline"
    executor.harness_profile_manifest = {
        "capabilities": {"strict_post_edit_physics": strict}
    }
    executor._pose_session = session
    executor._physics_obj = None
    executor._pose_dirty = False
    executor._freeform_edit_prepared = False
    executor._armed = {"mug#0"}
    executor._moved = {}
    executor.move_cap = 5
    executor._flipped = set()
    executor._flip_phys_rejected = set()
    executor._post_flip_investigation_required = None
    executor._post_flip_followup_events = []
    executor._edit_token_seq = 0
    executor.attempt_idx = 1
    executor.edit_history = []
    executor._ledger_history = []
    executor._graph_history = []
    executor._edit_meta = []
    executor.base_state = str(before)
    executor.blender_save = str(live)
    executor._guard_post_flip_tool_call = lambda tool: None
    executor._pose_session_get = lambda: session
    executor._ensure_undo_base = lambda: None
    executor._attach_physics = lambda current, resync: None
    executor._move_region_crop = lambda current: None
    executor._post_move_pending_note = lambda current, object_id, aspect: ""
    executor._oids_in = lambda text: text
    return executor, session


def _flip_result(*, infrastructure=False, applied=False):
    physics = {
        "accepted": applied,
        **(
            {
                "infrastructure_error": True,
                "failure_kind": "infrastructure_error",
                "reason": "Isaac RPC failed",
            }
            if infrastructure
            else {"failure_kind": "physics_rejection"}
        ),
    }
    if applied:
        physics["tilt_deg"] = 0.0
    score = {"score": 1.0, "iou": 1.0}
    return {
        "applied": applied,
        "before": score,
        "after": score,
        "physics": physics,
        "reason": physics.get("reason") or "unstable",
    }


def test_strict_flip_infrastructure_error_refunds_both_budgets_after_restore(
    tmp_path,
):
    executor, session = _flip_executor(tmp_path, _flip_result(infrastructure=True))

    result = executor.move_object("mug#0", "rotate_180")

    assert result["status"] == "error"
    assert result["output"]["failure_kind"] == "infrastructure_error"
    assert result["output"]["scene_mutation"] == "not_committed"
    assert result["output"]["recovery_succeeded"] is True
    assert result["output"]["budget_consumed"] is False
    assert "mug#0" not in executor._moved
    assert "mug#0" not in executor._flipped
    assert "mug#0" not in executor._flip_phys_rejected
    assert session.save_calls == 0
    assert session.rebuild_calls == 1


def test_strict_flip_exception_also_refunds_after_durable_restore(tmp_path):
    executor, _session = _flip_executor(
        tmp_path, RuntimeError("register transport exited")
    )

    result = executor.move_object("mug#0", "rotate_180")

    assert result["status"] == "error"
    assert result["output"]["failure_kind"] == "infrastructure_error"
    assert result["output"]["budget_consumed"] is False
    assert "mug#0" not in executor._moved
    assert "mug#0" not in executor._flipped


def test_object_state_table_does_not_recharge_recovered_flip_infrastructure():
    memory = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "target"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "investigate_objects",
                        "arguments": '{"objects":["mug#0"]}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "Investigated mug#0 (visit mug#0: 1/3)",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "move",
                        "arguments": ('{"object":"mug#0","aspect":"rotate_180"}'),
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "rotate_180 on mug#0 failed because the strict physics "
                "infrastructure errored (RPC failed). The pre-flip scene was "
                "restored; the one-shot and move budget were not consumed."
            ),
        },
    ]

    table = build_object_state_table(memory)

    assert "| mug#0 |" in table
    assert "| 1/3 | - |" in table


def test_strict_flip_recovery_failure_fails_closed_and_keeps_charge(tmp_path):
    executor, session = _flip_executor(tmp_path, _flip_result(infrastructure=True))
    executor._restore_failed_flip_transaction = lambda *args, **kwargs: (
        False,
        "restore transport failed",
    )

    result = executor.move_object("mug#0", "rotate_180")

    assert result["status"] == "error"
    assert result["output"]["scene_mutation"] == "unknown"
    assert result["output"]["recovery_succeeded"] is False
    assert result["output"]["budget_consumed"] is True
    assert executor._moved["mug#0"] == 1
    assert "mug#0" in executor._flipped
    assert "mug#0" not in executor._flip_phys_rejected
    assert executor._pose_session is None
    assert session.closed is True


def test_clean_flip_rejection_keeps_legacy_retry_and_move_accounting(tmp_path):
    executor, session = _flip_executor(tmp_path, _flip_result())

    result = executor.move_object("mug#0", "rotate_180")

    assert result["status"] == "success"
    assert executor._moved["mug#0"] == 1
    assert "mug#0" not in executor._flipped
    assert "mug#0" in executor._flip_phys_rejected
    assert session.save_calls == 1


@pytest.mark.parametrize(("strict", "budget_consumed"), [(False, True), (True, False)])
def test_flip_snapshot_infrastructure_refund_is_strict_profile_only(
    tmp_path, strict, budget_consumed
):
    executor, _session = _flip_executor(
        tmp_path,
        _flip_result(applied=True),
        strict=strict,
        save_error=RuntimeError("disk unavailable"),
    )

    result = executor.move_object("mug#0", "rotate_180")

    assert result["status"] == "error"
    assert (executor._moved.get("mug#0", 0) == 1) is budget_consumed
    assert ("mug#0" in executor._flipped) is budget_consumed
    if strict:
        assert result["output"]["failure_kind"] == "infrastructure_error"
        assert result["output"]["scene_mutation"] == "not_committed"
    else:
        assert "remain consumed" in result["output"]["text"][0]


class _SettlementPhysics:
    def __init__(self, error, recovery_errors=None):
        self.error = error
        self.recovery_errors = list(recovery_errors or [])
        self.recovery_calls = 0

    def settle_edited(self, session, intended_resting_modes=None):
        raise self.error

    def recover_after_error(self, session):
        self.recovery_calls += 1
        return self.recovery_errors


def _settlement_executor(error, *, recovery_errors=None):
    executor = Executor.__new__(Executor)
    executor.root_stage_name = "composition"
    executor.harness_profile = "gpt6_v1"
    executor.harness_profile_manifest = {
        "capabilities": {"strict_post_edit_physics": True}
    }
    executor._freeform_edit_prepared = True
    executor._pose_dirty = False
    executor._armed = set()
    executor._freeform = {}
    executor.edit_history = []
    executor.blender_save = None
    executor._pose_session = SimpleNamespace(save=lambda: None)
    executor._physics_obj = _SettlementPhysics(error, recovery_errors=recovery_errors)
    return executor


def test_direct_settlement_labels_clean_physics_rejection_without_recovery():
    executor = _settlement_executor(
        PhysicsSettlementRejected("would topple", reports=[{"accepted": False}])
    )

    result = executor._settle_after_edit()

    assert result["status"] == "error"
    assert result["failure_kind"] == "physics_rejection"
    assert executor._physics_obj.recovery_calls == 0


@pytest.mark.parametrize(
    ("recovery_errors", "recovered"), [([], True), (["Isaac unavailable"], False)]
)
def test_direct_settlement_labels_infrastructure_and_recovery_state(
    recovery_errors, recovered
):
    executor = _settlement_executor(
        RuntimeError("RPC transport failed"), recovery_errors=recovery_errors
    )

    result = executor._settle_after_edit()

    assert result["status"] == "error"
    assert result["failure_kind"] == "infrastructure_error"
    assert result["recovery_succeeded"] is recovered
    assert result["recovery_errors"] == recovery_errors
    assert executor._physics_obj.recovery_calls == 1


def test_typed_settle_intents_bind_exact_support_and_keep_mode_without_one():
    executor = Executor.__new__(Executor)
    executor._support_body_map = lambda: {
        "obj_mug_0": "table_0",
        "obj_other_0": "obj_tray_0",
    }

    intents = executor._typed_settle_intents(
        {"obj_mug_0": "preserve", "obj_book_0": "free"}
    )

    assert intents == {
        "obj_mug_0": {"mode": "preserve", "required_support": "table_0"},
        "obj_book_0": {"mode": "free"},
    }


def _direct_pose_executor(tmp_path, physics_result):
    executor = Executor.__new__(Executor)
    executor.root_stage_name = "composition"
    executor.harness_profile = "gpt6_v1"
    executor.harness_profile_manifest = {
        "capabilities": {
            "composition_direct_pose_edit": True,
            "strict_post_edit_physics": True,
        }
    }
    executor.blender_save = str(tmp_path / "live.blend")
    executor.edit_history = []
    executor._edit_meta = []
    executor._freeform = {"obj_mug_0": 2}
    executor._armed = set()
    executor._pose_dirty = False
    executor._last_code = "previous raw script"
    executor._begin_mutation = lambda kind, request: "composition-test"
    executor._pose_session_get = lambda: SimpleNamespace(ctx={"mug#0": object()})
    normalized = [
        {
            "object_id": "mug#0",
            "mesh_name": "obj_mug_0",
            "mode": "delta",
            "translation_m": [0.1, 0.0, 0.0],
            "expected_resting_mode": "preserve",
        }
    ]
    executor._normalize_typed_pose_edits = lambda edits, session: normalized
    executor._prepare_freeform_edit = lambda: {"status": "ready"}

    def execute(code, render=False):
        executor.edit_history.append(str(tmp_path / "after.blend"))
        executor._edit_meta.append({})
        return {"status": "success", "output": {}}

    executor.execute = execute

    def settle(*, intended_resting_modes=None):
        # Simulate an implementation which charged before a later failure; the
        # outer typed transaction must restore the pre-call counter on rollback.
        executor._freeform["obj_mug_0"] = 3
        return physics_result

    executor._settle_after_edit = settle
    executor._rollback_last_edit = lambda: True
    executor.audit_events = []
    executor._record_mutation_status = lambda txid, kind, status, details=None: (
        executor.audit_events.append((status, details))
    )
    executor.discard_calls = 0

    def discard():
        executor.discard_calls += 1
        return []

    executor._discard_composition_pose_runtime = discard
    return executor


def test_composition_typed_pose_passes_declared_support_intent(tmp_path):
    executor = _direct_pose_executor(
        tmp_path,
        {
            "status": "settled",
            "reports": [{"name": "obj_mug_0", "members": ["obj_mug_0"]}],
        },
    )
    executor._support_body_map = lambda: {"obj_mug_0": "table_0"}
    captured = []

    def settle(*, intended_resting_modes=None):
        captured.append(intended_resting_modes)
        return {
            "status": "settled",
            "reports": [{"name": "obj_mug_0", "members": ["obj_mug_0"]}],
        }

    executor._settle_after_edit = settle
    executor._settled_object_crop = lambda meshes: None

    result = executor.edit_composition_object_poses(
        [
            {
                "object": "mug#0",
                "mode": "delta",
                "translation_m": [0.1, 0.0, 0.0],
            }
        ],
        "visible target offset",
    )

    assert result["status"] == "success"
    assert captured == [
        {
            "obj_mug_0": {
                "mode": "preserve",
                "required_support": "table_0",
            }
        }
    ]


@pytest.mark.parametrize(
    ("physics_result", "scene_mutation", "audit_status", "discard_calls"),
    [
        (
            {
                "status": "error",
                "failure_kind": "physics_rejection",
                "reports": [],
                "reason": "would topple",
            },
            "not_committed",
            "rolled_back",
            0,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reports": [],
                "reason": "RPC failed",
                "recovery_succeeded": True,
                "recovery_errors": [],
            },
            "not_committed",
            "rolled_back",
            0,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reports": [],
                "reason": "RPC failed",
                "recovery_succeeded": False,
                "recovery_errors": ["restore failed"],
            },
            "unknown",
            "error",
            1,
        ),
    ],
)
def test_direct_pose_transaction_distinguishes_rejection_and_fails_closed(
    tmp_path, physics_result, scene_mutation, audit_status, discard_calls
):
    executor = _direct_pose_executor(tmp_path, physics_result)

    result = executor.edit_composition_object_poses(
        [
            {
                "object": "mug#0",
                "mode": "delta",
                "translation_m": [0.1, 0.0, 0.0],
            }
        ],
        "visible target offset",
    )

    assert result["status"] == "error"
    assert result["output"]["failure_kind"] == physics_result["failure_kind"]
    assert result["output"]["scene_mutation"] == scene_mutation
    assert executor.audit_events[-1][0] == audit_status
    assert executor.discard_calls == discard_calls
    assert executor._freeform == {"obj_mug_0": 2}


def test_direct_pose_outer_exception_restores_counter_after_successful_rollback(
    tmp_path,
):
    executor = _direct_pose_executor(tmp_path, {"status": "settled", "reports": []})
    audit_calls = 0

    def record_status(txid, kind, status, details=None):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 1:
            raise RuntimeError("terminal audit write failed")
        executor.audit_events.append((status, details))

    executor._record_mutation_status = record_status

    result = executor.edit_composition_object_poses(
        [
            {
                "object": "mug#0",
                "mode": "delta",
                "translation_m": [0.1, 0.0, 0.0],
            }
        ],
        "visible target offset",
    )

    assert result["status"] == "error"
    assert executor._freeform == {"obj_mug_0": 2}
    assert executor.audit_events[-1][0] == "rolled_back"


class _RawCompositionExecutor:
    def __init__(self, physics_result, *, strict=True, rollback=True):
        self.root_stage_name = "composition"
        self.strict = strict
        self.physics_result = physics_result
        self.rollback = rollback
        self._freeform = {"obj_mug_0": 2}
        self._pose_dirty = False
        self._armed = {"mug#0"}
        self.discard_calls = 0
        self.rollback_calls = 0

    def _has_capability(self, capability):
        return self.strict and capability == "strict_post_edit_physics"

    def _prepare_freeform_edit(self):
        return {"status": "ready"}

    def _pseudo_gt_views(self):
        return []

    def execute(self, code, render):
        return {"status": "success", "output": {"text": ["raw result"]}}

    def _settle_after_edit(self):
        self._freeform["obj_mug_0"] = 3
        return self.physics_result

    def _rollback_last_edit(self):
        self.rollback_calls += 1
        if isinstance(self.rollback, Exception):
            raise self.rollback
        return self.rollback

    def _discard_composition_pose_runtime(self):
        self.discard_calls += 1
        return []


@pytest.mark.parametrize(
    ("physics_result", "rollback", "scene_mutation", "discard_calls"),
    [
        (
            {
                "status": "error",
                "failure_kind": "physics_rejection",
                "reason": "would topple",
                "reports": [],
            },
            True,
            "not_committed",
            0,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reason": "RPC failed",
                "reports": [],
                "recovery_succeeded": True,
                "recovery_errors": [],
            },
            True,
            "not_committed",
            0,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reason": "RPC failed",
                "reports": [],
                "recovery_succeeded": False,
                "recovery_errors": ["Isaac restore failed"],
            },
            True,
            "unknown",
            1,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reason": "RPC failed",
                "reports": [],
                # Fail closed on internally inconsistent recovery telemetry.
                "recovery_succeeded": True,
                "recovery_errors": ["Blend restore remained stale"],
            },
            True,
            "unknown",
            1,
        ),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure_error",
                "reason": "RPC failed",
                "reports": [],
                "recovery_succeeded": True,
                "recovery_errors": [],
            },
            False,
            "unknown",
            1,
        ),
    ],
)
def test_raw_composition_strict_failure_metadata_and_fail_closed_cleanup(
    monkeypatch, physics_result, rollback, scene_mutation, discard_calls
):
    executor = _RawCompositionExecutor(physics_result, rollback=rollback)
    monkeypatch.setattr(blender_exec_module, "_executor", executor)
    monkeypatch.setattr(
        blender_exec_module,
        "_guard_pending_composition_followup",
        lambda tool: None,
    )

    result = blender_exec_module.execute_and_evaluate(code="pass")

    recovered = scene_mutation == "not_committed"
    assert result["status"] == "error"
    assert result["output"]["failure_kind"] == physics_result["failure_kind"]
    assert result["output"]["scene_mutation"] == scene_mutation
    assert result["output"]["retryable"] is recovered
    assert result["output"]["recovery_succeeded"] is recovered
    assert result["output"]["recovery_errors"] == physics_result.get(
        "recovery_errors", []
    )
    assert executor.discard_calls == discard_calls
    assert executor._freeform == ({"obj_mug_0": 2} if rollback else {"obj_mug_0": 3})


def test_raw_composition_baseline_failure_response_is_unchanged(monkeypatch):
    physics_result = {
        "status": "error",
        "failure_kind": "infrastructure_error",
        "reason": "RPC failed",
        "recovery_succeeded": False,
        "recovery_errors": ["restore failed"],
    }
    executor = _RawCompositionExecutor(physics_result, strict=False, rollback=True)
    monkeypatch.setattr(blender_exec_module, "_executor", executor)
    monkeypatch.setattr(
        blender_exec_module,
        "_guard_pending_composition_followup",
        lambda tool: None,
    )

    result = blender_exec_module.execute_and_evaluate(code="pass")

    assert result == {
        "status": "error",
        "output": {
            "text": [
                "Composition edit rolled back because physics settlement could "
                "not produce a reviewable result: error (RPC failed). Retry after "
                "the physics service recovers."
            ]
        },
    }
    assert executor.discard_calls == 0
    assert executor._freeform == {"obj_mug_0": 3}


def test_raw_composition_strict_rollback_exception_fails_closed(monkeypatch):
    physics_result = {
        "status": "error",
        "failure_kind": "infrastructure_error",
        "reason": "RPC failed",
        "recovery_succeeded": True,
        "recovery_errors": [],
    }
    executor = _RawCompositionExecutor(
        physics_result, rollback=RuntimeError("restore copy failed")
    )
    monkeypatch.setattr(blender_exec_module, "_executor", executor)
    monkeypatch.setattr(
        blender_exec_module,
        "_guard_pending_composition_followup",
        lambda tool: None,
    )

    result = blender_exec_module.execute_and_evaluate(code="pass")

    assert result["status"] == "error"
    assert result["output"]["scene_mutation"] == "unknown"
    assert result["output"]["retryable"] is False
    assert executor.discard_calls == 1
    assert executor._freeform == {"obj_mug_0": 3}
    assert "Blend rollback failed: restore copy failed" in result["output"]["text"][0]


def test_typed_pose_cap_redirects_to_move_after_two_committed_edits(tmp_path):
    settled = {
        "status": "settled",
        "reports": [{"name": "obj_mug_0", "members": ["obj_mug_0"]}],
    }
    executor = _direct_pose_executor(tmp_path, settled)
    executor._settled_object_crop = lambda meshes: None
    edit = [{"object": "mug#0", "mode": "delta", "translation_m": [0.05, 0.0, 0.0]}]
    assert executor.edit_composition_object_poses(edit, "first")["status"] == "success"
    assert executor.edit_composition_object_poses(edit, "second")["status"] == "success"
    third = executor.edit_composition_object_poses(edit, "third")
    assert third["status"] == "error"
    assert "typed pose cap (2 committed edits per object)" in third["output"]["text"][0]
    assert "use move(object, aspect)" in third["output"]["text"][0]
    assert executor._typed_pose == {"mug#0": 2}
    # Another object is unaffected by mug#0's cap.
    executor._normalize_typed_pose_edits = lambda edits, session: [
        {
            "object_id": "spoon#0",
            "mesh_name": "obj_spoon_0",
            "mode": "delta",
            "translation_m": [0.05, 0.0, 0.0],
            "expected_resting_mode": "preserve",
        }
    ]
    other = [{"object": "spoon#0", "mode": "delta", "translation_m": [0.05, 0.0, 0.0]}]
    assert executor.edit_composition_object_poses(other, "other")["status"] == "success"
