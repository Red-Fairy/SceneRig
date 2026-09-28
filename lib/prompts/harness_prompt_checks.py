"""Focused checks for profile-specific static-scene system prompts."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

# Developer/test checkouts intentionally omit these local launcher/credential modules.
# Stub only their import contracts; no prompt check needs credentials or subprocesses.
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

import lib.agents.prompt_builder as prompt_builder_module
from lib.agents.harness_profile import resolve_harness_profile
from lib.agents.prompt_builder import PromptBuilder
from lib.prompts import SYSTEM_PROMPTS, get_system_prompt
from lib.prompts.static_scene.generators.composition import (
    composition_generator_system,
)
from lib.prompts.static_scene.generators.initializer import (
    initializer_generator_system,
    static_scene_initializer_generator_system_scene_graph,
)
from lib.prompts.static_scene.scopes import TASK_PREAMBLE
from lib.prompts.static_scene.verifiers.initializer import (
    initializer_verifier_system,
    static_scene_initializer_verifier_system_scene_graph,
)
from lib.tools.blender.procedural_object import validate_authored_code
from lib.tools.geometry._yaw_test_artifact import write_unverified_yaw_observation
from lib.tools.geometry.relationship_constraints import write_initializer_constraints
from lib.utils.common import _openai_to_responses

_BASELINE_SHA256 = {
    "composition_generator": "7aecf8069663406118699bfa6b2304123fea806e6963f1fcf69ad11a98645091",
    "initializer_generator": "995f452e25f4099136d5f6c907ab28a44a66d2e7881fbec6f2b55669c14a4564",
    "initializer_verifier": "f70b33031402865acbbf2eb97ee5238680287506c4bd3788b668c09c4b1f25c9",
    "lighting_generator": "fcf734b31310e24c9f2d4b63230f32179884ccf5a7e9a0c886b720228c765ed1",
    "lighting_verifier": "3c0be5f5b8aa8b2fe2a10a9e20ba1cbe3432f01ec93ed8086c48b11f31f661d9",
    "texture_generator": "f868411e327b17e54d79117327b42fa2986adbbe4b8e4cd76ef804ff88f33c34",
    "texture_verifier": "0889a9480eb188386b4ca385fcb77fcc1bcdeb703514e933b426ca25ee3cade6",
}
_GPT6_SHA256 = {
    **_BASELINE_SHA256,
    "initializer_generator": "6e3ab28cdb0b154c8c498099e601d3974f62e92f0e07726372fb1c5b299de340",
    "initializer_verifier": "582edd75734f7a1a8559b18288cf5fb680e5f9222b22eafb43fd38b171b2f15d",
    "composition_generator": "3bb1270f9b7676cb1f28239e955c6ddcd64065fe626fcbea980b8e87cea95a74",
    "texture_generator": "b836e8378ec455487b918bf94f9451f8e6265d2d4b15e232fd67fe54f6778c02",
}


def _manifest(**overrides: bool) -> dict:
    manifest = resolve_harness_profile("gpt6_v1")
    manifest["capabilities"].update(overrides)
    return manifest


def test_baseline_prompt_constants_and_selection_are_byte_identical() -> None:
    assert set(SYSTEM_PROMPTS) == set(_BASELINE_SHA256)
    for key, baseline in SYSTEM_PROMPTS.items():
        assert hashlib.sha256(baseline.encode()).hexdigest() == _BASELINE_SHA256[key]
        assert get_system_prompt(key) == baseline
        assert get_system_prompt(key, harness_profile="baseline") == baseline

    assert initializer_generator_system() == (
        static_scene_initializer_generator_system_scene_graph
    )
    assert initializer_verifier_system() == (
        static_scene_initializer_verifier_system_scene_graph
    )


def test_gpt6_prompts_match_reviewed_contract_and_missing_manifest_policy() -> None:
    for key, expected in _GPT6_SHA256.items():
        prompt = get_system_prompt(key, "gpt6_v1", _manifest())
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected
        # 2026-09-15 owner decision: a missing manifest FAILS CLOSED — it renders exactly
        # like a manifest with every capability off (never re-enables a withdrawn tool).
        missing_manifest = get_system_prompt(key, "gpt6_v1")
        all_off = _manifest(**{c: False for c in _manifest()["capabilities"]})
        assert missing_manifest == get_system_prompt(key, "gpt6_v1", all_off)


def test_gpt6_initializer_generator_addendum_is_evidence_bound_and_code_authored() -> (
    None
):
    prompt = get_system_prompt(
        "initializer_generator",
        harness_profile="gpt6_v1",
        harness_profile_manifest=_manifest(),
    )

    assert prompt.startswith(TASK_PREAMBLE)
    assert prompt != static_scene_initializer_generator_system_scene_graph
    for text in (
        "MAJOR clearly visible object that is missing",
        "cannot stand as reconstructed",
        "missing a critical structural part",
        "reconstructed PROPORTIONS are",
        "must be WATERTIGHT",
        "target evidence",
        "execute_and_evaluate",
        "added_objects",
        "removed_objects",
        "COMPLETE authored Blender Python script",
        "ADDED_OBJECT_NAMES",
        "REMOVED_OBJECT_NAMES",
        "NOT environment variables",
        "one parentless EMPTY",
        "whole-object pose",
        "Do not save/open/export files",
        "isolated backend transaction",
        "verifies a clean GLB re-import",
        "logs the request and",
        "downstream USD inputs",
        "physics-estimation material/mass/friction record",
        "physics simulation after the complete code batch over exactly those objects' hierarchies",
        "separate dynamic bodies",
        "Building or editing a root surface alone",
        "does NOT simulate",
        "never rejected for overlapping an object",
        "reported in the result and the no-penetration rule lists it",
        "unchanged calls do not simulate",
        "pose_applied=baked",
        "pose_applied=static",
        "A converged topple is kept and reported as repair-needed",
        "malformed or incomplete results, and nonconvergence roll",
        "disconnected pieces are rejected",
        "support chain reaches the\nauthenticated main_support_id",
        "same-call additions",
        "when the main support is the\nroom floor",
        "Do not\nreconstruct background furniture",
        "Allowed imports are exactly bmesh, bpy",
        'oid = "item#1"',
        "part_name_map",
        "grase_part_label",
        "get-or-create",
    ):
        assert text in prompt


def test_gpt6_initializer_verifier_reviews_committed_results_not_artifacts() -> None:
    prompt = get_system_prompt(
        "initializer_verifier",
        harness_profile_manifest=_manifest(),
    )

    assert prompt.startswith(TASK_PREAMBLE)
    assert prompt != static_scene_initializer_verifier_system_scene_graph
    for text in (
        "Backend-COMMITTED runtime objects, mesh revisions, and full-pose edits",
        "members/revisions of the current scene",
        "Judge the RESULTING fidelity",
        "triggers one physics simulation over exactly the edited objects' hierarchies",
        "every other object and every registered root as static\ncolliders",
        "baked POST-SIMULATION",
        "converged topple is kept with",
        "nonconvergence roll back",
        "complete authored",
        "ADDED_OBJECT_NAMES and REMOVED_OBJECT_NAMES",
        "round-trip checks mesh/material export",
        "procedural code",
        "request corrective Blender code",
        "never ask the generator to falsify or hand-edit those artifacts",
        "Any current object can be removed",
        "support chain reaches the authenticated main support",
        "additions rooted at a different background\nsurface are out of scope",
    ):
        assert text in prompt


@pytest.mark.parametrize(
    "capability",
    ["initializer_code_transactions", "runtime_object_inventory", "mutation_journal"],
)
def test_disabled_capability_fails_closed_to_baseline(capability: str) -> None:
    disabled = _manifest(**{capability: False})
    assert (
        initializer_generator_system("gpt6_v1", disabled)
        == static_scene_initializer_generator_system_scene_graph
    )
    assert (
        initializer_verifier_system("gpt6_v1", disabled)
        == static_scene_initializer_verifier_system_scene_graph
    )


def test_composition_registry_uses_existing_profile_selector() -> None:
    manifest = _manifest()
    assert get_system_prompt(
        "composition_generator", "gpt6_v1", manifest
    ) == composition_generator_system("gpt6_v1", manifest)


def test_composition_mesh_repair_and_maskless_contract_are_capability_gated() -> None:
    prompt = composition_generator_system("gpt6_v1", _manifest())
    for text in (
        "After a successful investigate_objects call",
        "edit_object_mesh(object, code, reason)",
        "TARGET_OBJECT_NAME and",
        "one parentless EMPTY",
        "connected Boolean union",
        "same membership and order",
        "undo_last_step",
        "Only mask-bearing objects require investigation",
        "keeps the original object's photo mask",
        "Allowed\nimports are exactly bmesh, bpy",
        "part_name_map",
        "grase_part_label",
        "[Layout Editing]",
        "execute_and_evaluate leaves it armed",
        "physics-settled before the render",
        "re-cooks its collider",
        "uniform rescale in code",
        "move() and execute_and_evaluate are peers",
        "every investigate result reminds you of both routes",
        "both routes are open to you at any time",
    ):
        assert text in prompt
    for text in (
        "CLEARED by any execute_and_evaluate/undo",
        "Prefer move() for routine single-object pose fixes",
        "direct pose tool when it is offered",
    ):
        assert text not in prompt
    no_mesh = composition_generator_system(
        "gpt6_v1", _manifest(composition_mesh_edit=False)
    )
    assert "[Investigated Mesh Repair]" not in no_mesh
    assert "[Layout Editing]" in no_mesh
    no_pose = composition_generator_system(
        "gpt6_v1", _manifest(composition_direct_pose_edit=False)
    )
    assert "[Investigated Mesh Repair]" in no_pose
    assert "[Layout Editing]" in no_pose
    assert "[Direct Pose Editing]" not in no_pose
    assert "edit_object_poses" not in no_pose
    with_pose = composition_generator_system(
        "gpt6_v1", _manifest(composition_direct_pose_edit=True)
    )
    for text in (
        "[Direct Pose Editing]",
        "[Layout Editing]",
        "execute_and_evaluate leaves it armed",
        "physics-settled before the render",
        "re-cooks its collider",
    ):
        assert text in with_pose
    assert "CLEARED by any execute_and_evaluate/undo" not in with_pose
    no_strict = composition_generator_system(
        "gpt6_v1", _manifest(strict_post_edit_physics=False)
    )
    assert "[Layout Editing]" not in no_strict
    assert "CLEARED by any execute_and_evaluate/undo" in no_strict
    baseline = get_system_prompt("composition_generator")
    assert "[Authored Objects Without Scoring Masks]" not in baseline


def test_effective_prompts_have_no_profile_labels_or_superseded_contracts() -> None:
    prompts = (
        initializer_generator_system("gpt6_v1", _manifest()),
        initializer_verifier_system("gpt6_v1", _manifest()),
        composition_generator_system("gpt6_v1", _manifest()),
    )
    for prompt in prompts:
        lowered = prompt.casefold()
        for leaked_label in (
            "gpt-6",
            "harness profile",
            "profile override",
            "developer experiment",
            "supersedes the baseline",
        ):
            assert leaked_label not in lowered

    initializer = prompts[0]
    assert "NO physics simulation runs" not in initializer
    assert "`nudge_object` is the only initializer object-edit path" not in initializer
    # 2026-09-15 owner: gpt6_v1 has no nudge_object at all (menu, hint, prompts)
    assert "There is no nudge_object in this harness" in initializer
    assert "nudge_object for its" not in initializer and "bounded nudge" not in initializer
    # 2026-09-16: one direct-support contract for BOTH harnesses
    assert "COVER ALL OBJECTS WITH THE MAIN SUPPORT" not in initializer
    assert "COVER EACH OBJECT WITH ITS EXACT DIRECT SUPPORT" in initializer
    assert "COVER EACH OBJECT WITH ITS EXACT DIRECT SUPPORT" in get_system_prompt(
        "initializer_generator"
    )

    verifier = prompts[1]
    assert "NO physics simulation runs" not in verifier
    assert "A gate failure here is a MAIN-SUPPORT defect by default" not in verifier
    assert "its exact direct support" in verifier

    composition = prompts[2]
    assert "Every imported object must be inspected." not in composition
    assert "Work through the objects with two tools" not in composition
    assert "why the code only changes layout" not in composition
    assert "Every mask-bearing imported object must be inspected" in composition
    assert "Prefer move() for routine single-object pose fixes" not in composition
    assert "CLEARED by any execute_and_evaluate/undo" not in composition
    # 2026-09-15 owner-approved: no sentence frames code as the route AFTER a failed
    # move; the baseline keeps its move-first wording
    for after_failure in (
        "If that move is rejected or dead",
        "also qualifies for direct execution when",
    ):
        assert after_failure not in composition
        assert after_failure in SYSTEM_PROMPTS["composition_generator"]
    assert "fix it by either route" in composition
    assert "investigate that object once more before you end" in composition  # freshness
    assert "Neither a rejected move(object, 'rotation') nor a dead search" in composition
    typed_pose = composition_generator_system(
        "gpt6_v1", _manifest(composition_direct_pose_edit=True)
    )
    assert "call move(object, 'rotation') even when no YAW HINT" not in typed_pose
    assert "A visible fine-yaw mismatch qualifies for edit_object_poses" in typed_pose


def test_enhanced_texture_mapping_reminder_preserves_baseline() -> None:
    baseline = get_system_prompt("texture_generator")
    enhanced = get_system_prompt(
        "texture_generator", harness_profile_manifest=_manifest()
    )
    assert enhanced.startswith(baseline)
    assert "[Image Texture Coordinates]" in enhanced
    assert "coordinates outside [0,1] clamp to edge pixels" in enhanced
    assert "[Image Texture Coordinates]" not in baseline


def test_direct_pose_example_supplies_nullable_alternatives() -> None:
    prompt = composition_generator_system(
        "gpt6_v1", _manifest(composition_direct_pose_edit=True)
    )
    start = prompt.index('    {"edits":')
    end = prompt.index("\nFor a quaternion edit", start)
    request = json.loads(prompt[start:end])
    edit = request["edits"][0]
    assert edit["rotation_euler_deg"] == [0, 0, 25]
    assert edit["translation_m"] is None
    assert edit["rotation_quaternion_wxyz"] is None
    assert edit["expected_resting_mode"] == "preserve"
    assert "identity quaternion is a\nreal rotation" in prompt


def test_authored_object_examples_are_complete_python_with_allowed_imports() -> None:
    prompts_and_markers = (
        (
            initializer_generator_system("gpt6_v1", _manifest()),
            "A compact safe pattern is:",
        ),
        (
            composition_generator_system("gpt6_v1", _manifest()),
            "A minimal complete branching-solid example is below",
        ),
    )
    for prompt, marker in prompts_and_markers:
        start = prompt.index("    import bpy", prompt.index(marker))
        end = prompt.index("\nUse get-or-create for materials", start)
        code = textwrap.dedent(prompt[start:end])
        tree = ast.parse(code)
        validate_authored_code(code)
        assert {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } == {"bpy"}
        assigned = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        assert {"root", "part", "mat"} <= assigned
        assert 'end_fill_type="NGON"' in code
        assert "from mathutils import Vector" in code
        assert "edge-only Skin source" in prompt
        assert "Retain the original asset" in prompt


def _pose_scene(tmp_path: Path, records: dict) -> PromptBuilder:
    """Write the real minimal scene contracts consumed by PromptBuilder."""
    objects = [
        {
            "category": "bread slice",
            "instance": index,
            "mesh_name": f"obj_bread_slice_{index}",
            "center": [0, 0, 0.1],
            "size": [0.1, 0.1, 0.2],
        }
        for index in range(4)
    ]
    graph = {
        "main_support_id": "table#0",
        "relationship_adjudication": {"schema_version": 3},
        "relationships": [],
        "nodes": [
            {
                "id": "table#0",
                "category": "table",
                "kind": "root_surface",
                "main_support": True,
            },
            *[
                {
                    "id": f"bread slice#{index}",
                    "category": "bread slice",
                    "kind": "object",
                    "parent": "table#0",
                }
                for index in range(4)
            ],
        ],
    }
    (tmp_path / "scene_graph.json").write_text(json.dumps(graph))
    (tmp_path / "placement.json").write_text(json.dumps({"objects": objects}))
    yaw = write_unverified_yaw_observation(tmp_path, graph)
    write_initializer_constraints(tmp_path, graph, yaw_observation=yaw)
    (tmp_path / "physics").mkdir(exist_ok=True)
    (tmp_path / "physics/preprocess_pose_changes.json").write_text(
        json.dumps({"objects": records})
    )
    return PromptBuilder(
        None,
        {
            "generator_prompt_agent_type": "initializer_generator",
            "root_stage_name": "initializer",
            "harness_profile": "gpt6_v1",
            "harness_profile_manifest": _manifest(),
            "moge_dir": str(tmp_path),
        },
    )


def _outbound_text(builder: PromptBuilder, extra_messages: list | None = None) -> str:
    messages = builder.build_prompt("generator", "system") + (extra_messages or [])
    request = _openai_to_responses({"model": "test-model", "messages": messages})
    return "\n".join(
        part["text"]
        for item in request["input"]
        for part in item.get("content", [])
        if part.get("type") == "input_text"
    )


def test_preprocess_pose_history_reaches_outbound_request_once(tmp_path: Path) -> None:
    records = {
        f"obj_bread_slice_{index}": {
            "fell": True,
            "flip_accepted": None,
            "rollable": False,
            "tilt_deg": tilt,
            "rest_dz": 0.001,
            "converged": True,
            "retained_drop_attempt": 2,
            "drop_attempts": [
                {"id": 2, "disp_xy_mm": 0.3},
                {"id": 3, "disp_xy_mm": 999, "converged": False},
            ],
        }
        for index, tilt in [(0, 84.285), (3, 118.81914727216406)]
    }
    text = _outbound_text(_pose_scene(tmp_path, records))
    assert text.count("[Preprocessing pose history]") == 1
    rows = json.loads(text.split("\n")[-1])
    assert [row["object"] for row in rows] == ["bread slice#0", "bread slice#3"]
    assert [row["retained_tilt_deg"] for row in rows] == [84.285, 118.81914727216406]
    assert all(row["retained_drop_disp_xy_mm"] == 0.3 for row in rows)
    assert all(row["rest_dz_m"] == 0.001 for row in rows)
    assert "not cumulative motion" in text


def test_pose_history_preserves_flip_rollable_and_nonconvergence_semantics(
    tmp_path: Path,
) -> None:
    records = {
        "obj_bread_slice_0": {
            "fell": True,
            "flip_accepted": True,
            "rollable": False,
            "converged": True,
            "tilt_deg": 170,
        },
        "obj_bread_slice_1": {
            "fell": False,
            "rollable": True,
            "converged": True,
            "tilt_deg": 170,
        },
        "obj_bread_slice_2": {
            "fell": False,
            "rollable": False,
            "converged": False,
            "settle_failed": True,
            "tilt_deg": 0,
        },
        "obj_bread_slice_3": {
            "fell": True,
            "rollable": True,
            "converged": True,
            "tilt_deg": 5,
        },
    }
    text = _outbound_text(_pose_scene(tmp_path, records))
    rows = json.loads(text.split("\n")[-1])
    assert [row["object"] for row in rows] == [
        "bread slice#0",
        "bread slice#2",
        "bread slice#3",
    ]
    assert rows[0]["flip_accepted"] is True and rows[0]["converged"] is True
    assert rows[1]["settle_failed"] is True and rows[1]["converged"] is False
    assert rows[2]["rollable"] is True
    assert "accepted stable-face flip" in text


def test_pose_history_stays_historical_after_repair_and_omits_removed_objects(
    tmp_path: Path,
) -> None:
    builder = _pose_scene(
        tmp_path,
        {
            "obj_bread_slice_0": {"fell": True, "tilt_deg": 84.285},
            "obj_bread_slice_3": {"fell": True, "tilt_deg": 118.819},
        },
    )
    path = tmp_path / "placement.json"
    placement = json.loads(path.read_text())
    placement["objects"] = [obj for obj in placement["objects"] if obj["instance"] != 3]
    placement["objects"][0]["mesh_provider"] = "procedural"
    path.write_text(json.dumps(placement))
    text = _outbound_text(
        builder,
        [
            {
                "role": "user",
                "content": "Latest committed repair: bread slice#0 now upright.",
            }
        ],
    )
    assert text.count("[Preprocessing pose history]") == 1
    history = text.split("[Preprocessing pose history]", 1)[1]
    assert '"object": "bread slice#0"' in history
    assert '"object": "bread slice#3"' not in history
    assert "not a current-state verdict" in history
    assert "evidence take precedence" in history
    assert history.endswith("Latest committed repair: bread slice#0 now upright.")


@pytest.mark.parametrize(
    "profile,stage", [("baseline", "initializer"), ("gpt6_v1", "composition")]
)
def test_pose_history_does_not_change_other_profile_or_stage_prompts(
    tmp_path: Path,
    profile: str,
    stage: str,
) -> None:
    builder = _pose_scene(tmp_path, {"obj_bread_slice_0": {"fell": True}})
    builder.config.update(harness_profile=profile, root_stage_name=stage)
    # Deliberately corrupt only the history: excluded routes must never read it.
    (tmp_path / "physics/preprocess_pose_changes.json").write_text("invalid")
    assert "[Preprocessing pose history]" not in _outbound_text(builder)


def test_optional_missing_pose_history_preserves_prompt(tmp_path: Path) -> None:
    builder = _pose_scene(tmp_path, {})
    before = _outbound_text(builder)
    (tmp_path / "physics/preprocess_pose_changes.json").unlink()
    assert _outbound_text(builder) == before


def test_prompt_builder_forwards_profile_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    observed = {}

    def fake_get_system_prompt(
        agent_type: str,
        harness_profile: str | None = None,
        harness_profile_manifest: dict | None = None,
    ) -> str:
        observed.update(
            {
                "agent_type": agent_type,
                "harness_profile": harness_profile,
                "harness_profile_manifest": harness_profile_manifest,
            }
        )
        return "selected prompt"

    monkeypatch.setattr(
        prompt_builder_module, "get_system_prompt", fake_get_system_prompt
    )
    builder = PromptBuilder(
        None,
        {
            "generator_prompt_agent_type": "initializer_generator",
            "harness_profile": "gpt6_v1",
            "harness_profile_manifest": manifest,
        },
    )
    monkeypatch.setattr(builder, "_build_system_prompt", lambda prompts, _: prompts)

    assert builder.build_prompt("generator", "system") == {"system": "selected prompt"}
    assert observed == {
        "agent_type": "initializer_generator",
        "harness_profile": "gpt6_v1",
        "harness_profile_manifest": manifest,
    }


def test_unknown_prompt_key_still_fails() -> None:
    with pytest.raises(ValueError, match="No system prompt registered"):
        get_system_prompt("not_a_stage", harness_profile="gpt6_v1")
