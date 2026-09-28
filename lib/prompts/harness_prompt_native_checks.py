"""Blender-native checks for the runnable authored-object prompt examples.

Run explicitly: these tests launch bundled Blender in factory-empty, CPU-only mode. They
execute the exact code extracted from the rendered prompts rather than a copied fixture.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lib.agents.harness_profile import resolve_harness_profile
from lib.prompts.static_scene.generators.composition import (
    composition_generator_system,
)
from lib.prompts.static_scene.generators.initializer import (
    initializer_generator_system,
)
from lib.tools.blender.procedural_object import build_procedural_capture_script

# Deselected by the project addopts ("-m 'not native'"); run with `pytest -m native <file>`.
pytestmark = pytest.mark.native

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLENDER = PROJECT_ROOT / "lib/utils/third_party/blender-4.5/blender"


def _manifest() -> dict:
    return resolve_harness_profile("gpt6_v1")


def _example(prompt: str, marker: str) -> str:
    start = prompt.index("    import bpy", prompt.index(marker))
    end = prompt.index("\nUse get-or-create for materials", start)
    lines = prompt[start:end].splitlines()
    assert lines and all(not line or line.startswith("    ") for line in lines)
    return "\n".join(line[4:] if line else "" for line in lines) + "\n"


def _run_blender(
    tmp_path: Path,
    name: str,
    script: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    assert BLENDER.is_file(), f"bundled Blender is missing: {BLENDER}"
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(script)
    result = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--threads",
            "2",
            "--python-exit-code",
            "1",
            "--python",
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    (tmp_path / f"{name}.stdout.log").write_text(result.stdout)
    (tmp_path / f"{name}.stderr.log").write_text(result.stderr)
    assert not check or result.returncode == 0, (
        f"Blender failed ({result.returncode})\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result


def _capture_saved(
    tmp_path: Path,
    tag: str,
    script: str,
    name: str,
    object_id: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Use the live tool's authored-save then separate-process capture boundary."""
    authored = tmp_path / "authored.blend"
    _run_blender(
        tmp_path,
        tag + "_author",
        script + f"\nbpy.ops.wm.save_as_mainfile(filepath={str(authored)!r})\n",
    )
    capture = build_procedural_capture_script(
        target_object_name=name,
        target_object_id=object_id,
        mesh_glb_path=tmp_path / "asset.glb",
        capture_json_path=tmp_path / "capture.json",
        blend_output_path=tmp_path / "candidate.blend",
    )
    return _run_blender(
        tmp_path,
        tag + "_capture",
        f"import bpy\nbpy.ops.wm.open_mainfile(filepath={str(authored)!r})\n" + capture,
        check=check,
    )


def _assert_branch_capture(tmp_path: Path) -> None:
    capture = json.loads((tmp_path / "capture.json").read_text())
    assert capture["part_count"] == 3
    assert capture["connected_union"]["component_count"] == 1
    assert capture["connected_union"]["asset_mesh_count"] == 1
    assert capture["roundtrip"]["mesh_count"] == 1
    assert {row["label"] for row in capture["part_name_map"]} == {
        "stem",
        "left_branch",
        "right_branch",
    }
    assert (tmp_path / "candidate.blend").is_file()


def test_initializer_rendered_example_executes_in_blender(tmp_path: Path) -> None:
    snippet = _example(
        initializer_generator_system("gpt6_v1", _manifest()),
        "A compact safe pattern is:",
    )
    script = f"""import bpy
ADDED_OBJECT_NAMES = {{"item#1": "obj_item_1"}}
{snippet}
root = bpy.data.objects.get("obj_item_1")
assert root is not None and root.type == "EMPTY" and root.parent is None
assert len(root.children) == 3
for part in root.children:
    assert part.type == "MESH" and len(part.data.polygons) > 0
    assert [material.name for material in part.data.materials] == ["obj_item_1_material"]
    assert tuple(round(value, 6) for value in part.scale) == (1.0, 1.0, 1.0)
print("PROMPT_INITIALIZER_EXAMPLE_OK")
"""
    result = _capture_saved(
        tmp_path, "initializer_prompt_example", script, "obj_item_1", "item#1"
    )
    assert "GRASE_PROCEDURAL_CAPTURE_OK" in result.stdout
    _assert_branch_capture(tmp_path)


def test_composition_rendered_example_replaces_nested_tree_in_blender(
    tmp_path: Path,
) -> None:
    snippet = _example(
        composition_generator_system("gpt6_v1", _manifest()),
        "A minimal complete branching-solid example",
    )
    script = f"""import bpy
TARGET_OBJECT_ID = "item#0"
TARGET_OBJECT_NAME = "obj_item_0"
old = bpy.data.objects.new(TARGET_OBJECT_NAME, None)
bpy.context.scene.collection.objects.link(old)
old["prompt_test_sentinel"] = "old object"
child = bpy.data.objects.new("old_child", bpy.data.meshes.new("old_child_mesh"))
bpy.context.scene.collection.objects.link(child)
child.parent = old
grandchild = bpy.data.objects.new("old_grandchild", bpy.data.meshes.new("old_grandchild_mesh"))
bpy.context.scene.collection.objects.link(grandchild)
grandchild.parent = child
{snippet}
root = bpy.data.objects.get(TARGET_OBJECT_NAME)
assert root is not None and "prompt_test_sentinel" not in root
assert root.type == "EMPTY" and root.parent is None
assert bpy.data.objects.get("old_child") is None
assert bpy.data.objects.get("old_grandchild") is None
assert len(root.children) == 3
for part in root.children:
    assert part.type == "MESH" and len(part.data.polygons) > 0
    assert [material.name for material in part.data.materials] == ["obj_item_0_material"]
    assert tuple(round(value, 6) for value in part.scale) == (1.0, 1.0, 1.0)
print("PROMPT_COMPOSITION_EXAMPLE_OK")
"""
    result = _capture_saved(
        tmp_path, "composition_prompt_example", script, "obj_item_0", "item#0"
    )
    assert "GRASE_PROCEDURAL_CAPTURE_OK" in result.stdout
    _assert_branch_capture(tmp_path)


@pytest.mark.parametrize(
    "defect, expected",
    [
        ("gap", "disconnected"),
        ("duplicate_face", "not a closed manifold"),
        ("edge_only", "vertices and polygons"),
        ("open_shell", "closed"),
        ("open_shell_unlabeled", "closed"),
    ],
)
def test_authored_branch_failures_still_reject_capture(
    tmp_path: Path,
    defect: str,
    expected: str,
) -> None:
    snippet = _example(
        initializer_generator_system("gpt6_v1", _manifest()),
        "A compact safe pattern is:",
    )
    corrupt = {
        "gap": "part.location.x += 0.5\n",
        "duplicate_face": "faces.append(faces[0])\n",
        "edge_only": "faces = []\nedges = [(0, 1)]\n",
        "open_shell": "faces.pop()\n",
        "open_shell_unlabeled": (
            "faces.pop()\n"
            "del part['grase_part_label']\n"
            "part.name = '_grase_item_1_part_003'\n"
        ),
    }
    script = f"""import bpy
ADDED_OBJECT_NAMES = {{"item#1": "obj_item_1"}}
{snippet}
part = next(child for child in root.children if child.get('grase_part_label') == 'left_branch')
vertices = [tuple(vertex.co) for vertex in part.data.vertices]
faces = [list(face.vertices) for face in part.data.polygons]
edges = []
{corrupt[defect]}
mesh = bpy.data.meshes.new('deliberately_invalid_part')
mesh.from_pydata(vertices, edges, faces)
part.data = mesh
part.data.materials.append(mat)
"""
    result = _capture_saved(
        tmp_path, defect, script, "obj_item_1", "item#1", check=False
    )
    assert result.returncode != 0
    errors = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("RuntimeError:")
    ]
    assert expected in "\n".join(errors)
    if defect != "gap":
        label = (
            "_grase_item_1_part_003" if defect.endswith("unlabeled") else "left_branch"
        )
        assert f"authored part {'item#1/' + label!r}" in "\n".join(errors)
    else:
        assert "authored part" not in "\n".join(errors)
    assert not (tmp_path / "capture.json").exists()
    assert not (tmp_path / "candidate.blend").exists()


def test_wall_example_preserves_local_dimensions_through_oblique_capture(
    tmp_path: Path,
) -> None:
    prompt = initializer_generator_system("gpt6_v1", _manifest())
    start = prompt.index("      from mathutils import Matrix, Vector")
    end = prompt.index("\n  The diagonal above", start)
    snippet = "\n".join(line[6:] for line in prompt[start:end].splitlines())
    script = f"""import bpy
bpy.ops.mesh.primitive_cube_add(size=1)
W = bpy.context.object
point, normal = (0.48, -0.36, 0.75), (0.6, 0.8, 0)
corner_x, corner_y = 0, 0
width, thickness, height, z_center = 2.0, 0.04, 1.5, 0.75
{snippet}
bpy.context.view_layer.update()
corners = [W.matrix_world @ Vector(corner) for corner in W.bound_box]
spans = [max(p.dot(axis) for p in corners) - min(p.dot(axis) for p in corners) for axis in (A, N, U)]
assert max(abs(a-b) for a,b in zip(spans, (width, thickness, height))) < 1e-6, spans
assert W.matrix_world.to_3x3().determinant() > 0
assert max(p.x for p in corners) - min(p.x for p in corners) > 1.5
assert max(p.y for p in corners) - min(p.y for p in corners) > 1.1
root = bpy.data.objects.new('obj_wall_example_0', None)
bpy.context.scene.collection.objects.link(root)
W.parent = root
W['grase_part_label'] = 'wall_slab'
"""
    result = _capture_saved(
        tmp_path, "wall_local_axes", script, "obj_wall_example_0", "wall example#0"
    )
    assert "GRASE_PROCEDURAL_CAPTURE_OK" in result.stdout
    capture = json.loads((tmp_path / "capture.json").read_text())
    assert capture["connected_union"]["component_count"] == 1
    assert capture["roundtrip"]["mesh_count"] == 1


def test_prompt_conversion_materializes_edge_generated_faces_before_capture(
    tmp_path: Path,
) -> None:
    prompt = initializer_generator_system("gpt6_v1", _manifest())
    conversion = 'bpy.ops.object.convert(target="MESH")'
    assert conversion in prompt
    script = f"""import bpy
bpy.ops.object.select_all(action='DESELECT')
mesh = bpy.data.meshes.new('edge_source')
mesh.from_pydata([(0,0,0), (0,0,0.2)], [(0,1)], [])
part = bpy.data.objects.new('generated_tube', mesh)
bpy.context.scene.collection.objects.link(part)
part.select_set(True)
bpy.context.view_layer.objects.active = part
part.modifiers.new(name='Skin', type='SKIN')
for vertex in mesh.skin_vertices[0].data:
    vertex.radius = (0.02, 0.02)
assert len(part.data.polygons) == 0
{conversion}
part = bpy.context.object
assert len(part.data.polygons) > 0 and len(part.modifiers) == 0
root = bpy.data.objects.new('obj_tube_0', None)
bpy.context.scene.collection.objects.link(root)
part.parent = root
part['grase_part_label'] = 'tube'
"""
    result = _capture_saved(tmp_path, "converted_tube", script, "obj_tube_0", "tube#0")
    assert "GRASE_PROCEDURAL_CAPTURE_OK" in result.stdout
    capture = json.loads((tmp_path / "capture.json").read_text())
    assert capture["connected_union"]["component_count"] == 1
    assert capture["roundtrip"]["mesh_count"] == 1


def test_parenting_example_preserves_world_pose_under_transformed_root(
    tmp_path: Path,
) -> None:
    """Exercise the exact parenting example, including stale matrix evaluation."""
    prompt = initializer_generator_system("gpt6_v1", _manifest())
    start = prompt.index(
        "    bpy.context.view_layer.update()", prompt.index("When a part")
    )
    end = prompt.index("\nKeep each logical object root parentless", start)
    snippet = "\n".join(line[4:] for line in prompt[start:end].splitlines())
    script = f"""import bpy
from mathutils import Euler, Matrix
root = bpy.data.objects.new('obj_parent_0', None)
bpy.context.scene.collection.objects.link(root)
root.location = (1.0, -2.0, 0.5)
root.rotation_euler = (0.2, -0.3, 0.8)
root.scale = (1.5, 0.75, 2.0)
bpy.ops.mesh.primitive_cube_add()
part = bpy.context.object
part.location = (-0.4, 0.3, 0.15)
part.rotation_euler = (0.1, 0.2, -0.6)
part.scale = (0.2, 0.4, 0.3)
expected = Matrix.LocRotScale(part.location.copy(), part.rotation_euler.to_quaternion(), part.scale.copy())
{snippet}
assert part.parent == root and root.parent is None
assert max(abs(part.matrix_world[r][c] - expected[r][c]) for r in range(4) for c in range(4)) < 1e-6
# Repeating an attachment must preserve the evaluated world pose as well.
{snippet}
assert max(abs(part.matrix_world[r][c] - expected[r][c]) for r in range(4) for c in range(4)) < 1e-6
print('PROMPT_PARENTING_EXAMPLE_OK')
"""
    result = _run_blender(tmp_path, "parenting_prompt_example", script)
    assert "PROMPT_PARENTING_EXAMPLE_OK" in result.stdout
