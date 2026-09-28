"""Small Blender-only material helpers; no geometry, camera, or lighting edits.

Import these from generated Blender code. NumPy and Blender are the only runtime
dependencies; root material ownership is checked before any mutation.
"""

from __future__ import annotations

import math

import bpy
import numpy as np


def root_material(name: str):
    """Return an isolated current material for a root mesh, never an imported object."""
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH" or name.startswith("obj_"):
        raise ValueError(f"Expected a current non-obj_ root mesh, got {name!r}")
    parent = obj.parent
    imported_parent = False
    while parent is not None:
        imported_parent |= parent.name.startswith("obj_")
        parent = parent.parent
    if obj.data.users > 1 or imported_parent:
        raise ValueError(
            f"Root {name!r} has shared mesh data or an imported-object parent"
        )
    mat = obj.data.materials[0] if obj.data.materials else None
    if mat is None:
        mat = bpy.data.materials.new(f"Root {name} material")
    elif mat.users > 1 or mat.name.startswith("obj_"):
        mat = mat.copy()
        mat.name = f"Root {name} material"
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    mat.use_nodes = True
    return mat


def set_socket(node, name: str, value) -> None:
    """Set a real input with an informative name/type error, before shader execution."""
    socket = node.inputs.get(name)
    if socket is None:
        raise ValueError(
            f"{node.name!r} has no input {name!r}; available: {list(node.inputs.keys())}"
        )
    current = socket.default_value
    if isinstance(current, (float, int)):
        if isinstance(value, str) or not math.isfinite(float(value)):
            raise ValueError(
                f"{node.name}.{name} expects a finite number, got {value!r}"
            )
        value = float(value)
    else:
        value = tuple(float(v) for v in value)
        if len(value) != len(current) or not all(math.isfinite(v) for v in value):
            raise ValueError(
                f"{node.name}.{name} expects {len(current)} finite components"
            )
    socket.default_value = value


ROOT_IMAGE_TAG = (
    "grase_root_image"  # ID property stamped on every image this helper creates
)


def image_from_array(name: str, pixels: np.ndarray):
    """Create or REFRESH a packed RGBA root image from finite HxWx3/4 floats.

    Texture edits re-run the whole (patched) script on the live scene, so the second
    run meets the image the first run created. An image THIS helper made (stamped with
    ``ROOT_IMAGE_TAG``) is refreshed in place — resized if needed, pixels replaced,
    repacked — so a patch never fails on the unchanged ``image_from_array`` line and
    leaves no orphan textures (2026-09-17; 15 of 16 texture-stage code errors in the
    0916 batch were this collision). An image the helper did not create (an imported
    asset texture, or anything hand-made) is still protected: that raises.

    The returned image is packed. Repacking after ``pixels.foreach_set`` + ``update`` is
    the sequence that survives on both Blender 4.2 and 4.5 (a bare second ``pack()`` on
    4.2 discarded the buffer). Changing ``colorspace_settings.name`` needs no repack.
    """
    data = np.asarray(pixels, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] not in (3, 4) or not np.isfinite(data).all():
        raise ValueError(f"Expected finite HxWx3/4 pixels, got shape={data.shape}")
    height, width = data.shape[:2]
    if not 1 <= min(height, width) <= max(height, width) <= 4096:
        raise ValueError(f"Image dimensions outside 1..4096: {width=} {height=}")
    if data.shape[2] == 3:
        data = np.concatenate([data, np.ones((height, width, 1), np.float32)], axis=2)
    image = bpy.data.images.get(name)
    if image is not None:
        if not image.get(ROOT_IMAGE_TAG):
            # Never overwrite an image this helper did not create (asset textures).
            raise ValueError(
                f"Image {name!r} already exists and was not created by image_from_array; "
                "use a fresh root-image name"
            )
        if tuple(image.size) != (width, height):
            image.scale(width, height)
    else:
        image = bpy.data.images.new(name, width=width, height=height, alpha=True)
        image[ROOT_IMAGE_TAG] = True
    image.pixels.foreach_set(np.clip(data, 0, 1).ravel())
    image.update()
    image.pack()
    return image


def wood_material(
    root: str,
    *,
    color=(0.55, 0.43, 0.30),
    direction: str = "X",
    plank_width: float = 0.14,
    grain_scale: float = 35.0,
    variation: float = 0.2,
    roughness: float = 0.55,
    seed: float = 0.0,
):
    """Assign metric-scale wood with independent strips, irregular grain and shallow relief.

    Args:
        root: Exact current non-obj_ mesh name.
        color: Linear RGB base timber color.
        direction: World X or Y grain direction.
        plank_width: Width across the grain, in meters.
        grain_scale: Fine-grain frequency in inverse meters.
        variation: Bounded strip/grain color contrast (0..0.5).
        roughness: Principled roughness (0..1).
        seed: Deterministic pattern offset, not a random global-state change.
    """
    if direction not in {"X", "Y"} or not 0.01 <= plank_width <= 2.0:
        raise ValueError(f"Invalid wood mapping: {direction=} {plank_width=}")
    if (
        not 1 <= grain_scale <= 200
        or not 0 <= variation <= 0.5
        or not 0 <= roughness <= 1
    ):
        raise ValueError(
            f"Invalid wood parameters: {grain_scale=} {variation=} {roughness=}"
        )
    base = np.asarray(color, float)
    if base.shape != (3,) or not np.isfinite(base).all() or not np.isfinite(seed):
        raise ValueError(
            "Wood color must contain three finite linear RGB values; seed must be finite"
        )
    mat = root_material(root)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    geo = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Position"], separate.inputs[0])
    cells = nodes.new("ShaderNodeMath")
    cells.operation = "DIVIDE"
    cells.inputs[1].default_value = plank_width
    links.new(separate.outputs["Y" if direction == "X" else "X"], cells.inputs[0])
    floor = nodes.new("ShaderNodeMath")
    floor.operation = "FLOOR"
    links.new(cells.outputs[0], floor.inputs[0])
    board_coord = nodes.new("ShaderNodeCombineXYZ")
    links.new(floor.outputs[0], board_coord.inputs["X"])
    board_coord.inputs["Y"].default_value = seed
    board = nodes.new("ShaderNodeTexWhiteNoise")
    links.new(board_coord.outputs[0], board.inputs["Vector"])
    offset = nodes.new("ShaderNodeCombineXYZ")
    links.new(board.outputs["Value"], offset.inputs[direction])
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    links.new(geo.outputs["Position"], add.inputs[0])
    links.new(offset.outputs[0], add.inputs[1])
    stretch = nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    stretch.inputs[1].default_value = (0.3, 10, 2) if direction == "X" else (10, 0.3, 2)
    links.new(add.outputs[0], stretch.inputs[0])
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = grain_scale
    grain.inputs["Detail"].default_value = 3.0
    grain.inputs["Roughness"].default_value = 0.65
    links.new(stretch.outputs[0], grain.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*np.clip(base * (1 - variation), 0, 1), 1)
    ramp.color_ramp.elements[1].color = (*np.clip(base * (1 + variation), 0, 1), 1)
    links.new(grain.outputs["Fac"], ramp.inputs[0])
    board_tone = nodes.new("ShaderNodeMath")
    board_tone.operation = "MULTIPLY_ADD"
    board_tone.inputs[1].default_value = variation
    board_tone.inputs[2].default_value = 1 - variation / 2
    links.new(board.outputs["Value"], board_tone.inputs[0])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1
    links.new(ramp.outputs[0], mix.inputs[1])
    links.new(board_tone.outputs[0], mix.inputs[2])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.00015
    links.new(grain.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat
