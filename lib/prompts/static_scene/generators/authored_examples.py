"""Executable geometry examples shared by enabled object-authoring prompts."""

from __future__ import annotations

BRANCHING_SOLID_EXAMPLE = """
    from mathutils import Vector
    mat_name = root.name + "_material"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    # Illustrative metres: branch starts lie INSIDE the capped stem.
    segments = [
        ("stem", (0, 0, 0), (0, 0, 0.16), 0.018),
        ("left_branch", (0, 0, 0.11), (-0.07, 0, 0.23), 0.012),
        ("right_branch", (0, 0, 0.13), (0.065, 0.018, 0.24), 0.010),
    ]
    for label, start, end, radius in segments:
        a, b = Vector(start), Vector(end)
        direction = b - a
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=radius, depth=direction.length,
            end_fill_type="NGON", location=(a + b) * 0.5)
        part = bpy.context.object
        part.rotation_mode = "QUATERNION"
        part.rotation_quaternion = direction.to_track_quat("Z", "Y")
        part.parent = root                 # this example's root is identity
        part["grase_part_label"] = label   # unique semantic part label
        part.data.materials.append(mat)
"""

BRANCHING_SOLID_GUIDANCE = """
Each cylinder already has mesh faces and capped ends; the joins overlap by real
volume. Do not duplicate coincident faces/shells or leave disconnected decorations.
Keep each branch within this one physical object. An edge-only Skin source is
rejected by source checks before modifier evaluation: supply capped mesh faces,
or select only the generated part and run bpy.ops.object.convert(target="MESH")
in your authored code before submission, so its mesh datablock contains the
evaluated faces. A modifier on an otherwise empty edge graph is not enough.
Close rounded poles with one vertex rather than duplicate zero-radius rings;
avoid tiny voxel-remesh sizes as a generic repair. Retain the original asset if
you cannot produce a valid improvement.
"""
