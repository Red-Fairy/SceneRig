"""Bake procedural Blender materials to image textures, then export GLB.

Run headless by the demo server's GLB exporter:
    blender --background <scene>.blend --python site/bake_export.py -- <out.glb>

glTF can't represent procedural shader networks, so objects with procedural
materials export untextured (grey). Here we bake each unique procedural
material's diffuse colour to an image and wire it into the Principled base
colour so the in-browser viewer shows the right look. Objects that already use
an image texture (e.g. SAM3D assets) or have no material are left as-is. The
source .blend is never saved (this process loads it read-only).
"""

from __future__ import annotations

import sys

import bpy

OUT = sys.argv[-1]
RES = 512

scene = bpy.context.scene
scene.render.engine = "CYCLES"
try:
    scene.cycles.device = "GPU"
except Exception:
    pass
scene.cycles.samples = 8
scene.cycles.bake_type = "DIFFUSE"
scene.render.bake.use_pass_direct = False
scene.render.bake.use_pass_indirect = False
scene.render.bake.use_pass_color = True
scene.render.bake.margin = 4

vl = bpy.context.view_layer


def ensure_uv(obj):
    if obj.data.uv_layers:
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    vl.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def has_image(obj):
    return any(
        m and m.use_nodes and any(n.type == "TEX_IMAGE" for n in m.node_tree.nodes)
        for m in obj.data.materials
    )


meshes = [o for o in scene.objects if o.type == "MESH" and o.data.materials]
baked = 0
for obj in meshes:
    # Skip once a shared material already has a baked image (dedupes by material).
    if has_image(obj):
        continue
    try:
        ensure_uv(obj)
    except Exception as e:  # noqa: BLE001
        print("[bake] uv fail", obj.name, e)
        continue
    img = bpy.data.images.new(f"bake_{obj.name}", RES, RES)
    added = []
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = img
        mat.node_tree.nodes.active = node
        added.append((mat.node_tree, node))
    if not added:
        continue
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    vl.objects.active = obj
    try:
        bpy.ops.object.bake(type="DIFFUSE")
        baked += 1
    except Exception as e:  # noqa: BLE001
        print("[bake] bake fail", obj.name, e)
        continue
    for nt, node in added:
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            try:
                nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            except Exception:  # noqa: BLE001
                pass

print(f"[bake] baked {baked} materials of {len(meshes)} objects")
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_apply=True)
print("[bake] exported", OUT)
