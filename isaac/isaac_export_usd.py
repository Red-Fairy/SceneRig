"""Blender payload: export a GRASE final.blend to USD for Isaac.

Run inside the repo Blender (headless):
    blender -b <final.blend> --python isaac/isaac_export_usd.py -- <out.usdc>

Exports the whole scene (meshes, transforms, materials, textures, lights, camera) in Z-up meters.
Materials that are procedural node graphs (e.g. the wood-grain workbench) are first
Cycles-baked to image textures — the USD exporter silently drops procedural nodes,
which otherwise ships white surfaces. Physics is NOT added here — see isaac_add_physics.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import bpy

BAKE_RES = 1024
BAKE_SAMPLES = 16

out_path = sys.argv[sys.argv.index("--") + 1]


def needs_bake(mat):
    """Base color must be a flat value or a PLAIN image texture to export; anything
    else (procedural, image-through-mix) must bake."""
    if not mat or not mat.use_nodes:
        return False
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False
    links = bsdf.inputs["Base Color"].links
    return bool(links) and links[0].from_node.type != "TEX_IMAGE"


def bake_object(obj):
    todo = [s.material for s in obj.material_slots if needs_bake(s.material)]
    if not todo:
        return
    uv = obj.data.uv_layers.new(name="bake_uv")
    obj.data.uv_layers.active = uv
    for l in obj.data.uv_layers:
        l.active_render = l.name == "bake_uv"
    # select ONLY this object before edit mode: multi-object editing would
    # smart-project every selected mesh, overwriting their original UVMaps
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    staged = []
    for mat in todo:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        # camera/window/reflection coords are meaningless at bake time -> Generated
        for n in list(nt.nodes):
            if n.type == "TEX_COORD":
                for o in n.outputs:
                    if o.name in ("Camera", "Window", "Reflection") and o.links:
                        for lk in list(o.links):
                            nt.links.new(n.outputs["Generated"], lk.to_socket)
        img = bpy.data.images.new(f"bake_{obj.name}_{mat.name}", 1024, 1024)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        src = bsdf.inputs["Base Color"].links[0].from_socket
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(src, em.inputs["Color"])
        outn = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
        nt.nodes.active = node  # AFTER nodes.new calls: new nodes steal active status
        staged.append((nt, node, bsdf, img, outn, mat))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT")
    for nt, node, bsdf, img, outn, mat in staged:
        nt.links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
        nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        img.pack()
        px = list(img.pixels)[0::4][:60000]
        print(f"  baked {obj.name}/{mat.name} mean={sum(px) / len(px):.3f}")


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = BAKE_SAMPLES
scene.render.bake.use_pass_direct = False
scene.render.bake.use_pass_indirect = False

for obj in [o for o in scene.objects if o.type == "MESH"]:
    bake_object(obj)

bpy.ops.wm.usd_export(
    filepath=out_path,
    # Match the harness/current evaluated depsgraph, including modifier visibility
    # and subdivision levels. Export the evaluated surface, not a control cage.
    evaluation_mode="VIEWPORT",
    export_subdivision="TESSELLATE",
    export_materials=True,
    export_textures=True,
    selected_objects_only=False,
    export_animation=False,
    # bake the blend's camera (the GRASE input-photo viewpoint) and lights (capture-time
    # illumination) into the asset; consumers can still adjust or add their own at runtime
    export_cameras=True,
    export_lights=True,
)
print(f"ISAAC_EXPORT_OK {out_path}")

# The optional paired output is a fresh geometry-only collision stage. Keep the
# already-exported visual hierarchy/materials untouched; never save the input Blend.
extra = sys.argv[sys.argv.index("--") + 2 :]
if extra:
    if len(extra) != 2:
        raise RuntimeError("Expected <visual.usdc> [<collision.usdc> <request.json>]")
    collision_path, request_path = map(Path, extra)
    request = json.loads(request_path.read_text())
    helper_path = Path(__file__).with_name("blender_collision_source.py")
    spec = importlib.util.spec_from_file_location(
        "_grase_collision_source", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load collision-source helper: {helper_path!s}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import bmesh  # Optional dependency: only available inside Blender.

    records = helper.prepare_collision_source(request, bpy=bpy, bmesh=bmesh)
    bpy.ops.wm.usd_export(
        filepath=str(collision_path),
        evaluation_mode="VIEWPORT",
        export_subdivision="TESSELLATE",
        export_materials=False,
        export_textures=False,
        selected_objects_only=False,
        export_animation=False,
        export_cameras=False,
        export_lights=False,
    )
    hashes = {}
    for label, path in (("visual", Path(out_path)), ("collision", collision_path)):
        with path.open("rb") as stream:
            hashes[label + "_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    collision_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": request["policy"],
                "source_blend_sha256": request["blend_sha256"],
                **hashes,
                "objects": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"ISAAC_COLLISION_SOURCE_OK {collision_path}")
