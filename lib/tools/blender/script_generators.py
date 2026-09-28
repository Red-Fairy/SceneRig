"""Blender Script Generators.

Generates Python scripts for Blender operations used by investigator and
exec_blender tools. Contains reusable script generation methods for scene
inspection, rendering, and camera manipulation.

Object names that originate from the LLM / scene graph are interpolated into
the generated scripts via ``json.dumps`` (a valid Python string literal), never
raw inside ``'...'`` — a name containing a quote, backslash, or newline would
otherwise break the script or inject arbitrary code into the Blender process.
"""

import json
import math
import re
from typing import Optional

from lib.tools.geometry.contact_policy import CONTACT_TOLERANCE_M

_SUFFIX_RE = re.compile(
    r"^(.+)\.(\d{3,})$"
)  # Blender's auto-rename: "<base>.001", ".002", ...


COVERAGE_ID_SLOTS = 9  # red levels spaced 28/255 that survive the id round-trip


# This source is embedded in generated Blender scripts.  A multipart GPT-authored
# object is a canonical, parentless EMPTY named by the pipeline, with its editable
# mesh primitives below it.  The environment binding is essential: an arbitrary
# helper EMPTY in the scene must keep its historical independent-object semantics.
_PIPELINE_EMPTY_ROOTS_SCRIPT = r'''
try:
    _pipeline_names_payload = json.loads(
        os.environ.get("GRASE_PIPELINE_OBJECT_NAMES", "[]")
    )
except (TypeError, ValueError, json.JSONDecodeError):
    _pipeline_names_payload = []
if not isinstance(_pipeline_names_payload, list):
    _pipeline_names_payload = []
_pipeline_names = [
    name for name in _pipeline_names_payload if isinstance(name, str) and name
]
_pipeline_empty_parts = {}
_pipeline_part_owner = {}
for _root_name in _pipeline_names:
    _root = bpy.context.scene.objects.get(_root_name)
    if _root is None or _root.type != "EMPTY" or _root.parent is not None:
        continue
    _graph_id = _root.get("grase_graph_id")
    if not isinstance(_graph_id, str) or not _graph_id.strip():
        continue
    _descendants = tuple(_root.children_recursive)
    if not _descendants or any(child.type != "MESH" for child in _descendants):
        continue
    _parts = tuple(
        sorted(
            _descendants,
            key=lambda child: child.name,
        )
    )
    _pipeline_empty_parts[_root_name] = _parts
    for _part in _parts:
        _pipeline_part_owner[_part] = _root
'''


_LOGICAL_TARGET_SCRIPT = _PIPELINE_EMPTY_ROOTS_SCRIPT + r'''
def _target_parts(obj):
    if obj is None:
        return []
    if obj.type == "EMPTY" and obj.name in _pipeline_empty_parts:
        return list(_pipeline_empty_parts[obj.name])
    if obj.type != "MESH":
        return []
    owner = _pipeline_part_owner.get(obj)
    if owner is not None:
        return list(_pipeline_empty_parts[owner.name])
    return [obj]


def _logical_target_name(mesh):
    owner = _pipeline_part_owner.get(mesh)
    return owner.name if owner is not None else mesh.name


def _resolve_target_parts(name):
    exact = bpy.data.objects.get(name)
    parts = _target_parts(exact)
    if parts:
        return parts

    meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name not in ("Ground", "Plane")
    ]
    key = name.split("#")[0].strip().lower().replace(" ", "_")
    if key:
        # Prefer the declared logical root name over implementation-detail part names.
        for root_name in _pipeline_names:
            if root_name not in _pipeline_empty_parts:
                continue
            if key in root_name.lower().replace(" ", "_"):
                return list(_pipeline_empty_parts[root_name])
        for mesh in meshes:
            logical_name = _logical_target_name(mesh)
            if key in logical_name.lower().replace(" ", "_") or key in mesh.name.lower().replace(" ", "_"):
                return _target_parts(mesh)

    # Preserve the historical fallback: choose the largest individual mesh, then
    # expand it only when that mesh belongs to an explicitly bound Empty root.
    fallback = max(
        meshes,
        key=lambda mesh: mesh.dimensions.x * mesh.dimensions.y * mesh.dimensions.z,
        default=None,
    )
    return _target_parts(fallback)


def _world_bbox(parts):
    corners = [
        part.matrix_world @ Vector(corner)
        for part in parts
        for corner in part.bound_box
    ]
    if not corners:
        return None, None, None
    lo = Vector(tuple(min(corner[i] for corner in corners) for i in range(3)))
    hi = Vector(tuple(max(corner[i] for corner in corners) for i in range(3)))
    return lo, hi, (lo + hi) * 0.5
'''


def coverage_id_plan(names):
    ""
    names = list(names)
    ids = {i + 1: n for i, n in enumerate(names[:COVERAGE_ID_SLOTS])}
    return ids, names[COVERAGE_ID_SLOTS:]


def dedup_surface_plan(names):
    """Plan for collapsing Blender's auto-suffixed duplicate root surfaces (pure; unit-tested).

    The initializer agent may re-run full scene construction each iteration; ``obj.name =
    "Table"`` then collides with the existing surface, so Blender appends ``.001`` / ``.002``
    and a duplicate accumulates (coincident copies self-shadow black in Cycles; a *moved*
    rebuild leaves two separate tables). A ``.NNN`` suffix only ever appears on an exact name
    collision, so it is ALWAYS an accidental duplicate (genuinely distinct surfaces get
    distinct names like ``Wall`` / ``SideWall``). We therefore collapse each ``<base>`` group
    to a single survivor regardless of position: keep the NEWEST (the agent's latest build =
    highest suffix; the un-suffixed base is oldest), drop the rest, and rename the survivor
    back to ``<base>`` so later references stay stable.

    ``names``: iterable of object names (``obj_*`` object meshes are skipped — never touched).
    Returns ``(remove, rename)``: ``remove`` = names to delete, ``rename`` = ``{old: base}``.

    The guardrail in ``data/static_scene/generator_script.py`` mirrors this rule (it can't
    import this module: it runs inside Blender from a bare ``--python`` invocation)."""
    groups: dict[str, list[str]] = {}
    for n in names:
        if n.startswith("obj_"):  # imported SAM3D object meshes -> leave alone
            continue
        m = _SUFFIX_RE.match(n)
        groups.setdefault(m.group(1) if m else n, []).append(n)

    def _suffix(n: str) -> int:
        m = _SUFFIX_RE.match(n)
        return int(m.group(2)) if m else -1  # un-suffixed base sorts oldest

    remove: list[str] = []
    rename: dict[str, str] = {}
    for base, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=_suffix)
        keep = members[-1]  # newest = the agent's latest build
        remove.extend(members[:-1])
        if keep != base:
            rename[keep] = base
    return remove, rename


def generate_scene_info_script(output_path: str) -> str:
    """Generate script to extract scene information with bounding boxes.

    Args:
        output_path: Path where the JSON scene info will be saved.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os
import sys
from mathutils import Vector

{_PIPELINE_EMPTY_ROOTS_SCRIPT}

# Get scene information
scene_info = {{"objects": [], "materials": [], "lights": [], "cameras": [], "world": {{}}, "color_management": {{}}}}
import importlib.util
scene_info["runtime_capabilities"] = {{
    "blender_version": bpy.app.version_string,
    "python_version": sys.version.split()[0],
    "modules": {{name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "PIL")}},
    "material_helpers": "from lib.tools.blender.material_helpers import root_material, set_socket, image_from_array, wood_material",
    "code_contract": "Use current object.material_slots, not historical material names. Helper APIs are Blender-only. A missing optional module is not available just because the main pipeline has it.",
}}
scene_info["lighting_material_limits"] = []

# Root surfaces (ALL non-obj_ meshes: table/wall/floor/ceiling/cabinet) FIRST so the
# object cap below can never hide them.  An explicitly pipeline-bound Empty is one
# logical object: report its union world bounds/materials on the canonical root and
# suppress its implementation-detail mesh descendants.  Ordinary helper Empties keep
# the historical behavior because they are absent from _pipeline_empty_parts.
_scene_objs = [
    obj
    for obj in bpy.data.objects
    if obj.type not in ('CAMERA', 'LIGHT') and obj not in _pipeline_part_owner
]
# Canonical pipeline roots retain the required obj_* prefix, so the historical
# surface-first ordering remains exact for both mesh and Empty representations.
_scene_objs.sort(key=lambda o: 1 if o.name.startswith('obj_') else 0)
for obj in _scene_objs:
    # Calculate bounding box in world coordinates
    bbox = None
    _logical_parts = _pipeline_empty_parts.get(obj.name)
    if _logical_parts:
        bbox_corners = [
            part.matrix_world @ Vector(corner)
            for part in _logical_parts
            for corner in part.bound_box
        ]
    elif hasattr(obj, 'bound_box') and obj.bound_box:
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    else:
        bbox_corners = []
    if bbox_corners:
        min_x = min(corner.x for corner in bbox_corners)
        min_y = min(corner.y for corner in bbox_corners)
        min_z = min(corner.z for corner in bbox_corners)
        max_x = max(corner.x for corner in bbox_corners)
        max_y = max(corner.y for corner in bbox_corners)
        max_z = max(corner.z for corner in bbox_corners)
        bbox = {{
            "min": [round(min_x, 2), round(min_y, 2), round(min_z, 2)],
            "max": [round(max_x, 2), round(max_y, 2), round(max_z, 2)],
            "center": [round((min_x + max_x) / 2, 2), round((min_y + max_y) / 2, 2), round((min_z + max_z) / 2, 2)],
            "size": [round(max_x - min_x, 2), round(max_y - min_y, 2), round(max_z - min_z, 2)]
        }}

    _material_objects = _logical_parts or (obj,)
    if _logical_parts:
        _material_slots = []
        for _material_obj in _material_objects:
            for _material_name in [s.material.name for s in
                                   getattr(_material_obj, "material_slots", []) if s.material]:
                if _material_name not in _material_slots:
                    _material_slots.append(_material_name)
    else:
        _material_slots = [s.material.name for s in
                           getattr(obj, "material_slots", []) if s.material]
    _entry = {{
        "name": obj.name,
        "type": obj.type,
        "location": [round(x, 2) for x in obj.matrix_world.translation],
        "rotation": [round(x, 2) for x in obj.rotation_euler],
        "scale": [round(x, 2) for x in obj.scale],
        "visible": not (obj.hide_viewport or obj.hide_render),
        "bbox": bbox,
        "material_slots": _material_slots,
    }}
    if _logical_parts:
        _entry["mesh_parts"] = [part.name for part in _logical_parts]
    scene_info["objects"].append(_entry)
    if len(scene_info["objects"]) >= 40:
        break

# Root-surface materials (desk_mat / wall_mat) FIRST so the cap below can never
# truncate them -- the texture agent must see the wall/desk material to texture the
# room. Reconstructed-asset materials are named obj_*_mat (their baked SAM3D texture);
# they sort last and should be left alone (never cleared/rebuilt).
_frozen_materials = {{s.material.name for o in bpy.data.objects
                     if o.name.startswith("obj_") or o in _pipeline_part_owner
                     for s in getattr(o, "material_slots", []) if s.material}}
_mats = sorted(bpy.data.materials, key=lambda m: m.name in _frozen_materials)


def _principled(mat):
    # The SHADED values, which are what renders. mat.diffuse_color is the viewport display
    # colour and does NOT track the Principled base colour -- reporting it made a textured
    # surface look like flat default grey, the exact thing the texture verifier's check 1
    # ("still default gray/white") keys on.
    if not mat.use_nodes or not mat.node_tree:
        return {{}}
    node = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if node is None:
        return {{}}
    out = {{}}
    for key, field in (("base_color", "Base Color"), ("roughness", "Roughness"),
                       ("metallic", "Metallic")):
        sock = node.inputs.get(field)
        if sock is None:
            continue
        if sock.is_linked:
            out[key] = "textured"  # driven by a node graph, not a constant
        elif key == "base_color":
            out[key] = [round(v, 2) for v in sock.default_value[:3]]
        else:
            out[key] = round(sock.default_value, 2)
    return out


for mat in _mats:
    entry = {{"name": mat.name, "use_nodes": mat.use_nodes}}
    # Frozen BRDF scalars are diagnostic lighting context, never editing authority.
    entry.update(_principled(mat))
    entry["editable"] = mat.name not in _frozen_materials and not mat.name.startswith("obj_")
    if not entry["editable"] and isinstance(entry.get("roughness"), (int, float)) and entry["roughness"] >= 0.8:
        scene_info["lighting_material_limits"].append({{"material": mat.name,
            "roughness": entry["roughness"], "metallic": entry.get("metallic"),
            "note": "Frozen high-roughness BRDF: a sharp white highlight may be material-limited. This is not proof that lighting is correct."}})
    if entry["editable"] and mat.use_nodes and mat.node_tree:
        entry["nodes"] = [{{"name": n.name, "type": n.bl_idname,
            "inputs": list(n.inputs.keys())}} for n in list(mat.node_tree.nodes)[:12]]
    scene_info["materials"].append(entry)
    if len(scene_info["materials"]) >= 20:
        break

for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    scene_info["lights"].append({{
        "name": light.name,
        "type": light.data.type,
        "energy": light.data.energy,
        "color": [round(x, 2) for x in light.data.color],
        "location": [round(x, 2) for x in light.matrix_world.translation],
        "rotation": [round(x, 2) for x in light.rotation_euler]
    }})
    if len(scene_info["lights"]) >= 5:
        break

for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
    scene = bpy.context.scene
    scene_info["cameras"].append({{
        "name": cam.name,
        "lens": cam.data.lens,
        "location": [round(x, 2) for x in cam.matrix_world.translation],
        "rotation": [round(x, 2) for x in cam.rotation_euler],
        "is_active": cam == scene.camera,
    }})
    if len(scene_info["cameras"]) >= 3:
        break

# World (ambient/background) + color management — the LIGHTING stage's scope, so the agent
# reads current values instead of guessing (makes the get_scene_info "world settings" claim true).
scene = bpy.context.scene
_world = scene.world
_world_info = {{"name": None, "use_nodes": False, "strength": None, "color": None}}
if _world is not None:
    _world_info["name"] = _world.name
    _world_info["use_nodes"] = _world.use_nodes
    _bg = _world.node_tree.nodes.get("Background") if (_world.use_nodes and _world.node_tree) else None
    if _bg is not None:
        _world_info["strength"] = round(_bg.inputs["Strength"].default_value, 3)
        _world_info["color"] = [round(x, 2) for x in _bg.inputs["Color"].default_value[:3]]
    else:
        _world_info["color"] = [round(x, 2) for x in _world.color]
scene_info["world"] = _world_info

_vs = scene.view_settings
scene_info["color_management"] = {{
    "view_transform": _vs.view_transform,
    "look": _vs.look,
    "exposure": round(_vs.exposure, 3),
    "gamma": round(_vs.gamma, 3),
    "display_device": scene.display_settings.display_device,
}}

# Save to file for retrieval
with open("{output_path}", "w") as f:
    json.dump(scene_info, f)

print("Scene info extracted successfully")
sys.stdout.flush()
# This is a READ-ONLY info query. Exit now (the file is written + closed) so the
# shared wrapper does NOT run its trailing 512-sample Cycles render + blend save —
# pure cost here and, under GPU contention, the main failure surface for this tool.
# os._exit avoids raising SystemExit (which the wrapper's bare except would turn into
# a hard error).
os._exit(0)
'''


def identify_main_support(bodies: list[dict]) -> tuple[Optional[str], Optional[float]]:
    """Geometrically pick the main supporting surface from the penetration-script bodies.

    ``bodies``: ``[{"name", "is_obj", "lo":[x,y,z], "hi":[x,y,z]}]`` (world AABBs). The main
    support is the **horizontal** root-surface slab (its thinnest extent is the Z axis, so a
    wall is excluded) whose XY footprint covers at least one object's XY centre, taking the
    one with the **highest top** (so the table is chosen over the floor below it). Returns
    ``(name, top_z)`` or ``(None, None)`` if none qualifies."""
    objs = [b for b in bodies if b.get("is_obj")]
    obj_xy = [
        ((b["lo"][0] + b["hi"][0]) / 2.0, (b["lo"][1] + b["hi"][1]) / 2.0) for b in objs
    ]
    best = None
    for s in bodies:
        if s.get("is_obj"):
            continue
        lo, hi = s["lo"], s["hi"]
        ext = [hi[i] - lo[i] for i in range(3)]
        if ext[2] > min(ext[0], ext[1]):  # Z not the thinnest -> a wall, not a top
            continue
        covers = any(lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] for x, y in obj_xy)
        if not covers:
            continue
        if best is None or hi[2] > best[1]:  # highest top wins (table over floor)
            best = (s["name"], hi[2])
    return best if best else (None, None)


def _unit(v: Optional[list]) -> Optional[list]:
    if not v:
        return None
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return [v[0] / n, v[1] / n, v[2] / n] if n > 1e-9 else None


def _acute_angle_deg(a: Optional[list], b: Optional[list]) -> Optional[float]:
    """Unsigned 0..90 angle between two directions (sign-invariant; PCA normals lack sign)."""
    import math

    a, b = _unit(a), _unit(b)
    if not a or not b:
        return None
    d = max(0.0, min(1.0, abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))
    return math.degrees(math.acos(d))


def _signed_gap(A: dict, cB: list, nB: list) -> float:
    """Most-negative signed distance of A's footprint to B's plane (point ``cB``, unit normal
    ``nB``), oriented so A's own side of B is POSITIVE. ``> 0`` -> A's nearest point floats off B
    by that much; ``< 0`` -> A pokes THROUGH to the far side of B by that much. Uses A's TRUE
    convex-hull footprint (``A["hull"]`` XY points x its z-range) when available — a YAWED table's
    world AABB is far larger than the table, so AABB corners report phantom penetration against a
    diagonal wall; falls back to the 8 AABB corners when no hull was emitted. Used by both
    ``against`` (flush check) and ``corner`` (cut-through check)."""
    cA = A["c"]
    s = 1.0 if sum((cA[i] - cB[i]) * nB[i] for i in range(3)) >= 0 else -1.0
    hull = A.get("hull")
    if hull:
        pts = [(p[0], p[1], z) for p in hull for z in (A["lo"][2], A["hi"][2])]
    else:
        pts = [
            (cx, cy, cz)
            for cx in (A["lo"][0], A["hi"][0])
            for cy in (A["lo"][1], A["hi"][1])
            for cz in (A["lo"][2], A["hi"][2])
        ]
    return min(
        s * ((x - cB[0]) * nB[0] + (y - cB[1]) * nB[1] + (z - cB[2]) * nB[2])
        for x, y, z in pts
    )


def _out_dir(cA: list, cB: list, nB: Optional[list]) -> Optional[list]:
    """HORIZONTAL unit direction that pulls a body at ``cA`` AWAY from B's plane (point ``cB``,
    normal ``nB``), sign-resolved toward A's own side — the PCA normal's sign is arbitrary, so the
    raw normal is a coin flip for "out". z is zeroed (the main support's top must stay at z=0).
    None when ``nB`` is missing or near-vertical (no horizontal escape direction)."""
    n = _unit(nB)
    if not n:
        return None
    s = 1.0 if sum((cA[i] - cB[i]) * n[i] for i in range(3)) >= 0 else -1.0
    h = ((s * n[0]) ** 2 + (s * n[1]) ** 2) ** 0.5
    if h < 1e-6:
        return None
    return [s * n[0] / h, s * n[1] / h, 0.0]


def _anchored_built_wall(
    B: dict,
    sg_entry: Optional[dict],
    dist_tol: float = 0.15,
    ang_tol_deg: float = 25.0,
) -> bool:
    ""
    from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane

    if not sg_entry:
        return False
    wc = sg_entry.get("world_center")
    nm, kind = plumb_plane(sg_entry.get("normal"))
    if not wc or not nm or kind == "ambiguous" or not plane_is_reliable(sg_entry):
        return False
    c = B["c"]
    if abs(sum((c[i] - wc[i]) * nm[i] for i in range(3))) > dist_tol:
        return False
    ang = _acute_angle_deg(B.get("n"), nm)
    return ang is None or ang <= ang_tol_deg


def _min_obj_margin_to_plane(
    objs: list, cA: list, cP: list, nP: Optional[list]
) -> Optional[float]:
    """Smallest signed distance (A's side positive) from any object AABB corner to the plane
    (``cP``, ``nP``) — how much table remains under the deepest object once A's near edge
    retreats to that plane. None when there are no objects (nothing to uncover) or no
    usable normal. Objects on other supports only make the bound conservative."""
    n = _unit(nP)
    if not n or not objs:
        return None
    s = 1.0 if sum((cA[i] - cP[i]) * n[i] for i in range(3)) >= 0 else -1.0
    return min(
        s * ((x - cP[0]) * n[0] + (y - cP[1]) * n[1] + (z - cP[2]) * n[2])
        for ob in objs
        for x in (ob["lo"][0], ob["hi"][0])
        for y in (ob["lo"][1], ob["hi"][1])
        for z in (ob["lo"][2], ob["hi"][2])
    )


def _far_edge_margin(objs: list, A: dict, out: list) -> Optional[float]:
    """How far A's FAR edge (the one most along ``out``, i.e. away from the wall) extends
    past the outermost object corner, measured along ``out`` — the room A has to translate
    TOWARD the wall before that edge slides out from under an object. None when there are
    no objects. Uses A's true hull footprint when available (matches ``_signed_gap``)."""
    if not objs:
        return None
    hull = A.get("hull")
    pts = (
        hull
        if hull
        else [
            (x, y) for x in (A["lo"][0], A["hi"][0]) for y in (A["lo"][1], A["hi"][1])
        ]
    )
    far = max(p[0] * out[0] + p[1] * out[1] for p in pts)
    om = max(
        x * out[0] + y * out[1]
        for ob in objs
        for x in (ob["lo"][0], ob["hi"][0])
        for y in (ob["lo"][1], ob["hi"][1])
    )
    return far - om


def _relationship_enforcement(rel: dict) -> str:
    ""
    from lib.tools.geometry.surface_relations import relationship_schema_is_valid

    if not relationship_schema_is_valid(rel):
        return "none"
    return str(rel["enforcement"]).strip().lower()


def wall_yaw_candidates(
    nodes: list, relationships: list, main_id: str, max_nz: float = 0.3
) -> list[dict]:
    ""
    from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
    from lib.tools.geometry.surface_relations import surface_build_name

    by_id = {n.get("id"): n for n in (nodes or [])}
    out: list[dict] = []
    for index, rel in enumerate(relationships or []):
        rel_type = str(rel.get("type") or "").strip().lower()
        if rel_type not in ("against", "perpendicular"):
            continue
        aid, bid = rel.get("a"), rel.get("b")
        if main_id not in (aid, bid):
            continue
        wall_id = bid if aid == main_id else aid
        node = by_id.get(wall_id) or {}
        plane = node.get("plane") or {}
        relationship_enforcement = _relationship_enforcement(rel)
        # PERPENDICULAR owns contact/form only.  Even a confirmed hard row does
        # not pin the support's free in-plane yaw; retain its reliable wall run as
        # advisory visual evidence while reserving hard yaw authority for AGAINST.
        yaw_enforcement = (
            "advisory"
            if rel_type == "perpendicular" and relationship_enforcement == "hard"
            else relationship_enforcement
        )
        status = str(rel.get("status") or "legacy").strip().lower()
        relationship_id = rel.get("relationship_id") or (
            f"legacy:{index}:{rel_type}:{aid}:{bid}"
        )
        candidate = {
            "id": f"relationship:{relationship_id}",
            "relationship_id": relationship_id,
            "source": "measured_wall_plane",
            "surface": surface_build_name(wall_id) if wall_id else None,
            "surface_id": wall_id,
            "relationship": rel_type,
            "direction": None,
            "confidence": (
                "strong" if relationship_enforcement == "hard" else "advisory"
            ),
            "enforcement": yaw_enforcement,
            "relationship_enforcement": relationship_enforcement,
            "preprocess_status": status,
            "usable": False,
            "reason": None,
            "metrics": {
                "mask_fraction": plane.get("mask_frac"),
                "plane_inlier_fraction": plane.get("inlier_frac"),
            },
        }
        if relationship_enforcement != "hard":
            candidate["reason"] = (
                f"relationship enforcement is {relationship_enforcement}"
            )
        elif not node or node.get("kind") != "root_surface":
            candidate["reason"] = "wall surface is absent from the scene graph"
        elif not plane_is_reliable(plane):
            candidate["reason"] = "measured wall plane is unreliable"
        else:
            normal, kind = plumb_plane(plane.get("normal"))
            if not normal or kind != "wall" or abs(float(normal[2])) >= max_nz:
                candidate["reason"] = "measured partner is not a vertical wall plane"
            else:
                h = (float(normal[0]) ** 2 + float(normal[1]) ** 2) ** 0.5
                if h < 1e-6:
                    candidate["reason"] = "measured wall run is degenerate"
                else:
                    candidate["direction"] = [
                        -float(normal[1]) / h,
                        float(normal[0]) / h,
                        0.0,
                    ]
                    candidate["usable"] = True
                    if yaw_enforcement == "advisory":
                        candidate["reason"] = (
                            "PERPENDICULAR contact does not pin main-support yaw"
                        )
        out.append(candidate)
    return out


def wall_prior_run(
    nodes: list, relationships: list, main_id: str, max_nz: float = 0.3
) -> Optional[list]:
    """Expected RUN of the main support from an adjacent reliable WALL plane.

    A desk that is against / next to a wall runs parallel to it in practice (and the
    mod-90 fold makes parallel vs perpendicular irrelevant), and the wall's plane is
    RANSAC-anchored to actual 3D points — far more reliable than the desk mask's
    grazing z=0 unprojection (0709_eval_gpt1: wall prior -4.7° vs PCA noise -30°;
    0709_eval_bridge1: wall prior -27.7° tracks the genuine -25° rotation, because a
    rotated table sits against an equally rotated wall). Used as the yaw anchor when
    the mask-PCA is weak. Prefers ``against`` partners over ``perpendicular``, then
    the highest inlier fraction; walls must pass the shared reliable-plane contract
    (planarity >= 0.6, mask coverage >= 2%) and be near-vertical.

    The sliver rule is a HARD FILTER, not a demotion, because ``inlier_frac`` cannot express
    it: a wall seen edge-on unprojects to near-collinear points, so RANSAC reports a PERFECT
    fit for a plane whose horizontal run is unconstrained. Demoting by score would still let
    it win a scene where every candidate is a sliver. 0801_rdj_push_t_random: a
    0.56%-of-frame wall scored ``inlier_frac`` 1.0, beat the 6.6% back wall at 0.4586, and
    pinned the table 14deg off the photo — while L3's ``against`` check, reading BUILT runs,
    demanded +14 back off a different wall. ``coverage_report`` already refuses to grade
    these ("photo mask too small to compare"); this applies the same bar to the yaw anchor.
    ``inlier_frac`` stays a ranking key among reliable candidates. With no candidate
    left, the resolver falls through to its mask-PCA rungs rather than trusting an
    unmeasured direction."""
    cands = [c for c in wall_yaw_candidates(nodes, relationships, main_id, max_nz)
             if c.get("usable")]
    if not cands:
        return None
    winner = sorted(
        cands,
        key=lambda c: (
            0 if c.get("relationship") == "against" else 1,
            -float((c.get("metrics") or {}).get("plane_inlier_fraction") or 0.0),
            str(c.get("relationship_id")),
        ),
    )[0]
    return winner["direction"]


def mask_world_run(
    mask,
    cam_loc: list,
    look_at: list,
    lens_mm: float,
    plane_z: float = 0.0,
    sensor_mm: float = 36.0,
    max_pts: int = 4000,
) -> tuple[Optional[list], Optional[float]]:
    """Photo-anchored RUN direction of a level surface: unproject its segmentation-mask pixels
    through the reference camera onto the world plane ``z=plane_z``, PCA in world XY.

    Returns ``(run_xyz, ecc)`` — the dominant horizontal direction of the surface as the PHOTO
    shows it, and the footprint's elongation (sqrt eigenvalue ratio; ~1 = isotropic, no reliable
    direction). This is the exact-world-space variant of the L2 orientation check: comparing it
    to the BUILT surface's run (``surface_runs``) yields the true world-yaw delta, not an
    image-space approximation. Camera: ``location``/``look_at`` from pseudo_gt/cameras.json
    (roll-free after gravity de-roll) with a ``sensor_mm``-wide Blender default sensor, AUTO fit
    (the larger image dimension spans the sensor). ``(None, None)`` when degenerate (tiny mask,
    straight-down camera, rays missing the plane)."""
    import numpy as np

    m = np.asarray(mask) > 0
    ys, xs = np.nonzero(m)
    if len(xs) < 32:
        return None, None
    h, w = m.shape[:2]
    stride = max(1, int(len(xs) / max_pts))
    xs, ys = xs[::stride], ys[::stride]
    loc = np.asarray(cam_loc, float)
    fwd = np.asarray(look_at, float) - loc
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    nr = np.linalg.norm(right)
    if nr < 1e-6:  # camera looking straight down: image directions are yaw-ambiguous
        return None, None
    right /= nr
    upc = np.cross(right, fwd)
    tan_fit = (sensor_mm / 2.0) / lens_mm  # AUTO fit: larger dimension spans the sensor
    tan_h = tan_fit if w >= h else tan_fit * w / h
    tan_v = tan_fit * h / w if w >= h else tan_fit
    u = (xs + 0.5) / w - 0.5
    v = 0.5 - (ys + 0.5) / h
    rays = (
        fwd[None, :]
        + (2 * tan_h) * u[:, None] * right[None, :]
        + (2 * tan_v) * v[:, None] * upc[None, :]
    )
    dz = rays[:, 2]
    keep = (
        dz < -1e-6 if loc[2] > plane_z else dz > 1e-6
    )  # rays that hit the plane ahead
    rays = rays[keep]
    if len(rays) < 32:
        return None, None
    t = (plane_z - loc[2]) / rays[:, 2]
    pts = loc[None, :2] + t[:, None] * rays[:, :2]
    pts = pts - pts.mean(axis=0)
    cov = pts.T @ pts / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    ecc = float(np.sqrt(evals[1] / max(evals[0], 1e-12)))
    d = evecs[:, 1]
    return [float(d[0]), float(d[1]), 0.0], ecc


def mask_border_evidence(mask) -> dict:
    """Describe whether a 2-D mask is clipped by the image boundary.

    Eccentricity can look extremely confident after the image frame cuts away part
    of a support.  Preserve both the touched edge identities and quantitative border
    occupancy so yaw adjudication can demote that PCA without hiding why.
    """
    import numpy as np

    arr = np.asarray(mask, bool)
    empty = {
        "border_touch_edges": [],
        "border_touch_pixel_count": 0,
        "border_touch_fraction": 0.0,
        "mask_border_pixel_fraction": 0.0,
        "is_truncated": False,
    }
    if arr.ndim != 2 or arr.size == 0:
        return {**empty, "reason": "mask_missing_or_not_2d"}
    h, w = arr.shape
    edge_rows = {
        "top": arr[0, :],
        "right": arr[:, -1],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
    }
    touched = [name for name, values in edge_rows.items() if bool(values.any())]
    boundary = np.zeros_like(arr, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    border_count = int((arr & boundary).sum())
    perimeter = max(1, 2 * h + 2 * w - 4)
    mask_count = int(arr.sum())
    return {
        "border_touch_edges": touched,
        "border_touch_pixel_count": border_count,
        "border_touch_fraction": float(border_count / perimeter),
        "mask_border_pixel_fraction": float(border_count / max(mask_count, 1)),
        "is_truncated": bool(touched),
    }


def _rect_ratio(hull: Optional[list], run: Optional[list]) -> Optional[float]:
    """How rectangular a surface's XY footprint is: hull area / (its oriented bbox area), with
    the bbox axes given by the surface's dominant ``run`` direction. ~1.0 for a rectangle/square,
    ~0.785 for a disc, lower for irregular shapes. None when hull/run is missing or degenerate."""
    import math

    if not hull or len(hull) < 3 or not run:
        return None
    n = math.hypot(run[0], run[1])
    if n < 1e-9:
        return None
    ux, uy = run[0] / n, run[1] / n
    proj = [(p[0] * ux + p[1] * uy, -p[0] * uy + p[1] * ux) for p in hull]
    e1 = max(a for a, _ in proj) - min(a for a, _ in proj)
    e2 = max(b for _, b in proj) - min(b for _, b in proj)
    if e1 * e2 < 1e-9:
        return None
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    pts = sorted(
        hull, key=lambda p: math.atan2(p[1] - cy, p[0] - cx)
    )  # convex -> orderable
    area = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0 / (e1 * e2)


def _top_hull_run_info(hull: Optional[list], rect_min: float = 0.9) -> dict:
    """Fit an edge run and applicability from a finite TOP hull.

    This is the host-Python twin of the generated Blender dump's rotating-calipers
    fit.  Relationship yaw must never reuse a whole grouped furniture body's run:
    legs/pedestals can dominate that footprint while the contacting top edge points
    elsewhere.
    """
    if not hull or len(hull) < 3:
        return {
            "status": "unknown",
            "reason": "top_hull_missing",
            "run": None,
            "rectangularity": None,
        }
    try:
        points = [(float(point[0]), float(point[1])) for point in hull]
    except (TypeError, ValueError, IndexError):
        return {
            "status": "unknown",
            "reason": "top_hull_invalid",
            "run": None,
            "rectangularity": None,
        }
    area_twice = abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
    )
    hull_area = 0.5 * area_twice
    if hull_area < 1e-9:
        return {
            "status": "unknown",
            "reason": "top_hull_degenerate",
            "run": None,
            "rectangularity": None,
        }
    best_area = None
    best_run = None
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        ex, ey = nxt[0] - point[0], nxt[1] - point[1]
        length = math.hypot(ex, ey)
        if length < 1e-9:
            continue
        ux, uy = ex / length, ey / length
        along = [x * ux + y * uy for x, y in points]
        across = [-x * uy + y * ux for x, y in points]
        du = max(along) - min(along)
        dv = max(across) - min(across)
        box_area = du * dv
        if best_area is None or box_area < best_area:
            best_area = box_area
            best_run = [ux, uy, 0.0] if du >= dv else [-uy, ux, 0.0]
    if best_area is None or best_area < 1e-9 or best_run is None:
        return {
            "status": "unknown",
            "reason": "top_run_fit_failed",
            "run": None,
            "rectangularity": None,
        }
    rectangularity = hull_area / best_area
    return {
        "status": "applicable" if rectangularity >= rect_min else "not_applicable",
        "reason": (
            "edge_bearing_top"
            if rectangularity >= rect_min
            else "non_edge_bearing_top"
        ),
        "run": best_run if rectangularity >= rect_min else None,
        "rectangularity": rectangularity,
    }


def _yaw_delta_deg(run_a: Optional[list], run_b: Optional[list]) -> Optional[float]:
    """Rotation about the world Z (up) axis, in degrees, that makes A's dominant horizontal edge
    LINE parallel to B's — folded modulo 90° into [-45°, +45°] (a rectangular top has an edge every
    90°, and PCA line directions carry no sign). None when either run is missing (e.g. a round
    table has no dominant edge)."""
    import math

    if not run_a or not run_b:
        return None
    ang_a = math.degrees(math.atan2(run_a[1], run_a[0]))
    ang_b = math.degrees(math.atan2(run_b[1], run_b[0]))
    return (ang_b - ang_a + 45.0) % 90.0 - 45.0


def _required_yaw_feasible_set(
    bands: list[tuple[list, float]], built_run: Optional[list]
) -> dict:
    """Intersect required yaw bands on the 90-degree line-orientation circle.

    The returned correction is the nearest point in the *joint* feasible set.  It is
    intentionally computed once for the whole constraint set: callers must suppress
    individual rotation hints when bands disagree or when one combined correction is
    available.
    """

    if not bands:
        return {"feasible": True, "interval": None, "correction_degrees": None}

    centers = []
    for run, tolerance in bands:
        if run is None:
            return {"feasible": False, "interval": None, "correction_degrees": None}
        try:
            angle = math.degrees(math.atan2(float(run[1]), float(run[0]))) % 90.0
            tol = float(tolerance)
        except (TypeError, ValueError, IndexError):
            return {"feasible": False, "interval": None, "correction_degrees": None}
        if not math.isfinite(angle) or not math.isfinite(tol) or tol < 0.0:
            return {"feasible": False, "interval": None, "correction_degrees": None}
        if tol >= 45.0:
            tol = 45.0
        centers.append((angle, tol))

    # Work on several adjacent lifts of the 90-degree circle.  Intersections only
    # shrink, so five lifts are ample for all normalized centers and tolerances.
    c0, t0 = centers[0]
    intervals = [(c0 - t0 + 90.0 * k, c0 + t0 + 90.0 * k) for k in range(-2, 3)]
    for center, tolerance in centers[1:]:
        lifted = [
            (center - tolerance + 90.0 * k, center + tolerance + 90.0 * k)
            for k in range(-3, 4)
        ]
        intersections = []
        for left_lo, left_hi in intervals:
            for right_lo, right_hi in lifted:
                lo, hi = max(left_lo, right_lo), min(left_hi, right_hi)
                if lo <= hi + 1e-9:
                    intersections.append((lo, hi))
        if not intersections:
            return {
                "feasible": False,
                "interval": None,
                "correction_degrees": None,
            }
        intersections.sort()
        merged = []
        for lo, hi in intersections:
            if merged and lo <= merged[-1][1] + 1e-9:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        intervals = merged

    if not built_run:
        return {
            "feasible": True,
            "interval": None,
            "correction_degrees": None,
        }
    try:
        built_angle = math.degrees(
            math.atan2(float(built_run[1]), float(built_run[0]))
        ) % 90.0
    except (TypeError, ValueError, IndexError):
        return {
            "feasible": True,
            "interval": None,
            "correction_degrees": None,
        }

    choices = []
    for lo, hi in intervals:
        for k in range(-3, 4):
            built_lift = built_angle + 90.0 * k
            target = min(max(built_lift, lo), hi)
            correction = target - built_lift
            choices.append((abs(correction), correction, target, lo, hi))
    _, correction, target, lo, hi = min(choices)
    target_radians = math.radians(target)
    return {
        "feasible": True,
        "interval": {
            "lower_degrees_mod_90": lo % 90.0,
            "upper_degrees_mod_90": hi % 90.0,
            "wraps_zero": (lo % 90.0) > (hi % 90.0) and hi - lo < 90.0,
            "width_degrees": max(0.0, hi - lo),
        },
        "correction_degrees": correction,
        "nearest_run": [math.cos(target_radians), math.sin(target_radians), 0.0],
    }


def pin_yaw_delta(delta: float, prev: Optional[float]) -> float:
    ""
    if prev is None or delta == 0.0 or (delta > 0) == (prev > 0):
        return delta
    if abs(delta) + abs(prev) <= 75.0:
        return delta  # a real, different correction — not a fold-boundary twin
    return delta + (90.0 if prev > 0 else -90.0)


def _point_hull_distance_xy(point: list[float], hull: Optional[list]) -> Optional[float]:
    """Distance from a 2-D point to a closed polygon, zero when the point is inside."""
    if not hull or len(hull) < 3:
        return None
    px, py = float(point[0]), float(point[1])
    inside = False
    best = float("inf")
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        if (ay > py) != (by > py):
            x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < x_cross:
                inside = not inside
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom <= 1e-18 else max(
            0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom)
        )
        qx, qy = ax + t * dx, ay + t * dy
        best = min(best, ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5)
    return 0.0 if inside else best


def _finite_hull_xy(body: dict, hull_key: str = "hull") -> Optional[list[list[float]]]:
    ""
    hull = body.get(hull_key)
    if hull and len(hull) >= 3:
        try:
            out = [[float(p[0]), float(p[1])] for p in hull]
        except (TypeError, ValueError, IndexError):
            out = []
        if len(out) >= 3 and all(math.isfinite(x) for p in out for x in p):
            return out
    try:
        lo, hi = body["lo"], body["hi"]
        coords = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(x) for x in coords):
        return None
    x0, y0, x1, y1 = coords
    if x1 < x0 or y1 < y0:
        return None
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _segment_distance_xy(a, b, c, d) -> float:
    """Minimum distance between two closed 2-D segments."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def point_segment_distance(p, q, r):
        vx, vy = r[0] - q[0], r[1] - q[1]
        denom = vx * vx + vy * vy
        if denom <= 1e-18:
            return math.hypot(p[0] - q[0], p[1] - q[1])
        t = max(0.0, min(1.0, ((p[0] - q[0]) * vx + (p[1] - q[1]) * vy) / denom))
        return math.hypot(p[0] - (q[0] + t * vx), p[1] - (q[1] + t * vy))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    eps = 1e-10
    if (
        (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps)
        and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps)
    ):
        return 0.0
    # This also handles collinear overlap and endpoint contact.
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def _polygon_distance_xy(a: Optional[list], b: Optional[list]) -> Optional[float]:
    """Finite distance between two closed XY polygons (zero for overlap/contact)."""
    if not a or len(a) < 3 or not b or len(b) < 3:
        return None
    if any(_point_hull_distance_xy(p, b) == 0.0 for p in a):
        return 0.0
    if any(_point_hull_distance_xy(p, a) == 0.0 for p in b):
        return 0.0
    return min(
        _segment_distance_xy(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)])
        for i in range(len(a))
        for j in range(len(b))
    )


def _projection_interval_xy(
    hull: Optional[list], direction: Optional[list]
) -> Optional[list[float]]:
    """Finite scalar interval of an XY polygon along ``direction``."""
    u = _unit(direction)
    if not hull or len(hull) < 3 or not u:
        return None
    vals = [float(p[0]) * u[0] + float(p[1]) * u[1] for p in hull]
    return [min(vals), max(vals)] if vals and all(math.isfinite(x) for x in vals) else None


def _interval_overlap(a: Optional[list], b: Optional[list]) -> Optional[float]:
    """Signed interval overlap: positive overlap, zero touch, negative separation."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    return min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0]))


def _measured_corner_xy(a: Optional[dict], b: Optional[dict]) -> Optional[list[float]]:
    """Intersection of two reliable gravity-plumbed scene-graph wall planes in XY."""
    from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane

    if not a or not b or not plane_is_reliable(a) or not plane_is_reliable(b):
        return None
    wa, wb = a.get("world_center"), b.get("world_center")
    na, ka = plumb_plane(a.get("normal"))
    nb, kb = plumb_plane(b.get("normal"))
    if not wa or not wb or not na or not nb or ka != "wall" or kb != "wall":
        return None
    det = na[0] * nb[1] - na[1] * nb[0]
    if abs(det) < 1e-6:
        return None
    da = na[0] * wa[0] + na[1] * wa[1]
    db = nb[0] * wb[0] + nb[1] * wb[1]
    return [
        (da * nb[1] - na[1] * db) / det,
        (na[0] * db - da * nb[0]) / det,
    ]


_WALL_TILT_CATS = ("wall",)
_LEVEL_TILT_CATS = ("floor", "ground", "ceiling")


def _is_slab(body: dict, ratio: float) -> bool:
    """Is this body thin enough for its PCA normal to mean anything? A slab's smallest
    AABB extent is a small fraction of its largest; on a cube-ish body the smallest-
    variance axis is arbitrary, so the tilt check must not fire on one."""
    try:
        ext = sorted(float(body["hi"][d]) - float(body["lo"][d]) for d in range(3))
    except Exception:  # noqa: BLE001 - a malformed body is simply not gated
        return False
    return ext[2] > 1e-9 and ext[0] <= ratio * ext[2]


def surface_geometry_report(
    sg_surfaces: dict,
    bodies: list[dict],
    surface_determinants: Optional[dict] = None,
    surface_hulls: Optional[dict] = None,
    surface_normals: Optional[dict] = None,
    surface_centroids: Optional[dict] = None,
    surface_plane_bounds: Optional[dict] = None,
    determinant_tol: float = 1e-9,
    anchor_tol: float = 0.15,
    vertical_tol: float = 0.10,
    slab_ratio: float = 0.4,
    canonical_plane_offset_tol: float = 0.05,
    canonical_plane_angle_tol_deg: float = 5.0,
) -> tuple[bool, str]:
    ""
    import math

    from lib.tools.geometry.scene_graph import (
        PLUMB_SUPPORT_MIN_NZ,
        PLUMB_WALL_MAX_NZ,
        plane_is_reliable,
        plumb_plane,
    )
    from lib.tools.geometry.surface_relations import surface_build_name

    determinants = surface_determinants or {}
    hulls = surface_hulls or {}
    built_normals = surface_normals or {}
    centroids = surface_centroids or {}
    plane_bounds = surface_plane_bounds or {}
    built = {b.get("name"): b for b in bodies if not b.get("is_obj")}
    failures: list[str] = []
    checked = 0
    for sid, sg in sg_surfaces.items():
        name = surface_build_name(sid)
        body = built.get(name)
        if body is None:
            continue  # presence/visibility is owned by the coverage rule
        checked += 1
        cat = str(sg.get("category") or "").strip().lower()
        bn = built_normals.get(name)
        if bn is not None and _is_slab(body, slab_ratio):
            nz = min(1.0, abs(float(bn[2])))
            if cat in _WALL_TILT_CATS and nz > PLUMB_WALL_MAX_NZ:
                failures.append(
                    f"'{name}' is a WALL but leans {math.degrees(math.asin(nz)):.0f}deg "
                    "off vertical; rebuild it PLUMB (vertical edges along +Z). Only its "
                    "tilt vs gravity is constrained — its yaw and extent are not"
                )
            elif cat in _LEVEL_TILT_CATS and nz < PLUMB_SUPPORT_MIN_NZ:
                failures.append(
                    f"'{name}' is a {cat.upper()} but is "
                    f"{math.degrees(math.acos(nz)):.0f}deg off level; rebuild it FLAT "
                    "(its face normal along +Z)"
                )
        det = determinants.get(name)
        if det is None:
            failures.append(f"'{name}' transform determinant is unavailable")
        elif float(det) <= determinant_tol:
            failures.append(
                f"'{name}' has reflected/left-handed transform determinant {float(det):+.6g}; "
                "rebuild it with a right-handed basis and positive dimensions without moving "
                "its finite footprint"
            )

        normal, kind = plumb_plane(sg.get("normal"))
        anchor = sg.get("world_center")
        if (
            cat != "wall"
            or not anchor
            or not normal
            or kind != "wall"
            or not plane_is_reliable(sg)
        ):
            continue
        # A reliable measured wall is canonical for every later consumer.  Merely
        # containing one measured point lets a built slab rotate about that point, so
        # POSE (measured run) and CONTACT (built run) can issue opposite corrections.
        # Verify broad-face orientation and offset here, before either rule executes.
        if bn is None:
            failures.append(
                f"'{name}' canonical measured wall plane cannot be verified because "
                "its built broad-face normal is unavailable"
            )
        else:
            angle = _acute_angle_deg(bn, normal)
            center = centroids.get(name) or [
                (float(body["lo"][i]) + float(body["hi"][i])) / 2.0
                for i in range(3)
            ]
            offset = abs(
                sum(
                    (float(center[i]) - float(anchor[i])) * float(normal[i])
                    for i in range(3)
                )
            )
            if angle is None:
                failures.append(
                    f"'{name}' canonical measured wall orientation is unavailable"
                )
            elif angle > canonical_plane_angle_tol_deg:
                failures.append(
                    f"'{name}' built broad face is {angle:.1f}deg off its canonical "
                    f"measured wall plane (tolerance {canonical_plane_angle_tol_deg:.1f}deg); "
                    "rotate/rebuild the WALL to the measured plane — do not rotate the main "
                    "support to chase the incorrect built wall"
                )
            if offset > canonical_plane_offset_tol + 1e-6:
                failures.append(
                    f"'{name}' built broad plane is {offset:.3f}m off its canonical "
                    f"measured wall plane (tolerance {canonical_plane_offset_tol:.3f}m); "
                    "translate the WALL along the measured normal onto that plane"
                )
        distance = _point_hull_distance_xy(anchor, hulls.get(name))
        if distance is None:
            failures.append(f"'{name}' finite wall footprint is unavailable")
            continue
        finite_body = plane_bounds.get(name) or body
        z_outside = max(
            float(finite_body["lo"][2]) - float(anchor[2]),
            float(anchor[2]) - float(finite_body["hi"][2]),
            0.0,
        )
        if distance > anchor_tol or z_outside > vertical_tol:
            failures.append(
                f"'{name}' does not contain its measured wall point inside the finite face "
                f"(XY outside by {distance:.3f}m, Z outside by {z_outside:.3f}m); keep the "
                "wall on the half-line containing that point"
            )
    if failures:
        return False, "root surface geometry: FAIL — " + "; ".join(failures)
    return True, f"root surface geometry: PASS ({checked} built root surface(s))"


def relationship_report(
    relationships: list[dict],
    sg_surfaces: dict,
    bodies: list[dict],
    surface_normals: dict,
    against_gap: float = 0.01,
    under_gap: float = 0.05,
    against_pen: float = 0.01,
    against_yaw_deg: float = 3.0,
    rect_min: float = 0.9,
    corner_angle_deg: float = 85.0,
    perp_angle_deg: float = 88.0,
    main_name: Optional[str] = None,
    surface_runs: Optional[dict] = None,
    surface_hulls: Optional[dict] = None,
    surface_top_hulls: Optional[dict] = None,
    surface_plane_hulls: Optional[dict] = None,
    surface_plane_bounds: Optional[dict] = None,
    surface_plane_centroids: Optional[dict] = None,
    surface_plane_normals: Optional[dict] = None,
    edge_margin: float = 0.03,
    finite_contact_gap: float = 0.05,
    corner_reach_gap: float = 0.15,
    check_against_yaw: bool = True,
    evidence_out: Optional[list] = None,
) -> tuple[bool, str]:
    ""
    try:
        from lib.tools.geometry.surface_relations import surface_build_name
    except Exception:  # noqa: BLE001 - keep the rule importable outside the package

        def surface_build_name(sid):
            return (
                re.sub(r"[^a-zA-Z0-9_-]+", "_", sid.replace("#", "_"))
                .strip("_")
                .lower()[:48]
                or "surface"
            )

    built = {
        b["name"]: {
            "lo": b["lo"],
            "hi": b["hi"],
            "n": surface_normals.get(b["name"]),
            "plane_n": (
                (surface_plane_normals or {}).get(b["name"])
                or surface_normals.get(b["name"])
            ),
            "plane_c": (surface_plane_centroids or {}).get(b["name"]),
            "run": (surface_runs or {}).get(b["name"]),
            "hull": (surface_hulls or {}).get(b["name"]),
            "plane_hull": (surface_plane_hulls or {}).get(b["name"]),
            "plane_bounds": (surface_plane_bounds or {}).get(b["name"]),
            "run_hull": (
                (surface_top_hulls or {}).get(b["name"])
                if b["name"] == main_name
                else (surface_hulls or {}).get(b["name"])
            ),
            "c": [(b["lo"][i] + b["hi"][i]) / 2.0 for i in range(3)],
        }
        for b in bodies
        if not b.get("is_obj")
    }
    objs = [b for b in bodies if b.get("is_obj")]
    if (
        main_name in built
    ):  # the main support is LEVEL by rule -> its plane normal IS +Z
        built[main_name]["n"] = [0.0, 0.0, 1.0]
        built[main_name]["plane_n"] = [0.0, 0.0, 1.0]
    by_name = {name.lower(): name for name in built}

    def _canonical_wall(sid):
        """Canonical measured wall ``(point, normal, run)`` when reliable."""
        from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane

        entry = sg_surfaces.get(sid) or {}
        normal, kind = plumb_plane(entry.get("normal"))
        point = entry.get("world_center")
        if (
            not point
            or not normal
            or kind != "wall"
            or not plane_is_reliable(entry)
        ):
            return None
        h = (float(normal[0]) ** 2 + float(normal[1]) ** 2) ** 0.5
        if h < 1e-6:
            return None
        return (
            [float(x) for x in point],
            [float(x) for x in normal],
            [-float(normal[1]) / h, float(normal[0]) / h, 0.0],
        )

    def _match(sid):  # scene-graph id -> built body
        hit = by_name.get(
            surface_build_name(sid)
        )  # 1) the agent was told to name it this
        if hit:
            return hit
        s = (
            sg_surfaces.get(sid) or {}
        )  # 2) fallback: nearest built body on the sg plane (plumbed — the plane the
        # agent was told to build on; an ambiguous tilt matches nothing)
        from lib.tools.geometry.scene_graph import plumb_plane

        wc = s.get("world_center")
        n, kind = plumb_plane(s.get("normal"))
        if not wc or not n or kind == "ambiguous":
            return None
        best, bestd = None, 0.30
        for name, bb in built.items():
            c = bb["c"]
            d = abs(
                (c[0] - wc[0]) * n[0] + (c[1] - wc[1]) * n[1] + (c[2] - wc[2]) * n[2]
            )
            ang = _acute_angle_deg(bb.get("plane_n"), n)
            if d < bestd and (ang is None or ang <= 25.0):
                best, bestd = name, d
        return best

    def _contact_hull(body, *, root_plane=False):
        """Exact grouped/root hull when present; otherwise a finite bounds hull."""
        if root_plane:
            root_hull = body.get("plane_hull")
            if root_hull is not None:
                hull = _finite_hull_xy({"hull": root_hull})
                if hull:
                    return hull, "built_root_surface_hull"
            bounds = body.get("plane_bounds")
            if isinstance(bounds, dict):
                bounded = _finite_hull_xy(bounds)
                if bounded:
                    return bounded, "built_root_surface_bounds"
        hull = _finite_hull_xy(body)
        return hull, "built_group_hull" if body.get("hull") else "built_body_bounds"

    def _z_bounds(body, *, root_plane=False):
        source = body.get("plane_bounds") if root_plane else None
        source = source if isinstance(source, dict) else body
        try:
            lo, hi = float(source["lo"][2]), float(source["hi"][2])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            return None
        return [lo, hi]

    def _wall_run_from_normal(normal):
        n = _unit(normal)
        if not n:
            return None
        h = math.hypot(n[0], n[1])
        return [-n[1] / h, n[0] / h, 0.0] if h > 1e-8 else None

    def _is_wall_endpoint(sid, body):
        ""
        entry = sg_surfaces.get(sid) or {}
        category = str(entry.get("category") or "").strip().lower()
        if "wall" in re.split(r"[^a-z0-9]+", category):
            return True
        try:
            from lib.tools.geometry.scene_graph import plumb_plane

            _normal, kind = plumb_plane(entry.get("normal"))
            if kind == "wall":
                return True
        except Exception:  # noqa: BLE001 - retain legacy standalone behavior
            pass
        built_name = surface_build_name(sid) if sid else ""
        body_name = str((body or {}).get("name") or "")
        return built_name.startswith("wall_") or body_name.startswith("wall_")

    main_id = next(
        (sid for sid in sg_surfaces if surface_build_name(sid) == main_name), None
    )
    hard_against: list[tuple[str, list]] = []
    if check_against_yaw and main_id and main_name in built:
        main_rect = _rect_ratio(
            built[main_name].get("run_hull"), built[main_name].get("run")
        )
        if main_rect is not None and main_rect >= rect_min:
            for rel_index, rel in enumerate(relationships):
                if (
                    rel.get("type") != "against"
                    or _relationship_enforcement(rel) != "hard"
                    or main_id not in (rel.get("a"), rel.get("b"))
                ):
                    continue
                wall_id = rel.get("b") if rel.get("a") == main_id else rel.get("a")
                canonical = _canonical_wall(wall_id)
                if canonical:
                    rid = rel.get("relationship_id") or (
                        f"legacy:{rel_index}:against:{rel.get('a')}:{rel.get('b')}"
                    )
                    hard_against.append((rid, canonical[2]))
    conflicting_against_ids: set[str] = set()
    for i, (rid_a, run_a) in enumerate(hard_against):
        for rid_b, run_b in hard_against[i + 1 :]:
            delta = _yaw_delta_deg(run_a, run_b)
            if delta is not None and abs(delta) > 2.0 * against_yaw_deg:
                conflicting_against_ids.update((rid_a, rid_b))

    lines, ok = [], True
    for rel_index, r in enumerate(relationships):
        t, aid, bid = r.get("type"), r.get("a"), r.get("b")
        enforcement = _relationship_enforcement(r)
        preprocess_status = str(r.get("status") or "legacy")
        relationship_id = r.get("relationship_id") or (
            f"legacy:{rel_index}:{t}:{aid}:{bid}"
        )
        evidence = {
            "relationship_id": relationship_id,
            "type": t,
            "a": aid,
            "b": bid,
            "enforcement": enforcement,
            "preprocess_status": preprocess_status,
            "runtime_status": "not_checked",
            "measurements": {"preprocess": dict(r.get("measurements") or {})},
            "reasons": [str(x) for x in (r.get("adjudication_reasons") or [])],
        }
        if evidence_out is not None:
            evidence_out.append(evidence)
        if enforcement != "hard":
            evidence["runtime_status"] = (
                "ignored" if enforcement == "none" else "advisory"
            )
            if not evidence["reasons"]:
                evidence["reasons"].append(
                    f"preprocessing marked this relationship {enforcement}"
                )
            lines.append(
                f"{t} {aid}<->{bid}: {evidence['runtime_status'].upper()} "
                f"(preprocessing status {preprocess_status}; enforcement {enforcement} — "
                "not used as a hard CONTACT constraint)"
            )
            continue
        a, b = _match(aid), _match(bid)
        if not a or not b:
            ok = False
            evidence["runtime_status"] = "unverified"
            evidence["reasons"].append("surface not located in the build")
            lines.append(
                f"{t} {aid}<->{bid}: FAIL — hard relationship UNVERIFIED "
                "(surface not located in the build)"
            )
            continue
        if a == b:
            ok = False
            evidence["runtime_status"] = "unverified"
            evidence["reasons"].append("both ids resolved to the same built body")
            lines.append(
                f"{t} {aid}<->{bid}: FAIL — hard relationship UNVERIFIED "
                f"(both ids resolved to the same "
                f"built body '{a}')"
            )
            continue
        A, B = built[a], built[b]
        if t == "against":
            nA_eff, nB_eff = _unit(A.get("plane_n")), _unit(B.get("plane_n"))
            if (
                not r.get("_compiled_constraint")
                and nA_eff
                and nB_eff
                and abs(nA_eff[2]) < 0.5 < abs(nB_eff[2])
            ):
                a, b, aid, bid, A, B = b, a, bid, aid, B, A
            binding = str(
                ((r.get("_constraint_reference") or {}).get("plane_binding") or "")
            )
            canonical = None if binding == "current_built_root" else _canonical_wall(bid)
            if binding == "canonical_scene_graph" and canonical is None:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append(
                    "compiled canonical wall reference is unavailable"
                )
                lines.append(
                    f"against '{a}'<->'{b}': FAIL — required canonical wall "
                    "reference is unavailable"
                )
                continue
            cB = canonical[0] if canonical else (B.get("plane_c") or B["c"])
            nB = canonical[1] if canonical else B.get("plane_n")
            root_plane_run = _wall_run_from_normal(nB)
            runB = canonical[2] if canonical else (root_plane_run or B.get("run"))
            evidence["measurements"]["reference_plane_source"] = (
                "canonical_measured_wall" if canonical else "built_wall"
            )
            evidence["measurements"]["reference_run_source"] = (
                "canonical_measured_wall"
                if canonical
                else "built_root_surface_plane"
                if root_plane_run is not None
                else "built_group_run"
            )
            if relationship_id in conflicting_against_ids:
                ok = False
                evidence["runtime_status"] = "inconsistent"
                evidence["reasons"].append(
                    "hard AGAINST wall runs have no jointly satisfiable edge family"
                )
                lines.append(
                    f"against '{a}'<->'{b}': FAIL — CONSTRAINTS INCONSISTENT (this "
                    "canonical wall run conflicts with another hard AGAINST wall run; "
                    "no rectangular support yaw can satisfy both within tolerance. Do not "
                    "alternate rotations or move an anchored wall; correct/demote the "
                    "upstream relationship set.)"
                )
                continue
            if not nB:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append("reference plane unavailable")
                lines.append(
                    f"against '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED "
                    "(no reference plane)"
                )
                continue
            # AGAINST is contact with a FINITE wall, not merely its infinite plane.
            # Use the exact root slab rather than grouped detail children (a frame or
            # backdrop must not make a short wall appear to reach the support).  The
            # canonical measured plane/run remains authoritative for orientation;
            # these built-root intervals certify that the actual slab reaches the
            # asserted contact in both run and height.
            hullA, hullA_source = _contact_hull(A)
            hullB, hullB_source = _contact_hull(B, root_plane=True)
            zA, zB = _z_bounds(A), _z_bounds(B, root_plane=True)
            intervalA = _projection_interval_xy(hullA, runB)
            intervalB = _projection_interval_xy(hullB, runB)
            run_overlap = _interval_overlap(intervalA, intervalB)
            vertical_overlap = _interval_overlap(zA, zB)
            run_gap = None if run_overlap is None else max(-run_overlap, 0.0)
            vertical_gap = (
                None if vertical_overlap is None else max(-vertical_overlap, 0.0)
            )
            evidence["measurements"].update(
                finite_support_hull_source=hullA_source,
                finite_wall_hull_source=hullB_source,
                wall_run_axis_xy=(runB[:2] if runB else None),
                support_wall_run_interval_m=intervalA,
                wall_run_interval_m=intervalB,
                wall_run_gap_m=run_gap,
                wall_run_overlap_m=(
                    None if run_overlap is None else max(run_overlap, 0.0)
                ),
                support_vertical_interval_m=zA,
                wall_vertical_interval_m=zB,
                vertical_gap_m=vertical_gap,
                vertical_overlap_m=(
                    None
                    if vertical_overlap is None
                    else max(vertical_overlap, 0.0)
                ),
                finite_contact_tolerance_m=finite_contact_gap,
            )
            if run_overlap is None or vertical_overlap is None:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append(
                    "finite wall run or vertical span is unavailable"
                )
                lines.append(
                    f"against '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED "
                    "(finite wall run/vertical-span evidence unavailable; infinite-plane "
                    "contact is insufficient)"
                )
                continue
            if run_gap > finite_contact_gap or vertical_gap > finite_contact_gap:
                ok = False
                evidence["runtime_status"] = "fail"
                reasons = []
                fixes = []
                if run_gap > finite_contact_gap:
                    reasons.append(
                        f"finite wall run misses support by {-run_overlap:.3f}m"
                    )
                    fixes.append(
                        (
                            f"extend '{b}' along its run by at least {-run_overlap:.3f}m "
                            "while retaining its measured point"
                            if canonical
                            else f"extend/reposition '{b}' along its run by at least "
                            f"{-run_overlap:.3f}m"
                        )
                    )
                if vertical_gap > finite_contact_gap:
                    reasons.append(
                        f"vertical spans miss by {-vertical_overlap:.3f}m"
                    )
                    fixes.append(
                        f"extend '{b}' vertically by at least {-vertical_overlap:.3f}m"
                    )
                evidence["reasons"].extend(reasons)
                fixed = "; ".join(fixes)
                plane_note = (
                    " while keeping its canonical measured plane fixed"
                    if canonical
                    else ""
                )
                lines.append(
                    f"against '{a}'<->'{b}': FAIL (infinite planes approach each other, "
                    f"but finite contact does not exist: {', '.join(reasons)}; {fixed}"
                    f"{plane_note}"
                    + (
                        f"; do NOT translate '{b}' — if its photo-matched extent is "
                        f"already correct, rebuild/recenter '{a}' instead"
                        if canonical
                        else ""
                    )
                    + ")"
                )
                continue
            delta = _yaw_delta_deg(A.get("run"), runB)
            rect = _rect_ratio(A.get("run_hull"), A.get("run"))
            evidence["measurements"].update(
                yaw_delta_degrees_mod_90=delta,
                rectangularity=rect,
                yaw_reference_run=runB,
            )
            if (
                check_against_yaw
                and
                delta is not None
                and abs(delta) > against_yaw_deg
                and rect is not None
                and rect >= rect_min
            ):
                ok = False
                evidence["runtime_status"] = "fail"
                evidence["reasons"].append("edge is not parallel to canonical wall run")
                lines.append(
                    f"against '{a}'<->'{b}': FAIL (edge not parallel — '{a}'s edge runs "
                    f"{delta:+.0f}° off '{b}'s "
                    f"{'canonical measured' if canonical else 'built'} run; rotate '{a}' by "
                    f"{delta:+.0f}° around the world Z "
                    f"(up) axis about its own centre C=({A['c'][0]:+.2f}, {A['c'][1]:+.2f}, 0) — "
                    f"obj.matrix_world = Matrix.Translation(C) @ Matrix.Rotation(math.radians("
                    f"{delta:.0f}), 4, 'Z') @ Matrix.Translation(-C) @ obj.matrix_world. Objects "
                    f"resting on it do NOT rotate with it — momentary overhang is EXPECTED, do "
                    f"NOT undo; re-run the check after rotating (flush distance is judged then))"
                )
                continue
            # Signed nearest-corner distance to B's plane (A's side positive): >0 = A floats off
            # B; <0 = A crosses through B.
            signed = _signed_gap(A, cB, nB)
            evidence["measurements"].update(
                signed_gap_m=signed,
                near_edge_gap_m=signed,
            )
            float_ok = signed <= against_gap  # not floating away from B
            pen_ok = signed >= -against_pen  # not crossing through B
            if float_ok and pen_ok:
                evidence["runtime_status"] = "pass"
                lines.append(
                    f"against '{a}'<->'{b}': PASS (flush {signed * 100:+.1f}cm to "
                    f"{'canonical measured plane' if canonical else 'built plane'})"
                )
            else:
                ok = False
                evidence["runtime_status"] = "fail"
                out = _out_dir(A["c"], cB, nB)
                pen = not pen_ok
                state = (
                    f"gap {signed * 100:.1f}cm > {against_gap * 100:.0f}cm"
                    if not float_ok
                    else f"'{a}' PENETRATES it by {-signed * 100:.1f}cm"
                )
                dist = abs(signed)
                move_a, pinned = None, False  # (unit dir, slack) when A should move
                if out is not None and canonical is not None:
                    if pen:  # retreat A's near edge OUT of the wall, away from it
                        m = _min_obj_margin_to_plane(objs, A["c"], cB, nB)
                        if m is None or m >= edge_margin:
                            move_a = (out, m)
                        else:
                            pinned = True
                    else:  # close the gap: slide A toward the wall
                        fm = _far_edge_margin(objs, A, out)
                        if fm is None or fm >= dist + edge_margin:
                            move_a = ([-out[0], -out[1], 0.0], None)
                        else:
                            pinned = True
                if move_a is not None:
                    da, slack = move_a
                    note = (
                        f" (the deepest object keeps {slack:.2f}m of '{a}' under it)"
                        if slack is not None
                        else ""
                    )
                    lines.append(
                        f"against '{a}'<->'{b}': FAIL ({state} — translate '{a}' by "
                        f"~{dist:.3f}m along world ({da[0]:+.2f}, {da[1]:+.2f}, 0) "
                        f"(obj.location.x += {dist * da[0]:+.3f}, obj.location.y += "
                        f"{dist * da[1]:+.3f}) so its edge sits flush in '{b}'s plane — its "
                        f"top stays at z=0 and the objects do NOT move with it (they are "
                        f"FINAL and stay covered{note}); do NOT move '{b}' — its plane was "
                        f"MEASURED from the image)"
                    )
                    continue
                if canonical is not None:
                    # A reliable wall is no longer a repair target. If translating A
                    # would violate object support, the hard constraints are jointly
                    # inconsistent; moving the wall would only make STRUCTURE move it
                    # back on the next ladder pass.
                    evidence["runtime_status"] = "inconsistent"
                    evidence["reasons"].append(
                        "canonical wall is immutable and the support is pinned by objects"
                    )
                    lines.append(
                        f"against '{a}'<->'{b}': FAIL — CONSTRAINTS INCONSISTENT ({state}; "
                        f"'{b}' is fixed to its canonical measured plane, while the objects "
                        f"leave insufficient safe translation room for '{a}'. Do NOT move "
                        f"'{b}': that would fail STRUCTURE on the next check. Rebuild/resize "
                        f"'{a}' so its edge reaches the canonical plane while all objects "
                        "remain supported, or correct the upstream relationship if the photo "
                        "does not show this contact.)"
                    )
                    continue
                if out is None:
                    fix = f"translate '{b}' along its own normal to meet '{a}'s edge"
                else:
                    d = [signed * out[0], signed * out[1]]
                    fix = (
                        f"translate '{b}' by ~{abs(signed):.3f}m along world "
                        f"({d[0] / abs(signed):+.2f}, {d[1] / abs(signed):+.2f}, 0) "
                        f"(obj.location.x += {d[0]:+.3f}, obj.location.y += {d[1]:+.3f})"
                    )
                why_a_fixed = (
                    f"the settled objects pin '{a}' (no room to translate it), so "
                    f"'{b}' must yield despite its measured plane"
                    if pinned
                    else f"do NOT move '{a}' — its pose is set by the photo (POSE rule)"
                )
                lines.append(
                    f"against '{a}'<->'{b}': FAIL ({state} — {fix} so its plane sits flush "
                    f"on '{a}'s edge; {why_a_fixed})"
                )
        elif t == "under":
            # A hard UNDER involving a wall means contact with the exact root slab,
            # not with an allowed detail child (frame/backdrop) grouped under it.
            # Furniture/support endpoints intentionally retain the complete grouped
            # body so floor-under-table still tests the table base's real underside.
            a_root_plane = _is_wall_endpoint(aid, A)
            b_root_plane = _is_wall_endpoint(bid, B)
            zA = _z_bounds(A, root_plane=a_root_plane)
            zB = _z_bounds(B, root_plane=b_root_plane)
            dz = None if zA is None or zB is None else zB[0] - zA[1]
            hullA, hullA_source = _contact_hull(A, root_plane=a_root_plane)
            hullB, hullB_source = _contact_hull(B, root_plane=b_root_plane)
            axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            a_intervals = [_projection_interval_xy(hullA, axis) for axis in axes]
            b_intervals = [_projection_interval_xy(hullB, axis) for axis in axes]
            interval_overlaps = [
                _interval_overlap(ia, ib)
                for ia, ib in zip(a_intervals, b_intervals)
            ]
            xy_distance = _polygon_distance_xy(hullA, hullB)
            finite_known = (
                xy_distance is not None
                and all(x is not None for x in interval_overlaps)
                and dz is not None
                and math.isfinite(float(dz))
            )
            xy_axis_gaps = (
                [max(-float(x), 0.0) for x in interval_overlaps]
                if finite_known
                else None
            )
            xy_overlap_extents = (
                [max(float(x), 0.0) for x in interval_overlaps]
                if finite_known
                else None
            )
            interval_overlap = bool(
                finite_known and all(float(x) >= 0.0 for x in interval_overlaps)
            )
            overlap = bool(
                finite_known and interval_overlap and float(xy_distance) <= 1e-6
            )
            xy_gap = max(xy_axis_gaps) if xy_axis_gaps is not None else None
            evidence["measurements"].update(
                gap_m=dz,
                vertical_contact_gap_m=(None if dz is None else abs(dz)),
                lower_plane_height_m=(None if zA is None else float(zA[1])),
                upper_vertical_interval_m=zB,
                a_vertical_bounds_source=(
                    "built_root_surface_bounds"
                    if a_root_plane and isinstance(A.get("plane_bounds"), dict)
                    else "built_group_bounds"
                ),
                b_vertical_bounds_source=(
                    "built_root_surface_bounds"
                    if b_root_plane and isinstance(B.get("plane_bounds"), dict)
                    else "built_group_bounds"
                ),
                a_xy_hull_source=hullA_source,
                b_xy_hull_source=hullB_source,
                a_xy_intervals_m=a_intervals,
                b_xy_intervals_m=b_intervals,
                xy_axis_gaps_m=xy_axis_gaps,
                xy_overlap_extents_m=xy_overlap_extents,
                xy_gap_m=xy_gap,
                xy_polygon_distance_m=xy_distance,
                xy_intervals_overlap=interval_overlap,
                xy_overlap=overlap,
            )
            if not finite_known:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append("finite XY/contact evidence unavailable")
                lines.append(
                    f"under '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED "
                    "(finite XY/contact evidence unavailable)"
                )
            elif abs(dz) <= under_gap and overlap:
                evidence["runtime_status"] = "pass"
                lines.append(
                    f"under '{a}'<->'{b}': PASS (vertical contact gap "
                    f"{dz * 100:+.1f}cm; finite XY footprints overlap)"
                )
            else:
                ok = False
                evidence["runtime_status"] = "fail"
                evidence["reasons"].append(
                    "vertical contact gap" if abs(dz) > under_gap else "no xy overlap"
                )
                why = (
                    f"gap {dz * 100:+.1f}cm — rest '{b}' on '{a}'"
                    if abs(dz) > under_gap
                    else "finite XY footprints do not overlap — center/extend them"
                )
                lines.append(f"under '{a}'<->'{b}': FAIL ({why})")
        elif t == "corner":
            # Two walls meeting at a room corner must be ~perpendicular and, when both measured
            # planes are reliable, each FINITE footprint must reach their plane-intersection
            # corner. Extending PAST that point is still tolerated; this only rejects disconnected
            # finite slabs that happen to lie on the correct infinite planes.
            ang = _acute_angle_deg(A.get("plane_n"), B.get("plane_n"))
            evidence["measurements"]["acute_normal_angle_degrees"] = ang
            evidence["measurements"]["normal_angle_deg"] = ang
            if ang is None:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append("surface plane unavailable")
                lines.append(
                    f"corner '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED (no plane)"
                )
                continue
            if ang < corner_angle_deg:
                ok = False
                evidence["runtime_status"] = "fail"
                evidence["reasons"].append("walls are not perpendicular")
                lines.append(
                    f"corner '{a}'<->'{b}': FAIL (walls at {ang:.0f}° — should meet at "
                    f"~90°; make '{a}' and '{b}' perpendicular)"
                )
            else:
                corner = r.get("_corner_xy_m")
                if corner is None:
                    corner = _measured_corner_xy(
                        sg_surfaces.get(aid), sg_surfaces.get(bid)
                    )
                root_hull_a, source_a = _contact_hull(A, root_plane=True)
                root_hull_b, source_b = _contact_hull(B, root_plane=True)
                da = _point_hull_distance_xy(corner, root_hull_a) if corner else None
                db = _point_hull_distance_xy(corner, root_hull_b) if corner else None
                z_a = _z_bounds(A, root_plane=True)
                z_b = _z_bounds(B, root_plane=True)
                z_overlap = _interval_overlap(z_a, z_b)
                z_gap = None if z_overlap is None else max(-z_overlap, 0.0)
                xy_gap = _polygon_distance_xy(root_hull_a, root_hull_b)
                evidence["measurements"].update(
                    measured_corner_xy=corner,
                    a_corner_distance_m=da,
                    b_corner_distance_m=db,
                    a_corner_hull_source=source_a,
                    b_corner_hull_source=source_b,
                    a_vertical_interval_m=z_a,
                    b_vertical_interval_m=z_b,
                    vertical_gap_m=z_gap,
                    vertical_overlap_m=(
                        None if z_overlap is None else max(z_overlap, 0.0)
                    ),
                    finite_xy_distance_m=xy_gap,
                    finite_contact_tolerance_m=finite_contact_gap,
                )
                if corner is None:
                    if xy_gap is None or z_gap is None:
                        ok = False
                        evidence["runtime_status"] = "unverified"
                        evidence["reasons"].append(
                            "finite root-wall proximity is unavailable"
                        )
                        lines.append(
                            f"corner '{a}'<->'{b}': FAIL — hard relationship "
                            "UNVERIFIED (no reliable measured corner and finite root-wall "
                            "proximity is unavailable)"
                        )
                    elif xy_gap > finite_contact_gap or z_gap > finite_contact_gap:
                        ok = False
                        evidence["runtime_status"] = "fail"
                        evidence["reasons"].append(
                            "finite root walls do not meet"
                        )
                        lines.append(
                            f"corner '{a}'<->'{b}': FAIL (walls are ~perpendicular "
                            f"{ang:.0f}° but no reliable measured corner is available "
                            f"and their finite root slabs miss by {xy_gap:.2f}m in XY / "
                            f"{z_gap:.2f}m vertically; make the root wall slabs meet)"
                        )
                    else:
                        evidence["runtime_status"] = "pass"
                        lines.append(
                            f"corner '{a}'<->'{b}': PASS (~perpendicular {ang:.0f}°; "
                            "finite root wall slabs meet; extending past the corner is fine)"
                        )
                elif da is None or db is None or z_gap is None or xy_gap is None:
                    ok = False
                    evidence["runtime_status"] = "unverified"
                    evidence["reasons"].append(
                        "finite root-wall reach or vertical extent is unavailable"
                    )
                    lines.append(
                        f"corner '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED "
                        "(measured corner exists but finite root-wall reach/vertical "
                        "extent is unavailable)"
                    )
                elif z_overlap is None or z_overlap <= 0.0:
                    ok = False
                    evidence["runtime_status"] = "fail"
                    evidence["reasons"].append(
                        "finite root walls do not positively overlap vertically"
                    )
                    lines.append(
                        f"corner '{a}'<->'{b}': FAIL (walls reach the measured XY "
                        f"corner ({corner[0]:+.2f}, {corner[1]:+.2f}) but their root "
                        "slabs do not positively overlap vertically; extend/reposition "
                        "them so they share a finite vertical corner edge)"
                    )
                elif xy_gap > finite_contact_gap:
                    ok = False
                    evidence["runtime_status"] = "fail"
                    evidence["reasons"].append("finite root walls do not actually meet")
                    lines.append(
                        f"corner '{a}'<->'{b}': FAIL (both walls approach the canonical "
                        f"corner, but their finite root slabs remain {xy_gap:.3f}m apart; "
                        f"make their actual XY gap <= {finite_contact_gap:.3f}m)"
                    )
                elif max(da, db) > corner_reach_gap:
                    ok = False
                    evidence["runtime_status"] = "fail"
                    evidence["reasons"].append("finite wall misses measured corner")
                    lines.append(
                        f"corner '{a}'<->'{b}': FAIL (walls are ~perpendicular {ang:.0f}° but "
                        f"their finite footprints miss measured corner ({corner[0]:+.2f}, "
                        f"{corner[1]:+.2f}) by {da:.2f}m / {db:.2f}m; extend them to that "
                        "corner without changing their measured planes)"
                    )
                else:
                    evidence["runtime_status"] = "pass"
                    lines.append(
                        f"corner '{a}'<->'{b}': PASS (~perpendicular {ang:.0f}°; extending past "
                        f"the corner is fine — it hides behind the other wall)"
                    )
        elif t == "perpendicular":
            # Preprocessing grants a hard PERPENDICULAR only when the measured finite
            # surfaces meet.  Preserve that semantic at runtime: an angle match between
            # remote surfaces must not pass merely because their infinite planes would
            # intersect somewhere.
            ang = _acute_angle_deg(A.get("plane_n"), B.get("plane_n"))
            evidence["measurements"]["acute_normal_angle_degrees"] = ang
            evidence["measurements"]["normal_angle_deg"] = ang
            if ang is None:
                ok = False
                evidence["runtime_status"] = "unverified"
                evidence["reasons"].append("surface plane unavailable")
                lines.append(
                    f"perpendicular '{a}'<->'{b}': FAIL — hard relationship UNVERIFIED "
                    "(no plane)"
                )
            elif ang < perp_angle_deg:
                ok = False
                evidence["runtime_status"] = "fail"
                evidence["reasons"].append("surfaces are not perpendicular")
                lines.append(
                    f"perpendicular '{a}'<->'{b}': FAIL (surfaces at {ang:.0f}° — should "
                    f"be perpendicular (~90°); make '{a}' and '{b}' meet at a right angle "
                    f"(one level, one vertical))"
                )
            else:
                nA, nB = _unit(A.get("plane_n")), _unit(B.get("plane_n"))
                support = wall = None
                wall_id = None
                wall_name = None
                canonical_wall = False
                if nA and nB and abs(nA[2]) >= 0.5 > abs(nB[2]):
                    support, wall, wall_id = A, B, bid
                    wall_name = b
                elif nA and nB and abs(nB[2]) >= 0.5 > abs(nA[2]):
                    support, wall, wall_id = B, A, aid
                    wall_name = a

                if support is not None:
                    canonical = _canonical_wall(wall_id)
                    canonical_wall = canonical is not None
                    wall_run = (
                        canonical[2]
                        if canonical
                        else _wall_run_from_normal(wall.get("plane_n"))
                    )
                    support_hull, support_source = _contact_hull(
                        support, root_plane=True
                    )
                    wall_hull, wall_source = _contact_hull(wall, root_plane=True)
                    support_z = _z_bounds(support, root_plane=True)
                    wall_z = _z_bounds(wall, root_plane=True)
                    support_interval = _projection_interval_xy(
                        support_hull, wall_run
                    )
                    wall_interval = _projection_interval_xy(wall_hull, wall_run)
                    run_overlap = _interval_overlap(support_interval, wall_interval)
                    vertical_overlap = _interval_overlap(support_z, wall_z)
                    run_gap = (
                        None if run_overlap is None else max(-run_overlap, 0.0)
                    )
                    vertical_gap = (
                        None
                        if vertical_overlap is None
                        else max(-vertical_overlap, 0.0)
                    )
                    xy_gap = _polygon_distance_xy(support_hull, wall_hull)
                    evidence["measurements"].update(
                        finite_support_hull_source=support_source,
                        finite_wall_hull_source=wall_source,
                        wall_run_axis_xy=(wall_run[:2] if wall_run else None),
                        support_wall_run_interval_m=support_interval,
                        wall_run_interval_m=wall_interval,
                        wall_run_gap_m=run_gap,
                        wall_run_overlap_m=(
                            None if run_overlap is None else max(run_overlap, 0.0)
                        ),
                        support_vertical_interval_m=support_z,
                        wall_vertical_interval_m=wall_z,
                        vertical_gap_m=vertical_gap,
                        vertical_overlap_m=(
                            None
                            if vertical_overlap is None
                            else max(vertical_overlap, 0.0)
                        ),
                        finite_xy_distance_m=xy_gap,
                        finite_contact_tolerance_m=finite_contact_gap,
                    )
                    finite_known = all(
                        x is not None for x in (run_gap, vertical_gap, xy_gap)
                    )
                    finite_pass = bool(
                        finite_known
                        and run_gap <= finite_contact_gap
                        and vertical_gap <= finite_contact_gap
                        and xy_gap <= finite_contact_gap
                    )
                else:
                    hullA, sourceA = _contact_hull(A, root_plane=True)
                    hullB, sourceB = _contact_hull(B, root_plane=True)
                    zA = _z_bounds(A, root_plane=True)
                    zB = _z_bounds(B, root_plane=True)
                    xy_gap = _polygon_distance_xy(hullA, hullB)
                    vertical_overlap = _interval_overlap(zA, zB)
                    vertical_gap = (
                        None
                        if vertical_overlap is None
                        else max(-vertical_overlap, 0.0)
                    )
                    evidence["measurements"].update(
                        a_finite_hull_source=sourceA,
                        b_finite_hull_source=sourceB,
                        finite_xy_distance_m=xy_gap,
                        vertical_gap_m=vertical_gap,
                        finite_contact_tolerance_m=finite_contact_gap,
                    )
                    finite_known = xy_gap is not None and vertical_gap is not None
                    finite_pass = bool(
                        finite_known
                        and math.hypot(xy_gap, vertical_gap) <= finite_contact_gap
                    )

                if not finite_known:
                    ok = False
                    evidence["runtime_status"] = "unverified"
                    evidence["reasons"].append(
                        "finite proximity/contact evidence unavailable"
                    )
                    lines.append(
                        f"perpendicular '{a}'<->'{b}': FAIL — hard relationship "
                        "UNVERIFIED (finite proximity/contact evidence unavailable; "
                        "an infinite-plane angle is insufficient)"
                    )
                elif finite_pass:
                    evidence["runtime_status"] = "pass"
                    lines.append(
                        f"perpendicular '{a}'<->'{b}': PASS ({ang:.0f}°; finite "
                        "surfaces meet)"
                    )
                else:
                    ok = False
                    evidence["runtime_status"] = "fail"
                    evidence["reasons"].append(
                        "finite surfaces do not meet within contact tolerance"
                    )
                    lines.append(
                        f"perpendicular '{a}'<->'{b}': FAIL (planes are {ang:.0f}° "
                        f"apart, but the finite surfaces do not meet within "
                        f"{finite_contact_gap:.2f}m; "
                        + (
                            f"keep '{wall_name}' fixed to its "
                            "canonical measured plane/point and either extend it without "
                            "translating, or rebuild/recenter the support"
                            if canonical_wall
                            else "extend/reposition them to share a finite edge/contact"
                        )
                        + " without changing the required angle)"
                    )
        else:  # unknown type: INFO (never fails)
            ang = _acute_angle_deg(A["n"], B["n"])
            evidence["runtime_status"] = "informational"
            evidence["measurements"]["acute_normal_angle_degrees"] = ang
            lines.append(
                f"{t} '{a}'<->'{b}': info"
                + (f" (face angle {ang:.0f}°)" if ang is not None else "")
            )
    if not relationships:
        return True, "relationships: PASS (none to check)"
    head = "relationships hold." if ok else "relationships FAIL — fix below:"
    return ok, "relationships: " + head + "".join("\n      - " + ln for ln in lines)


def _relationship_pose_constraint_report_v1(
    constraints: list[dict],
    sg_surfaces: dict,
    bodies: list[dict],
    surface_runs: dict,
    surface_top_hulls: dict,
    surface_plane_normals: dict,
    *,
    main_name: Optional[str] = None,
    main_support_run: Optional[dict] = None,
    independent_yaw_evidence: Optional[dict] = None,
    rect_min: float = 0.9,
    evidence_out: Optional[list] = None,
) -> tuple[bool, str]:
    ""
    try:
        from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
        from lib.tools.geometry.surface_relations import surface_build_name
    except Exception:  # pragma: no cover - package imports are required in production
        plane_is_reliable = lambda _plane: False

        def plumb_plane(_normal):
            return None, "ambiguous"

        def surface_build_name(sid):
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(sid).replace("#", "_")).lower()

    pose = [
        c
        for c in constraints or []
        if c.get("authority") == "required"
        and c.get("stage") == "POSE"
        and c.get("kind") == "edge_parallel_to_surface"
    ]
    if not pose:
        return True, "relationship POSE constraints: PASS (none to check)"

    body_by_name = {
        str(b.get("name")): b for b in bodies or [] if not b.get("is_obj")
    }

    def _build_name(sid: str) -> str:
        node = sg_surfaces.get(sid) or {}
        return str(node.get("build_name") or surface_build_name(sid))

    def _run_from_normal(normal):
        n = _unit(normal)
        if not n:
            return None
        h = math.hypot(float(n[0]), float(n[1]))
        return [-float(n[1]) / h, float(n[0]) / h, 0.0] if h > 1e-8 else None

    def _canonical_run(sid: str):
        entry = sg_surfaces.get(sid) or {}
        plane = entry.get("plane") or {
            "normal": entry.get("normal"),
            "inlier_frac": entry.get("inlier_frac"),
            "mask_frac": entry.get("mask_frac"),
        }
        normal, kind = plumb_plane(plane.get("normal"))
        if kind != "wall" or not normal or not plane_is_reliable(plane):
            return None
        return _run_from_normal(normal)

    contexts: list[dict] = []
    for constraint in pose:
        targets = list(constraint.get("targets") or [])
        target_id = str(
            (constraint.get("applicability") or {}).get("target_surface_id")
            or (targets[0] if targets else "")
        )
        reference = constraint.get("reference") or {}
        reference_id = str(
            reference.get("surface_id")
            or (targets[1] if len(targets) > 1 else "")
        )
        target_name = _build_name(target_id) if target_id else ""
        reference_name = _build_name(reference_id) if reference_id else ""
        binding = str(reference.get("run_binding") or "")
        # Always derive target yaw from its exact TOP hull.  ``surface_runs`` is the
        # complete grouped-body footprint and may be dominated by a rotated base/legs.
        top_run_info = _top_hull_run_info(
            (surface_top_hulls or {}).get(target_name), rect_min=rect_min
        )
        target_run = top_run_info.get("run")
        if binding == "canonical_scene_graph":
            reference_run = _canonical_run(reference_id)
            reference_source = "canonical_scene_graph"
        elif binding == "current_built_root":
            reference_run = _run_from_normal(
                (surface_plane_normals or {}).get(reference_name)
            )
            reference_source = "current_built_root"
        else:
            reference_run = None
            reference_source = "invalid_binding"

        # Main-support structured evidence is produced by the exact same top hull in
        # Blender and carries the richer disc/non-edge reason; use it as the authority
        # while retaining the independently host-fitted run above.
        if target_name == main_name and main_support_run:
            app_status = str(main_support_run.get("status") or "unknown")
            app_reason = str(
                main_support_run.get("reason") or "main_top_applicability_missing"
            )
            rectangularity = main_support_run.get("rectangularity")
            if app_status == "applicable":
                target_run = main_support_run.get("run") or target_run
            else:
                target_run = None
        else:
            app_status = str(top_run_info.get("status") or "unknown")
            app_reason = str(top_run_info.get("reason") or "top_edge_geometry_unavailable")
            rectangularity = top_run_info.get("rectangularity")
        contexts.append(
            {
                "constraint": constraint,
                "target_id": target_id,
                "reference_id": reference_id,
                "target_name": target_name,
                "reference_name": reference_name,
                "binding": binding,
                "reference_source": reference_source,
                "target_run": target_run,
                "reference_run": reference_run,
                "app_status": app_status,
                "app_reason": app_reason,
                "rectangularity": rectangularity,
            }
        )

    # Detect impossible required yaw sets before producing alternating repair hints.
    conflicts: dict[str, list[dict]] = {}
    for i, left in enumerate(contexts):
        if left["app_status"] != "applicable" or left["reference_run"] is None:
            continue
        ltol = float(
            (left["constraint"].get("parameters") or {}).get(
                "max_delta_degrees_mod_90", 3.0
            )
        )
        for right in contexts[i + 1 :]:
            if (
                right["target_id"] != left["target_id"]
                or right["app_status"] != "applicable"
                or right["reference_run"] is None
            ):
                continue
            rtol = float(
                (right["constraint"].get("parameters") or {}).get(
                    "max_delta_degrees_mod_90", 3.0
                )
            )
            delta = _yaw_delta_deg(left["reference_run"], right["reference_run"])
            if delta is not None and abs(delta) > ltol + rtol:
                item = {
                    "constraint_ids": [
                        left["constraint"].get("constraint_id"),
                        right["constraint"].get("constraint_id"),
                    ],
                    "delta_degrees_mod_90": delta,
                    "joint_tolerance_degrees": ltol + rtol,
                }
                for context in (left, right):
                    conflicts.setdefault(
                        str(context["constraint"].get("constraint_id")), []
                    ).append(item)

    # A strong independent PCA target is a separate required yaw observation.  It
    # must not be silently discarded when a relationship-derived target exists.
    independent = independent_yaw_evidence or {}
    selection = independent.get("selection") or {}
    verdict = independent.get("verdict") or {}
    selected_candidate = next(
        (
            candidate
            for candidate in independent.get("anchor_candidates", []) or []
            if candidate.get("id") == selection.get("candidate_id")
        ),
        None,
    )
    independent_run = (selected_candidate or {}).get("direction")
    independent_required = (
        selection.get("enforcement") == "hard"
        and independent_run is not None
        and verdict.get("status") not in {"unverified", "not_applicable"}
    )
    independent_tol = float(verdict.get("tolerance_degrees") or 12.0)
    if independent_required:
        for context in contexts:
            if context["app_status"] != "applicable" or context["reference_run"] is None:
                continue
            c = context["constraint"]
            ctol = float(
                (c.get("parameters") or {}).get("max_delta_degrees_mod_90", 3.0)
            )
            delta = _yaw_delta_deg(context["reference_run"], independent_run)
            if delta is not None and abs(delta) > ctol + independent_tol:
                conflicts.setdefault(str(c.get("constraint_id")), []).append(
                    {
                        "constraint_ids": [
                            c.get("constraint_id"),
                            "legacy-independent-photo-yaw",
                        ],
                        "delta_degrees_mod_90": delta,
                        "joint_tolerance_degrees": ctol + independent_tol,
                    }
                )

    lines: list[str] = []
    ok = True
    for context in contexts:
        constraint = context["constraint"]
        cid = str(constraint.get("constraint_id") or "")
        tolerance = float(
            (constraint.get("parameters") or {}).get(
                "max_delta_degrees_mod_90", 3.0
            )
        )
        result = {
            "constraint_id": cid,
            "source_relationship_id": constraint.get("source_relationship_id"),
            "kind": constraint.get("kind"),
            "stage": "POSE",
            "runtime_status": "not_checked",
            "targets": list(constraint.get("targets") or []),
            "measurements": {
                "target_build_name": context["target_name"],
                "reference_build_name": context["reference_name"],
                "reference_run_source": context["reference_source"],
                "reference_run": context["reference_run"],
                "built_target_run": context["target_run"],
                "rectangularity": context["rectangularity"],
                "tolerance_degrees_mod_90": tolerance,
            },
            "reason_codes": [],
        }
        if evidence_out is not None:
            evidence_out.append(result)
        if cid in conflicts:
            ok = False
            result["runtime_status"] = "inconsistent"
            result["reason_codes"].append("required_yaw_constraints_inconsistent")
            result["measurements"]["conflicts"] = conflicts[cid]
            lines.append(
                f"{cid}: FAIL — CONSTRAINT SET INCONSISTENT; do not alternate "
                "rotations or move a canonical wall"
            )
            continue
        if not context["target_id"] or not context["reference_id"]:
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("constraint_endpoint_missing")
            lines.append(f"{cid}: UNVERIFIED — constraint endpoint is missing")
            continue
        if (
            context["target_name"] not in body_by_name
            or context["reference_name"] not in body_by_name
        ):
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("built_surface_missing")
            lines.append(
                f"{cid}: UNVERIFIED — '{context['target_name']}' or "
                f"'{context['reference_name']}' is absent"
            )
            continue
        if context["app_status"] == "not_applicable":
            result["runtime_status"] = "not_applicable"
            result["reason_codes"].append(context["app_reason"])
            lines.append(
                f"{cid}: NOT APPLICABLE — '{context['target_name']}' has no "
                "rectangular edge-bearing top"
            )
            continue
        if context["app_status"] != "applicable":
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append(context["app_reason"])
            lines.append(
                f"{cid}: UNVERIFIED — target top-edge applicability is unknown "
                f"({context['app_reason']})"
            )
            continue
        if context["binding"] not in {"canonical_scene_graph", "current_built_root"}:
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("invalid_reference_binding")
            lines.append(f"{cid}: UNVERIFIED — invalid wall-run binding")
            continue
        if context["reference_run"] is None or context["target_run"] is None:
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("yaw_run_unavailable")
            lines.append(f"{cid}: UNVERIFIED — support or wall run is unavailable")
            continue
        delta = _yaw_delta_deg(context["target_run"], context["reference_run"])
        result["measurements"]["delta_degrees_mod_90"] = delta
        if delta is None:
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("yaw_delta_unavailable")
            lines.append(f"{cid}: UNVERIFIED — yaw delta cannot be measured")
        elif abs(delta) <= tolerance:
            result["runtime_status"] = "pass"
            lines.append(
                f"{cid}: PASS ('{context['target_name']}' edge is {delta:+.1f}deg "
                f"from '{context['reference_name']}'s {context['reference_source']} run)"
            )
        else:
            ok = False
            result["runtime_status"] = "fail"
            result["reason_codes"].append("edge_not_parallel_to_wall_run")
            body = body_by_name.get(context["target_name"]) or {}
            lo, hi = body.get("lo") or [0, 0, 0], body.get("hi") or [0, 0, 0]
            center = [(float(lo[i]) + float(hi[i])) / 2.0 for i in range(3)]
            ownership = (
                "; the wall is canonical and must not move"
                if context["binding"] == "canonical_scene_graph"
                else ""
            )
            lines.append(
                f"{cid}: FAIL — rotate '{context['target_name']}' by {delta:+.1f}deg "
                f"about world Z through its own centre ({center[0]:+.2f}, "
                f"{center[1]:+.2f}, {center[2]:+.2f}) so its edge is parallel to "
                f"'{context['reference_name']}' within {tolerance:.1f}deg{ownership}"
            )
    for result in evidence_out or []:
        if result.get("stage") == "POSE" and result.get("runtime_status") == "not_checked":
            ok = False
            result["runtime_status"] = "unverified"
            result.setdefault("reason_codes", []).append(
                "pose_evaluator_no_terminal_status"
            )
    head = "PASS" if ok else "FAIL"
    return ok, "relationship POSE constraints: " + head + "\n      - " + "\n      - ".join(lines)


def relationship_pose_constraint_report(
    constraints: list[dict],
    sg_surfaces: dict,
    bodies: list[dict],
    surface_runs: dict,
    surface_top_hulls: dict,
    surface_plane_normals: dict,
    *,
    main_name: Optional[str] = None,
    main_support_run: Optional[dict] = None,
    rect_min: float = 0.9,
    evidence_out: Optional[list] = None,
) -> tuple[bool, str]:
    """Evaluate every compiled required yaw target as one feasible POSE set.

    Hard AGAINST rows resolve a canonical or live wall run.  Independent source-yaw
    rows use only their frozen compiled run.  All bands for the same support are
    intersected before any repair text is emitted, so the agent receives exactly one
    jointly feasible correction—or one inconsistency with no rotation hint.
    """
    try:
        from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
        from lib.tools.geometry.surface_relations import surface_build_name
    except Exception:  # pragma: no cover - package imports are required in production
        plane_is_reliable = lambda _plane: False

        def plumb_plane(_normal):
            return None, "ambiguous"

        def surface_build_name(sid):
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(sid).replace("#", "_")).lower()

    pose = [
        constraint
        for constraint in constraints or []
        if constraint.get("authority") == "required"
        and constraint.get("stage") == "POSE"
        and constraint.get("kind")
        in {"edge_parallel_to_surface", "main_support_edge_yaw"}
    ]
    if not pose:
        return True, "yaw POSE constraints: PASS (none to check)"

    body_by_name = {
        str(body.get("name")): body for body in bodies or [] if not body.get("is_obj")
    }

    def _build_name(surface_id: str) -> str:
        node = sg_surfaces.get(surface_id) or {}
        return str(node.get("build_name") or surface_build_name(surface_id))

    def _run_from_normal(normal):
        unit = _unit(normal)
        if not unit:
            return None
        horizontal = math.hypot(float(unit[0]), float(unit[1]))
        if horizontal < 1e-8:
            return None
        return [-float(unit[1]) / horizontal, float(unit[0]) / horizontal, 0.0]

    def _canonical_run(surface_id: str):
        entry = sg_surfaces.get(surface_id) or {}
        plane = entry.get("plane") or {
            "normal": entry.get("normal"),
            "inlier_frac": entry.get("inlier_frac"),
            "mask_frac": entry.get("mask_frac"),
        }
        normal, kind = plumb_plane(plane.get("normal"))
        if kind != "wall" or normal is None or not plane_is_reliable(plane):
            return None
        return _run_from_normal(normal)

    contexts: list[dict] = []
    for constraint in pose:
        targets = list(constraint.get("targets") or [])
        applicability = constraint.get("applicability") or {}
        target_id = str(
            applicability.get("target_surface_id")
            or (targets[0] if targets else "")
        )
        target_name = _build_name(target_id) if target_id else ""
        top_info = _top_hull_run_info(
            (surface_top_hulls or {}).get(target_name), rect_min=rect_min
        )
        target_run = top_info.get("run")
        if target_name == main_name and main_support_run:
            app_status = str(main_support_run.get("status") or "unknown")
            app_reason = str(
                main_support_run.get("reason") or "main_top_applicability_missing"
            )
            rectangularity = main_support_run.get("rectangularity")
            target_run = (
                main_support_run.get("run") or target_run
                if app_status == "applicable"
                else None
            )
        else:
            app_status = str(top_info.get("status") or "unknown")
            app_reason = str(top_info.get("reason") or "top_edge_geometry_unavailable")
            rectangularity = top_info.get("rectangularity")

        kind = str(constraint.get("kind") or "")
        reference = constraint.get("reference") or {}
        reference_id = ""
        reference_name = ""
        binding = str(reference.get("run_binding") or "")
        if kind == "main_support_edge_yaw":
            reference_run = reference.get("run_xy")
            reference_source = "frozen_source_consensus"
        else:
            reference_id = str(
                reference.get("surface_id")
                or (targets[1] if len(targets) > 1 else "")
            )
            reference_name = _build_name(reference_id) if reference_id else ""
            if binding == "canonical_scene_graph":
                reference_run = _canonical_run(reference_id)
                reference_source = "canonical_scene_graph"
            elif binding == "current_built_root":
                reference_run = _run_from_normal(
                    (surface_plane_normals or {}).get(reference_name)
                )
                reference_source = "current_built_root"
            else:
                reference_run = None
                reference_source = "invalid_binding"
        contexts.append(
            {
                "constraint": constraint,
                "kind": kind,
                "target_id": target_id,
                "target_name": target_name,
                "reference_id": reference_id,
                "reference_name": reference_name,
                "binding": binding,
                "reference_source": reference_source,
                "reference_run": reference_run,
                "target_run": target_run,
                "app_status": app_status,
                "app_reason": app_reason,
                "rectangularity": rectangularity,
                "result": None,
                "terminal": False,
            }
        )

    lines: list[str] = []
    ok = True
    grouped: dict[str, list[dict]] = {}
    for context in contexts:
        constraint = context["constraint"]
        constraint_id = str(constraint.get("constraint_id") or "")
        tolerance = float(
            (constraint.get("parameters") or {}).get(
                "max_delta_degrees_mod_90", 3.0
            )
        )
        result = {
            "constraint_id": constraint_id,
            "source_relationship_id": constraint.get("source_relationship_id"),
            "source_yaw_evidence_id": constraint.get("source_yaw_evidence_id"),
            "kind": context["kind"],
            "stage": "POSE",
            "runtime_status": "not_checked",
            "targets": list(constraint.get("targets") or []),
            "measurements": {
                "target_build_name": context["target_name"],
                "reference_build_name": context["reference_name"] or None,
                "reference_run_source": context["reference_source"],
                "reference_run": context["reference_run"],
                "built_target_run": context["target_run"],
                "rectangularity": context["rectangularity"],
                "tolerance_degrees_mod_90": tolerance,
            },
            "reason_codes": [],
        }
        context["result"] = result
        context["tolerance"] = tolerance
        if evidence_out is not None:
            evidence_out.append(result)

        if not context["target_id"] or context["target_name"] not in body_by_name:
            ok = False
            context["terminal"] = True
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("built_target_surface_missing")
            lines.append(
                f"{constraint_id}: UNVERIFIED — target support is absent"
            )
            continue
        if context["app_status"] == "not_applicable":
            context["terminal"] = True
            if context["kind"] == "main_support_edge_yaw":
                ok = False
                result["runtime_status"] = "fail"
                result["reason_codes"].append("source_build_shape_inconsistent")
                lines.append(
                    f"{constraint_id}: FAIL — source evidence confirms an edge-bearing "
                    f"top, but '{context['target_name']}' was built without one "
                    "(shape_inconsistent)"
                )
            else:
                result["runtime_status"] = "not_applicable"
                result["reason_codes"].append(context["app_reason"])
                lines.append(
                    f"{constraint_id}: NOT APPLICABLE — '{context['target_name']}' "
                    "has no rectangular edge-bearing top"
                )
            continue
        if context["app_status"] != "applicable":
            ok = False
            context["terminal"] = True
            result["runtime_status"] = "unverified"
            result["reason_codes"].append(context["app_reason"])
            lines.append(
                f"{constraint_id}: UNVERIFIED — target top-edge applicability is "
                f"unknown ({context['app_reason']})"
            )
            continue
        if context["kind"] == "edge_parallel_to_surface" and (
            not context["reference_id"]
            or context["reference_name"] not in body_by_name
        ):
            ok = False
            context["terminal"] = True
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("built_reference_surface_missing")
            lines.append(
                f"{constraint_id}: UNVERIFIED — reference wall is absent"
            )
            continue
        if context["reference_run"] is None or context["target_run"] is None:
            ok = False
            context["terminal"] = True
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("yaw_run_unavailable")
            lines.append(
                f"{constraint_id}: UNVERIFIED — support or reference run is unavailable"
            )
            continue
        grouped.setdefault(context["target_id"], []).append(context)

    for target_id, group in sorted(grouped.items()):
        target_name = group[0]["target_name"]
        built_run = group[0]["target_run"]
        feasible = _required_yaw_feasible_set(
            [
                (context["reference_run"], context["tolerance"])
                for context in group
            ],
            built_run,
        )
        constraint_ids = [
            str(context["constraint"].get("constraint_id")) for context in group
        ]
        if not feasible.get("feasible"):
            ok = False
            for context in group:
                result = context["result"]
                result["runtime_status"] = "inconsistent"
                result["reason_codes"].append(
                    "required_yaw_constraints_have_no_common_solution"
                )
                result["measurements"]["constraint_set_ids"] = constraint_ids
                result["measurements"]["joint_feasible_interval"] = None
                result["measurements"]["combined_correction_degrees"] = None
            lines.append(
                f"required yaw set for '{target_name}': CONSTRAINT SET INCONSISTENT "
                "— required yaw bands have no common solution; suppressing all "
                "rotation hints (" + ", ".join(constraint_ids) + ")"
            )
            continue

        correction = feasible.get("correction_degrees")
        group_pass = correction is not None and abs(float(correction)) <= 1e-7
        for context in group:
            result = context["result"]
            delta = _yaw_delta_deg(context["target_run"], context["reference_run"])
            result["measurements"].update(
                delta_degrees_mod_90=delta,
                constraint_set_ids=constraint_ids,
                joint_feasible_interval=feasible.get("interval"),
                combined_correction_degrees=correction,
            )
            if delta is None:
                ok = False
                result["runtime_status"] = "unverified"
                result["reason_codes"].append("yaw_delta_unavailable")
            elif abs(delta) <= context["tolerance"] + 1e-9:
                result["runtime_status"] = "pass"
            else:
                ok = False
                result["runtime_status"] = "fail"
                result["reason_codes"].append("outside_required_yaw_band")
        if group_pass:
            lines.append(
                f"required yaw set for '{target_name}': PASS — all "
                f"{len(group)} required band(s) share the current yaw"
            )
        else:
            ok = False
            body = body_by_name.get(target_name) or {}
            lo, hi = body.get("lo") or [0, 0, 0], body.get("hi") or [0, 0, 0]
            center = [(float(lo[i]) + float(hi[i])) / 2.0 for i in range(3)]
            lines.append(
                f"required yaw set for '{target_name}': FAIL — rotate once by "
                f"{float(correction):+.1f}deg about world Z through its own centre "
                f"({center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f}) into the "
                "common feasible interval; do not follow individual target hints"
            )

    for context in contexts:
        result = context["result"]
        if result.get("runtime_status") == "not_checked":
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("pose_evaluator_no_terminal_status")
    return ok, "yaw POSE constraints: " + ("PASS" if ok else "FAIL") + (
        "\n      - " + "\n      - ".join(lines) if lines else ""
    )


def relationship_contact_constraint_report(
    constraints: list[dict],
    sg_surfaces: dict,
    bodies: list[dict],
    surface_normals: dict,
    *,
    main_name: Optional[str] = None,
    surface_runs: Optional[dict] = None,
    surface_hulls: Optional[dict] = None,
    surface_top_hulls: Optional[dict] = None,
    surface_plane_hulls: Optional[dict] = None,
    surface_plane_bounds: Optional[dict] = None,
    surface_plane_centroids: Optional[dict] = None,
    surface_plane_normals: Optional[dict] = None,
    evidence_out: Optional[list] = None,
) -> tuple[bool, str]:
    """Evaluate required compiled relationship CONTACT obligations.

    The mature finite-geometry implementation remains centralized in
    :func:`relationship_report`, but the runtime input is now the immutable compiled
    constraints rather than raw relationship rows.  AGAINST yaw is explicitly disabled
    here because its named constraint is evaluated once at POSE.
    """
    contact = [
        c
        for c in constraints or []
        if c.get("authority") == "required" and c.get("stage") == "CONTACT"
    ]
    if not contact:
        return True, "relationship CONTACT constraints: PASS (none to check)"
    kind_to_type = {
        "finite_against": "against",
        "finite_under": "under",
        "finite_wall_corner": "corner",
        "finite_perpendicular": "perpendicular",
    }
    ok = True
    lines: list[str] = []

    def _reason_codes(
        kind: str,
        status: str,
        measurements: dict,
        legacy_reasons: list,
        params: dict,
    ) -> list[str]:
        """Translate the geometry verdict into stable machine reason codes.

        ``relationship_report`` predates structured telemetry and retains prose repair
        reasons for its standalone callers.  Compiled constraints expose a deliberately
        smaller, stable vocabulary derived from the measured failure predicate—not that
        prose—so downstream consumers never need to parse English.
        """
        if status in {"pass", "not_applicable"}:
            return []
        reason_text = " ".join(str(reason).lower() for reason in legacy_reasons)
        if "surface not located" in reason_text:
            return ["built_surface_missing"]
        if "same built body" in reason_text:
            return ["constraint_endpoints_alias_same_body"]
        if "canonical wall reference" in reason_text:
            return ["canonical_reference_unavailable"]
        if status == "inconsistent":
            return ["required_constraints_inconsistent"]
        if kind == "finite_against":
            if status == "unverified":
                if "reference plane" in reason_text:
                    return ["reference_plane_unavailable"]
                return ["finite_extent_evidence_unavailable"]
            finite_tol = max(
                float(params.get("max_finite_run_gap_m", 0.05)),
                float(params.get("max_vertical_gap_m", 0.05)),
            )
            run_gap = measurements.get("wall_run_gap_m")
            vertical_gap = measurements.get("vertical_gap_m")
            codes = []
            if run_gap is not None and float(run_gap) > finite_tol:
                codes.append("finite_run_gap_exceeded")
            if vertical_gap is not None and float(vertical_gap) > finite_tol:
                codes.append("vertical_gap_exceeded")
            signed_gap = measurements.get("signed_gap_m")
            if signed_gap is not None:
                signed_gap = float(signed_gap)
                if signed_gap > float(params.get("max_float_gap_m", 0.01)):
                    codes.append("float_gap_exceeded")
                if signed_gap < -float(params.get("max_penetration_m", 0.01)):
                    codes.append("penetration_exceeded")
            return codes or ["finite_against_failed"]
        if kind == "finite_under":
            if status == "unverified":
                return ["finite_contact_evidence_unavailable"]
            codes = []
            gap = measurements.get("vertical_contact_gap_m")
            if gap is not None and float(gap) > float(
                params.get("max_vertical_gap_m", 0.05)
            ):
                codes.append("vertical_gap_exceeded")
            if measurements.get("xy_overlap") is False:
                codes.append("xy_overlap_required")
            return codes or ["finite_under_failed"]
        if kind == "finite_wall_corner":
            if status == "unverified":
                return [
                    "plane_or_finite_extent_evidence_unavailable"
                    if measurements.get("normal_angle_deg") is None
                    else "finite_extent_evidence_unavailable"
                ]
            codes = []
            angle = measurements.get("normal_angle_deg")
            if angle is not None and float(angle) < float(
                params.get("minimum_normal_angle_degrees", 85.0)
            ):
                codes.append("normal_angle_below_minimum")
            vertical_overlap = measurements.get("vertical_overlap_m")
            if vertical_overlap is not None and float(vertical_overlap) <= 0.0:
                codes.append("positive_vertical_overlap_required")
            xy_gap = measurements.get("finite_xy_distance_m")
            if xy_gap is not None and float(xy_gap) > float(
                params.get("max_vertical_gap_m", 0.05)
            ):
                codes.append("finite_xy_gap_exceeded")
            reach = [
                measurements.get("a_corner_distance_m"),
                measurements.get("b_corner_distance_m"),
            ]
            if all(value is not None for value in reach) and max(
                float(value) for value in reach
            ) > float(params.get("max_built_reach_error_m", 0.05)):
                codes.append("corner_reach_exceeded")
            return codes or ["finite_wall_corner_failed"]
        if kind == "finite_perpendicular":
            if status == "unverified":
                return [
                    "plane_evidence_unavailable"
                    if measurements.get("normal_angle_deg") is None
                    else "finite_contact_evidence_unavailable"
                ]
            angle = measurements.get("normal_angle_deg")
            if angle is not None and float(angle) < float(
                params.get("minimum_normal_angle_degrees", 88.0)
            ):
                return ["normal_angle_below_minimum"]
            return ["finite_contact_gap_exceeded"]
        return ["contact_constraint_evaluation_failed"]

    for constraint in contact:
        cid = str(constraint.get("constraint_id") or "")
        kind = str(constraint.get("kind") or "")
        targets = list(constraint.get("targets") or [])
        result = {
            "constraint_id": cid,
            "source_relationship_id": constraint.get("source_relationship_id"),
            "kind": kind,
            "stage": "CONTACT",
            "runtime_status": "not_checked",
            "targets": targets,
            "measurements": {},
            "reason_codes": [],
        }
        if evidence_out is not None:
            evidence_out.append(result)
        rel_type = kind_to_type.get(kind)
        if rel_type is None or len(targets) != 2:
            ok = False
            result["runtime_status"] = "unverified"
            result["reason_codes"].append("unsupported_or_malformed_constraint")
            lines.append(f"{cid}: UNVERIFIED — unsupported or malformed constraint")
            continue
        params = constraint.get("parameters") or {}
        reference = constraint.get("reference") or {}
        pseudo = {
            "relationship_id": cid,
            "type": rel_type,
            "a": targets[0],
            "b": targets[1],
            "status": "confirmed",
            "enforcement": "hard",
            "_compiled_constraint": True,
            "_constraint_reference": reference,
        }
        if kind == "finite_wall_corner":
            pseudo["_corner_xy_m"] = reference.get("corner_xy_m")
        legacy_evidence: list[dict] = []
        constraint_ok, message = relationship_report(
            [pseudo],
            sg_surfaces,
            bodies,
            surface_normals,
            against_gap=float(params.get("max_float_gap_m", 0.01)),
            under_gap=float(params.get("max_vertical_gap_m", 0.05)),
            against_pen=float(params.get("max_penetration_m", 0.01)),
            against_yaw_deg=float(params.get("max_delta_degrees_mod_90", 3.0)),
            corner_angle_deg=float(params.get("minimum_normal_angle_degrees", 85.0)),
            perp_angle_deg=float(params.get("minimum_normal_angle_degrees", 88.0)),
            main_name=main_name,
            surface_runs=surface_runs,
            surface_hulls=surface_hulls,
            surface_top_hulls=surface_top_hulls,
            surface_plane_hulls=surface_plane_hulls,
            surface_plane_bounds=surface_plane_bounds,
            surface_plane_centroids=surface_plane_centroids,
            surface_plane_normals=surface_plane_normals,
            finite_contact_gap=float(
                params.get(
                    "max_finite_contact_gap_m",
                    max(
                        float(params.get("max_finite_run_gap_m", 0.05)),
                        float(params.get("max_vertical_gap_m", 0.05)),
                    ),
                )
            ),
            corner_reach_gap=float(params.get("max_built_reach_error_m", 0.05)),
            check_against_yaw=False,
            evidence_out=legacy_evidence,
        )
        legacy = legacy_evidence[0] if legacy_evidence else {}
        legacy_status = str(legacy.get("runtime_status") or "")
        if legacy_status not in {"pass", "fail", "unverified", "inconsistent"}:
            legacy_status = "unverified"
            constraint_ok = False
        result["runtime_status"] = legacy_status
        result["measurements"] = dict(legacy.get("measurements") or {})
        result["reason_codes"] = _reason_codes(
            kind,
            legacy_status,
            result["measurements"],
            list(legacy.get("reasons") or []),
            params,
        )
        ok = ok and constraint_ok
        detail = message.split("\n      - ", 1)[-1]
        lines.append(f"{cid}: {detail}")
    head = "PASS" if ok else "FAIL"
    return ok, "relationship CONTACT constraints: " + head + "\n      - " + "\n      - ".join(lines)


def compiled_contact_owner_pairs(
    constraints: list[dict], build_name_by_surface_id: dict[str, str]
) -> set[frozenset[str]]:
    """Return exact built pairs whose dedicated compiled contact rule owns overlap.

    Only active required ``finite_against``/``finite_under`` obligations may suppress
    generic penetration. Advisory/rejected relationship rows never reach this input;
    CORNER/PERPENDICULAR intentionally own no penetration semantics.
    """
    pairs: set[frozenset[str]] = set()
    for constraint in constraints or []:
        owned = (constraint.get("effects") or {}).get("owns_contact_pair")
        if (
            constraint.get("authority") != "required"
            or constraint.get("stage") != "CONTACT"
            or constraint.get("kind") not in {"finite_against", "finite_under"}
            or not isinstance(owned, list)
            or len(owned) != 2
            or any(endpoint not in build_name_by_surface_id for endpoint in owned)
        ):
            continue
        pairs.add(
            frozenset(
                (
                    build_name_by_surface_id[owned[0]],
                    build_name_by_surface_id[owned[1]],
                )
            )
        )
    return pairs


def main_support_top_report(
    bodies: list[dict],
    main_name: Optional[str] = None,
    tol: float = 0.001,
    form: Optional[str] = None,
    floor_name: Optional[str] = None,
) -> tuple[bool, str]:
    """Rule: the main support's TOP face must sit at z=0. Returns ``(passed, message)``;
    on failure the message names the surface and the exact z-shift to apply.

    The main support is taken by NAME (``main_name`` — the scene graph's form-tagged surface,
    e.g. ``cutting_board_0``) when given: authoritative, so a mis-placed table is never mistaken for
    the floor below it and the rule can't vacuously pass while the real support exists. Only when no
    name is supplied (older graphs) does it fall back to the geometric guess
    (``identify_main_support``). If the named support isn't found among the built surfaces it is
    reported as not-yet-built (non-blocking) rather than silently skipped."""
    name = top_z = None
    if main_name:
        b = next(
            (x for x in bodies if not x.get("is_obj") and x.get("name") == main_name),
            None,
        )
        if b is not None:
            name, top_z = main_name, b["hi"][2]
        else:  # named support not built yet -> say so
            return True, (
                f"main support top at z=0: not verified — the main support '{main_name}' "
                f"is not among the built surfaces; build it and name it exactly "
                f"'{main_name}'."
            )
    else:
        name, top_z = identify_main_support(bodies)
    if name is None:
        return (
            True,
            "main support top at z=0: PASS (no main support identified to check)",
        )
    if abs(top_z) <= tol:
        return True, f"main support top at z=0: PASS ('{name}' top z={top_z:+.4f})"
    dz = -top_z
    if form == "table" and floor_name and floor_name != name:
        # Full-piece table whose base rests on a floor: a Z-shift moves the base too, so co-move the
        # floor by the same dz -> the top reaches z=0 AND the base stays flush on the floor (any FAIL
        # here means the height is off / the base isn't on the floor; shifting both preserves the
        # intended base-on-floor contact). A pure resize would splay the legs, so it's forbidden.
        return False, (
            f"main support top at z=0: FAIL — '{name}' top is at z={top_z:+.4f}. '{name}' is a FULL "
            f"table whose base rests on the floor '{floor_name}', so a Z-shift moves the base too. "
            f"Co-translate BOTH by {dz:+.4f} in Z (obj.location.z += {dz:+.4f} on '{name}' AND on "
            f"'{floor_name}') so the top reaches z=0 while the base stays flush on the floor. PURE "
            f"translation on each — do NOT rebuild or rescale (keeps the legs attached to the top "
            f"and the base on the floor)."
            " (If the top landed at about HALF the height you computed after a fresh build, you derived obj.scale from half-extents on a size=1 cube — it spans ±0.5, not ±1; size box parts with obj.dimensions = FULL sizes.)"
        )
    return False, (
        f"main support top at z=0: FAIL — '{name}' top is at z={top_z:+.4f}. Translate the EXISTING "
        f"'{name}' by {dz:+.4f} in Z (obj.location.z += {dz:+.4f}) so its TOP reaches z=0. Do this "
        f"as a PURE translation — do NOT rebuild or rescale it; recomputing location.z from the "
        f"mesh half-height is error-prone (the top is location.z + half-height, not location.z)."
        " (If the top landed at about HALF the height you computed after a fresh build, you derived obj.scale from half-extents on a size=1 cube — it spans ±0.5, not ±1; size box parts with obj.dimensions = FULL sizes.)"
    )


def main_support_bottom_report(
    bodies: list[dict],
    main_name: Optional[str],
    form: Optional[str],
    min_height: float = 0.15,
) -> tuple[bool, str]:
    """Rule: a ``form == "table"`` main support (a FULL piece, either classified directly or
    required by an explicit floor-under-main relationship) must actually extend BELOW its top
    slab: total built height ``hi[2] - lo[2]`` must exceed ``min_height``. Catches the lazy build
    where the agent makes a bare thin slab for the full-piece form — that passes the connected
    check (a single island is trivially connected) and the z=0 check, yet has no bottom.
    Non-blocking PASS for the effective ``tabletop`` form (a slab IS the correct build), when
    name/form is unknown, or when the named support isn't built yet (the top rule already reports
    that)."""
    if form != "table" or not main_name:
        return True, "main support bottom: PASS (not a full-piece table form)"
    b = next(
        (x for x in bodies if not x.get("is_obj") and x.get("name") == main_name), None
    )
    if b is None:
        return True, f"main support bottom: not verified ('{main_name}' not built yet)"
    height = float(b["hi"][2]) - float(b["lo"][2])
    if height > min_height:
        return (
            True,
            f"main support bottom: PASS ('{main_name}' extends {height:.2f}m below its top)",
        )
    return False, (
        f"main support bottom: FAIL — the main support's form is a FULL table but '{main_name}' is "
        f"only {height * 100:.0f}cm tall (a bare slab). The table must have a BOTTOM: build its "
        f"legs/pedestal/base extending DOWN from the top's underside (to the floor if one exists), "
        f"joined into the same '{main_name}' object — see [TABLE ASSEMBLY]."
    )


def _fmt_box(lo: list, hi: list) -> str:
    return f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]"


def _axis_gaps(a: list, b: list) -> list[float]:
    """Per-axis separation between two AABBs (0.0 where they overlap on that axis)."""
    (lo_a, hi_a), (lo_b, hi_b) = a, b
    return [max(lo_b[d] - hi_a[d], lo_a[d] - hi_b[d], 0.0) for d in range(3)]


def main_support_connected_report(
    islands: Optional[list], gap: float = 0.02
) -> tuple[bool, str]:
    """Rule: every part of the built main support must be CONNECTED — no leg/base detached from the
    top. ``islands`` are the world AABBs (``[[lo, hi], ...]``) of the main-support object's mesh
    loose-parts (the top slab + each leg), from the penetration script. Two islands are LINKED when
    their AABBs overlap or lie within ``gap`` on ALL three axes; the support passes when the islands
    form a SINGLE connected component. PASS (non-blocking) with 0/1 island — a solid slab or an
    un-split mesh is connected by construction — or when no island data was emitted. Catches the
    observed failure where a non-uniform ``obj.scale`` splays the legs off the top (in XY or Z).

    On FAIL the message is DIAGNOSTIC (the agent cannot see the islands any other way — even
    get_scene_info only shows the joined object's whole bbox): it lists each disconnected piece's
    union AABB (topmost first), the closest cross-piece gap with its separating axes and the move
    that closes it, and clarifies that box overlap suffices (no boolean/weld needed) — without
    this, agents were observed burning whole attempts guessing the check's semantics."""
    if not islands or len(islands) <= 1:
        return True, "main support connected: PASS (single piece)"
    n = len(islands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a: list, b: list) -> bool:
        (lo_a, hi_a), (lo_b, hi_b) = a, b
        return all(
            min(hi_a[d], hi_b[d]) >= max(lo_a[d], lo_b[d]) - gap for d in range(3)
        )

    for i in range(n):
        for j in range(i + 1, n):
            if near(islands[i], islands[j]):
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    if len(comps) == 1:
        return True, f"main support connected: PASS ({n} parts joined)"
    # Diagnostic FAIL: list each piece's union AABB (topmost first) and the closest
    # cross-piece gap so the agent sees WHERE the detachment is and how far to move.
    pieces = sorted(comps.values(), key=lambda c: -max(islands[i][1][2] for i in c))

    def union_box(idxs: list[int]) -> tuple[list, list]:
        lo = [min(islands[i][0][d] for i in idxs) for d in range(3)]
        hi = [max(islands[i][1][d] for i in idxs) for d in range(3)]
        return lo, hi

    listed = "; ".join(
        f"#{k + 1} ({len(c)} part{'s' if len(c) > 1 else ''}) {_fmt_box(*union_box(c))}"
        for k, c in enumerate(pieces[:6])
    )
    if len(pieces) > 6:
        listed += f"; … (+{len(pieces) - 6} more)"
    best = None  # (worst-axis gap, gaps, piece a, piece b, island a, island b)
    for ka in range(len(pieces)):
        for kb in range(ka + 1, len(pieces)):
            for i in pieces[ka]:
                for j in pieces[kb]:
                    g = _axis_gaps(islands[i], islands[j])
                    if best is None or max(g) < best[0]:
                        best = (max(g), g, ka, kb, i, j)
    _, gaps, ka, kb, ia, ib = best
    moves = []
    for d, ax in enumerate("XYZ"):
        if gaps[d] > gap:
            # islands[ib] (in the lower piece) needs +ax when it sits below/before islands[ia]
            sign = "+" if islands[ib][1][d] < islands[ia][0][d] else "-"
            moves.append(f"{gaps[d]:.2f}m along {ax} (move {sign}{ax})")
    return False, (
        f"main support connected: FAIL — the main support is in {len(comps)} disconnected "
        f"pieces (of {n} parts).\n"
        f"Pieces (world boxes, m): {listed}\n"
        f"Closest gap: piece #{kb + 1} is separated from piece #{ka + 1} by "
        + " and ".join(moves)
        + f" — translate/extend it by those amounts plus ~{gap:.2f}m so their boxes overlap.\n"
        f"Connection is judged by box overlap (within {gap * 100:.0f}cm): volumetric overlap is "
        f"ENOUGH — do NOT boolean-union or weld vertices. Keep base parts within the top's "
        f"footprint, overlapping up into its underside; never resize the joined table with a "
        f"non-uniform obj.scale (build the parts at their final sizes instead). If a gap "
        f"persists after a move that should have closed it (or every part lands at HALF its "
        f"intended size), you derived obj.scale from half-extents on a size=1 cube (it spans "
        f"\u00b10.5) — rebuild ONLY the base part with obj.dimensions = FULL sizes, spanning "
        f"from the floor up INTO the top slab, then re-join."
    )


def resolve_yaw_anchor(
    built_run,
    photo_run,
    ecc,
    wall_run,
    main_against: bool,
    ecc_gate: float = 1.5,
    ecc_gate_no_anchor: float = 1.15,
):
    ""
    if built_run is None:
        return None, None
    if main_against and wall_run is not None:
        return wall_run, "against-wall"
    if photo_run is not None and ecc is not None and ecc >= ecc_gate:
        return photo_run, "pca"
    if wall_run is not None:
        return wall_run, "wall"
    if photo_run is not None and ecc is not None and ecc >= ecc_gate_no_anchor:
        return photo_run, "pca-low"
    return None, None


def resolve_yaw_evidence(
    main_surface: Optional[str],
    applicability: dict,
    built_run,
    photo_run,
    ecc,
    wall_candidates: Optional[list[dict]] = None,
    yaw_tol_deg: float = 12.0,
    against_yaw_tol_deg: float = 3.0,
    ecc_gate: float = 1.5,
    ecc_gate_no_anchor: float = 1.15,
    photo_border_evidence: Optional[dict] = None,
) -> dict:
    ""
    status = str((applicability or {}).get("status") or "unknown")
    reason = str((applicability or {}).get("reason") or "applicability_missing")
    geometry_source = str(
        (applicability or {}).get("geometry_source") or "main_top_hull"
    )
    rectangularity = (applicability or {}).get("rectangularity")
    record = {
        "schema_version": 1,
        "main_surface": main_surface,
        "applicability": {
            "status": status,
            "reason": reason,
            "geometry_source": geometry_source,
            "rectangularity": rectangularity,
        },
        "built": {
            "run": built_run,
            "source": geometry_source,
            "rectangularity": rectangularity,
        },
        "anchor_candidates": [],
        "selection": None,
        "verdict": {
            "status": "unverified",
            "delta_degrees_mod_90": None,
            "tolerance_degrees": float(yaw_tol_deg),
            "reason": "not_evaluated",
        },
    }

    for candidate in wall_candidates or []:
        normalized = dict(candidate)
        # Enforce the authority boundary here as well as in wall_yaw_candidates().
        # Cached/tool-provided candidate records may predate that normalizer; a hard
        # PERPENDICULAR contact must never silently become a hard POSE yaw constraint.
        if (
            normalized.get("relationship") == "perpendicular"
            and normalized.get("enforcement") == "hard"
        ):
            normalized.setdefault("relationship_enforcement", "hard")
            normalized["enforcement"] = "advisory"
            normalized["reason"] = (
                "PERPENDICULAR contact does not pin main-support yaw"
            )
        record["anchor_candidates"].append(normalized)
    border = dict(photo_border_evidence or {})
    border_edges = [str(x) for x in (border.get("border_touch_edges") or [])]
    truncated = bool(border.get("is_truncated") or border_edges)
    photo_metrics = {
        "eccentricity": ecc,
        "border_touch_edges": border_edges,
        "border_touch_pixel_count": border.get("border_touch_pixel_count"),
        "border_touch_fraction": border.get("border_touch_fraction"),
        "mask_border_pixel_fraction": border.get("mask_border_pixel_fraction"),
        "is_truncated": truncated,
    }
    if photo_run is not None:
        confidence = (
            "strong"
            if ecc is not None and float(ecc) >= ecc_gate and not truncated
            else "weak"
            if ecc is not None and float(ecc) >= ecc_gate_no_anchor
            else "insufficient"
        )
        record["anchor_candidates"].append(
            {
                "id": "photo_mask_pca",
                "source": "photo_mask_pca",
                "surface": main_surface,
                "surface_id": None,
                "relationship_id": None,
                "relationship": None,
                "direction": photo_run,
                "confidence": confidence,
                "enforcement": (
                    "hard" if confidence == "strong" else
                    "advisory" if confidence == "weak" else "none"
                ),
                "preprocess_status": None,
                "usable": confidence != "insufficient",
                "reason": (
                    "photo mask touches image border; PCA may describe the crop"
                    if truncated and confidence == "weak"
                    else None
                    if confidence != "insufficient"
                    else "photo mask axis is too weak"
                ),
                "metrics": photo_metrics,
            }
        )
    else:
        record["anchor_candidates"].append(
            {
                "id": "photo_mask_pca",
                "source": "photo_mask_pca",
                "surface": main_surface,
                "surface_id": None,
                "relationship_id": None,
                "relationship": None,
                "direction": None,
                "confidence": "insufficient",
                "enforcement": "none",
                "preprocess_status": None,
                "usable": False,
                "reason": "photo mask run is unavailable",
                "metrics": photo_metrics,
            }
        )

    if status == "not_applicable":
        record["verdict"].update(status="not_applicable", reason=reason)
        return record
    if status != "applicable" or built_run is None:
        record["verdict"].update(
            status="unverified",
            reason=reason if status == "unknown" else "built_run_missing",
        )
        return record

    usable_walls = [
        c for c in record["anchor_candidates"]
        if c.get("source") == "measured_wall_plane"
        and c.get("usable")
        and c.get("enforcement") == "hard"
    ]
    advisory_walls = [
        c
        for c in record["anchor_candidates"]
        if c.get("source") == "measured_wall_plane"
        and c.get("usable")
        and c.get("enforcement") == "advisory"
    ]
    against = [c for c in usable_walls if c.get("relationship") == "against"]
    # Several hard AGAINST annotations must identify one compatible edge family.
    # If they do not, choosing one merely makes CONTACT demand the opposite rotation.
    conflicts = []
    for i, a in enumerate(against):
        for b in against[i + 1 :]:
            d = _yaw_delta_deg(a.get("direction"), b.get("direction"))
            if d is not None and abs(d) > 2.0 * against_yaw_tol_deg:
                conflicts.append(
                    {
                        "a": a.get("relationship_id"),
                        "b": b.get("relationship_id"),
                        "delta_degrees_mod_90": d,
                        "joint_feasibility_tolerance_degrees": (
                            2.0 * against_yaw_tol_deg
                        ),
                    }
                )
    if conflicts:
        record["selection"] = {
            "candidate_id": None,
            "source": None,
            "surface": None,
            "relationship": "against",
            "confidence": "strong",
            "enforcement": "hard",
            "reason": "conflicting_hard_against_anchors",
            "conflicts": conflicts,
        }
        record["verdict"].update(
            status="fail", reason="conflicting_hard_against_anchors"
        )
        return record

    strong_photo = next(
        (
            c for c in record["anchor_candidates"]
            if c.get("id") == "photo_mask_pca" and c.get("confidence") == "strong"
        ),
        None,
    )
    weak_photo = next(
        (
            c for c in record["anchor_candidates"]
            if c.get("id") == "photo_mask_pca" and c.get("confidence") == "weak"
        ),
        None,
    )
    if against:
        ranked_against = sorted(
            against,
            key=lambda c: (
                -float((c.get("metrics") or {}).get("plane_inlier_fraction") or 0.0),
                str(c.get("relationship_id")),
            ),
        )
        selected = ranked_against[0]
        if len(ranked_against) > 1:
            # A compatible multi-AGAINST set has a small shared feasible band.
            # Target its centre so CONTACT can satisfy every +/-3deg constraint;
            # choosing one endpoint can still alternate on the other wall forever.
            import math

            base = selected.get("direction")
            deltas = [
                _yaw_delta_deg(base, c.get("direction")) for c in ranked_against
            ]
            if all(d is not None for d in deltas):
                mean_delta = sum(float(d) for d in deltas) / len(deltas)
                theta = math.atan2(float(base[1]), float(base[0])) + math.radians(
                    mean_delta
                )
                relationship_ids = [c.get("relationship_id") for c in ranked_against]
                selected = {
                    "id": "against_consensus:" + ",".join(map(str, relationship_ids)),
                    "relationship_id": None,
                    "relationship_ids": relationship_ids,
                    "source": "measured_wall_plane_consensus",
                    "surface": None,
                    "surface_ids": [c.get("surface_id") for c in ranked_against],
                    "relationship": "against",
                    "direction": [math.cos(theta), math.sin(theta), 0.0],
                    "confidence": "strong",
                    "enforcement": "hard",
                    "usable": True,
                    "reason": "centre of jointly feasible hard AGAINST wall runs",
                    "metrics": {
                        "member_count": len(ranked_against),
                        "member_delta_degrees_mod_90": deltas,
                    },
                }
                record["anchor_candidates"].append(selected)
        source = "against-wall"
        enforcement = "hard"
    elif strong_photo:
        selected, source, enforcement = strong_photo, "pca", "hard"
    elif usable_walls:
        selected = sorted(
            usable_walls,
            key=lambda c: (
                -float((c.get("metrics") or {}).get("plane_inlier_fraction") or 0.0),
                str(c.get("relationship_id")),
            ),
        )[0]
        source, enforcement = "wall", "hard"
    elif advisory_walls:
        selected = sorted(
            advisory_walls,
            key=lambda c: (
                -float((c.get("metrics") or {}).get("plane_inlier_fraction") or 0.0),
                str(c.get("relationship_id")),
            ),
        )[0]
        source, enforcement = "wall", "advisory"
    elif weak_photo:
        selected, source, enforcement = weak_photo, "pca-low", "advisory"
    else:
        record["verdict"].update(status="unverified", reason="no_usable_anchor")
        return record

    delta = _yaw_delta_deg(built_run, selected.get("direction"))
    record["selection"] = {
        "candidate_id": selected.get("id"),
        "source": source,
        "surface": selected.get("surface"),
        "relationship_id": selected.get("relationship_id"),
        "relationship_ids": selected.get("relationship_ids"),
        "relationship": selected.get("relationship"),
        "confidence": selected.get("confidence"),
        "enforcement": enforcement,
        "reason": (
            "hard_against_relationship" if source == "against-wall" else
            "strong_photo_axis" if source == "pca" else
            "reliable_related_wall" if source == "wall" else
            "weak_photo_axis_without_wall"
        ),
    }
    record["verdict"].update(
        delta_degrees_mod_90=delta,
        status=(
            "unverified" if delta is None else
            "advisory" if enforcement == "advisory" and abs(delta) > yaw_tol_deg else
            "fail" if abs(delta) > yaw_tol_deg else "pass"
        ),
        reason=(
            "selected_anchor_delta_missing" if delta is None else
            "outside_tolerance" if abs(delta) > yaw_tol_deg else "within_tolerance"
        ),
    )
    return record


def bind_compiled_yaw_evidence(
    main_surface: Optional[str],
    applicability: dict,
    built_run,
    resolved_source: dict,
) -> dict:
    """Bind immutable preprocessed yaw evidence to current built-top geometry.

    This is the post-cutover runtime path.  It performs no PCA, wall selection, or
    authority promotion: ``source_decision`` is copied from the digest-validated
    initializer artifact and only the current built delta is measured here.
    """

    source = resolved_source if isinstance(resolved_source, dict) else {}
    decision = dict(source.get("source_decision") or {})
    candidates = [
        dict(candidate)
        for candidate in source.get("source_candidates", []) or []
        if isinstance(candidate, dict)
    ]
    authority = str(decision.get("authority") or "unverified")
    status = str((applicability or {}).get("status") or "unknown")
    reason = str((applicability or {}).get("reason") or "applicability_missing")
    target = decision.get("target_run_xy")
    target_run = (
        [float(target[0]), float(target[1]), 0.0]
        if isinstance(target, (list, tuple)) and len(target) >= 2
        else None
    )
    delta = _yaw_delta_deg(built_run, target_run)
    tolerance = decision.get("tolerance_degrees_mod_90")
    try:
        tolerance = float(tolerance) if tolerance is not None else 12.0
    except (TypeError, ValueError):
        tolerance = 12.0

    if status == "not_applicable":
        runtime_status = "not_applicable"
        runtime_reason = reason
    elif status != "applicable" or built_run is None:
        runtime_status = "unverified"
        runtime_reason = reason if status == "unknown" else "built_run_missing"
    elif authority == "required":
        runtime_status = (
            "unverified"
            if delta is None
            else "fail"
            if abs(delta) > tolerance
            else "pass"
        )
        runtime_reason = (
            "selected_anchor_delta_missing"
            if delta is None
            else "outside_tolerance"
            if abs(delta) > tolerance
            else "within_tolerance"
        )
    elif authority == "advisory":
        runtime_status = "advisory"
        runtime_reason = "machine_resolution_pending"
    elif authority == "conflicting":
        runtime_status = "conflicting"
        runtime_reason = "source_evidence_conflicting"
    elif authority == "not_applicable":
        runtime_status = "not_applicable"
        runtime_reason = "source_yaw_not_applicable"
    else:
        runtime_status = "unverified"
        runtime_reason = "source_yaw_unverified"

    selection = None
    if target_run is not None:
        selection = {
            "candidate_id": source.get("decision_id"),
            "candidate_ids": list(decision.get("candidate_ids") or []),
            "source": "compiled_source_yaw",
            "confidence": "strong" if authority == "required" else "weak",
            "enforcement": "hard" if authority == "required" else "advisory",
            "reason": ",".join(decision.get("reason_codes") or []),
            "direction": target_run,
        }
    return {
        "schema_version": 2,
        "main_surface": main_surface,
        "applicability": {
            "status": status,
            "reason": reason,
            "geometry_source": str(
                (applicability or {}).get("geometry_source") or "main_top_hull"
            ),
            "rectangularity": (applicability or {}).get("rectangularity"),
        },
        "built": {
            "run": built_run,
            "source": str(
                (applicability or {}).get("geometry_source") or "main_top_hull"
            ),
            "rectangularity": (applicability or {}).get("rectangularity"),
        },
        "source_candidates": candidates,
        "source_decision": decision,
        "selection": selection,
        "verdict": {
            "status": runtime_status,
            "delta_degrees_mod_90": delta,
            "tolerance_degrees": tolerance,
            "reason": runtime_reason,
        },
    }


def objects_resting_report(
    bodies: list[dict],
    supports: dict[str, str],
    float_tol: float = 0.015,
) -> tuple[bool, str]:
    """Rule: every object RESTS on SOMETHING beneath it — its AABB bottom sits within
    ``float_tol`` of the top of its support OR of any other body directly under it.
    Penetration and inboard checks cannot catch a FLOATING object (no overlap, XY fine),
    so an agent "fixing" a wall penetration by tilting/lifting an object shipped a fork
    hovering mid-air (0709_eval3_raw1). An object legitimately resting on a NON-support
    NEIGHBOR (a mug perched on a keyboard, a tool on a box) is NOT floating, so we compare
    to the highest body actually beneath it, not only its scene-graph support — otherwise
    a physics-settled object resting on a neighbor gets a false "FLOATS … lower it" demand
    that would drive it down into that neighbor. ``supports``: object body name -> its
    support's body name (surface or parent object); pairs whose support body is absent
    (name-only walls) with nothing else beneath are skipped. Returns ``(pass, message)``
    with an exact lower-by demand per genuine floater."""
    box = {b["name"]: b for b in bodies}

    def _below_top(ob):
        """Highest body top at/below ``ob``'s base that overlaps it in XY (what it
        actually rests on), or None if nothing is under it."""
        olo, ohi = ob["lo"], ob["hi"]
        best = None
        for n, b in box.items():
            if b is ob:
                continue
            if olo[0] > b["hi"][0] or b["lo"][0] > ohi[0]:
                continue  # no X footprint overlap
            if olo[1] > b["hi"][1] or b["lo"][1] > ohi[1]:
                continue  # no Y footprint overlap
            top = float(b["hi"][2])
            if top > float(olo[2]) + float_tol:
                continue  # not beneath the base (lateral or above)
            if best is None or top > best:
                best = top
        return best

    fails = []
    for obj, sup in sorted(supports.items()):
        ob = box.get(obj)
        if ob is None:
            continue
        tops = []
        sb = box.get(sup)
        if sb is not None:
            tops.append(float(sb["hi"][2]))
        below = _below_top(ob)
        if below is not None:
            tops.append(below)
        if not tops:
            continue  # support absent AND nothing beneath -> skip (name-only wall)
        gap = float(ob["lo"][2]) - max(tops)
        if gap > float_tol:
            fails.append(
                f"  - '{obj}' FLOATS {gap * 100:.1f}cm above '{sup}': lower it "
                f"with nudge_object(object='{obj}', translation=[0,0,-{gap:.3f}], "
                f"reason='resting_defect'); keep its rotation — do NOT tilt"
            )
    if not fails:
        return True, "objects resting: PASS (every object sits on its support)"
    return False, "objects resting: FAIL — floating objects:\n" + "\n".join(fails)


def coverage_report(
    rendered: dict,
    photo: dict,
    main_name: Optional[str],
    main_band: tuple[float, float] = (0.5, 2.0),
    other_band: tuple[float, float] = (0.4, 2.5),
    visible_floor: float = 0.01,
    skip_photo_below: float = 0.02,
    waive_visibility_below: float = 0.05,
    main_yaw_delta: Optional[float] = None,
    main_yaw_src: Optional[str] = None,
    main_yaw_anchor: Optional[dict] = None,
    yaw_tol_deg: float = 12.0,
    known_names: Optional[set] = None,
    required_walls: Optional[set] = None,
    attached_walls: Optional[list] = None,
    anchored_walls: Optional[set] = None,
    yaw_unverified: bool = False,
    main_form: Optional[str] = None,
    bypassed: Optional[set] = None,
) -> tuple[bool, str]:
    """POSE rule: each root surface's REFERENCE-VIEW footprint must roughly match the photo.

    ``rendered``: per build-name ``{"frac": visible pixel fraction from the reference camera,
    "cx","cy": centroid in [0,1] image coords, y from the TOP}`` (from the coverage script).
    ``photo``: same shape, measured from that surface's segmentation mask of the target photo
    (ground truth); a surface ABSENT from ``photo`` has no mask (description-only) and gets
    only the visibility floor. Bands are generous — the masks are approximate and the build is
    idealized; the rule exists to catch the gross errors every visual judge missed (a 2x
    oversized table hiding the walls, a half-size table, a wall built out of frame), not to
    pixel-match. ``main_yaw_delta`` is now supplied only for a frozen residual
    advisory source decision. Required source and AGAINST yaw are evaluated together
    by the compiled POSE constraint set, never here. There is deliberately NO centroid check: for
    a full-piece ``table`` form the rendered silhouette includes the base/pedestal while the
    photo mask covers only the top face, biasing the rendered centroid downward past the old
    15%-of-frame tolerance on a correctly placed table (0723_e2e4_accelerate_comp_abc2: 36% of
    rendered pixels were pedestal; the FAIL tug-of-warred against the L3 ``against`` flush fix
    and the agent escaped by cutting 0.18m off the table depth). Position is owned by the
    relationship rules + objects-inboard instead. ``main_yaw_src="source-advisory"``
    identifies that non-authoritative compiled decision; historical source strings
    remain accepted only for offline replay tests. Occlusion is fair by construction: the photo mask is the surface's VISIBLE
    pixels, and the rendered fraction is also post-occlusion. ``known_names``: the scene
    graph's root-surface build names — a rendered surface with NO photo mask that is also NOT
    in the graph is a HELPER the pipeline itself mandated (the invented floor under a full
    table when the photo shows none) and is EXEMPT from the visibility floor: 4 of 5
    `0703_yaw` failures were the gate demanding >=1% visibility from a floor it required the
    agent to add under a table that fills the frame bottom. A floor's shortfall is likewise
    tolerated (reported as PASS) once the main support fully passes size+yaw+centroid — the
    remaining deficit is genuine occlusion by a correctly-posed table, not a fixable error.
    An oversized ``table``-form MAIN support (both the FAIL and the small-mask CAUTION
    branches) carries a wrong-base-type hint: the initializer sometimes builds a solid
    block/pedestal under a table the photo shows as LEGGED, and that base — not the top —
    is the excess area; a bare shrink demand loops the agent instead of fixing the base.
    ``required_walls`` is the exact set of wall build names from the scene graph: each must
    exist in the build and render at least one reference-camera pixel unless its visibility
    is explicitly bypassed.  The 2% threshold skips unreliable wall plane/yaw/size evidence;
    it never erases the wall identity or silently permits zero visibility."""
    lines, ok = [], True
    main_ok = True  # does the main support fully pass (size + yaw)?
    required_walls = set(required_walls or ())
    for name in sorted(required_walls - set(rendered)):
        ok = False
        lines.append(
            f"coverage '{name}': FAIL (current registered wall was not built — create "
            f"or repair this already-registered root with execute_and_evaluate under "
            f"this exact name. build_root_surface is only for a distinct wall plane "
            f"that is missing from the current scene graph; never edit the graph file "
            f"directly)"
        )
    for name in sorted(
        rendered, key=lambda n: (n != main_name, n)
    ):  # main judged first
        r = rendered[name]
        p = photo.get(name)
        tag = f"coverage '{name}'"
        if bypassed and name in bypassed:
            # Agent-waived via the bypass tool: either the GT mask is wrong or a
            # registered wall is genuinely absent from the target. Presence in the
            # build was checked above and cannot be bypassed.
            lines.append(
                f"{tag}: PASS (agent-bypassed — coverage/visibility waived; recorded)"
            )
            continue
        is_required_wall = name in required_walls
        if is_required_wall and float(r.get("frac", 0.0)) <= 0.0:
            ok = False
            lines.append(
                f"{tag}: FAIL (renders 0 pixels from the input camera. Every listed "
                f"wall must be visible from that camera. Correct its corner, finite "
                f"extension half-line, or occlusion so it enters the corresponding "
                f"visible image region. Call render_bev to inspect where the current "
                f"walls are placed. If the target genuinely does not show this listed "
                f"wall, call bypass([\"{name}\"]) to waive visibility for that wall.)"
            )
            continue
        is_floor = name.startswith(("floor", "ground", "ceiling"))
        if p is None:  # no photo mask
            if is_required_wall:
                lines.append(
                    f"{tag}: PASS (visible, {r['frac']:.3%} of frame; registered wall has "
                    f"no usable photo mask, so detailed size grading is skipped)"
                )
            elif is_floor:
                lines.append(
                    f"{tag}: PASS (floor/ceiling support helper with no photo mask — "
                    f"visibility not required)"
                )
            elif known_names is not None and name not in known_names:
                # a helper the pipeline itself mandated — it exists for physical
                # support, not visual match
                lines.append(
                    f"{tag}: PASS (helper surface, not in the photo — visibility not required)"
                )
            elif r["frac"] >= visible_floor:
                lines.append(
                    f"{tag}: PASS (visible, {r['frac']:.0%} of frame; no photo mask)"
                )
            else:
                ok = False
                if name == main_name:
                    main_ok = False
                lines.append(
                    f"{tag}: FAIL (renders {r['frac']:.1%} of the reference frame — "
                    f"invisible from the reference camera: it is out of frame or fully "
                    f"occluded; move/extend it into view (a wall may extend UP above the "
                    f"main support's far edge), or remove it if the photo has no such surface)"
                )
            continue
        if p["frac"] < skip_photo_below:
            if is_required_wall:
                lines.append(
                    f"{tag}: PASS (visible, {r['frac']:.3%} of frame; photo mask "
                    f"too small for reliable plane, yaw, or detailed size comparison, "
                    f"{p['frac']:.1%})"
                )
            else:
                lines.append(
                    f"{tag}: PASS (photo mask too small to compare, {p['frac']:.1%})"
                )
            continue
        lo, hi = main_band if name == main_name else other_band
        g = min(2.5, max(1.0, (0.25 / p["frac"]) ** 0.5))
        lo, hi = lo / g, hi * g
        was_ok = ok
        ratio = r["frac"] / p["frac"] if p["frac"] > 0 else float("inf")
        if r["frac"] < visible_floor:
            if is_floor and main_ok:
                lines.append(
                    f"{tag}: PASS (renders {r['frac']:.1%} vs {p['frac']:.0%} in the photo — "
                    f"shortfall tolerated: the main support passes size+yaw, so the floor is "
                    f"genuinely occluded from this camera)"
                )
                continue
            if p["frac"] < waive_visibility_below:
                lines.append(
                    f"{tag}: PASS (renders {r['frac']:.1%} vs {p['frac']:.0%} in the "
                    f"photo — visibility not demanded for a sliver mask (<"
                    f"{waive_visibility_below:.0%} of frame): at some camera pitches "
                    f"its physically-correct pose projects out of frame)"
                )
                continue
            ok = False
            lines.append(
                f"{tag}: FAIL (renders {r['frac']:.1%} of the reference frame vs {p['frac']:.0%} "
                f"in the photo — out of frame or fully occluded: extend it into view (a wall may "
                f"extend UP above the main support's far edge); if the main support's own "
                f"coverage FAILs oversized, shrink THAT first — it is the occluder)"
            )
        elif ratio > hi:
            base_hint = ""
            if name == main_name and main_form == "table":
                # Wrong-base-type hint: a legged table built with a solid
                # block/pedestal under the top renders far more silhouette than
                # the photo's top-face(+thin legs) mask — the excess is the BASE,
                # and a bare shrink demand starts the shrink->objects-don't-fit
                # ->grow loop. Say so instead of only demanding a shrink.
                base_hint = (
                    "; NOTE a common cause: the photo's table is LEGGED (open air "
                    "beneath the top) but the build has a solid block/pedestal "
                    "filling that space — the excess area is then the BASE, not the "
                    "top: compare your render's underside against the photo and "
                    "rebuild the base as separate slender legs BEFORE shrinking the "
                    "top"
                )
            if name.startswith("wall"):
                ok = False
                if name in (anchored_walls or set()):
                    # A reliable wall's plane is fixed by STRUCTURE.  Normal motion
                    # would merely trade this L2 failure for an L1 failure next round.
                    lines.append(
                        f"{tag}: FAIL (covers {r['frac']:.0%} of the reference frame vs "
                        f"{p['frac']:.0%} in the photo ({ratio:.1f}x too LARGE) — keep its "
                        "canonical measured plane FIXED: do NOT move it toward/away from the "
                        "camera. Adjust only its finite in-plane width/height/extension or "
                        "recenter that extent on the measured wall point, preserving all "
                        "required contacts and the photo's visible wall span)"
                    )
                else:
                    lines.append(
                        f"{tag}: FAIL (covers {r['frac']:.0%} of the reference frame vs "
                        f"{p['frac']:.0%} in the photo ({ratio:.1f}x too LARGE) — the wall is "
                        f"too CLOSE: move it BACK (away from the camera, along its own normal); "
                        f"do NOT reduce its height below the photo's visible extent)"
                    )
            elif name == main_name and main_form == "table" and p["frac"] < 0.15:
                lines.append(
                    f"{tag}: PASS with CAUTION (covers {r['frac']:.0%} vs {p['frac']:.0%} "
                    f"in the photo, {ratio:.1f}x — but the photo mask is SMALL and may "
                    f"cover only the support's TOP FACE while the build includes the "
                    f"full piece (top + base): visually compare the FULL unit against "
                    f"the photo; shrink ONLY if the whole unit is genuinely smaller, "
                    f"and never below what the resting objects need{base_hint})"
                )
            else:
                ok = False
                attach = ""
                if name == main_name and attached_walls:
                    fixed = sorted(set(attached_walls) & set(anchored_walls or set()))
                    movable = sorted(set(attached_walls) - set(fixed))
                    notes = []
                    if fixed:
                        notes.append(
                            "keep canonical attached wall(s) "
                            f"({', '.join(fixed)}) FIXED on their measured planes and "
                            "resize/recenter/translate the support while preserving flush contact"
                        )
                    if movable:
                        if fixed:
                            notes.append(
                                f"move only unanchored attached wall(s) ({', '.join(movable)}) "
                                "with it so their relationships keep holding"
                            )
                        else:
                            notes.append(
                                f"move its attached wall(s) ({', '.join(movable)}) together "
                                "with it so their relationships keep holding"
                            )
                    attach = "; " + "; ".join(notes)
                lines.append(
                    f"{tag}: FAIL (covers {r['frac']:.0%} of the reference frame vs {p['frac']:.0%} "
                    f"in the photo ({ratio:.1f}x too LARGE) — shrink its footprint ~x{1 / ratio:.2f} "
                    f"per side and/or move it back, keeping every object on it{attach}{base_hint})"
                )
        elif ratio < lo:
            if is_floor and main_ok:
                lines.append(
                    f"{tag}: PASS (covers {r['frac']:.0%} vs {p['frac']:.0%} in the photo — "
                    f"shortfall tolerated: the main support passes size+yaw, so the floor is "
                    f"genuinely occluded from this camera)"
                )
                continue
            ok = False
            lines.append(
                f"{tag}: FAIL (covers {r['frac']:.0%} of the reference frame vs {p['frac']:.0%} "
                f"in the photo ({ratio:.1f}x too SMALL) — enlarge its footprint ~x{1 / ratio:.2f})"
            )
        else:
            if (
                name == main_name
                and main_yaw_delta is not None
                and abs(main_yaw_delta) > yaw_tol_deg
                and main_yaw_src != "pca-low"  # weak anchor -> CAUTION below, not FAIL
                and (main_yaw_anchor or {}).get("enforcement") != "advisory"
            ):
                ok = False
                main_ok = False
                anchor_desc = "the photo's table"
                if main_yaw_src == "against-wall":
                    ids = (main_yaw_anchor or {}).get("relationship_ids") or []
                    surface = (main_yaw_anchor or {}).get("surface")
                    relation = (main_yaw_anchor or {}).get("relationship_id")
                    if ids:
                        anchor_desc = (
                            "the consensus canonical runs of hard AGAINST relationships "
                            + ", ".join(map(str, ids))
                        )
                    elif surface:
                        anchor_desc = (
                            f"the canonical measured run of '{surface}'"
                            + (f" (AGAINST {relation})" if relation else "")
                        )
                elif main_yaw_src == "wall" and (main_yaw_anchor or {}).get("surface"):
                    anchor_desc = (
                        f"the canonical measured run of "
                        f"'{(main_yaw_anchor or {}).get('surface')}'"
                    )
                lines.append(
                    f"{tag}: FAIL (size OK, {ratio:.1f}x — but its edge runs "
                    f"{main_yaw_delta:+.0f}° off {anchor_desc}: rotate '{name}' by "
                    f"{main_yaw_delta:+.0f}° around the world Z (up) axis about its own centre "
                    f"(objects resting on it do NOT rotate with it — momentary overhang is "
                    f"EXPECTED, do NOT undo; re-run the check after rotating))"
                )
                continue
            unv = ""
            if name == main_name and main_yaw_delta is None and yaw_unverified:
                unv = (
                    "; yaw UNVERIFIED — the structured source resolver selected no "
                    "numeric target; follow the emitted advisory_requirement/manual-review "
                    "state rather than inventing or averaging a rotation hint"
                )
            elif (
                name == main_name
                and main_yaw_delta is not None
                and main_yaw_src == "source-advisory"
                and (main_yaw_anchor or {}).get("enforcement") == "advisory"
            ):
                unv = (
                    f"; yaw ADVISORY — the frozen source resolver estimates a "
                    f"{main_yaw_delta:+.0f}deg delta, but did not grant hard "
                    "authority; complete the machine-verifiable yaw resolution"
                )
            elif (
                name == main_name
                and main_yaw_delta is not None
                and main_yaw_src == "wall"
                and (main_yaw_anchor or {}).get("enforcement") == "advisory"
            ):
                surface = (main_yaw_anchor or {}).get("surface") or "related wall"
                relationship = (
                    (main_yaw_anchor or {}).get("relationship") or "relationship"
                ).upper()
                unv = (
                    f"; yaw ADVISORY — its edge is {main_yaw_delta:+.0f}° from the "
                    f"reliable measured run of '{surface}', but REQUIRED {relationship} "
                    "does not pin main-support yaw: visually prefer the support edges in "
                    "the photo; do not rotate solely to satisfy this advisory"
                )
            elif (
                name == main_name
                and main_yaw_delta is not None
                and main_yaw_src == "pca-low"
                and abs(main_yaw_delta) > yaw_tol_deg
            ):
                d2 = main_yaw_delta - 90.0 if main_yaw_delta > 0 else main_yaw_delta + 90.0
                unv = (
                    f"; CAUTION — weak photo evidence suggests its edge MAY run "
                    f"{main_yaw_delta:+.0f}° off the photo's table. This number is the "
                    f"mask footprint's dominant axis folded mod 90 — rotating the FULL "
                    f"{main_yaw_delta:+.0f}° OR the FULL {d2:+.0f}° would each align "
                    f"with it — and on a corner-on/near-square table that axis can be "
                    f"the mask's DIAGONAL (up to ~45° biased), so it may keep printing "
                    f"even when your yaw is correct. Resolve it ONCE per attempt: "
                    f"(a) if you have measured TWO NON-PARALLEL visible table edges' "
                    f"image slopes (or one edge plus a corner position) in your render "
                    f"and in the photo, endpoints stated, each agreeing within ~3°, "
                    f"cite them — resolved, no probe (ONE edge alone settles nothing: "
                    f"a single slope can match at a wrong yaw); otherwise (b) probe "
                    f"by rotating the FULL {main_yaw_delta:+.0f}° or FULL {d2:+.0f}° "
                    f"(whichever is smaller in magnitude) — NEVER a partial step: a "
                    f"midpoint between fold-equivalent poses always looks worse and "
                    f"proves nothing — and keep the better pose (undo_last_step "
                    f"otherwise). Once resolved, later repeats of this caution in this "
                    f"attempt are already answered — do not re-probe; state the "
                    f"resolution once before ending"
                )
            lines.append(
                f"{tag}: PASS ({r['frac']:.0%} vs photo {p['frac']:.0%}, {ratio:.1f}x{unv})"
            )
        if name == main_name and ok != was_ok:
            main_ok = False
    if not lines:
        return True, "coverage: PASS (nothing to compare)"
    return ok, "\n".join(lines)


def _point_margin_in_convex_poly(px: float, py: float, poly: list) -> float:
    """Signed clearance from a point to a convex polygon's boundary.

    Positive is inboard, zero is on an edge, and negative is outside.  The polygon
    may be clockwise or counter-clockwise.  For a convex support, requiring every
    object-footprint vertex to have clearance >= M is exactly containment in the
    support footprint eroded inward by M.
    """
    if len(poly) < 3:
        return float("-inf")
    area2 = sum(
        float(poly[i][0]) * float(poly[(i + 1) % len(poly)][1])
        - float(poly[(i + 1) % len(poly)][0]) * float(poly[i][1])
        for i in range(len(poly))
    )
    if abs(area2) <= 1e-12:
        return float("-inf")
    orient = 1.0 if area2 > 0.0 else -1.0
    clearance = float("inf")
    for i in range(len(poly)):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = (
            float(poly[(i + 1) % len(poly)][0]),
            float(poly[(i + 1) % len(poly)][1]),
        )
        vx, vy = bx - ax, by - ay
        edge_len = (vx * vx + vy * vy) ** 0.5
        if edge_len <= 1e-12:
            continue
        cross = vx * (py - ay) - vy * (px - ax)
        clearance = min(clearance, orient * cross / edge_len)
    return clearance


def directional_pull_depth(
    spans_entry: Optional[dict], mover: str, other: str, out_xy
) -> Optional[float]:
    """Exact pull distance for "PULL ``mover`` out of ``other`` along ``out_xy``".

    ``spans_entry`` is the penetration dump's per-pair projection record along
    ``other``'s plane normal: ``{"n": [nx,ny,nz], mover: [lo,hi], other: [lo,hi]}``.
    Sign-resolve the normal against the caller's outward direction, then the distance
    is how far ``mover`` must translate along it until its lowest projection clears
    ``other``'s highest — exact for convex bodies, immune to the AABB inflation that
    made a -30° desk pull 0.20m for a 0.053m true penetration (0709_oc3_gpt1)."""
    if not spans_entry or mover not in spans_entry or other not in spans_entry:
        return None
    n = spans_entry.get("n")
    if not n or out_xy is None:
        return None
    sgn = 1.0 if (n[0] * out_xy[0] + n[1] * out_xy[1]) >= 0 else -1.0
    lo_mover = min(sgn * x for x in spans_entry[mover])
    hi_other = max(sgn * x for x in spans_entry[other])
    return max(hi_other - lo_mover, 0.0)


INBOARD_FRAC_CONTAINED = 0.97
INBOARD_FRAC_OVERHANG_OK = 0.65


def _poly_area(poly: list) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _containment_fraction(subject: list, clip: list) -> float:
    """Area fraction of convex polygon ``subject`` inside convex polygon ``clip``
    (Sutherland-Hodgman clip + shoelace). Degenerate subject -> 1.0."""
    import math

    area = _poly_area(subject)
    if area < 1e-12 or len(clip) < 3:
        return 1.0
    cx = sum(float(p[0]) for p in clip) / len(clip)
    cy = sum(float(p[1]) for p in clip) / len(clip)
    clip_ccw = sorted(clip, key=lambda p: math.atan2(float(p[1]) - cy, float(p[0]) - cx))
    out = [[float(p[0]), float(p[1])] for p in subject]
    n = len(clip_ccw)
    for i in range(n):
        ax, ay = float(clip_ccw[i][0]), float(clip_ccw[i][1])
        bx, by = float(clip_ccw[(i + 1) % n][0]), float(clip_ccw[(i + 1) % n][1])
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            px, py = inp[j]
            qx, qy = inp[(j + 1) % len(inp)]
            ps = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            qs = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
            if ps >= 0:
                out.append([px, py])
            if (ps >= 0) != (qs >= 0):
                t = ps / (ps - qs)
                out.append([px + t * (qx - px), py + t * (qy - py)])
    return min(1.0, _poly_area(out) / area) if out else 0.0


def objects_on_direct_supports_report(
    bodies: list[dict],
    supports: dict[str, str],
    margin: float = 0.02,
    support_hulls: Optional[dict] = None,
    object_hulls: Optional[dict] = None,
) -> tuple[bool, str]:
    """POSE sub-rule (both harnesses): cover each current object by its direct support.

    ``supports`` is the
    executor's authoritative direct scene-graph body map; this function deliberately
    does not walk ancestors. Root-surface supports use their emitted top hull, object
    supports use their projected object hull, and either falls back to its body AABB.
    Missing, malformed, self, or absent support bindings fail closed.
    """
    if not isinstance(supports, dict):
        return False, (
            "objects on declared direct supports: UNVERIFIED — the authenticated "
            "direct-support body map is unavailable"
        )

    body_by_name: dict[str, dict] = {}
    duplicate_names: set[str] = set()
    for body in bodies:
        name = body.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in body_by_name:
            duplicate_names.add(name)
        body_by_name[name] = body

    binding_errors: list[str] = []
    if duplicate_names:
        binding_errors.append(
            "duplicate penetration bodies: " + ", ".join(sorted(duplicate_names))
        )
    routed: list[tuple[dict, str, dict]] = []
    for body in bodies:
        if not body.get("is_obj"):
            continue
        name = body.get("name")
        if not isinstance(name, str) or not name:
            binding_errors.append("an object body has no valid name")
            continue
        if name not in supports:
            binding_errors.append(f"'{name}' has no direct support mapping")
            continue
        support = supports[name]
        if not isinstance(support, str) or not support:
            binding_errors.append(f"'{name}' has an invalid direct support {support!r}")
            continue
        if support == name:
            binding_errors.append(f"'{name}' declares itself as its direct support")
            continue
        support_body = body_by_name.get(support)
        if support_body is None:
            binding_errors.append(
                f"'{name}' direct support '{support}' is not a current penetration body"
            )
            continue
        routed.append((body, support, support_body))
    if binding_errors:
        return False, (
            "objects on declared direct supports: UNVERIFIED — "
            + "; ".join(binding_errors)
            + ". Restore the exact current scene-graph support/body binding; do not "
            "silently substitute the main support or collapse to an ancestor."
        )

    def aabb_xy(body: dict) -> list[list[float]]:
        return [
            [float(body["lo"][0]), float(body["lo"][1])],
            [float(body["hi"][0]), float(body["lo"][1])],
            [float(body["hi"][0]), float(body["hi"][1])],
            [float(body["lo"][0]), float(body["hi"][1])],
        ]

    failures: dict[str, tuple[str, float, float, bool]] = {}
    tolerated: dict[str, tuple[str, float]] = {}
    for body, support, support_body in routed:
        name = body["name"]
        footprint = (object_hulls or {}).get(name)
        if not footprint or len(footprint) < 3:
            footprint = aabb_xy(body)
        on_object = bool(support_body.get("is_obj"))
        hulls = object_hulls if on_object else support_hulls
        support_poly = (hulls or {}).get(support)
        if not support_poly or len(support_poly) < 3:
            support_poly = aabb_xy(support_body)
        clearance = min(
            _point_margin_in_convex_poly(float(point[0]), float(point[1]), support_poly)
            for point in footprint
        )
        fraction = _containment_fraction(footprint, support_poly)
        if on_object:
            if fraction >= INBOARD_FRAC_OVERHANG_OK:
                if fraction < INBOARD_FRAC_CONTAINED:
                    tolerated[name] = (support, fraction)
            else:
                failures[name] = (support, clearance, fraction, True)
            continue
        if fraction >= INBOARD_FRAC_CONTAINED:
            if clearance + 1e-9 < margin:
                failures[name] = (support, clearance, fraction, False)
        elif fraction >= INBOARD_FRAC_OVERHANG_OK:
            tolerated[name] = (support, fraction)
        else:
            failures[name] = (support, clearance, fraction, False)

    if not failures:
        note = ""
        if tolerated:
            note = " NOTE — " + "; ".join(
                f"{name} on '{support}': partial overhang tolerated ({fraction:.0%} "
                "of its footprint on that direct support; verify the PHOTO shows it "
                "overhanging — do NOT enlarge the support past the photo's edge to "
                "swallow it)"
                for name, (support, fraction) in sorted(tolerated.items())
            )
        return True, (
            "objects on declared direct supports: PASS (each object keeps "
            f"{margin * 100:.1f}cm from the edge of its root-surface support or has "
            f">={INBOARD_FRAC_OVERHANG_OK:.0%} of its footprint on its object support, "
            f"or is a tolerated partial overhang).{note}"
        )

    details = "; ".join(
        f"{name} on '{support}' (clearance {clearance * 100:+.1f}cm, "
        f"{fraction:.0%} of footprint on that direct support; "
        + (
            f"it needs >={INBOARD_FRAC_OVERHANG_OK:.0%} of its footprint on that object "
            "support)"
            if on_object
            else f"an object fully on a root surface must keep {margin * 100:.1f}cm from "
            f"the surface's edge, overhangs need >={INBOARD_FRAC_OVERHANG_OK:.0%} on it)"
        )
        for name, (support, clearance, fraction, on_object) in sorted(failures.items())
    )
    return False, (
        f"objects on declared direct supports: FAIL — {details}. Correct only the "
        "object or its exact direct support when paired target/render evidence proves "
        "which is wrong; never reparent it to the main support, substitute another "
        "support, or collapse the check to an ancestor merely to pass this gate."
    )


def generate_coverage_script(output_path: str, res: int = 320) -> str:
    """READ-ONLY script: flat per-surface id render through the REAL reference camera, then
    per-root-surface visible pixel fraction + centroid (image coords, y from the TOP) written
    to ``output_path`` as ``{name: {frac, cx, cy}}``. Workbench OBJECT-color (the render_bev
    machinery pointed at the scene camera); objects render dim-gray so they occlude surfaces
    exactly as they do in the photo masks. Nothing is saved — engine/color changes die with
    the process."""
    return f'''import bpy
import json
import os

{_PIPELINE_EMPTY_ROOTS_SCRIPT}

scene = bpy.context.scene
cam = scene.camera
result = {{}}
if cam is not None:
    _orig_engine = scene.render.engine  # restored for the BEAUTY pass below
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sh.show_object_outline = False
    sh.show_cavity = False
    sh.show_shadows = False
    scene.display.render_aa = "OFF"  # exact id colors, no edge blending
    surf_ids = {{}}
    id_overflow = []  # gradeable surfaces past the 9 available id slots (F5)
    i = 0
    _orig_colors = {{}}  # restored before the BEAUTY pass (see below)
    for o in scene.objects:
        if o.type != "MESH":
            continue
        _orig_colors[o.name] = tuple(o.color)
        if o.name.startswith("obj_") or o in _pipeline_part_owner:
            # Every mesh primitive below one exact pipeline-declared Empty root is
            # part of that one physical object.  Paint all of it object-dark so no
            # arbitrary child name consumes a gradeable root-surface id.
            o.color = (0.15, 0.15, 0.15, 1.0)
        elif o.parent is not None and o.parent.type == "MESH" and not o.parent.name.startswith("obj_"):
            o.color = (0.15, 0.15, 0.15, 1.0)
        elif i < 9:
            # id-channel encoding: red levels spaced 28/255 apart so the sRGB write/read
            # round-trip (+ 8-bit quantization) can never shift a pixel to a neighbour id
            i += 1
            o.color = (i * 28.0 / 255.0, 0.0, 1.0, 1.0)
            surf_ids[i] = o.name
        else:
            o.color = (0.15, 0.15, 0.15, 1.0)
            id_overflow.append(o.name)
    # keep the scene's aspect ratio (a square render would change the FOV and skew the
    # fractions against the photo masks)
    ar = scene.render.resolution_y / max(1, scene.render.resolution_x)
    scene.render.resolution_x = {int(res)}
    scene.render.resolution_y = max(1, round({int(res)} * ar))
    scene.render.resolution_percentage = 100
    # EXR is scene-referred: the id colors land in the file EXACTLY (a PNG goes through the
    # display/view transform, which warps the spaced red levels and scrambles the ids).
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = "{output_path}.exr"
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load("{output_path}.exr")
    img.colorspace_settings.name = "Non-Color"  # raw readback, no conversion
    w, h = img.size
    px = list(img.pixels)  # RGBA floats, row 0 = image BOTTOM
    n_ch = len(px) // (w * h)
    counts = {{}}
    sums = {{}}
    for y in range(h):
        for x in range(w):
            k = (y * w + x) * n_ch
            if px[k + 2] < 0.5:  # blue channel tags surfaces; objects/world have b<0.5
                continue
            sid = round(px[k] * 255.0 / 28.0)
            if sid in surf_ids:
                counts[sid] = counts.get(sid, 0) + 1
                s = sums.setdefault(sid, [0.0, 0.0])
                s[0] += x
                s[1] += (h - 1 - y)  # flip: report y from the TOP like the photo masks
    total = float(w * h)
    for sid, name in surf_ids.items():
        c = counts.get(sid, 0)
        if c:
            result[name] = {{"frac": c / total,
                             "cx": sums[sid][0] / c / w, "cy": sums[sid][1] / c / h}}
        else:
            result[name] = {{"frac": 0.0, "cx": 0.5, "cy": 0.5}}
    # per-surface id map dump (best-effort): the coverage-mismatch visual tints
    # each surface's RENDERED silhouette next to its photo GT mask so the agent
    # can judge whether the GT mask itself is wrong (the bypass-tool flow).
    try:
        import numpy as _np
        arr = _np.array(px, dtype=_np.float32).reshape(h, w, n_ch)
        arr = arr[::-1]  # row 0 at the TOP, like the photo masks
        surf = arr[:, :, 2] >= 0.5
        sid_arr = _np.rint(arr[:, :, 0] * 255.0 / 28.0).astype(_np.int32)
        ids = _np.where(surf, sid_arr, 0).astype(_np.uint8)
        objects = (~surf) & (arr[:, :, 0] > 0.05)  # dim-gray objects; world is ~0
        names = _np.array([surf_ids.get(i, "") for i in range(10)], dtype=object)
        _np.savez("{output_path}_ids.npz", ids=ids, names=names, objects=objects)
        for _o in scene.objects:
            if _o.type == "MESH" and _o.name in _orig_colors:
                _o.color = _orig_colors[_o.name]
        sh.light = "STUDIO"
        sh.show_shadows = True
        sh.color_type = "TEXTURE"
        scene.display.render_aa = "FXAA"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.filepath = "{output_path}_beauty.png"
        bpy.ops.render.render(write_still=True)
    except Exception:
        pass

if id_overflow:
    result["_id_overflow"] = id_overflow  # popped by _rule_coverage before grading

with open("{output_path}", "w") as f:
    json.dump(result, f)
'''


def generate_penetration_script(
    output_path: str,
    main_name: Optional[str] = None,
    eps: float = CONTACT_TOLERANCE_M,
    sections: str = "full",
) -> str:
    ""
    head = f'''import bpy
import hashlib
import json
import os
import sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

_pipeline_names = json.loads(os.environ.get("GRASE_PIPELINE_OBJECT_NAMES", "[]"))
_pipeline_empty_root_names = set()
for _pipeline_name in _pipeline_names:
    _pipeline_root = bpy.data.objects.get(_pipeline_name)
    if (
        _pipeline_root is None
        or _pipeline_root.type != "EMPTY"
        or _pipeline_root.parent is not None
        or not isinstance(_pipeline_root.get("grase_graph_id"), str)
        or not _pipeline_root.get("grase_graph_id").strip()
    ):
        continue
    _pipeline_descendants = tuple(_pipeline_root.children_recursive)
    if _pipeline_descendants and all(
        child.type == "MESH" for child in _pipeline_descendants
    ):
        _pipeline_empty_root_names.add(_pipeline_name)


def root_key(o):
    p = o
    while p.parent is not None:
        p = p.parent
    return p.name


# Group mesh objects (with faces, render-visible) by root ancestor so an object's own
# parts count as one body and never self-report.
groups = {{}}
for o in bpy.data.objects:
    if o.type != "MESH" or len(o.data.polygons) == 0 or o.hide_render:
        continue
    groups.setdefault(root_key(o), []).append(o)


def world_aabb(objs):
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for k in range(3):
                lo[k] = min(lo[k], w[k])
                hi[k] = max(hi[k], w[k])
    return lo, hi


def world_verts_polys(objs):
    verts, polys, base = [], [], 0
    for o in objs:
        mw = o.matrix_world
        verts.extend([mw @ v.co for v in o.data.vertices])
        for p in o.data.polygons:
            polys.append([base + i for i in p.vertices])
        base += len(o.data.vertices)
    return verts, polys


def aabb_overlap(a, b, eps=1e-6):
    return all(a[0][i] <= b[1][i] + eps and b[0][i] <= a[1][i] + eps for i in range(3))


keys = list(groups.keys())
trees, boxes, label, verts, polys = {{}}, {{}}, {{}}, {{}}, {{}}
for k in keys:
    label[k] = (
        k
        if k in _pipeline_empty_root_names
        else min((o.name for o in groups[k]), key=len)
    )
    boxes[k] = world_aabb(groups[k])
    v, p = world_verts_polys(groups[k])
    verts[k] = v
    polys[k] = p
    trees[k] = BVHTree.FromPolygons(v, p, all_triangles=False)


def _is_obj(k):
    return k in _pipeline_empty_root_names or label[k].startswith("obj_")


def _small_clearance_candidates(ka, kb):
    _MTD_CAP = 0.05
    mover, other = (ka, kb) if len(verts[ka]) <= len(verts[kb]) else (kb, ka)
    if not _is_obj(mover):
        if not _is_obj(other):
            return []
        mover, other = other, mover
    vm, pm, tree_o = verts[mover], polys[mover], trees[other]

    def _hits(shift):
        return BVHTree.FromPolygons([v + shift for v in vm], pm, all_triangles=False).overlap(tree_o)

    cm = sum(vm, Vector()) / len(vm)
    co = sum(verts[other], Vector()) / len(verts[other])
    ax = cm - co
    axh = Vector((ax.x, ax.y, 0.0))
    dirs = [Vector(d) for d in ((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
    for v in (ax, -ax, axh, -axh):
        if v.length > 1e-9:
            dirs.append(v.normalized())
    result = []
    for u in dirs:
        if not _hits(u * {eps}):
            return [{{"object": label[mover], "translation": list(u * {eps}),
                     "direction": list(u), "distance": {eps}}}]
        if _hits(u * _MTD_CAP):
            continue
        lo, hi = {eps}, _MTD_CAP
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if _hits(u * mid):
                lo = mid
            else:
                hi = mid
        result.append({{"object": label[mover], "translation": list(u * hi),
                        "direction": list(u), "distance": hi}})
    return sorted(result, key=lambda row: row["distance"])[:3]


_NEST_XY_FRAC = 0.9      # mover footprint (AABB) share inside the other's footprint
_NEST_RISE_M = 0.010     # the other must rise this far above the mover's base


def _nested(ka, kb):
    def _fits(mover, other):
        if not _is_obj(mover):
            return False
        (mlo, mhi), (olo, ohi) = boxes[mover], boxes[other]
        ax = max(0.0, min(mhi[0], ohi[0]) - max(mlo[0], olo[0]))
        ay = max(0.0, min(mhi[1], ohi[1]) - max(mlo[1], olo[1]))
        area = max((mhi[0] - mlo[0]) * (mhi[1] - mlo[1]), 1e-12)
        if ax * ay / area < _NEST_XY_FRAC:
            return False
        return not (olo[2] > mlo[2] + {eps} or ohi[2] < mlo[2] + _NEST_RISE_M)

    def _area(k):
        return (boxes[k][1][0] - boxes[k][0][0]) * (boxes[k][1][1] - boxes[k][0][1])

    fits_ab, fits_ba = _fits(ka, kb), _fits(kb, ka)
    if fits_ab and (not fits_ba or _area(ka) <= _area(kb)):
        return ka, kb
    if fits_ba:
        return kb, ka
    return None, None


def _inside_depth(mover, other):
    # Max distance of a mover vertex INSIDE the other body's solid. Candidates are the
    # mover's vertices inside the other's AABB; "inside" is the nearest-surface test
    # (the nearest face's normal points AWAY from the point), which is robust enough
    # for the near-closed SAM3D / authored unions and costs one BVH query per vertex.
    # Well defined for a body nested in a concave container, where a separating
    # translation is not. Deep burials read at least the distance to the nearest
    # surface, i.e. far over any class tolerance.
    olo, ohi = boxes[other]
    tree = trees[other]
    best = 0.0
    for v in verts[mover]:
        if not (olo[0] <= v.x <= ohi[0] and olo[1] <= v.y <= ohi[1] and olo[2] <= v.z <= ohi[2]):
            continue
        loc, nrm, _idx, dist = tree.find_nearest(v)
        if loc is None or dist <= best:
            continue
        if nrm.dot(v - loc) < 0.0:
            best = float(dist)
    return best


def _fit_normal(objs):
    # PCA plane normal (smallest-variance axis) over a surface body's world verts, for the
    # caller's relationship rules (face-angle of 'against', etc). None if too few verts.
    import numpy as np
    pts = []
    for o in objs:
        mw = o.matrix_world
        pts.extend([(mw @ v.co)[:] for v in o.data.vertices])
    if len(pts) < 3:
        return None
    P = np.asarray(pts, float)
    if len(P) > 5000:                        # subsample big meshes for speed
        P = P[np.linspace(0, len(P) - 1, 5000).astype(int)]
    P = P - P.mean(0)
    n = np.linalg.svd(P, full_matrices=False)[2][-1]   # smallest singular vector = normal
    ln = float(np.linalg.norm(n)) or 1.0
    return [float(n[0] / ln), float(n[1] / ln), float(n[2] / ln)]


pairs = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        ka, kb = keys[i], keys[j]
        if not aabb_overlap(boxes[ka], boxes[kb]):
            continue
        # Allow a SMALL intersection: the overlap extent in the thinnest axis approximates
        # the depth needed to separate the bodies; below {eps} m it is a shallow clip, skip.
        ext = [min(boxes[ka][1][d], boxes[kb][1][d]) - max(boxes[ka][0][d], boxes[kb][0][d])
               for d in range(3)]
        depth = min(ext)
        if depth < {eps}:
            continue
        if not trees[ka].overlap(trees[kb]):
            continue
        nested_mover, nested_other = _nested(ka, kb)
        depth_source = None
        if nested_mover is not None:
            depth = _inside_depth(nested_mover, nested_other)
            if depth <= {eps}:
                continue
            repairs = []
            depth_source = "inside_depth"
        else:
            repairs = _small_clearance_candidates(ka, kb) if (_is_obj(ka) or _is_obj(kb)) else []
            measured_clearance = min((r["distance"] for r in repairs), default=None)
            if measured_clearance is not None:
                if measured_clearance <= {eps}:
                    continue
                depth = measured_clearance
        # Projection spans of BOTH bodies along each SURFACE side's plane normal — the
        # caller derives the exact directional pull distance from these (the AABB
        # min-axis `depth` badly overstates for rotated bodies).
        spans = {{}}
        for ks in (ka, kb):
            if _is_obj(ks):
                continue
            n = _fit_normal(groups[ks])
            if n is None:
                continue
            ent = {{"n": n}}
            for kv in (ka, kb):
                pr = [vv[0] * n[0] + vv[1] * n[1] + vv[2] * n[2] for vv in
                      (tuple(w) for w in verts[kv])]
                ent[label[kv]] = [min(pr), max(pr)]
            spans[label[ks]] = ent
        # Emit EVERY interpenetrating pair (incl. surface<->surface) with its depth; the caller
        # reports object<->surface and object<->object always, and
        # surface<->surface only for an over-tolerance main support in the initializer.
        pairs.append((depth, {{"a": label[ka], "b": label[kb], "depth": depth,
                               "aabb_bound": min(ext),
                               "separation_direction": repairs[0]["direction"] if repairs else None,
                               "spans": spans, "suggested_repairs": repairs,
                               "nested_in": label[nested_other] if nested_mover is not None else None,
                               "depth_source": depth_source or ("mesh_mtd" if repairs else "aabb_overlap_bound")}}))
pairs.sort(key=lambda dp: -dp[0])           # deepest intersection first
pairs = [p for _, p in pairs]

'''
    rules = f'''# World AABB centre of each SURFACE body, so the caller can match it back to its scene-graph
# root node (by position) and tell whether that surface was geometrically anchored.
_surf_c = {{label[k]: [(boxes[k][0][d] + boxes[k][1][d]) / 2.0 for d in range(3)]
            for k in keys if not _is_obj(k)}}




_surf_n = {{label[k]: _fit_normal(groups[k]) for k in keys if not _is_obj(k)}}


def _canonical_surface_meshes(k):
    # Root-slab evidence must not be polluted by permitted detail children
    # (frames/glass/trim). Prefer the exact canonical mesh; older builds whose root
    # is an Empty fall back to the group so evidence remains available.
    exact = next((o for o in groups[k] if o.name == k), None)
    return [exact] if exact is not None else groups[k]


def _world_bounds_center(objs):
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    if not pts:
        return None
    return [float((min(p[d] for p in pts) + max(p[d] for p in pts)) / 2.0)
            for d in range(3)]


_surf_plane_n = {{label[k]: _fit_normal(_canonical_surface_meshes(k))
                  for k in keys if not _is_obj(k)}}
_surf_plane_c = {{label[k]: _world_bounds_center(_canonical_surface_meshes(k))
                  for k in keys if not _is_obj(k)}}


def _hull_xy(objs, top_only=False, exact=False):
    # Convex hull of the body's XY-projected world verts (monotone chain). ``top_only``
    # keeps only the body's highest near-coplanar band: a full-piece table's legs/base
    # must not inflate the top face that actually covers the objects.
    import numpy as np
    pts = []
    for o in objs:
        mw = o.matrix_world
        pts.extend([(mw @ v.co)[:] for v in o.data.vertices])
    if len(pts) < 3:
        return None
    P3 = np.asarray(pts, float)
    if top_only:
        zlo, zhi = float(P3[:, 2].min()), float(P3[:, 2].max())
        ztol = max(0.002, min(0.01, 0.02 * max(zhi - zlo, 0.0)))
        P3 = P3[P3[:, 2] >= zhi - ztol]
    P = np.unique(np.round(P3[:, :2], 6), axis=0)
    if len(P) < 3:
        return None
    if len(P) > 5000 and not exact:
        P = P[np.linspace(0, len(P) - 1, 5000).astype(int)]
    Ps = P[np.lexsort((P[:, 1], P[:, 0]))]

    def _chain(rows):
        h = []
        for p in rows:
            while len(h) >= 2 and float(np.cross(h[-1] - h[-2], p - h[-2])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    hull = _chain(list(Ps)) + _chain(list(Ps[::-1]))
    if len(hull) < 3:
        return None
    return [[float(p[0]), float(p[1])] for p in hull]


def _fit_run_info(hull):
    # Dominant HORIZONTAL edge direction via the MIN-AREA bounding rectangle of the convex hull
    # (rotating calipers). Unlike PCA this recovers edge orientation for a SQUARE top. Circular
    # and other valid non-rectangular tops have no rectangular edge family, so yaw is explicitly
    # NOT APPLICABLE. UNKNOWN is reserved for missing/degenerate geometry or a failed fit.
    import numpy as np
    if not hull:
        return {{"status": "unknown", "reason": "top_hull_missing",
                 "geometry_source": "main_top_hull", "rectangularity": None,
                 "circularity": None, "run": None}}
    H = np.asarray(hull, float)
    hull_area = 0.5 * abs(float(np.dot(H[:, 0], np.roll(H[:, 1], -1))
                                - np.dot(H[:, 1], np.roll(H[:, 0], -1))))
    if hull_area < 1e-9:
        return {{"status": "unknown", "reason": "top_hull_degenerate",
                 "geometry_source": "main_top_hull", "rectangularity": None,
                 "circularity": None, "run": None}}
    best_area, best_run = None, None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        n = float(np.hypot(e[0], e[1]))
        if n < 1e-9:
            continue
        d = e / n
        u = H @ d
        v = H @ np.array([-d[1], d[0]])
        du, dv = float(u.max() - u.min()), float(v.max() - v.min())
        if best_area is None or du * dv < best_area:
            best_area = du * dv
            best_run = d if du >= dv else np.array([-d[1], d[0]])  # the LONGER side
    if best_run is None or best_area is None or best_area < 1e-9:
        return {{"status": "unknown", "reason": "top_run_fit_failed",
                 "geometry_source": "main_top_hull", "rectangularity": None,
                 "circularity": None, "run": None}}
    rectangularity = float(hull_area / best_area)
    edges = H - np.roll(H, 1, axis=0)
    perimeter = float(np.hypot(edges[:, 0], edges[:, 1]).sum())
    circularity = float(4.0 * np.pi * hull_area / max(perimeter * perimeter, 1e-12))
    if rectangularity >= 0.9:
        return {{"status": "applicable", "reason": "edge_bearing_main_top",
                 "geometry_source": "main_top_hull", "rectangularity": rectangularity,
                 "circularity": circularity,
                 "run": [float(best_run[0]), float(best_run[1]), 0.0]}}
    if circularity >= 0.90:
        return {{"status": "not_applicable", "reason": "axisymmetric_main_top",
                 "geometry_source": "main_top_hull", "rectangularity": rectangularity,
                 "circularity": circularity, "run": None}}
    return {{"status": "not_applicable", "reason": "non_edge_bearing_main_top",
             "geometry_source": "main_top_hull", "rectangularity": rectangularity,
             "circularity": circularity, "run": None}}


def _fit_run(hull):
    return _fit_run_info(hull)["run"]


_main_name = {main_name!r}
_surf_hull = {{label[k]: _hull_xy(groups[k]) for k in keys if not _is_obj(k)}}
_surf_plane_hull = {{label[k]: _hull_xy(_canonical_surface_meshes(k), exact=True)
                     for k in keys if not _is_obj(k)}}
_surf_plane_bounds = {{
    label[k]: {{
        "lo": [float(v) for v in world_aabb(_canonical_surface_meshes(k))[0]],
        "hi": [float(v) for v in world_aabb(_canonical_surface_meshes(k))[1]],
    }}
    for k in keys if not _is_obj(k)
}}
_surf_top_hull = {{}}
for k in keys:
    if _is_obj(k):
        continue
    name = label[k]
    top = _hull_xy(groups[k], top_only=True, exact=True)
    # Never fall back to the grouped body: every relationship-derived support yaw
    # belongs to the exact TOP, including a non-main desk/counter. Legs/pedestals can
    # otherwise rotate the reported run. A missing top remains explicit UNKNOWN.
    _surf_top_hull[name] = top
_obj_hull = {{label[k]: _hull_xy(groups[k], exact=True) for k in keys if _is_obj(k)}}
_surf_run = {{k2: _fit_run(v2) for k2, v2 in _surf_hull.items()}}
_main_run_info = (
    _fit_run_info(_surf_top_hull.get(_main_name))
    if _main_name else
    {{"status": "unknown", "reason": "main_support_identity_missing",
      "geometry_source": "main_top_hull", "rectangularity": None,
      "circularity": None, "run": None}}
)
if _main_name:
    _surf_run[_main_name] = _main_run_info["run"]
# Determinant of each root surface object's WORLD transform. A negative value is a reflected
# basis even when the rendered vertices look unchanged; downstream contact/normals cannot treat
# that as simulation-ready geometry. Child wall details do not replace the root's contract.
_surf_det = {{}}
for k in keys:
    if _is_obj(k):
        continue
    root = next((o for o in groups[k] if o.name == k), None)
    if root is None:
        root = min(groups[k], key=lambda o: len(o.name))
    _surf_det[label[k]] = float(root.matrix_world.to_3x3().determinant())


'''
    tail = f'''def _plain(v):
    """Stable JSON-safe representation of common Blender RNA values."""
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        return round(v, 9)
    try:
        return [round(float(x), 9) for x in v]
    except Exception:
        return str(v)


def _material_payload(mat):
    if mat is None:
        return None
    out = {{
        "name": mat.name,
        "diffuse_color": _plain(mat.diffuse_color),
        "use_nodes": bool(mat.use_nodes),
    }}
    for key in ("metallic", "roughness", "blend_method"):
        if hasattr(mat, key):
            out[key] = _plain(getattr(mat, key))
    if mat.use_nodes and mat.node_tree:
        out["nodes"] = []
        for node in sorted(mat.node_tree.nodes, key=lambda n: n.name):
            inputs = []
            for socket in node.inputs:
                if hasattr(socket, "default_value"):
                    inputs.append([socket.name, _plain(socket.default_value)])
            image = getattr(node, "image", None)
            image_ref = None
            if image is not None:
                image_ref = [image.name, image.filepath, image.source,
                             image.colorspace_settings.name]
            out["nodes"].append([node.name, node.bl_idname, inputs, image_ref])
        out["links"] = sorted(
            [link.from_node.name, link.from_socket.name,
             link.to_node.name, link.to_socket.name]
            for link in mat.node_tree.links
        )
    return out


def _semantic_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _world_coordinate(value):
    # PoseSession.prepare recentres imported-object origins before physics.  That
    # rewrites local coordinates while preserving world geometry, with only a few
    # ulps of matrix round-trip noise.  Ten micrometres is far below any permitted
    # agent edit (millimetres) while being comfortably above that numeric residue.
    rounded = round(float(value), 5)
    return 0.0 if rounded == 0.0 else rounded


def _mesh_digests(mesh, matrix_world, world):
    """(vertex digest, topology digest, uv digests, n_verts, n_edges, n_polys) of a mesh
    via numpy: world-space vertices rounded to 5 decimals (``world=True``, matching
    ``_world_coordinate``) or raw local float32 coordinates (``world=False``)."""
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if world:
        M = np.array([list(r) for r in matrix_world], dtype=np.float64)
        pts = np.round(co @ M[:3, :3].T + M[:3, 3], 5)
        pts[pts == 0.0] = 0.0  # fold -0.0 like _world_coordinate
    else:
        pts = co.astype(np.float32)
    vertex_digest = hashlib.sha256(pts.tobytes()).hexdigest()
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int64); mesh.edges.foreach_get("vertices", ev)
    lv = np.empty(len(mesh.loops), dtype=np.int64); mesh.loops.foreach_get("vertex_index", lv)
    ls = np.empty(len(mesh.polygons), dtype=np.int64); mesh.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(mesh.polygons), dtype=np.int64); mesh.polygons.foreach_get("loop_total", lt)
    mi = np.empty(len(mesh.polygons), dtype=np.int64); mesh.polygons.foreach_get("material_index", mi)
    topo = hashlib.sha256()
    for a in (ev, lv, ls, lt, mi):
        topo.update(a.tobytes())
    uv_layers = []
    for uv in mesh.uv_layers:
        a = np.empty(len(uv.data) * 2, dtype=np.float64); uv.data.foreach_get("uv", a)
        uv_layers.append([uv.name, hashlib.sha256(np.round(a, 9).tobytes()).hexdigest()])
    return vertex_digest, topo.hexdigest(), uv_layers, n, len(mesh.edges), len(mesh.polygons)


def _world_semantics(root):
    """Canonical rendered semantics with separately auditable component digests.

    World geometry remains independent of local matrices, origins, and rotation mode,
    which PoseSession may normalise without moving a rendered vertex.  Hierarchy is
    recorded separately: exact world hashes can authorize harmless representation
    normalization, while the numerical-vertex fallback additionally requires the
    parent graph to be unchanged.  Component digests avoid embedding every vertex in
    the JSON dump.
    """
    descendants = [root] + sorted(list(root.children_recursive), key=lambda o: o.name)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geometry, topology, hierarchy = [], [], []
    materials, modifiers, visibility = [], [], []
    vertex_count = edge_count = polygon_count = mesh_count = 0
    for o in descendants:
        geometry_row = {{"name": o.name, "type": o.type}}
        topology_row = {{"name": o.name, "type": o.type}}
        if o.type == "MESH" and o.data is not None:
            evaluated = o.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                world_digest, topo_digest, uv_layers, n_verts, n_edges, n_polys = (
                    _mesh_digests(mesh, evaluated.matrix_world, world=True)
                )
            finally:
                evaluated.to_mesh_clear()
            geometry_row["world_vertices_sha256"] = world_digest
            geometry_row["vertex_count"] = n_verts
            topology_row.update(
                {{"topology_sha256": topo_digest, "uv_layers": uv_layers,
                  "counts": [n_edges, n_polys]}}
            )
            mesh_count += 1
            vertex_count += n_verts
            edge_count += n_edges
            polygon_count += n_polys
        geometry.append(geometry_row)
        topology.append(topology_row)
        hierarchy.append(
            {{
                "name": o.name,
                "type": o.type,
                "parent": o.parent.name if o.parent else None,
            }}
        )
        materials.append(
            {{
                "name": o.name,
                "slots": [_material_payload(slot.material) for slot in o.material_slots],
            }}
        )
        modifiers.append(
            {{
                "name": o.name,
                "items": sorted(
                    [modifier.name, modifier.type, bool(modifier.show_render)]
                    for modifier in o.modifiers
                ),
            }}
        )
        visibility.append(
            {{
                "name": o.name,
                "hide_render": bool(o.hide_render),
                "hide_viewport": bool(o.hide_viewport),
                "hide_get": bool(o.hide_get()),
            }}
        )
    detail = {{
        "schema_version": 1,
        "world_coordinate_decimals": 5,
        "part_count": len(descendants),
        "mesh_count": mesh_count,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "polygon_count": polygon_count,
        "geometry_sha256": _semantic_sha256(geometry),
        "topology_sha256": _semantic_sha256(topology),
        "hierarchy_sha256": _semantic_sha256(hierarchy),
        "material_sha256": _semantic_sha256(materials),
        "modifier_sha256": _semantic_sha256(modifiers),
        "visibility_sha256": _semantic_sha256(visibility),
    }}
    detail["sha256"] = _semantic_sha256(detail)
    return detail


def _object_integrity(root):
    descendants = [root] + sorted(list(root.children_recursive), key=lambda o: o.name)
    payload = []
    content_payload = []
    for o in descendants:
        row = {{
            "name": o.name,
            "type": o.type,
            "parent": o.parent.name if o.parent else None,
            # Root translation is the one authorized mutable field. Its 3x3 basis
            # still pins rotation/scale; descendants keep their full local matrix.
            "matrix_local": (
                [[round(float(x), 9) for x in r] for r in o.matrix_local.to_3x3()]
                if o is root else
                [[round(float(x), 9) for x in r] for r in o.matrix_local]
            ),
            "scale": _plain(o.scale),
            "rotation_mode": o.rotation_mode,
            "hide_render": bool(o.hide_render),
            "hide_viewport": bool(o.hide_viewport),
            "hide_get": bool(o.hide_get()),
            "materials": [_material_payload(s.material) for s in o.material_slots],
            "modifiers": sorted([m.name, m.type, bool(m.show_render)] for m in o.modifiers),
        }}
        if o.type == "MESH" and o.data is not None:
            v_digest, t_digest, uv_layers, n_v, n_e, n_p = _mesh_digests(
                o.data, o.matrix_world, world=False
            )
            row["mesh"] = {{
                "name": o.data.name,
                "vertices_sha256": v_digest,
                "topology_sha256": t_digest,
                "uv_layers": uv_layers,
                "counts": [n_v, n_e, n_p],
            }}
        payload.append(row)
        # GPT-6 typed transactions separately compare the root world matrix.  This
        # companion digest therefore omits only the root's local transform while
        # retaining its mesh/material/etc. and every descendant's full row/matrix.
        content_row = dict(row)
        if o is root:
            content_row.pop("matrix_local", None)
        content_payload.append(content_row)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    result = {{
        "name": root.name,
        "matrix": [[float(x) for x in r] for r in root.matrix_world],
        "scale": [float(x) for x in root.scale],
        "parent": root.parent.name if root.parent else None,
        "hide_render": bool(root.hide_render),
        "hide_viewport": bool(root.hide_viewport),
        "integrity_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }}
    if os.environ.get("GRASE_TYPED_WORLD_SEMANTICS") == "1":
        content_raw = json.dumps(
            content_payload, sort_keys=True, separators=(",", ":")
        )
        result["content_sha256"] = hashlib.sha256(
            content_raw.encode("utf-8")
        ).hexdigest()
        result["world_semantics"] = _world_semantics(root)
    return result


_surface_geometry = {{}}
if os.environ.get("GRASE_TYPED_WORLD_SEMANTICS") == "1":
    depsgraph = bpy.context.evaluated_depsgraph_get()
    owned = set()
    for name in _pipeline_names:
        root = bpy.data.objects.get(name)
        if root is not None:
            owned.update(o.name for o in [root, *root.children_recursive])
    # Match register's static colliders, including hidden meshes. Dynamic parts
    # belong to their prepared root even when their names lack the obj_ prefix.
    for o in sorted(bpy.data.objects, key=lambda obj: obj.name):
        if o.type != "MESH" or o.name in owned or o.name.startswith("obj_"):
            continue
        evaluated = o.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            w_digest, t_digest, _uvs, n_v, n_e, n_p = _mesh_digests(
                mesh, evaluated.matrix_world, world=True
            )
        finally:
            evaluated.to_mesh_clear()
        _surface_geometry[o.name] = _semantic_sha256([w_digest, t_digest, n_v, n_p])

_obj_integrity = {{}}
# Canonical placement roots plus any unexpected live root obj_* identity. Internal
# GLB children stay inside their canonical root's digest and are not double-counted.
_integrity_names = set(_pipeline_names)
_integrity_names.update(
    o.name for o in bpy.data.objects
    if o.name.startswith("obj_") and o.parent is None
)
for _name in sorted(_integrity_names):
    _root = bpy.data.objects.get(_name)
    if _root is not None:
        _obj_integrity[_name] = _object_integrity(_root)
# Per-body world AABB (for the caller's geometric rules, e.g. main-support-top-at-z=0).
_bodies = [{{"name": label[k], "is_obj": _is_obj(k),
             "lo": [boxes[k][0][d] for d in range(3)],
             "hi": [boxes[k][1][d] for d in range(3)]}} for k in keys]

# World AABB of each connected mesh loose-part of the MAIN SUPPORT (top slab + each leg), so the
# caller can check they form ONE connected assembly (a scale-splayed leg shows up as a stray island).
_main_islands = None
if _main_name:
    _mo = [o for o in bpy.data.objects if o.type == "MESH"
           and (o.name == _main_name or o.name.startswith(_main_name + "."))]
    if _mo:
        _main_islands = []
        for o in _mo:
            adj = {{}}
            for e in o.data.edges:
                a, b = e.vertices
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            seen = set()
            for v in range(len(o.data.vertices)):
                if v in seen:
                    continue
                stack = [v]; comp = []
                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue
                    seen.add(x); comp.append(x)
                    stack.extend(adj.get(x, set()) - seen)
                ws = [o.matrix_world @ o.data.vertices[i].co for i in comp]
                _main_islands.append([[min(w[d] for w in ws) for d in range(3)],
                                      [max(w[d] for w in ws) for d in range(3)]])

with open("{output_path}", "w") as f:
    json.dump({{"penetrating_pairs": pairs, "surface_centroids": _surf_c,
                "surface_normals": _surf_n, "surface_runs": _surf_run,
                "surface_plane_centroids": _surf_plane_c,
                "surface_plane_normals": _surf_plane_n,
                "surface_plane_hulls": _surf_plane_hull,
                "surface_plane_bounds": _surf_plane_bounds,
                "surface_determinants": _surf_det,
                "surface_hulls": _surf_hull, "surface_top_hulls": _surf_top_hull,
                "object_hulls": _obj_hull, "bodies": _bodies,
                "object_integrity": _obj_integrity,
                **({{"surface_geometry": _surface_geometry}}
                   if os.environ.get("GRASE_TYPED_WORLD_SEMANTICS") == "1" else {{}}),
                "main_support_run": _main_run_info,
                "main_support_islands": _main_islands}}, f)
print("Penetration check complete:", len(pairs), "pairs")
sys.stdout.flush()
# READ-ONLY query: exit before the shared wrapper's trailing render/save (see scene-info).
os._exit(0)
'''
    if sections == "full":
        return head + rules + tail
    if sections != "transaction":
        raise ValueError(f"unknown penetration dump sections {sections!r}")
    stub = f'''
# geometry — surface plane fits, XY hulls, runs, determinants — that only
# check_rules_enforced reads; on a 320k-vertex scene it was 5 s of an 11.6 s dump.
_main_name = {main_name!r}
_surf_c = {{}}; _surf_n = {{}}; _surf_plane_n = {{}}; _surf_plane_c = {{}}
_surf_plane_hull = {{}}; _surf_plane_bounds = {{}}; _surf_hull = {{}}
_surf_top_hull = {{}}; _obj_hull = {{}}; _surf_run = {{}}; _surf_det = {{}}
_main_run_info = {{"status": "not_computed", "reason": "transaction_dump", "run": None}}
'''
    return head + stub + tail


def generate_render_script() -> str:
    """Generate script to render the current scene.

    Renders the scene to RENDER_DIR/output.png using Cycles engine
    while preserving the scene aspect ratio with the longer side at 512.

    Returns:
        Blender Python script as a string.
    """
    return """import bpy
import os

render_dir = os.environ.get("RENDER_DIR", "/tmp")

# Basic render settings
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512

# Single render
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

print("Render completed to", bpy.context.scene.render.filepath)
"""


def generate_camera_focus_script(object_name: str, base_path: str) -> str:
    """Generate script to focus camera on a specific object.

    Creates a track-to constraint to point the camera at the target object,
    renders the scene, and saves camera info to JSON.

    Args:
        object_name: Name of the Blender object to focus on.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    object_name_literal = json.dumps(object_name)
    return f'''import bpy
import json
import math
import os
from mathutils import Vector

{_LOGICAL_TARGET_SCRIPT}

# Get target geometry (robust to scene-graph ids / unknown names).  An exact
# pipeline-bound Empty resolves to all of its mesh parts as one logical object.
target_meshes = _resolve_target_parts({object_name_literal})
if not target_meshes:
    raise ValueError("No mesh objects to focus on")

# Get camera
camera = bpy.context.scene.camera
if not camera:
    # Find first camera
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera
    else:
        raise ValueError("No camera found in scene")

# Aim at the union WORLD-SPACE BOUNDING-BOX CENTER, never matrix_world.translation:
# preprocessed obj_* meshes can have their poses baked into vertices, while a new
# procedural object's canonical Empty pivot need not be at its geometry center.
if len(target_meshes) == 1:
    target_obj = target_meshes[0]
    target_pos = sum(
        (target_obj.matrix_world @ Vector(c) for c in target_obj.bound_box), Vector()
    ) / 8
else:
    _target_lo, _target_hi, target_pos = _world_bbox(target_meshes)
camera_pos = camera.matrix_world.translation
distance = (camera_pos - target_pos).length

for c in list(camera.constraints):
    if c.type == 'TRACK_TO':
        camera.constraints.remove(c)
camera.rotation_euler = (target_pos - camera_pos).to_track_quat('-Z', 'Y').to_euler()

# Render after focus
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

# update camera info — the rotation is written directly above (no constraint), so the
# evaluated transform equals the raw one; keep the evaluated read for uniformity with
# older blends that may still carry constraints elsewhere.
_dg = bpy.context.evaluated_depsgraph_get()
_cw = camera.evaluated_get(_dg).matrix_world
camera_info = [{{
    "location": list(_cw.translation),
    "rotation": list(_cw.to_euler())
}}]
rotate_info = {{
    "radius": distance,
    "theta": math.atan2(*(camera_pos[i] - target_pos[i] for i in (1,0))),
    "phi": math.asin((camera_pos.z - target_pos.z)/distance)
}}

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)
with open(f"{base_path}/tmp/rotate_info.json", "w") as f:
    json.dump(rotate_info, f)

print("Camera focused on object and rendered")
'''


def generate_camera_set_script(
    location: list[float], rotation_euler: list[float], base_path: str
) -> str:
    """Generate script to set camera position and rotation.

    Sets the camera to a specific location and rotation, renders the scene,
    and saves camera info to JSON.

    Args:
        location: Camera location as [x, y, z] coordinates.
        rotation_euler: Camera rotation as [rx, ry, rz] Euler angles.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

# Get camera
camera = bpy.context.scene.camera
if not camera:
    # Find first camera
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera
    else:
        raise ValueError("No camera found in scene")

for _c in list(camera.constraints):
    camera.constraints.remove(_c)

# Set camera location and rotation
camera.location = {location}
camera.rotation_euler = {rotation_euler}

# Render after setting camera
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}}]

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera set to location and rotation and rendered")
'''


def generate_visibility_script(
    show_objects: list[str], hide_objects: list[str], base_path: str
) -> str:
    """Generate script to set object visibility and render.

    Sets visibility for specified objects, renders the scene, and saves
    camera info to JSON.

    Args:
        show_objects: List of object names to make visible.
        hide_objects: List of object names to hide.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

show_list = {show_objects}
hide_list = {hide_objects}

# Apply visibility changes
for obj in bpy.data.objects:
    if obj.name in hide_list:
        obj.hide_viewport = True
        obj.hide_render = True
    if obj.name in show_list:
        obj.hide_viewport = False
        obj.hide_render = False

# Render after visibility update
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(bpy.context.scene.camera.location),
    "rotation": list(bpy.context.scene.camera.rotation_euler)
}}]

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Visibility updated and rendered: show", show_list, ", hide", hide_list)
'''


def generate_camera_move_script(
    target_obj_name: str, radius: float, theta: float, phi: float, base_path: str
) -> str:
    """Generate script to move camera around a target object.

    Positions the camera in spherical coordinates relative to the target
    object, renders the scene, and saves camera info to JSON.

    Args:
        target_obj_name: Name of the object to orbit around.
        radius: Distance from the target object.
        theta: Azimuth angle in radians.
        phi: Elevation angle in radians.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    target_obj_name_literal = json.dumps(target_obj_name)
    return f'''import bpy
import json
import math
import os
from mathutils import Vector

{_LOGICAL_TARGET_SCRIPT}

# Get target geometry (robust to scene-graph ids / unknown names).  An exact
# pipeline-bound Empty resolves to all of its mesh parts as one logical object.
target_meshes = _resolve_target_parts({target_obj_name_literal})
if not target_meshes:
    raise ValueError("No mesh objects to orbit")

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Calculate new camera position around the union WORLD-SPACE BBOX CENTER (same rule
# as focus: neither a baked mesh origin nor an authored Empty pivot is the center).
if len(target_meshes) == 1:
    target_obj = target_meshes[0]
    target_pos = sum(
        (target_obj.matrix_world @ Vector(c) for c in target_obj.bound_box), Vector()
    ) / 8
else:
    _target_lo, _target_hi, target_pos = _world_bbox(target_meshes)
x = {radius} * math.cos({phi}) * math.cos({theta})
y = {radius} * math.cos({phi}) * math.sin({theta})
z = {radius} * math.sin({phi})

new_pos = Vector((target_pos.x + x, target_pos.y + y, target_pos.z + z))
camera.matrix_world.translation = new_pos
# Aim directly (the focus script no longer installs a TRACK_TO to do it for us).
for c in list(camera.constraints):
    if c.type == 'TRACK_TO':
        camera.constraints.remove(c)
camera.rotation_euler = (target_pos - new_pos).to_track_quat('-Z', 'Y').to_euler()

# Render after moving
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

# EVALUATED transform kept for uniformity; rotation is now written directly above.
_dg = bpy.context.evaluated_depsgraph_get()
_cw = camera.evaluated_get(_dg).matrix_world
camera_info = [{{
    "location": list(_cw.translation),
    "rotation": list(_cw.to_euler())
}}]

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera moved to position and rendered")
'''


def generate_keyframe_script(frame_number: int, base_path: str) -> str:
    """Generate script to set the current frame and render.

    Sets the timeline to a specific frame number, renders the scene,
    and saves camera info to JSON.

    Args:
        frame_number: Target frame number to set.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

scene = bpy.context.scene
current_frame = scene.frame_current

# Ensure frame number is within valid range
target_frame = max(scene.frame_start, min(scene.frame_end, {frame_number}))
scene.frame_set(target_frame)

# Render after frame change
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
try:
    bpy.context.scene.render.engine = render_engine
except Exception:
    bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(bpy.context.scene.camera.location),
    "rotation": list(bpy.context.scene.camera.rotation_euler)
}}]

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Changed to frame", target_frame, "(was", current_frame, ") and rendered")
'''


def generate_viewpoint_script(object_names: list[str], base_path: str) -> str:
    """Generate script to initialize viewpoints around objects.

    Creates four viewpoints around the bounding box of specified objects,
    renders from each viewpoint, and saves camera info to JSON.

    Args:
        object_names: List of object names to observe. If empty, observes
            all mesh objects except Ground and Plane.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import math
import os
from mathutils import Vector

{_LOGICAL_TARGET_SCRIPT}

object_names = {json.dumps(object_names)}
meshes = [o for o in bpy.data.objects
          if o.type == 'MESH' and o.name not in ['Ground', 'Plane']]
objects = []


def _resolve(name):
    # Exact Blender-object match first.  A bound Empty root or any one of its
    # implementation-detail children expands to the complete logical object.
    exact = bpy.data.objects.get(name)
    parts = _target_parts(exact) if (
        exact in meshes or (exact is not None and exact.name in _pipeline_empty_parts)
    ) else []
    if parts:
        return parts
    # else fuzzy-match a scene-graph id ("table#0", "tissue box#0") to Blender names
    # ("Table", "obj_tissue_box_0"): prefer a declared logical root, then meshes.
    key = name.split('#')[0].strip().lower().replace(' ', '_')
    if not key:
        return []
    hits = []
    for root_name in _pipeline_names:
        if root_name in _pipeline_empty_parts and key in root_name.lower().replace(' ', '_'):
            for part in _pipeline_empty_parts[root_name]:
                if part not in hits:
                    hits.append(part)
    for mesh in meshes:
        logical_name = _logical_target_name(mesh)
        if key not in logical_name.lower().replace(' ', '_') and key not in mesh.name.lower().replace(' ', '_'):
            continue
        for part in _target_parts(mesh):
            if part not in hits:
                hits.append(part)
    return hits


# Find objects to observe; fall back to ALL meshes if names don't resolve (e.g. the
# caller passed scene-graph ids that don't match Blender object names) so we still render.
if object_names:
    for nm in object_names:
        for o in _resolve(nm):
            if o not in objects:
                objects.append(o)
if not objects:
    objects = list(meshes)

if not objects:
    raise ValueError("No valid objects found")

# Frame the cameras on foreground content only: room-scale background planes
# (walls/floors/ceilings) blow the bounding box up to metres and push all four
# cameras so far out that everything else renders as a speck. They stay visible
# in the renders — they just don't dictate the camera distance.
_bg = ('wall', 'floor', 'ceiling')
frame_objects = [o for o in objects
                 if not _logical_target_name(o).lower().removeprefix('obj_').startswith(_bg)]
if not frame_objects:
    frame_objects = objects

# Calculate bounding box
min_x = min_y = min_z = float('inf')
max_x = max_y = max_z = float('-inf')

for obj in frame_objects:
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    for corner in bbox_corners:
        min_x = min(min_x, corner.x)
        min_y = min(min_y, corner.y)
        min_z = min(min_z, corner.z)
        max_x = max(max_x, corner.x)
        max_y = max(max_y, corner.y)
        max_z = max(max_z, corner.z)

center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

size_x = max_x - min_x
size_y = max_y - min_y
size_z = max_z - min_z
max_size = max(size_x, size_y, size_z)
margin = max_size * 0.5

camera_positions = [
    (center_x - margin, center_y - margin, center_z + margin),
    (center_x + margin, center_y - margin, center_z + margin),
    (center_x - margin, center_y + margin, center_z + margin),
    (center_x + margin, center_y + margin, center_z + margin)
]

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Store original position
original_location = camera.location.copy()
original_rotation = camera.rotation_euler.copy()
camera_infos = []

# Set up viewpoints and render each
render_dir = os.environ.get("RENDER_DIR", "/tmp")
for i, pos in enumerate(camera_positions):
    camera.location = pos
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))

    # Look at center
    direction = Vector((center_x, center_y, center_z)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    # Render per viewpoint
    render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
    try:
        bpy.context.scene.render.engine = render_engine
    except Exception:
        bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    width = max(1, bpy.context.scene.render.resolution_x)
    height = max(1, bpy.context.scene.render.resolution_y)
    if width >= height:
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
    else:
        bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
        bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.filepath = os.path.join(render_dir, str(i+1)+".png")
    bpy.ops.render.render(write_still=True)

    camera_infos.append({{
        "location": list(camera.location),
        "rotation": list(camera.rotation_euler)
    }})

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_infos, f)

# Restore original position
camera.location = original_location
camera.rotation_euler = original_rotation

print("Viewpoints initialized and rendered for", len(objects), "objects")
'''


# --------------------------------------------------------------------------- #
# Bird's-eye-view (BEV) render: an ORTHOGRAPHIC top-down flat-tint of the      #
# current scene, centred on the az/el pivot. Each root surface gets a distinct #
# flat colour (the "tinted mask"); objects render dim-gray for context. A      #
# sidecar JSON carries per-surface footprints + colours + the source camera so #
# `bev_overlay` can draw ids/legend/camera-arrow/axes on top. READ-ONLY (no    #
# save). Advertised to the initializer + composition agents to check layout    #
# (which side each wall is on vs the camera) and catch flipped surfaces.        #
# --------------------------------------------------------------------------- #
_BEV_BODY = r"""
import bpy, json, os, sys, math
from mathutils import Vector

RENDER_DIR = sys.argv[-1]
OUT_PNG = os.path.join(RENDER_DIR, "bev.png")
OUT_JSON = os.path.join(RENDER_DIR, "bev.json")
scene = bpy.context.scene

# reference (source-view) camera BEFORE we add the ortho BEV camera
_ref = scene.camera or next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
cam_pos = [round(v, 4) for v in _ref.matrix_world.translation] if _ref else None
cam_fwd = [round(v, 4) for v in (_ref.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)))] if _ref else None

meshes = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render and o.data.polygons]
def _corners(o):
    return [o.matrix_world @ Vector(c) for c in o.bound_box]

def _is_bg(name):
    return name.startswith('floor_') or name.startswith('ceiling_')   # huge background planes

px, py = float(PIVOT[0]), float(PIVOT[1])
rad, maxz = 0.0, float(PIVOT[2])
for o in meshes:
    for w in _corners(o):
        maxz = max(maxz, w.z)
        if not _is_bg(o.name):     # a very large floor/ceiling must NOT blow up the framing
            rad = max(rad, math.hypot(w.x - px, w.y - py))
size_m = float(SIZE_OVERRIDE) if SIZE_OVERRIDE else max(2.0, min(8.0, 2.0 * rad * 1.2))

PALETTE = [(0.35, 0.58, 0.98), (0.92, 0.27, 0.27), (0.27, 0.84, 0.37), (0.98, 0.75, 0.18),
           (0.70, 0.40, 0.95), (0.20, 0.80, 0.80), (0.98, 0.50, 0.75), (0.60, 0.80, 0.30)]
surfaces = sorted([o for o in meshes if not o.name.startswith('obj_')], key=lambda o: o.name)
surf_info, _pi = [], 0
for o in surfaces:
    if _is_bg(o.name):
        r, g, b = 0.16, 0.16, 0.18            # background plane -> dim, so it doesn't flood the frame
    else:
        r, g, b = PALETTE[_pi % len(PALETTE)]; _pi += 1
    o.color = (r, g, b, 1.0)
    cs = _corners(o); xs = [w.x for w in cs]; ys = [w.y for w in cs]
    surf_info.append({"name": o.name, "color": [int(r * 255), int(g * 255), int(b * 255)],
                      "aabb": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
                      "centroid": [round((min(xs) + max(xs)) / 2.0, 4), round((min(ys) + max(ys)) / 2.0, 4)]})
for o in meshes:
    if o.name.startswith('obj_'):
        o.color = (0.28, 0.28, 0.30, 1.0)   # dim-gray context

# flat, unlit top-down render via the Workbench engine (object colours, no lighting dependence)
scene.render.engine = 'BLENDER_WORKBENCH'
sh = scene.display.shading
sh.light = 'FLAT'; sh.color_type = 'OBJECT'; sh.show_object_outline = False
sh.background_type = 'VIEWPORT'; sh.background_color = (0.08, 0.08, 0.09)
scene.render.film_transparent = False

cd = bpy.data.cameras.new("BEV_CAM"); cd.type = 'ORTHO'
cd.ortho_scale = size_m; cd.clip_start = 0.01; cd.clip_end = max(100.0, maxz + 50.0)
bev = bpy.data.objects.new("BEV_CAM", cd); scene.collection.objects.link(bev)
bev.location = (px, py, maxz + 5.0)
bev.rotation_euler = (0.0, 0.0, 0.0)     # look down -Z, up +Y -> image-right = +X, image-up = +Y
scene.camera = bev
scene.render.resolution_x = RES; scene.render.resolution_y = RES; scene.render.resolution_percentage = 100
scene.render.filepath = OUT_PNG
bpy.ops.render.render(write_still=True)

json.dump({"size_m": round(size_m, 4), "pivot": [px, py, float(PIVOT[2])], "res": RES,
           "surfaces": surf_info, "camera": {"pos": cam_pos, "forward": cam_fwd}}, open(OUT_JSON, "w"))
print("BEV_RENDERED", OUT_PNG, "size_m", round(size_m, 2), len(surf_info), "surfaces")
sys.stdout.flush()
os._exit(0)   # READ-ONLY: exit before the shared wrapper's trailing render/save
"""


def generate_bev_script(pivot, size_m=None, res: int = 1024) -> str:
    """Blender script for the top-down BEV render (see module comment). ``pivot`` = the az/el
    centre (world XYZ); ``size_m`` overrides the auto-fit square side; ``res`` the square
    resolution. Reads the render dir from argv; writes ``bev.png`` + ``bev.json`` there."""
    header = (
        f"PIVOT = {[float(x) for x in pivot]!r}\n"
        f"SIZE_OVERRIDE = {float(size_m) if size_m else None!r}\n"
        f"RES = {int(res)}\n"
    )
    return header + _BEV_BODY
