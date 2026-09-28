"""Persistent Blender render/transform server for pose registration.

Loads a blend (imported SAM3D meshes + the locked MoGE camera) once, then answers
many isolate / set-pose / render commands over a newline-delimited JSON protocol on
stdin/stdout. ``PoseSession`` uses it for composition investigate/move cycles;
the retained preprocessing flip checker and composition collider-dump path share the
same server. These paths do many render/transform cycles, so the process stays warm
instead of paying a Blender launch (~5 s) for each command.

Legacy mesh objects keep their existing registration behavior: sub-parts are parented
to one mesh handle and its origin is moved to the bounds centre.  A procedural object
instead has one canonical parentless EMPTY handle with live MESH descendants.  That
captured Empty matrix/pivot is preserved, and all descendant geometry is treated as one
logical render/collider body.
``set_pose`` is a DELTA from the imported base pose (translate + euler added, scale
multiplied), so the orchestrator can try candidates and revert by re-sending.

Protocol (one JSON object per line):
  {"cmd":"prepare","objects":[...],"parents":{child:parent},
   "allow_empty_roots":bool?,
   "authored_empty_ids":{canonical_name:graph_id}?}
      -> {"ok":true,"prepared":[...],"surfaces":[...]}
  {"cmd":"isolate","visible":[...],"holdout":[...]} -> {"ok":true}
      (visible=null shows all; visible=null + "hide":[names] shows all EXCEPT those;
       holdout bodies occlude but do not contribute silhouette pixels)
  {"cmd":"ensure_light"}                              -> {"ok":true,...}
  {"cmd":"set_pose","name":..,"translate":[x,y,z],"euler":[x,y,z],"scale":s}
      -> {"ok":true,"resolve_offset":[x,y,z],...}
      (after applying the delta: push out of penetration with table/parent, then
      SEAT vertically on whatever is actually below — all-scene BVH, deadbanded so
      an already-seated pose is returned bit-exact; see _seat_z)
  {"cmd":"set_matrix","name":..,"M":4x4,"rebase":bool?} -> {"ok":true}
      (premultiply matrix_world by the WORLD delta M — no penetration resolve, no
      re-seat: physics owns the pose. rebase=true commits the result as the new
      set_pose base, so later (t,euler,s) deltas apply relative to the rested pose)
  {"cmd":"get_matrix","names":[..]}    -> {"ok":true,"matrices":{n:4x4}}
      (matrix_world per handle; translation = the object's origin = the set_pose pivot)
  {"cmd":"dump_npz","names":[..],"out":dir} -> {"ok":true,"paths":{n:path}}
      (world-frame evaluated mesh per handle/surface as a 1-part npz — CoACD/collider
      input for the Isaac composition session; authoritative current poses)
  {"cmd":"geom_sig","names":[..]}      -> {"ok":true,"sigs":{n:{nv,lo,hi,c,sd,vd}}}
      (LOCAL-frame vertex stats per handle — count, AABB, mean, per-axis std. A raw
      data edit like `v.co *= s` never touches matrix_world, so the composition
      physics authority diffs these to fold mesh-data scales into its Isaac mirror)
  {"cmd":"render","out":"/path.png"}                 -> {"ok":true}
  {"cmd":"save","path":"/path.blend"} | {"cmd":"ping"} | {"cmd":"shutdown"}
  -> {"ready":true} after the blend loads.
Run: blender --background --python register_blender_server.py -- <blend> [res]
"""

import json
import math
import re
import sys

import bpy
import mathutils
from mathutils.bvhtree import BVHTree

_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
BLEND = _argv[0]
RES = int(_argv[1]) if len(_argv) > 1 else 512

_base: dict[str, tuple] = {}  # name -> (location, euler, scale) base pose
_parts: dict[str, list] = {}  # name -> [sub-part object names]
_authored_empty_bindings: dict[str, str] = {}  # opted-in handle -> exact graph id
_parents: dict[str, str] = {}  # object handle -> parent object handle (stacking)
_support_cache: dict = {"name": None, "bvh": None}  # per-object support BVHTree
_RESOLVE_GAP = 0.001  # leave a 1 mm gap after separating
_RESOLVE_ITERS = 12  # push-out iterations (deepest contact each pass)
_RESOLVE_MAX_VERTS = 1500  # subsample big object meshes for the vertex test
# --- all-scene vertical seat (2026-08-07, owner-approved) -------------------------
# ``_seat_z`` replaced ``_rest_on_support``: the old drop ray-cast only against the
# SCENE-GRAPH support, so with a wrong graph edge (0806 food_packing: can#1 parented
# to the table, not its tray) every set_pose seated the can THROUGH the tray onto the
# table — 28 mm inside a solid neighbor — and silently undid the physics commit's
# rested pose. The seat probes the WHOLE scene instead and corrects z only when the
# candidate is genuinely buried or floating.
_seat_cache: dict = {"name": None, "sig": None, "bvh": None}  # per-object scene BVH
_pose_rev: dict[str, int] = {}  # handle -> pose-write counter (seat-BVH invalidation)
_SEAT_DEADBAND = 0.003  # |dz| <= this: candidate untouched BIT-EXACT (a physics-
# rested pose must survive set_pose — resting-contact reads ~GAP, PhysX noise ~2 mm)
_SEAT_PCT = 8.0  # seat on the level supported by >= this % of footprint columns —
# a leaning pen's 1-2 contact columns must not hoist the whole notebook onto it
_SEAT_OVERHEAD_TOL = 0.005  # bodies whose bbox bottom is at/above the object's top
# (minus this band) are riders/overheads, never seats — a tray must not be lifted
# onto its own cargo even when the graph parents the cargo elsewhere
_SEAT_RECAST_MAX = 8  # backface re-cast bound per column
_SEAT_DZ_CAP = 0.5  # give up (return 0) on absurd shifts; Isaac owns those poses


def _respond(obj):
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _mesh_objs():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def _surface_names():
    """Root-surface meshes = initializer-built primitives (table/walls/floor); every
    prepared object's descendant mesh is excluded even when its editable part names do
    not use the legacy ``obj_*`` prefix."""
    owned = {part for parts in _parts.values() for part in parts}
    return [
        o.name
        for o in _mesh_objs()
        if o.name not in owned and not o.name.startswith("obj_")
    ]


def _double_side(objs):
    """Disable backface culling on every material so SAM3D open-shell meshes render
    SOLID (not see-through) for the VLM flip check, regardless of the GLB's doubleSided
    flag. Inward-facing normals + single-sided materials otherwise render transparent,
    which would corrupt the VLM's orientation judgment."""
    for o in objs:
        if o.type != "MESH":
            continue
        for slot in o.material_slots:
            m = slot.material
            if m is None:
                continue
            m.use_backface_culling = False
            for attr in (
                "use_backface_culling_shadow",
                "use_backface_culling_lightprobe_volume",
            ):
                if hasattr(m, attr):
                    setattr(m, attr, False)


def _mesh_name_for_graph_id(graph_id):
    """Canonical pipeline handle encoded by an exact ``category#instance`` id."""
    category, separator, instance = graph_id.rpartition("#")
    if not separator or not category or not instance.isdigit():
        return None
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", category).strip("_").lower()[:48]
    slug = slug or "object"
    return f"obj_{slug}_{int(instance)}"


def _validate_authored_transform(obj, label):
    """Match the authored exporter: finite, nondegenerate, and non-mirrored."""
    values = [float(value) for row in obj.matrix_world for value in row]
    determinant = float(obj.matrix_world.to_3x3().determinant())
    if (
        not all(math.isfinite(value) for value in values)
        or not math.isfinite(determinant)
        or determinant <= 1.0e-12
    ):
        raise RuntimeError(f"{label} requires a finite, non-mirrored world transform")


def _validate_authored_empty(name, graph_id=None):
    """Return the exact live part list for an opted-in canonical authored Empty.

    The binding is minted only by ``prepare(..., allow_empty_roots=True)``.  Recheck
    it at every collider dump so a later script cannot turn an unrelated Empty into
    the expensive/special Boolean path or silently change the logical object tree.
    """
    bound_id = _authored_empty_bindings.get(name)
    if bound_id is None:
        return None
    if graph_id is not None and bound_id != graph_id:
        raise RuntimeError(f"procedural object root {name!r} changed graph binding")
    root = bpy.data.objects.get(name)
    if (
        root is None
        or root.type != "EMPTY"
        or root.parent is not None
        or root.get("grase_graph_id") != bound_id
        or _mesh_name_for_graph_id(bound_id) != name
    ):
        raise RuntimeError(f"procedural object root {name!r} is no longer canonical")
    owners = [obj for obj in bpy.data.objects if obj.get("grase_graph_id") == bound_id]
    if owners != [root]:
        raise RuntimeError(
            f"procedural graph identity {bound_id!r} is not bound to exactly {name!r}"
        )
    _validate_authored_transform(root, f"procedural object root {name!r}")
    descendants = sorted(root.children_recursive, key=lambda obj: obj.name)
    if not descendants or any(obj.type != "MESH" for obj in descendants):
        raise RuntimeError(
            f"procedural object root {name!r} must retain only MESH descendants"
        )
    if any(obj.get("grase_graph_id") not in (None, "") for obj in descendants):
        raise RuntimeError(
            f"only procedural object root {name!r} may carry a graph identity"
        )
    for obj in descendants:
        _validate_authored_transform(obj, f"procedural object part {obj.name!r}")
    expected = _parts.get(name)
    if expected is None or [obj.name for obj in descendants] != sorted(expected):
        raise RuntimeError(
            f"procedural object root {name!r} changed its prepared part set"
        )
    return descendants


def _prepare(names, allow_empty_roots=False, authored_empty_ids=None):
    """Prepare one logical handle per requested object.

    The exact canonical EMPTY path is the procedural representation.  It remains the
    live transform authority and every MESH descendant becomes one logical part set.
    The fallback below is intentionally the pre-existing MESH-root implementation, so
    scenes outside the opted-in procedural profile retain byte-for-byte semantics.
    """
    strict_authored_ids = None
    if authored_empty_ids is not None:
        if not allow_empty_roots or not isinstance(authored_empty_ids, dict):
            raise RuntimeError(
                "authored_empty_ids requires allow_empty_roots and a mapping"
            )
        strict_authored_ids = {}
        for name, graph_id in authored_empty_ids.items():
            if (
                not isinstance(name, str)
                or name not in names
                or not isinstance(graph_id, str)
                or _mesh_name_for_graph_id(graph_id) != name
            ):
                raise RuntimeError(
                    f"invalid inventory-authored Empty binding {name!r}: {graph_id!r}"
                )
            strict_authored_ids[name] = graph_id
    for name in names:
        root = bpy.data.objects.get(name)
        if (
            strict_authored_ids is not None
            and root is not None
            and root.type == "EMPTY"
        ):
            graph_id = root.get("grase_graph_id")
            if name not in strict_authored_ids or strict_authored_ids[name] != graph_id:
                raise RuntimeError(
                    f"procedural object root {name!r} is not authorized by the "
                    "validated runtime inventory"
                )
        if (
            allow_empty_roots
            and root is not None
            and root.type == "EMPTY"
            and isinstance(root.get("grase_graph_id"), str)
            and root.get("grase_graph_id").strip()
        ):
            graph_id = root.get("grase_graph_id")
            if (
                strict_authored_ids is not None
                and strict_authored_ids.get(name) != graph_id
            ):
                raise RuntimeError(
                    f"procedural object root {name!r} is not authorized by the "
                    "validated runtime inventory"
                )
            if (
                graph_id != graph_id.strip()
                or _mesh_name_for_graph_id(graph_id) != name
            ):
                raise RuntimeError(
                    f"procedural object root {name!r} has a non-canonical graph binding"
                )
            if root.parent is not None:
                raise RuntimeError(
                    f"procedural object root {name!r} must be parentless"
                )
            _validate_authored_transform(root, f"procedural object root {name!r}")
            owners = [
                obj for obj in bpy.data.objects if obj.get("grase_graph_id") == graph_id
            ]
            if owners != [root]:
                raise RuntimeError(
                    f"procedural graph identity {graph_id!r} must have one root owner"
                )
            descendants = sorted(root.children_recursive, key=lambda obj: obj.name)
            non_mesh = [o for o in descendants if o.type != "MESH"]
            if non_mesh:
                labels = ", ".join(f"{o.name}:{o.type}" for o in non_mesh)
                raise RuntimeError(
                    f"procedural object root {name!r} has non-MESH descendants: "
                    + labels
                )
            parts = [o for o in descendants if o.type == "MESH"]
            if not parts:
                raise RuntimeError(
                    f"procedural object root {name!r} has no MESH descendants"
                )
            if any(o.get("grase_graph_id") not in (None, "") for o in parts):
                raise RuntimeError(
                    f"only procedural object root {name!r} may carry a graph identity"
                )
            for part in parts:
                _validate_authored_transform(
                    part, f"procedural object part {part.name!r}"
                )
            _double_side(parts)
            root_world = root.matrix_world.copy()
            root.rotation_mode = "XYZ"
            root.matrix_world = root_world
            bpy.context.view_layer.update()
            _base[name] = (
                root.location.copy(),
                root.rotation_euler.copy(),
                root.scale.copy(),
            )
            _parts[name] = [o.name for o in parts]
            _authored_empty_bindings[name] = graph_id
            continue

        if strict_authored_ids is not None and name in strict_authored_ids:
            raise RuntimeError(
                f"inventory-authored object {name!r} is not a canonical Empty root"
            )
        parts = [
            o for o in _mesh_objs() if o.name == name or o.name.startswith(name + "_")
        ]
        if not parts:
            continue
        _double_side(
            parts
        )  # render solid for the VLM (open-shell meshes are see-through otherwise)
        main = next((o for o in parts if o.name == name), parts[0])
        corners = [
            o.matrix_world @ mathutils.Vector(c) for o in parts for c in o.bound_box
        ]
        centre = sum(corners, mathutils.Vector()) / len(corners)
        bpy.ops.object.select_all(action="DESELECT")
        main.select_set(True)
        bpy.context.view_layer.objects.active = main
        if (main.matrix_world.translation - centre).length > 1e-6:
            # Re-centre only when the origin is actually off. origin_set rewrites
            # every vertex as (co - new origin) in float32, so re-importing an
            # already-centred body (every session rebuild) perturbed its mesh data
            # by ~1e-7 m — enough to flip a byte digest and re-cook the collider
            # for an edit that never happened (0916 batch: 972 re-cooks, ~10 h).
            bpy.context.scene.cursor.location = centre
            bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        for c in parts:
            if c is not main:
                c.select_set(True)
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
        # glTF import leaves objects in QUATERNION mode, where rotation_euler is
        # IGNORED -- so set_pose's euler deltas would silently do nothing. Switch to
        # XYZ (identity quat -> euler 0) so rotations actually apply.
        main.rotation_mode = "XYZ"
        _base[name] = (
            main.location.copy(),
            main.rotation_euler.copy(),
            main.scale.copy(),
        )
        _parts[name] = [o.name for o in parts]
    if strict_authored_ids is not None:
        actual = {
            name: _authored_empty_bindings[name]
            for name in names
            if name in _authored_empty_bindings
        }
        if actual != strict_authored_ids:
            raise RuntimeError("prepared authored Empty inventory is incomplete")


def _set_pose(name, translate, euler, scale, render_only=False):
    """Apply base+deltas. ``render_only=True`` (F0a, 2026-08-08) skips the penetration
    resolver and the vertical seat: DIAGNOSTIC renders (scale/position hints, flip
    previews) must not mutate a physics-validated pose. Without it, restoring the
    CURRENT pose for a measurement re-ran the BVH heuristics and seated the object on
    the RENDER mesh — ~6 mm inside the CoACD hull its rest was computed on for cargo in
    a concave support (still_life apple: the certify free sim then popped it out and
    rolled it 262 mm off the plate). Same principle ``_set_matrix`` documents below;
    the diagnostic path just never had the escape. Line-search scoring keeps the
    resolver+seat: there the seated pose IS the candidate being evaluated."""
    _pose_rev[name] = _pose_rev.get(name, 0) + 1
    bl, be, bs = _base[name]
    o = bpy.data.objects[name]
    o.location = bl + mathutils.Vector(translate)
    o.rotation_euler = mathutils.Euler(
        (be[0] + euler[0], be[1] + euler[1], be[2] + euler[2])
    )
    o.scale = (bs[0] * scale, bs[1] * scale, bs[2] * scale)
    bpy.context.view_layer.update()  # parts' matrix_world reflect the new pose
    if render_only:
        return [0.0, 0.0, 0.0]
    off = _resolve_penetration(name)  # push OUT of penetration (builds the support BVH)
    off[2] += _seat_z(name)  # seat on whatever is ACTUALLY below (all-scene, vertical)
    return off


def _set_matrix(name, M, rebase=False):
    """Apply a WORLD-frame delta (e.g. an Isaac rested pose, possibly tilted) on top of
    the current pose. Deliberately no ``_resolve_penetration``/``_seat_z`` —
    the caller's physics produced this pose and the BVH heuristics must not re-mangle
    it. ``rebase`` commits the result as the new ``set_pose`` base so the next line
    search's (t,euler,s) deltas — which OVERWRITE loc/rot/scale from base — start from
    the rested pose instead of clobbering its tilt."""
    _pose_rev[name] = _pose_rev.get(name, 0) + 1
    o = bpy.data.objects[name]
    o.matrix_world = mathutils.Matrix([list(r) for r in M]) @ o.matrix_world
    bpy.context.view_layer.update()
    if rebase:
        _base[name] = (
            o.location.copy(),
            o.rotation_euler.copy(),
            o.scale.copy(),
        )


_authored_geometry = None


def _authored_geometry_module():
    """Lazy authored-only loader; ordinary Mesh/static registration never imports it."""
    global _authored_geometry
    if _authored_geometry is None:
        import importlib.util
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "blender"
            / "authored_root_local_geometry.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_grase_authored_root_local_geometry", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load authored root-local geometry helper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _authored_geometry = module
    return _authored_geometry


def _authored_empty_union(name, parts):
    """Build the exact shared centered root-local union and return float64 world arrays."""
    import bmesh

    helper = _authored_geometry_module()
    root = bpy.data.objects.get(name)
    if root is None:
        raise RuntimeError(f"procedural object root {name!r} disappeared")
    try:
        with helper.authored_union(root, parts, bpy=bpy, bmesh=bmesh) as payload:
            vertices, faces = helper.register_arrays(payload)
            # Keep _dump_npz's legacy list contract (`if not verts`) and perform its
            # one existing explicit float64/int64 conversion at the persistence seam.
            return vertices.tolist(), faces.tolist(), helper.part_arrays(payload)
    except RuntimeError as exc:
        if "union is disconnected" in str(exc):
            raise RuntimeError(
                f"procedural collider union for {name!r} has disconnected solid "
                "components; connect the authored primitives with real overlapping "
                "structural geometry"
            ) from exc
        raise


def _dump_npz(names, out_dir):
    """Evaluated world-frame mesh per handle or surface as one collider input.

    An opted-in canonical authored Empty is an exact Boolean union of disposable
    evaluated copies, matching its connected exported asset while preserving the live
    editable hierarchy.  Every legacy Mesh/surface keeps the original concatenation
    path.  The blend is the authoritative current pose source for both branches.
    """
    import os

    import numpy as np

    paths = {}
    for name in names:
        authored_parts = _validate_authored_empty(name)
        part_meshes = []
        if authored_parts is not None:
            verts, faces, part_meshes = _authored_empty_union(name, authored_parts)
        else:
            deps = bpy.context.evaluated_depsgraph_get()
            verts, faces, off = [], [], 0
            for nm in _parts.get(name, [name]):
                o = bpy.data.objects.get(nm)
                if o is None or o.type != "MESH":
                    continue
                ev = o.evaluated_get(deps)
                me = ev.to_mesh()
                me.calc_loop_triangles()
                mw = ev.matrix_world
                verts.extend([tuple(mw @ v.co) for v in me.vertices])
                faces.extend(
                    [tuple(off + i for i in t.vertices) for t in me.loop_triangles]
                )
                off += len(me.vertices)
                ev.to_mesh_clear()
        if not verts:
            continue
        path = os.path.join(out_dir, f"mesh_{name}.npz")
        # Authored Empties also carry their evaluated parts (part_v{i}/part_f{i}) so
        # the cook hulls each part instead of re-slicing the union (collision.py).
        parts_data = {"n_parts": np.asarray(len(part_meshes))} if part_meshes else {}
        for i, (pv, pf) in enumerate(part_meshes):
            parts_data[f"part_v{i}"] = np.asarray(pv, dtype=np.float64)
            parts_data[f"part_f{i}"] = np.asarray(pf, dtype=np.int64)
        np.savez(
            path, n=np.asarray(1),
            v0=np.asarray(verts, dtype=np.float64),
            f0=np.asarray(faces, dtype=np.int64),
            **parts_data,
        )  # fmt: skip
        paths[name] = path
    return paths


def _geom_sig(names):
    """Local-frame geometry signature per handle over all sub-parts, expressed in
    the handle's own object frame so pose changes cancel out: ``nv``/``nf`` vertex
    and face counts, AABB/mean/per-axis std (legacy statistics fit), and ``V`` —
    the vertex array itself as base64 little-endian float32 rows (Blender's own
    precision, lossless; decoded by composition_physics ``decode_local_vertices``),
    which the client fits vertex by vertex (``fit_local_affine``). Raw
    (non-evaluated) vertices: that is the data a freeform script mutates
    (`v.co *= s`), and raw-vs-raw comparison is self-consistent. Order is Blender's
    vertex index order, part by part, so an in-place edit stays comparable per
    vertex. No byte digest: float32 re-centring noise (~1e-7 m) is a displacement
    the client can threshold, not a hash flip."""
    import base64

    import numpy as np

    sigs = {}
    for name in names:
        root = bpy.data.objects.get(name)
        if root is None:
            continue
        Winv = np.asarray(root.matrix_world.inverted(), dtype=np.float64)
        chunks = []
        nf = 0
        for nm in _parts.get(name, [name]):
            o = bpy.data.objects.get(nm)
            if o is None or o.type != "MESH" or len(o.data.vertices) == 0:
                continue
            arr = np.empty(len(o.data.vertices) * 3, dtype=np.float64)
            o.data.vertices.foreach_get("co", arr)
            nf += len(o.data.polygons)
            if o is root:
                # the handle's own mesh: its raw coordinates ARE the handle frame.
                # Going through Winv @ W would add ~1e-8 of float noise per pose,
                # so the bytes stay pose-invariant only on this direct path.
                chunks.append(arr.reshape(-1, 3))
                continue
            M = Winv @ np.asarray(o.matrix_world, dtype=np.float64)
            chunks.append(arr.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        if not chunks:
            continue
        V = np.vstack(chunks)
        sigs[name] = {
            "nv": int(len(V)),
            "nf": int(nf),
            "lo": [float(x) for x in V.min(0)],
            "hi": [float(x) for x in V.max(0)],
            "c": [float(x) for x in V.mean(0)],
            "sd": [float(x) for x in V.std(0)],
            "V": base64.b64encode(
                np.ascontiguousarray(V, dtype="<f4").tobytes()
            ).decode("ascii"),
        }
    return sigs


def _descendants_of(name, parents):
    """Transitive scene-graph children of ``name`` per the handle->parent map."""
    kids: dict = {}
    for c, p in parents.items():
        kids.setdefault(p, []).append(c)
    out, stack = set(), list(kids.get(name, []))
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(kids.get(n, []))
    return out


def _seat_filter(name, handles, aabbs, parents):
    """Handles allowed to SEAT ``name`` (support it from below): every other handle
    EXCEPT ``name`` itself, its scene-graph descendants, and OVERHEAD bodies — any
    handle whose bbox bottom sits at/above ``name``'s bbox top (minus tolerance).
    The overhead rule is geometric on purpose: it catches true riders even when the
    GRAPH parents them elsewhere (0806 food_packing), so a tray is never hoisted
    onto its own cargo, and shelves above never read as floors. It uses only z
    extents (no xy test) so the result — and the seat BVH built from it — is stable
    across a whole xy candidate ladder."""
    lo, hi = aabbs[name]
    skip = _descendants_of(name, parents)
    out = []
    for h in handles:
        if h == name or h in skip:
            continue
        box = aabbs.get(h)
        if box is None:
            continue
        if box[0][2] >= hi[2] - _SEAT_OVERHEAD_TOL:
            continue  # rider / overhead body — never a seat
        out.append(h)
    return out


def _footprint_columns(verts, cells=12):
    """Lowest vertex per xy-cell — the object's BOTTOM surface. The seat percentile
    must range over the FOOTPRINT, never all vertices: on a tall object the bottom
    ring is <8% of the mesh, so an all-vertex percentile lands 30-80mm up the SIDE
    and the 'seat' pulls a correctly-rested object down by that height on every
    set_pose (0807_bulk_misc_online3: lamp sank exactly 31.5mm, flowerpot 77.1mm,
    into the desk — reproduced live with an identity set_pose on the final blend).
    Flat/squat objects escaped (bottom >= 8% of verts), which is why the food
    packing replays validated clean."""
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0, y0 = min(xs), min(ys)
    dx = (max(xs) - x0) / cells or 1e-9
    dy = (max(ys) - y0) / cells or 1e-9
    best: dict = {}
    for x, y, z in verts:
        k = (min(cells - 1, int((x - x0) / dx)), min(cells - 1, int((y - y0) / dy)))
        if k not in best or z < best[k][2]:
            best[k] = (x, y, z)
    return list(best.values())


def _column_clearances(ray, columns, z_top):
    """Per-column clearance of ``columns`` (world (x,y,z) vertex samples) above the
    scene: ``ray((x,y,z)) -> (hit_z, normal_z) | None`` casts straight DOWN. Rays
    start just above ``z_top`` (the object's OWN top), so anything above the object
    can never be its floor. Hits with ``normal_z <= 0`` (the underside of a leaning
    or overhanging feature — e.g. a pen shaft crossing the footprint) are skipped by
    re-casting from just below, so the column reports the real surface under it."""
    eps = 1e-4
    out = []
    for x, y, vz in columns:
        oz = z_top + eps
        for _ in range(_SEAT_RECAST_MAX):
            hit = ray((x, y, oz))
            if hit is None:
                break
            hz, nz = hit
            if nz > 1e-6:
                out.append(vz - hz)
                break
            oz = hz - eps
    return out


def _seat_offset(clearances):
    """Vertical shift that seats the object on the level supported by >= _SEAT_PCT%
    of its footprint columns. Percentile, NOT min/max: a leaning pen's 1-2 contact
    columns must not hoist the whole notebook (lateral contact is Isaac's job).
    Positive = lift OUT of a buried candidate (xy slid into a neighbor's volume);
    negative = drop a genuine floater (xy slid off its support / a scale-down lifted
    the base — the old ``_rest_on_support`` founding case). The deadband keeps
    already-seated candidates BIT-EXACT so a physics-rested pose survives set_pose
    untouched — the invariant whose absence buried 0806 food_packing's can."""
    if not clearances:
        return 0.0
    ordered = sorted(clearances)
    c = ordered[min(len(ordered) - 1, int(len(ordered) * _SEAT_PCT / 100.0))]
    dz = _RESOLVE_GAP - c
    if abs(dz) <= _SEAT_DEADBAND or abs(dz) > _SEAT_DZ_CAP:
        return 0.0
    return dz


def _world_aabb(name):
    """World AABB of a handle over all its sub-parts' bound_box corners, or None."""
    pts = []
    for nm in _parts.get(name, [name]):
        o = bpy.data.objects.get(nm)
        if o is None or o.type != "MESH":
            continue
        pts.extend(o.matrix_world @ mathutils.Vector(c) for c in o.bound_box)
    if not pts:
        return None
    return (
        [min(p[i] for p in pts) for i in range(3)],
        [max(p[i] for p in pts) for i in range(3)],
    )


def _seat_meshes(name):
    """Mesh names eligible to seat ``name``: all root surfaces (walls excluded) and
    every other object handle, filtered by ``_seat_filter`` (self / descendants /
    overhead bodies removed), expanded to sub-part meshes."""
    part2handle = {part: handle for handle, parts in _parts.items() for part in parts}
    handles = set()
    surfaces = set(_surface_names())
    for obj in _mesh_objs():
        name_or_part = obj.name
        if name_or_part in part2handle:
            handles.add(part2handle[name_or_part])
        elif name_or_part.startswith("obj_"):
            # Preserve the legacy fallback for an unprepared object-looking mesh.
            handles.add(name_or_part)
        elif name_or_part in surfaces and not _is_wall_surface(name_or_part):
            handles.add(name_or_part)
    handles.add(name)
    aabbs = {h: _world_aabb(h) for h in handles}
    if aabbs.get(name) is None:
        return []
    keep = _seat_filter(name, sorted(handles), aabbs, _parents)
    return [m for h in keep for m in _parts.get(h, [h])]


def _seat_z(name):
    """Seat ``name`` vertically on whatever is ACTUALLY below it, whole scene —
    replaces ``_rest_on_support`` (which trusted the scene-graph support and seated
    objects THROUGH un-graphed neighbors). Applies and returns one z-shift, 0.0 when
    the candidate is already seated (deadband — bit-exact no-op) or nothing lies
    below. The BVH is cached per object and invalidated when any OTHER handle's pose
    is written, so a whole candidate ladder shares one build."""
    main = bpy.data.objects.get(name)
    if main is None:
        return 0.0
    sig = tuple(sorted((h, r) for h, r in _pose_rev.items() if h != name))
    if _seat_cache.get("name") != name or _seat_cache.get("sig") != sig:
        v, p = _world_polys(_seat_meshes(name))
        _seat_cache["name"] = name
        _seat_cache["sig"] = sig
        _seat_cache["bvh"] = (
            BVHTree.FromPolygons(v, p, all_triangles=False) if v else None
        )
    bvh = _seat_cache["bvh"]
    if bvh is None:
        return 0.0
    verts = _object_world_verts(name)
    if not verts:
        return 0.0
    z_top = max(v.z for v in verts)
    down = mathutils.Vector((0.0, 0.0, -1.0))

    def ray(origin):
        loc, nrm, _i, _d = bvh.ray_cast(mathutils.Vector(origin), down)
        return None if loc is None else (loc.z, nrm.z)

    dz = _seat_offset(
        _column_clearances(
            ray, _footprint_columns([(v.x, v.y, v.z) for v in verts]), z_top
        )
    )
    if dz:
        main.location.z += dz
        bpy.context.view_layer.update()
    return dz


def _is_wall_surface(name):
    """Walls are named ``wall*`` by the initializer (``wall_0``, ``wall_bg``); the surfaces
    objects rest on are ``table*`` / ``floor*``. Walls are excluded from the support set --
    a thin vertical wall makes nearest-surface separation ambiguous. We key on the name
    rather than geometry on purpose: a single-view table whose top is occluded (e.g. covered
    by a mat) reconstructs as a mostly-vertical slab, so any flat-vs-tall geometric test
    would wrongly reject it and drop everything onto the floor. Name/category is robust."""
    return name.startswith("wall")


def _support_meshes(name):
    """Mesh names ``name`` must not penetrate: the root surfaces it rests on (table / floor,
    i.e. all root surfaces EXCEPT walls) plus its parent object's parts (when stacked, e.g. a
    cushion on a sofa). Vertical walls are excluded. Excludes the object's own parts."""
    names = [nm for nm in _surface_names() if not _is_wall_surface(nm)]
    parent = _parents.get(name)
    if parent:
        names += _parts.get(parent, [parent])
    own = set(_parts.get(name, [name]))
    return [n for n in names if n not in own]


def _world_polys(names):
    verts, polys, base = [], [], 0
    for nm in names:
        o = bpy.data.objects.get(nm)
        if o is None or o.type != "MESH" or len(o.data.polygons) == 0:
            continue
        mw = o.matrix_world
        # A mirrored transform (negative determinant — e.g. an initializer-built table
        # with all-negative scale, 0708_iso_bridge1) flips triangle winding, so the BVH
        # face normals point INTO the solid: the penetration inside-test then reads
        # backwards and pushes resting objects THROUGH the surface (every object was
        # driven 3.9cm down into the table box). Reverse the index order to restore
        # outward normals.
        flip = mw.to_3x3().determinant() < 0.0
        verts.extend([mw @ v.co for v in o.data.vertices])
        for p in o.data.polygons:
            idx = [base + i for i in p.vertices]
            polys.append(idx[::-1] if flip else idx)
        base += len(o.data.vertices)
    return verts, polys


def _object_world_verts(name):
    out = []
    for nm in _parts.get(name, [name]):
        o = bpy.data.objects.get(nm)
        if o is None or o.type != "MESH":
            continue
        mw = o.matrix_world
        out.extend([mw @ v.co for v in o.data.vertices])
    if len(out) > _RESOLVE_MAX_VERTS:  # subsample for the per-vertex test
        out = out[:: len(out) // _RESOLVE_MAX_VERTS + 1]
    return out


def _resolve_penetration(name):
    """Minimal-translation separation: push ``name`` out of any penetration with its
    support. Each pass finds the DEEPEST object vertex inside the support and pushes the
    whole object along that contact's outward surface normal by the penetration depth
    (+gap); iterating clears multi-sided contacts (e.g. a cushion wedged between two sofa
    faces). Generalises a vertical re-rest: a flat table contributes a +Z normal, a sofa
    side a horizontal one. Returns the total applied world offset."""
    support = _support_meshes(name)
    if not support:
        return [0.0, 0.0, 0.0]
    if _support_cache.get("name") != name:  # build the support BVH once per object
        v, p = _world_polys(support)
        _support_cache["name"] = name
        _support_cache["bvh"] = (
            BVHTree.FromPolygons(v, p, all_triangles=False) if v else None
        )
    bvh = _support_cache["bvh"]
    main = bpy.data.objects.get(name)
    if bvh is None or main is None:
        return [0.0, 0.0, 0.0]
    verts = _object_world_verts(name)  # at the current pose; pure translation
    applied = mathutils.Vector((0.0, 0.0, 0.0))
    for _ in range(_RESOLVE_ITERS):
        deepest, push = 0.0, None
        for v in verts:
            loc, nrm, _i, _d = bvh.find_nearest(v + applied)
            if loc is None:
                continue
            signed = (v + applied - loc).dot(
                nrm
            )  # < 0 => the vertex is inside the support
            if signed < 0.0 and -signed > deepest:
                deepest, push = -signed, nrm.normalized()
        if push is None or deepest < 1e-5:
            break
        applied = applied + push * (deepest + _RESOLVE_GAP)
    if applied.length > 1e-6:
        main.location = main.location + applied
        bpy.context.view_layer.update()
    return [applied.x, applied.y, applied.z]


def _isolate(visible, holdout=None, hide=None):
    """Show ``visible`` normally; show ``holdout`` objects as HOLDOUTS — they still
    occlude (punch alpha-0 holes in) whatever is behind them but contribute no
    silhouette pixels themselves. Used for the IoU render: the object is visible, its
    stacking ancestors are holdouts, so a tray occludes its child without joining the
    child's silhouette union (which deflated every stacked child's IoU toward
    |child|/|parent|).

    ``visible=None`` shows everything; with ``hide`` it shows everything EXCEPT those
    objects (investigate's de-occluded render: the occluders a redetect's edit removed
    are hidden so the render matches the edited photo). Show-all-except is server-side
    on purpose — the session cannot reliably enumerate every scene mesh."""
    hold = set()
    for name in holdout or []:
        hold.update(_parts.get(name, [name]))
    if visible is None:  # show everything (full-scene render for investigate crops)
        hid = set()
        for name in hide or []:
            hid.update(_parts.get(name, [name]))
        for o in _mesh_objs():
            o.hide_render = o.name in hid
            o.is_holdout = False
        return
    vis = set()
    for name in visible:
        vis.update(_parts.get(name, [name]))
    for o in _mesh_objs():
        o.hide_render = o.name not in vis and o.name not in hold
        o.is_holdout = o.name in hold


def _ensure_light():
    """Add flat ambient + a sun so an UNLIT scene (e.g. preprocessing, before the
    initializer builds lights) renders visibly. In-memory only -- never saved."""
    sc = bpy.context.scene
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        # Unlit scene (preprocessing flip stage): bright, even lighting so objects of ANY
        # albedo read CLEARLY under the Standard tone-map — strong ambient + a key sun and an
        # opposite fill so no face goes black. Tuned (book + plush) so dark/gray meshes are
        # clearly visible while a light object still isn't blown out.
        if bg is not None:
            bg.inputs["Strength"].default_value = 3.0
        for nm, energy, rot in (
            ("FlipSun", 3.0, (0.6, 0.2, 0.3)),
            ("FlipFill", 2.0, (-0.6, -0.2, -1.0)),
        ):
            data = bpy.data.lights.new(nm, type="SUN")
            data.energy = energy
            obj = bpy.data.objects.new(nm, data)
            bpy.context.collection.objects.link(obj)
            obj.rotation_euler = rot
    elif bg is not None:  # already-lit scene: just keep ambient non-zero
        bg.inputs["Strength"].default_value = max(
            bg.inputs["Strength"].default_value, 1.0
        )


def _render(out, transparent=True, width=None, height=None, standard=False):
    """``transparent`` alpha=silhouette (IoU render). ``width``/``height`` force an
    exact resolution (the GT framing for the VLM context render); otherwise the blend
    aspect is scaled so its long side is ``RES``.

    EEVEE is forced for speed, but EVERY render setting we touch is restored afterwards so
    the saved blend keeps the pipeline's engine + color management. (Previously this set
    ``engine=EEVEE`` + ``view_transform=Standard`` without restoring them, so the saved
    ``registered.blend`` leaked a flat 'Standard' tone-map into the composition stage --
    the IoU silhouette uses only alpha, so the view transform never mattered here anyway.)"""
    sc = bpy.context.scene
    saved = (
        sc.render.engine,
        sc.render.film_transparent,
        sc.render.image_settings.file_format,
        sc.render.image_settings.color_mode,
        sc.render.resolution_x,
        sc.render.resolution_y,
        sc.view_settings.view_transform,
    )
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    # Flip renders only: use a Standard (linear-sRGB) tone-map -- the blend's default AgX
    # crushes low-albedo objects to near-black in these flat preprocessing renders. Restored
    # below, so it never leaks into the saved blend / later stages (register/IoU keep AgX).
    if standard:
        sc.view_settings.view_transform = "Standard"
    sc.render.film_transparent = bool(transparent)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    if width and height:
        sc.render.resolution_x, sc.render.resolution_y = int(width), int(height)
    else:
        s = RES / max(rx, ry)
        sc.render.resolution_x, sc.render.resolution_y = int(rx * s), int(ry * s)
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    (
        sc.render.engine,
        sc.render.film_transparent,
        sc.render.image_settings.file_format,
        sc.render.image_settings.color_mode,
        sc.render.resolution_x,
        sc.render.resolution_y,
        sc.view_settings.view_transform,
    ) = saved


def main():
    bpy.ops.wm.open_mainfile(filepath=BLEND)
    _respond({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "error": "bad json"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "shutdown":
                break
            if cmd == "ping":
                _respond({"ok": True, "pong": True})
            elif cmd == "prepare":
                if "authored_empty_ids" in req and not isinstance(
                    req["authored_empty_ids"], dict
                ):
                    raise RuntimeError("authored_empty_ids must be a mapping")
                strict_ids = (
                    req.get("authored_empty_ids")
                    if "authored_empty_ids" in req
                    else None
                )
                _prepare(
                    req["objects"],
                    allow_empty_roots=bool(req.get("allow_empty_roots", False)),
                    authored_empty_ids=strict_ids,
                )
                _parents.update(req.get("parents") or {})  # stacking map for separation
                _support_cache["name"] = None  # invalidate any stale support BVH
                _seat_cache["name"] = None  # ... and the all-scene seat BVH
                response = {
                    "ok": True,
                    "prepared": list(_base),
                    "surfaces": _surface_names(),
                }
                if strict_ids is not None:
                    response["authored_empty_roots"] = {
                        name: _authored_empty_bindings[name]
                        for name in req["objects"]
                        if name in _authored_empty_bindings
                    }
                    response["logical_parts"] = {
                        name: list(_parts[name])
                        for name in response["authored_empty_roots"]
                    }
                    response["body_parts"] = {
                        name: list(_parts[name])
                        for name in req["objects"]
                        if name in _parts
                    }
                _respond(response)
            elif cmd == "isolate":
                _isolate(req["visible"], req.get("holdout"), req.get("hide"))
                _respond({"ok": True})
            elif cmd == "ensure_light":
                _ensure_light()
                _respond({"ok": True})
            elif cmd == "set_pose":
                off = _set_pose(
                    req["name"],
                    req.get("translate", [0, 0, 0]),
                    req.get("euler", [0, 0, 0]),
                    req.get("scale", 1.0),
                    render_only=bool(req.get("render_only", False)),
                )
                _respond({"ok": True, "resolve_offset": off})
            elif cmd == "set_matrix":
                _set_matrix(req["name"], req["M"], req.get("rebase", False))
                _respond({"ok": True})
            elif cmd == "get_matrix":
                ms = {}
                for n in req["names"]:
                    o = bpy.data.objects.get(n)
                    if o is not None:
                        ms[n] = [list(r) for r in o.matrix_world]
                _respond({"ok": True, "matrices": ms})
            elif cmd == "dump_npz":
                _respond({"ok": True, "paths": _dump_npz(req["names"], req["out"])})
            elif cmd == "geom_sig":
                _respond({"ok": True, "sigs": _geom_sig(req["names"])})
            elif cmd == "render":
                _render(
                    req["out"],
                    req.get("transparent", True),
                    req.get("width"),
                    req.get("height"),
                    req.get("standard", False),
                )
                _respond({"ok": True, "out": req["out"]})
            elif cmd == "save":
                for o in _mesh_objs():  # un-isolate: never persist hidden/holdout state
                    o.hide_render = False
                    o.is_holdout = False
                bpy.ops.wm.save_as_mainfile(filepath=req["path"])
                _respond({"ok": True})
            else:
                _respond({"ok": False, "error": f"unknown cmd {cmd}"})
        except Exception as exc:  # noqa: BLE001 - keep the server alive on bad input
            _respond({"ok": False, "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
