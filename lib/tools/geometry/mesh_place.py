"""Place a SAM3D mesh at its MoGE metric transform.

SAM3D reconstructs object SHAPE (in a camera-style frame, +Z = depth) at its own
metric guess. MoGE gives the trustworthy metric PLACEMENT (world center + size).
We keep SAM3D's geometry + orientation but anchor position and scale to MoGE:

  1. rotate the mesh from the camera frame into the GRASE Z-up world (same
     OpenCV->world rotation used for the MoGE point cloud, so the object lands
     upright and un-mirrored);
  2. recenter to the origin and uniformly scale so its bbox diagonal matches the
     MoGE size diagonal (uniform keeps SAM3D's reconstructed proportions rather
     than forcing MoGE's noisier per-axis box);
  3. translate to the MoGE world center.
"""

from __future__ import annotations

import numpy as np


def sam3d_to_world(verts: np.ndarray, flip_x: bool = False) -> np.ndarray:
    """SAM3D mesh frame -> GRASE Z-up world.

    SAM3D's worker output is Y-up, Z-forward (depth): its +Y is the object's up
    and +Z is into the scene. Map to the GRASE world (Z up, camera at origin looking
    -Y, image-right = world -X):
      world up    (+Z) =  +y   (SAM3D up)
      world depth (-Y) =  +z   (SAM3D forward) -> world Y = -z
      world right (+X) =  +x   (SAM3D image-right)
    so world = (x, -z, +y) -- a PROPER rotation (det +1) that preserves the raw mesh's
    chirality. Validated against the source image (8219 reference-camera render vs GT):
    meshes land upright with correct left-right chirality. ``flip_x=True`` negates X,
    an improper reflection (det -1) -- kept only for debugging; do not use for placement
    (it mirrors the object and reads as an orientation flip)."""
    v = np.asarray(verts, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    sx = -x if flip_x else x
    return np.stack([sx, -z, y], axis=-1)


def canonical_y_up_to_world(verts: np.ndarray) -> np.ndarray:
    """Canonical Y-up mesh frame -> SceneRig Z-up world.

    SAM3D pristine fallback meshes are unposed canonical glTFs: +Y is object-up and
    the mesh is already gravity-aligned. Rotate Y-up -> Z-up, ``(x,y,z) -> (x,-z,y)``,
    and do not apply the camera gravity rotation ``R``."""
    v = np.asarray(verts, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, -z, y], axis=-1)


def quat_to_rotmat(q) -> np.ndarray:
    """Quaternion (w, x, y, z, real-first -- PyTorch3D convention) -> 3x3 rotation matrix.
    Pure-numpy mirror of pytorch3d.transforms.quaternion_to_matrix so the main venv can
    inspect SAM3D poses without importing torch."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def euler_xyz_matrix(euler) -> np.ndarray:
    """Blender 'XYZ' Euler (radians) -> 3x3 rotation matrix (R = Rz @ Ry @ Rx), so the
    matrix matches what the register render server applies via ``rotation_euler``."""
    rx, ry, rz = (float(a) for a in euler)
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx),
        np.sin(rx),
        np.cos(ry),
        np.sin(ry),
        np.cos(rz),
        np.sin(rz),
    )
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


def place_mesh_vertices(
    verts: np.ndarray,
    center,
    size,
    to_world: bool = True,
    flip_x: bool = False,
    R=None,
    backend: str = "sam3d",
    flip_euler=None,
    ext_diag=None,
) -> np.ndarray:
    """Mesh vertices (N, 3) -> GRASE world, at the MoGE center/size.

    ``backend='sam3d'``: SAM3D reconstructs in the (camera-tilted) view frame, so the
    gravity rotation ``R`` is applied to make the object upright. ``flip_x=False`` (default)
    uses ``sam3d_to_world`` as a PROPER rotation (det +1) that matches the reference camera,
    so meshes keep the orientation they already have at the source view -- empirically
    correct and deterministic (no per-object VLM flip needed). ``flip_x=True`` is the legacy
    improper reflection (det -1), which can tip flat objects over; production leaves it off.
    ``backend='canonical_y_up'``: the mesh is CANONICAL (gravity-aligned,
    Y-up glTF), so ``R`` is ignored and an un-mirrored Y-up->Z-up map is used. ``flip_euler`` is an optional orientation override (a
    discrete 180deg flip) baked in about the object centre; unused by default. The target
    ``center`` is in the gravity-aligned world."""
    if to_world:
        v = (
            sam3d_to_world(verts, flip_x=flip_x)
            if backend == "sam3d"
            else canonical_y_up_to_world(verts)
        )
    else:
        v = np.asarray(verts, dtype=np.float64)
    if R is not None and backend == "sam3d":
        v = v @ np.asarray(R, dtype=v.dtype).T
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = v - (lo + hi) / 2.0
    ext = hi - lo  # bbox diagonal is flip-invariant
    if flip_euler is not None:
        v = v @ euler_xyz_matrix(flip_euler).T  # rotate about the centred origin
    # Denominator = the mesh's own diagonal. Default: AABB of the (rotated) verts. When the
    # caller passes ``ext_diag`` (the mesh's rotation-invariant 3D OBB diagonal), use it: a
    # flat object placed at a yaw has an inflated AABB, so AABB matching under-scales it (a
    # laptop shrank ~17%); the OBB diagonal is yaw-independent, so the metric ``size``
    # diagonal maps to the true object size. ``size`` must be the matching cloud OBB box.
    denom = float(ext_diag) if ext_diag else float(np.linalg.norm(ext))
    scale = float(np.linalg.norm(size)) / max(denom, 1e-6)
    return v * scale + np.asarray(center, dtype=np.float64)


def _pre_rotate_for_gltf(world: np.ndarray) -> np.ndarray:
    """Pre-rotate Z-up world verts so Blender's glTF importer (which bakes a
    +90deg X, Y-up->Z-up rotation mapping (x,y,z)->(x,-z,y)) restores them.
    Inverse is (x,y,z)->(x,-z,y)."""
    x, y, z = world[:, 0], world[:, 1], world[:, 2]
    return np.stack([x, z, -y], axis=-1)


def _unpre_rotate_for_gltf(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_pre_rotate_for_gltf`: glTF-import-frame verts -> Z-up world,
    mapping (x,y,z)->(x,-z,y)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([x, -z, y], axis=-1)


def apply_world_translation_to_glb(in_glb: str, out_glb: str, delta) -> dict:
    """Translate an already-placed GLB by a world-space ``delta`` (x, y, z), preserving the
    gltf-import pre-rotation. Used to bake the rest-on-support z-shift into the placed mesh."""
    import os

    import trimesh

    mesh = trimesh.load(in_glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    world = world + np.asarray(delta, dtype=np.float64)
    mesh.vertices = _pre_rotate_for_gltf(world)
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    return {"out_glb": out_glb, "delta": [float(x) for x in delta]}


def apply_world_rotation_to_glb(in_glb: str, out_glb: str, R_world, center) -> dict:
    """Rotate an already-placed GLB about a world-space ``center`` by the world rotation
    ``R_world`` (3x3), preserving the gltf-import pre-rotation. Used to bake the physics
    pitch/roll tilt into the placed mesh, keeping its MoGE center fixed."""
    import os

    import trimesh

    mesh = trimesh.load(in_glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    c = np.asarray(center, dtype=np.float64)
    world = (world - c) @ np.asarray(R_world, dtype=np.float64).T + c
    mesh.vertices = _pre_rotate_for_gltf(world)
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    return {"out_glb": out_glb, "center": [float(x) for x in c]}


def placed_glb_obb_extents(glb: str):
    """Sorted-descending OBB extents ``[long, mid, short]`` (metres) of an already-placed
    GLB, in world axes. OBB extents are rotation-invariant, so this reads the object's true
    width/depth/height regardless of its world yaw. ``None`` if the OBB solve fails."""
    import trimesh
    from trimesh.bounds import oriented_bounds

    mesh = trimesh.load(glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    try:
        _, extents = oriented_bounds(world)
    except Exception:  # noqa: BLE001 - degenerate mesh
        return None
    return np.sort(np.asarray(extents, dtype=np.float64))[::-1]


def placed_glb_obb_frame(glb: str) -> dict[str, np.ndarray] | None:
    """Measure one immutable OBB frame for an already-placed GLB.

    ``oriented_bounds`` is allowed to choose its minimum-volume frame exactly once.
    Callers that anisotropically deform the mesh must retain this frame: asking the OBB
    solver to choose again afterwards can select a different, equally plausible frame and
    falsely report that a correct scale missed its target.
    """
    import trimesh
    from trimesh.bounds import oriented_bounds

    try:
        mesh = trimesh.load(glb, force="mesh")
        world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
        if world.ndim != 2 or world.shape[1] != 3 or len(world) < 4:
            return None
        if not np.isfinite(world).all():
            return None
        to_origin, extents = oriented_bounds(world)
    except Exception:  # noqa: BLE001 - missing/degenerate geometry is not measurable
        return None
    to_origin = np.asarray(to_origin, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    if (
        to_origin.shape != (4, 4)
        or extents.shape != (3,)
        or not np.isfinite(to_origin).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 1e-8)
    ):
        return None
    order = np.argsort(extents)[::-1]
    return {
        "to_origin": to_origin,
        "extents": extents,
        "axis_order": order,
        "sorted_extents": extents[order],
    }


def rescale_corresponding_glbs_to_extents(
    canonical_glb: str,
    posed_glb: str,
    canonical_out_glb: str,
    posed_out_glb: str,
    target_sorted,
    *,
    canonical_frame: dict[str, np.ndarray] | None = None,
    correspondence_tol: float = 1e-5,
) -> dict:
    """Apply one frozen canonical scale tensor to a corresponding GLB pair.

    ``canonical_glb`` is the metric pristine candidate (``*_pcand.glb``) and
    ``posed_glb`` is the same reconstructed mesh in its SAM3D pose.  Their matching
    triangle corners recover the rigid pose exactly.  The target is measured and applied
    only in the pristine OBB frame, then transported to the posed twin; a second OBB solve
    is deliberately never used for validation.

    Both outputs retain their original world-space bottom-centre anchor.  The tiny uniform
    scale recovered between the two metric placements is normalized away so both emitted
    meshes have the same physical target box.  The caller is responsible for staging both
    outputs and committing them atomically.
    """
    import os

    import trimesh

    from lib.tools.geometry.collision import corner_similarity

    def _load(path: str):
        mesh = trimesh.load(path, force="mesh")
        world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if (
            world.ndim != 2
            or world.shape[1] != 3
            or len(world) < 4
            or faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) == 0
            or not np.isfinite(world).all()
            or int(faces.min()) < 0
            or int(faces.max()) >= len(world)
        ):
            raise ValueError(f"invalid or non-finite GLB geometry: {path}")
        return mesh, world, faces

    def _anchor(world: np.ndarray) -> np.ndarray:
        lo, hi = world.min(axis=0), world.max(axis=0)
        return np.array(
            [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]],
            dtype=np.float64,
        )

    def _write(mesh, world: np.ndarray, path: str) -> None:
        if not np.isfinite(world).all():
            raise ValueError(
                f"same-size transform produced non-finite geometry: {path}"
            )
        mesh.vertices = _pre_rotate_for_gltf(world)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mesh.export(path)

    canonical_mesh, canonical_world, canonical_faces = _load(canonical_glb)
    posed_mesh, posed_world, posed_faces = _load(posed_glb)
    if canonical_faces.shape != posed_faces.shape:
        raise ValueError("pristine/posed triangle topology differs")

    src_corners = canonical_world[canonical_faces].reshape(-1, 3)
    dst_corners = posed_world[posed_faces].reshape(-1, 3)
    if not np.isfinite(correspondence_tol) or correspondence_tol <= 0:
        raise ValueError("correspondence_tol must be finite and positive")
    similarity = corner_similarity(
        src_corners, dst_corners, tol=float(correspondence_tol)
    )
    if similarity is None:
        raise ValueError("pristine/posed correspondence is not a rigid similarity")
    source_scale, rotation, translation = similarity
    if (
        not np.isfinite(source_scale)
        or source_scale <= 1e-8
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("pristine/posed similarity is invalid")

    frame = canonical_frame or placed_glb_obb_frame(canonical_glb)
    if frame is None:
        raise ValueError("metric pristine OBB frame is unavailable")
    to_origin = np.asarray(frame.get("to_origin"), dtype=np.float64)
    extents = np.asarray(frame.get("extents"), dtype=np.float64)
    order = np.asarray(frame.get("axis_order"), dtype=np.int64)
    if (
        to_origin.shape != (4, 4)
        or extents.shape != (3,)
        or order.shape != (3,)
        or sorted(order.tolist()) != [0, 1, 2]
        or not np.isfinite(to_origin).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 1e-8)
    ):
        raise ValueError("metric pristine OBB frame is malformed")
    canonical_local = trimesh.transform_points(canonical_world, to_origin)
    measured_source = np.ptp(canonical_local, axis=0)
    if not np.allclose(measured_source, extents, rtol=0.002, atol=1e-5):
        raise ValueError("metric pristine OBB frame is stale")

    target = np.asarray(target_sorted, dtype=np.float64)
    if (
        target.shape != (3,)
        or not np.isfinite(target).all()
        or np.any(target <= 1e-8)
        or np.any(np.diff(target) > 1e-8)
    ):
        raise ValueError("same-size target must be finite [long, mid, short] extents")
    factors = np.ones(3, dtype=np.float64)
    for rank, axis in enumerate(order):
        factors[axis] = target[rank] / extents[axis]

    # Scale the metric pristine mesh in its one frozen frame.
    canonical_scaled_local = canonical_local * factors
    canonical_scaled = trimesh.transform_points(
        canonical_scaled_local, np.linalg.inv(to_origin)
    )
    canonical_delta = _anchor(canonical_world) - _anchor(canonical_scaled)
    canonical_scaled += canonical_delta

    # Bring the posed twin back to that same frame, apply the SAME tensor, and restore
    # only its rigid pose. Deliberately omit ``source_scale`` on the forward map: the
    # output pair must share one physical metric size, not retain placement round-off.
    posed_in_canonical = (posed_world - translation) @ rotation / source_scale
    posed_local = trimesh.transform_points(posed_in_canonical, to_origin)
    posed_scaled_canonical = trimesh.transform_points(
        posed_local * factors, np.linalg.inv(to_origin)
    )
    posed_scaled = posed_scaled_canonical @ rotation.T + translation
    posed_delta = _anchor(posed_world) - _anchor(posed_scaled)
    posed_scaled += posed_delta

    _write(canonical_mesh, canonical_scaled, canonical_out_glb)
    _write(posed_mesh, posed_scaled, posed_out_glb)

    # Validate the serialized files in the preserved frame. Translation does not affect
    # extents, but explicitly removing each anchor delta also makes the frame provenance
    # auditable and prevents a later implementation from silently changing the contract.
    canonical_check_mesh, canonical_check, canonical_check_faces = _load(
        canonical_out_glb
    )
    del canonical_check_mesh
    posed_check_mesh, posed_check, posed_check_faces = _load(posed_out_glb)
    del posed_check_mesh
    anchor_tolerance = max(1e-5, 1e-4 * float(target[0]))
    for label, before, after in (
        ("metric pristine", canonical_world, canonical_check),
        ("posed", posed_world, posed_check),
    ):
        if not np.allclose(
            _anchor(after), _anchor(before), rtol=0.0, atol=anchor_tolerance
        ):
            raise ValueError(f"{label} bottom-centre anchor changed")
    canonical_fixed = trimesh.transform_points(
        canonical_check - canonical_delta, to_origin
    )
    posed_check_canonical = (posed_check - posed_delta - translation) @ rotation
    posed_fixed = trimesh.transform_points(posed_check_canonical, to_origin)
    canonical_achieved = np.ptp(canonical_fixed, axis=0)[order]
    posed_achieved = np.ptp(posed_fixed, axis=0)[order]
    for label, achieved in (
        ("metric pristine", canonical_achieved),
        ("posed", posed_achieved),
    ):
        if not np.allclose(achieved, target, rtol=0.002, atol=1e-5):
            raise ValueError(
                f"{label} fixed-frame extents missed target: "
                f"{achieved.tolist()} vs {target.tolist()}"
            )

    post_similarity = corner_similarity(
        canonical_check[canonical_check_faces].reshape(-1, 3),
        posed_check[posed_check_faces].reshape(-1, 3),
        tol=float(correspondence_tol),
    )
    if post_similarity is None:
        raise ValueError("scaled pristine/posed outputs lost shape correspondence")
    post_scale, post_rotation, post_translation = post_similarity
    post_residual = float(
        np.abs(
            canonical_check[canonical_check_faces].reshape(-1, 3)
            @ (post_scale * post_rotation).T
            + post_translation
            - posed_check[posed_check_faces].reshape(-1, 3)
        ).max()
    )
    if not np.isclose(post_scale, 1.0, rtol=0.002, atol=1e-5):
        raise ValueError(
            f"scaled pristine/posed outputs differ in metric scale: {post_scale}"
        )

    def _result(world: np.ndarray, achieved: np.ndarray, path: str) -> dict:
        return {
            "out_glb": path,
            "size": [float(x) for x in np.ptp(world, axis=0)],
            "obb_sorted": [float(x) for x in achieved],
        }

    return {
        "canonical": _result(canonical_check, canonical_achieved, canonical_out_glb),
        "posed": _result(posed_check, posed_achieved, posed_out_glb),
        "source_similarity_scale": float(source_scale),
        "post_similarity_scale": float(post_scale),
        "post_similarity_max_residual_m": post_residual,
        "measurement_frame": "metric_pristine_frozen_obb",
    }


def rescale_glb_to_extents(in_glb: str, out_glb: str, target_sorted) -> dict:
    """Anisotropically rescale an already-placed GLB so its OBB extents match
    ``target_sorted`` (``[long, mid, short]`` metres), scaling ALONG the mesh's own OBB
    axes — so width/depth/height are set independently of world yaw — about the object's
    BOTTOM-CENTER (base stays put; the physics settle re-rests it). Preserves the
    gltf-import pre-rotation. Returns the achieved AABB ``size`` and sorted OBB extents."""
    import os

    import trimesh
    from trimesh.bounds import oriented_bounds

    mesh = trimesh.load(in_glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    to_origin, extents = oriented_bounds(world)  # extents along local x,y,z
    tgt = np.asarray(target_sorted, dtype=np.float64)
    order = np.argsort(extents)[::-1]  # local axes ranked long -> short
    f = np.ones(3)
    for rank, axis in enumerate(order):
        f[axis] = tgt[rank] / max(float(extents[axis]), 1e-6)
    local = trimesh.transform_points(world, to_origin)
    local = local * f
    world_new = trimesh.transform_points(local, np.linalg.inv(to_origin))
    # Anchor: keep the original xy centre and bottom (min z) fixed.
    lo0, hi0 = world.min(axis=0), world.max(axis=0)
    lo1, hi1 = world_new.min(axis=0), world_new.max(axis=0)
    anchor0 = np.array([(lo0[0] + hi0[0]) / 2, (lo0[1] + hi0[1]) / 2, lo0[2]])
    anchor1 = np.array([(lo1[0] + hi1[0]) / 2, (lo1[1] + hi1[1]) / 2, lo1[2]])
    world_new = world_new + (anchor0 - anchor1)
    mesh.vertices = _pre_rotate_for_gltf(world_new)
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    aabb = world_new.max(axis=0) - world_new.min(axis=0)
    return {
        "out_glb": out_glb,
        "size": [float(x) for x in aabb],
        "obb_sorted": [float(x) for x in np.sort(extents * f)[::-1]],
    }


def frame_glb_native(
    in_glb: str, out_glb: str, for_gltf_import: bool = True, R=None
) -> dict:
    """Place a SAM3D GLB at SAM3D's OWN estimated transform (no MoGE anchoring).

    SAM3D's native X position is correct, but its mesh geometry is internally
    X-mirrored. So we frame-convert WITHOUT the global X flip (keeps the position
    un-mirrored) and instead reflect each mesh about its OWN centroid X — fixing
    chirality (e.g. mug handle) while leaving the object where SAM3D placed it.
    Scale, depth and rotation are SAM3D's estimates, untouched."""
    import os

    import trimesh

    mesh = trimesh.load(in_glb, force="mesh")
    world = sam3d_to_world(
        mesh.vertices, flip_x=False
    )  # correct position, mirrored geom
    cx = 0.5 * (world[:, 0].min() + world[:, 0].max())
    world[:, 0] = 2.0 * cx - world[:, 0]  # reflect about own X -> fix chirality
    if R is not None:  # gravity-align position + orientation
        world = world @ np.asarray(R, dtype=world.dtype).T
    mesh.vertices = _pre_rotate_for_gltf(world) if for_gltf_import else world
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    c = (world.min(0) + world.max(0)) / 2.0
    return {"out_glb": out_glb, "world_center": [float(x) for x in c]}


def _set_glb_double_sided(path: str) -> None:
    """Mark every material in a binary GLB ``doubleSided`` (in place). SAM3D's single-sided
    (``doubleSided=False``) materials make Blender backface-cull open shells, so meshes whose
    reconstructed normals face away render see-through; double-sided renders both faces."""
    import json
    import struct

    with open(path, "rb") as f:
        data = bytearray(f.read())
    if bytes(data[:4]) != b"glTF":  # not a binary GLB; nothing to patch
        return
    total = struct.unpack("<I", data[8:12])[0]
    off, json_start, json_end = 12, None, None
    while off < total:
        clen, ctype = struct.unpack("<II", data[off : off + 8])
        if ctype == 0x4E4F534A:  # 'JSON'
            json_start, json_end = off + 8, off + 8 + clen
            break
        off += 8 + clen
    if json_start is None:
        return
    doc = json.loads(bytes(data[json_start:json_end]))
    mats = doc.get("materials") or []
    if not mats or all(m.get("doubleSided") for m in mats):
        return
    for m in mats:
        m["doubleSided"] = True
    new_json = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((-len(new_json)) % 4)  # 4-byte pad with spaces (glTF spec)
    rest = bytes(data[json_end:])  # BIN chunk header+data, unchanged
    body = struct.pack("<II", len(new_json), 0x4E4F534A) + new_json + rest
    with open(path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)


def place_glb(
    in_glb: str,
    out_glb: str,
    center,
    size,
    to_world: bool = True,
    for_gltf_import: bool = True,
    flip_x: bool = False,
    R=None,
    backend: str = "sam3d",
    flip_euler=None,
    use_obb: bool = False,
) -> dict:
    """Load a reconstructed GLB, place it at the MoGE transform, write the placed GLB.

    ``flip_x=False`` (default) is the proper-rotation, source-view-matching convention (see
    :func:`place_mesh_vertices`) -- deterministically upright, no VLM flip needed.
    ``for_gltf_import`` pre-rotates so Blender's glTF importer lands the mesh at
    the intended Z-up world coords (the demo's GLB scene-export then re-applies
    Z-up->Y-up, so the web viewer is correct too). ``backend`` selects the source
    frame: ``'sam3d'`` (view-frame, gravity-aligned by ``R``) or
    ``'canonical_y_up'`` (already gravity-aligned; ``R`` ignored). The ``center``
    is already gravity-aligned."""
    import os

    import trimesh

    mesh = trimesh.load(in_glb, force="mesh")
    # When ``use_obb``, scale by the mesh's rotation-invariant 3D OBB diagonal (``size`` is
    # then the cloud OBB box). Fall back to the AABB path if the OBB solve fails (rare).
    ext_diag = None
    if use_obb:
        try:
            ext_diag = float(
                np.linalg.norm(mesh.bounding_box_oriented.primitive.extents)
            )
        except Exception:  # noqa: BLE001
            ext_diag = None
    world = place_mesh_vertices(
        mesh.vertices,
        center,
        size,
        to_world=to_world,
        flip_x=flip_x,
        R=R,
        backend=backend,
        flip_euler=flip_euler,
        ext_diag=ext_diag,
    )
    placed_center = [float(x) for x in (world.min(0) + world.max(0)) / 2.0]
    placed_extents = [float(x) for x in (world.max(0) - world.min(0))]
    mesh.vertices = _pre_rotate_for_gltf(world) if for_gltf_import else world
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    # SAM3D raw GLBs are open shells with single-sided (doubleSided=False) materials; on
    # import Blender backface-culls them, so any mesh whose reconstructed surface normals
    # point away from the camera renders transparent / see-through (e.g. the tissue box's
    # slot shows through to the underside). We can't reliably reorient open-shell normals
    # (signed volume is ambiguous and flipping it breaks meshes that were already fine, like
    # the laptop), so instead mark the materials doubleSided so both faces always render --
    # fixes the see-through artifact without touching geometry/orientation.
    _set_glb_double_sided(out_glb)
    return {
        "out_glb": out_glb,
        "placed_center": placed_center,
        "placed_extents": placed_extents,
    }
