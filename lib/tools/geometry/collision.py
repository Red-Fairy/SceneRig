"""CoACD convex decomposition of placed GLBs into world-frame collider parts.

A convex hull of a concave object (open laptop, mug, bowl) fills its cavities, so a body
resting *in* or *against* the concavity sits on phantom volume and the settle tips or
ejects it. CoACD (collision-aware approximate convex decomposition, Wei et al. 2022)
splits the mesh into a small set of convex pieces whose union hugs the real surface;
physics engines then collide against the pieces as a compound body — exactly (each piece
is convex), fast, and with a cavity that actually exists.

Parts are decomposed fresh from the current GLB at preprocess (placed GLBs are mutated in
place by settle / pose-match, so blindly cached parts would go stale). Later sessions
(composition boot, certify) REUSE the preprocess decomposition instead of re-cooking: the
exact pose delta is recovered per object via ``corner_similarity`` on face-corner
correspondence, which self-invalidates (falls back to a fresh cook) whenever the geometry
actually changed. This keeps the settle ladder's validated collider choice — a fresh CoACD
of the same object can rest differently (0715 wendy1 marker capsized on a re-cook of a
pristine-rescued collider) — and saves ~27 s/object/session. Parts are saved in the Z-up
world frame the physics runs in (``_unpre_rotate_for_gltf`` — same frame contract as the
grounding in ``physics.incremental_settle``). CoACD's ``auto`` preprocessing voxel-remeshes
the open SAM3D shells to a watertight solid first, which also makes PhysX's density-derived
mass/COM/inertia come from a solid, not a shell.

Consumers: ``physics.incremental_settle`` / ``composition_physics`` and
``isaac/build_collision.py`` -> ``isaac/isaac_add_physics.py`` (authors the parts as USD
collision prims).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

# Concavity tolerance (normalized 0.01-1). 0.05 keeps tabletop objects at ~2-15 parts;
# lower is more faithful but slower to decompose and to simulate.
DEFAULT_THRESHOLD = 0.05
# Part cap: bumpy organic shapes (plush toys) otherwise shatter into 60+ hulls whose
# surface bumps carry no contact information but multiply every scene reset's collision
# cost. CoACD keeps the most significant cuts when capped.
# 32 since 2026-08-27 (was 16): 51% of the 0825 batch's objects were truncated at 16, and a
# 119-object probe showed 32 costs nothing (0.95x wall-clock). Ring-bottomed objects (mugs,
# cups) get several base hulls instead of one — a single spanning base hull creeps under
# PhysX at physical mu (roborig_depth/CHANGELOG 2026-08-27). The plush-toy rationale above
# still holds at 32: CoACD stops on the concavity threshold long before the cap for most.
MAX_PARTS = 32
# coacd's own default, kept explicit for contrast with the support tier below: the
# auto-preprocess voxel remesh runs at 50 cells across the bbox, ~10 mm on a 0.5 m
# container — coarse enough to close a wire weave into a solid before decomposition.
DEFAULT_PREP_RESOLUTION = 50
# Finer budget for SUPPORT objects (anything cargo rests in/on: basket, tray, plate,
# mug). At the default budget a container's cavity gets bridged by solid wedges —
# 0722_bulkfix_groot2's wire basket carried 44 mm median phantom fill over its floor
# plus cm-scale part-seam holes, leaving the pepper floating on fill and the corn
# wedged over a gap. Benchmarked on that scene: this tier costs +19-53 s per support
# (basket 28->47 s) but is 3-440x slower on organic NON-supports (bell peppers) for
# zero contact value — hence the gate. Non-supports rest ON things; only supports
# need faithful cavities.
SUPPORT_THRESHOLD = 0.02
SUPPORT_MAX_PARTS = 64
# Support prep resolution is VOXEL-TARGETED, not fixed (2026-08-22). CoACD's
# preprocess_resolution is cells across the longest bbox extent, so a fixed 100
# meant ~5.3 mm voxels on benchmark_0822_rerun2/abc_5's 0.53 m tray: the remesh
# quantization put a +3.1 mm median / +7.8 mm p90 margin over the tray floor with
# 5.9% phantom-wall cells (+37..66 mm) narrowing the cargo strip — bottles could
# never stand there (39 deg implied lean; 13 preprocess re-drops). Targeting
# SUPPORT_TARGET_VOXEL_M instead: resolution = ceil(extent / 0.003), clamped to
# [100, 200] — the floor keeps small supports at (or above) the old fidelity, and
# the cap bounds the O(res^3) remesh cost on metre-scale supports. The rule keys
# off the extent of the mesh being COOKED: on the pristine-first path that is the
# pristine (pcand) sibling, which today shares metric scale with the placed GLB
# by the placement contract (preprocess places and same-size-rescales both
# together), so the derived placed collider's effective voxel is ~3 mm too. If
# that contract ever loosens (pristine->placed scale s != 1), the target merely
# drifts to s * 3 mm — corner_similarity recovers any s exactly, so geometry
# stays correct regardless.
SUPPORT_TARGET_VOXEL_M = 0.003
SUPPORT_PREP_RESOLUTION_MIN = 100
SUPPORT_PREP_RESOLUTION_MAX = 200
# Cap-hit escalation: a support decomposition returning EXACTLY SUPPORT_MAX_PARTS
# means CoACD truncated its cut tree (abc_5's tray hit 64 exactly and its floor
# strip kept phantom walls a further cut would have opened). One bounded retry at
# this cap, supports only (decompose_world_mesh logs the old/new part counts).
SUPPORT_ESCALATED_MAX_PARTS = 128


def support_prep_resolution(max_extent_m: float) -> int:
    """Support-tier CoACD preprocess resolution: cells across the longest bbox
    extent targeting ``SUPPORT_TARGET_VOXEL_M`` voxels (derivation at the
    constants above)."""
    cells = int(np.ceil(max_extent_m / SUPPORT_TARGET_VOXEL_M))
    return min(max(cells, SUPPORT_PREP_RESOLUTION_MIN), SUPPORT_PREP_RESOLUTION_MAX)


def decompose_glb(
    glb: str, out_npz: str, support: bool = False, frame_glb: str | None = None
) -> int:
    """Decompose a placed GLB into convex parts in the Z-up world frame.

    Saves ``out_npz`` with ``n`` (part count) and per-part ``v{i}`` (float64 [k,3] world
    vertices) / ``f{i}`` (int64 [m,3] faces). Returns the part count. ``support``
    selects the finer SUPPORT_* budget (objects other objects rest in/on).
    ``frame_glb``: the object's PRISTINE sibling for the exact local-frame overshoot
    clamp (see ``_pristine_frame``)."""
    import trimesh

    from lib.tools.geometry.mesh_place import _unpre_rotate_for_gltf

    mesh = trimesh.load(glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    return decompose_world_mesh(world, mesh.faces, out_npz, support, frame_glb)


def _pristine_frame(world, faces, frame_glb):
    """Exact LOCAL clamp frame from the SAM3D pristine sibling (owner insight
    2026-08-05: SAM3D's pristine save is axis-aligned in its own frame — a heuristic
    that holds in practice). Pristine, raw, and placed GLBs are the SAME mesh under
    similarity transforms (identical topology), so face-corner correspondence
    recovers the placed pose's rotation from the axis-aligned frame EXACTLY — no
    PCA, no estimation error. Returns ``(s, R, t, lo, hi)`` with the placed VISUAL
    verts' box in that frame, or None (missing sibling / broken correspondence,
    e.g. a non-uniform mesh edit — the tol guard makes the fallback safe by
    construction)."""
    if not frame_glb or not os.path.exists(frame_glb):
        return None
    try:
        src = glb_face_corners(frame_glb)
        dst = np.asarray(world, dtype=np.float64)[
            np.asarray(faces, dtype=np.int64)
        ].reshape(-1, 3)
        sim = corner_similarity(src, dst)
        if sim is None:
            return None
        s, R, t = sim
        P = (world - t) @ R / s  # placed visual verts in the pristine frame
        return s, R, t, P.min(0), P.max(0)
    except Exception:  # noqa: BLE001 - the clamp is an enhancement, never fatal
        return None


# CoACD's MCTS is stochastic: unseeded, byte-identical meshes produced a different
# decomposition on every call (2026-08-16, replay_real_trajectory repro: banana
# 384b0837ea vs 1657ab0370, bowl three different hashes in three runs), which made
# grasp outcomes irreproducible. A fixed seed makes the decomposition a pure
# function of the mesh. Its default of 0 in run_coacd is NOT a fixed seed.
COACD_SEED = 20260816
# Opt-in hull vertex cap. CoACD (decimate=False) emits hulls at raw surface-vertex density
# (2,600-27,000 verts seen on eibin objects); PhysX then cooks every convexHull to 64 verts
# (physxConvexHullCollision:hullVertexLimit default), so the simulated shape is NOT the
# authored one. On flat-bottomed objects the cooked base is no longer planar and resting
# objects creep at constant velocity (eibin ae341146 mug 6 mm/s, apple ~10 mm/s, 2026-08-27).
# Capping at authoring makes cooking an identity. Set GRASE_COACD_MAX_VERTS=64 to enable.
COACD_MAX_VERTS = int(os.environ.get("GRASE_COACD_MAX_VERTS", "0"))


def coacd_diagnostic_root(out_npz: str) -> Path:
    """Return the output-local directory containing retained native cook inputs."""
    output = Path(out_npz)
    return output.parent / "coacd_diagnostics" / output.stem


@lru_cache(maxsize=4)
def _coacd_file_identity(path: str, size: int, mtime_ns: int) -> dict:
    """Hash each installed binary once per worker while its file identity is stable."""
    return {
        "path": path,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "size_bytes": size,
        "mtime_ns": mtime_ns,
    }


def _run_coacd_recorded(
    world: np.ndarray, faces: np.ndarray, out_npz: str, kwargs: dict
) -> list:
    """Retain submitted arrays and library identity if a native cook exits abruptly.

    Native stderr retains its existing owning tool/core-log destination. Successful
    calls remove only their own temporary evidence, never an earlier failed attempt.
    """
    import coacd  # Optional native dependency: imported only at the cooking boundary.

    mesh = coacd.Mesh(world, faces)
    root = coacd_diagnostic_root(out_npz)
    root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=root))
    source = attempt / "input.npz"
    np.savez(source, vertices=mesh.vertices, faces=mesh.indices)
    identities = {}
    for name, path in (
        ("python_wrapper", coacd.__file__),
        ("native_library", coacd._lib._name),
    ):
        path = str(Path(path).resolve())
        stat = Path(path).stat()
        identities[name] = _coacd_file_identity(path, stat.st_size, stat.st_mtime_ns)
    metadata = {
        "schema_version": 1,
        "state": "entering_native",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_npz": str(Path(out_npz).resolve()),
        "input_npz": "input.npz",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "vertices_shape": list(mesh.vertices.shape),
        "vertices_dtype": str(mesh.vertices.dtype),
        "faces_shape": list(mesh.indices.shape),
        "faces_dtype": str(mesh.indices.dtype),
        "input_semantics": "coacd.Mesh arrays; checker normalization/casting occurs internally",
        "call_kwargs": kwargs,
        "unspecified_arguments": "defaults from the recorded Python wrapper",
        "package_version": importlib.metadata.version("coacd"),
        "installed_files_at_entry": identities,
        "stderr": "inherited owning tool/core log; no per-cook stderr file",
    }
    (attempt / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[collision] native cook {Path(out_npz).stem}: pending input evidence {attempt}",
        file=sys.stderr,
        flush=True,
    )
    parts = coacd.run_coacd(mesh, **kwargs)
    shutil.rmtree(attempt)
    return parts


def decompose_world_mesh(
    world: np.ndarray,
    faces: np.ndarray,
    out_npz: str,
    support: bool = False,
    frame_glb: str | None = None,
    seed: int = COACD_SEED,
) -> int:
    """Decompose an already-world-frame mesh (e.g. dumped from the exported USD, which
    is the authoritative geometry — GLB poses can be stale after agent stages).
    ``frame_glb`` enables the exact local-frame clamp (``_pristine_frame``)."""
    import coacd

    coacd.set_log_level("error")
    world = np.asarray(world, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if support:
        threshold, max_parts = SUPPORT_THRESHOLD, SUPPORT_MAX_PARTS
        prep = support_prep_resolution(float((world.max(0) - world.min(0)).max()))
    else:
        threshold, max_parts = DEFAULT_THRESHOLD, MAX_PARTS
        prep = DEFAULT_PREP_RESOLUTION
    kwargs = dict(
        threshold=threshold,
        max_convex_hull=max_parts,
        preprocess_resolution=prep,
        seed=seed,
        **(
            {"decimate": True, "max_ch_vertex": COACD_MAX_VERTS}
            if COACD_MAX_VERTS
            else {}
        ),
    )
    parts = _run_coacd_recorded(world, faces, out_npz, kwargs)
    if support and len(parts) == SUPPORT_MAX_PARTS:
        # Exactly at the cap = CoACD truncated the cut tree (see the constant's
        # note). ONE bounded retry with more part headroom, supports only.
        parts = _run_coacd_recorded(
            world,
            faces,
            out_npz,
            {**kwargs, "max_convex_hull": SUPPORT_ESCALATED_MAX_PARTS},
        )
        print(
            f"[collision] {os.path.basename(out_npz)}: support decomposition hit "
            f"the {SUPPORT_MAX_PARTS}-part cap; retried once at "
            f"{SUPPORT_ESCALATED_MAX_PARTS} -> {len(parts)} parts",
            file=sys.stderr,
        )
    return _save_parts(parts, world, faces, out_npz, frame_glb)


def _save_parts(
    parts: list, world: np.ndarray, faces: np.ndarray, out_npz: str, frame_glb: str | None
) -> int:
    """Clamp cooked parts to the visual mesh and write the ``{n, v{i}, f{i}}`` npz."""
    # Clamp part vertices to the visual mesh's z-RANGE: the voxel remesh overshoots
    # the surface by ~1 mm (up to ~half a voxel) on EVERY face. Bottom clamp — an
    # object authored in exact resting contact would otherwise start its collider
    # penetrated into the support, and PhysX's depenetration nudge makes narrow
    # objects lean at sim start (8334 glue bottle). Top clamp (2026-08-04, owner-
    # approved) — the mirror interface: a NEIGHBOR resting on this object starts
    # penetrated into the top overshoot instead. Stacks of thin slabs get it from
    # both faces (0804_e2e_online6: book_0's collider top sat 2.7mm inside book_1 on
    # a 30mm book) — joint settles absorb that mutually, but a single-object
    # move-settle turns it into an ejection impulse: the moved book flipped 60-174deg
    # and EVERY composition move on a stacked book was capsize-rejected. Parts stay
    # convex (PhysX hulls the vertices), and only the systematic stacking axis (z)
    # matters — lateral overshoot carries no resting weight and is left alone.
    # LOCAL-frame clamp first (2026-08-05, owner-approved): the world clamp planes
    # sit at the AABB extremes, so on a TILTED slab they trim only the high-corner
    # region and ~90% of the face keeps its overshoot (0804_e2ef books: collider
    # tops +1.1mm above visual after settle -> 0.8mm visible inter-book margins).
    # The pristine-frame clamp trims uniformly across every face and is
    # pose-invariant, so the transform-reuse machinery propagates it exactly.
    frame = _pristine_frame(world, faces, frame_glb)
    z_min = float(world[:, 2].min())
    z_max = float(world[:, 2].max())
    data: dict[str, np.ndarray] = {"n": np.asarray(len(parts))}
    for i, (v, f) in enumerate(parts):
        v = np.asarray(v, dtype=np.float64).copy()
        if frame is not None:
            s, R, t, lo, hi = frame
            v = np.clip((v - t) @ R / s, lo, hi) @ (s * R).T + t
        # world z-clamp stays as the OUTER invariant: the tilted local box's bottom
        # corners can dip below the visual bottom plane, and the 8334 resting-contact
        # guarantee (collider never below the support plane) must survive verbatim.
        v[:, 2] = np.clip(v[:, 2], z_min, z_max)
        data[f"v{i}"] = v
        data[f"f{i}"] = np.asarray(f, dtype=np.int64)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, **data)
    return len(parts)


# Agent-authored objects: one convex hull PER AUTHORED PART instead of CoACD on the fused
# union (2026-09-17, owner-approved). CoACD re-slices the union along its own planes and
# handed clutter_shelf's base slab back as a 46-vertex hull with a bulged top; resting on
# 5 of its vertices PhysX pumped energy in every step and the shelf launched 1.2 m in the
# metric sim (cap 24/64 + OMP_NUM_THREADS=1, corner filters, subsampling all still
# flipped; four clean corners for either face held 0.0 mm). The agent's part boundaries
# are the cut planes CoACD cannot infer from the fused surface, and primitives hull to
# a handful of vertices (a box slab -> 8). A part whose mesh volume is below this
# fraction of its hull volume is non-convex (torus rim, carved cup, cut slab) and is
# CoACD'd ALONE, so no cut can ever span two parts. 189/196 gpt6v1_0917 parts are >=0.95.
AUTHORED_HULL_FILL_RATIO = 0.9


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    t = vertices[faces] - vertices.mean(0)
    return abs(float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6)


def decompose_authored_parts(
    part_meshes: list[tuple[np.ndarray, np.ndarray]],
    world: np.ndarray,
    faces: np.ndarray,
    out_npz: str,
    frame_glb: str | None = None,
) -> int:
    """Per-part hulls for an authored object; ``world``/``faces`` is the fused union the
    clamps and any whole-object fallback use. Non-convex parts are CoACD'd alone; if the
    hull set would exceed ``MAX_PARTS`` the whole union falls back to ``decompose_world_mesh``.
    """
    from scipy.spatial import ConvexHull

    world = np.asarray(world, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    coacd_parts = []
    for pv, pf in part_meshes:
        pv = np.asarray(pv, dtype=np.float64)
        pf = np.asarray(pf, dtype=np.int64)
        hull = ConvexHull(pv)
        if _mesh_volume(pv, pf) >= AUTHORED_HULL_FILL_RATIO * hull.volume:
            hv = pv[hull.vertices]
            parts.append((hv, ConvexHull(hv).simplices.astype(np.int64)))
            continue
        coacd_parts.append((pv, pf))
    if coacd_parts:
        import coacd

        coacd.set_log_level("error")
        kwargs = dict(
            threshold=DEFAULT_THRESHOLD,
            max_convex_hull=MAX_PARTS,
            preprocess_resolution=DEFAULT_PREP_RESOLUTION,
            seed=COACD_SEED,
        )
        for pv, pf in coacd_parts:
            parts.extend(_run_coacd_recorded(pv, pf, out_npz, kwargs))
    if len(parts) > MAX_PARTS:
        print(
            f"[collision] {os.path.basename(out_npz)}: {len(parts)} per-part hulls exceed "
            f"the {MAX_PARTS}-part cap; cooking the fused union instead",
            file=sys.stderr,
        )
        return decompose_world_mesh(world, faces, out_npz, False, frame_glb)
    print(
        f"[collision] {os.path.basename(out_npz)}: authored object -> {len(parts)} hulls "
        f"({len(part_meshes) - len(coacd_parts)} convex parts, {len(coacd_parts)} CoACD'd)",
        file=sys.stderr,
    )
    return _save_parts(parts, world, faces, out_npz, frame_glb)


def dump_part_meshes(d, prefix: str = "part_") -> list[tuple[np.ndarray, np.ndarray]]:
    """Authored part meshes stored in a dump as ``<prefix>v{i}``/``<prefix>f{i}`` with
    ``n_parts`` (register server) — empty for a plain single-mesh dump."""
    count = int(d["n_parts"]) if "n_parts" in d else 0
    return [(d[f"{prefix}v{i}"], d[f"{prefix}f{i}"]) for i in range(count)]


def decompose_dump(
    dump_npz: str,
    out_npz: str,
    support: bool = False,
    frame_glb: str | None = None,
) -> int:
    """``decompose_world_mesh`` on a register-server mesh dump ({v0,f0}). File-path
    signature so boot-time cache misses can run in spawn workers (CoACD in-process
    poisons later torch/scipy imports, and 12 sequential cooks cost ~5 min)."""
    d = np.load(dump_npz)
    part_meshes = dump_part_meshes(d)
    if part_meshes and not support:
        return decompose_authored_parts(part_meshes, d["v0"], d["f0"], out_npz, frame_glb)
    return decompose_world_mesh(d["v0"], d["f0"], out_npz, support, frame_glb)


def load_parts(npz: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load ``decompose_glb`` output back as ``[(world_vertices, faces), ...]``."""
    d = np.load(npz)
    return [(d[f"v{i}"], d[f"f{i}"]) for i in range(int(d["n"]))]


def glb_face_corners(glb: str) -> np.ndarray:
    """Z-up world face-corner positions [3*n_faces, 3]. Face corners — not vertices —
    are the correspondence unit that survives Blender roundtrips: import/export
    splits vertices (per-corner normals/UVs) but preserves triangle order."""
    import trimesh

    from lib.tools.geometry.mesh_place import _unpre_rotate_for_gltf

    mesh = trimesh.load(glb, force="mesh")
    world = _unpre_rotate_for_gltf(np.asarray(mesh.vertices, dtype=np.float64))
    return world[np.asarray(mesh.faces, dtype=np.int64)].reshape(-1, 3)


def corner_similarity(
    src: np.ndarray, dst: np.ndarray, tol: float = 0.002
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Exact similarity (s, R, t) with ``dst ~= s*R@src + t`` from corresponding
    face-corner arrays (closed-form Umeyama). Returns None when the correspondence
    is broken — count mismatch or max residual above ``tol`` (meters) — so callers
    fall back to a fresh decomposition instead of trusting a wrong transform."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or len(src) < 3:
        return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(a))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    s = float((S * [1.0, 1.0, d]).sum() / ((a * a).sum() / len(a)))
    t = mu_d - s * R @ mu_s
    if float(np.abs(src @ (s * R).T + t - dst).max()) > tol:
        return None
    return s, R, t


def transform_parts_npz(
    src_npz: str,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
    out_npz: str,
    *,
    pre_transform_flatten_mm: float | None = None,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Apply a similarity to every convex part of a collider npz (convexity is
    preserved under similarity, so the validated decomposition stays valid).
    ``pre_transform_flatten_mm`` applies the stabilization bundle's base clamp in
    the SOURCE frame before mapping.  That ordering is material: flattening the
    already-rotated destination collider against world Z selects a different base
    slice, so it does not reproduce the validated stabilized body.
    ``z_range`` re-applies the destination frame's world z-clamp after the
    transform (the 8334 resting-contact invariant: a tilted part's lateral
    overshoot corners can dip below the destination visual bottom plane)."""
    d = np.load(src_npz)
    parts = [
        (
            np.asarray(d[f"v{i}"], dtype=np.float64),
            np.asarray(d[f"f{i}"], dtype=np.int64),
        )
        for i in range(int(d["n"]))
    ]
    if pre_transform_flatten_mm is not None:
        value = float(pre_transform_flatten_mm)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "pre-transform flatten distance must be finite and nonnegative"
            )
        if value:
            from lib.tools.geometry.settle_geometry import flatten_base

            parts = flatten_base(parts, value)
    data: dict[str, np.ndarray] = {"n": d["n"]}
    for i, (source_vertices, source_faces) in enumerate(parts):
        v = source_vertices @ (s * R).T + t
        if z_range is not None:
            v[:, 2] = np.clip(v[:, 2], z_range[0], z_range[1])
        data[f"v{i}"] = v
        data[f"f{i}"] = source_faces
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, **data)


def _quat_to_matrix_xyzw(value) -> np.ndarray:
    """Finite normalized ``[x,y,z,w]`` quaternion to a rotation matrix."""
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("principal_axes must be a finite [x,y,z,w] quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("principal_axes quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(value: np.ndarray) -> list[float]:
    """Proper rotation matrix to a deterministic ``[x,y,z,w]`` quaternion."""
    M = np.asarray(value, dtype=np.float64)
    if M.shape != (3, 3) or not np.isfinite(M).all():
        raise ValueError("principal-axis rotation must be a finite 3x3 matrix")
    trace = float(np.trace(M))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                (M[2, 1] - M[1, 2]) / scale,
                (M[0, 2] - M[2, 0]) / scale,
                (M[1, 0] - M[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        i = int(np.argmax(np.diag(M)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = math.sqrt(max(1.0 + M[i, i] - M[j, j] - M[k, k], 1e-12)) * 2.0
        q = np.array([0.0, 0.0, 0.0, (M[k, j] - M[j, k]) / scale])
        q[i] = 0.25 * scale
        q[j] = (M[j, i] + M[i, j]) / scale
        q[k] = (M[k, i] + M[i, k]) / scale
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("transformed principal-axis quaternion is invalid")
    q /= norm
    # q and -q encode the same basis.  Fix the largest component's sign so audit
    # records and tests remain deterministic even for a rotation near pi.
    if q[int(np.argmax(np.abs(q)))] < 0.0:
        q *= -1.0
    return [float(item) for item in q]


def transform_stabilization_bundle(
    bundle: dict,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> dict:
    """Carry a complete world-frame stabilization bundle through a similarity.

    For ``x' = s R x + t``, the CoM follows the point transform, mass scales as
    ``s^3``, principal moments as ``s^5``, and the world principal basis becomes
    ``R Q``.  This mirrors ``isaac_settle_server._apply_delta`` exactly and avoids
    importing scipy in a process which may already have run CoACD.
    """
    if not isinstance(bundle, dict):
        raise ValueError("stabilization bundle must be an object")
    scale = float(s)
    rotation = np.asarray(R, dtype=np.float64)
    translation = np.asarray(t, dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("stabilization similarity scale must be finite and positive")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("stabilization similarity rotation must be finite 3x3")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("stabilization similarity translation must be finite length 3")
    if (
        float(np.abs(rotation.T @ rotation - np.eye(3)).max()) > 1e-6
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6
    ):
        raise ValueError("stabilization similarity rotation must be proper orthonormal")

    com = np.asarray(bundle.get("com_world"), dtype=np.float64)
    inertia = np.asarray(bundle.get("diagonal_inertia"), dtype=np.float64)
    try:
        mass = float(bundle.get("solver_mass_kg"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stabilization solver_mass_kg must be finite and positive"
        ) from exc
    if com.shape != (3,) or not np.isfinite(com).all():
        raise ValueError("stabilization com_world must be finite length 3")
    if (
        inertia.shape != (3,)
        or not np.isfinite(inertia).all()
        or np.any(inertia <= 0.0)
    ):
        raise ValueError("stabilization diagonal_inertia must contain three positives")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("stabilization solver_mass_kg must be finite and positive")
    axes = _quat_to_matrix_xyzw(bundle.get("principal_axes"))

    out = dict(bundle)
    out["com_world"] = [float(item) for item in scale * rotation @ com + translation]
    out["diagonal_inertia"] = [float(item) for item in inertia * scale**5]
    out["principal_axes"] = _matrix_to_quat_xyzw(rotation @ axes)
    out["solver_mass_kg"] = float(mass * scale**3)
    if bundle.get("flatten_base_mm") is not None:
        try:
            flatten = float(bundle["flatten_base_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stabilization flatten_base_mm must be nonnegative"
            ) from exc
        if not np.isfinite(flatten) or flatten < 0.0:
            raise ValueError("stabilization flatten_base_mm must be nonnegative")
        out["flatten_base_mm"] = float(flatten * scale)
    return out


def placed_collider_from_pristine(
    pristine_npz: str, pristine_glb: str, placed_glb: str, out_npz: str
) -> bool:
    """Derive the PLACED collider by similarity-transforming the PRISTINE
    decomposition — never CoACD the placed mesh of a support (2026-08-22).

    CoACD's voxel remesh is grid-aligned with the axis-aligned pristine save, so
    the pristine decomposition's floor is flat at a uniform quantization offset;
    remeshing the same mesh TILTED staircases the floor across the grid instead
    (abc_5's ~4 deg tray: +3.1 mm median margin, 12.9 mm plane residual, 5.9%
    phantom-wall cells vs a 2-6x flatter pristine cook). Pristine and placed GLBs
    are the SAME mesh under a similarity (owner insight 2026-08-05), so
    ``corner_similarity`` recovers the placed pose exactly and convexity survives
    the transform (``transform_parts_npz``). The pristine cook's world z-clamp IS
    its exact local-frame clamp (the pristine is axis-aligned in world), so it
    propagates exactly; the placed frame's z-clamp is re-applied as the outer
    resting-contact invariant. Returns False — writing nothing — on a broken
    correspondence (mismatched topology, missing file), so callers fall back to
    a fresh placed decomposition; this path must never hard-fail preprocess."""
    try:
        dst = glb_face_corners(placed_glb)
        sim = corner_similarity(glb_face_corners(pristine_glb), dst)
        if sim is None:
            return False
        transform_parts_npz(
            pristine_npz,
            *sim,
            out_npz,
            z_range=(float(dst[:, 2].min()), float(dst[:, 2].max())),
        )
        return True
    except Exception:  # noqa: BLE001 - fall back to a fresh placed cook, never fatal
        return False
