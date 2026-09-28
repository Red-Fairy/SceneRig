"""Point-cloud pose matching: refine a placed object's similarity transform against its
masked MoGE point cloud (SimFoundry's "Pose Matching" stage, adapted to monocular input).

The cloud is a 2.5D FRONT SHELL (only the camera-facing surface), so the fit is strictly
one-directional — cloud points to nearest point ON THE MESH SURFACE — never mesh-to-cloud
(that direction drags the mesh centroid into the shell). DoF are gravity- AND
contact-restricted: [x, y, yaw, isotropic scale] about the mesh's BOTTOM-CENTER pivot —
z is owned by the physics settle (this runs AFTER it), and a scale/yaw about a pivot ON
the contact plane provably keeps the resting bottom where settle put it, so no re-ground
is needed for the object itself. Out-of-plane rotation stays the settle/mesh prior's
job.
Per iteration the update is closed-form: a trimmed 2D similarity (Umeyama on xy) for
yaw+scale+xy-translation, with the z-component of the increment zeroed. Scale is blended
toward 1 each step (monocular depth couples scale with z — an unregularized fit trades
one for the other).

Placement already consumed this cloud's CENTER and EXTENT (mesh_place.py); the new signal
here is surface-SHAPE alignment — yaw above all.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

# glTF export pre-rotation: stored GLB coords are world (x, y, z) -> (x, z, -y), so
# file -> world is (fx, -fz, fy).
GLB_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Degenerate-cloud skip gate (2026-07-21, ICP_DEGENERATE_GUARD proposal): a quasi-1D or
# tiny-area cloud cannot constrain the 4-DOF fit (a specular pan's MoGE depth collapsed
# below the support plane; the z-clip left only its handle — a 4mm stripe the fit dragged
# the whole pan onto). Thresholds calibrated on a 54-run sweep (310 accepted fits): the
# gate blocks exactly the two observed pan failures (footprint 6.1%/11.8%) and nothing
# else; nearest legit survivors are a spoon (footprint 20.9%) and the thin abc tray
# (kept 6.3% but footprint 62-96%, cleared by the relaxed limb).
MIN_FOOTPRINT_FRAC = 0.15  # cloud xy-hull area / mesh xy-hull area
MIN_KEEP_FRAC = 0.20  # cloud pts / eroded mask px, gates only with the limb below
RELAXED_FOOTPRINT_FRAC = 0.30


def object_cloud(
    points_hw3: np.ndarray,
    obj_mask: np.ndarray,
    valid_mask: Optional[np.ndarray],
    R: Optional[list],
    T: Optional[list],
    erode_px: int = 2,
    support_z_clip: float = 0.01,
    max_pts: int = 5000,
    stats: Optional[dict] = None,
) -> np.ndarray:
    """Masked MoGE points -> cleaned Nx3 object cloud in the canonical Z-up world.

    Cleanup: erode the mask (boundary pixels are depth-mixed with the background), drop
    points within ``support_z_clip`` of the support plane z=0 (mask bleed onto the table),
    trim the farthest 2% from the cloud median (flying-pixel outliers), downsample.
    ``stats`` (if a dict is passed) is filled with ``n_mask_px`` (post-erode, valid) and
    ``kept_frac`` (surviving points / mask px, PRE-downsample) for the degenerate gate."""
    from lib.tools.geometry import moge_camera as mc

    m = np.asarray(obj_mask) > 0
    if valid_mask is not None:
        m = m & (np.asarray(valid_mask) > 0)
    if erode_px > 0:
        import scipy.ndimage as ndi

        m = ndi.binary_erosion(m, iterations=erode_px)
    pts = np.asarray(points_hw3)[m]
    if stats is not None:
        stats["n_mask_px"] = int(len(pts))
        stats["kept_frac"] = 0.0
    if len(pts) == 0:
        return pts.reshape(0, 3)
    world = mc.moge_points_to_world(pts.astype(np.float64), R, T)
    world = world[world[:, 2] > support_z_clip]
    if len(world) > 8:
        med = np.median(world, axis=0)
        d = np.linalg.norm(world - med, axis=1)
        world = world[d <= np.quantile(d, 0.98)]
    if stats is not None:
        stats["kept_frac"] = float(len(world) / len(pts))
    if len(world) > max_pts:
        idx = np.random.default_rng(0).choice(len(world), max_pts, replace=False)
        world = world[idx]
    return world


def load_placed_mesh(glb_path: str):
    """Placed-object GLB -> a single trimesh.Trimesh in the canonical Z-up world frame."""
    import trimesh

    sc = trimesh.load(glb_path, force="scene")
    # dump() applies the scene-graph NODE transforms (sc.geometry is the raw, untransformed
    # geometry — reading it silently drops any transform stored on a glTF node, e.g. the
    # pose-match bake's root-node correction)
    parts = sc.dump() if hasattr(sc, "dump") else [sc]
    parts = [g for g in parts if hasattr(g, "vertices")]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices @ GLB_TO_WORLD.T
    return mesh


def bottom_center_pivot(verts: np.ndarray) -> np.ndarray:
    """The transform pivot: xy centroid at the LOWEST z (the resting-contact plane).
    Yaw/scale about this point leave the mesh's bottom height invariant."""
    v = np.asarray(verts)
    return np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()])


def _apply(params: dict, pts: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """World-space correction: scale+yaw about ``pivot``, then translate."""
    c = math.cos(math.radians(params["yaw_deg"]))
    s = math.sin(math.radians(params["yaw_deg"]))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (pts - pivot) @ rz.T * params["scale"] + pivot + np.asarray(params["t"])


def fit_similarity(
    mesh,
    cloud: np.ndarray,
    iters: int = 60,
    trim: float = 0.98,
    scale_blend: float = 0.5,
    max_yaw_deg: float = 30.0,
    scale_range: tuple[float, float] = (0.85, 1.18),
    min_pts: int = 150,
    accept_rms_ratio: float = 0.8,
    freeze_xy: bool = False,
    cloud_stats: Optional[dict] = None,
) -> dict[str, Any]:
    """Fit the [x, y, yaw, scale] correction that best aligns ``mesh`` to ``cloud``.

    z is free during iteration but dropped from the result (t[2] == 0, fitted value in
    ``dz_dropped``), and the pivot is the mesh's bottom-center — so the resting contact
    from the physics settle is preserved by construction (scale about a point on the
    contact plane cannot move the bottom). Returns the correction, before/after trimmed RMS
    point-to-mesh distances, and ``accepted`` (RMS improved by >= 1-accept_rms_ratio AND the
    cloud is big enough). The mesh is NOT modified; callers apply the correction themselves.

    Per-DOF freezes: ``freeze_xy=True`` locks the xy translation to 0 (z stays free during
    iteration as usual); ``max_yaw_deg=0`` locks yaw exactly (the increment is zeroed
    BEFORE it enters the translation update, not just clipped after); scale is frozen via
    the existing ``scale_range=(1.0, 1.0), scale_blend=0.0``.
    """
    import trimesh
    from scipy.spatial import cKDTree

    out: dict[str, Any] = {
        "yaw_deg": 0.0,
        "scale": 1.0,
        "t": [0.0, 0.0, 0.0],
        "dz_dropped": 0.0,
        "n_pts": int(len(cloud)),
        "rms_before": None,
        "rms_after": None,
        "accepted": False,
        "skipped": None,
    }
    if len(cloud) < min_pts:
        out["skipped"] = f"cloud too small ({len(cloud)} < {min_pts})"
        return out
    # Degenerate-cloud gate: skip when the cloud's xy footprint cannot constrain the
    # 4-DOF fit, or is both a sliver of the mask (depth collapsed — specular surfaces)
    # AND well under the mesh footprint. Keeping the settled pose beats a degenerate fit.
    kept_frac = (cloud_stats or {}).get("kept_frac")
    try:
        from scipy.spatial import ConvexHull

        fr = float(
            ConvexHull(cloud[:, :2]).volume / ConvexHull(mesh.vertices[:, :2]).volume
        )
    except Exception:  # noqa: BLE001 - collinear/tiny cloud has no hull: fully degenerate
        fr = 0.0
    out["footprint_frac"] = round(fr, 4)
    if kept_frac is not None:
        out["kept_frac"] = round(float(kept_frac), 4)
    if fr < MIN_FOOTPRINT_FRAC or (
        kept_frac is not None
        and kept_frac < MIN_KEEP_FRAC
        and fr < RELAXED_FOOTPRINT_FRAC
    ):
        kf = f", kept {kept_frac:.0%} of {(cloud_stats or {}).get('n_mask_px')} mask px" \
            if kept_frac is not None else ""  # fmt: skip
        out["skipped"] = f"degenerate cloud (footprint {fr:.0%} of mesh{kf})"
        return out
    # nearest-surface queries via a dense surface SAMPLING + KDTree (~mm sampling error,
    # negligible at cm-scale RMS; avoids trimesh's exact on_surface, which needs rtree)
    surf, _ = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    tree = cKDTree(np.asarray(surf))
    pivot = bottom_center_pivot(mesh.vertices)

    def correspondences(
        params: dict, cur_trim: float = trim
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # pull the cloud into the mesh's frame via the INVERSE correction (one static
        # KD-tree instead of re-transforming the mesh every step)
        c = math.cos(math.radians(-params["yaw_deg"]))
        s = math.sin(math.radians(-params["yaw_deg"]))
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = (cloud - pivot - np.asarray(params["t"])) / params[
            "scale"
        ] @ rz.T + pivot
        dist, idx = tree.query(local)
        # robust threshold: adapts to the residual distribution (5x median keeps the
        # informative outer ring of a mis-scaled/mis-yawed shell, cuts far flying pixels),
        # with a floor for the converged/noise regime and a hard cap on the extreme tail
        thr = min(
            max(5.0 * float(np.median(dist)), 0.008), float(np.quantile(dist, cur_trim))
        )
        keep = dist <= thr
        near = np.asarray(surf)[idx[keep]]
        rms = float(np.sqrt(np.mean((dist[keep] * params["scale"]) ** 2)))
        # matched mesh points mapped back to world under the current correction
        return rms, _apply(params, near, pivot), cloud[keep]

    params = {"yaw_deg": 0.0, "scale": 1.0, "t": np.zeros(3)}
    rms0, _, _ = correspondences(params)
    out["rms_before"] = rms0
    prev = rms0
    for it in range(iters):
        cur_trim = (
            0.98  # the robust threshold does the real trimming; cap the extreme tail
        )
        _, m_w, q_w = correspondences(params, cur_trim)
        # exact incremental 2D similarity (yaw + scale) about the pivot, from the centered
        # covariance of the matched pairs; then the EXACT incremental translation
        # dt = (qc - pivot) - ds*Rz(da)*(mc - pivot)  (a dt of "bm - am" is only right at
        # zero rotation — with an off-pivot shell centroid it injects wrong translation)
        a0 = m_w[:, :2] - m_w[:, :2].mean(axis=0)
        b0 = q_w[:, :2] - q_w[:, :2].mean(axis=0)
        cov = a0.T @ b0 / len(a0)
        da = math.atan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
        if max_yaw_deg == 0:  # frozen yaw: keep the increment out of dt too
            da = 0.0
        var = (a0**2).sum() / len(a0)
        ds = float(np.trace(cov) / (var * math.cos(da))) if var > 1e-12 else 1.0
        ds = 1.0 + (ds - 1.0) * scale_blend  # regularize toward 1
        c, si = math.cos(da), math.sin(da)
        rz = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        mc, qc = m_w.mean(axis=0), q_w.mean(axis=0)
        dt = (qc - pivot) - ds * (rz @ (mc - pivot))
        # compose increment onto the accumulated params: yaw/scale add/multiply;
        # t_new = ds*Rz(da)*t_old + dt
        params["yaw_deg"] = float(
            np.clip(params["yaw_deg"] + math.degrees(da), -max_yaw_deg, max_yaw_deg)
        )
        params["scale"] = float(
            np.clip(params["scale"] * ds, scale_range[0], scale_range[1])
        )
        params["t"] = ds * (rz @ params["t"]) + dt
        if freeze_xy:
            params["t"][0] = params["t"][1] = 0.0
        rms, _, _ = correspondences(params, cur_trim)
        if it >= 8 and prev - rms < 1e-5:
            break
        prev = rms
    out["rms_after"] = prev
    out["yaw_deg"], out["scale"] = params["yaw_deg"], params["scale"]
    # z stays FREE during iteration (early correspondences have transient vertical bias
    # that would otherwise pollute the robust trim and the xy covariance) but is DROPPED
    # from the reported correction: z is owned by the settle contact, and with the
    # bottom-center pivot a zero t_z provably keeps the resting bottom in place. The
    # fitted value is logged as dz_dropped (post-settle it should be ~0; a large one
    # flags a cloud/support height disagreement worth investigating).
    out["dz_dropped"] = float(params["t"][2])
    out["t"] = [float(params["t"][0]), float(params["t"][1]), 0.0]
    # audit-only degenerate-fit signature (no gating: |t| magnitude provably cannot
    # separate good from bad — legit croissant corrections reach 0.78x their diagonal)
    if scale_range[0] != scale_range[1]:
        out["scale_at_cap"] = out["scale"] in (scale_range[0], scale_range[1])
    out["accepted"] = bool(prev <= accept_rms_ratio * rms0)
    return out


def icp_freeze_kwargs(freeze: Optional[list[str]]) -> dict[str, Any]:
    """Map --icp-freeze DOF names to ``fit_similarity`` kwargs.

    ``xy`` -> freeze_xy=True; ``yaw`` -> max_yaw_deg=0 (exact, see fit_similarity);
    ``scale`` -> scale_range=(1,1) + scale_blend=0 (the same-size mechanism)."""
    kw: dict[str, Any] = {}
    for dof in freeze or []:
        if dof == "xy":
            kw["freeze_xy"] = True
        elif dof == "yaw":
            kw["max_yaw_deg"] = 0.0
        elif dof == "scale":
            kw.update(scale_range=(1.0, 1.0), scale_blend=0.0)
        else:
            raise ValueError(f"unknown ICP freeze DOF {dof!r} (xy|yaw|scale)")
    return kw


def correction_matrix(params: dict, pivot: np.ndarray) -> np.ndarray:
    """The fitted correction as a 4x4 world-frame matrix (same map as ``_apply``)."""
    c = math.cos(math.radians(params["yaw_deg"]))
    s = math.sin(math.radians(params["yaw_deg"]))
    sc = params["scale"]
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) * sc
    m[:3, 3] = (
        np.asarray(params["t"]) + np.asarray(pivot) - m[:3, :3] @ np.asarray(pivot)
    )
    return m


_BAKE_SCRIPT = r"""
import json, sys
import bpy
from mathutils import Matrix
jobs = json.load(open(sys.argv[-1]))
for job in jobs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=job["glb_in"])
    M = Matrix(job["matrix"])
    # Bake the corrected WORLD transform into the vertex data (identity nodes), matching how
    # the original placed GLBs are stored — the glTF exporter does not reliably carry a root
    # EMPTY's transform through the round trip, so node-level transforms get silently lost.
    for o in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        world = M @ o.matrix_world
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.data.transform(world)
    for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.export_scene.gltf(filepath=job["glb_out"], export_format="GLB")
print("POSE_MATCH_BAKE_DONE", len(jobs))
"""


def object_cloud_for(
    o: dict, out_dir: str, points: np.ndarray, valid, R, T,
    stats: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Mask/redetect-aware MoGE cloud for one placement entry (None if no usable mask).
    Source for the incremental settle's interleaved per-object ICP. ``stats`` is
    passed through to ``object_cloud``."""
    import os as _os

    name = (o.get("mesh_name") or "").removeprefix("obj_")
    mask_p = _os.path.join(out_dir, "masks", f"{name}.npy")
    obj_points, obj_valid = points, valid
    # Accepted bank assets may have been registered against a sparse, measured
    # RGB-aligned depth map while the main scene uses a dense/hole-completed point map.
    # Keep the post-settle residual ICP on the same evidence so it cannot undo the
    # calibrated registration.  The path is stored relative to the scene for portable
    # skip-preprocess runs.
    asset_points = o.get("asset_registration_points_npy")
    using_asset_points = False
    if asset_points:
        asset_points_path = str(asset_points)
        if not _os.path.isabs(asset_points_path):
            asset_points_path = _os.path.join(out_dir, asset_points_path)
        if _os.path.exists(asset_points_path):
            candidate_points = np.load(asset_points_path)
            if candidate_points.shape == points.shape:
                obj_points = candidate_points
                obj_valid = (
                    np.isfinite(candidate_points).all(axis=2)
                    & (candidate_points[..., 2] > 0)
                )
                using_asset_points = True
    rd = (o.get("redetect") or {}) if isinstance(o, dict) else {}
    if rd.get("mask_path") and _os.path.exists(rd["mask_path"]):
        mask_p = rd["mask_path"]
        if (
            not using_asset_points
            and rd.get("points_npy")
            and _os.path.exists(rd["points_npy"])
        ):
            obj_points, obj_valid = np.load(rd["points_npy"]), None
    if not _os.path.exists(mask_p):
        return None
    return object_cloud(obj_points, np.load(mask_p), obj_valid, R, T, stats=stats)


def bake_world_matrices(
    jobs: list[dict], out_dir: str, blender_cmd: str
) -> None:
    """Bake per-object world matrices into GLB copies via Blender (texture-preserving;
    trimesh's glTF round trip is not). ``jobs`` = [{glb_in, glb_out, matrix(4x4)}]."""
    import json as _json
    import os as _os
    import subprocess as _sp

    job_json = _os.path.join(out_dir, "physics", "settle_bake_jobs.json")
    _os.makedirs(_os.path.dirname(job_json), exist_ok=True)
    with open(job_json, "w") as f:
        _json.dump(jobs, f)
    script = _os.path.join(out_dir, "physics", "pose_match_bake.py")
    with open(script, "w") as f:
        f.write(_BAKE_SCRIPT)
    proc = _sp.run(
        [blender_cmd, "--background", "--factory-startup", "--python", script,
         "--", job_json],
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    if "POSE_MATCH_BAKE_DONE" not in (proc.stdout or ""):
        raise RuntimeError(f"settle bake failed: {(proc.stderr or proc.stdout)[-400:]}")
