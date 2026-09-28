"""Scene-graph support inference from MoGE geometry.

Pure geometry helpers used by preprocessing to turn the VLM's proposed support
tree into a verified one: fit planes to root surfaces, find each object's base /
top / footprint, score how well an object rests on a candidate support, and
resolve the final parent (flag VLM/geometry disagreements -> the VLM decides).

Everything here is a pure function over numpy arrays (no VLM / Blender), so it
unit-tests with synthetic point clouds. The "support" test targets the common
case of objects RESTING on a horizontal-ish surface or stacked on each other;
wall/ceiling attachments get a low geometric score and fall back to the VLM.

All heights/footprints use a single scene ``up`` axis (default world +Z, gravity up;
optionally the RANSAC ground normal) so every node shares one basis.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np

FOOT_PCTL = (2.0, 98.0)  # P1: outlier-robust footprint bbox (a depth-bleed strip
# at an occlusion boundary stretched a statue's min/max base bbox to 1.15m and
# poisoned its `covered` factor to 0.14 — just under the 0.15 flag floor)
INCONT_FRAC = 0.8  # P2: >=80% of the object's base footprint inside the rim bbox
INCONT_MIN_DEPTH = 0.03  # P2: a container must have >=3cm of interior depth
# (a placemat is not a container)
SUPPORT_EPS = 0.04  # the resting tolerance every factor shares (meters)


def _score_v2() -> bool:
    return os.environ.get("GRASE_SUPPORT_SCORE_V2", "1") != "0"


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def basis_perp(up) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal axes spanning the plane perpendicular to ``up``."""
    u = unit(up)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(u, a))
    e2 = unit(np.cross(u, e1))
    return e1, e2


def _finite(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p[np.isfinite(p).all(axis=1)]


# --------------------------------------------------------------------------- #
# Plane fitting (root surfaces)                                                #
# --------------------------------------------------------------------------- #
def fit_plane(
    points: np.ndarray,
    thresh: float = 0.01,
    iters: int = 200,
    up=(0, 0, 1),
    seed: int = 0,
    max_pts: int = 5000,
) -> Optional[dict[str, Any]]:
    """RANSAC plane fit -> {normal (unit, oriented toward ``up``), d, centroid,
    inliers, inlier_frac}. ``normal . x + d = 0``. None if too few points.

    Large surfaces (100k+ points) are subsampled to ``max_pts`` for the RANSAC +
    refit so cost stays bounded; ``inlier_frac`` is measured on the sample."""
    pts = _finite(points)
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    best = None
    for _ in range(iters):
        i = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p0
        inl = np.abs(pts @ n + d) < thresh
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best is None:  # every sampled triple was collinear (a 1-D sliver of points)
        return None
    inl = best[1]
    ip = pts[inl]
    c = ip.mean(axis=0)
    # refit: plane normal = smallest right-singular vector (thin SVD: U is never
    # materialized, so this is O(N) not O(N^2)).
    _, _, vt = np.linalg.svd(ip - c, full_matrices=False)
    n = unit(vt[-1])
    d = float(-n @ c)
    if n @ unit(up) < 0:  # orient toward up so "above plane" is well defined
        n, d = -n, -d
    return {
        "normal": n,
        "d": d,
        "centroid": c,
        "inliers": int(inl.sum()),
        "inlier_frac": float(inl.mean()),
    }


MIN_PLANE_INLIER = 0.6  # RANSAC fit quality: is the surface actually planar?
MIN_PLANE_MASK_FRAC = 0.02  # frame coverage; mirrors coverage_report's skip_photo_below


def plane_is_reliable(
    plane: Optional[dict[str, Any]],
    min_inlier: float = MIN_PLANE_INLIER,
    min_mask_frac: float = MIN_PLANE_MASK_FRAC,
) -> bool:
    """Is a scene-graph surface's MEASURED plane trustworthy enough to build on / anchor to?

    Two INDEPENDENT failure modes, both disqualifying:

    * ``inlier_frac`` — the points aren't planar (a grazing monocular fit, a mixed
      fit over two surfaces). Long-standing gate, 0.6.
    * ``mask_frac`` — the surface is a SLIVER. A wall seen edge-on covers a few
      hundred pixels whose unprojections are near-collinear, so RANSAC scores a
      perfect ``inlier_frac`` on a plane whose horizontal RUN is unconstrained.
      0801_rdj_push_t_random: two 0.56%-of-frame walls fit at 1.0 / 0.9994 with runs
      20deg from parallel when they were opposite walls, and the yaw ladder anchored
      the table on one of them (-14deg) while L3 demanded +14 off a third wall.
      ``coverage_report`` already refuses to grade these ("photo mask too small to
      compare") — this applies the same bar upstream, where the plane is consumed.

    A missing field is permissive (older scene graphs carry no ``mask_frac``), matching
    the pre-existing ``inl is None`` behaviour. ``None``/empty plane -> False."""
    if not plane:
        return False
    inl, frac = plane.get("inlier_frac"), plane.get("mask_frac")
    if inl is not None and inl < min_inlier:
        return False
    return not (frac is not None and frac < min_mask_frac)


# Wall-vs-support classification of a MEASURED root-surface normal (|nz| = |normal . up|).
# Shared by the initializer prompt, the built-scene checkers, and infer_against so all
# three read ONE definition. The bands leave a deliberate ambiguous gap: a fit tilted
# ~45-72 deg is almost never real geometry in these scenes, and snapping it 30-60 deg
# with false confidence is worse than building that surface by eye.
PLUMB_SUPPORT_MIN_NZ = 0.7  # |nz| >= this -> level support, snap to +Z (tilt < ~45 deg)
PLUMB_WALL_MAX_NZ = 0.3  # |nz| <= this -> plumb wall, zero z (within ~17.5 deg of vertical)


def plumb_plane(normal) -> tuple[Optional[list], str]:
    ""
    if normal is None:
        return None, "ambiguous"
    v = np.asarray(normal, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return None, "ambiguous"
    v = v / n
    if abs(v[2]) >= PLUMB_SUPPORT_MIN_NZ:
        return [0.0, 0.0, 1.0], "support"
    if abs(v[2]) <= PLUMB_WALL_MAX_NZ:
        h = float(np.hypot(v[0], v[1]))
        return [float(v[0] / h), float(v[1] / h), 0.0], "wall"
    return [float(x) for x in v], "ambiguous"


def footprint_bbox(
    points: np.ndarray, up=(0, 0, 1), pctl: Optional[tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    ""
    e1, e2 = basis_perp(up)
    pts = _finite(points)
    proj = np.stack([pts @ e1, pts @ e2], axis=1)
    if pctl is not None:
        return (
            np.percentile(proj, pctl[0], axis=0),
            np.percentile(proj, pctl[1], axis=0),
        )
    return proj.min(axis=0), proj.max(axis=0)


# --------------------------------------------------------------------------- #
# Object base / top / footprint                                               #
# --------------------------------------------------------------------------- #
def object_base_top(
    points: np.ndarray, up=(0, 0, 1), lo: float = 5.0, hi: float = 95.0
) -> Optional[dict[str, Any]]:
    """Base (low) and top (high) bands of an object along ``up``, with footprints.

    Returns {base_h, top_h, centroid, base_pts, foot_bbox (base footprint),
    top_bbox (top footprint)}. Percentiles shrug off mask-edge outliers.
    """
    pts = _finite(points)
    if len(pts) < 3:
        return None
    u = unit(up)
    h = pts @ u
    base_h, top_h = np.percentile(h, lo), np.percentile(h, hi)
    band = 0.15 * (top_h - base_h) + 1e-6
    base_pts = pts[h <= base_h + band]
    top_pts = pts[h >= top_h - band]
    pctl = FOOT_PCTL if _score_v2() else None
    return {
        "base_h": float(base_h),
        "top_h": float(top_h),
        "centroid": pts.mean(axis=0),
        "base_pts": base_pts,
        "foot_bbox": footprint_bbox(base_pts, u, pctl=pctl),
        "top_bbox": footprint_bbox(top_pts, u, pctl=pctl),
    }


def _bbox_overlap_frac(a: tuple, b: tuple) -> float:
    """Fraction of bbox ``a``'s area that lies inside bbox ``b`` (a,b = (min,max))."""
    amn, amx = np.asarray(a[0]), np.asarray(a[1])
    bmn, bmx = np.asarray(b[0]), np.asarray(b[1])
    inter = np.clip(np.minimum(amx, bmx) - np.maximum(amn, bmn), 0, None).prod()
    area_a = np.clip(amx - amn, 1e-9, None).prod()
    return float(inter / area_a)


def _bbox_iou(a: tuple, b: tuple) -> float:
    amn, amx = np.asarray(a[0]), np.asarray(a[1])
    bmn, bmx = np.asarray(b[0]), np.asarray(b[1])
    inter = np.clip(np.minimum(amx, bmx) - np.maximum(amn, bmn), 0, None).prod()
    ua = np.clip(amx - amn, 0, None).prod() + np.clip(bmx - bmn, 0, None).prod() - inter
    return float(inter / ua) if ua > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# Support scoring                                                             #
# --------------------------------------------------------------------------- #
def in_container(obj_bt: dict[str, Any], cand: dict[str, Any], eps: float = SUPPORT_EPS) -> bool:
    ""
    if cand["top_h"] - cand["base_h"] < INCONT_MIN_DEPTH:
        return False
    # INTERIOR test: the base must sit clearly below the rim (margin = the same
    # min-depth constant), else stacking ON a tall candidate's top face would read
    # as containment (a can resting exactly on a box top has base_h == top_h).
    if not (cand["base_h"] - eps <= obj_bt["base_h"] <= cand["top_h"] - INCONT_MIN_DEPTH):
        return False
    return _bbox_overlap_frac(obj_bt["foot_bbox"], cand["top_bbox"]) >= INCONT_FRAC


def support_score(
    obj_bt: dict[str, Any], candidate: dict[str, Any], eps: float
) -> float:
    """How well ``obj_bt`` (from object_base_top) rests on ``candidate`` ∈ [0,1].

    candidate kinds:
      surface -> {kind:'surface', plane, extent:(min2,max2)}  (object base near the
                 plane, footprint over the surface, object above the plane)
      object  -> {kind:'object', top_h, top_bbox}             (object base near the
                 candidate's top, footprints overlap, object above it)
    """
    if candidate["kind"] == "surface":
        plane, extent = candidate["plane"], candidate["extent"]
        n, d = plane["normal"], plane["d"]
        # gap = how far the object's lowest points float above the plane (signed;
        # a low percentile of the base band, robust to a few outliers).
        gap = max(0.0, float(np.percentile(obj_bt["base_pts"] @ n + d, 10)))
        contact = max(0.0, 1.0 - gap / eps)
        covered = _bbox_overlap_frac(obj_bt["foot_bbox"], extent)
        above = (obj_bt["centroid"] @ n + d) > -eps
        return contact * covered * (1.0 if above else 0.0)
    if _score_v2():
        # P2: in-container — score contact against the container's BASE, not its
        # rim (bottle-in-bin scored 0.000 against the bin under the rim formula).
        if "base_h" in candidate and in_container(obj_bt, candidate, eps):
            gap = max(0.0, obj_bt["base_h"] - candidate["base_h"])
            covered = _bbox_overlap_frac(obj_bt["foot_bbox"], candidate["top_bbox"])
            return max(0.0, 1.0 - gap / eps) * covered
        gap = obj_bt["base_h"] - candidate["top_h"]
        contact = max(0.0, 1.0 - abs(gap) / eps)
        covered = _bbox_overlap_frac(obj_bt["foot_bbox"], candidate["top_bbox"])
        above = obj_bt["base_h"] > candidate["top_h"] - eps
        return contact * covered * (1.0 if above else 0.0)
    gap = obj_bt["base_h"] - candidate["top_h"]
    contact = max(0.0, 1.0 - abs(gap) / eps)
    overlap = _bbox_iou(obj_bt["foot_bbox"], candidate["top_bbox"])
    above = obj_bt["base_h"] > candidate["top_h"] - eps
    return contact * overlap * (1.0 if above else 0.0)


# --------------------------------------------------------------------------- #
# Parent resolution (flag disagreements -> VLM decides)                        #
# --------------------------------------------------------------------------- #
def resolve_parents(
    vlm_parents: dict[str, Optional[str]],
    scores: dict[str, dict[str, float]],
    decide: Optional[Callable[[str, Optional[str], str, dict[str, float]], str]] = None,
    tau_weak: float = 0.15,
) -> dict[str, Any]:
    """Fuse the VLM-proposed parents with geometric support scores.

    For each object: pick the best-scoring geometric parent ``pg``. If geometry is
    weak (< ``tau_weak``) keep the VLM parent; if ``pg`` agrees with the VLM, accept;
    otherwise it's a flagged disagreement -> ``decide`` (the VLM) picks. Returns
    {parents: {obj: parent}, flags: [{obj, vlm, geom, score}]}. Cycles are broken
    by falling back to the VLM parent.
    """
    parents: dict[str, Optional[str]] = {}
    flags: list[dict[str, Any]] = []
    for obj, sc in scores.items():
        pv = vlm_parents.get(obj)
        pg = max(sc, key=sc.get) if sc else None
        best = sc.get(pg, 0.0) if pg else 0.0
        if pg is None or best < tau_weak:
            parents[obj] = pv
        elif pg == pv:
            parents[obj] = pv
        else:
            flags.append({"obj": obj, "vlm": pv, "geom": pg, "score": round(best, 3)})
            parents[obj] = decide(obj, pv, pg, sc) if decide else pv
    _break_cycles(parents, vlm_parents)
    return {"parents": parents, "flags": flags}


def _break_cycles(
    parents: dict[str, Optional[str]], fallback: dict[str, Optional[str]]
) -> None:
    for obj in list(parents):
        seen, cur = set(), obj
        while parents.get(cur) is not None:
            cur = parents[cur]
            if cur in seen or cur == obj:  # cycle -> sever this object's edge
                parents[obj] = fallback.get(obj) if fallback.get(obj) != obj else None
                break
            seen.add(cur)
