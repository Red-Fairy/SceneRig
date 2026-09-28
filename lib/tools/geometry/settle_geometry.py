"""Pure geometry shared by the incremental settle server and its tests.

Convex-part utilities with no Isaac/USD dependencies: half-space overlap tests for
the lift-to-clear search, collider base flattening, and npz part loading. The server
(isaac/isaac_settle_server.py) imports this via a repo-root sys.path insert.
"""

from __future__ import annotations

import numpy as np


SUPPORT_CONTAINER_FRAC = 0.5


def support_relation(olo, blo, bhi, frac: float, tol: float) -> str:
    """Classify a broad ``supports_of`` base/footprint candidate.

    The first two results identify a definite under-support or enclosing/spanning
    body. ``ambiguous_same_level`` is the lateral-contact class: its AABB happens to
    extend under the queried object, but it must remain eligible for contact-dependent
    re-simulation when either neighbour moves.
    """
    if bhi[2] <= olo[2] + tol:
        return "strict_below"
    if blo[2] < olo[2] - tol or frac >= SUPPORT_CONTAINER_FRAC:
        return "clear_below_or_container"
    return "ambiguous_same_level"


def load_parts(npz):
    d = np.load(npz)
    return [
        (np.array(d[f"v{i}"], dtype=float), np.array(d[f"f{i}"]))
        for i in range(int(d["n"]))
    ]


def flatten_base(parts, mm):
    """Clamp collider vertices within ``mm`` of the compound zmin onto zmin (flat base
    facet spanning the true contact footprint — 'observed at rest => flat base')."""
    zmin = min(float(v[:, 2].min()) for v, _ in parts)
    out = []
    for v, f in parts:
        v = v.copy()
        v[v[:, 2] < zmin + mm / 1000.0, 2] = zmin
        out.append((v, f))
    return out


_SIMPLIFY_MIN_VERTS = 256  # parts already this small are kept exact
_SIMPLIFY_VOXEL = 0.002  # 2 mm simplification grid (floored to extent/8 for
# thin parts so a knife blade never collapses to a degenerate plane)


class Hull:
    """Half-space form of one convex part (A @ x <= b) + its AABB, for overlap tests.
    A degenerate sliver (Qhull failure: planar/collinear points) encloses no volume, so
    as a CONTAINER it contains nothing — an AABB fallback here produced false positives
    that made lift-to-clear overshoot by the sliver's full extent. Overlap with a sliver
    is still caught from the other side (its vertices inside the other hull).

    SIMPLIFICATION (T1a, 2026-07-23): organic CoACD parts are hull meshes of
    smooth surfaces — ~all verts are extreme (bell_pepper_1: 37 958, pruning is
    a no-op), and every overlap/scan matmul paid for the tessellation density.
    Large parts are voxel-downsampled on a 2 mm grid, re-hulled, and the
    half-spaces dilated OUTWARD by the measured containment error, so the
    simplified hull strictly CONTAINS the raw part — overlap can only be
    over-reported (~0.3 mm), never missed; the dilation is MANDATORY (without
    it the error flips to missed penetrations). ``pad`` carries the dilation
    into the AABB and analytic transforms. Raw parts stay untouched on the
    owning object (centroid/zmin/USD authoring never see simplified hulls).
    ``simplify=False`` keeps the exact legacy hull (tests' golden reference)."""

    def __init__(self, verts, simplify=True):
        from scipy.spatial import ConvexHull, QhullError

        raw = np.asarray(verts, dtype=float)
        v = raw
        self.pad = 0.0
        if simplify and len(raw) > _SIMPLIFY_MIN_VERTS:
            ext = float((raw.max(0) - raw.min(0)).max())
            vox = min(_SIMPLIFY_VOXEL, max(ext / 8.0, 1e-6))
            _, idx = np.unique(
                np.round(raw / vox).astype(np.int64), axis=0, return_index=True
            )
            v = raw[np.sort(idx)]
        try:
            h = ConvexHull(v)
            A, b = h.equations[:, :3], -h.equations[:, 3]
            if v is not raw:
                eps = float(np.max(raw @ A.T - b))  # worst raw vert outside
                if eps > 0.0:
                    b = b + eps
                    self.pad = eps
                v = v[h.vertices]  # probes: the simplified extreme points
            self.A, self.b = A, b
        except QhullError:
            self.A = self.b = None
            v = raw  # degenerate: raw verts stay the probe set (legacy semantics)
        self.v = v
        self.lo = raw.min(0) - self.pad  # AABB must contain the DILATED hull
        self.hi = raw.max(0) + self.pad

    def contains_any(self, pts, tol=1e-6):
        if self.A is None:
            return False  # zero-volume container
        inside = (pts @ self.A.T <= self.b + tol).all(axis=1)
        return bool(inside.any())

    def translated(self, offset) -> "Hull":
        """Exact analytic translation (keeps the dilation — re-hulling the
        simplified verts would silently shed ``pad``)."""
        off = np.asarray(offset, dtype=float)
        h = Hull.__new__(Hull)
        h.v = self.v + off
        h.lo, h.hi = self.lo + off, self.hi + off
        h.pad = self.pad
        if self.A is None:
            h.A = h.b = None
        else:
            h.A = self.A
            h.b = self.b + self.A @ off
        return h

    def transformed_rigid(self, R, t) -> "Hull":
        """Exact analytic rigid transform: rows of A stay unit-norm under an
        orthonormal R (the ``tol`` dilation in hulls_overlap assumes unit
        normals), so qhull never re-runs on a pose change. Callers must route
        SCALED deltas through a full rebuild instead."""
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        h = Hull.__new__(Hull)
        h.v = self.v @ R.T + t
        h.pad = self.pad
        h.lo = h.v.min(0) - self.pad
        h.hi = h.v.max(0) + self.pad
        if self.A is None:
            h.A = h.b = None
        else:
            h.A = self.A @ R.T
            h.b = self.b + h.A @ t
        return h


def hulls_overlap(
    a: Hull, b: Hull, dz: float, tol: float = 0.0, dx: float = 0.0, dy: float = 0.0
) -> bool:
    """Vertex-level overlap of hull ``a`` shifted by (dx, dy, dz) against ``b`` (AABB
    prefiltered; a's half-spaces are shifted analytically: b' = b + A @ shift).
    ``tol`` dilates both hulls (isotropic outward offset of the unit-normal
    half-spaces) so a NEAR-contact within ``tol`` counts as overlapping -- the
    composition dependency query uses this to free a leaner that rests AGAINST a
    neighbor with a hair of gap (abc3: a battery 3.5mm off a mic-stand tripod).
    ``tol=0`` is exact interpenetration -- the default the strict penetration gate
    keeps."""
    if (a.lo[0] + dx > b.hi[0] + tol) or (b.lo[0] > a.hi[0] + dx + tol):
        return False
    if (a.lo[1] + dy > b.hi[1] + tol) or (b.lo[1] > a.hi[1] + dy + tol):
        return False
    if (a.lo[2] + dz > b.hi[2] + tol) or (b.lo[2] > a.hi[2] + dz + tol):
        return False
    s = np.array([dx, dy, dz])
    av = a.v + s
    if b.contains_any(av, tol=tol + 1e-6):
        return True
    if a.A is not None:
        inside = (b.v @ a.A.T <= (a.b + a.A @ s) + tol + 1e-6).all(axis=1)
        return bool(inside.any())
    return False  # both directions checked; a degenerate `a` cannot contain b's verts


_CHUNK_ELEMS = 4_000_000  # rhs matmul budget (verts_chunk x faces ~ 32 MB f64):
# organic raw parts reach ~60k hull faces; a fixed vertex chunk exploded to GB-
# scale transients and allocation thrash ate the interval method's win.


def _vert_intervals(lo_acc, hi_acc, V, A, bvec, cu, slack, flip):
    """Per-vertex d-intervals for one containment direction, appended to
    ``lo_acc``/``hi_acc``. Each vertex v of ``V`` is inside the hull (A, bvec)
    shifted relative to it along d*u for exactly one interval of d (each face is
    one linear inequality in d): with ``flip=False`` the VERTICES move (+d*u into
    a static hull, constraint A.v + d*cu <= b+slack); with ``flip=True`` the HULL
    moves (+d*u past static vertices, constraint A.v <= b + d*cu + slack). ``cu``
    = A @ u per face. Faces with cu ~ 0 are d-independent: they gate feasibility
    outright."""
    sgn = -1.0 if flip else 1.0  # flip: d*cu moves to the rhs
    pos = sgn * cu > 1e-12
    neg = sgn * cu < -1e-12
    zer = ~(pos | neg)
    chunk = max(64, _CHUNK_ELEMS // max(len(bvec), 1))
    for i in range(0, len(V), chunk):
        R = (bvec + slack)[None, :] - V[i:i + chunk] @ A.T  # rhs per (vert, face)
        n = len(R)
        hi = np.min(R[:, pos] / (sgn * cu[pos]), axis=1) if pos.any() else np.full(n, np.inf)
        lo = np.max(R[:, neg] / (sgn * cu[neg]), axis=1) if neg.any() else np.full(n, -np.inf)
        ok = (R[:, zer] >= 0.0).all(axis=1) if zer.any() else np.ones(n, bool)
        keep = ok & (lo <= hi)
        lo_acc.append(lo[keep])
        hi_acc.append(hi[keep])


def _pair_overlap_intervals(a: Hull, b: Hull, u: np.ndarray) -> tuple:
    """The exact set of d >= anything for which ``hulls_overlap(a, b, d*u)``
    (tol=0) holds, as (lo_array, hi_array) of possibly-overlapping intervals —
    the closed-form replacement for stepping d. Reproduces the legacy test's
    semantics term by term: the AABB gate (no slack) becomes one d-interval that
    everything is clipped to; each containment direction (a's verts in b, b's
    verts in shifted a, both with the +1e-6 slack) contributes one interval per
    vertex; the pair's overlap set is their union. Convexity makes every
    per-vertex set a single interval, so non-monotone multi-interval unions
    (clear gap between two blockers) survive exactly."""
    # AABB gate as a d-interval (legacy rejects on strict >, so bounds inclusive)
    g_lo, g_hi = -np.inf, np.inf
    for k in range(3):
        span_lo = b.lo[k] - a.hi[k]  # d*u[k] >= span_lo
        span_hi = b.hi[k] - a.lo[k]  # d*u[k] <= span_hi
        if abs(u[k]) <= 1e-12:
            if 0.0 < span_lo or 0.0 > span_hi:
                return np.empty(0), np.empty(0)
        elif u[k] > 0:
            g_lo = max(g_lo, span_lo / u[k])
            g_hi = min(g_hi, span_hi / u[k])
        else:
            g_lo = max(g_lo, span_hi / u[k])
            g_hi = min(g_hi, span_lo / u[k])
    if g_lo > g_hi:
        return np.empty(0), np.empty(0)
    lo_acc: list = []
    hi_acc: list = []
    if b.A is not None:  # a's vertices travel +d*u into static b
        _vert_intervals(lo_acc, hi_acc, a.v, b.A, b.b, b.A @ u, 1e-6, flip=False)
    if a.A is not None:  # a's half-spaces travel +d*u past b's static vertices
        _vert_intervals(lo_acc, hi_acc, b.v, a.A, a.b, a.A @ u, 1e-6, flip=True)
    if not lo_acc:
        return np.empty(0), np.empty(0)
    lo = np.clip(np.concatenate(lo_acc), g_lo, None)
    hi = np.clip(np.concatenate(hi_acc), None, g_hi)
    keep = lo <= hi
    return lo[keep], hi[keep]


def clear_dist(probe_hulls, other_hulls, direction, step=0.002, cap=1.0) -> float:
    """Smallest displacement along unit 3-vector ``direction`` (on the ``step``
    grid) at which the probe overlaps none of the others. Returns ``cap`` if
    never clear (the sim's depenetration then owns it).

    Closed-form (T1b, 2026-07-23): instead of re-testing every hull pair at
    every 2 mm step (100-500 full passes on deep overlaps — the measured 45 s
    wedged-probe / 48-59 s contacts costs), each pair contributes its overlap
    d-intervals in ONE (verts x faces) pass (:func:`_pair_overlap_intervals`),
    and the answer is the first grid point not covered by any interval — cost
    independent of step count and cap, exact multi-interval semantics (the
    first clear gap between two blockers wins, as in the legacy loop). A short
    stepping PREFIX (d = 0..4 steps, same predicate) keeps already-clear and
    shallow-contact queries — the common case — at legacy cost or better.
    Direct replacement, no legacy fallback — equivalence is pinned by the
    golden tests in settle_geometry_test.py."""
    ux, uy, uz = direction
    n = int(np.floor(cap / step + 1e-9)) + 1  # grid: 0, step, ..., <= cap
    # Stepping PREFIX (same predicate, not a fallback): shallow answers — 0 to a
    # few grid steps — are the common case (resting contacts are 2-8 mm deep in
    # hull space), and there the legacy-style short-circuiting pass is cheaper
    # than materializing both directions' full interval sets. Intervals take
    # over only when the block is genuinely deep — where they are 5-100x the
    # step loop (measured 45 s wedged probes at cap/step = 100).
    prefix = min(5, n)
    for k in range(prefix):
        d = k * step
        if not any(
            hulls_overlap(h, s, uz * d, dx=ux * d, dy=uy * d)
            for h in probe_hulls
            for s in other_hulls
        ):
            return d
    if prefix == n:
        return cap
    u = np.array([ux, uy, uz], dtype=float)
    covered = np.zeros(n, dtype=bool)
    covered[:prefix] = True  # just proven overlapping at the prefix points
    diff = np.zeros(n + 1, dtype=np.int64)  # interval-stabbing accumulator
    for h in probe_hulls:
        for s in other_hulls:
            lo, hi = _pair_overlap_intervals(h, s, u)
            if not lo.size:
                continue
            i0 = np.maximum(np.ceil((lo - 1e-9) / step).astype(np.int64), 0)
            i1 = np.minimum(np.floor((hi + 1e-9) / step).astype(np.int64), n - 1)
            m = i0 <= i1
            np.add.at(diff, i0[m], 1)
            np.add.at(diff, i1[m] + 1, -1)
    covered |= np.cumsum(diff[:-1]) > 0
    idx = np.flatnonzero(~covered)
    return float(idx[0] * step) if idx.size else cap


def clear_dz(probe_hulls, other_hulls, step=0.002, cap=1.0) -> float:
    """Smallest +z shift at which the probe overlaps none of the others -- the
    settle LIFT metric (how high to release a body so it drops from clear)."""
    return clear_dist(probe_hulls, other_hulls, (0.0, 0.0, 1.0), step=step, cap=cap)


def drop_dz(probe_hulls, other_hulls, cap=1.0) -> float:
    """Largest downward shift d in [0, cap] with the probe overlap-free at EVERY
    intermediate shift -- quasi-static lowering to FIRST CONTACT (the settle
    SNAP-DOWN metric for floating releases: a MoGE-placed pose can float, and
    releasing from height free-falls and topples what a contact release keeps
    upright). 0 if already touching/overlapping; ``cap`` if nothing blocks
    within it (the caller's slab clamp owns the landing). Exact interval math
    (:func:`_pair_overlap_intervals`, so no grid and no tunneling through thin
    blockers); callers add their own release nudge on top."""
    u = np.array([0.0, 0.0, -1.0])
    first = cap
    for h in probe_hulls:
        for s in other_hulls:
            lo, hi = _pair_overlap_intervals(h, s, u)
            if not lo.size:
                continue
            m = hi >= 0.0  # intervals reached by a DOWNWARD (d >= 0) shift
            if not m.any():
                continue
            lo_m = lo[m]
            if (lo_m <= 0.0).any():
                return 0.0  # already in contact/overlap at the current pose
            first = min(first, float(lo_m.min()))
    return min(first, cap)


_PUSH_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))


def min_clear_dist(probe_hulls, other_hulls, step=0.002, cap=1.0) -> float:
    """Penetration DEPTH for the rules gate: the smallest single-direction push
    (+-x, +-y, +z) that separates the probe from the others. The z-only lift
    metric read a hairline LATERAL hull kiss between two lying utensils as the
    neighbor's full height (0720_leanfix_abc1: spoon<->fork reported ~40mm
    'penetration' while the visual meshes were ~1cm apart, at the sibling
    exemption's 20mm cap); the min over push directions reports the true escape
    distance. -z is excluded: separating downward through the support is never
    physically achievable. Each direction's scan is capped at the best depth so
    far, so shallow contacts stay cheap."""
    best = cap
    for u in _PUSH_DIRS:
        best = min(best, clear_dist(probe_hulls, other_hulls, u, step=step, cap=best))
        if best <= 0.0:
            return 0.0
    return best


def pair_clear_dist(a_hulls, b_hulls, step=0.002, cap=1.0) -> float:
    """SYMMETRIC penetration depth of an object<->object pair for the rules gate: the
    smaller of the two ``min_clear_dist`` readings (push a out of b, push b out of a).
    Pushing a CONTAINER out of the cargo seated in it reads the cargo's escape distance,
    not the contact overlap (0904 fruits_plate_to_bowl: plate pushed out of a tomato
    resting in its dish read 44 mm, tomato pushed out of the plate 4-6 mm; the probe
    was simply the alphabetically-first body, so plate<->tomato false-FAILed the gate
    while lemon<->plate passed). The second scan is capped at the first's result, so a
    shallow contact stays cheap."""
    d = min_clear_dist(a_hulls, b_hulls, step=step, cap=cap)
    if d <= 0.0:
        return 0.0
    return min(d, min_clear_dist(b_hulls, a_hulls, step=step, cap=d))


def common_child_lift(child_hulls, static_hulls, nudge=0.002, step=0.002, cap=1.0) -> float:
    """ONE +z shift that clears EVERY child (list-of-hull-lists) of the ``static_hulls``
    (the resettled parent + surfaces) for the scale-commit re-drop. Uniform over all
    children so their mutual stacking is preserved and the lift is order-independent
    (a per-child ``clear_dz`` would depend on which child is lifted first and could
    push one into another). Siblings are deliberately NOT obstacles here — they ride
    up together. Returns ``nudge`` when nothing overlaps (a tiny release gap, e.g. a
    scale-DOWN where children already float clear) and also when a child never clears
    within ``cap`` vertically (a lateral wall overlap): drop in place and let the sim
    depenetrate, mirroring the parent's ``dz>=cap`` fallback."""
    needed = max(
        (clear_dz(hs, static_hulls, step=step, cap=cap) for hs in child_hulls),
        default=0.0,
    )
    return nudge if needed >= cap else nudge + needed
