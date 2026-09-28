"""Per-object planar pose alignment to the reference — the composition backend.

``PoseSession`` is the live machinery behind the composition stage's
investigate_objects/move tools: isolate the object (IoU render = object + its
object-ancestors, transparent silhouette), score candidate poses with the IoU+depth
objective (``register_objective``; ROTATION moves additionally weigh a DINO
patch-similarity term — ``feature_metric`` — since IoU is blind to a ~180-yaw on
near-symmetric objects), and gate accepted winners through the Isaac physics
authority (``composition_physics``). Also home to the hint machinery (orientation /
position / scale) and the shared low-level pieces (RenderClient, _score_pose,
candidate ladders). Renders/transforms run on the warm ``register_blender_server``.

This is the ONLY pose-refinement path. The pre-agent ONE-SHOT register stage (DFS + VLM
axis proposer, ``register_legacy.py``) and its ``--legacy-register`` flag were deleted on
2026-07-27; its one still-live helper now lives in ``ref_render.py``.

World convention: the scene is GRAVITY-ALIGNED -- +Z is up and the XY plane is
horizontal (the ground/table). The camera sits at the world origin, tilted to look down
into the scene. So objects are aligned with 4 DoF: the two horizontal translations (x, y),
the in-plane spin about +Z (rotation), and uniform scale. After EVERY candidate pose, the
render server (a) pushes the object out of any penetration with its support (parent object
+ root surfaces) by a minimal-translation separation (``_resolve_penetration``), then (b)
SEATS it vertically on whatever is actually below it — an all-scene probe (``_seat_z``),
deadbanded so an already-seated candidate keeps its z bit-exact. (b) replaced the old
graph-support-only drop on 2026-08-07: with a wrong support edge it seated candidates
THROUGH un-graphed neighbors (0806 food_packing: every xy candidate re-buried the can
28 mm inside its tray, silently undoing each physics commit's rescue).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from lib.tools.geometry import moge_camera as mc
from lib.tools.geometry import register_objective as ro
from lib.tools.geometry.agentic_mask import slugify
from lib.tools.geometry.surface_relations import surface_build_name

# axis -> (kind, index). kind: 't' translate, 'r' rotate(euler), 's' scale. The world is
# gravity-aligned: x/y are the two horizontal ground-plane directions, rotation is about
# +Z (up). No vertical translate / pitch / roll -- objects stay upright on their support.
# z is assigned by the server on EVERY candidate: push out of support penetration, then
# seat vertically on whatever is actually below (all-scene, deadbanded — see _seat_z), so
# a scale-up never sinks through the table and an xy slide over a neighbor rides ON it.
_AXES = {
    "x-axis": ("t", 0),
    "y-axis": ("t", 1),
    "xy": ("xy", None),  # snap-align: registration-computed planar jump + refine ring
    "rotation": ("r", 2),
    "scale": ("s", 0),
}
_TRANS = (0.05, 0.15, 0.4)  # x object size
_ROT = tuple(math.radians(a) for a in (5, 15, 40))
_SCALE = (0.8, 0.9, 1.1, 1.25)

# Closed-form silhouette scale estimate (``_scale_estimate``): plausibility gate on
# the RAW estimate, clamp applied to the ladder CANDIDATE ("snapscale"), minimum
# centroid-aligned overlap below which the estimate is a mis-registration, the
# radial percentiles matched, and the |s*-1| that triggers the investigate hint.
_SCALE_EST_RANGE = (0.6, 1.6)
_SCALE_CAND_CLAMP = (0.7, 1.4)
_SCALE_MIN_OVERLAP = 0.2
# Elongated-silhouette regime for _scale_estimate: when both silhouettes'
# covariance elongation (sqrt eigenvalue ratio) clears _ELONG_MIN, scale is the
# major-axis extent ratio at P_ELONG_PCT instead of the radial median (which a
# pixel-dense bowl swamps — see _scale_estimate). Values validated offline on
# the fleet's saved scale-hint overlays (logs/replay_0715/scale_elong/).
_ELONG_MIN = 3.0
_ELONG_PCT = 95.0
_SCALE_PCTS = (50, 70, 90)
_SCALE_HINT_THRESH = 0.10
# ASPECT-MISMATCH ceiling for the SIZE hint (see size_hint_aspect_suppressed). `area_scale`
# is an AREA ratio, blind to rotation; `scale_est` is a principal-axis EXTENT ratio. For a
# UNIFORMLY mis-sized object the two measure the same thing and agree; they diverge only
# when the SHAPE disagrees (wrong mesh aspect, occlusion-truncated mask, wrong pose). Since
# move('scale') applies a UNIFORM factor, |log(scale_est/area_scale)| is close to a direct
# measure of how much of the mismatch a resize CANNOT reach. Tuned on the 0721-0803 corpus
# (audits/HINT_LADDER_FALLBACK_PROPOSAL_2026_08_03.md) over 1198 landed scale moves and
# 1416 scale-move outcomes:
#     spread          median dIoU   gained <0.01   moves that DIED
#     < 0.07             +0.120         10%           9% / 25%   (too small / too large)
#     0.07-0.15          +0.040         24%          11% / 30%
#     >= 0.15            +0.030         38%          24% / 47%
#     >= 0.15 & too big  +0.020         43%              47%
# i.e. in that last cell ~70% of the moves return nothing. 0.10 was too aggressive (28% of
# SIZE hints), 0.20 too weak (6%).
_SIZE_ASPECT_MAX = 0.15
# Displacement symptom: how much better the shape fits once translation is
# factored out (centered_overlap − IoU). Gap-based, NOT absolute IoU — thin
# objects (cutlery) sit at low IoU even when perfectly placed (abc2 spoon:
# placed at IoU 0.26, gap 0.04, its 1.47 scale estimate CORRECT; misplaced at
# IoU 0.09, gap 0.33, estimate 1.16 vs true 1.47). Two thresholds, tuned on the
# 125 investigate points of the 0715 fleet (logs/replay_0715 + poshint sweep):
# - POSITION hint fires at gap >= 0.12: precision 0.82 / recall 0.60; the
#   0.125-0.15 band holds 7 confirmed position fixes at identical precision,
#   while 0.10 sits inside the 2-dp IoU rounding noise of well-placed objects.
# - The scale MAGNITUDE is deferred at gap >= 0.25: in the 0.12-0.25 band 3/15
#   flagged estimates were correct and immediately useful (abc1 croissant#2,
#   abc2 cup#0, abc3 stand#1) — there the number is shown with a may-change
#   caveat; every gap >= 0.28 estimate was wrong-or-moot until the object was
#   placed.
#
# WHICH THRESHOLD HIDES THE SIZE NUMBER DEPENDS ON THE SITE — the two disclose
# differently ON PURPOSE, so retuning one constant moves only half the surface:
#   gap        investigate (_object_hint_lines)   post-move (_post_move_size_note)
#   < 0.12     SIZE/DEPTH HINT with the number    number, no caveat
#   0.12-0.25  NO number ("unreliable until       number + "placement may still be
#              placed" — POSITION hint instead)   slightly off" caveat
#   >= 0.25    NO number (same)                   "size not re-measured"
# The 0.25 band above therefore describes the POST-MOVE note only; _SCALE_DEFER_GAP
# is imported at that site alone. Investigate keys its withholding off _POS_HINT_GAP
# because the object has not moved yet: a tempting "~30% too small" there steers the
# agent to move('scale') when the right next call is move('xy'), and scale distorts a
# mesh whose real-world size is trusted. After a move the reading is credible enough
# to show with a caveat. (HARNESS_AUDIT_2026_07_26 N8: the comment used to state the
# 0.25 rule as if it were global.)
_POS_HINT_GAP = 0.12
_SCALE_DEFER_GAP = 0.25
# ROTATION (small-yaw) hint: principal-axis angle of the render silhouette vs the
# GT mask (translation/scale-invariant, so it isolates in-plane yaw from
# position/size). Fires in a BAND, not a floor: below _YAW_HINT_MIN is noise
# (well-aligned median ~2.8 deg on the 0715 fleet). Gated on anisotropy (a
# near-circular silhouette has no defined axis). Tuned on
# logs/replay_0715/rot_probe: 12 keeps the abc4 shelf (14.6) + abc3 mics (24-37),
# drops the occluded abc1 tray (10.9) and near-symmetric mug (11.8).
# The UPPER edge is conditional on elongation — see yaw_hint_fires.
_YAW_HINT_MIN = 12.0
_YAW_HINT_MAX = 40.0
_YAW_HINT_MAX_ELONG = 90.0
_YAW_ANISO_MIN = 1.5
# The hint is an IMAGE-PLANE angle; the move is a world-Z rotation. _yaw_jacobian
# measures the local ratio between them so the seed can be expressed in move degrees.
# Probe size: big enough to clear silhouette quantisation, small enough to stay local.
_YAW_PROBE_DEG = 5.0
# Below this |ratio| the mapping is too degenerate to invert (a 20deg hint would demand
# >80deg of yaw): fall back to seeding the raw hint angle, i.e. the pre-2026-07-27
# behaviour. Guard rails, not tuned values — no fleet distribution measured yet.
_YAW_JACOBIAN_MIN = 0.25
# Past ~90deg of yaw the footprint's orientation has flipped: that is a different pose,
# not a correction, and rotate_180/the coarse ladder own that regime.
_YAW_SEED_MAX_DEG = 90.0
# YAW-REGRESSION tolerance: how much the measured yaw may RISE across a rotation with a
# reliable SUB-band pre reading (pre < _YAW_HINT_MIN) before the move is refused (see the
# gate in optimize_axis). One repeatability p90 — two consecutive readings of an object
# that did NOT rotate differ by p90 1.9-2.8 deg for aniso >= 1.5 (0730 fleet, 1258
# same-object pairs), so 3.0 is the noise band and anything above it is a real regression.
# Every one of the 6 harmful landings measured on static_scene_eval rose by +5.9 to
# +36.7 deg; none of the 4 successes comes near 3.
_YAW_REGRESS_TOL = 3.0
# YAW-IMPROVEMENT demand (2026-08-21): with a reliable pre reading AT/ABOVE _YAW_HINT_MIN
# — an actionable error — mere non-regression is not enough: the misc_online5 stapler's
# round-13 wrong-direction pick re-measured within +3 deg of pre 17 on its foreshortened
# silhouette and slipped through, shipping a ~45 deg residual. A landed rotation must
# either REDUCE the yaw by _YAW_IMPROVE_MIN or END below _YAW_DONE_DEG (safely under
# the 12-deg actionable band — well-aligned objects measure median ~2.8; a landing
# there needs no demand on the delta, e.g. a correct 48-deg fix ending at ~5). 2.0
# clears only the LOWER end of the 1.9-2.8 deg repeatability p90 band above, so at
# the 2.8-deg end a wrong pick can still re-measure lower by noise alone and pass;
# accepted because every wrong-pick post observed on the corpus regressed PAST pre
# (16.7/20.3/21.7/26.5/46.6 deg — all refused), so the residual hole is empirically
# narrow.
_YAW_IMPROVE_MIN = 2.0
_YAW_DONE_DEG = 8.0
# NEAR-MISS clause (2026-08-21, G4): the two rules above leave a hole just past the
# actionable edge — a real partial fix from mid-band pre that lands a hair ABOVE
# _YAW_DONE_DEG but UNDER the actionable band's edge+1 was refused for missing the full
# 2-deg delta (toast-rack 14.3 -> 12.5: a genuine improvement to a sub-actionable
# residual, refused). Accept when the post BOTH moved down by >= _YAW_NEAR_MISS_IMPROVE
# (rules out the flat retry: pre 12.6 -> 12.5 still refuses) AND landed under
# _YAW_NEAR_MISS_DEG = _YAW_HINT_MIN + 1 (rules out far-from-done shuffles: pre 17 ->
# 16 still refuses).
_YAW_NEAR_MISS_IMPROVE = 1.0
_YAW_NEAR_MISS_DEG = 13.0
# 180-FLIP hint deferral: for a STRONGLY elongated silhouette (near-straight —
# cutlery reads 4.2-11, vs ~2.1 for a bent V-shaped tool), an in-band yaw
# misalignment means the DINO flip compare ran on non-comparable stretched crops
# and its margin is noise — defer the flip recommendation until the yaw lands.
# Tested 2026-07-21 on all 229 new-era flags (see CHANGELOG): >= 3.0 suppresses
# exactly the 7 abc2 fork/knife false positives (aniso 4.2-6.8, yaw 12.4-44.8)
# and keeps every known true flip, incl. the reversed+yawed wendy1 screwdrivers
# (aniso 2.1) — a 180-yaw preserves the principal axis for ANY shape, so an
# in-band yaw does NOT rule out a reversal; only the compare's reliability.
_FLIP_DEFER_ANISO_MIN = 3.0
# Which cached hint channels a LANDED move invalidates. Anything absent invalidates
# nothing — see PoseSession._invalidate_hints for the per-channel reasoning.
# "appearance" (the pose-parameterized appearance_agreement cache) follows the
# orientation verdict's rules exactly: a reorientation makes every cached number
# stale (a physics accept rebases poses to zero, so the SAME zero-delta digest
# names a DIFFERENT world pose afterwards), while translations/resizes are
# survived by design — the crop is position/scale-invariant (_orient_object_crop).
_HINT_STALE = {
    "rotate_180": ("orient", "yaw", "scale", "appearance"),
    "rotation": ("orient", "yaw", "scale", "appearance"),
}
# When a HINT fired for an object, its move accepts a smaller gain than the default:
# the measurement is a WEAK contributor to IoU (a correct resize can even LOWER IoU
# until a follow-up placement re-aligns it — abc2 spoon#1: scale gain 0.0077 at one
# pose, 0.0121 after an xy move), so a real hint-backed move hovers at the normal gate.
# The hint is the reliable signal, so trust it at a lower bar. SCALE (size hint) only
# since phase C1: rotation selects and gates on the centered metric, whose response to
# a correct yaw is strong enough to need no relaxation (see _ROT_CEN_MIN_GAIN).
# NOTE those anchor gains are raw-score-era numbers: since the scale cutover the gated
# gain swaps iou for centered_iou, which removes most of the misalignment dip that
# motivated the relaxation. Retained anyway (a lower bar on a stronger-responding
# metric only ADMITS moves the size hint already vouches for) pending a centered-era
# recalibration — the scale rounds' `selection` log now records what it needs.
_HINT_MIN_GAIN = 0.005
# ROTATION acceptance bar on the CENTERED selection score (phase C1, 2026-08-21 — the
# translation-invariant metric now SELECTS rotation candidates, see optimize_axis).
# The raw-IoU bars above/min_gain are calibrated on raw-IoU gains and do NOT carry
# over. Calibrated on the logged 0819+0820+0821 static_scene_benchmark corpus (198
# rotation rounds with a shadow_centered.gain, 66 of them 0821). The bands OVERLAP —
# there is no clean noise/signal gap in 0.010-0.020: the noise population (dead
# rounds whose pick is a fine rung jittering the pose at raw gain 0.0) reads
# <= 0.0032 on 0821 but reaches 0.0104/0.0150/0.0198 across the three days
# (misc_wendy1 screwdriver#0, robodojo mallet#0, robolab spoon#0), while 9
# previously-applied rotations carried centered gains 0.000-0.010 (e.g. 0820 box#0
# 0.0000 at raw 0.0191, 0821 bin#0 0.0100) and every audited CORRECT fix reads
# 0.098-0.52. 0.012 sits inside the audit's sanctioned ~0.01-0.02 window, above
# most of the noise mass; rounds it flips in the marginal band are UNAUDITED —
# the logged gains understate the new search (the fine ladder now recenters on the
# centered winner) and the strengthened yaw gate (see _YAW_IMPROVE_MIN) backstops
# wrong-direction winners there. One bar for seeded and unseeded rotations.
_ROT_CEN_MIN_GAIN = 0.012


def mesh_name_for(category: str, instance: int) -> str:
    return f"obj_{slugify(category)}_{instance}"


def _wall_holdout_names(graph: dict, prepared_surfaces: list[str]) -> list[str]:
    """Exact registered wall roots that exist in the live render scene.

    Walls are depth-only holdouts in object-mask renders: they may hide target pixels,
    but must never contribute their own alpha to the target silhouette. Tables and
    floors stay out deliberately because contact jitter against those support surfaces
    can shave otherwise-correct object masks.
    """
    available = set(prepared_surfaces)
    out: list[str] = []
    for node in graph.get("nodes", []):
        if node.get("kind") != "root_surface" or node.get("category") != "wall":
            continue
        node_id = str(node.get("id") or "").strip()
        build_name = str(
            node.get("build_name") or (surface_build_name(node_id) if node_id else "")
        ).strip()
        if build_name in available and build_name not in out:
            out.append(build_name)
    return out


# --------------------------------------------------------------------------- #
# Blender render-server client                                                #
# --------------------------------------------------------------------------- #
class RenderClient:
    def __init__(
        self,
        blender_cmd: str,
        server: str,
        blend: str,
        res: int = 512,
        log_path: Optional[str] = None,
        ready_timeout: float = 300.0,
    ):
        import time

        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [blender_cmd, "--background", "--python", server, "--", blend, str(res)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("register server exited before ready")
            try:
                if json.loads(line).get("ready"):
                    return
            except json.JSONDecodeError:
                continue
        raise RuntimeError("register server not ready in time")

    def rpc(self, req: dict) -> Optional[dict]:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def close(self):
        try:
            self.rpc({"cmd": "shutdown"})
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()


# --------------------------------------------------------------------------- #
# Per-object alignment                                                        #
# --------------------------------------------------------------------------- #
def _candidates(axis: str, pose: dict, size: float) -> list[dict]:
    """Pose candidates perturbing one axis around the current pose (+ the no-op)."""
    kind, idx = _AXES[axis]
    out = [dict(pose, _tag="keep")]
    if kind == "s":
        for f in _SCALE:
            out.append({**pose, "scale": f, "_tag": f"scale={f}"})
        return out
    steps = [s * size for s in _TRANS] if kind == "t" else list(_ROT)
    field = "translate" if kind == "t" else "euler"
    for st in steps:
        for sign in (+1, -1):
            v = list(pose[field])
            v[idx] = v[idx] + sign * st
            out.append({**pose, field: v, "_tag": f"{axis}{sign:+}{st:.3g}"})
    return out


def _lerp_pose(p0: dict, p1: dict, f: float) -> dict:
    """Interpolate translate/euler/scale from ``p0`` toward ``p1`` by fraction ``f`` —
    used by the feasibility clamp to reduce a single-axis winner move to its largest
    non-penetrating fraction (linear in the yaw angle / translation / scale)."""
    return {
        "translate": [a + f * (b - a) for a, b in zip(p0["translate"], p1["translate"])],
        "euler": [a + f * (b - a) for a, b in zip(p0["euler"], p1["euler"])],
        "scale": p0["scale"] + f * (p1["scale"] - p0["scale"]),
    }


_FINE_SCALE = (0.95, 0.975, 1.025, 1.05)


def _fine_candidates(axis: str, winner: dict, size: float) -> list[dict]:
    """Stage-2 probes around the coarse winner at 1/4 of the smallest coarse step.
    When ``keep`` won stage 1 this polishes around the current pose — the case
    where every coarse step overshoots a near-optimal placement."""
    kind, idx = _AXES[axis]
    if kind == "s":
        return [
            {**winner, "scale": winner["scale"] * f, "_tag": f"fine*{f}"}
            for f in _FINE_SCALE
        ]
    fine = _TRANS[0] / 4 * size if kind == "t" else _ROT[0] / 4
    field = "translate" if kind == "t" else "euler"
    out = []
    for st in (fine, 2 * fine):
        for sign in (+1, -1):
            v = list(winner[field])
            v[idx] = v[idx] + sign * st
            out.append({**winner, field: v, "_tag": f"fine{axis}{sign:+}{st:.3g}"})
    return out


def _phase_shift(a: np.ndarray, b: np.ndarray) -> Optional[tuple[float, float]]:
    """Pixel shift (du, dv) that best moves silhouette ``a`` onto ``b`` — phase
    correlation, with a mask-centroid fallback when the spectral peak is too weak
    (heavy occlusion / shape mismatch). None when either silhouette is empty."""
    if not a.any() or not b.any():
        return None
    A, B = np.fft.rfft2(a.astype(float)), np.fft.rfft2(b.astype(float))
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    dv, du = np.unravel_index(int(np.argmax(r)), r.shape)
    if r[dv, du] < 0.03:  # weak peak: fall back to centroid displacement
        ya, xa = np.nonzero(a)
        yb, xb = np.nonzero(b)
        return float(xb.mean() - xa.mean()), float(yb.mean() - ya.mean())
    if dv > a.shape[0] // 2:
        dv -= a.shape[0]
    if du > a.shape[1] // 2:
        du -= a.shape[1]
    return float(du), float(dv)


def _centered_overlap(render_sil: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU of the two silhouettes after shifting the GT mask's centroid onto the
    render's (integer shift, zero fill) — a shape-compatibility gate that ignores
    the translation error the estimator deliberately factors out."""
    a, b = np.asarray(render_sil) > 0, np.asarray(gt_mask) > 0
    if not a.any() or not b.any():
        return 0.0
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    b = ro._shift2d(
        b, int(round(ya.mean() - yb.mean())), int(round(xa.mean() - xb.mean()))
    )
    return float((a & b).sum() / max((a | b).sum(), 1))


def _axis_angle(mask: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """(major-axis angle in [0,180) degrees, anisotropy) of a silhouette's pixel
    covariance. Anisotropy = sqrt eigenvalue ratio (1 = isotropic, no defined
    axis). None when too few pixels."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) < 20:
        return None, None
    p = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(float)
    w, V = np.linalg.eigh(p.T @ p / len(p))  # ascending
    aniso = math.sqrt(float(w[1]) / max(float(w[0]), 1e-9))
    ang = math.degrees(math.atan2(V[1, 1], V[0, 1])) % 180.0
    return ang, aniso


def _yaw_discrepancy(
    render_sil: np.ndarray, gt_mask: np.ndarray
) -> tuple[Optional[float], Optional[float]]:
    """Undirected principal-axis angle difference (in [0,90]) between the render
    silhouette and the GT mask — the small-yaw signal — plus the min anisotropy of
    the two (the axis is only meaningful when both silhouettes are elongated).
    None when either axis is undefined. Angle is translation/scale-invariant, so a
    difference here is in-plane rotation, not displacement or size."""
    a1, an1 = _axis_angle(render_sil)
    a2, an2 = _axis_angle(gt_mask)
    if a1 is None or a2 is None:
        return None, None
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d), min(an1, an2)


def yaw_hint_fires(yaw: Optional[float], aniso: Optional[float]) -> bool:
    """Is a measured (yaw, aniso) a RECOMMENDABLE move('rotation')? The single
    predicate behind BOTH the agent's YAW HINT and the search's hint seed —
    they must not drift apart.

    The upper edge widens with elongation. `aniso` measures how well DEFINED the
    axis is (it is min(render, mask), so both silhouettes must be elongated), and
    the fleet outcome of every rotation move splits on it, not on the yaw
    magnitude (306 moves, 244 with a known pre-move hint):

        aniso >= 3    yaw 12-40:  31% applied,  6% no-improve, median IoU +0.12
        aniso >= 3    yaw 40-60:  62% applied,  0% no-improve, median IoU +0.24
        aniso >= 3    yaw 60-90:  67% applied,  0% no-improve, median IoU +0.10
        1.5-3         yaw 60-90:  27% applied, 27% no-improve, median IoU +0.01
        aniso < 1.5   yaw 40-90:   0% applied, 75% no-improve

    So a strongly elongated silhouette is recommendable at ANY measured yaw: the
    old "beyond the search's reach (~40 deg)" cap does not hold, because world-Z
    yaw maps non-linearly to the image-plane angle under a tilted camera — the
    +-40 rung took 0716_e2e4_wendy1 marker#0 from 71.4 to 23.1 deg. NOTE that this
    non-linearity cuts BOTH ways (an earlier version of this docstring claimed only
    the favourable direction): the ratio was 0.66 on the 0726_audit_abc4 shelf, where
    the same rung reaches only ~26 deg of image angle. The band above is justified by
    measured OUTCOMES, not by that mechanism; _yaw_jacobian handles the conversion.
    Below
    _FLIP_DEFER_ANISO_MIN the axis is weak, a large reading is as likely to be an
    attitude error (a lying rod at ~90 deg is neither a 'rotation' nor a 180), and
    the move buys ~nothing — so the 40 deg cap stands there."""
    if yaw is None or aniso is None or aniso < _YAW_ANISO_MIN:
        return False
    hi = _YAW_HINT_MAX_ELONG if aniso >= _FLIP_DEFER_ANISO_MIN else _YAW_HINT_MAX
    return _YAW_HINT_MIN <= yaw <= hi


def size_hint_aspect_suppressed(
    scale_est: Optional[float],
    area_scale: Optional[float],
    yaw_aniso: Optional[float],
) -> bool:
    """Is a FLAGGED size reading too aspect-inconsistent to recommend move('scale')?

    ``area_scale`` (AREA ratio) and ``scale_est`` (principal-axis EXTENT ratio) measure the
    same thing for a UNIFORMLY mis-sized object and agree; they diverge only when the SHAPE
    disagrees. move('scale') is uniform, so a large spread means a resize cannot reach the
    mismatch — see ``_SIZE_ASPECT_MAX`` for the measured outcome split.

    Three guards, each forced by the data (all figures in that comment):

    - ``area_scale < 1`` (render LARGER than the photo): the evidence is much weaker on the
      too-small half (24% of moves died vs 47%), so it stays out for now.
    - ``yaw_aniso < _FLIP_DEFER_ANISO_MIN``: above it the relation is NON-monotone
      (33 / 14 / 36% dead by spread band), so this would misfire on cutlery — where scale
      moves fail for an unrelated reason (the settle, see TODO).
    - a MISSING measurement never suppresses. ``scale_est`` is None on 2.8% of flagged rows
      (gated out by ``_SCALE_MIN_OVERLAP``) and ``area_scale`` is None when a silhouette is
      empty; both leave the hint FIRING, i.e. today's behaviour.

    Deliberately NOT applied to the size-locked DEPTH branch: every outcome measured here is
    from an ``axis == "scale"`` move, and a depth correction is a different operator.
    """
    if not scale_est or not area_scale or yaw_aniso is None:
        return False
    if area_scale >= 1.0 or yaw_aniso >= _FLIP_DEFER_ANISO_MIN:
        return False
    return abs(math.log(scale_est / area_scale)) >= _SIZE_ASPECT_MAX


def _axis_extent(ys: np.ndarray, xs: np.ndarray, pct: float) -> tuple[float, float]:
    """(elongation, extent): sqrt eigenvalue ratio of the pixel covariance, and
    the ``pct``-percentile of |projection onto the major axis| about the centroid."""
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
    cov = pts.T @ pts / len(pts)
    w, V = np.linalg.eigh(cov)  # ascending
    elong = math.sqrt(float(w[1]) / max(float(w[0]), 1e-9))
    return elong, float(np.percentile(np.abs(pts @ V[:, 1]), pct))


def _scale_estimate(
    render_sil: np.ndarray,
    gt_mask: np.ndarray,
    min_overlap: float = _SCALE_MIN_OVERLAP,
) -> Optional[float]:
    """Closed-form silhouette scale: center each silhouette on its OWN centroid
    and return s* — the factor the RENDER should grow by to match the photo mask.
    Two regimes: COMPACT shapes use the median over ``_SCALE_PCTS`` of the
    radial-distance percentile ratios (robust to occlusion holes / mask noise);
    ELONGATED shapes (both silhouettes' covariance elongation >= ``_ELONG_MIN``)
    use the major-axis extent ratio at P``_ELONG_PCT`` instead — a spoon is a
    pixel-dense bowl plus a thin handle, so the radial median reads 1.04 while
    the handle is 20% short (0715_fix_abc2 spoon#1: P50 0.98 / P90 1.19); length
    IS the signal there. Centroid alignment factors out translation; radii /
    per-shape principal axes factor out rotation. None when either silhouette is
    empty, the centroid-aligned overlap is under ``min_overlap``
    (mis-registration — the shapes don't even roughly agree), or s* falls
    outside ``_SCALE_EST_RANGE`` (implausible). Pure numpy."""
    a, b = np.asarray(render_sil) > 0, np.asarray(gt_mask) > 0
    if not a.any() or not b.any():
        return None
    if _centered_overlap(a, b) < min_overlap:
        return None
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    el_a, ext_a = _axis_extent(ya, xa, _ELONG_PCT)
    el_b, ext_b = _axis_extent(yb, xb, _ELONG_PCT)
    if min(el_a, el_b) >= _ELONG_MIN:
        s = ext_b / max(ext_a, 1e-9)
    else:
        ra = np.hypot(xa - xa.mean(), ya - ya.mean())
        rb = np.hypot(xb - xb.mean(), yb - yb.mean())
        s = float(
            np.median(
                [np.percentile(rb, p) / max(np.percentile(ra, p), 1e-9)
                 for p in _SCALE_PCTS]  # fmt: skip
            )
        )
    lo, hi = _SCALE_EST_RANGE
    return s if lo <= s <= hi else None


_RING = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _xy_ring(winner: dict, size: float) -> list[dict]:
    """8-direction refinement ring around the xy winner (one fine planar step)."""
    st = 0.05 * size
    out = []
    for dx, dy in _RING:
        k = st / math.sqrt(2) if (dx and dy) else st
        v = list(winner["translate"])
        v[0], v[1] = v[0] + dx * k, v[1] + dy * k
        out.append({**winner, "translate": v, "_tag": f"ring{dx:+}{dy:+}"})
    return out


def _isolate_iou(client: RenderClient, visible_iou: list) -> None:
    """IoU isolation: the target visible, every later entry a depth-only HOLDOUT.

    Holdouts include applicable object ancestors/occluders and registered walls. They
    occlude the target without contributing silhouette pixels themselves.
    """
    client.rpc(
        {"cmd": "isolate", "visible": visible_iou[:1], "holdout": visible_iou[1:]}
    )


def _score_pose(
    client: RenderClient,
    name: str,
    pose: dict,
    mask: np.ndarray,
    center_y: float,
    moge_depth: float,
    tmp_png: str,
    apply: bool = True,
) -> dict:
    """``apply=False`` scores the scene AS IT STANDS (no set_pose) — used after a
    physics settle applied a full matrix that (t,euler,s) can't express; ``pose`` then
    only feeds the depth prior."""
    resp = None
    if apply:
        resp = client.rpc(
            {
                "cmd": "set_pose",
                "name": name,
                "translate": pose["translate"],
                "euler": pose["euler"],
                "scale": pose["scale"],
            }
        )
    client.rpc({"cmd": "render", "out": tmp_png})
    from PIL import Image

    rgba = np.asarray(Image.open(tmp_png).convert("RGBA"))
    depth = -(center_y + pose["translate"][1])  # distance along -Y view dir
    out = ro.objective(rgba, mask, depth=depth, moge_depth=moge_depth)
    out["resolve_offset"] = (resp or {}).get(
        "resolve_offset"
    )  # support-separation shift
    return out


def _alpha_bbox(png_path: str):
    """(x0, y0, x1, y1) of the non-transparent pixels, or None."""
    from PIL import Image

    a = np.asarray(Image.open(png_path).convert("RGBA"))[:, :, 3] > 0
    ys, xs = np.where(a)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _feature_crops(render_png: str, ctx: dict, pad_frac: float = 0.25):
    """Comparable APPEARANCE crops for the DINO patch-similarity term: the isolated
    RGBA render composited on neutral gray, the SAME window from the object's photo
    (``ctx["obj_image"]`` — the de-occluded edit when the object was redetected, so
    the two sides stay consistent), and the GT mask selecting the patches to average.
    GT-mask-only (not mask ∪ render silhouette): it keeps the compared patch set
    FIXED across pose candidates and measures "does the render show the right thing
    where the photo has the object" — the union variant diluted the reversed-spoon
    signal below zero on 0714_flipfirst_abc1. Window = GT-mask bbox + ``pad_frac``
    margin at GT resolution (the render is upsampled to GT res first). None when the
    mask is empty. (The orientation hint uses ``_orient_object_crop`` instead — a
    position-invariant object-on-white crop — because the scoring window here is
    anchored to the GT mask and an agent move shifts the object out of it.)"""
    from PIL import Image

    gw, gh = int(ctx["gt_w"]), int(ctx["gt_h"])
    m = np.load(ctx["mask_path"])
    if m.shape != (gh, gw):
        m = (
            np.asarray(Image.fromarray((m > 0).astype("uint8") * 255).resize((gw, gh)))
            > 127
        )
    else:
        m = m > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    px, py = int((x1 - x0 + 1) * pad_frac), int((y1 - y0 + 1) * pad_frac)
    win = (max(0, x0 - px), max(0, y0 - py), min(gw, x1 + px + 1), min(gh, y1 + py + 1))
    r = Image.open(render_png).convert("RGBA").resize((gw, gh))
    bg = Image.new("RGBA", (gw, gh), (128, 128, 128, 255))
    render_crop = Image.alpha_composite(bg, r).convert("RGB").crop(win)
    photo_crop = Image.open(ctx["obj_image"]).convert("RGB").resize((gw, gh)).crop(win)
    mask_crop = Image.fromarray((m * 255).astype("uint8")).crop(win)
    return render_crop, photo_crop, mask_crop


def _orient_object_crop(img, obj_mask=None, sq: int = 224):
    """POSITION/SCALE-invariant crop for the orientation DINO compare: put the object on
    white, zoom-crop TIGHT to its OWN bbox, and resize that bbox to ``sq`` x ``sq``.
    ``obj_mask`` (bool, same HxW) selects the object for a photo; for an RGBA render pass
    None and the alpha channel is used.

    Two properties matter here. (1) The crop FOLLOWS the object, so an agent xy-move can't
    shift it out of a fixed window and collapse the margin (the 0720 abc1 runtime bug:
    spoon moved before its first hint -> render fell outside the GT-mask window -> +0.144
    read as +0.010, unflagged). (2) Resizing the bbox to a fixed SQUARE normalises shape:
    a round object's cur and flip fill the same square identically (margin ~0, correctly a
    keep -- fixes the donut/plate false positives), while an elongated object's 180-yaw
    still reads as a strong directional change. Square-PADDING instead (aspect-preserving)
    kept those round-object false positives; the stretch is deliberate."""
    from PIL import Image

    arr = np.asarray(img)
    if obj_mask is None:  # RGBA render -> alpha is the object
        a = arr[..., 3] > 127
        rgb = arr[..., :3]
    else:
        a = obj_mask
        rgb = arr[..., :3] if arr.ndim == 3 else arr
    ys, xs = np.where(a)
    if len(xs) == 0:
        return Image.new("RGB", (sq, sq), (255, 255, 255))
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    canv = np.full(rgb.shape, 255, np.uint8)
    canv[a] = rgb[a]
    crop = canv[y0 : y1 + 1, x0 : x1 + 1]
    return Image.fromarray(crop).resize((sq, sq))


def _pose_digest(pose) -> str:
    """Stable cache key for the appearance cache: a session delta-pose dict or a
    4x4 world matrix, flattened and rounded to 1e-6 so float round-tripping never
    splits entries (``round(x, 6) + 0.0`` also folds -0.0 into 0.0 — a bare
    ``%.6f`` prints ``-0.000000`` for a -1e-9 and the digest would flap). Delta
    poses and world matrices deliberately live in separate namespaces ("p:" /
    "m:"): an equivalent pair costs at most one duplicate render, never a wrong
    cache hit."""
    if isinstance(pose, dict):
        vals = [*pose["translate"], *pose["euler"], pose["scale"]]
        tag = "p"
    else:
        vals = np.asarray(pose, dtype=np.float64).reshape(-1).tolist()
        tag = "m"
    return tag + ":" + ",".join(f"{round(float(v), 6) + 0.0:.6f}" for v in vals)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def _carry_parents(nodes: dict, obj_nodes: list) -> dict[str, str]:
    """Carry map (mesh-name -> parent mesh-name) for subtree moves, preferring
    ``support`` over ``parent``. Carrying is a PHYSICAL riding relation, and on
    graphs from before the two fields were unified the VLM's ``support`` is the
    reliable one: the geometric challenge that produced divergent ``parent``
    edges was systematically blind on thin supports (abc2 spoon#0: parent=table,
    support=tray — the tray slid out from under the uncarried spoon). A field is
    used only when it names a known non-root-surface node; otherwise fall through
    to the next."""
    out: dict[str, str] = {}
    for n in obj_nodes:
        for p in (n.get("support"), n.get("parent")):
            if p and p in nodes and nodes[p].get("kind") != "root_surface":
                out[mesh_name_for(*_cat_inst(n))] = mesh_name_for(
                    *_cat_inst(nodes[p])
                )
                break
    return out


# --------------------------------------------------------------------------- #
# PoseSession: live backend for the composition stage's investigate/move tools #
# --------------------------------------------------------------------------- #
def _profile_capability_enabled(
    harness_profile: str,
    harness_profile_manifest: Optional[dict],
    capability: str,
) -> bool:
    """True only for an explicitly selected GPT-6 harness capability.

    The resolved manifest is authoritative, including an explicit false capability; a
    missing manifest fails CLOSED (2026-09-15 owner decision). Baseline never opts in.
    """
    if harness_profile != "gpt6_v1":
        return False
    capabilities = (harness_profile_manifest or {}).get("capabilities")
    # 2026-09-15 owner decision: a missing manifest fails CLOSED.
    return bool(capabilities.get(capability, False)) if capabilities else False


def _runtime_mask_path(value: str, inventory_path: Path) -> str:
    """Resolve a runtime mask path while preserving absolute producer paths."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    scene_relative = inventory_path.parent.parent / path
    inventory_relative = inventory_path.parent / path
    return str(
        scene_relative if scene_relative.exists() else inventory_relative
    )


def _merge_runtime_pose_inventory(
    graph: dict,
    masks: dict[tuple[str, int], dict],
    place: dict[tuple[str, int], dict],
    inventory_path: Path,
) -> None:
    """Merge committed runtime objects into the pose-session input inventories.

    ``masks.json`` remains immutable preprocessing truth.  Runtime masks live in the
    GPT-6 overlay, while graph/placement rows are normally also committed to their
    canonical files for USD export.  Equivalent canonical rows are accepted; an
    identity conflict fails closed instead of silently scoring the wrong object.
    """
    if not inventory_path.exists():
        return
    payload = json.loads(inventory_path.read_text())
    records = payload.get("objects") or []
    if not isinstance(records, list):
        raise ValueError(f"{inventory_path}: 'objects' must be a list")

    graph_nodes = graph.setdefault("nodes", [])
    by_id = {n.get("id"): n for n in graph_nodes if isinstance(n, dict)}
    roots = graph.setdefault("roots", [])
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{inventory_path}: runtime object record is not an object")
        if record.get("status") != "committed":
            raise ValueError(
                f"{inventory_path}: active runtime inventory contains a "
                f"non-committed record ({record.get('id')!r})"
            )
        try:
            category = str(record["category"])
            instance = int(record["instance"])
            object_id = str(record["id"])
            origin = str(record["origin"])
            mask_policy = str(record.get("mask_policy", "tracked"))
            node = dict(record["graph_node"])
            placement = dict(record["placement"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{inventory_path}: incomplete committed runtime object: {exc}"
            ) from exc
        if mask_policy not in {"tracked", "excluded"}:
            raise ValueError(
                f"{inventory_path}: unknown mask_policy {mask_policy!r} "
                f"for {object_id!r}"
            )
        if mask_policy == "excluded":
            nonnull_mask_fields = [
                field
                for field in (
                    "mask_path",
                    "mask_sha256",
                    "scoring_mask_path",
                    "scoring_mask_sha256",
                )
                if record.get(field) is not None
            ]
            if nonnull_mask_fields:
                raise ValueError(
                    f"{inventory_path}: excluded runtime object {object_id!r} has "
                    "non-null mask fields: "
                    + ", ".join(nonnull_mask_fields)
                )
            mask_path = scoring_mask_path = None
        else:
            raw_mask_path = record.get("mask_path")
            raw_scoring_mask_path = record.get("scoring_mask_path") or raw_mask_path
            if (
                not isinstance(raw_mask_path, str)
                or not raw_mask_path.strip()
                or not isinstance(raw_scoring_mask_path, str)
                or not raw_scoring_mask_path.strip()
            ):
                raise ValueError(
                    f"{inventory_path}: tracked runtime object {object_id!r} has "
                    "no nonempty effective mask"
                )
            mask_path = raw_mask_path
            scoring_mask_path = raw_scoring_mask_path
        expected_id = f"{category}#{instance}"
        expected_mesh = mesh_name_for(category, instance)
        if origin not in {"runtime_added", "source_revised"}:
            raise ValueError(
                f"{inventory_path}: unknown origin {origin!r} for {object_id!r}"
            )
        if node.get("kind") != "object":
            raise ValueError(
                f"{inventory_path}: {expected_id} graph node is not an object"
            )
        if object_id != expected_id or node.get("id") != expected_id:
            raise ValueError(
                f"{inventory_path}: runtime identity mismatch for {object_id!r}; "
                f"expected {expected_id!r}"
            )
        if placement.get("category") != category or int(
            placement.get("instance", -1)
        ) != instance:
            raise ValueError(
                f"{inventory_path}: placement identity mismatch for {expected_id}"
            )
        if placement.get("mesh_name") != expected_mesh:
            raise ValueError(
                f"{inventory_path}: {expected_id} must use mesh_name {expected_mesh!r}"
            )
        if mask_policy == "excluded" and placement.get("mask_path") is not None:
            raise ValueError(
                f"{inventory_path}: excluded runtime object {expected_id} has an "
                "active placement mask"
            )

        key = (category, instance)
        existing_node = by_id.get(expected_id)
        if origin == "source_revised":
            if existing_node is None or key not in masks or key not in place:
                raise ValueError(
                    f"{inventory_path}: source_revised object {expected_id} is not "
                    "present in the preprocessing graph/mask/placement inventory"
                )
            if existing_node.get("runtime_added") is True or place[key].get(
                "runtime_added"
            ) is True:
                raise ValueError(
                    f"{inventory_path}: source_revised object {expected_id} resolves "
                    "to a runtime-added canonical record"
                )
            source_mask = _runtime_mask_path(str(masks[key].get("mask_path")), inventory_path)
            revision_mask_value = (
                record.get("source_mask_path")
                if mask_policy == "excluded"
                else mask_path
            )
            if not isinstance(revision_mask_value, str) or not revision_mask_value.strip():
                raise ValueError(
                    f"{inventory_path}: source_revised object {expected_id} has no "
                    "immutable source mask provenance"
                )
            revision_mask = _runtime_mask_path(revision_mask_value, inventory_path)
            if os.path.realpath(source_mask) != os.path.realpath(revision_mask):
                raise ValueError(
                    f"{inventory_path}: source_revised object {expected_id} must "
                    "retain its immutable source mask"
                )
            # The canonical files should already contain this revision; assigning the
            # overlay snapshot makes the live composition view deterministic even for
            # callers that loaded a pre-commit placement mapping.
            if mask_policy == "excluded":
                # The source mask remains immutable provenance, not active scoring
                # evidence for direct-code replacement geometry.
                masks.pop(key, None)
            else:
                resolved_scoring_mask = _runtime_mask_path(
                    scoring_mask_path, inventory_path
                )
                if os.path.realpath(resolved_scoring_mask) != os.path.realpath(
                    source_mask
                ):
                    revised_mask = dict(masks[key])
                    revised_mask["mask_path"] = resolved_scoring_mask
                    masks[key] = revised_mask
            place[key] = placement
            continue

        if node.get("runtime_added") is not True or placement.get(
            "runtime_added"
        ) is not True:
            raise ValueError(
                f"{inventory_path}: runtime_added object {expected_id} is missing "
                "runtime provenance flags"
            )
        if existing_node is None:
            graph_nodes.append(node)
            by_id[expected_id] = node
            parent = node.get("parent")
            if parent:
                if parent not in by_id:
                    raise ValueError(
                        f"{inventory_path}: {expected_id} has unknown parent {parent!r}"
                    )
                children = by_id[parent].setdefault("children", [])
                if expected_id not in children:
                    children.append(expected_id)
            elif expected_id not in roots:
                roots.append(expected_id)
        elif (
            _cat_inst(existing_node) != (category, instance)
            or existing_node.get("runtime_added") is not True
            or existing_node.get("parent") != node.get("parent")
        ):
            raise ValueError(
                f"{inventory_path}: graph identity conflict for {expected_id}"
            )

        existing_mask = masks.get(key)
        if mask_policy == "excluded":
            if existing_mask is not None:
                raise ValueError(
                    f"{inventory_path}: excluded runtime object {expected_id} "
                    "collides with an active scoring mask"
                )
        else:
            runtime_mask = {
                "category": category,
                "instance": instance,
                "mask_path": _runtime_mask_path(scoring_mask_path, inventory_path),
                "runtime_added": True,
            }
            if existing_mask is not None and os.path.realpath(
                str(existing_mask.get("mask_path"))
            ) != os.path.realpath(runtime_mask["mask_path"]):
                raise ValueError(
                    f"{inventory_path}: mask conflict for runtime object {expected_id}"
                )
            masks.setdefault(key, runtime_mask)

        existing_place = place.get(key)
        if existing_place is not None and (
            existing_place.get("mesh_name") != expected_mesh
            or existing_place.get("runtime_added") is not True
        ):
            raise ValueError(
                f"{inventory_path}: placement conflict for runtime object {expected_id}"
            )
        place[key] = placement


class PoseSession:
    """The register machinery as a LIVE session instead of a one-shot stage.

    The merged composition stage exposes two agent tools — ``investigate_objects``
    (comparable render/photo crops) and ``move(object, aspect)`` (backend-chosen
    1-D correction) — and this class is their engine: the warm
    ``register_blender_server`` holding the shared blend, one register ``ctx`` per
    meshed object, per-object pose state (deltas relative to the session load), and
    the register-style ``register/register.json`` log the web demo already reads.
    ``save()`` writes the current poses back to the shared blend (the composition
    executor re-opens that file per call, so its renders pick the moves up);
    ``rebuild()`` reloads after a manual scene edit made the server copy stale."""

    def __init__(
        self,
        blend: str,
        scene_graph_path: str,
        masks_json: str,
        placement_json: str,
        image_path: str,
        work_dir: str,
        blender_cmd: str,
        moge_json: Optional[str] = None,
        harness_profile: str = "baseline",
        harness_profile_manifest: Optional[dict] = None,
    ) -> None:
        self._init_args = (
            blend, scene_graph_path, masks_json, placement_json,
            image_path, work_dir, blender_cmd, moge_json,
            harness_profile, harness_profile_manifest,
        )
        self.harness_profile = harness_profile
        self.harness_profile_manifest = harness_profile_manifest
        self.strict_post_edit_physics = _profile_capability_enabled(
            harness_profile,
            harness_profile_manifest,
            "strict_post_edit_physics",
        )
        self.allow_procedural_empty_roots = _profile_capability_enabled(
            harness_profile,
            harness_profile_manifest,
            "runtime_object_inventory",
        )
        self.blend = blend
        self.image_path = image_path
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        graph = json.load(open(scene_graph_path))
        masks = {
            (r["category"], r["instance"]): r
            for r in json.load(open(masks_json))["instances"]
        }
        place = {
            (o["category"], o["instance"]): o
            for o in json.load(open(placement_json))["objects"]
        }
        if self.allow_procedural_empty_roots:
            _merge_runtime_pose_inventory(
                graph,
                masks,
                place,
                Path(placement_json).parent / "runtime_objects" / "inventory.json",
            )
        nodes = {n["id"]: n for n in graph["nodes"]}
        cam_cfg = None
        if moge_json and os.path.exists(moge_json):
            mj = json.load(open(moge_json))
            grav = mj.get("gravity") or {}
            R, Tc = grav.get("R"), grav.get("T")
            cam_cfg = mc.camera_config_from_moge(
                mj,
                R=np.array(R) if R is not None else None,
                T=np.array(Tc) if Tc is not None else None,
            )
        if cam_cfg is not None:
            self.gt_w, self.gt_h = (
                int(cam_cfg["resolution_x"]),
                int(cam_cfg["resolution_y"]),
            )
        else:
            from PIL import Image

            self.gt_w, self.gt_h = Image.open(image_path).size

        # DFS object order (parents before children) plus the stacking map.
        order, seen = [], set()

        def visit(nid):
            n = nodes.get(nid)
            if not n or nid in seen:
                return
            seen.add(nid)
            if n.get("kind") != "root_surface":
                order.append(nid)
            for c in n.get("children", []):
                visit(c)

        for r in graph.get("roots", []):
            visit(r)
        obj_nodes = [
            n for n in (nodes[i] for i in order) if n.get("kind") != "root_surface"
        ]
        parents = _carry_parents(nodes, obj_nodes)
        server = os.path.join(os.path.dirname(__file__), "register_blender_server.py")
        # Scoring renders at GT resolution: the server's default (width-less) render
        # scales the blend's long side to `res`, and _score_pose is the only default-
        # res consumer on this client. At 512 a thin object was scored on ~150 px
        # (spoon#1: IoU quantized, size_loss 2x overstated); GT-res costs ~2.9x per
        # render (183 -> 525 ms on H100 EEVEE) — negligible against the pipeline —
        # and lets the objective compare the mask natively (no downsample).
        self.client = RenderClient(
            blender_cmd,
            server,
            blend,
            res=max(self.gt_w, self.gt_h),
            log_path=str(self.work / "server.log"),
        )
        prep = self.client.rpc(
            {
                "cmd": "prepare",
                "parents": parents,
                "objects": [mesh_name_for(*_cat_inst(n)) for n in obj_nodes],
                "allow_empty_roots": self.allow_procedural_empty_roots,
            }
        )
        prepared = set(prep.get("prepared", [])) if prep else set()
        self.surfaces = list(prep.get("surfaces", [])) if prep else []
        self.wall_holdouts = _wall_holdout_names(graph, self.surfaces)
        self.prepared = sorted(prepared)  # every blend handle (physics colliders)
        self.parents = parents  # mesh-name -> parent mesh-name (subtree carries)
        self.physics = None  # CompositionPhysics, attached lazily by composition.
        self.ctx: dict[str, dict] = {}
        self.pose: dict[str, dict] = {}
        self.cur: dict[str, Optional[dict]] = {}
        self.trace: dict[str, list] = {}
        # last (yaw_deg, aniso) from scale_hint per object — the rotation search
        # seeds on it, and the yaw gate reuses it as the pre-move reading.
        self._yaw_hint: dict[str, tuple] = {}
        # objects whose SIZE hint fired (area mismatch) -> the scale move relaxes its
        # accept gate to _HINT_MIN_GAIN (set in scale_hint).
        self._scale_hint: dict[str, float] = {}
        # cached 180-orientation verdict per object. Computed ONCE at the object's
        # first (stable, ~post-settle) pose and frozen -- recomputing on the agent's
        # perturbed poses each investigate made the DINO margin swing sign (0720 abc1
        # spoon: −0.106 → +0.072 across inv5/6/7) and false-flagged the moved mug.
        # Invalidated on an APPLIED rotate_180 so the post-flip re-check reflects the
        # new orientation (see rotate_180 / orientation_hint).
        self._orient_hint: dict[str, dict] = {}
        # pose-parameterized appearance scores (obj_id -> {pose digest: sim}) — the
        # FACING check's sim0 quantity served per pose (see appearance_agreement).
        # Same lifecycle as _orient_hint: cleared per object by a LANDED
        # reorientation (_invalidate_hints) and wholesale by a session rebuild
        # (this __init__) — a stale appearance number is worse than none.
        self._appearance_cache: dict[str, dict] = {}
        # per-object photo-side appearance crop (_orient_object_crop of obj_image
        # under the GT mask) — the photo never changes within a session.
        self._photo_crop_cache: dict = {}
        # per-object occluders (obj_id -> [occluder obj_id]) from the preprocessing
        # generative-resegment pass (objects resting on / in front of it). Held out in the
        # size-metric render so its silhouette carries the SAME holes as the modal SAM3
        # mask -- otherwise the isolated (full) render vs the occluded mask reads as a
        # false size mismatch (0720 abc1 placemat: 5 objects on it -> a bogus "17% too large").
        self._occluders_of: dict[str, list] = {}
        try:
            _edges = json.loads(
                (Path(masks_json).parent / "generative_resegment.json").read_text()
            )["edges"]
            for _e in _edges:
                if _e.get("occluded") and _e.get("occluder"):
                    self._occluders_of.setdefault(_e["occluded"], []).append(
                        _e["occluder"]
                    )
        except Exception:  # noqa: BLE001 - occlusion record is best-effort
            pass
        # SUPPORT-GRAPH CHILDREN, transitively (parent obj_id -> [descendant obj_id]).
        # Unioned into the holdout below: a child resting ON the parent is excluded from
        # the parent's MODAL mask by construction (SAM3 gives those px to the child), so
        # it holes that mask whether or not the VLM occlusion pass listed it. The VLM
        # catches 87% of parent<-child edges fleet-wide (299 of 343) — this is the
        # deterministic backstop for the other 44. 0727_tl_e2e_abc4 stand#0 listed only
        # 1 of its 4 cups: the render came out 12416 px too full against a 68246 px mask
        # (area_scale 0.802 vs 0.854 corrected, IoU 0.579 vs 0.634, yaw 24.2 vs 19.1deg).
        # A holdout that does not overlap the target punches nothing, so a false
        # inclusion costs zero while a false exclusion is a permanent size bias.
        self._children_of: dict[str, list] = {}
        for _n in graph["nodes"]:
            if _n.get("kind") != "object":
                continue
            _p = _n.get("parent")
            while _p and _p in nodes and nodes[_p].get("kind") != "root_surface":
                self._children_of.setdefault(_p, []).append(_n["id"])
                _p = nodes[_p].get("parent")
        # occluders ERASED from a redetected object's amodal mask (redetect['removed']).
        # For a redetected object the mask is the de-occluded extent, so those occluders
        # must NOT be held out in the size render -- holding them out re-punches holes the
        # mask no longer has (0720 abc1 tray#0 read a false 37% "too small"). See
        # _occluder_holdout: for a redetected object we hold out occluders MINUS removed.
        self._removed_of: dict[str, list] = {}
        try:
            for _r in json.loads(Path(masks_json).read_text()).get("instances", []):
                _rd = _r.get("redetect") or {}
                if _rd.get("removed"):
                    self._removed_of[f"{_r['category']}#{_r['instance']}"] = list(
                        _rd["removed"]
                    )
        except Exception:  # noqa: BLE001 - redetect record is best-effort
            pass
        self.rounds = 0
        for n in obj_nodes:
            cat, inst = _cat_inst(n)
            key = (cat, inst)
            name = mesh_name_for(cat, inst)
            if key not in masks or key not in place or name not in prepared:
                continue
            visible_iou = [name]
            p = n.get("parent")
            while p and p in nodes and nodes[p].get("kind") != "root_surface":
                visible_iou.append(mesh_name_for(*_cat_inst(nodes[p])))
                p = nodes[p].get("parent")
            center = place[key]["center"]
            obj_image, obj_mask = image_path, masks[key]["mask_path"]
            rd = masks[key].get("redetect") or {}
            # Border completion re-frames the mask/image (pad-shifted); register renders
            # through the ORIGINAL GT camera, so the shifted mask would never align —
            # score the original visible silhouette instead (its placement was already
            # corrected upstream by the completed cloud).
            is_rd = bool(
                rd.get("mask_path")
                and os.path.exists(rd["mask_path"])
                and not rd.get("border")
            )
            if is_rd:
                obj_mask = rd["mask_path"]
                if rd.get("edited_image") and os.path.exists(rd["edited_image"]):
                    obj_image = rd["edited_image"]
                # the redetect mask is the FULL de-occluded extent: carving the
                # render with ancestor holdouts guaranteed a permanent mismatch
                # (capped IoU, biased snap) — score the full silhouette instead.
                visible_iou = [name]
            self.ctx[n["id"]] = {
                "mesh_name": name,
                "mask_path": obj_mask,
                # Agent-facing composition crops use the original/modal mask when
                # they show the original photograph.  ``mask_path`` may instead be
                # an amodal redetect and remains the source for internal pose metrics.
                "visible_mask_path": masks[key]["mask_path"],
                "obj_image": obj_image,
                "redetect": is_rd,
                "visible_iou": visible_iou,
                "visible_ctx": visible_iou + self.surfaces,
                "gt_w": self.gt_w,
                "gt_h": self.gt_h,
                "size": float(max(place[key]["size"])),
                "center_y": float(center[1]),
                "depth": float(place[key]["depth"]),
            }
            self.pose[n["id"]] = {
                "translate": [0.0, 0.0, 0.0],
                "euler": [0.0, 0.0, 0.0],
                "scale": 1.0,
            }
            self.cur[n["id"]] = None
            self.trace[n["id"]] = []
        # Scoring isolation = ancestors (visible_iou) PLUS the object's occluders and
        # registered walls held out. A wall in front therefore punches transparent holes
        # in the target silhouette; a wall behind it changes nothing. Walls are holdouts,
        # never visible contributors, so their own large alpha cannot inflate the target
        # mask. Redetect-aware object-occluder handling remains unchanged. Second pass
        # because _occluder_holdout reads OTHER objects' ctx, so ctx must be complete.
        for _oid in self.ctx:
            self.ctx[_oid]["visible_score"] = list(
                dict.fromkeys(
                    self.ctx[_oid]["visible_iou"]
                    + self._occluder_holdout(_oid)
                    + self.wall_holdouts
                )
            )
        # Warm the DINO feature worker (separate spawned process — importing torch
        # in THIS process after CoACD hangs) now, before any physics attach, so the
        # first rotation move / orientation hint doesn't pay the model-load latency.
        try:
            from lib.tools.geometry import feature_metric as fm

            fm.warmup()
        except Exception as exc:  # noqa: BLE001 - appearance term is optional
            print(f"[feature-metric] warmup failed: {exc}", file=sys.stderr)

    # -- queries ------------------------------------------------------------- #
    def objects(self) -> list[str]:
        """Ids of every meshed, masked object the session can score/move."""
        return list(self.ctx)

    def score(self, obj_id: str) -> dict:
        """Score ``obj_id`` at its current pose (cached until a move changes it)."""
        if self.cur.get(obj_id) is None:
            c = self.ctx[obj_id]
            _isolate_iou(self.client, c["visible_score"])
            self.cur[obj_id] = _score_pose(
                self.client,
                c["mesh_name"],
                self.pose[obj_id],
                np.load(c["mask_path"]),
                c["center_y"],
                c["depth"],
                str(self.work / f"{c['mesh_name']}_iou.png"),
            )
            # Persist the baseline as soon as it exists. _dump() is otherwise only reached
            # from a move round / flip / rebuild, so a scene where every placement was
            # accepted as-is finished with NO register.json at all and read as unmeasurable
            # downstream — robodojo_fold_clothes_standard did exactly that in 3 of 7
            # benchmark passes (it has a single registerable object, so "no move" is common;
            # its obj_*_iou.png was written here while the score died with the process).
            # An end-of-session dump would not help: PoseSession.close() is never called.
            self._dump()
        return self.cur[obj_id]

    def _snap_candidates(self, obj_id: str, pose: dict) -> list[dict]:
        """The ``xy`` snap: phase-correlate the current silhouette against the GT
        mask for the pixel offset, estimate the pixel<-world Jacobian with two
        eps-probes (one render each), and solve the 2x2 for the planar jump. One
        computed diagonal move instead of two axis line-searches. Returns [] when
        registration is unusable (empty/occluded silhouette, singular Jacobian,
        implausibly large jump) — the caller then just runs the refinement ring."""
        from PIL import Image

        c = self.ctx[obj_id]
        name = c["mesh_name"]
        # GT resolution (matches the gt_w x gt_h renders below): _phase_shift FFT-
        # correlates cur against gt and needs identical shapes -- a raw np.load of a
        # mask stored at a different resolution than gt_w/gt_h broadcast-fails
        # (0717 e2e8 abc1 placemat: render 768x681 vs mask 864x769).
        gt = self._gt_mask_full(obj_id)
        tmp = str(self.work / "_snap.png")

        def sil(p: dict) -> np.ndarray:
            # render_only: measurement render, must not resolve/seat (F0a)
            self.client.rpc(
                {"cmd": "set_pose", "name": name, "translate": p["translate"],
                 "euler": p["euler"], "scale": p["scale"], "render_only": True}  # fmt: skip
            )
            self.client.rpc(
                {"cmd": "render", "out": tmp, "transparent": True,
                 "width": self.gt_w, "height": self.gt_h}  # fmt: skip
            )
            return np.asarray(Image.open(tmp).convert("RGBA"))[..., 3] > 0

        cur = sil(pose)
        target = _phase_shift(cur, gt)
        if target is None:
            return []
        eps = 0.05 * c["size"]
        probes = []
        for i in (0, 1):
            v = list(pose["translate"])
            v[i] += eps
            probes.append(_phase_shift(cur, sil({**pose, "translate": v})))
        if any(p is None for p in probes):
            return []
        J = np.array([[probes[0][0], probes[1][0]], [probes[0][1], probes[1][1]]])
        J /= eps
        if abs(float(np.linalg.det(J))) < 1e-6:
            return []
        dxy = np.linalg.solve(J, np.asarray(target, dtype=float))
        if not np.isfinite(dxy).all() or float(np.linalg.norm(dxy)) > 2.0 * c["size"]:
            return []  # a snap beyond 2 object sizes is a mis-registration
        v = list(pose["translate"])
        v[0], v[1] = v[0] + float(dxy[0]), v[1] + float(dxy[1])
        return [{**pose, "translate": v, "_tag": "snap"}]

    def _occluder_holdout(self, obj_id: str) -> list:
        """Mesh names of ``obj_id``'s preprocessing occluders (objects resting on / in
        front of it, from generative_resegment) UNION its support-graph children, that
        are in the session and not already in its ``visible_iou``. Held out during the
        SIZE-metric render so the rendered silhouette carries the SAME holes the mask
        has (an object on a placemat punches a hole in both), instead of a full
        silhouette that reads as 'too large'.

        The children term (2026-07-27) is a deterministic backstop for the VLM pass,
        which misses some of them — see _children_of for the fleet split and the
        stand#0 measurement. It changes nothing for an object without children, and a
        child that does not overlap the parent on screen punches no hole, so a false
        inclusion is free.

        The holes must match WHICH mask the object uses (see the ctx build):
        - NON-redetected (case A, incl. boundary-cut that wasn't redetected, and boundary
          redetects where ``is_rd`` is False): the mask is the ORIGINAL modal mask with
          every object hole -> hold out ALL occluders.
        - REDETECTED by object occlusion (case B1, ``ctx['redetect']``): the mask is the
          AMODAL de-occluded extent with the ``redetect['removed']`` occluders erased ->
          hold out occluders MINUS ``removed`` (empirically all of them, so usually none;
          holding an erased occluder re-punches a hole the mask no longer has -> false
          'too small', 0720 abc1 tray#0). Any un-removed occluder still holes the amodal
          mask, so it IS held out (correct even if de-occlusion is only partial)."""
        c = self.ctx[obj_id]
        seen = set(c.get("visible_iou") or [])
        removed = (
            set(getattr(self, "_removed_of", {}).get(obj_id, []))
            if c.get("redetect")
            else set()
        )
        out = []
        # occluders FIRST, then children, deduped — the `removed` and `seen` filters
        # below then apply to both, which is what keeps the redetect cases correct: a
        # child erased from an amodal mask is subtracted here and ends up HIDDEN, not
        # held out (re-punching that hole is the 0720 abc1 tray#0 false 'too small').
        cand = list(
            dict.fromkeys(
                list(getattr(self, "_occluders_of", {}).get(obj_id, []))
                + list(getattr(self, "_children_of", {}).get(obj_id, []))
            )
        )
        for occ in cand:
            if occ in removed:  # erased from the amodal mask -> no hole to reproduce
                continue
            mn = (self.ctx.get(occ) or {}).get("mesh_name")
            if mn and mn not in seen:
                out.append(mn)
                seen.add(mn)
        return out

    def _object_sil(self, obj_id: str, pose: dict) -> np.ndarray:
        """GT-resolution silhouette (alpha>0) of the object at ``pose``, rendered
        OCCLUSION-CONSISTENT with the modal SAM3 mask: the object visible, its ancestors
        AND its preprocessing occluders and registered walls as holdouts (objects/walls
        in front punch the same holes the photo mask has). The canonical scoring
        isolation remains active afterwards."""
        from PIL import Image

        c = self.ctx[obj_id]
        tmp = str(self.work / "_scale_sil.png")
        # Use the same wall-aware isolation as the objective. Otherwise scale/yaw hints
        # can still reward an object that the contextual wall completely hides.
        isolation = list(
            dict.fromkeys(
                c["visible_iou"]
                + self._occluder_holdout(obj_id)
                + list(getattr(self, "wall_holdouts", []))
            )
        )
        _isolate_iou(self.client, isolation)
        # render_only (F0a): a measurement must not mutate the pose it measures. The
        # plain set_pose re-ran the penetration resolver + vertical seat even when
        # RESTORING the current pose, silently re-seating physics-rested cargo onto the
        # render mesh (~6 mm inside its CoACD hull in a concave support) — the origin of
        # the still_life certify topples (CARGO_STABILITY_FIX_PLAN_2026_08_08 R1).
        self.client.rpc(
            {"cmd": "set_pose", "name": c["mesh_name"], "translate": pose["translate"],
             "euler": pose["euler"], "scale": pose["scale"], "render_only": True}  # fmt: skip
        )
        self.client.rpc(
            {"cmd": "render", "out": tmp, "transparent": True,
             "width": self.gt_w, "height": self.gt_h}  # fmt: skip
        )
        sil = np.asarray(Image.open(tmp).convert("RGBA"))[..., 3] > 0
        return sil

    def _yaw_jacobian(self, obj_id: str, pose: dict) -> Optional[float]:
        """d(image-plane principal-axis angle) / d(euler[2]), measured locally by a
        central difference around ``pose``. None when it cannot be measured.

        The ROTATION hint is an IMAGE-PLANE angle (``_yaw_discrepancy``: the angle
        between two 2D silhouettes' principal axes) but the move rotates about world
        Z. Under an oblique camera those are different quantities that happen to share
        a unit, and the ratio between them is neither 1 nor constant — measured 0.66 on
        the 0726_audit_abc4 shelf (a 21.7deg seed removed only 14.4deg of discrepancy)
        and >1.1 on the 0716_e2e4_wendy1 marker. It also depends on the CURRENT yaw, so
        it is a local derivative, not a per-object constant: measure it, don't cache it.

        Cost: 2 renders, on top of the 11-13 a rotation move already spends.
        Best-effort like every other advisory measurement here: a failed probe returns
        None and the caller seeds the raw hint angle (the pre-2026-07-27 behaviour)."""
        try:
            angs = []
            for sgn in (1, -1):
                e = list(pose["euler"])
                e[2] += sgn * math.radians(_YAW_PROBE_DEG)
                ang, _ = _axis_angle(self._object_sil(obj_id, {**pose, "euler": e}))
                if ang is None:
                    break
                angs.append(ang)
            # _object_sil restores the PLAIN iou isolation, but a scoring caller set up
            # visible_score (iou + occluder holdout) — put it back or every later
            # candidate renders without the holes the mask has.
            _isolate_iou(self.client, self.ctx[obj_id]["visible_score"])
            if len(angs) != 2:
                return None
            # principal axes live in [0,180): wrap the difference into (-90, 90]
            d = (angs[0] - angs[1] + 90.0) % 180.0 - 90.0
            return d / (2.0 * _YAW_PROBE_DEG)
        except Exception as exc:  # noqa: BLE001 - probe is advisory; fall back to raw
            print(f"[yaw-jacobian] {obj_id}: {exc}", file=sys.stderr)
            return None

    def mean_iou(self, mesh_names: list) -> Optional[float]:
        """Mean silhouette IoU over the given BLEND handles at the CURRENT blend
        state (matrices as-is — no set_pose): the carry-decision metric for
        parent moves (children follow only when following scores higher).
        Unknown handles are skipped; None when nothing is scoreable."""
        try:
            from PIL import Image

            mesh2oid = {v["mesh_name"]: k for k, v in self.ctx.items()}
            tmp = str(self.work / "_carry_iou.png")
            vals = []
            for m in mesh_names:
                oid = mesh2oid.get(m)
                if oid is None:
                    continue
                c = self.ctx[oid]
                _isolate_iou(self.client, c["visible_score"])
                self.client.rpc(
                    {"cmd": "render", "out": tmp, "transparent": True,
                     "width": self.gt_w, "height": self.gt_h}  # fmt: skip
                )
                sil = np.asarray(Image.open(tmp).convert("RGBA"))[..., 3] > 0
                mask = self._gt_mask_full(oid)
                union = float(np.logical_or(sil, mask).sum())
                vals.append(
                    float(np.logical_and(sil, mask).sum()) / union if union else 0.0
                )
            return round(sum(vals) / len(vals), 4) if vals else None
        except Exception as exc:  # noqa: BLE001 - decision degrades to always-carry
            print(f"[carry-iou] {exc}", file=sys.stderr)
            return None

    def _gt_mask_full(self, obj_id: str) -> np.ndarray:
        """The object's GT mask as a bool array at GT resolution."""
        from PIL import Image

        m = np.load(self.ctx[obj_id]["mask_path"])
        if m.shape != (self.gt_h, self.gt_w):
            m = (
                np.asarray(
                    Image.fromarray((m > 0).astype("uint8") * 255).resize(
                        (self.gt_w, self.gt_h)
                    )
                )
                > 127
            )
        return m > 0

    def _scale_snap_candidate(self, obj_id: str, pose: dict) -> Optional[dict]:
        """The closed-form "snapscale" ladder candidate: the uniform factor that makes
        the render's silhouette AREA match the mask's — ``sqrt(A_mask/A_render)`` — applied
        to the current scale, clamped to ``_SCALE_CAND_CLAMP``. Area (not the principal-axis
        EXTENT) is what a uniform scale can actually achieve and what the objective's
        size term optimizes, so the seeded candidate is one the search will accept; an
        extent deficit that exceeds the area deficit is an aspect mismatch a uniform
        scale can't fix (surfaced by the SIZE hint instead). None when unusable — the
        ladder then behaves exactly as today. Best-effort."""
        try:
            sil = self._object_sil(obj_id, pose)
            mask = self._gt_mask_full(obj_id)
            a_ren, a_msk = float((sil > 0).sum()), float((mask > 0).sum())
            if not a_ren or not a_msk:
                return None
            s = float(np.clip((a_msk / a_ren) ** 0.5, *_SCALE_CAND_CLAMP))
            return {**pose, "scale": pose["scale"] * s, "_tag": "snapscale"}
        except Exception as exc:  # noqa: BLE001 - estimator is an extra rung only
            print(f"[scale-snap] {obj_id}: {exc}", file=sys.stderr)
            return None

    def _rotation_feat(self, obj_id: str):
        """Callable(render_png) -> DINO patch-similarity of the isolated render vs
        the object's photo crop, or None when the feature worker is unavailable
        (rotation selection then drops the appearance term and runs on the
        centered score alone — see optimize_axis, phase C1)."""
        try:
            from lib.tools.geometry import feature_metric as fm

            if not fm.available():
                return None
        except Exception:  # noqa: BLE001
            return None
        c = self.ctx[obj_id]

        def fn(png: str) -> float:
            try:
                crops = _feature_crops(png, c)
                if crops is None:
                    return 0.0
                return float(fm.patch_sim(crops[0], crops[1], mask=crops[2]))
            except Exception as exc:  # noqa: BLE001 - keep scoring on IoU alone
                print(f"[feature-metric] {obj_id}: {exc}", file=sys.stderr)
                return 0.0

        return fn

    # -- the move backend ------------------------------------------------------ #
    def optimize_axis(self, obj_id: str, axis: str, min_gain: float = 0.01) -> dict:
        """ONE register round on a FIXED axis: coarse candidate ladder, then a fine
        ladder around the coarse winner (1/4 of the smallest coarse step); keep the
        overall best if the TOTAL gain clears ``min_gain`` — except ROTATION and
        SCALE, which select and gate on the CENTERED (translation-invariant) score:
        rotation against ``_ROT_CEN_MIN_GAIN`` (phase C1), scale against the same
        numeric bars as before the cutover (``min_gain`` / ``_HINT_MIN_GAIN`` — see
        the acceptance comment below). Every candidate is separated from
        its support before scoring (the server's set_pose), so anti-penetration +
        rest-on-support hold by construction. Returns ``applied``, ``axis``, ``before``,
        ``after``, ``pose``, and the diagnostic fields ``physics``, ``clamp_reason``,
        ``clamped_frac``, ``yaw_regressed``, ``range_limited``, and — on an APPLIED
        'rotation' move, None everywhere else — the edit-feedback evidence pairs
        ``app_pre``/``app_post`` (appearance agreement vs the photo at the pre-move /
        rested pose) and ``yaw_pre``/``yaw_post`` with ``yaw_pre_aniso``/
        ``yaw_post_aniso`` (the gate-measured pair + the anisos it was measured
        at, only when BOTH readings are reliable and the full winner landed); it
        also logs a register-style trace row."""
        if axis not in _AXES:
            raise ValueError(f"unknown axis {axis!r}; options: {sorted(_AXES)}")
        strict_physics = (
            getattr(self, "harness_profile", "baseline") == "gpt6_v1"
            and bool(getattr(self, "strict_post_edit_physics", False))
        )
        c = self.ctx[obj_id]
        name = c["mesh_name"]
        mask = np.load(c["mask_path"])
        iou_png = str(self.work / f"{name}_iou.png")
        _isolate_iou(self.client, c["visible_score"])
        before = self.score(obj_id)
        pose = self.pose[obj_id]
        # P1 (bridge_6): snapshot the pre-move WORLD matrix — a physics accept
        # rebases the pose to zero deltas, so the keep pose stops being
        # expressible as (t, euler, s) and the landed-rotation appearance pair
        # below must render it from this matrix. One get_matrix RPC (no render),
        # rotation only; best-effort — None just omits app_pre.
        pre_world_m = (
            self.world_matrices([obj_id]).get(obj_id) if axis == "rotation" else None
        )
        # ROTATION only: add a DINO appearance term to each candidate's SELECTION
        # score — silhouette IoU is blind to a ~180-deg yaw on near-symmetric
        # objects. Selection/gain use score + LAMBDA_FEAT*sim; the stored/reported
        # objective stays unaugmented (comparable across axes and physics checks).
        feat = self._rotation_feat(obj_id) if axis == "rotation" else None
        # yaw HINT active? (the SAME band the agent's YAW HINT used — ONE
        # predicate, yaw_hint_fires, so the two can never drift). A firing hint SEEDS
        # a candidate at the hint angle. The hint no longer buys a relaxed gain bar:
        # rotation gates on the centered score (_ROT_CEN_MIN_GAIN), whose response to
        # a correct yaw is strong (audited fixes 0.098-0.52 vs the ~0.01 raw-IoU
        # hover that motivated the old _HINT_MIN_GAIN relaxation).
        yaw_hint, yaw_aniso = getattr(self, "_yaw_hint", {}).get(obj_id, (None, None))
        yaw_seed = axis == "rotation" and yaw_hint_fires(yaw_hint, yaw_aniso)
        # A fired SIZE hint relaxes the scale gate (see _HINT_MIN_GAIN): a correct
        # resize can dip raw IoU until a follow-up placement re-aligns it, so it
        # hovers at the normal bar — trust the hint and let it land in one shot.
        # (The gated gain is centered since the cutover, which shrinks that dip;
        # the relaxation is retained pending centered-era recalibration — see
        # the _HINT_MIN_GAIN comment.)
        scale_active = axis == "scale" and obj_id in getattr(self, "_scale_hint", {})
        from lib.tools.geometry.feature_metric import LAMBDA_FEAT

        def _aug(sc: dict) -> float:
            if feat is None:
                return sc["score"]
            s = feat(iou_png)
            sc["feat_sim"] = round(s, 4)
            return sc["score"] + LAMBDA_FEAT * s

        # PHASE C1 (2026-08-21) — the CENTERED metric SELECTS on the rotation axis. `iou` is
        # the term that actually moves selection, and it is not translation-invariant: rotating
        # about an object's own centre swings its extremities, so on a displaced object the IoU
        # preference between the two seeded directions tracks "which way swings me onto the
        # mask", not "which way aligns my axis". Six of ten landed rotations on static_scene_eval
        # RAISED the yaw while IoU rose in all ten; the phase-C0 shadow run then measured the
        # centred metric disagreeing with the production pick on 46/66 of the 0821 benchmark's
        # rotation rounds, correct in every case audited by eye (misc_online5 stapler,
        # airoa power plug). audits/ROTATION_SIGN_DISPLACEMENT_PROPOSAL_2026_08_03.md and
        # audits/ROTATION_HINT_AUDIT_2026_08_21.md.
        def _cen(sc: dict) -> Optional[float]:
            # None when the field is absent — a score dict not produced by ro.objective (a
            # stub, an older cached entry). Selection then falls back to `aug` for that
            # candidate: the FALLBACK must never crash a move. feat_sim is a side effect of
            # _aug, which always runs first on the same dict.
            v = sc.get("centered_iou")
            if v is None:
                return None
            if axis == "scale":
                # Swap ONLY the iou term of the objective for its translation-
                # invariant twin, KEEPING scale's axis-tuned auxiliary terms: the
                # w_size size loss (IoU is a poor SIZE signal on an imperfect
                # shape — the term was tuned FOR this axis, abc2 0717 grows an
                # under-sized spoon 1.0->~1.3) and the depth regularizer (which
                # cancels across same-translate scale candidates anyway). This is
                # the true mirror of rotation below, which keeps ITS axis-specific
                # feat term — and it is what makes the raw bar transfer exact at
                # aligned centroids (see the acceptance comment).
                return sc["score"] - sc["iou"] + v
            return v + (LAMBDA_FEAT * sc.get("feat_sim", 0.0) if feat is not None else 0.0)

        rot = axis == "rotation"
        # SCALE CUTOVER (2026-08-21): scale joins rotation on centered selection — raw
        # IoU's response to a resize on a DISPLACED object tracks "which way grows me
        # onto the mask" the same way it tracks rotation direction (the 0819-0822
        # corpora hold applied scale rounds whose raw-IoU delta is NEGATIVE while the
        # size actually improved). Same machinery as C1: per-candidate fallback to
        # ``aug`` when centered_iou is absent; the snapscale rung stays in the ladder;
        # xy/x/y/depth untouched. On scale ``_cen`` is the full objective score with
        # iou swapped for centered_iou (size/depth terms retained — see above).
        cen_sel = rot or axis == "scale"

        def _sel(sc: dict, aug: float) -> float:
            """Selection score for one candidate: the centered score on the rotation
            and scale axes (per-candidate fallback to ``aug`` when centered_iou is
            absent), ``aug`` everywhere else."""
            if not cen_sel:
                return aug
            cen = _cen(sc)
            return aug if cen is None else cen

        # rotation/scale diagnostics: the raw (aug) preference over the candidates the
        # NEW search scores — logged as the trace `selection` field (the inverse of
        # the retired phase-C0 shadow_centered log). `cen_seen` records whether any
        # candidate carried the centered field, i.e. whether the centered metric
        # actually chose.
        iou_base = iou_best = None
        sel_pick = iou_pick = None
        cen_seen = False

        best_pose, best = pose, before
        # with the feat term or centered selection the baseline comes from the freshly
        # scored "keep" candidate (same pose, with its appearance sim / centered value);
        # without either, `before` — exactly the pre-C1 behavior. It MUST track `cen_sel`,
        # not `rot`: seeding it with before["score"] on a centered-selecting axis compares
        # a centered candidate against a RAW baseline (and subtracts the two for `gain`).
        best_sel = None if (feat is not None or cen_sel) else before["score"]
        base_sel = best_sel
        keep_feat = 0.0  # the keep candidate's feat_sim (rotation held check)
        if axis == "xy":
            coarse = [dict(pose, _tag="keep")] + self._snap_candidates(obj_id, pose)
        else:
            coarse = _candidates(axis, pose, c["size"])
        if axis == "scale":
            # closed-form silhouette-measured rung (candidate #1, after "keep");
            # the blind _SCALE rungs stay, scoring/gates unchanged.
            snap = self._scale_snap_candidate(obj_id, pose)
            if snap is not None:
                coarse.insert(1, snap)
        if yaw_seed:
            # Seed the hint angle directly (the +-15 ladder rung only approximates an
            # ~18deg hint, and no rung at all reaches a 60deg one); BOTH signs, since
            # the principal-axis hint is undirected.
            #
            # The hint is measured in the IMAGE PLANE and euler[2] rotates about world
            # Z: converting between them needs the local ratio, or the seed lands short
            # (shelf, ratio 0.66) or long (marker, ratio >1.1) and the fine ladder
            # (+-2.5deg) cannot close the gap. Worse, a short landing leaves a residual
            # UNDER _YAW_HINT_MIN, so the hint stops firing and the error ships.
            seed_deg = yaw_hint
            j = self._yaw_jacobian(obj_id, pose)
            if j is not None and abs(j) >= _YAW_JACOBIAN_MIN:
                seed_deg = min(yaw_hint / abs(j), _YAW_SEED_MAX_DEG)
            for sgn in (1, -1):
                e = list(pose["euler"])
                e[2] = e[2] + sgn * math.radians(seed_deg)
                coarse.insert(
                    1, {**pose, "euler": e, "_tag": f"yawhint{sgn:+d}x{seed_deg:.0f}"}
                )
        for cand in coarse:
            sc = _score_pose(
                self.client, name, cand, mask, c["center_y"], c["depth"], iou_png
            )
            aug = _aug(sc)
            sel = _sel(sc, aug)
            if cand.get("_tag") == "keep" and base_sel is None:
                base_sel = sel
                # the keep candidate's live feat reading — the held check below
                # needs it to detect a worker that died between this loop and
                # the settle (its rested re-score would read feat 0.0).
                keep_feat = sc.get("feat_sim", 0.0)
            if best_sel is None or sel > best_sel + 1e-6:
                best_pose = {k: cand[k] for k in ("translate", "euler", "scale")}
                best, best_sel = sc, sel
                sel_pick = cand.get("_tag")
            if cen_sel:
                cen_seen = cen_seen or sc.get("centered_iou") is not None
                if cand.get("_tag") == "keep" and iou_base is None:
                    iou_base = aug
                if iou_best is None or aug > iou_best + 1e-6:
                    iou_best, iou_pick = aug, cand.get("_tag")
        if base_sel is None:  # defensive: keep is always first, but stay safe
            base_sel = _sel(before, before["score"])
        # stage 2: refine between the coarse rungs around the winner — for rotation
        # and scale that is the CENTERED winner (best_pose tracks the selection metric).
        fine = (
            _xy_ring(best_pose, c["size"])
            if axis == "xy"
            else _fine_candidates(axis, best_pose, c["size"])
        )
        for cand in fine:
            sc = _score_pose(
                self.client, name, cand, mask, c["center_y"], c["depth"], iou_png
            )
            aug = _aug(sc)
            sel = _sel(sc, aug)
            if sel > best_sel + 1e-6:
                best_pose = {k: cand[k] for k in ("translate", "euler", "scale")}
                best, best_sel = sc, sel
                sel_pick = cand.get("_tag")
            if cen_sel:
                cen_seen = cen_seen or sc.get("centered_iou") is not None
                if iou_best is None or aug > iou_best + 1e-6:
                    iou_best, iou_pick = aug, cand.get("_tag")
        gain = best_sel - base_sel
        # rotation gates on its own centered bar (see _ROT_CEN_MIN_GAIN); the other axes
        # keep the raw-IoU bars — a fired SIZE hint still relaxes scale.
        #
        # SCALE BAR TRANSFER (2026-08-21): scale now SELECTS on the centered metric but
        # keeps the raw-calibrated numbers, unlike rotation. The rotation recalibration was
        # needed because its raw gains hover at the noise floor (a correct yaw barely moves
        # raw IoU), so a raw-tuned bar sat inside the centered signal band. Scale's
        # transfer is EXACT by construction, not by analogy: the gated gain has the same
        # composition as the recorded raw-score gains — iou swapped for centered_iou with
        # the w_size/depth terms retained (see _cen) — and centered_iou equals raw iou at
        # aligned centroids, where scale rounds run (post-xy; final centered/raw IoU ratio
        # median 1.03 over the 761 scale-round objects logging both in the 0819-0822
        # corpora). So every previously-applied aligned round gates on the SAME number it
        # cleared before, and those gains sit far above the bar (901 applied / 224 dead
        # scale rounds: applied gains median 0.124, p10 0.035, only 7 of 901 inside
        # [0.005, 0.01) and NONE below 0.005 — b3_scale_bar_check.py). Displaced rounds
        # gate on a gain that differs only through the centered_iou term (typically >=
        # raw there — the intended rescue direction). NOTE the earlier bare-centered
        # draft of this cutover did NOT
        # transfer: dropping the w_size term left a bare-IoU gain whose aligned-case
        # proxy put 16% of previously-applied rounds under the 0.01 bar
        # (rev3_diou_check.py) — the size term is load-bearing for the bar as well as
        # for selection. The truly-centered gain distribution on scale is still
        # unmeasured pre-cutover; the selection log below makes it a one-benchmark-day
        # recalibration if the marginal band ever needs a scale-specific bar.
        applied = gain >= (
            _ROT_CEN_MIN_GAIN
            if rot
            else (_HINT_MIN_GAIN if scale_active else min_gain)
        )
        # YAW GATE (2026-08-21, extends the 08-03 seeded-only regression gate). Whenever a
        # PAIRED reliable yaw reading exists — pre-yaw at the pre-move pose with
        # aniso >= _YAW_ANISO_MIN, post-yaw re-measured at the winner — gate seeded AND
        # unseeded rotations: the 08-19 UNKNOWN-routing prompt sends agents to unseeded
        # rotations exactly where the old gate was blind. Strengthened from non-regression
        # to IMPROVEMENT for an actionable pre (>= _YAW_HINT_MIN): refuse unless
        # post <= pre - _YAW_IMPROVE_MIN, or post <= _YAW_DONE_DEG, or the NEAR-MISS
        # clause fires (down by >= _YAW_NEAR_MISS_IMPROVE AND under _YAW_NEAR_MISS_DEG —
        # see the constants). The stapler round-13 pick re-measured within +3 deg of
        # pre 17 and slipped through the old tolerance.
        # A sub-band pre keeps the old noise tolerance (_YAW_REGRESS_TOL). No reliable
        # paired reading -> no gate: can't gate blind.
        #
        # Measured through _object_sil — the SAME silhouette source scale_hint derived
        # yaw_hint from — so pre and post are apples-to-apples. The iou_png already on disk
        # from the scoring loop is cheaper but is rendered under a different isolation set and
        # is NOT comparable. Costs 1 render, +1 more only for an unseeded rotation with no
        # cached pre reading, against the 11-13 a rotation already spends.
        yaw_regressed = None
        # P3: the pair the gate measures doubles as the applied-rotation
        # feedback's "measured yaw pre -> post" evidence — hoisted (with the post
        # ANISO the gate itself never needed, free off the same _yaw_discrepancy
        # call) so the return below can stash it instead of re-rendering a
        # display number the gate already computed.
        pre_yaw = pre_aniso = post_yaw = post_aniso = None
        if applied and rot:
            # pre-yaw: the cached scale_hint reading was measured at the CURRENT pose (a
            # landed reorientation clears it — _invalidate_hints — and translations/resizes
            # don't change the yaw), so reuse it; render only when nothing is cached.
            pre_yaw, pre_aniso = yaw_hint, yaw_aniso
            try:
                # MASK PARITY: measure against _gt_mask_full, NOT the raw `mask` this method
                # loaded. scale_hint derived yaw_hint from _gt_mask_full (resized to
                # gt_h x gt_w) and _object_sil renders at that same resolution, whereas the
                # raw np.load is NOT resized and the stored mask is not always already that
                # shape (0717_e2e8_abc1 placemat: mask 864x769 vs render 768x681). The
                # mismatch would be SILENT — _axis_angle reads each silhouette's OWN principal
                # axis, so mismatched shapes do not raise, they return a wrong angle.
                gt = self._gt_mask_full(obj_id)
                if pre_yaw is None:
                    pre_yaw, pre_aniso = _yaw_discrepancy(
                        self._object_sil(obj_id, pose), gt
                    )
                if pre_yaw is not None and (pre_aniso or 0.0) >= _YAW_ANISO_MIN:
                    post_yaw, post_aniso = _yaw_discrepancy(
                        self._object_sil(obj_id, best_pose), gt
                    )
                # Restore the scoring isolation once the renders are done: _object_sil
                # leaves the PLAIN iou isolation behind, but a scoring caller set up
                # visible_score (ancestors + the redetect-aware occluder holdout). Miss it and
                # every later render in this call loses the holes the photo mask has. Same
                # re-restore _yaw_jacobian already has to do.
                _isolate_iou(self.client, c["visible_score"])
            except Exception as exc:  # noqa: BLE001 - guarded like every silhouette probe here
                # Best-effort for the same reason _yaw_jacobian and scale_hint are: this runs
                # AFTER the winner was applied to the blend, so an unhandled raise would abort
                # mid-commit and leave a pose the trace never records. Failing open is also the
                # safe direction — no measurement means no rejection, exactly as a None angle.
                print(f"[yaw-regress] {obj_id}: {exc}", file=sys.stderr)
            if post_yaw is not None:
                if pre_yaw >= _YAW_HINT_MIN:
                    ok = (
                        post_yaw <= pre_yaw - _YAW_IMPROVE_MIN
                        or post_yaw <= _YAW_DONE_DEG
                        or (
                            post_yaw <= pre_yaw - _YAW_NEAR_MISS_IMPROVE
                            and post_yaw < _YAW_NEAR_MISS_DEG
                        )
                    )
                else:
                    ok = post_yaw <= pre_yaw + _YAW_REGRESS_TOL
                if not ok:
                    applied = False
                    yaw_regressed = [round(float(pre_yaw), 1), round(float(post_yaw), 1)]
        # PRE-PHYSICS FEASIBILITY CLAMP (translation/rotation): reduce the winner move to
        # the largest fraction that keeps the carried compound out of penetration. Reject
        # if it's boxed in (< FEASIBLE_MIN_FRAC of the move reachable) or the reachable
        # part no longer clears the gain bar. Scale is excluded (a neighbor-blocked resize
        # must not silently shrink to dodge it) — instead its drop-based commit is gated
        # POST-settle on penetration below. The clamp complements rather than replaces
        # that gate; a settle can still end wedged when lift-to-clear bails against a wall.
        # Physics reads the winner's pose off the LIVE blend (max_feasible and
        # commit_move both call _blend_matrices), but the search loop left the blend
        # holding the LAST fine candidate — always an offset from best_pose, never the
        # winner itself. Apply the winner now so BOTH the feasibility clamp and the
        # settle evaluate the pose we actually intend to commit (post set_pose resolve).
        if applied and self.physics is not None:
            _score_pose(
                self.client, name, best_pose, mask, c["center_y"], c["depth"], iou_png
            )
        clamp_reason = None
        clamped_frac = None  # set when a PARTIAL winner (fraction f) was kept
        physics_failure = None
        if applied and self.physics is not None and axis != "scale":
            from lib.tools.geometry.composition_physics import FEASIBLE_MIN_FRAC

            try:
                f = self.physics.max_feasible(self, name, axis)
            except Exception as exc:  # noqa: BLE001 - strict harness rejects atomically
                if not strict_physics:
                    raise
                recovery = self.physics.recover_after_error(self)
                physics_failure = (
                    f"physics feasibility check failed: {exc}"
                    + (f"; recovery warnings: {'; '.join(recovery)}" if recovery else "")
                )
                applied = False
                f = 0.0
            if f < FEASIBLE_MIN_FRAC:
                if physics_failure is not None:
                    clamp_reason = physics_failure
                else:
                    applied, clamp_reason = False, (
                        f"boxed in — {name} can only move ~{f * 100:.0f}% toward the "
                        "target before wedging into a neighbor; move the neighbor or leave it"
                    )
            elif f < 1.0 - 1e-6:
                best_pose = _lerp_pose(pose, best_pose, f)
                sc = _score_pose(
                    self.client, name, best_pose, mask, c["center_y"], c["depth"], iou_png
                )
                sel_c = _sel(sc, _aug(sc))
                if sel_c - base_sel < (_ROT_CEN_MIN_GAIN if rot else min_gain):
                    applied, clamp_reason = False, (
                        f"the reachable part of the move (~{f * 100:.0f}% before a "
                        "neighbor blocks it) no longer improves the match"
                    )
                else:
                    best, best_sel = sc, sel_c
                    clamped_frac = f
        physics = (
            {
                "accepted": False,
                "infrastructure_error": True,
                "reason": physics_failure,
            }
            if physics_failure is not None
            else None
        )
        commit = None
        if applied and strict_physics and self.physics is None:
            applied = False
            physics = {
                "accepted": False,
                "infrastructure_error": True,
                "reason": "composition physics authority is unavailable",
            }
        winner_pose_rec = winner_world_t = rested_world_t = None
        # the applied delta as of the accept — a physics ACCEPT zero-rebases
        # best_pose/self.pose below, which made the return-time range_limited
        # compare 0 vs 0 and structurally never fire on a physics-accepted move
        # (misc_online5 stapler: a -42.5deg ladder-edge landing never got the
        # "repeat this SAME aspect" nudge). Snapshot the winner before the rebase.
        moved_pose = None
        if applied:
            # trace diagnostics (2026-07-21): the attempted winner — recorded even
            # when physics later rejects it (a rejected round's target pose was
            # previously unrecoverable from artifacts: the real8219 plush round-1).
            winner_pose_rec = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                               for k, v in best_pose.items()}  # fmt: skip
            winner_world_t = self._world_t(name)
        if applied and self.physics is not None:
            # winner already applied to the blend above (before the feasibility clamp);
            # the f<1.0 lerp branch re-applied its clamped pose, so the blend is current.
            from lib.tools.geometry.composition_physics import TILT_CAP_DEG

            # STABILITY CLAMP (2026-07-21): a winner whose settle CAPSIZES is retried
            # at a smaller fraction of the move (mirrors the pre-physics feasibility
            # clamp) so the agent gets partial progress toward the photo instead of a
            # flat reject. Capsize-only — pen/score failures keep single-shot
            # semantics; scale keeps its own drop-based gate (no fractions).
            # Ladder trimmed [1.0, 0.75, 0.5, 0.25] -> [1.0, 0.5] (2026-07-24):
            # 55 clamp-era runs measured 4 rescues vs 37 exhausted ladders, and
            # 3 of the 4 rescues landed at >=0.5 — most capsizes are 90-170 deg
            # flips where the instability is at the destination, so shorter
            # fractions can't help; each rung costs a full deps+settle commit.
            winner_full = dict(best_pose)
            fractions = [1.0] + ([0.5] if axis != "scale" else [])
            # freed-dependency cache shared across the rungs: every failed rung
            # ends in an exact two-sided reject, so each rung's _contact_deps
            # query sees identical pre-move state — compute once, reuse.
            deps_cache: dict = {}
            for f in fractions:
                if f < 1.0:
                    best_pose = _lerp_pose(pose, winner_full, f)
                    sc = _score_pose(
                        self.client, name, best_pose, mask, c["center_y"],
                        c["depth"], iou_png,
                    )  # fmt: skip
                    if _sel(sc, _aug(sc)) - base_sel < (
                        _ROT_CEN_MIN_GAIN if rot else min_gain
                    ):
                        # the standable fraction no longer clears the gain bar:
                        # the whole move dies (the full-move capsize reason from
                        # the previous iteration would be misleading here)
                        physics = {
                            "accepted": False,
                            "reason": (
                                f"{name} tips over at the full move, and the "
                                "smaller fraction that might stand no longer "
                                "improves the match"
                            ),
                        }
                        applied = False
                        break
                try:
                    commit = self.physics.commit_move(
                        self, name, deps_cache=deps_cache
                    )
                except Exception as exc:  # noqa: BLE001 - a dead Isaac must not kill
                    import sys as _sys

                    # Baseline retains the historical kinematic fallback. GPT-6 strict
                    # mode rejects the move and restores the last accepted snapshot.
                    if strict_physics:
                        recovery = self.physics.recover_after_error(self)
                        physics = {
                            "accepted": False,
                            "infrastructure_error": True,
                            "reason": (
                                f"physics settlement failed: {exc}"
                                + (
                                    f"; recovery warnings: {'; '.join(recovery)}"
                                    if recovery
                                    else ""
                                )
                            ),
                        }
                        applied = False
                    else:
                        # stderr: inside the exec MCP server stdout is the RPC transport
                        print(f"[composition] physics disabled after error: {exc}",
                              file=_sys.stderr)  # fmt: skip
                        self.physics = None
                    commit = None
                    break
                rested_world_t = self._world_t(name)
                tilt = max(commit["cum_tilt_deg"].values(), default=0.0)
                # the carry decision's mean_iou() re-isolated per MEMBER — restore
                # OUR isolation before scoring the rested pose, or the moved object
                # renders invisible and "rested WORSE" fires spuriously
                # (0715_fix3_abc1: both stay-decisions rejected this way)
                _isolate_iou(self.client, c["visible_score"])
                rested = _score_pose(
                    self.client, name, best_pose, mask, c["center_y"], c["depth"],
                    iou_png, apply=False,
                )  # fmt: skip
                # don't-lose-ground, NOT a second min_gain: selection already charged
                # the full anti-churn margin, so re-charging it here rejected marginal
                # winners for ordinary mm-scale seating drift (0715_hint_real8226
                # mouse: gain +0.013, settle tilt 1.2 deg, rejected for ~0.003 drift).
                # caps: per-member class-aware stability (rollable -> displacement,
                # else cum tilt); replaces the old blanket max-tilt check, which read
                # a lying mic's benign roll as a 118-180 deg capsize.
                caps = commit.get("capsized")
                if caps is None:  # kinematic/legacy commit without the field
                    caps = (
                        [{"name": name, "tilt_deg": round(tilt, 1)}]
                        if tilt > TILT_CAP_DEG
                        else []
                    )
                # Penetration gate (ALL axes). SCALE has no pre-physics clamp; the other
                # axes start contact-clear via the clamp, but the settle can still END
                # wedged when lift-to-clear bails — an xy-overlap with a WALL static can't
                # be lifted away, so the compound is released in place for PhysX
                # depenetration, which may not finish within the micro budget. _pen_worst is
                # delta-based (worsened > PEN_NEW_MM), so a move that doesn't deepen a
                # pre-existing overlap still passes.
                pen = commit.get("penetration")
                converged = bool(commit.get("converged", True))
                # ... and it runs on the SELECTION metric wherever selection does
                # (rotation AND scale — `cen_sel`), where a correct winner can be
                # raw-IoU-neutral or even raw-negative (flagship C1 rounds: centered
                # gain 0.08-0.37 at raw gain ~0.0) — a raw comparison would ACCEPT on
                # one metric and then re-veto on another, killing exactly the fixes
                # centered selection exists to land the moment the settle drifts the
                # raw score below the pre-move value. xy/x/y select on aug, so they
                # keep the raw comparison. base_sel is the keep candidate's selection
                # score.
                held_bar = base_sel
                if cen_sel:
                    rested_sel = _sel(rested, _aug(rested))
                    # ROTATION cross-state guard: base_sel was computed with a
                    # LIVE feat term during the candidate loop, but _aug scores
                    # a worker that died (or a crop failure / fully occluded
                    # render) before the settle as feat 0.0 — a "no measurement"
                    # sentinel, not a similarity. Comparing that deflated rested
                    # score against a feat-carrying bar spuriously rejects a
                    # physics-good rotation as "rested WORSE than before the
                    # move"; drop the keep candidate's feat term so both sides
                    # compare feat-free.
                    if feat is not None and not rested.get("feat_sim"):
                        held_bar = base_sel - LAMBDA_FEAT * keep_feat
                held = (
                    rested_sel >= held_bar
                    if cen_sel
                    else rested["score"] >= before["score"]
                )
                ok = (
                    not caps
                    and pen is None
                    and held
                    and (converged or not strict_physics)
                )
                physics = {
                    "members": commit["members"],
                    "cum_tilt_deg": round(tilt, 1),
                    "lift_mm": round(commit["lift_mm"], 1),
                    "accepted": ok,
                    **({"converged": converged} if strict_physics else {}),
                    **({"capsized": caps} if caps else {}),
                    **({"penetration": pen} if pen else {}),
                    **({"stability_clamped_frac": f} if ok and f < 1.0 else {}),
                    **(
                        {"followed": commit["followed"],
                         "carry_decision": commit["carry_decision"]}  # fmt: skip
                        if "followed" in commit
                        else {}
                    ),
                }
                if ok:
                    self.physics.accept(self, commit)
                    # rebase: every carried member's pose deltas restart at zero from
                    # the rested pose (the set_pose base was just re-committed).
                    zero = {"translate": [0.0, 0.0, 0.0], "euler": [0.0, 0.0, 0.0],
                            "scale": 1.0}  # fmt: skip
                    mesh2oid = {v["mesh_name"]: k for k, v in self.ctx.items()}
                    # members that FOLLOWED a reorientation turned with the parent:
                    # their yaw/orient/scale/appearance channels were measured at
                    # the pre-carry pose and are stale exactly like the moved
                    # object's (the moved object gets its own _invalidate_hints
                    # below). No-op off the reorienting axes (_HINT_STALE) and when
                    # the carry decision left the children in place (followed False
                    # — a solo-probe commit whose kids only settled as free bodies).
                    carried_turned = commit.get("followed", True)
                    for m in commit["members"]:
                        oid = mesh2oid.get(m)
                        if oid is not None:
                            self.pose[oid] = dict(zero)
                            self.cur[oid] = None
                            if oid != obj_id and carried_turned:
                                self._invalidate_hints(oid, axis)
                    moved_pose = best_pose  # pre-rebase winner (range_limited)
                    best_pose, best = dict(zero), rested
                    break
                self.physics.reject(self, commit)
                if caps and not pen and f != fractions[-1]:
                    continue  # capsized: try a smaller fraction of the move
                if strict_physics and not converged:
                    physics["reason"] = "physics settlement did not converge"
                elif caps:
                    c0 = caps[0]
                    physics["reason"] = (
                        f"{c0['name']} rolled/slid ~{c0['disp_mm']:.0f}mm away "
                        f"from where the move put it"
                        if c0.get("rollable")
                        else f"{c0['name']} cannot rest at the target pose — it "
                        f"tips over (ends {c0.get('tilt_deg', tilt):.0f} deg "
                        f"from upright)"
                    )
                    if len(fractions) > 1:
                        physics["reason"] += (
                            f" (tried down to {fractions[-1] * 100:.0f}% of the "
                            "move — every fraction tips)"
                        )
                elif pen:
                    physics["reason"] = (
                        f"{name} ends ~{pen['after_mm']:.0f}mm wedged inside "
                        f"'{pen['other']}' after the settle"
                    )
                else:
                    physics["reason"] = (
                        "the rested pose scored WORSE than before the move"
                    )
                applied = False
                break
            # (no for-else: every path breaks — accept, terminal reject, gain-bar
            # death, or the Isaac-died kinematic fallback, which leaves `applied`
            # untouched so the blend-authoritative pose still lands.)
        if applied:
            self.pose[obj_id], self.cur[obj_id] = best_pose, best
            self._invalidate_hints(obj_id, axis)
        # leave the server holding the (possibly unchanged) accepted pose — except
        # after a physics accept, where set_pose would re-run the BVH resolve/rest
        # heuristics on a pose that PhysX already owns.
        if physics is None or not physics["accepted"]:
            if self.physics is not None:
                # A move that did NOT stick with an active physics session (physics
                # REJECT, feasibility-clamp reject, or gain-fail) must leave the scene
                # EXACTLY as it was. A kinematic set_pose "restore" is NOT identity: it
                # re-runs _resolve_penetration (which separates only from the object's
                # SUPPORT, never a lateral sibling) + the all-scene _seat_z (whose
                # deadband makes a rested pose a no-op, but a tilted rest expressed as
                # (t,euler,s) still isn't reachable), and it swaps a physics-rested
                # attitude for a kinematic one (0717 knife-into-spoon after a rejected
                # rotation). reject() already restored Isaac + any carried members;
                # restore the blend handle to its exact synced (pre-move) matrix.
                self.physics.restore_blend(self, name)
            else:
                # kinematic session (no Isaac): set_pose both COMMITS an accepted
                # kinematic move and restores a rejected one — the only mechanism here.
                _score_pose(
                    self.client,
                    name,
                    self.pose[obj_id],
                    mask,
                    c["center_y"],
                    c["depth"],
                    iou_png,
                )
        # P1 applied-rotation capture (bridge_6 / misc_online5): the appearance
        # pair is the reliable pose-comparison instrument where raw IoU dips and
        # a weak-axis yaw pair aliases across a CORRECT landed yaw fix. Two
        # appearance_agreement calls on a LANDED rotation only (vs the 11-13
        # renders the search spent, and zero cost on every other axis): pre at
        # the snapshotted pre-move world matrix, post at the final (rested)
        # pose — ALSO as a world matrix, so the cache entry lives in the
        # absolute "m:" namespace: a zero-delta "p:" digest written here would
        # name a DIFFERENT world pose after any later physics rebase (which by
        # _HINT_STALE design invalidates nothing for translations), and on the
        # kinematic lane the world matrix carries the commit's resolve/seat
        # offset the deltas don't. Falls back to the delta pose only when the
        # snapshot RPC hiccups (best-effort, one extra get_matrix). AFTER
        # _invalidate_hints so the fresh values survive in the cache; a dead
        # feature worker just yields None-None.
        app_pre = app_post = None
        if applied and rot:
            if pre_world_m is not None:
                app_pre = self.appearance_agreement(obj_id, pre_world_m)
            post_world_m = self.world_matrices([obj_id]).get(obj_id)
            app_post = self.appearance_agreement(
                obj_id,
                post_world_m if post_world_m is not None else self.pose[obj_id],
            )
        self.rounds += 1
        after = self.cur[obj_id] if applied else before
        self.trace[obj_id].append(
            {
                "round": self.rounds,
                "axis": axis,
                "score": round(after["score"], 4),
                "iou": round(after["iou"], 4),
                "gain": round(max(gain, 0.0), 4),
                **({} if applied else {"dead": True}),
                # T3b (2026-07-23): boxed-in / clamped-gain rejects were bare
                # `dead` rows — indistinguishable from no-gain searches in
                # artifacts, and the reason never reached the agent either
                **({"clamp_reason": clamp_reason} if clamp_reason else {}),
                # a rotation refused by the yaw-regression gate is NOT a no-gain search —
                # it FOUND gain and was vetoed. Recorded as [pre, post] degrees so the
                # band calibration can finally be recomputed against yaw improvement
                # instead of "applied" (see the audit).
                **({"yaw_regressed": yaw_regressed} if yaw_regressed else {}),
                # PHASE C1 selection log (replaces the C0 shadow_centered field, which
                # nothing at runtime read): the centered metric SELECTS on rotation AND
                # scale (`cen_sel`), and BOTH log it — a scale round that selects on one
                # metric while logging nothing is blind to the next calibration sweep,
                # which is how the raw-IoU bars went unaudited for so long.
                # `iou_pick`/`iou_gain` record the raw-IoU (aug) preference over the
                # NEW candidate set (coarse rungs are shared, but fine rungs follow
                # the CENTERED winner), so this is not the pre-C1 counterfactual and
                # is not apples-to-apples with the C0 shadow logs. On a seeded rotation a
                # disagreement between the yawhint+1x<deg>/yawhint-1x<deg> tags is
                # literally the sign flip; on scale a snapscale-vs-blind-rung
                # disagreement is the closed-form estimate winning on centered alone.
                # Rejected/dead rounds keep recording it.
                # metric == "iou" only when no candidate carried centered_iou (stub
                # score dicts) and selection degraded to aug wholesale.
                **(
                    {
                        "selection": {
                            "metric": "centered" if cen_seen else "iou",
                            "pick": sel_pick,
                            "gain": round(gain, 4),
                            "iou_pick": iou_pick,
                            "iou_gain": (
                                round(iou_best - iou_base, 4)
                                if iou_best is not None and iou_base is not None
                                else None
                            ),
                            "agrees": sel_pick == iou_pick,
                        }
                    }
                    if (cen_sel and sel_pick is not None)
                    else {}
                ),
                **({"physics": physics} if physics else {}),
                # diagnostics: the attempted winner + world poses — recorded for
                # REJECTED rounds too (previously unrecoverable from artifacts)
                **({"winner_pose": winner_pose_rec} if winner_pose_rec else {}),
                **({"winner_world_t": winner_world_t} if winner_world_t else {}),
                **({"rested_world_t": rested_world_t} if rested_world_t else {}),
            }
        )
        self._dump()
        # a physics accept rebased best_pose to zero above; the ladder-edge test
        # must read the snapshotted winner delta instead (still vs the pre-move
        # `pose`, which shares the same base).
        mv = best_pose if moved_pose is None else moved_pose
        # P2/P3: the yaw pair is surfaced ONLY when the landed pose is the very
        # winner the gate measured (a clamped fraction lands a DIFFERENT yaw —
        # the gate's post reading would misdescribe it) AND both readings are
        # reliable (aniso >= _YAW_ANISO_MIN on each side). Weak on either side
        # -> both None: edit-feedback surfaces omit weak yaw numbers SILENTLY
        # (bridge_6: the aniso-1.09 pair aliased 28 -> 50 across a correct fix,
        # then 55.8 -> 12.7 across a scale-only move). ONE predicate here so no
        # frontend can drift.
        yaw_pair = (
            [round(float(pre_yaw), 1), round(float(post_yaw), 1)]
            if (
                applied
                and post_yaw is not None
                and clamped_frac is None
                and not (physics or {}).get("stability_clamped_frac")
                and (pre_aniso or 0.0) >= _YAW_ANISO_MIN
                and (post_aniso or 0.0) >= _YAW_ANISO_MIN
            )
            else None
        )
        return {
            "applied": applied,
            "axis": axis,
            "before": before,
            "after": after,
            "pose": dict(self.pose[obj_id]),
            "physics": physics,
            "clamp_reason": clamp_reason,
            "clamped_frac": clamped_frac,
            "yaw_regressed": yaw_regressed,
            # P1/P3 edit-feedback evidence on an APPLIED 'rotation' move (None on
            # every other axis and on unapplied moves): appearance agreement vs
            # the photo at the pre-move pose / the final rested pose, and the
            # gate-measured yaw pair under the both-reliable rule above, with
            # the anisos it was measured at (exec re-checks them defensively —
            # absent anisos read as 0.0 there and would mute a reliable pair).
            "app_pre": app_pre,
            "app_post": app_post,
            "yaw_pre": yaw_pair[0] if yaw_pair else None,
            "yaw_post": yaw_pair[1] if yaw_pair else None,
            "yaw_pre_aniso": round(float(pre_aniso), 2) if yaw_pair else None,
            "yaw_post_aniso": round(float(post_aniso), 2) if yaw_pair else None,
            # winner landed at the coarse ladder's outermost rung — the per-call
            # search is bounded, so the optimum may lie beyond; a SAME-aspect
            # repeat searches onward from the new pose. xy is excluded (snap
            # candidates solve the offset directly, not a bounded ladder).
            "range_limited": bool(
                applied and clamped_frac is None and (
                    (_AXES[axis][0] == "t" and axis != "xy" and abs(
                        mv["translate"][_AXES[axis][1]]
                        - pose["translate"][_AXES[axis][1]]
                    ) >= 0.9 * max(_TRANS) * c["size"])
                    or (_AXES[axis][0] == "r" and abs(
                        mv["euler"][_AXES[axis][1]]
                        - pose["euler"][_AXES[axis][1]]
                    ) >= 0.9 * max(_ROT))
                    or (axis == "scale"
                        and not (min(_SCALE) * 1.05 < mv["scale"] < max(_SCALE) * 0.95))
                )
            ),
        }

    def _world_t(self, mesh_name: str):
        """Blend world translation of a handle — trace diagnostics only,
        best-effort (a get_matrix hiccup must never fail a move)."""
        try:
            m = self.client.rpc({"cmd": "get_matrix", "names": [mesh_name]})[
                "matrices"
            ][mesh_name]
            return [round(float(m[i][3]), 4) for i in range(3)]
        except Exception:  # noqa: BLE001
            return None

    def world_matrices(self, oids: Optional[list] = None) -> dict:
        """World matrix (4x4 row-major lists) per scene-graph id off the live
        blend — the snapshot appearance_agreement can score AFTER a settle
        rebuild / physics rebase makes the pose unreachable as a delta (exec
        captures it before a freeform edit; optimize_axis captures the pre-move
        pose with it). Best-effort like _world_yaws: {} on any failure, unknown
        handles skipped — a snapshot hiccup only costs the advisory pair."""
        try:
            ids = list(oids) if oids is not None else list(self.ctx)
            names = {o: self.ctx[o]["mesh_name"] for o in ids if o in self.ctx}
            ms = self.client.rpc(
                {"cmd": "get_matrix", "names": list(names.values())}
            )["matrices"]
            return {o: ms[n] for o, n in names.items() if n in ms}
        except Exception:  # noqa: BLE001 - snapshot is advisory
            return {}

    def _descendant_meshes(self, mesh_name: str) -> list[str]:
        """Transitive scene-graph children of a blend handle (mesh-name dialect)."""
        kids: dict[str, list[str]] = {}
        for c, p in self.parents.items():
            kids.setdefault(p, []).append(c)
        out, stack = [], list(kids.get(mesh_name, []))
        while stack:
            n = stack.pop(0)
            out.append(n)
            stack.extend(kids.get(n, []))
        return out

    def flip_180(self, obj_id: str) -> dict:
        """One-shot SEMANTIC 180-degree yaw about the object's center, applied
        regardless of the match score — a near-symmetric object's silhouette cannot
        distinguish front from back, so the caller's visual judgment decides, not
        IoU. Children deliberately stay in place (SAM3D orientation errors are the
        parent's problem, not the stack's). With physics attached the flip is
        settled solo (descendants excluded from the sim) and REVERTED if the rested
        pose tilts past FLIP_TILT_CAP_DEG — a flip that cannot stand is wrong.

        Geometry: with Blender's XYZ euler order, euler.z += pi is exactly a
        world-frame yaw about the object origin even on a tilted base pose:
        Rz(ez+pi)@Ry@Rx == Rz(pi)@[Rz(ez)@Ry@Rx]."""
        import math as _math

        strict_physics = (
            getattr(self, "harness_profile", "baseline") == "gpt6_v1"
            and bool(getattr(self, "strict_post_edit_physics", False))
        )
        c = self.ctx[obj_id]
        name = c["mesh_name"]
        mask = np.load(c["mask_path"])
        iou_png = str(self.work / f"{name}_iou.png")
        _isolate_iou(self.client, c["visible_score"])
        before = self.score(obj_id)
        pose = self.pose[obj_id]
        flipped = {
            "translate": list(pose["translate"]),
            "euler": [pose["euler"][0], pose["euler"][1],
                      pose["euler"][2] + _math.pi],  # fmt: skip
            "scale": pose["scale"],
        }
        after = _score_pose(
            self.client, name, flipped, mask, c["center_y"], c["depth"], iou_png
        )
        applied, physics, reason = True, None, None
        if strict_physics and self.physics is None:
            applied = False
            reason = "composition physics authority is unavailable"
            physics = {
                "accepted": False,
                "infrastructure_error": True,
                "failure_kind": "infrastructure_error",
                # The tentative yaw only exists in the register-server scene.
                # The common non-applied tail below restores it before this
                # result can be returned; an exception there is handled from the
                # executor's durable pre-flip snapshot instead.
                "recovery_succeeded": True,
                "recovery_errors": [],
                "reason": reason,
            }
        if self.physics is not None:
            from lib.tools.geometry.composition_physics import (
                FLIP_TILT_CAP_DEG,
                TILT_CAP_DEG,
            )

            try:
                commit = self.physics.commit_move(self, name, carry_children=False)
            except Exception as exc:  # noqa: BLE001 - same fallback as optimize_axis
                import sys as _sys

                if strict_physics:
                    try:
                        recovery = self.physics.recover_after_error(self)
                    except Exception as recovery_exc:  # noqa: BLE001
                        recovery = [f"physics recovery failed: {recovery_exc}"]
                    reason = (
                        f"physics settlement failed: {exc}"
                        + (
                            f"; recovery warnings: {'; '.join(recovery)}"
                            if recovery
                            else ""
                        )
                    )
                    physics = {
                        "accepted": False,
                        "infrastructure_error": True,
                        "failure_kind": "infrastructure_error",
                        "recovery_succeeded": not recovery,
                        "recovery_errors": list(recovery),
                        "reason": reason,
                    }
                    applied = False
                else:
                    print(f"[composition] physics disabled after error: {exc}",
                          file=_sys.stderr)  # fmt: skip
                    self.physics = None
                commit = None
            if commit is not None:
                tilt = float(commit.get("tilt_deg", 0.0))
                rested = _score_pose(
                    self.client, name, flipped, mask, c["center_y"], c["depth"],
                    iou_png, apply=False,
                )  # fmt: skip
                pen = commit.get("penetration")
                # a ROLLABLE object (lying cylinder) may roll during the flip
                # settle — tilt is meaningless for it; gate on displacement
                roll_p = bool((commit.get("rollable") or {}).get(name))
                disp_p = float((commit.get("disp_mm") or {}).get(name, 0.0))
                from lib.tools.geometry.composition_physics import (
                    FLAT_DISP_CAP_MM,
                )

                target_ok = (
                    disp_p <= FLAT_DISP_CAP_MM
                    if roll_p
                    else tilt <= FLIP_TILT_CAP_DEG
                )
                # neighbours: the commit's class-aware per-member list — welded
                # members only, freed contact-deps EXCLUDED by design ("a leaner
                # falling once its support moved is the expected reaction, not a
                # failure of this move"). The old blanket max over cum_tilt_deg
                # re-included them and let a bystander that tumbles on EVERY
                # settle veto a clean flip with a bogus reason (0721_perffix5
                # gpt1: pen at 54.4 killed the notebook flip that rested at 0.0;
                # a freed dep ending WEDGED still rejects via the penetration
                # gate, which covers every settled body). Legacy commits without
                # the field keep the old cum check.
                caps = commit.get("capsized")
                if caps is None:
                    # legacy/kinematic commit: old blanket max-cum check, but
                    # attributed to the worst body (incl. the target)
                    tilts = commit["cum_tilt_deg"]
                    worst = max(tilts, key=tilts.get, default=None)
                    cum = float(tilts.get(worst, 0.0)) if worst else 0.0
                    caps = (
                        [{"name": worst, "rollable": False,
                          "tilt_deg": round(cum, 1)}]  # fmt: skip
                        if cum > TILT_CAP_DEG
                        else []
                    )
                # caps covers the WELDED members (for a flip: the target alone —
                # its lifetime-cum 45-deg cap, complementing target_ok's
                # this-settle 35-deg cap, exactly the old tilt<=35 AND cum<=45)
                converged = bool(commit.get("converged", True))
                ok = (
                    target_ok
                    and not caps
                    and pen is None
                    and (converged or not strict_physics)
                )
                physics = {
                    "members": commit["members"],
                    "tilt_deg": round(tilt, 1),
                    # per-body (vs the composition-input pose) — the collapsed
                    # max made round-16's 54.4 unattributable in the artifacts
                    "cum_tilt_deg": {
                        m: round(float(v), 1)
                        for m, v in commit["cum_tilt_deg"].items()
                    },
                    "lift_mm": round(commit["lift_mm"], 1),
                    "accepted": ok,
                    **({"converged": converged} if strict_physics else {}),
                    **(
                        {"failure_kind": "physics_rejection"}
                        if strict_physics and not ok
                        else {}
                    ),
                    **({"capsized": caps} if caps else {}),
                    **({"penetration": pen} if pen else {}),
                }
                if ok:
                    self.physics.accept(self, commit)
                    self.pose[obj_id] = {
                        "translate": [0.0, 0.0, 0.0],
                        "euler": [0.0, 0.0, 0.0],
                        "scale": 1.0,
                    }
                    after = rested
                else:
                    self.physics.reject(self, commit)
                    if strict_physics and not converged:
                        reason = "physics settlement did not converge"
                    elif not target_ok:
                        reason = (
                            f"after the 180-yaw the object rolled/slid "
                            f"~{disp_p:.0f}mm away"
                            if roll_p
                            else f"after the 180-yaw the object cannot rest "
                            f"upright (settle tilt {tilt:.0f} deg > "
                            f"{FLIP_TILT_CAP_DEG:.0f})"
                        )
                    elif caps:
                        c0 = caps[0]
                        reason = (
                            f"the flip settle rolled {c0['name']} "
                            f"~{c0['disp_mm']:.0f}mm away"
                            if c0.get("rollable")
                            else f"the flip settle leaves {c0['name']} toppled "
                            f"({c0.get('tilt_deg', 0):.0f} deg from upright > "
                            f"{TILT_CAP_DEG:.0f})"
                        )
                    else:
                        reason = (
                            f"after the 180-yaw the object ends wedged "
                            f"~{pen['after_mm']:.0f}mm into {pen['other']}"
                        )
                    applied = False
        if applied and physics is None:
            # kinematic backend (or physics just disabled itself): keep the flip
            self.pose[obj_id] = flipped
        if applied:
            self.cur[obj_id] = after
            # every hint measured at the pre-flip pose is stale now (this used to drop
            # the flip verdict ALONE, leaving the yaw + size channels armed from the
            # reversed pose to feed the next move's relaxed gate).
            self._invalidate_hints(obj_id, "rotate_180")
            mesh2oid = {v["mesh_name"]: k for k, v in self.ctx.items()}
            for m in self._descendant_meshes(name):
                oid = mesh2oid.get(m)
                if oid is not None:
                    self.cur[oid] = None  # their holdout silhouette changed
        else:
            # restore the pre-flip pose in the blend (physics already inverted its
            # side; set_pose overwrites from the unchanged base)
            _score_pose(
                self.client, name, pose, mask, c["center_y"], c["depth"], iou_png
            )
            after = before
        self.rounds += 1
        self.trace[obj_id].append(
            {
                "round": self.rounds,
                "axis": "rotate_180",
                "score": round(after["score"], 4),
                "iou": round(after["iou"], 4),
                "gain": round(after["score"] - before["score"], 4),
                **({} if applied else {"dead": True}),
                **({"physics": physics} if physics else {}),
            }
        )
        self._dump()
        return {
            "applied": applied,
            "before": before,
            "after": after,
            "physics": physics,
            "reason": reason,
            "pose": dict(self.pose[obj_id]),
        }

    def _render_at(self, obj_id: str, pose, out_png: str) -> None:
        """Isolated TRANSPARENT GT-res render of ``obj_id`` at ``pose`` — a session
        delta dict ({translate, euler, scale}), or a 4x4 WORLD matrix (row-major)
        for a pose that is no longer reachable as a delta (exec's pre-edit
        snapshot after a settle rebuild, the pre-move pose after a physics
        rebase). Measurement-only (F0a): the delta path is a render_only
        set_pose; the matrix path premultiplies the world delta via set_matrix
        (no BVH resolve/seat, no rebase — physics-clean by construction). EITHER
        WAY the live pose is restored before returning by an EXACT-matrix
        put-back: the live world matrix is snapshotted before the excursion and
        restored with a set_matrix premultiply (restore_blend's discipline).
        Restoring by re-set_pose'ing ``self.pose`` deltas is NOT exact on the
        kinematic lane — a kinematic commit's non-render_only set_pose seats the
        object with a resolve/seat offset ON TOP of base+deltas, which a delta
        put-back silently drops (the applied-rotation appearance capture then
        persisted the un-seated pose via session.save()). Transparency matters:
        the masked-white crops need to knock the render bg out to white
        (_score_pose renders OPAQUE — the round-object FP bug). The
        visible_score isolation is left active, exactly as the scoring loop
        leaves it."""
        c = self.ctx[obj_id]
        name = c["mesh_name"]
        _isolate_iou(self.client, c["visible_score"])
        w0 = np.asarray(
            self.client.rpc({"cmd": "get_matrix", "names": [name]})[
                "matrices"
            ][name],
            dtype=np.float64,
        )
        if isinstance(pose, dict):
            self.client.rpc(
                {"cmd": "set_pose", "name": name, "translate": pose["translate"],
                 "euler": pose["euler"], "scale": pose["scale"],
                 "render_only": True}  # fmt: skip
            )
        else:
            M = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            self.client.rpc(
                {"cmd": "set_matrix", "name": name,
                 "M": (M @ np.linalg.inv(w0)).tolist()}  # fmt: skip
            )
        self.client.rpc(
            {"cmd": "render", "out": out_png, "transparent": True,
             "width": int(c["gt_w"]), "height": int(c["gt_h"])}  # fmt: skip
        )
        # where the excursion actually landed: the matrix path lands exactly on
        # M (premultiply algebra); the delta path needs one get_matrix read.
        w1 = (
            np.asarray(
                self.client.rpc({"cmd": "get_matrix", "names": [name]})[
                    "matrices"
                ][name],
                dtype=np.float64,
            )
            if isinstance(pose, dict)
            else np.asarray(pose, dtype=np.float64).reshape(4, 4)
        )
        self.client.rpc(
            {"cmd": "set_matrix", "name": name,
             "M": (w0 @ np.linalg.inv(w1)).tolist()}  # fmt: skip
        )

    def _photo_crop_sq(self, obj_id: str):
        """The PHOTO side of the appearance/orientation compare: ``obj_image``
        under the GT-resolution mask, on white, bbox-stretched to the fixed
        square (_orient_object_crop). Cached per object — the photo and its mask
        never change within a session (the cache dies with a rebuild). Shared by
        orientation_hint and appearance_agreement so the two sides can never
        drift."""
        # getattr: sessions built via __new__ in tests may not define the cache
        cache = getattr(self, "_photo_crop_cache", None)
        if cache is None:
            cache = self._photo_crop_cache = {}
        cached = cache.get(obj_id)
        if cached is not None:
            return cached
        from PIL import Image

        c = self.ctx[obj_id]
        gw, gh = int(c["gt_w"]), int(c["gt_h"])
        mask = np.load(c["mask_path"])
        if mask.shape != (gh, gw):
            m = (
                np.asarray(
                    Image.fromarray((mask > 0).astype("uint8") * 255).resize(
                        (gw, gh)
                    )
                )
                > 127
            )
        else:
            m = mask > 0
        photo = np.asarray(Image.open(c["obj_image"]).convert("RGB").resize((gw, gh)))
        cache[obj_id] = _orient_object_crop(photo, obj_mask=m)
        return cache[obj_id]

    def appearance_agreement(self, obj_id: str, pose) -> Optional[float]:
        """APPEARANCE agreement vs the photo at the GIVEN pose — the FACING
        check's ``sim0`` quantity (orientation_hint), parameterized by pose:
        isolated transparent render at ``pose`` -> position/scale-invariant
        object-on-white crop (_orient_object_crop) -> DINO patch similarity
        against the object's cached photo crop. Higher = this pose looks more
        like the photo. It is pose-COMPARISON evidence (a before -> after pair
        across ONE edit), NOT a global quality score — and it is the reliable
        channel exactly where silhouette yaw cannot be measured (bridge_6
        execute#1: the CORRECT +12deg fix read 0.4343 -> 0.4927 here while raw
        score/IoU dipped and the aniso-1.09 yaw pair aliased 28 -> 50).

        ``pose`` is a session delta dict or a 4x4 WORLD matrix for a pose that
        no longer exists as a delta (see _render_at); the live scene is restored
        either way. Cached per (object, pose digest) beside the other hint
        caches with the ORIENTATION cache's exact invalidation rules
        (_HINT_STALE "appearance" + session rebuild): a landed reorientation
        drops the object's entries — after the physics rebase the SAME
        zero-delta digest names a DIFFERENT world pose — while translations and
        resizes are survived by the position/scale-invariant crop. Best-effort:
        None on a dead feature worker, an empty render (no measurement, NOT
        agreement — the scale_hint rule), or any exception; an appearance
        number must never break a move."""
        try:
            from lib.tools.geometry import feature_metric as fm

            if not fm.available():
                return None
            # getattr: __new__-built test sessions may not define the cache
            cache = getattr(self, "_appearance_cache", None)
            if cache is None:
                cache = self._appearance_cache = {}
            digest = _pose_digest(pose)
            hit = cache.get(obj_id, {}).get(digest)
            if hit is not None:
                return hit
            from PIL import Image

            c = self.ctx[obj_id]
            adir = self.work / "appearance"
            adir.mkdir(parents=True, exist_ok=True)
            png = str(adir / f"{c['mesh_name']}_appearance.png")
            self._render_at(obj_id, pose, png)
            r = Image.open(png).convert("RGBA").resize(
                (int(c["gt_w"]), int(c["gt_h"]))
            )
            if not (np.asarray(r)[..., 3] > 127).any():
                return None  # rendered nothing (fully occluded): no measurement
            sim = round(
                float(fm.patch_sim(_orient_object_crop(r), self._photo_crop_sq(obj_id))),
                4,
            )
            cache.setdefault(obj_id, {})[digest] = sim
            with open(adir / "appearance.jsonl", "a") as f:
                f.write(json.dumps({"obj": obj_id, "digest": digest, "sim": sim}) + "\n")
            return sim
        except Exception as exc:  # noqa: BLE001 - advisory, must never break a move
            print(f"[appearance] {obj_id}: {exc}", file=sys.stderr)
            return None

    def orientation_hint(self, obj_id: str, prefix: str = "hint") -> Optional[dict]:
        """Automatic 180-reversal check: render the isolated object at its CURRENT
        pose and at yaw+180 (pose restored afterwards), score each against the
        photo crop with the DINO patch-similarity, and flag when the flip matches
        as well or better (sim180 >= sim0 + HINT_MARGIN). Both sides use the
        masked-white crop (isolated object on white) so the neighbour + background
        don't dominate the features. The verdict is computed ONCE (at the object's
        first, stable pose) and cached in ``self._orient_hint``; later investigates
        return the frozen result -- recomputing on perturbed poses is noisy. Cache
        is cleared by any LANDED reorientation, rotate_180 or rotation (see
        _invalidate_hints), and survives translations/resizes by design.
        Best-effort: returns None when the
        feature worker is unavailable or anything fails. All artifacts (crops + a
        hints.jsonl row) land under work/orientation_hints/."""
        try:
            from lib.tools.geometry import feature_metric as fm

            if not fm.available():
                return None
            cached = self._orient_hint.get(obj_id)
            if cached is not None:
                with open(
                    self.work / "orientation_hints" / "hints.jsonl", "a"
                ) as f:
                    f.write(json.dumps({"prefix": prefix, "cached": True, **cached}) + "\n")
                return dict(cached)
            c = self.ctx[obj_id]
            name = c["mesh_name"]
            hdir = self.work / "orientation_hints"
            hdir.mkdir(parents=True, exist_ok=True)
            stem = f"{prefix}_{name}"
            pose = self.pose[obj_id]
            flipped = {
                "translate": list(pose["translate"]),
                "euler": [pose["euler"][0], pose["euler"][1],
                          pose["euler"][2] + math.pi],  # fmt: skip
                "scale": pose["scale"],
            }
            raw0 = str(hdir / f"{stem}_cur_raw.png")
            raw1 = str(hdir / f"{stem}_flip_raw.png")
            # TRANSPARENT isolated renders at the two poses, live pose restored
            # after each — the shared pose-parameterized measurement path
            # (_render_at, also the engine of appearance_agreement).
            self._render_at(obj_id, pose, raw0)
            self._render_at(obj_id, flipped, raw1)
            # POSITION/SCALE-invariant crops (object-on-white, bbox zoom-cropped and
            # stretched to a fixed square) -- so an agent xy-move before the hint can't
            # shift the object out of a fixed window and collapse the margin, and round
            # objects normalise to ~0 (see _orient_object_crop). patch_sim over the whole
            # crop (both sides are the object on white). The photo side comes from the
            # shared per-object cache (_photo_crop_sq).
            from PIL import Image

            gw, gh = int(c["gt_w"]), int(c["gt_h"])
            photo_sq = self._photo_crop_sq(obj_id)
            cur_sq = _orient_object_crop(Image.open(raw0).convert("RGBA").resize((gw, gh)))
            flip_sq = _orient_object_crop(Image.open(raw1).convert("RGBA").resize((gw, gh)))
            sim0 = float(fm.patch_sim(cur_sq, photo_sq))
            sim180 = float(fm.patch_sim(flip_sq, photo_sq))
            paths = {
                "cur_png": str(hdir / f"{stem}_cur.png"),
                "flip_png": str(hdir / f"{stem}_flip.png"),
                "photo_png": str(hdir / f"{stem}_photo.png"),
            }
            cur_sq.save(paths["cur_png"])
            flip_sq.save(paths["flip_png"])
            photo_sq.save(paths["photo_png"])
            out = {
                "obj": obj_id,
                "sim0": round(sim0, 4),
                "sim180": round(sim180, 4),
                "flagged": bool(sim180 >= sim0 + fm.HINT_MARGIN),
                **paths,
            }
            self._orient_hint[obj_id] = dict(out)  # freeze the stable-pose verdict
            with open(hdir / "hints.jsonl", "a") as f:
                f.write(json.dumps({"prefix": prefix, **out}) + "\n")
            return out
        except Exception as exc:  # noqa: BLE001 - hints are advisory
            print(f"[orientation-hint] {obj_id}: {exc}", file=sys.stderr)
            return None

    def scale_hint(self, obj_id: str, prefix: str = "hint") -> Optional[dict]:
        """Closed-form silhouette scale check at the CURRENT pose. Returns
        {"scale_est", "area_scale", "overlap", "flagged", "shift_px", "yaw_deg",
        "yaw_aniso", "render_px", "mask_px", "overlay_png"} or None on failure.
        ``area_scale`` (= sqrt of the mask/render AREA ratio) is the uniform resize the
        operator can achieve, and is **None when either silhouette is empty** — no
        measurement, NOT agreement (it used to fall back to 1.0, which reported perfect
        size for an object that rendered nothing). ``render_px``/``mask_px`` are the two
        pixel counts, logged so a degenerate row is self-diagnosing and so the caller can
        say WHICH side was empty. ``scale_est`` is the principal-axis EXTENT ratio (None
        when gated out — still logged) used only to detect an aspect mismatch: it never
        reaches the agent, but ``size_hint_aspect_suppressed`` reads it to withhold a
        SIZE hint whose extent and area disagree, and this method clears the
        ``_scale_hint`` gate channel on the same predicate so hint and search cannot
        drift. Flags when |area_scale - 1| >= ``_SCALE_HINT_THRESH``. ``shift_px`` is the phase-
        correlation (du, dv) moving the render silhouette onto the mask — the
        POSITION hint's measured offset, free since both silhouettes are already
        in hand. Every computation is appended to work/scale_hints.jsonl; a
        centered red(render)/green(mask) silhouette overlay is saved under
        work/scale_hints/ for inspection. Best-effort."""
        try:
            from PIL import Image

            c = self.ctx[obj_id]
            _isolate_iou(self.client, c["visible_iou"])
            sil = self._object_sil(obj_id, self.pose[obj_id])
            mask = self._gt_mask_full(obj_id)
            overlap = _centered_overlap(sil, mask)
            s = _scale_estimate(sil, mask)
            # overlay: both silhouettes shifted to a common centroid — red =
            # render, green = photo mask, yellow = agreement; cropped to content.
            hdir = self.work / "scale_hints"
            hdir.mkdir(parents=True, exist_ok=True)
            overlay_png = str(hdir / f"{prefix}_{c['mesh_name']}_overlay.png")
            shifted = mask
            if sil.any() and mask.any():
                ya, xa = np.nonzero(sil)
                yb, xb = np.nonzero(mask)
                shifted = ro._shift2d(
                    mask,
                    int(round(ya.mean() - yb.mean())),
                    int(round(xa.mean() - xb.mean())),
                )
            rgb = np.zeros((*sil.shape, 3), np.uint8)
            rgb[..., 0][sil] = 255
            rgb[..., 1][shifted] = 255
            ys, xs = np.nonzero(sil | shifted)
            if xs.size:
                pad = 10
                rgb = rgb[
                    max(0, ys.min() - pad) : ys.max() + pad,
                    max(0, xs.min() - pad) : xs.max() + pad,
                ]
            Image.fromarray(rgb).save(overlay_png)
            shift = _phase_shift(sil, mask)
            yaw, yaw_aniso = _yaw_discrepancy(sil, mask)
            # AREA factor — the uniform resize the scale operator can actually achieve
            # (sqrt of the area ratio = linear size). The SIZE hint fires on THIS, not the
            # EXTENT (`scale_est`): the mesh is trusted, so scale_est is kept only as a
            # logged diagnostic (see scale_hints.jsonl) and is NEVER surfaced to the agent.
            a_ren, a_msk = float(sil.sum()), float(mask.sum())
            # An EMPTY silhouette on either side is NO MEASUREMENT -> None, like every other
            # field in this row. The old `else 1.0` fallback asserted PERFECT size agreement
            # for an object that rendered no pixels, so `flagged` came out False and nothing
            # downstream ever reported the object missing: 40 of 11248 fleet rows / 31
            # objects (0802_bulk_misc_online5 pen#0 at IoU 0.000 -- GT mask intact at 7133
            # px, render silhouette empty, fully occluded by its own stacking ANCESTOR,
            # which _isolate_iou deliberately keeps as a non-contributing holdout).
            # `render_px`/`mask_px` are logged so the degenerate rows are self-diagnosing and
            # so _object_hint_lines can say WHICH side was empty.
            area_scale = (a_msk / a_ren) ** 0.5 if a_ren and a_msk else None
            flagged = bool(
                area_scale is not None and abs(area_scale - 1.0) >= _SCALE_HINT_THRESH
            )
            out = {
                "obj": obj_id,
                "scale_est": round(s, 4) if s is not None else None,  # EXTENT (logged diagnostic only)
                "area_scale": None if area_scale is None else round(area_scale, 4),  # AREA (achievable target)
                "render_px": int(a_ren),
                "mask_px": int(a_msk),
                "overlap": round(overlap, 4),
                "flagged": flagged,
                "shift_px": None if shift is None else
                            [round(shift[0], 1), round(shift[1], 1)],  # fmt: skip
                # small-yaw signal (see _yaw_discrepancy) — the ROTATION hint input
                "yaw_deg": None if yaw is None else round(yaw, 1),
                "yaw_aniso": None if yaw_aniso is None else round(yaw_aniso, 2),
                "overlay_png": overlay_png,
            }
            with open(self.work / "scale_hints.jsonl", "a") as f:
                f.write(json.dumps({"prefix": prefix, **out}) + "\n")
            # feed the rotation search (seed + the yaw gate's pre reading); CLEAR on an
            # unmeasurable yaw (degenerate silhouette -> _yaw_discrepancy None), or a
            # hint from an older pose would outlive its own re-measurement — mirrors
            # the _scale_hint pop below.
            if yaw is not None:
                self._yaw_hint[obj_id] = (float(yaw), float(yaw_aniso or 0.0))
            else:
                self._yaw_hint.pop(obj_id, None)
            # feed the scale search's relaxed gate (mirrors _yaw_hint); clear when the
            # area matches so a later well-sized pose drops back to the normal bar. A hint
            # SUPPRESSED for aspect inconsistency must clear it too: leaving the cache set
            # would still hand a manually-requested move('scale') the relaxed
            # _HINT_MIN_GAIN bar for a reading we just declined to recommend — the opposite
            # of the intent. ONE predicate for the agent hint and the gate, same discipline
            # as yaw_hint_fires.
            if flagged and not size_hint_aspect_suppressed(s, area_scale, yaw_aniso):
                self._scale_hint[obj_id] = float(area_scale)
            else:
                self._scale_hint.pop(obj_id, None)
            return out
        except Exception as exc:  # noqa: BLE001 - hints are advisory
            print(f"[scale-hint] {obj_id}: {exc}", file=sys.stderr)
            return None

    def _invalidate_hints(self, obj_id: str, axis: str) -> None:
        """Drop the cached hint channels a LANDED ``axis`` move made stale. The ONE
        place that knows this rule — it used to be spelled out at two call sites and
        missing from a third (an applied flip cleared only the flip verdict, leaving
        the yaw + size channels holding pre-flip numbers; 9 fleet moves consumed one).

        Four caches feed different consumers, so "stale" differs per channel:

            channel            written by            read by                 cleared after
            _orient_hint       orientation_hint      the composer's chain    reorientation
            _yaw_hint          scale_hint            search: seed + yaw gate reorientation
            _scale_hint        scale_hint            search: gate only       reorientation
            _appearance_cache  appearance_agreement  move/freeform feedback  reorientation

        A REORIENTATION (rotate_180 / rotation) invalidates all four: the flip
        verdict was scored at the pre-move orientation (for a yaw fix, at the exact
        yaw that made it unreliable — see _FLIP_DEFER_ANISO_MIN), both silhouette
        readings were measured at the pre-move pose, and the appearance entries name
        poses by digest — after the physics rebase the SAME zero-delta digest names a
        DIFFERENT world pose, so a stale appearance number would be served for the
        new orientation. Left stale, the next move('rotation') re-seeds +-(old angle)
        — and the yaw gate would treat the pre-fix angle as the current pre-move
        reading.

        A TRANSLATION or RESIZE invalidates nothing, on purpose: the frozen flip
        verdict is deliberately immune to pose perturbation (the 0720 moved-mug false
        flag), a translation does not change the object's yaw, the size channel is
        re-measured by exec's _post_move_size_note, which re-runs scale_hint for
        exactly those aspects, and the appearance crop is position/scale-invariant
        (_orient_object_crop) — the same reason the flip verdict survives."""
        chans = _HINT_STALE.get(axis, ())
        # getattr: sessions built via __new__ in tests may not define every cache
        if "orient" in chans:
            getattr(self, "_orient_hint", {}).pop(obj_id, None)
        if "yaw" in chans:
            getattr(self, "_yaw_hint", {}).pop(obj_id, None)
        if "scale" in chans:
            getattr(self, "_scale_hint", {}).pop(obj_id, None)
        if "appearance" in chans:
            getattr(self, "_appearance_cache", {}).pop(obj_id, None)

    # -- investigate rendering -------------------------------------------------- #
    def full_render(self, out_png: str, hide: tuple = ()) -> str:
        """Full-scene render (nothing isolated) from the GT camera at GT resolution.
        ``hide`` (mesh names) renders the scene WITHOUT those objects — investigate's
        de-occluded crops hide exactly the occluders a redetect's edit removed, so
        IMAGE 1 matches the edited photo (owner-approved 2026-08-04)."""
        req = {"cmd": "isolate", "visible": None}  # None = show all
        if hide:
            req["hide"] = list(hide)
        self.client.rpc(req)
        self.client.rpc(
            {
                "cmd": "render",
                "out": out_png,
                "transparent": False,
                "width": self.gt_w,
                "height": self.gt_h,
            }
        )
        return out_png

    def rendered_bbox(self, obj_ids: list[str]) -> Optional[tuple]:
        """Union bbox (x0, y0, x1, y1) of the CURRENT rendered silhouettes of
        ``obj_ids`` at GT resolution (one isolated transparent render). Registered
        walls are depth-only holdouts, so an object hidden by a wall has no fake bbox."""
        names = [self.ctx[o]["mesh_name"] for o in obj_ids]
        tmp = str(self.work / "_investigate_bbox.png")
        self.client.rpc(
            {
                "cmd": "isolate",
                "visible": names,
                "holdout": list(getattr(self, "wall_holdouts", [])),
            }
        )
        self.client.rpc(
            {
                "cmd": "render",
                "out": tmp,
                "transparent": True,
                "width": self.gt_w,
                "height": self.gt_h,
            }
        )
        return _alpha_bbox(tmp)

    # -- persistence -------------------------------------------------------------- #
    def save(self, path: Optional[str] = None) -> None:
        """Write current poses to the shared blend (un-isolates all meshes)."""
        self.client.rpc({"cmd": "save", "path": path or self.blend})

    def rebuild(self) -> None:
        """Reload after a manual scene edit made the server copy stale. The move/
        flip TRACE survives the rebuild — wiping it made register.json under-report
        every move that preceded a freeform edit (0714_hr_abc1's spoon flip left no
        trace row). Pose state deliberately resets: the blend was reloaded."""
        keep_trace, keep_rounds = self.trace, self.rounds
        self.close()
        self.__init__(*self._init_args)
        for oid, rows in keep_trace.items():
            if oid in self.trace and rows:
                self.trace[oid] = rows
        self.rounds = max(self.rounds, keep_rounds)
        self._dump()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass

    def _dump(self) -> None:
        """register.json in the legacy schema so the demo panels keep working."""
        objs = []
        for oid in self.ctx:
            if not self.trace[oid] and self.cur.get(oid) is None:
                continue
            fin = self.cur.get(oid) or {}
            objs.append(
                {
                    "id": oid,
                    "pose": self.pose[oid],
                    "final": {
                        k: v for k, v in fin.items() if isinstance(v, (int, float))
                    },
                    "trace": self.trace[oid],
                }
            )
        payload = {"out_blend": self.blend, "objects": objs, "in_progress": None}
        reg_json = self.work / "register.json"
        tmp = reg_json.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, reg_json)


def _cat_inst(node: dict) -> tuple[str, int]:
    cat, inst = node["id"].rsplit("#", 1)
    return cat, int(inst)
