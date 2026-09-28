"""Isaac-backed physics authority for the composition stage.

Every pose the composition agent COMMITS is PhysX-rested: the kinematic line search
(``PoseSession.optimize_axis``) stays the scorer, but its winner is settled on the
persistent Isaac server (micro budget) before acceptance, and the rested pose — which
may legitimately include pitch/roll — is applied to the blend via ``set_matrix`` with
no BVH re-resolve. Moves of an object with scene-graph descendants carry the whole
subtree: the members are welded into ONE compound rigid body (``move_group``), so a
tray move takes its spoon along and the loaded assembly settles under its true CoM.

SCALE is the exception (``_commit_scale`` / server ``scale_resettle``): a resize is
about the object origin, never sliding out from under its cargo, so it skips the
carry decision entirely. It DROPS the resized parent alone to rest on its support,
after independent lateral bodies settle one at a time; descendants and contact
dependents then re-seat last in destination-support order, each with its own
lift-to-clear. Because scale
alone has no pre-physics feasibility clamp (a size fix must not silently shrink to
dodge a neighbor), its accept gate additionally rejects on the post-drop penetration
of the stack into its support.

Boot (once per scene, warmed eagerly at stage entry): dump every object/surface's
CURRENT world mesh from the live register-server blend (the authoritative pose source
— GLBs can be stale after agent stages), REUSE the preprocess ladder's validated
colliders mapped to the current pose (fresh CoACD only on correspondence failure, in
spawn workers — see ``_boot_colliders``), feed surfaces directly as static colliders. One-way sync per edit type: Isaac -> blend for move results
(set_matrix + rebase), blend -> Isaac for freeform ``execute_and_evaluate`` edits
(matrix_world diffs pushed as exact transforms on the next physics call).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import sys
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from lib.tools.geometry.collision import (
    coacd_diagnostic_root,
    corner_similarity,
    glb_face_corners,
    transform_stabilization_bundle,
)
from lib.tools.geometry.contact_policy import (
    HULL_SIBLING_TOL_M,
    HULL_SUBTOL_M,
    LATERAL_SIBLING,
    contact_class,
)
from lib.tools.geometry.initializer_pose_policy import (
    INITIALIZER_POSE_EXCLUSION_POLICY,
    initializer_pose_exclusions,
)
from lib.tools.geometry.physics import CAPSIZE_DEG, CAPSIZE_DISP_MM, capsized

# Capsize thresholds live in physics.py (ONE definition, shared with the settle ladder,
# both certify passes and the demo — see ``capsized``). These names are kept because the
# move-rejection and carry-decision call sites read better with them.
TILT_CAP_DEG = CAPSIZE_DEG  # reject a move whose settle capsizes any carried member
FLIP_TILT_CAP_DEG = 35.0  # rotate_180: reject only a genuine capsize (60-90 deg).
PEN_CAP_MM = HULL_SUBTOL_M * 1000.0  # 8 mm: the composition rules gate's generic hull allowance
PEN_NEW_MM = 5.0
PEN_SIBLING_CAP_MM = HULL_SIBLING_TOL_M * 1000.0
# Pre-physics move clamp: the search winner's move is reduced to the largest fraction
# (of its magnitude) that keeps the carried COMPOUND out of penetration (delta rule:
# after <= PEN_CAP_MM and worsening <= PEN_NEW_MM). If even that fraction is below
# FEASIBLE_MIN_FRAC the object is boxed in and the move is rejected; otherwise the
# clamped move is settled. This complements the post-settle penetration gate: the
# clamp handles reachable-fraction feasibility up front, while the per-body gate still
# rejects wedged landings when lift-to-clear cannot escape a wall. Scale is EXCLUDED (a genuine
# size fix blocked by a neighbor should move the neighbor, not silently shrink).
FEASIBLE_MIN_FRAC = 0.2
FOLLOW_TILT_DEG = 25.0
# Class-aware stability: a ROLLABLE object (VLM-judged from the reference image —
# a LYING marker/mic/bottle, a round pastry; falls back to server extents on old
# runs) rolls benignly, so tilt is meaningless for it — its roll about its own
# axis read as "capsize 118-180 deg" and burned 5-11 rejected moves per run on
# abc3's lying mic. Rollables are gated on settle DISPLACEMENT instead: rolling
# in place passes, rolling away rejects.
FLAT_DISP_CAP_MM = CAPSIZE_DISP_MM
CERT_DXY_CAP_M = 0.05
CERT_TILT_CAP_DEG = 10.0
CONTACT_DEP_TOL = 0.020
# ``supports_of`` deliberately reports a third, broad AABB-overlap class so its
# other consumers keep complete telemetry.  Only these two relations are definite
# enough to protect a touching body from contact-dependent re-simulation.  A
# co-level leaner (``ambiguous_same_level``) must stay free when its neighbour moves.
_PROTECTED_SUPPORT_RELATIONS = {"strict_below", "clear_below_or_container"}
_SYNC_EPS = 1e-4  # matrix_world drift below this is numeric noise, not an edit
_EDIT_MIN_MM = 2.0
_EDIT_MIN_DEG = 0.5
_EDIT_MIN_SCALE = 0.005
# Collider reuse may tolerate millimetric visual round-trip residue, but carrying a
# recorded center of mass is a stronger claim.  Use the same 20um bound as the typed
# world-semantics comparison before treating the similarity as a mass-property map.
_STABILIZATION_MAP_TOL_M = 2e-5


class PhysicsSettlementRejected(RuntimeError):
    """A strict GPT-6 edit produced no transactionally acceptable settled pose."""

    def __init__(self, reason: str, reports: list[dict] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.reports = list(reports or [])


class PhysicsSupportQueryFailed(RuntimeError):
    """A strict required-support postcondition could not be authenticated.

    This is deliberately distinct from :class:`PhysicsSettlementRejected`: a
    missing/malformed server answer is an infrastructure failure, not evidence that
    the requested edit itself was physically invalid.
    """

    def __init__(self, reason: str, reports: list[dict] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.reports = list(reports or [])


def _protected_support_names(records) -> set[str]:
    """Definite under-support/container names from a ``supports_of`` response.

    Servers predating the relation field remain conservative: an unlabeled record is
    protected exactly as it was before.  New ``ambiguous_same_level`` records are not
    protected, so a touching co-level neighbour is re-simulated instead of left aloft.
    """
    return {
        r["name"]
        for r in records
        if not r.get("relation") or r["relation"] in _PROTECTED_SUPPORT_RELATIONS
    }


def _validated_support_records(
    response: object, *, subject: str | None = None
) -> list[dict]:
    """Validate the fresh ``supports_of`` proof used by the strict edit gate.

    The ordinary contact-dependency path is intentionally best-effort, but an
    explicit ``required_support`` promise must fail closed when the server answer
    cannot be authenticated.  A missing ``relation`` remains compatible with old
    settle servers; an explicit unknown relation is retained as telemetry but is
    not definite according to :func:`_protected_support_names`.
    """
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise ValueError("supports_of returned no successful response object")
    records = response.get("supports")
    if not isinstance(records, list):
        raise ValueError("supports_of response has no supports list")
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"supports_of record {index} is not an object")
        support_name = record.get("name")
        if (
            not isinstance(support_name, str)
            or not support_name
            or support_name != support_name.strip()
        ):
            raise ValueError(f"supports_of record {index} has no exact non-empty name")
        if subject is not None and support_name == subject:
            raise ValueError(f"supports_of record {index} names the object itself")
        if "relation" in record and (
            not isinstance(record["relation"], str)
            or not record["relation"]
            or record["relation"] != record["relation"].strip()
        ):
            raise ValueError(f"supports_of record {index} has malformed relation")
        validated.append(dict(record))
    return validated


def _really_moved(D: np.ndarray) -> bool:
    """True when a world-delta exceeds the physical edit thresholds
    (translation > 2mm, rotation > 0.5deg, or uniform scale change > 0.5%)."""
    if float(np.linalg.norm(D[:3, 3])) * 1000.0 > _EDIT_MIN_MM:
        return True
    L = D[:3, :3]
    s = abs(float(np.linalg.det(L))) ** (1.0 / 3.0)
    if abs(s - 1.0) > _EDIT_MIN_SCALE:
        return True
    cos = (float(np.trace(L / max(s, 1e-9))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > _EDIT_MIN_DEG


_GEOM_SPAN_EPS = 1e-6  # axis span below this: degenerate, per-axis scale unrecoverable
_GEOM_NOEDIT_SCALE = 2e-3  # |s-1| below this on every axis: numeric noise, not an edit
_GEOM_NOEDIT_M = 5e-4  # |t| below this (0.5mm local): noise, not an edit
_GEOM_FIT_RTOL = 0.02  # affine-fit residual tolerance (fraction of the axis scale)
_GEOM_NOEDIT_VERTEX_M = 1e-5  # max per-vertex displacement below this (10um): no edit


def encode_local_vertices(V) -> str:
    """Wire form of a local-frame vertex array (``geom_sig`` ``V``): base64 of the
    little-endian float32 rows. float32 is Blender's own vertex precision, so the
    round trip is lossless; the register server writes the same form inline."""
    return base64.b64encode(np.ascontiguousarray(V, dtype="<f4").tobytes()).decode("ascii")


def decode_local_vertices(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").reshape(-1, 3).astype(np.float64)


def _affine_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = s
    A[:3, 3] = t
    return A


def _fit_vertices(Vo: np.ndarray, Vn: np.ndarray) -> np.ndarray | str | None:
    """Vertex-level form of :func:`fit_local_affine` (same nv, Blender index order)."""
    if Vo.shape != Vn.shape:
        return "rebuild"
    if np.abs(Vn - Vo).max() < _GEOM_NOEDIT_VERTEX_M:
        return None
    c_o, c_n = Vo.mean(0), Vn.mean(0)
    Xo, Xn = Vo - c_o, Vn - c_n
    s = np.ones(3)
    for i in range(3):
        if np.abs(Xo[:, i]).max() < _GEOM_SPAN_EPS:  # flat axis: no scale to recover
            if np.abs(Xn[:, i]).max() >= _GEOM_SPAN_EPS:
                return "rebuild"  # a flat axis grew: not a scale of the old axis
            continue
        s[i] = float(Xo[:, i] @ Xn[:, i]) / float(Xo[:, i] @ Xo[:, i])
    if (s <= 0).any():
        return "rebuild"  # mirrored data: not a scale the collider path can apply
    t = c_n - s * c_o
    span = np.maximum(Vo.max(0) - Vo.min(0), Vn.max(0) - Vn.min(0))
    tol = np.maximum(_GEOM_NOEDIT_M, _GEOM_FIT_RTOL * span)
    if (np.abs(Vo * s + t - Vn).max(0) > tol).any():
        return "rebuild"  # some vertex does not follow the map: a deformation
    if (np.abs(s - 1.0) < _GEOM_NOEDIT_SCALE).all() and (np.abs(t) < _GEOM_NOEDIT_M).all():
        return None
    return _affine_matrix(s, t)


def fit_local_affine(old: dict, new: dict) -> np.ndarray | str | None:
    """Explain a geom-signature change as an axis-aligned LOCAL affine A (4x4,
    per-axis scale + translation: new_verts = A @ old_verts) — the shape of every
    CoACD part survives such a map, so the collider is rescaled (qhull on the
    existing parts), not re-cooked. Returns None when the mesh data is unchanged,
    the matrix A for an affine-explainable edit, and the sentinel string
    ``"rebuild"`` for everything else (vertex or face count changed, or the
    vertices do not follow one map).

    With ``V`` (the local vertex array, register_blender_server ``_geom_sig``) the
    decision is made on the vertices themselves: a max per-vertex displacement
    under ``_GEOM_NOEDIT_VERTEX_M`` is no edit (float32 re-centring noise is
    1e-8..1e-7 m), the per-axis scale is a least-squares fit on centred
    coordinates, and the map must reproduce EVERY vertex within ``_GEOM_FIT_RTOL``
    of the span — an in-box deformation that leaves bounds, mean and std alone is
    caught here. Signatures without ``V`` (older servers, kinematic test mocks)
    fall back to the statistics fit below, which is noise-tolerant but blind to
    such deformations. The raw-byte sha256 that D1-a (09-16) used as the exact key
    is gone: the origin re-centre on every session rebuild rewrote vertices by
    ~1e-7 m, the digest flipped on untouched objects, and the 0916 batch paid 972
    CoACD re-cooks (~10 h) for edits that never happened."""
    if int(old["nv"]) != int(new["nv"]):
        return "rebuild"
    if "nf" in old and "nf" in new and int(old["nf"]) != int(new["nf"]):
        return "rebuild"
    if "V" in old and "V" in new:
        return _fit_vertices(decode_local_vertices(old["V"]), decode_local_vertices(new["V"]))
    lo_o, hi_o = np.asarray(old["lo"], float), np.asarray(old["hi"], float)
    lo_n, hi_n = np.asarray(new["lo"], float), np.asarray(new["hi"], float)
    c_o, c_n = np.asarray(old["c"], float), np.asarray(new["c"], float)
    sd_o, sd_n = np.asarray(old["sd"], float), np.asarray(new["sd"], float)
    span_o, span_n = hi_o - lo_o, hi_n - lo_n
    s, t = np.ones(3), np.zeros(3)
    for i in range(3):
        if span_o[i] < _GEOM_SPAN_EPS:
            if span_n[i] >= _GEOM_SPAN_EPS:
                return "rebuild"  # a flat axis grew: not a scale of the old axis
            t[i] = c_n[i] - c_o[i]
        else:
            s[i] = span_n[i] / span_o[i]
            t[i] = lo_n[i] - s[i] * lo_o[i]
    if (np.abs(s - 1.0) < _GEOM_NOEDIT_SCALE).all() and (np.abs(t) < _GEOM_NOEDIT_M).all():
        return None
    # the fitted map must also carry the mean and per-axis std, or the change is
    # a deformation the AABB fit merely brackets
    tol = np.maximum(_GEOM_NOEDIT_M, _GEOM_FIT_RTOL * np.maximum(span_n, span_o))
    if (np.abs(s * c_o + t - c_n) > tol).any():
        return "rebuild"
    if (np.abs(s * sd_o - sd_n) > np.maximum(_GEOM_NOEDIT_M, _GEOM_FIT_RTOL * sd_n
                                             )).any():  # fmt: skip
        return "rebuild"
    return _affine_matrix(s, t)


_VERIFY_TRANS_TOL_M = 1e-3
_VERIFY_ROT_TOL_DEG = 0.1
_VERIFY_SCALE_TOL = 1e-3


def _mat(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(4, 4)


def _rigid_of(C: np.ndarray) -> tuple[np.ndarray, float]:
    """Split a similarity delta into (orthonormal R, uniform scale s)."""
    s = float(np.cbrt(max(np.linalg.det(C[:3, :3]), 1e-12)))
    U, _, Vt = np.linalg.svd(C[:3, :3] / s)
    return U @ Vt, s


def _pose_drift(requested: np.ndarray, settled: np.ndarray) -> dict[str, float]:
    """Origin translation and attitude drift from requested to settled pose."""
    translation_m = float(np.linalg.norm(settled[:3, 3] - requested[:3, 3]))
    requested_r, _ = _rigid_of(requested)
    settled_r, _ = _rigid_of(settled)
    cos = (float(np.trace(settled_r @ requested_r.T)) - 1.0) / 2.0
    rotation_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return {
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
    }


def _authenticated_pose_pair(
    requested: np.ndarray | None, settled: np.ndarray | None
) -> bool:
    """Whether both internally-read world matrices can safely define edit drift.

    The matrices passed by ``settle_edited`` come directly from the register server
    immediately before and after the physics commit.  Validate their homogeneous
    shape and nondegenerate linear parts here so a malformed/missing pair never
    suppresses the established cumulative-capsize gate.
    """
    for value in (requested, settled):
        if value is None:
            return False
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return False
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            return False
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not np.isfinite(determinant) or determinant <= 1e-12:
            return False
    return True


def _strict_settlement_rejection(
    name: str,
    commit: dict,
    intent: str | dict | None = None,
    *,
    requested: np.ndarray | None = None,
    settled: np.ndarray | None = None,
) -> str | None:
    ""
    if not bool(commit.get("converged", True)):
        moving = list(commit.get("unconverged_bodies") or [])
        return "physics settlement did not converge" + (
            f" (still moving: {', '.join(moving)})" if moving else ""
        )
    penetration = commit.get("penetration")
    if penetration:
        body = penetration.get("body") or name
        return (
            f"{body} ended ~{float(penetration.get('after_mm', 0.0)):.0f}mm "
            f"inside {penetration.get('other', 'another body')} after settlement"
        )

    if intent is None:
        spec: dict = {"mode": "preserve"}
    elif isinstance(intent, str):
        spec = {"mode": intent}
    elif isinstance(intent, dict):
        spec = dict(intent)
    else:
        return "invalid intended resting mode specification"
    mode = str(spec.get("mode") or "preserve").lower()
    if mode not in {"preserve", "side", "free"}:
        return (
            f"unknown intended resting mode {mode!r}; expected preserve, side or free"
        )

    capsized_members = list(commit.get("capsized") or [])
    trusted_pair = _authenticated_pose_pair(requested, settled)
    sole_target_boot_capsize = (
        mode == "preserve"
        and trusted_pair
        and len(capsized_members) == 1
        and capsized_members[0].get("name") == name
        and capsized_members[0].get("rollable") is False
    )
    disallowed_capsizes = [
        record
        for record in capsized_members
        if (
            record.get("name") != name
            or (mode not in {"side", "free"} and not sole_target_boot_capsize)
        )
    ]
    if disallowed_capsizes:
        record = disallowed_capsizes[0]
        if record.get("rollable"):
            return (
                f"{record.get('name', name)} rolled/slid "
                f"~{float(record.get('disp_mm', 0.0)):.0f}mm away"
            )
        return (
            f"{record.get('name', name)} violated the intended {mode} resting "
            f"mode (tilt {float(record.get('tilt_deg', 0.0)):.0f} degrees)"
        )

    if mode != "free" and trusted_pair:
        drift = _pose_drift(requested, settled)
        rotation_value = spec.get(
            "max_settle_rotation_deg",
            15.0 if mode == "preserve" else 35.0,
        )
        try:
            max_rotation = float(rotation_value)
        except (TypeError, ValueError):
            max_rotation = math.nan
        if (
            isinstance(rotation_value, bool)
            or not math.isfinite(max_rotation)
            or max_rotation < 0.0
        ):
            return "invalid intended resting mode max_settle_rotation_deg"
        if drift["rotation_deg"] > max_rotation:
            return (
                f"{name} violated the intended {mode} resting mode: physics "
                f"rotated it {drift['rotation_deg']:.1f} degrees from the "
                f"requested pose (limit {max_rotation:.1f})"
            )
        default_translation = 0.05 if mode == "preserve" else None
        max_translation_value = spec.get(
            "max_settle_translation_m", default_translation
        )
        if max_translation_value is None and "max_settle_translation_m" in spec:
            return "invalid intended resting mode max_settle_translation_m"
        if max_translation_value is not None:
            try:
                max_translation = float(max_translation_value)
            except (TypeError, ValueError):
                max_translation = math.nan
            if (
                isinstance(max_translation_value, bool)
                or not math.isfinite(max_translation)
                or max_translation < 0.0
            ):
                return "invalid intended resting mode max_settle_translation_m"
            if drift["translation_m"] > max_translation:
                return (
                    f"{name} violated the intended {mode} resting mode: physics "
                    f"moved it {drift['translation_m']:.3f}m from the requested "
                    f"pose (limit {max_translation:.3f}m)"
                )
    return None


def _support_names(work: Path) -> set[str]:
    """Support-object mesh names for the finer CoACD budget on cache-miss cooks.
    ``scene_graph.json`` lives two levels up from both boot workdirs
    (<scene>/physics/composition and <scene>/physics/composition_certify).
    Best-effort: {} keeps every cook at the default budget."""
    from lib.tools.geometry.physics import support_mesh_names

    try:
        g = json.loads((work.parent.parent / "scene_graph.json").read_text())
        return support_mesh_names(g.get("nodes", []))
    except Exception:  # noqa: BLE001 - budget selection must never block a boot
        return set()


def _preprocess_records(work: Path) -> dict:
    ""
    path = work.parent / "pose_changes.json"
    try:
        return json.loads(path.read_text()).get("objects") or {}
    except Exception:  # noqa: BLE001 - overrides are best-effort
        import sys

        print(
            f"[physics-boot] WARNING: {path} missing/unreadable — no preprocess "
            "ladder records; every object boots with its RAW collider and NO "
            "stabilization bundle (pristine/CoM rescues lost). If this is a "
            "--skip-preprocess run, the staging forgot to copy it.",
            file=sys.stderr,
        )
        return {}


def _rollable_flags(work: Path) -> dict:
    """{mesh_name: bool} for objects whose placement record carries the VLM's
    state-aware ``rollable`` judgment; objects without it are omitted (the
    server then falls back to its extents heuristic)."""
    try:
        placement = json.loads((work.parent.parent / "placement.json").read_text())[
            "objects"
        ]
    except Exception:  # noqa: BLE001 - flags are best-effort
        return {}
    return {
        r["mesh_name"]: bool(r["rollable"])
        for r in placement
        if isinstance(r, dict) and r.get("mesh_name") and r.get("rollable") is not None
    }


def _capsized(members: list, tilts: dict, disp: dict, roll: dict) -> list:
    """Per-member stability violations for a settled move (``physics.capsized``: a
    ROLLABLE member is judged by settle displacement, everything else by cumulative
    tilt). Empty list = stable."""
    out = []
    for m in members:
        tilt, d, rollable = tilts.get(m, 0.0), disp.get(m, 0.0), bool(roll.get(m))
        if not capsized(tilt, d, rollable):
            continue
        out.append(
            {"name": m, "rollable": True, "disp_mm": round(d, 1)} if rollable
            else {"name": m, "rollable": False, "tilt_deg": round(tilt, 1)}
        )  # fmt: skip
    return out


def _joint_view(joint: dict, members: list[str]) -> dict:
    """Per-object view of a joint commit (:meth:`CompositionPhysics._commit_joint`):
    the same commit contract restricted to ``members`` (one moved hierarchy) so the
    strict gates judge each moved object on its own evidence — its riders' capsize,
    its own wedges — while accept/reject still act on the whole joint commit."""
    mset = set(members)
    # a freed dependent (member of the joint commit but not one of the moved hierarchies)
    # that ended wedged is judged in EVERY view (audit F-M2): the first view rejects
    deps = set(joint.get("members") or []) - set(joint.get("bodies") or joint.get("members") or [])
    per = {
        b: w for b, w in (joint.get("penetrations") or {}).items() if b in mset or b in deps
    }
    return {
        **joint,
        "members": list(members),
        "cum_tilt_deg": {m: joint["cum_tilt_deg"].get(m, 0.0) for m in members},
        "capsized": [c for c in joint.get("capsized") or [] if c.get("name") in mset],
        "penetration": (
            max(per.values(), key=lambda w: w["worsened_mm"]) if per else None
        ),
        "penetrations": per,
    }


def _would_topple(free_probe: dict) -> list:
    ""
    return sorted(
        k for k, p in free_probe.items()
        if (p["disp"] > FLAT_DISP_CAP_MM if p["rollable"]
            else p["tilt"] > FOLLOW_TILT_DEG)
    )  # fmt: skip


def _hull_cap_mm(
    a: str, b: str, support: dict | None, stacked: bool | None = None
) -> float:
    """Hull-depth allowance for the pair ``a``/``b`` under the composition rules gate's
    class ladder: lateral siblings (same support, SIDE BY SIDE) 20 mm, else 8 mm.
    ``stacked`` (server ``body_stacked``: one body's bottom at or above the other's top)
    turns a same-support pair into the stricter stacked class (audit F-M11); None keeps
    the lateral reading. Without a support map every pair is judged at 8 mm."""
    if not support:
        return PEN_CAP_MM
    direction = None if stacked is None else ((0.0, 0.0, 1.0) if stacked else (1.0, 0.0, 0.0))
    cls = contact_class(a, b, support, direction)
    return PEN_SIBLING_CAP_MM if cls == LATERAL_SIBLING else PEN_CAP_MM


def _body_pen_report(r: dict, support: dict | None = None) -> tuple[dict | None, dict]:
    ""
    per: dict = {}
    for b, maps in (r.get("body_pen") or {}).items():
        after = maps.get("after") or {}
        stacked = (r.get("body_stacked") or {}).get(b) or {}
        caps = {other: _hull_cap_mm(b, other, support, stacked.get(other)) for other in after}
        w = _pen_worst(maps.get("before") or {}, after, caps)
        if w is not None:
            per[b] = {**w, "body": b}
    if not per:
        return None, {}
    worst = max(per.values(), key=lambda w: w["worsened_mm"])
    return worst, per


def _penetration_report(r: dict, support: dict | None = None) -> tuple[dict | None, dict]:
    ""
    if r.get("body_pen") is not None:
        return _body_pen_report(r, support)
    return _pen_worst(r.get("pen_before_mm") or {}, r.get("pen_after_mm") or {}), {}


def _pen_worst(
    pen_before: dict, pen_after: dict, caps: dict | None = None
) -> dict | None:
    """Worst NEW interpenetration a settle left: the pair that ends deeper than its cap
    (``caps[other]``, default PEN_CAP_MM) AND worsened by more than PEN_NEW_MM
    (delta-based, so objects near a pre-existing overlap stay movable). None = clean."""
    worst = None
    for other, after in pen_after.items():
        worsened = after - pen_before.get(other, 0.0)
        if after > (caps or {}).get(other, PEN_CAP_MM) and worsened > PEN_NEW_MM:
            if worst is None or worsened > worst["worsened_mm"]:
                worst = {
                    "other": other,
                    "after_mm": round(after, 1),
                    "worsened_mm": round(worsened, 1),
                }
    return worst


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_artifact(scene: Path, path: Path) -> str:
    """Require ``path`` to exist inside ``scene``; return its scene-relative path."""
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(scene.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"stabilization source escapes scene root: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"stabilization source is missing: {relative}")
    return relative


def _mapped_stabilization_transport(
    work: Path,
    *,
    name: str,
    current_record: dict,
    source_npz: Path,
    chosen_glb: Path,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_residual_m: float,
) -> tuple[dict | None, float | None, dict]:
    """Transform one immutable preprocess stabilization bundle to the mapped pose.

    A mapped collider alone is not enough to carry a stale CoM: the settle-bake job
    must associate the chosen visual mesh with this object, and the current live
    record must retain the same collider choice and bundle as the immutable
    preprocess pose record.  Failures are diagnostic and fall back to a
    fresh-collider solve in :func:`_add_overrides`.
    """
    scene = work.parent.parent
    provenance: dict = {
        "transported": False,
        "mode": "mapped_preprocess_stabilization",
        "max_residual_m": float(max_residual_m),
    }
    try:
        if not isinstance(current_record, dict) or not current_record.get(
            "physics_overrides"
        ):
            raise ValueError("current pose record has no stabilization bundle")
        if max_residual_m > _STABILIZATION_MAP_TOL_M:
            raise ValueError(
                f"similarity residual {max_residual_m:.9g}m exceeds "
                f"{_STABILIZATION_MAP_TOL_M:.9g}m transport limit"
            )
        # ``placement.json`` is a live GPT-6 transaction overlay (typed add/edit
        # rewrites it).  The immutable settle-bake job is the per-object source
        # association: it binds the chosen input GLB to the placed output while the
        # normalized output stem binds the mesh name.
        jobs_path = scene / "physics" / "settle_bake_jobs.json"
        jobs_relative = _scene_artifact(scene, jobs_path)
        jobs = json.loads(jobs_path.read_text())
        if not isinstance(jobs, list):
            raise ValueError("immutable settle-bake jobs are not a list")

        def _job_path(value) -> Path:
            return scene / "meshes" / Path(value).name

        def _job_mesh_name(value) -> str:
            stem = Path(value).stem
            if stem.endswith("_pm"):
                stem = stem[: -len("_pm")]
            for suffix in ("_pcand", "_pristine", "_raw"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            return f"obj_{stem}"

        matches = [
            item
            for item in jobs
            if isinstance(item, dict)
            and _job_mesh_name(item.get("glb_out", "")) == name
            and _job_path(item.get("glb_in", "")).resolve() == chosen_glb.resolve()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"immutable settle-bake jobs have {len(matches)} source associations "
                f"for {name}"
            )
        placed_relative = _scene_artifact(scene, _job_path(matches[0]["glb_out"]))

        pose_path = scene / "physics" / "preprocess_pose_changes.json"
        pose_relative = _scene_artifact(scene, pose_path)
        collision_relative = _scene_artifact(scene, source_npz)
        glb_relative = _scene_artifact(scene, chosen_glb)
        immutable = json.loads(pose_path.read_text())
        immutable_record = (immutable.get("objects") or {}).get(name)
        if not isinstance(immutable_record, dict):
            raise ValueError(f"immutable preprocess pose has no record for {name}")
        if immutable_record.get("chosen") != current_record.get("chosen"):
            raise ValueError(
                "current collider choice differs from immutable preprocess"
            )
        source_bundle = immutable_record.get("physics_overrides")
        if source_bundle != current_record.get("physics_overrides"):
            raise ValueError(
                "current stabilization bundle differs from immutable preprocess"
            )
        mapped_bundle = transform_stabilization_bundle(
            source_bundle, scale, rotation, translation
        )
        flatten_mm = float(source_bundle.get("flatten_base_mm"))
        if not np.isfinite(flatten_mm) or flatten_mm < 0.0:
            raise ValueError("source stabilization flatten distance is invalid")
        provenance.update(
            {
                "transported": True,
                "source_collision": {"path": collision_relative},
                "source_visual": {"path": glb_relative},
                "source_bake_job": {
                    "path": jobs_relative,
                    "placed_visual_path": placed_relative,
                },
                "source_pose_changes": {"path": pose_relative},
                "similarity": {
                    "scale": float(scale),
                    "rotation": np.asarray(rotation, dtype=float).tolist(),
                    "translation": np.asarray(translation, dtype=float).tolist(),
                },
                "source_flatten_base_mm": flatten_mm,
                "flatten_baked_before_transform": True,
                "transformed_mass_properties": {
                    key: mapped_bundle[key]
                    for key in (
                        "com_world",
                        "diagonal_inertia",
                        "principal_axes",
                        "solver_mass_kg",
                    )
                },
            }
        )
        return mapped_bundle, flatten_mm, provenance
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        provenance["error"] = str(exc)
        return None, None, provenance


def _boot_colliders(
    work: Path, paths: dict, names: list[str], records: dict
) -> tuple[dict[str, str], dict[str, dict]]:
    """Collider npz per object for a session boot (composition AND certify).

    For each object, first try to REUSE the preprocess ladder's chosen collider
    (raw / pristine per pose_changes.json) by recovering the exact similarity from
    the chosen GLB's face corners to the current blend dump and transforming the
    validated parts — a fresh CoACD of the same object is a different decomposition
    with no stability guarantee (0715 wendy1: a pristine-rescued marker re-cooked
    fresh capsized at certify), and each cook costs ~27 s. Correspondence failures
    (geometry edited, files missing) fall back to fresh cooks, batched in spawn
    workers (CoACD in-process poisons later torch/scipy imports)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from lib.tools.geometry.collision import (
        corner_similarity,
        decompose_dump,
        glb_face_corners,
        transform_parts_npz,
    )

    try:
        placement = json.loads((work.parent.parent / "placement.json").read_text())[
            "objects"
        ]
    except Exception:  # noqa: BLE001 - reuse is best-effort
        placement = []
    by_name = {r.get("mesh_name"): r for r in placement if isinstance(r, dict)}
    supports = _support_names(work)
    out: dict[str, str] = {}
    bindings: dict[str, dict] = {}
    cook: list[tuple[str, str, bool]] = []
    for name in names:
        if name not in paths:
            continue
        out_npz = str(work / f"collision_{name}.npz")
        out[name] = out_npz
        binding = _map_validated(
            work,
            paths[name],
            name,
            records,
            by_name,
            out_npz,
            corner_similarity,
            glb_face_corners,
            transform_parts_npz,
        )
        if binding:
            bindings[name] = binding
            continue  # fmt: skip
        binding = _map_runtime_cook(
            work, paths[name], name, name in supports, out_npz,
            corner_similarity, transform_parts_npz,
        )  # fmt: skip
        if binding:
            bindings[name] = binding
            continue
        # fresh cook: pass the pristine sibling as the exact local-frame clamp
        # source (collision._pristine_frame). If the reuse failed because the
        # geometry actually changed, the correspondence tol guard no-ops it.
        cook.append(
            (
                paths[name],
                out_npz,
                name in supports,
                (by_name.get(name) or {}).get("pristine_glb"),
            )
        )
    if cook:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(4, len(cook)), mp_context=ctx) as ex:
            list(ex.map(decompose_dump, *zip(*cook)))
        for dump_path, out_npz, support, _frame in cook:
            _remember_runtime_cook(work, dump_path, out_npz, support)
    return out, bindings


def _runtime_collider_paths(work: Path, out_npz: str, support: bool) -> tuple[Path, Path]:
    """(collider npz, face-corner npy) of the remembered cook for this object; the
    support flag is part of the key because a support cooks at a finer budget."""
    root = work.parent / "runtime_colliders"
    stem = Path(out_npz).stem + ("__support" if support else "")
    return root / f"{stem}.npz", root / f"{stem}_corners.npy"


def _dump_face_corners(dump_path: str) -> np.ndarray:
    d = np.load(dump_path)
    return np.asarray(d["v0"], dtype=np.float64)[np.asarray(d["f0"])].reshape(-1, 3)


def _remember_runtime_cook(work: Path, dump_path: str, out_npz: str, support: bool) -> None:
    """Keep a fresh cook + the world-space face corners it was cooked from (~350 KB for
    a 30k-vertex object) as the next boot's reuse source. Best-effort."""
    try:
        npz, corners = _runtime_collider_paths(work, out_npz, support)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.save(corners, _dump_face_corners(dump_path).astype(np.float32))
        shutil.copyfile(out_npz, npz)
    except Exception as exc:  # noqa: BLE001 - remembering is an optimisation only
        print(f"[collision] could not remember cook {Path(out_npz).stem}: {exc}", file=sys.stderr)


def _map_runtime_cook(
    work, dump_path, name, support, out_npz, similarity, transform
) -> dict | None:
    """Reuse the remembered cook when the current dump is that geometry under an exact
    similarity (identical face-corner count, residual <= 2 mm) — the same test the
    preprocess-collider path applies. None means cook fresh."""
    try:
        npz, corners_path = _runtime_collider_paths(work, out_npz, support)
        if not (npz.is_file() and corners_path.is_file()):
            return None
        src = np.load(corners_path).astype(np.float64)
        dst = _dump_face_corners(dump_path)
        sim = similarity(src, dst)
        if sim is None:
            return None
        scale, rotation, translation = sim
        residual = float(np.abs(src @ (scale * rotation).T + translation - dst).max())
        transform(str(npz), float(scale), np.asarray(rotation, dtype=np.float64),
                  np.asarray(translation, dtype=np.float64), out_npz)  # fmt: skip
        print(
            f"[collision] reused remembered cook for {name} (residual {residual * 1000:.2f} mm)",
            file=sys.stderr,
        )
        return {
            "source_npz": str(npz),
            "source": "runtime_cook",
            "scale": float(scale),
            "rotation": np.asarray(rotation, dtype=float).tolist(),
            "translation": np.asarray(translation, dtype=float).tolist(),
            "face_corner_count": int(len(src)),
            "max_residual_m": residual,
        }
    except Exception:  # noqa: BLE001 - any surprise means "cook fresh", never crash
        return None


def _map_validated(
    work, dump_path, name, records, by_name, out_npz, similarity, corners, transform
) -> dict | None:
    """Map one validated collider and return its similarity/authentication binding.

    ``None`` still means the caller must cook fresh. A truthy result always means
    collider reuse succeeded, while ``stabilization_bundle`` is present only for a
    separately transported mass-property bundle.
    """
    try:
        pm_glb = (by_name.get(name) or {}).get("mesh_glb") or ""
        if not pm_glb.endswith("_pm.glb"):
            return None
        chosen_value = pm_glb[: -len("_pm.glb")] + ".glb"
        local_chosen = work.parent.parent / "meshes" / Path(chosen_value).name
        chosen_glb = str(local_chosen if local_chosen.is_file() else chosen_value)
        suffix = (
            "_pristine" if (records.get(name) or {}).get("chosen") == "pristine" else ""
        )
        src_npz = work.parent / f"collision_{name}{suffix}.npz"
        if not (os.path.exists(chosen_glb) and src_npz.exists()):
            return None
        d = np.load(dump_path)
        dst = np.asarray(d["v0"], dtype=np.float64)[
            np.asarray(d["f0"], dtype=np.int64)
        ].reshape(-1, 3)
        source_corners = corners(chosen_glb)
        sim = similarity(source_corners, dst)
        if sim is None:
            return None
        scale, rotation, translation = sim
        residual = float(
            np.abs(
                np.asarray(source_corners, dtype=np.float64)
                @ (float(scale) * np.asarray(rotation, dtype=np.float64)).T
                + np.asarray(translation, dtype=np.float64)
                - dst
            ).max()
        )
        mapped_bundle, flatten_mm, provenance = _mapped_stabilization_transport(
            work,
            name=name,
            current_record=records.get(name) or {},
            source_npz=Path(src_npz),
            chosen_glb=Path(chosen_glb),
            scale=float(scale),
            rotation=np.asarray(rotation, dtype=np.float64),
            translation=np.asarray(translation, dtype=np.float64),
            max_residual_m=residual,
        )
        transform(
            str(src_npz),
            float(scale),
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
            out_npz,
            **(
                {"pre_transform_flatten_mm": flatten_mm}
                if mapped_bundle is not None
                else {}
            ),
        )
        return {
            "source_npz": str(src_npz),
            "chosen_glb": str(chosen_glb),
            "scale": float(scale),
            "rotation": np.asarray(rotation, dtype=float).tolist(),
            "translation": np.asarray(translation, dtype=float).tolist(),
            "face_corner_count": int(len(source_corners)),
            "max_residual_m": residual,
            "stabilization_bundle": mapped_bundle,
            "stabilization_provenance": provenance,
            "flatten_baked_before_transform": mapped_bundle is not None,
        }
    except Exception:  # noqa: BLE001 - any surprise means "cook fresh", never crash
        return None


def _log_boot_overrides(
    work: Path, records: dict, applied: dict, bindings: dict | None = None
) -> None:
    ""
    try:
        mapping = bindings or {}
        (work / "boot_overrides.json").write_text(
            json.dumps(
                {
                    n: {
                        "applied": ov,
                        "chosen": (records.get(n) or {}).get("chosen"),
                        "com_missing": (
                            "com" not in ov
                            and bool((records.get(n) or {}).get("physics_overrides"))
                        ),
                        **(
                            {
                                "mapped_stabilization": mapping[n].get(
                                    "stabilization_provenance"
                                )
                            }
                            if n in mapping
                            and mapping[n].get("stabilization_provenance")
                            else {}
                        ),
                    }
                    for n, ov in applied.items()
                },
                indent=2,
            )
        )
    except Exception:  # noqa: BLE001 - logging must never block a boot
        pass


def _add_overrides(
    records: dict,
    name: str,
    parts_npz: str,
    *,
    binding: dict | None = None,
    strict: bool = False,
) -> dict:
    """Stabilization bundle for ``add``: an object that needed the CoM rung in
    preprocess CAPSIZES as a raw body (real8219 plush: raw 80 deg vs stabilized
    0.3 deg), so every later settle — winner commits AND certify — must carry the
    bundle. A mapped collider transports the complete stored mass
    properties through the same similarity and bakes the source-frame flatten into
    the mapped geometry. Otherwise the CoM/inertia bundle is recomputed from the
    fresh collider; strict GPT-6 boot fails closed if that recomputation is invalid."""
    rec = (records.get(name) or {}).get("physics_overrides")
    if not rec:
        return {}
    import subprocess
    import sys

    from lib.tools.geometry.physics import REPO_ROOT

    mapped = binding or {}
    provenance = mapped.get("stabilization_provenance") or {}
    transported = (
        mapped.get("stabilization_bundle")
        if provenance.get("transported") is True
        and mapped.get("flatten_baked_before_transform") is True
        else None
    )
    active = transported if isinstance(transported, dict) else rec
    out = {
        "friction": active.get("friction"),
        "damping": active.get("angular_damping"),
        # Authenticated reuse already contains T(flatten(source)). A second
        # destination-world-Z flatten would select a different base slice.
        "flatten_mm": (
            None
            if transported is not None
            and bool(mapped.get("flatten_baked_before_transform"))
            else active.get("flatten_base_mm")
        ),
    }
    if transported is not None:
        out.update(
            {
                "com": active["com_world"],
                "diagonal_inertia": active["diagonal_inertia"],
                "principal_axes": active["principal_axes"],
                "solver_mass_kg": active["solver_mass_kg"],
            }
        )
        return {key: value for key, value in out.items() if value is not None}
    # The solve runs in a CLEAN subprocess, twice over: (1) after an in-process
    # CoACD decomposition, importing scipy SEGFAULTS (OpenMP runtime clash —
    # same reason _decompose_all uses spawn workers); (2) this code path lives
    # inside the exec MCP server, whose stdout is the JSON-RPC transport, and
    # _stabilize_params prints (sliver-guard refusal). Reproduced live on
    # real8219: the boot killed the server at exactly this call.
    code = (
        "import json, sys\n"
        "from lib.tools.geometry.physics import _stabilize_params\n"
        "s = _stabilize_params(sys.argv[1])\n"
        "print('STAB' + json.dumps(\n"
        "    None if s is None else {k: s.get(k) for k in\n"
        "    ('com_world', 'diagonal_inertia', 'principal_axes',\n"
        "     'solver_mass_kg')}))\n"
    )

    def _solve_com():
        try:
            r = subprocess.run(
                [sys.executable, "-c", code, parts_npz],
                capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
            )  # fmt: skip
            if getattr(r, "returncode", 0) != 0:
                return None
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("STAB")]
            decoded = json.loads(lines[-1][len("STAB") :]) if lines else None
            validated = transform_stabilization_bundle(
                decoded, 1.0, np.eye(3), np.zeros(3)
            )
            return {
                key: validated[key]
                for key in (
                    "com_world",
                    "diagonal_inertia",
                    "principal_axes",
                    "solver_mass_kg",
                )
            }
        except Exception:  # noqa: BLE001
            return None

    stab = _solve_com()
    if stab is None:
        stab = _solve_com()  # one retry: a transient subprocess hiccup must not
        # silently strip the CoM a stabilized object NEEDS to stand
    if stab is not None:
        out["com"] = stab["com_world"]
        out["diagonal_inertia"] = stab["diagonal_inertia"]
        out["principal_axes"] = stab["principal_axes"]
        # solve_com's density-250 mass: _merge_vlm_physics rescales the inertia
        # to a VLM mass with it, then pops it (never sent to the server).
        out["solver_mass_kg"] = stab.get("solver_mass_kg")
    elif (records.get(name) or {}).get("chosen") == "stabilized" or rec.get(
        "com_world"
    ) is not None:
        detail = (
            f"{name} requires a preprocess stabilization bundle but has neither "
            "a mapped stabilization nor a valid fresh-collider "
            "CoM solve"
        )
        if provenance.get("error"):
            detail += f" (mapped transport failed: {provenance['error']})"
        if strict:
            raise RuntimeError(detail)
        print(
            f"[physics-boot] WARNING: {detail} — booting WITHOUT the CoM "
            "override (expect instability)",
            file=sys.stderr,
        )
    return {k: v for k, v in out.items() if v is not None}


def _merge_vlm_physics(work: Path, name: str, dump_npz: str, ov: dict) -> dict:
    """VLM physics (preprocess estimate) under the stabilization bundle
    (per-key precedence: overrides > VLM): mass is the estimate rescaled to the
    CURRENT visual mesh via the OBB-extents anchor (robust to any resize since
    estimation), friction fills in only when the bundle carries none, and an
    explicit bundle inertia is rescaled to the new mass (inertia is linear in
    mass; ref = solve_com's density-250 mass) so the (mass, CoM, inertia)
    triple stays self-consistent. Missing estimate -> ``ov`` unchanged."""
    out = dict(ov)
    ref_mass = out.pop("solver_mass_kg", None)
    from lib.tools.geometry.physics_estimate import (
        load_estimates,
        obb_extents,
        persistable_mass,
        rescaled_mass,
    )

    entry = load_estimates(work.parent / "physics_vlm.json").get(name)
    if not entry or entry.get("mass_kg") is None:
        return out
    try:
        d = np.load(dump_npz)
        v = np.vstack([d[f"v{i}"] for i in range(int(d["n"]))])
        mass, _ = rescaled_mass(entry, obb_extents(v), label=name)
    except Exception:  # noqa: BLE001 - estimates are best-effort
        return out
    out["mass"] = persistable_mass(mass, mass, mass)[0]
    if out.get("friction") is None and entry.get("friction") is not None:
        out["friction"] = float(entry["friction"])
    if out.get("diagonal_inertia") and ref_mass:
        out["diagonal_inertia"] = [
            float(x) * out["mass"] / float(ref_mass) for x in out["diagonal_inertia"]
        ]
    return out


class CompositionPhysics:
    """Persistent Isaac session mirroring the composition blend (see module doc)."""

    def __init__(self, session, work_dir: str, isaac_python: str | None = None):
        from lib.tools.geometry.physics import DEFAULT_ISAAC_PYTHON, SettleClient

        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.parents = dict(session.parents)  # mesh-name -> parent mesh-name
        names = sorted(session.prepared)
        surfaces = list(session.surfaces)
        paths = session.client.rpc(
            {"cmd": "dump_npz", "names": names + surfaces, "out": str(self.work)}
        )["paths"]
        self.client = SettleClient(
            self.work,
            isaac_python or DEFAULT_ISAAC_PYTHON,
            shared_dir=self.work.parent,  # <scene>/physics — reuse preprocess's boot
        )
        records = _preprocess_records(self.work)
        rollable = _rollable_flags(self.work)
        colliders, bindings = _boot_colliders(self.work, paths, names, records)
        strict = getattr(session, "harness_profile", "baseline") == "gpt6_v1" and bool(
            getattr(session, "strict_post_edit_physics", False)
        )
        applied: dict[str, list[str]] = {}
        for name, parts_npz in colliders.items():
            ov = _merge_vlm_physics(
                self.work, name, paths[name],
                _add_overrides(
                    records,
                    name,
                    parts_npz,
                    binding=bindings.get(name),
                    strict=strict,
                ),
            )  # fmt: skip
            applied[name] = sorted(ov)
            self.client.rpc(
                {
                    "cmd": "add",
                    "name": name,
                    "npz": parts_npz,
                    **({"rollable": rollable[name]} if name in rollable else {}),
                    **ov,
                }  # fmt: skip
            )
        _log_boot_overrides(self.work, records, applied, bindings)
        for name in surfaces:
            if name in paths:
                self.client.rpc({"cmd": "add_static", "name": name, "npz": paths[name]})
        self._names = [n for n in names if n in paths]
        self._synced = self._blend_matrices(session)  # last state pushed to Isaac
        # blend matrices at add() time: verify_sync's reference frame — the server's
        # cumulative ``pose`` total per body must map _boot to the current blend pose
        self._boot = {n: W.copy() for n, W in self._synced.items()}
        self._records = records  # boot-time add() overrides, for collider rebuilds
        self._rollable = rollable
        self._dump_paths = dict(paths)
        self._colliders = dict(colliders)
        self._collider_bindings = dict(bindings)
        self._geom = self._geom_sigs(session)  # mesh-DATA edit baselines

    # -- sync (blend -> Isaac) --------------------------------------------------- #
    def _blend_matrices(self, session) -> dict[str, np.ndarray]:
        ms = session.client.rpc({"cmd": "get_matrix", "names": self._names})["matrices"]
        return {n: _mat(m) for n, m in ms.items()}

    def _geom_sigs(self, session) -> dict:
        """Local-frame geometry signatures per handle from the register server;
        {} when the server predates cmd geom_sig (kinematic test mocks) — the
        mesh-data edit detection then simply stays off."""
        try:
            r = session.client.rpc({"cmd": "geom_sig", "names": self._names})
        except Exception:  # noqa: BLE001 - mocks/old servers
            return {}
        return (r.get("sigs") or {}) if isinstance(r, dict) else {}

    def _absorb_geom_edits(self, session) -> list[str]:
        """Detect mesh-DATA edits (see ``fit_local_affine``) and reflect them into
        the Isaac mirror. An affine-explainable edit A is FOLDED into the sync
        baselines (``_synced``/``_boot`` @= inv(A)) so the very next matrix diff
        carries A: ``sync`` pushes it as an exact transform, ``verify_sync`` heals
        it, and ``settle_edited``'s commit routes it through the scale commit
        (collider parts rescale, cargo re-seats, gates run). A non-affine edit
        re-cooks the collider instead (``_rebuild_body``). Returns the names that
        took the rebuild path — their matrix diff stays identity, so callers that
        settle must force-include them."""
        base = getattr(self, "_geom", None)
        if not base:
            return []
        import sys as _sys

        rebuilt = []
        for n, sig in self._geom_sigs(session).items():
            old = base.get(n)
            if old is None:
                base[n] = sig
                continue
            A = fit_local_affine(old, sig)
            if A is None:
                continue
            if isinstance(A, str):  # "rebuild"
                self._rebuild_body(session, n)
                rebuilt.append(n)
                print(
                    f"[physics-sync] {n}: non-affine mesh-data edit — collider "
                    "re-cooked from a fresh dump",
                    file=_sys.stderr,
                )
            else:
                Ai = np.linalg.inv(A)
                self._synced[n] = self._synced[n] @ Ai
                if n in (getattr(self, "_boot", None) or {}):
                    self._boot[n] = self._boot[n] @ Ai
                print(
                    f"[physics-sync] {n}: mesh-data edit (local scale "
                    f"{[round(float(v), 4) for v in np.diag(A)[:3]]}) folded "
                    "into the pose diff",
                    file=_sys.stderr,
                )
            base[n] = sig
        return rebuilt

    def _rebuild_body(self, session, name: str) -> None:
        """Re-cook + re-add one body whose mesh data changed non-affinely: fresh
        world dump -> fresh CoACD (never remap the validated parts — the geometry
        they were validated for no longer exists) -> remove/add at the current
        blend pose with the boot-time overrides. Resets the sync/boot baselines
        (the re-added body's cumulative total restarts at identity)."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from lib.tools.geometry.collision import decompose_dump

        paths = session.client.rpc(
            {"cmd": "dump_npz", "names": [name], "out": str(self.work)}
        )["paths"]
        npz = str(self.work / f"collision_{name}.npz")
        ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                ex.submit(
                    decompose_dump, paths[name], npz, name in _support_names(self.work)
                ).result()
        except BrokenProcessPool as exc:
            raise RuntimeError(
                f"CoACD worker exited abruptly while rebuilding {name!r}; "
                f"cook evidence directory: {coacd_diagnostic_root(npz)}. "
                "Native stderr remains in the owning tool/core log."
            ) from exc
        records = getattr(self, "_records", None) or {}
        roll = getattr(self, "_rollable", None) or {}
        if getattr(self, "_collider_bindings", None) is not None:
            self._collider_bindings.pop(name, None)
        strict = getattr(session, "harness_profile", "baseline") == "gpt6_v1" and bool(
            getattr(session, "strict_post_edit_physics", False)
        )
        ov = _merge_vlm_physics(
            self.work,
            name,
            paths[name],
            _add_overrides(records, name, npz, strict=strict),
        )
        self.client.rpc({"cmd": "remove", "name": name})
        self.client.rpc(
            {
                "cmd": "add",
                "name": name,
                "npz": npz,
                **({"rollable": roll[name]} if name in roll else {}),
                **ov,
            }  # fmt: skip
        )
        cur = self._blend_matrices(session)[name]
        self._synced[name] = cur
        if getattr(self, "_boot", None) is not None:
            self._boot[name] = cur.copy()
        if getattr(self, "_dump_paths", None) is not None:
            self._dump_paths[name] = paths[name]
        if getattr(self, "_colliders", None) is not None:
            self._colliders[name] = npz

    def sync(self, session) -> list[str]:
        """Push freeform blend edits (execute_and_evaluate / undo) to Isaac as exact
        transforms. Mesh-DATA edits are absorbed first (``_absorb_geom_edits``): an
        affine one rides the same exact-transform push; a rebuilt body is already
        at blend truth (mirrored, but NOT settled — this is the mirror path)."""
        rebuilt = self._absorb_geom_edits(session)
        cur = self._blend_matrices(session)
        moved = list(rebuilt)
        for n, W in cur.items():
            D = W @ np.linalg.inv(self._synced[n])
            if np.abs(D - np.eye(4)).max() > _SYNC_EPS:
                self.client.rpc({"cmd": "transform", "name": n, "M": D.tolist()})
                moved.append(n)
        self._synced = cur
        return moved

    def verify_sync(self, session, names) -> list[str]:
        """Isaac<->blend pose invariant: for each body, the server's cumulative delta
        (cmd ``pose`` total) applied to the boot blend matrix must reproduce the
        CURRENT blend matrix within tolerance. On a mismatch the server is resynced
        to blend truth (one ``transform``) and the drift is logged loudly — a silent
        divergence otherwise persists for the whole stage (0720_orinit3_abc1: buried
        croissant passed the settle notes AND the rules gate). Returns the resynced
        names. Skips silently on servers without ``pose`` (kinematic test mocks) and
        on sessions built before the _boot snapshot existed."""
        boot = getattr(self, "_boot", None)
        if boot is None:
            return []
        import sys as _sys

        # absorb mesh-DATA edits first: a folded affine makes `expected` carry the
        # data-level scale, so the invariant check below heals it into the hulls
        # before the caller (the rules gate) trusts contact_pairs
        self._absorb_geom_edits(session)
        cur = self._blend_matrices(session)
        fixed = []
        for n in names:
            if n not in boot or n not in cur:
                continue
            try:
                total = _mat(self.client.rpc({"cmd": "pose", "name": n})["total"])
            except Exception:  # noqa: BLE001 - mocks/old servers: invariant unavailable
                return fixed
            expected = cur[n] @ np.linalg.inv(boot[n])
            p = np.append(boot[n][:3, 3], 1.0)
            t_err = float(np.linalg.norm((total @ p)[:3] - (expected @ p)[:3]))
            Rt, st = _rigid_of(total)
            Re, se = _rigid_of(expected)
            c = (float(np.trace(Rt @ Re.T)) - 1.0) / 2.0
            a_err = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            s_err = abs(st / max(se, 1e-12) - 1.0)
            if (
                t_err <= _VERIFY_TRANS_TOL_M
                and a_err <= _VERIFY_ROT_TOL_DEG
                and s_err <= _VERIFY_SCALE_TOL
            ):
                continue
            self.client.rpc(
                {
                    "cmd": "transform",
                    "name": n,
                    "M": (expected @ np.linalg.inv(total)).tolist(),
                }  # fmt: skip
            )
            fixed.append(n)
            print(
                f"[physics-sync] {n}: Isaac pose drifted from the blend "
                f"({t_err * 1000:.1f}mm / {a_err:.2f}deg / scale {s_err:.4f}) "
                "— resynced to blend truth",
                file=_sys.stderr,
            )
        return fixed

    # -- the winner commit -------------------------------------------------------- #
    def _settle_support_map(self) -> dict[str, str]:
        """Support map for the settle gate's hull-allowance ladder: each body's carry
        parent, and one shared root for every body without one (side-by-side bodies on
        the main support are lateral siblings, exactly as the rules gate classes them)."""
        base = dict(getattr(self, "support_map", None) or {})  # rules-gate map, if handed over
        return {n: base.get(n) or self.parents.get(n) or "__root__" for n in self._names}

    def _descendants(self, name: str) -> list[str]:
        """All Isaac-known descendants of ``name``, parents before children."""
        kids: dict[str, list[str]] = {}
        for c, p in self.parents.items():
            kids.setdefault(p, []).append(c)
        out, stack = [], list(kids.get(name, []))
        while stack:
            n = stack.pop(0)
            if n in self._names:
                out.append(n)
            stack.extend(kids.get(n, []))
        return out

    def _contact_deps(
        self,
        name: str,
        exclude: set,
        members: list[str] | None = None,
        cache: dict | None = None,
    ) -> list[str]:
        """Movable objects TOUCHING ``name`` that depend on it for stability — its
        lateral leaners (a mug propped on the keyboard) plus anything perched on it,
        via cmd_contacts with a CONTACT_DEP_TOL dilation (so LATERAL leans are caught,
        not just on-top, AND a near-rest leaner with a few-mm gap counts). Excludes
        ``name``, its members/kids (in ``exclude``), and every DEFINITE body that ANY
        weld member rests ON or is CONTAINED BY (cmd_supports_of relations
        ``strict_below`` / ``clear_below_or_container`` over ``members``, not just the
        parent — a body a carried CHILD rests on must not be freed out from under it
        either; never free your own support: a fork must not free the rimmed tray it
        sits in). The broad ``ambiguous_same_level`` result is deliberately NOT
        excluded: it describes the bagel/leaner class this method exists to re-settle.
        These re-settle when ``name`` moves out from under/against them instead of
        hanging frozen. CARGO CLOSURE: a freed body's own riders join the
        free set (0720_orinit3_abc1 t18: the tray was freed out from under its
        mid-air static cargo) — they re-settle after it in the server's
        support-first phase-B order. Best-effort: any RPC gap (kinematic mock-client
        tests) yields [] -> the old always-weld behavior.

        ``cache`` (per-move, from optimize_axis's stability-clamp ladder): every
        failed rung ends in an exact two-sided reject, so re-queries within one
        move see IDENTICAL scene state — the cached set is returned without an
        RPC. Never persisted across accepts (an accept changes poses).
        ``pairs_only`` skips the server's per-pair depth scan — this query reads
        only the pair names, and the depth was ~57 of a 59 s stacked-scene sweep."""
        key = (name, frozenset(exclude), tuple(members or ()))
        if cache is not None and key in cache:
            return list(cache[key])
        try:
            pairs = (
                self.client.rpc(
                    {"cmd": "contacts", "tol": CONTACT_DEP_TOL, "pairs_only": True}
                )
                or {}
            ).get("pairs", [])
            supports: set = set()
            for m in members if members else [name]:
                supports |= _protected_support_names(
                    (self.client.rpc({"cmd": "supports_of", "name": m}) or {}).get(
                        "supports", []
                    )
                )
        except Exception as exc:  # noqa: BLE001 - degrade to no contact re-settle
            print(
                f"[contact-deps] query failed for {name}: {type(exc).__name__}: {exc} "
                "— no dependents will be re-settled (riders stay static)",
                file=sys.stderr,
            )
            return []
        adj: dict[str, set] = {}
        for p in pairs:
            a, b = p.get("a"), p.get("b")
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        deps = [
            d for d in sorted(adj.get(name, ()))
            if d in self._names and d not in exclude and d not in supports
        ]  # fmt: skip
        seen, queue = set(deps), list(deps)
        while queue:
            d = queue.pop(0)
            for r in sorted(adj.get(d, ())):
                if r in seen or r in exclude or r in supports or r not in self._names:
                    continue
                try:
                    sup_r = _protected_support_names(
                        (self.client.rpc({"cmd": "supports_of", "name": r}) or {}).get(
                            "supports", []
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - closure is best-effort
                    print(
                        f"[contact-deps] supports_of failed for {r} (rider of {d}): "
                        f"{type(exc).__name__}: {exc} — left static",
                        file=sys.stderr,
                    )
                    continue
                if d in sup_r:  # r rides on the freed body d -> free it too
                    seen.add(r)
                    deps.append(r)
                    queue.append(r)
        if cache is not None:
            cache[key] = list(deps)
        return deps

    def _carry_deltas(self, name: str, kids: list, W_cand: np.ndarray) -> dict:
        """Per-member world deltas for a rigid (weld) carry of ``name`` + ``kids`` to
        the parent target pose ``W_cand``: parent gets ``W_cand @ inv(synced)``; each
        child gets the same rotation about the parent's synced origin plus a radial
        scale term. Shared by the commit and the pre-physics feasibility clamp."""
        C = {name: W_cand @ np.linalg.inv(self._synced[name])}
        R, s = _rigid_of(C[name])
        p = self._synced[name][:3, 3]  # parent origin at move start = set_pose pivot
        p2 = (C[name] @ np.append(p, 1.0))[:3]
        for c in kids:
            radial = (s - 1.0) * (R @ (self._synced[c][:3, 3] - p))
            radial[2] = 0.0  # z re-seated by physics, not geometry
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, p2 - R @ p + radial
            C[c] = M
        return C

    def max_feasible(
        self, session, name: str, aspect: str, resolution: float = 0.125
    ) -> float:
        ""
        if aspect == "scale":
            return 1.0
        kids = self._descendants(name)
        members = [name] + kids
        W_synced = self._synced[name]
        W_cand = self._blend_matrices(session)[name]
        Cpar = W_cand @ np.linalg.inv(W_synced)
        R, _ = _rigid_of(Cpar)
        t = Cpar[:3, 3]
        p = W_synced[:3, 3]
        is_rot = aspect == "rotation"
        theta = math.atan2(R[1, 0], R[0, 0]) if is_rot else 0.0

        def _W_at(f: float) -> np.ndarray:
            if is_rot:
                cf, sf = math.cos(f * theta), math.sin(f * theta)
                Rz = np.eye(4)
                Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = cf, -sf, sf, cf
                Tp, Tm = np.eye(4), np.eye(4)
                Tp[:3, 3], Tm[:3, 3] = p, -p
                Cf = Tp @ Rz @ Tm
            else:  # translation
                Cf = np.eye(4)
                Cf[:3, 3] = f * t
            return Cf @ W_synced

        def _pen_at(f: float) -> dict:
            Cf = self._carry_deltas(name, kids, _W_at(f))
            r = self.client.rpc({
                "cmd": "penetration", "members": members,
                "deltas": {m: Cf[m].tolist() for m in members},
            })  # fmt: skip
            return (r or {}).get("pen", {}) or {}

        before = _pen_at(0.0)
        if _pen_worst(before, _pen_at(1.0)) is None:
            return 1.0
        lo, hi = 0.0, 1.0  # lo feasible by definition (f=0 = baseline), hi infeasible
        while hi - lo > resolution:
            mid = (lo + hi) / 2.0
            if _pen_worst(before, _pen_at(mid)) is None:
                lo = mid
            else:
                hi = mid
        return lo

    def commit_move(
        self,
        session,
        name: str,
        carry_children: bool = True,
        deps_cache: dict | None = None,
    ) -> dict:
        ""
        if carry_children:
            _, s = _rigid_of(
                self._blend_matrices(session)[name] @ np.linalg.inv(self._synced[name])
            )
            if abs(s - 1.0) >= 1e-3:
                return self._commit_scale(
                    session, name, self._descendants(name), deps_cache
                )
        kids = self._descendants(name) if carry_children else []
        exclude = [] if carry_children else self._descendants(name)
        mean_iou = getattr(session, "mean_iou", None)
        strict = getattr(session, "harness_profile", "baseline") == "gpt6_v1" and bool(
            getattr(session, "strict_post_edit_physics", False)
        )
        if not kids or mean_iou is None or strict:
            return self._commit_carry(session, name, kids, exclude, deps_cache)
        U = self._commit_solo(session, name, kids)
        topple = _would_topple(U["free_probe"])
        decision = {"topple": topple, "avg_iou_stay": None, "avg_iou_follow": None}
        if not topple:
            decision["avg_iou_stay"] = mean_iou([name] + kids)
        self.reject(session, U)
        commit = self._commit_carry(session, name, kids, [], deps_cache)
        if not topple:
            decision["avg_iou_follow"] = mean_iou([name] + kids)
        follow = bool(topple) or (
            decision["avg_iou_stay"] is None
            or decision["avg_iou_follow"] is None
            or decision["avg_iou_follow"] > decision["avg_iou_stay"]
        )
        if not follow:
            self.reject(session, commit)
            for m in U["members"]:
                self.client.rpc(
                    {"cmd": "transform", "name": m, "M": U["totals"][m].tolist()}
                )
                session.client.rpc(
                    {
                        "cmd": "set_matrix",
                        "name": m,
                        "M": U["blend_deltas"][m].tolist(),
                    }  # fmt: skip
                )
            commit = U
        commit["followed"] = follow
        commit["carry_decision"] = decision
        return commit

    def _commit_solo(self, session, name: str, kids: list) -> dict:
        ""
        W_cand = self._blend_matrices(session)[name]
        C = W_cand @ np.linalg.inv(self._synced[name])
        r = self.client.rpc(
            {
                "cmd": "move_group",
                "members": [name],
                "budget": "micro",
                "deltas": {name: C.tolist()},
                "free": kids,
                "pen_ignore": kids,
                "sequential": True,
            }  # fmt: skip
        )
        totals = {name: _mat(r["D"]) @ C}
        tilts = dict(r["cum_tilt_deg"])
        disp = dict(r.get("disp_mm") or {})
        roll = dict(r.get("rollable") or {})
        free_probe = {}
        for k in kids:
            fr = (r.get("free") or {}).get(k) or {}
            totals[k] = _mat(fr.get("D") or np.eye(4).tolist())
            tilts[k] = float(fr.get("cum_tilt_deg", 0.0))
            disp[k] = float(fr.get("disp_mm", 0.0))
            roll[k] = bool(fr.get("rollable", False))
            free_probe[k] = {
                "tilt": float(fr.get("tilt_deg", 0.0)),
                "disp": disp[k],
                "rollable": roll[k],
            }
        deltas = {name: totals[name] @ np.linalg.inv(C)}
        deltas.update({k: totals[k] for k in kids})
        for m in [name] + kids:
            session.client.rpc(
                {"cmd": "set_matrix", "name": m, "M": deltas[m].tolist()}
            )
        penetration, penetrations = _penetration_report(r, self._settle_support_map())
        return {
            "members": [name] + kids,
            "totals": totals,
            "blend_deltas": deltas,
            "cum_tilt_deg": tilts,
            "disp_mm": disp,
            "rollable": roll,
            "capsized": _capsized([name] + kids, tilts, disp, roll),
            "tilt_deg": r["tilt_deg"],
            "lift_mm": r["lift_mm"],
            "penetration": penetration,
            "penetrations": penetrations,
            "free_probe": free_probe,
            "converged": bool(r.get("converged", True)),
        }

    def _commit_scale(
        self, session, name: str, kids: list, deps_cache: dict | None = None
    ) -> dict:
        """SCALE commit (see ``commit_move``): resize the parent and DROP it to rest on
        its support while lateral contact-deps remain present as colliders, then re-seat
        its true descendants (kept at their ORIGINAL xy — no radial carry), and finally
        run a no-lift reciprocal closure with every affected body independently dynamic.
        One ``scale_resettle`` RPC performs the ordered phases server-side.
        Descendants gate the move (capsize); contact-deps are members (baked by
        accept/reject) but EXCLUDED from the capsize gate — a freed leaner falling once
        its support resized is the expected reaction. The caller re-scores and, for
        scale, gates on ``penetration`` (the stack bedding into its support) since scale
        skips the pre-physics feasibility clamp."""
        W_cand = self._blend_matrices(session)[name]
        C = W_cand @ np.linalg.inv(self._synced[name])
        deps = self._contact_deps(
            name, {name} | set(kids), members=[name] + kids, cache=deps_cache
        )
        free = kids + deps
        # kids ride ON the parent -> temporarily ABSENT during its drop (static cargo
        # would pin the resizing parent); lateral deps are never absent and their
        # colliders constrain the parent before the reciprocal dynamic closure.
        r = self.client.rpc(
            {
                "cmd": "scale_resettle",
                "parent": name,
                "delta": C.tolist(),
                "free": free,
                "absent": kids,
                "budget": "micro",
            }  # fmt: skip
        )
        D = {m: _mat(v) for m, v in r["D"].items()}
        # parent total = settle @ scale; a free body carried no delta (Cm = I)
        Cm = {name: C, **{c: np.eye(4) for c in free}}
        members_all = [name] + free
        totals = {m: D[m] @ Cm[m] for m in members_all}
        deltas = {name: totals[name] @ np.linalg.inv(C)}
        deltas.update({c: totals[c] for c in free})
        for m in members_all:
            session.client.rpc(
                {"cmd": "set_matrix", "name": m, "M": deltas[m].tolist()}
            )
        penetration, penetrations = _penetration_report(r, self._settle_support_map())
        return {
            "members": members_all,
            "totals": totals,
            "blend_deltas": deltas,
            "cum_tilt_deg": dict(r["cum_tilt_deg"]),
            "disp_mm": dict(r.get("disp_mm") or {}),
            "rollable": dict(r.get("rollable") or {}),
            # descendants gate; contact-deps excluded (their fall is expected)
            "capsized": _capsized(
                [name] + kids,
                r["cum_tilt_deg"],
                r.get("disp_mm") or {},
                r.get("rollable") or {},
            ),  # fmt: skip
            "tilt_deg": r["tilt_deg"],
            "lift_mm": r["lift_mm"],
            "penetration": penetration,
            "penetrations": penetrations,
            "converged": bool(r.get("converged", True)),
        }

    def _commit_carry(
        self,
        session,
        name: str,
        kids: list,
        exclude: list,
        deps_cache: dict | None = None,
    ) -> dict:
        """Variant F (and the no-children / rotate_180 path): the weld (rigid) carry —
        members welded into ONE compound and settled together. SCALE no longer reaches
        here (``commit_move`` routes it to ``_commit_scale``), so ``s`` is ~1 by
        construction and there is no scale-reseat branch."""
        members = [name] + kids
        W_cand = self._blend_matrices(session)[name]
        C = self._carry_deltas(name, kids, W_cand)
        free_deps = self._contact_deps(
            name, set(members), members=members, cache=deps_cache
        )
        r = self.client.rpc(
            {
                "cmd": "move_group",
                "members": members,
                "budget": "micro",
                "deltas": {m: C[m].tolist() for m in members},
                **(
                    {"free": free_deps, "sequential": True, "free_static": True}
                    if free_deps
                    else {}
                ),
                **({"exclude": exclude} if exclude else {}),
            }  # fmt: skip
        )
        D = _mat(r["D"])
        totals = {m: D @ C[m] for m in members}
        tilts = dict(r["cum_tilt_deg"])
        disp = dict(r.get("disp_mm") or {})
        roll = dict(r.get("rollable") or {})
        lift, step_tilt = r["lift_mm"], r["tilt_deg"]
        for d in free_deps:  # bake each freed dependent's settle (like _commit_solo)
            fr = (r.get("free") or {}).get(d) or {}
            totals[d] = _mat(fr.get("D") or np.eye(4).tolist())
            tilts[d] = float(fr.get("cum_tilt_deg", 0.0))
            disp[d] = float(fr.get("disp_mm", 0.0))
            roll[d] = bool(fr.get("rollable", False))
        penetration, body_per = _penetration_report(r, self._settle_support_map())
        # freed contact-dependents that actually re-settled become members so
        # accept/reject bake them uniformly.
        settled_deps = [d for d in free_deps if d in totals]
        out_members = members + settled_deps
        # apply rested poses to the blend: the parent already sits at W_cand, so it
        # gets total minus the carry it has; untouched children + freed deps get their full total.
        deltas = {name: totals[name] @ np.linalg.inv(C[name])}
        deltas.update({c: totals[c] for c in kids})
        deltas.update({d: totals[d] for d in settled_deps})
        for m in out_members:
            session.client.rpc(
                {"cmd": "set_matrix", "name": m, "M": deltas[m].tolist()}
            )
        return {
            "members": out_members,
            "totals": {m: totals[m] for m in out_members},
            "blend_deltas": deltas,
            "cum_tilt_deg": tilts,
            "disp_mm": disp,
            "rollable": roll,
            # per-member class-aware stability violations (rollable -> settle
            # displacement, else cumulative tilt); the caller rejects on any. Freed
            # contact-deps are EXCLUDED: a leaner falling once its support moved is the
            # expected reaction, not a failure of this move.
            "capsized": _capsized(members, tilts, disp, roll),
            "tilt_deg": step_tilt,  # THIS settle's tilt (the flip gate)
            "lift_mm": lift,
            # worst NEW interpenetration the settled winner leaves (None = clean);
            # the caller rejects on it — see PEN_CAP_MM/PEN_NEW_MM. With body_pen
            # this covers EVERY settled body (incl. freed deps), each vs the scene.
            "penetration": penetration,
            "penetrations": body_per,
            "converged": bool(r.get("converged", True)),
        }

    def _commit_joint(
        self, session, moved: list[str], deps_cache: dict | None = None
    ) -> dict:
        ""
        W = self._blend_matrices(session)
        C: dict[str, np.ndarray] = {}
        groups: list[list[str]] = []  # one hierarchy per moved object (server lifts each alone)
        for name in moved:  # parents first: a moved descendant overrides the carry
            kids = [k for k in self._descendants(name) if k not in moved]
            C.update(self._carry_deltas(name, kids, W[name]))
            groups.append([name] + kids)
        bodies = [n for n in self._names if n in C]  # stable scene order
        deps: list[str] = []
        for name in moved:
            hier = [name] + [k for k in self._descendants(name) if k not in moved]
            for d in self._contact_deps(
                name, set(bodies), members=hier, cache=deps_cache
            ):
                if d not in deps and d not in bodies:
                    deps.append(d)
        r = self.client.rpc(
            {
                "cmd": "settle_set",
                "bodies": bodies,
                "deltas": {b: C[b].tolist() for b in bodies},
                "groups": groups,
                "budget": "micro",
                **({"free": deps} if deps else {}),
            }  # fmt: skip
        )
        D = {n: _mat(M) for n, M in r["D"].items()}
        totals = {b: D[b] @ C[b] for b in bodies}
        settled_deps = [d for d in deps if d in D]
        totals.update({d: D[d] for d in settled_deps})
        out_members = bodies + settled_deps
        # blend: a moved object already sits at its requested pose -> only the settle
        # delta; an unmoved rider or freed dependent still sits at synced -> full total
        deltas = {
            n: (totals[n] @ np.linalg.inv(C[n])) if n in moved else totals[n]
            for n in out_members
        }
        for m in out_members:
            session.client.rpc(
                {"cmd": "set_matrix", "name": m, "M": deltas[m].tolist()}
            )
        tilts = {n: float(v) for n, v in r["cum_tilt_deg"].items()}
        disp = {n: float(v) for n, v in (r.get("disp_mm") or {}).items()}
        roll = {n: bool(v) for n, v in (r.get("rollable") or {}).items()}
        penetration, body_per = _penetration_report(r, self._settle_support_map())
        return {
            "members": out_members,
            "bodies": list(bodies),  # the moved hierarchies; members minus these = deps
            "totals": {m: totals[m] for m in out_members},
            "blend_deltas": deltas,
            "cum_tilt_deg": tilts,
            "disp_mm": disp,
            "rollable": roll,
            # riders of a moved object count; freed dependents are the expected reaction
            "capsized": _capsized(bodies, tilts, disp, roll),
            "tilt_deg": max((float(v) for v in r["tilt_deg"].values()), default=0.0),
            "lift_mm": float(r["lift_mm"]),
            "penetration": penetration,
            "penetrations": body_per,
            "converged": bool(r.get("converged", True)),
            "unconverged_bodies": list(r.get("unconverged_bodies") or []),
            "joint": True,
        }

    def accept(self, session, commit: dict) -> None:
        """Rebase each member's set_pose base onto the rested pose, mark synced, and
        check the Isaac<->blend pose invariant for the touched members (auto-resync
        on drift — see :meth:`verify_sync`)."""
        eye = np.eye(4).tolist()
        for m in commit["members"]:
            session.client.rpc(
                {"cmd": "set_matrix", "name": m, "M": eye, "rebase": True}
            )
        self._synced = self._blend_matrices(session)
        self.verify_sync(session, commit["members"])

    def reject(self, session, commit: dict) -> None:
        """Exact inverse on both sides — no half-transformed scene."""
        for m in reversed(commit["members"]):
            self.client.rpc(
                {
                    "cmd": "transform",
                    "name": m,
                    "M": np.linalg.inv(commit["totals"][m]).tolist(),
                }  # fmt: skip
            )
            session.client.rpc(
                {
                    "cmd": "set_matrix",
                    "name": m,
                    "M": np.linalg.inv(commit["blend_deltas"][m]).tolist(),
                }  # fmt: skip
            )

    def recover_after_error(self, session) -> list[str]:
        """Best-effort fail-closed recovery to the last accepted pose snapshot.

        An RPC may fail after ``commit_move`` has already transformed one or more
        Blend handles. GPT-6 strict mode must not reinterpret that half-commit as a
        kinematic success. Restore every known Blend handle from ``_synced`` using
        only the register server; then, when Isaac is still responsive, restore its
        cumulative body transforms to the same snapshot. Returns recovery errors for
        diagnostics; callers still reject the edit even when recovery succeeds.
        """
        errors: list[str] = []
        desired = {name: matrix.copy() for name, matrix in self._synced.items()}
        try:
            current = self._blend_matrices(session)
        except Exception as exc:  # noqa: BLE001 - preserve the primary physics error
            current = {}
            errors.append(f"Blend recovery snapshot failed: {exc}")
        for name, target in desired.items():
            if name not in current:
                continue
            try:
                delta = target @ np.linalg.inv(current[name])
                session.client.rpc(
                    {"cmd": "set_matrix", "name": name, "M": delta.tolist()}
                )
                session.client.rpc(
                    {
                        "cmd": "set_matrix",
                        "name": name,
                        "M": np.eye(4).tolist(),
                        "rebase": True,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - continue restoring other handles
                errors.append(f"Blend recovery failed for {name}: {exc}")

        boot = getattr(self, "_boot", None) or {}
        for name, target in desired.items():
            if name not in boot:
                continue
            try:
                total = _mat(self.client.rpc({"cmd": "pose", "name": name})["total"])
                expected = target @ np.linalg.inv(boot[name])
                self.client.rpc(
                    {
                        "cmd": "transform",
                        "name": name,
                        "M": (expected @ np.linalg.inv(total)).tolist(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - server may be the failure source
                errors.append(f"Isaac recovery failed for {name}: {exc}")
                break  # a dead RPC transport must not cost one timeout per object
        return errors

    def restore_blend(self, session, name: str) -> None:
        """Set the blend handle ``name`` back to its EXACT synced (pre-move) world
        matrix via ``set_matrix`` (a world-delta premultiply, no BVH). This is the
        reject/clamp/gain-fail restore: a kinematic ``set_pose`` restore is NOT identity
        — it re-runs ``_resolve_penetration`` (which separates only from the object's
        SUPPORT, never a lateral sibling) + the all-scene ``_seat_z`` (deadbanded, so a
        rested pose is a no-op — but a tilted rest expressed as (t,euler,s) still isn't
        reachable), and it swaps a physics-rested attitude for a kinematic one (0717
        knife-into-spoon). Isaac is already at the
        synced pose (``reject`` restored carried members; a clamp/gain-fail never moved
        it), so only the blend handle needs fixing."""
        cur = self._blend_matrices(session)[name]
        session.client.rpc(
            {
                "cmd": "set_matrix",
                "name": name,
                "M": (self._synced[name] @ np.linalg.inv(cur)).tolist(),
            }  # fmt: skip
        )

    # -- freeform-edit settle (execute_and_evaluate escape hatch) ------------------- #
    def _ancestors(self, name: str) -> list[str]:
        """Support chain of ``name`` (nearest first) via the carry-parent map."""
        out, cur = [], self.parents.get(name)
        while cur is not None and cur not in out:
            out.append(cur)
            cur = self.parents.get(cur)
        return out

    def settle_edited(
        self,
        session,
        intended_resting_modes: dict[str, str | dict] | None = None,
        *,
        force_rebuild_names: list[str] | None = None,
    ) -> list[dict]:
        """Physics-resolve a freeform ``execute_and_evaluate`` RELOCATION. Diffs the
        edited blend against the last-synced pose to find what MOVED, then runs the
        winner commit (:meth:`commit_move`) on each — consuming the raw teleport AS the
        move, so it gets the SAME lift-to-clear + settle + carry the ``move`` tool gives:
        the lift clears interpenetration the teleport created, the settle rests it, and
        descendants ride along. Supports settle before their moved dependents. Each is
        In the baseline profile each result is ACCEPTED (kept), preserving the existing
        keep/undo workflow. Under ``gpt6_v1`` + ``strict_post_edit_physics``, convergence,
        new/worsened penetration, capsize, and the optional intended resting mode are
        gates: the current commit is rejected and :class:`PhysicsSettlementRejected`
        tells the executor to roll back the outer edit transaction. A dict intent may
        additionally name an exact ``required_support``; after the other gates pass,
        a fresh ``supports_of`` result must prove that definite support before accept.
        Query/protocol failures raise :class:`PhysicsSupportQueryFailed` so the
        executor treats them as infrastructure failures rather than bad physics.
        Returns one report dict per settled object; ``[]`` for a no-op / non-relocation
        edit.

        Mesh-DATA edits are absorbed first: an affine resize folds into the pose
        diff (the commit below then routes it through ``_commit_scale``, which
        rescales the collider parts and re-seats cargo); a rebuilt body's matrix
        diff is identity, so it is force-included and its commit is the plain
        lift-to-clear + settle at the new geometry.

        GPT-6 mesh transactions explicitly pass ``force_rebuild_names``: replacing
        a model must discard its old collider and stabilization even if geometry
        signatures happen to be identical or explainable by an affine transform.
        Ordinary callers retain the existing heuristic and simulation ordering.
        """
        strict = getattr(session, "harness_profile", "baseline") == "gpt6_v1" and bool(
            getattr(session, "strict_post_edit_physics", False)
        )
        intents = intended_resting_modes or {}
        mesh_to_oid = {
            value.get("mesh_name"): object_id
            for object_id, value in (getattr(session, "ctx", None) or {}).items()
        }
        forced = set(force_rebuild_names or [])
        if forced:
            if not strict:
                raise ValueError("forced mesh rebuild requires strict GPT-6 physics")
            unknown = forced.difference(self._names)
            if unknown:
                raise ValueError(
                    f"forced mesh rebuild names are unknown: {sorted(unknown)!r}"
                )
            current_rollable = _rollable_flags(self.work)
            for name in sorted(forced):
                for attr in ("_records", "_collider_bindings", "_rollable"):
                    cache = getattr(self, attr, None)
                    if cache is not None:
                        cache.pop(name, None)
                if name in current_rollable:
                    if getattr(self, "_rollable", None) is None:
                        self._rollable = {}
                    self._rollable[name] = current_rollable[name]
                self._rebuild_body(session, name)
            signatures = self._geom_sigs(session)
            if getattr(self, "_geom", None) is not None:
                for name in forced:
                    self._geom.pop(name, None)
                    if name in signatures:
                        self._geom[name] = signatures[name]
        # Blend matrices are read ONCE here: absorbing mesh-data edits below changes the
        # mirror baselines, never the blend, so the same read serves the moved-set
        # computation (and fixtures that meter _blend_matrices calls).
        cur = self._blend_matrices(session)
        rebuilt = forced | set(self._absorb_geom_edits(session))
        moved = [
            n for n in self._names
            if n in rebuilt or _really_moved(cur[n] @ np.linalg.inv(self._synced[n]))
        ]  # fmt: skip
        moved.sort(
            key=lambda n: len(set(self._ancestors(n)) & set(moved))
        )  # parents 1st
        pre = {n: self._synced[n].copy() for n in moved}
        pending, out = set(moved), []
        joint = None
        if strict and len(moved) > 1 and not any(
            abs(_rigid_of(cur[n] @ np.linalg.inv(self._synced[n]))[1] - 1.0) >= 1e-3
            for n in moved
        ):
            joint = self._commit_joint(session, moved)
            settled_all = self._blend_matrices(session)
        for name in moved:
            if name not in pending:  # already carried along by a settled parent
                continue
            if joint is not None:
                commit = _joint_view(
                    joint,
                    [name]
                    + [
                        k for k in self._descendants(name)
                        if k in joint["members"] and k not in moved
                    ],
                )  # fmt: skip
                settled = settled_all.get(name)
            else:
                commit = self.commit_move(
                    session, name
                )  # freeform edit consumed as the move
                settled = self._blend_matrices(session).get(name) if strict else None
            intent = intents.get(name, intents.get(mesh_to_oid.get(name)))
            if intent is None:
                # freeform code edit: physics owns the final pose, and the drift from
                # the requested pose is REPORTED (requested_to_settled), not capped —
                # convergence, penetration and rider/neighbor capsize still gate.
                intent = "free"
            tilt = max(commit["cum_tilt_deg"].values(), default=0.0)
            report = {
                "name": name,
                "members": commit["members"],
                "tilt_deg": round(tilt, 1),
                "lift_mm": round(commit["lift_mm"], 1),
                "capsized": commit.get("capsized") or [],
                "penetration": commit.get("penetration"),
                # per-body wedges (incl. freed deps) + sim convergence — the
                # freeform note names each wedged body honestly
                "penetrations": commit.get("penetrations") or {},
                "converged": bool(commit.get("converged", True)),
            }
            if strict and cur.get(name) is not None and settled is not None:
                report.update(
                    {
                        "requested_matrix": cur[name].tolist(),
                        "settled_matrix": settled.tolist(),
                        "requested_to_settled": _pose_drift(cur[name], settled),
                        "intended_resting_mode": (
                            intent.get("mode", "preserve")
                            if isinstance(intent, dict)
                            else intent or "preserve"
                        ),
                    }
                )
            if name in forced:
                report["forced_mesh_rebuild"] = True
            if joint is not None:
                report["joint_settle"] = True
            rejection = (
                _strict_settlement_rejection(
                    name,
                    commit,
                    intent,
                    requested=cur.get(name),
                    settled=settled,
                )
                if strict
                else None
            )
            rejection_code = None
            if (
                strict
                and rejection is None
                and isinstance(intent, dict)
                and "required_support" in intent
            ):
                required = intent["required_support"]
                invalid_required = None
                if (
                    not isinstance(required, str)
                    or not required
                    or required != required.strip()
                ):
                    invalid_required = (
                        "required_support must be an exact non-empty object name"
                    )
                elif required == name:
                    invalid_required = "required_support cannot name the object itself"

                if invalid_required is not None:
                    rejection = (
                        f"invalid required_support for {name}: {invalid_required}"
                    )
                    rejection_code = "invalid_required_support"
                    report["required_support_postcondition"] = {
                        "required_support": (
                            required if isinstance(required, str) else None
                        ),
                        "requested_value_repr": repr(required),
                        "verified": False,
                        "satisfied": False,
                        "observed_supports": [],
                        "definite_supports": [],
                        "validation_error": invalid_required,
                    }
                else:
                    try:
                        support_records = _validated_support_records(
                            self.client.rpc({"cmd": "supports_of", "name": name}),
                            subject=name,
                        )
                        definite_supports = sorted(
                            _protected_support_names(support_records)
                        )
                    except Exception as exc:  # noqa: BLE001 - strict proof fails closed
                        query_reason = (
                            f"could not verify required support {required!r} for "
                            f"{name}: {exc}"
                        )
                        report.update(
                            {
                                "accepted": False,
                                "rejection_code": "required_support_query_failed",
                                "rejection_reason": query_reason,
                                "required_support_postcondition": {
                                    "required_support": required,
                                    "verified": False,
                                    "satisfied": None,
                                    "observed_supports": [],
                                    "definite_supports": [],
                                    "query_error": str(exc),
                                    "query_error_type": type(exc).__name__,
                                },
                            }
                        )
                        try:
                            self.reject(session, joint or commit)
                        except Exception as rollback_exc:  # noqa: BLE001
                            report["required_support_postcondition"][
                                "rollback_error"
                            ] = str(rollback_exc)
                        raise PhysicsSupportQueryFailed(
                            query_reason, reports=[*out, report]
                        ) from exc

                    support_evidence = {
                        "required_support": required,
                        "verified": True,
                        "satisfied": required in definite_supports,
                        "observed_supports": support_records,
                        "definite_supports": definite_supports,
                    }
                    report["required_support_postcondition"] = support_evidence
                    if not support_evidence["satisfied"]:
                        rejection = (
                            f"{name} did not remain on required support "
                            f"{required!r} after settlement"
                        )
                        rejection_code = "required_support_missing"
            if rejection is not None:
                report.update(
                    {
                        "accepted": False,
                        "rejection_code": (
                            rejection_code or "strict_settlement_rejected"
                        ),
                        "rejection_reason": rejection,
                    }
                )
                self.reject(session, joint or commit)
                raise PhysicsSettlementRejected(rejection, reports=[*out, report])
            if joint is None:
                self.accept(session, commit)  # keep; agent decides keep/undo
            for m in commit["members"]:
                pending.discard(m)
            # accept() re-reads ALL _synced, but Isaac only moved this commit's members;
            # restore pre-edit _synced for the still-pending moved objects so the next
            # commit still sees their edit as the move (Isaac↔synced stay consistent).
            for n in pending:
                self._synced[n] = pre[n]
            out.append(report)
        if joint is not None:
            self.accept(session, joint)  # every gate passed: bake all hierarchies
        return out

    # -- rules gate ----------------------------------------------------------------- #
    def contact_pairs(self) -> list[dict]:
        """Overlap pairs at current poses in the penetration-dump schema. Depth (m)
        is the min single-direction push (+-x/+-y/+z) to separate the pair — see
        cmd_contacts — not the settle z-lift."""
        r = self.client.rpc({"cmd": "contacts"})
        return [
            {"a": p["a"], "b": p["b"], "depth": p["depth_mm"] / 1000.0}
            for p in r["pairs"]
        ]

    def close(self) -> None:
        try:
            self.client.disconnect()  # shared server stays warm for certify
        except Exception:  # noqa: BLE001
            pass


def delivered_vs_placed(
    records: dict, composition_deltas: dict, centroids: dict, rollable: dict
) -> dict:
    """How far each object's DELIVERED pose sits from the pose the photo put it in.

    The per-stage drifts each look small and none of them answers this: the ladder
    reports its own rungs, the preprocess certify reports drift from the settled pose,
    and the composition certify reports drift from the COMPOSED pose — so an object that
    capsized at preprocess certify is measured from its capsized pose thereafter and
    reads clean forever (0725_snapdown2 vase: 82 deg / 1063 mm at preprocess certify,
    then 0.0 deg / 0.1 mm at composition certify, `fell` false — invisible everywhere).

    Composition of world-frame deltas, placed -> delivered::

        T = composition_delta @ settle_total

    ``settle_total`` is the settle server's cumulative matrix (ladder + ICP + re-drop +
    preprocess certify, from the placed pose). Composition deltas must describe actual
    WORLD GEOMETRY motion, not ``M_final @ inverse(M_import)``: Blender's origin_set
    changes both the local vertex frame and matrix_world without moving the geometry.
    Displacement uses the final vertex centroid, matching ``_com_drift`` semantics.

    Args:
        records: Preprocessing records, keyed by object name.
        composition_deltas: World transforms from settled GLBs to delivered geometry.
        centroids: Delivered, post-certification world vertex means.
        rollable: Class-aware capsize exemptions, keyed by object name.

    Returns:
        Cumulative displacement, tilt, and capsize reports for measurable objects.
        Missing inputs are omitted rather than replaced by an unsafe matrix baseline.
    """

    out = {}
    for name, rec in records.items():
        S, D = rec.get("settle_total"), composition_deltas.get(name)
        c1 = centroids.get(name)
        if S is None or D is None or c1 is None:
            continue
        T = np.asarray(D, float) @ np.asarray(S, float)
        R = T[:3, :3]
        s = float(
            np.cbrt(max(np.linalg.det(R), 1e-12))
        )  # ICP + agent resizes ride here
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2]) / s))))
        c0 = np.linalg.inv(T) @ np.array([*c1, 1.0])  # the placed centroid
        d = np.asarray(c1, float) - c0[:3]
        dxy = float(np.linalg.norm(d[:2]))
        out[name] = {
            "dxy": dxy,
            "dz": float(d[2]),
            "tilt_deg": tilt,
            "capsized": capsized(tilt, dxy * 1000.0, bool(rollable.get(name))),
        }
    return out


def _resolve_certification_artifact(scene: Path, value: object) -> Path:
    """Resolve a validated inventory path without trusting the process cwd."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("runtime revision has no nonempty mesh_glb")
    raw = Path(value)
    candidates = (
        [raw.resolve()]
        if raw.is_absolute()
        else [
            (scene / raw).resolve(),
            (Path(__file__).resolve().parents[3] / raw).resolve(),
            raw.resolve(),
        ]
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise ValueError(f"runtime revision mesh is missing: {candidates[0]}")
    return found


def _gpt6_certification_contract(
    scene: Path, requested_names: list[str]
) -> tuple[list[str], dict[str, str], dict[str, dict]]:
    """Resolve final dynamic names and authored revision references.

    The Blender property is only a representation check. Authorization comes from
    the already validated committed runtime overlay and its current revision.
    """
    from lib.tools.geometry.inventory_contract import validate_scene_artifacts
    from lib.tools.geometry.runtime_object_repair import (
        load_runtime_inventory,
        procedural_capture_of,
    )

    report = validate_scene_artifacts(
        scene,
        allow_runtime_additions=True,
        allow_runtime_inventory=True,
        validate_runtime_physics=True,
    )
    by_id = dict(report["object_mesh_names"])
    expected = sorted(by_id.values())
    if expected != sorted(requested_names):
        raise RuntimeError(
            "GPT-6 certification graph/placement inventory differs from requested "
            f"objects: expected={expected!r}, requested={sorted(requested_names)!r}"
        )
    placement = json.loads((scene / "placement.json").read_text())
    procedural = {
        row["mesh_name"]: row
        for row in placement.get("objects", [])
        if isinstance(row, dict)
        and isinstance(row.get("procedural_capture"), dict)
        and row["procedural_capture"].get("source_schema")
        == "gpt6_blender_empty_tree_v1"
    }
    inventory = load_runtime_inventory(scene) or {"objects": []}
    authored_ids: dict[str, str] = {}
    references: dict[str, dict] = {}
    for record in inventory.get("objects", []):
        # authored geometry = maskless initializer additions OR a composition mesh
        # replacement that kept its photo mask (tracked) — both are Empty+parts captures
        if record.get("mask_policy") != "excluded" and procedural_capture_of(record) is None:
            continue
        object_id = str(record["id"])
        snapshot = record["placement"]
        mesh_name = str(snapshot["mesh_name"])
        if by_id.get(object_id) != mesh_name or mesh_name not in procedural:
            raise RuntimeError(
                f"authored runtime object {object_id!r} has no exact procedural "
                "graph/placement binding"
            )
        capture = snapshot["procedural_capture"]
        if (
            capture.get("root_object_type") != "EMPTY"
            or capture.get("source_schema") != "gpt6_blender_empty_tree_v1"
            or capture.get("roundtrip_verified") is not True
        ):
            raise RuntimeError(
                f"authored runtime object {object_id!r} has invalid capture evidence"
            )
        current_number = record["current_mesh_revision"]
        matches = [
            revision
            for revision in record.get("mesh_revisions", [])
            if isinstance(revision, dict)
            and revision.get("revision") == current_number
            and revision.get("status") == "committed"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"authored runtime object {object_id!r} has no unique current revision"
            )
        revision = matches[0]
        if revision.get("mesh_glb") != snapshot.get("mesh_glb"):
            raise RuntimeError(
                f"authored runtime object {object_id!r} placement is not its current revision"
            )
        mesh_path = _resolve_certification_artifact(scene, revision.get("mesh_glb"))
        transaction_id = revision.get("transaction_id")
        placement_transaction_id = snapshot.get("runtime_mesh_revision_transaction_id")
        record_transaction_id = record.get("transaction_id")
        # Initial additions bind their first mesh to the creation transaction.
        # Only subsequent mesh edits write the revision-specific placement key.
        # Never fall back from an explicitly present but invalid revision binding.
        if (
            "runtime_mesh_revision_transaction_id" not in snapshot
            and record.get("origin") == "runtime_added"
            and snapshot.get("runtime_added") is True
            and current_number == 1
            and snapshot.get("runtime_transaction_id") == record_transaction_id
        ):
            placement_transaction_id = snapshot.get("runtime_transaction_id")
        if (
            not isinstance(current_number, int)
            or isinstance(current_number, bool)
            or current_number < 1
            or not isinstance(transaction_id, int)
            or isinstance(transaction_id, bool)
            or transaction_id < 1
            or not isinstance(placement_transaction_id, int)
            or isinstance(placement_transaction_id, bool)
            or placement_transaction_id != transaction_id
            or not isinstance(record_transaction_id, int)
            or isinstance(record_transaction_id, bool)
            or record_transaction_id < 1
        ):
            raise RuntimeError(
                f"authored runtime object {object_id!r} has a stale revision transaction"
            )
        authored_ids[mesh_name] = object_id
        references[mesh_name] = {
            "object_id": object_id,
            "mesh_name": mesh_name,
            "mesh_glb": str(mesh_path),
            "sha256": revision.get("sha256"),
            "revision_sha256": revision.get("sha256"),
            "capture_glb_sha256": capture.get("glb_sha256"),
            "reference_pose": "authored_revision",
            "revision": int(current_number),
            "current_revision": int(current_number),
            "transaction_id": transaction_id,
            "placement_transaction_id": placement_transaction_id,
            "record_transaction_id": record_transaction_id,
            "capture_root_matrix": capture.get("root_matrix_world"),
        }
    if set(procedural) != set(authored_ids):
        raise RuntimeError(
            "procedural placement objects are not exactly bound by the active "
            f"runtime inventory: placement={sorted(procedural)!r}, "
            f"inventory={sorted(authored_ids)!r}"
        )
    return expected, authored_ids, references


def delivered_from_meshes(
    moge_dir: Path,
    records: dict,
    paths: dict[str, str],
    rollable: dict,
    *,
    baked_deltas: dict | None = None,
    authored_references: dict[str, dict] | None = None,
    diagnostics: dict[str, dict] | None = None,
    pose_exclusions: dict[str, dict] | None = None,
) -> dict:
    """Measure placed-to-delivered motion without depending on Blender object origins.

    Recover the actual composition similarity from corresponding triangle corners of
    the settled ``*_pm.glb`` and the delivered world mesh. This is the same geometry
    correspondence used for validated-collider reuse, with a tighter 10-micrometer
    residual tolerance for telemetry. It also handles origin changes and transforms
    baked into mesh data. Missing or changed geometry is explicitly unmeasurable.

    Args:
        moge_dir: Scene directory containing placement.json and settled meshes.
        records: Preprocessing physics records with cumulative settle_total matrices.
        paths: World-mesh NPZ dumps keyed by object name (possibly multiple parts).
        rollable: Class-aware capsize exemptions.
        baked_deltas: Only the final-certification world deltas actually baked into
            Blender. Applied to both corners and centroids of pre-certification dumps.
            Omit when paths already describe the delivered scene, e.g. for backfills.
        pose_exclusions: Validated committed initializer edits, keyed by active mesh
            name. Excludes only reference-to-delivered measurements, never simulation
            or final-settle drift. Requires diagnostics to retain explicit exclusions.

    Returns:
        Reports for verified geometry correspondences. Unmeasurable objects emit a
        warning and are omitted; this reporting path never changes physics or poses.
        Policy-excluded objects are labeled in diagnostics, not reported as measured
        and not given fabricated zero displacement or tilt.
    """
    exclusions = pose_exclusions or {}
    if exclusions and diagnostics is None:
        raise ValueError("initializer pose exclusions require explicit diagnostics")
    for name, exclusion in exclusions.items():
        if (
            name not in paths
            or not isinstance(exclusion, dict)
            or exclusion.get("reason_code") != "initializer_edited"
            or exclusion.get("stage") != "initializer"
        ):
            raise ValueError(f"invalid initializer pose exclusion for {name!r}")
    try:
        placement = json.loads((moge_dir / "placement.json").read_text())["objects"]
        by_name = {row["mesh_name"]: row for row in placement}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.warn(
            f"Delivered pose telemetry unavailable for {moge_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    deltas, centroids = {}, {}
    enhanced_telemetry = authored_references is not None or diagnostics is not None
    reference_meta: dict[str, dict] = {}
    for name, path in paths.items():
        if name in exclusions:
            # Eligibility was established from committed initializer history. These
            # bodies still participate in both physics passes; only the comparison
            # against an intentionally superseded reference is out of scope.
            diagnostics[name] = {**exclusions[name], "status": "excluded"}
            continue
        if (records.get(name) or {}).get("settle_total") is None:
            if diagnostics is not None:
                diagnostics[name] = {
                    "status": "unmeasurable",
                    "reason": "physics record has no settle_total",
                }
            continue
        reference_pose = "source_placed"
        reference_digest = None
        try:
            authored = (
                authored_references.get(name)
                if authored_references is not None
                else None
            )
            if authored is not None:
                reference_pose = "authored_revision"
                pm = Path(authored["mesh_glb"])
                reference_digest = _sha256_file(pm)
                if reference_digest != authored.get("sha256"):
                    raise ValueError("authorized authored revision GLB changed")
            else:
                pm = Path(by_name[name]["mesh_glb"])
                if not pm.name.endswith("_pm.glb"):
                    raise ValueError(f"expected a settled *_pm.glb baseline, got {pm}")
                if not pm.is_file():
                    # Staged runs can retain the source run's path in placement.json.
                    pm = moge_dir / "meshes" / pm.name
            source = glb_face_corners(str(pm))
            with np.load(path) as dump:
                parts = [
                    np.asarray(dump[f"v{i}"], float) for i in range(int(dump["n"]))
                ]
                corners = np.vstack(
                    [
                        vertices[np.asarray(dump[f"f{i}"], int)].reshape(-1, 3)
                        for i, vertices in enumerate(parts)
                    ]
                )
                vertices = np.vstack(parts)
            bake = np.asarray((baked_deltas or {}).get(name, np.eye(4)), float)
            vertices = vertices @ bake[:3, :3].T + bake[:3, 3]
            correspondence_meta = {}
            if authored is not None:
                if not all(
                    np.isfinite(value).all() for value in (source, corners, vertices)
                ):
                    raise ValueError("non-finite geometry")
                with np.errstate(over="ignore", invalid="ignore"):
                    authored_centroid = vertices.mean(0)
                if not np.isfinite(authored_centroid).all():
                    raise ValueError("non-finite delivered centroid")
                from lib.tools.geometry.authored_delivery import (
                    measure_authored_delivery,
                )

                measured = measure_authored_delivery(
                    object_name=name,
                    binding=authored,
                    actual_reference_sha256=reference_digest,
                    reference_triangles=source,
                    current_triangles=corners,
                    capture_root_matrix=authored["capture_root_matrix"],
                    current_root_matrix=authored["current_root_matrix"],
                    baked_delta=bake,
                )
                delta = np.asarray(measured["delta"], dtype=float)
                correspondence_meta = {
                    "triangle_count": measured["triangle_count"],
                    "max_corner_error_m": measured["max_corner_error_m"],
                    "matching_path": measured["matching_path"],
                    "fallback_triangle_count": measured["fallback_triangle_count"],
                    "fallback_candidate_edges": measured["fallback_candidate_edges"],
                    "pose_uniform_scale": measured["pose_uniform_scale"],
                    "baked_uniform_scale": measured["baked_uniform_scale"],
                    "transform_source": "authenticated_empty_root_matrices",
                }
            else:
                # Preserve the historical source-placed path exactly.
                corners = corners @ bake[:3, :3].T + bake[:3, 3]
                if not all(np.isfinite(v).all() for v in (source, corners, vertices)):
                    raise ValueError("non-finite geometry")
                if np.linalg.matrix_rank(source - source.mean(0)) < 2:
                    raise ValueError("degenerate geometry cannot establish orientation")
                similarity = corner_similarity(source, corners, tol=1e-5)
                if similarity is None:
                    raise ValueError(
                        "settled-to-delivered geometry correspondence failed"
                    )
                scale, rotation, translation = similarity
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"invalid geometry scale {scale=}")
                delta = np.eye(4)
                delta[:3, :3], delta[:3, 3] = scale * rotation, translation
            deltas[name], centroids[name] = (
                delta,
                authored_centroid if authored is not None else vertices.mean(0),
            )
            if enhanced_telemetry:
                if reference_digest is None:
                    reference_digest = _sha256_file(pm)
                reference_meta[name] = {
                    "status": "measured",
                    "reference_pose": reference_pose,
                    "reference_mesh_sha256": reference_digest,
                } | correspondence_meta
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            np.linalg.LinAlgError,
        ) as exc:
            if diagnostics is not None:
                diagnostics[name] = {
                    "status": "unmeasurable",
                    "reference_pose": reference_pose,
                    "reason": str(exc),
                }
                reason_code = getattr(exc, "code", None)
                if reference_pose == "authored_revision" and isinstance(
                    reason_code, str
                ):
                    diagnostics[name]["reason_code"] = reason_code
            warnings.warn(
                f"Delivered pose telemetry unavailable for {name!r} in {moge_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    reports = delivered_vs_placed(records, deltas, centroids, rollable)
    if not enhanced_telemetry:
        return reports
    for name, meta in reference_meta.items():
        if name not in reports:
            if diagnostics is not None:
                diagnostics[name] = {
                    "status": "unmeasurable",
                    "reference_pose": meta["reference_pose"],
                    "reason": "placed-to-delivered report could not be derived",
                }
            continue
        if diagnostics is not None:
            diagnostics[name] = dict(meta)
        if meta["reference_pose"] == "authored_revision":
            reports[name]["reference_pose"] = "authored_revision"
            reports[name]["reference_mesh_sha256"] = meta["reference_mesh_sha256"]
    return reports


def _require_exact_names(label: str, actual, expected) -> None:
    values = list(actual) if isinstance(actual, (dict, list, tuple, set)) else None
    if (
        values is None
        or len(values) != len(set(values))
        or set(values) != set(expected)
    ):
        raise RuntimeError(
            f"{label} coverage mismatch: expected={sorted(expected)!r}, actual={values!r}"
        )


def _validate_velocity_rest(response: dict, names: list[str]) -> dict:
    """Authenticate the requested rest criterion before trusting a warm server."""
    from isaac.physics_config import (
        ANGULAR_REST_SPEED_DEG_S,
        EXPORT_VELOCITY_REST_POLICY,
        LINEAR_REST_SPEED_M_S,
    )

    rest = response.get("rest")
    limits = {
        "linear_velocity_m_s": LINEAR_REST_SPEED_M_S,
        "angular_velocity_deg_s": ANGULAR_REST_SPEED_DEG_S,
    }
    if (
        response.get("rest_policy") != EXPORT_VELOCITY_REST_POLICY
        or not isinstance(rest, dict)
        or rest.get("thresholds") != limits
        or not isinstance(response.get("converged"), bool)
    ):
        raise RuntimeError(
            "Isaac did not acknowledge the requested export velocity rest criterion"
        )
    if not isinstance(rest.get("objects"), dict):
        raise RuntimeError("Isaac rest evidence is not a mapping")
    _require_exact_names("Isaac rest evidence", rest.get("objects"), names)
    for name, row in rest["objects"].items():
        if not isinstance(row, dict):
            raise RuntimeError(f"Isaac rest evidence for {name!r} is malformed")
        if response["converged"] and (
            row.get("valid") is not True
            or row.get("below_limits") is not True
            or any(
                type(row.get(key)) not in (int, float)
                or not math.isfinite(row[key])
                or not 0 <= row[key] < limit
                for key, limit in limits.items()
            )
        ):
            raise RuntimeError(
                f"Isaac claimed convergence with invalid/moving body {name!r}"
            )
    if response["converged"] and (
        rest.get("status") != "passed" or rest.get("offending_objects") != []
    ):
        raise RuntimeError("Isaac convergence contradicts its rest report")
    return rest


def _validate_joint_certification(
    response: dict, names: list[str], surfaces: list[str]
) -> dict:
    """Shared coverage/finite-pose contract for initializer and final joint bakes."""
    if (
        response.get("ok") is not True
        or response.get("support_policy") != "actual_surfaces_v1"
        or response.get("support_proxy") is not False
    ):
        raise RuntimeError(
            "Isaac did not acknowledge actual_surfaces_v1 certification policy"
        )
    _require_exact_names(
        "Isaac actual-support static", response.get("surfaces"), surfaces
    )
    for key in ("total", "drift"):
        if not isinstance(response.get(key), dict):
            raise RuntimeError(f"Isaac certification {key} is not a mapping")
    _require_exact_names(
        "Isaac certification total", response.get("total"), names + surfaces
    )
    _require_exact_names("Isaac certification drift", response.get("drift"), names)
    for name, row in response["drift"].items():
        if not isinstance(row, dict):
            raise RuntimeError(
                f"Isaac certification drift for {name!r} is not an object"
            )
        for key in ("dxy", "dz", "tilt_deg"):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                raise RuntimeError(
                    f"Isaac certification drift for {name!r} has invalid finite numeric non-boolean {key}"
                )
    for name, total in response["total"].items():
        try:
            matrix = np.asarray(total, dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Isaac certification total for {name!r} is not a finite 4x4 matrix"
            ) from exc
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise RuntimeError(
                f"Isaac certification total for {name!r} is not a finite 4x4 matrix"
            )
        if name in surfaces and not np.allclose(matrix, np.eye(4), atol=1e-9, rtol=0.0):
            raise RuntimeError(f"Isaac certification changed static total for {name!r}")
    return _validate_velocity_rest(response, names)


def certify_composed_scene(
    moge_dir: str,
    blend: str,
    blender_cmd: str,
    isaac_python: str | None = None,
    *,
    runtime_inventory_profile: bool = False,
) -> dict:
    ""
    from lib.tools.geometry.physics import (
        DEFAULT_ISAAC_PYTHON,
        SettleClient,
        merge_pose_changes,
        run_id_for,
    )
    from lib.tools.geometry.register import RenderClient, _cat_inst, mesh_name_for

    md = Path(moge_dir)
    graph = json.load(open(md / "scene_graph.json"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    # instance comes from the node ID (`cat#N`) — the field itself can be None
    names = sorted(
        {
            mesh_name_for(*_cat_inst(n))
            for n in nodes.values()
            if n.get("kind") != "root_surface"
        }
    )
    authored_ids: dict[str, str] = {}
    authored_references: dict[str, dict] = {}
    pose_exclusions: dict[str, dict] = {}
    if runtime_inventory_profile:
        expected, authored_ids, authored_references = _gpt6_certification_contract(
            md, names
        )
        if expected != names:
            raise RuntimeError("validated GPT-6 dynamic inventory changed ordering")
        pose_exclusions = initializer_pose_exclusions(md, names)

    def require_exact_names(label: str, actual, expected) -> None:
        if isinstance(actual, dict):
            values = list(actual)
        elif isinstance(actual, (list, tuple, set)):
            values = list(actual)
        else:
            raise RuntimeError(f"{label} is not a name collection")
        if len(values) != len(set(values)) or set(values) != set(expected):
            missing = sorted(set(expected) - set(values))
            extra = sorted(set(values) - set(expected))
            raise RuntimeError(
                f"{label} coverage mismatch: missing={missing!r}, extra={extra!r}"
            )

    work = md / "physics" / "composition_certify"
    work.mkdir(parents=True, exist_ok=True)
    server = os.path.join(os.path.dirname(__file__), "register_blender_server.py")
    rc = RenderClient(blender_cmd, server, blend, log_path=str(work / "blender.log"))
    isaac = None
    try:
        prepare_request = {"cmd": "prepare", "objects": names, "parents": {}}
        if runtime_inventory_profile:
            prepare_request.update(
                allow_empty_roots=True,
                authored_empty_ids=authored_ids,
            )
        prep = rc.rpc(prepare_request)
        surfaces = list(prep.get("surfaces", []))
        if runtime_inventory_profile:
            require_exact_names(
                "register prepared dynamic", prep.get("prepared"), names
            )
            if prep.get("authored_empty_roots") != authored_ids:
                raise RuntimeError(
                    "register authored Empty bindings differ from validated inventory"
                )
            logical_parts = prep.get("logical_parts")
            if not isinstance(logical_parts, dict) or set(logical_parts) != set(
                authored_ids
            ):
                raise RuntimeError(
                    "register did not report exact authored Empty descendant ownership"
                )
            descendants: list[str] = []
            for root_name, part_names in logical_parts.items():
                if (
                    not isinstance(part_names, list)
                    or not part_names
                    or any(not isinstance(part, str) or not part for part in part_names)
                    or len(part_names) != len(set(part_names))
                ):
                    raise RuntimeError(
                        f"register returned invalid logical parts for {root_name!r}"
                    )
                descendants.extend(part_names)
            if len(descendants) != len(set(descendants)):
                raise RuntimeError("authored Empty descendants are multiply owned")
            if set(descendants) & set(surfaces):
                raise RuntimeError(
                    "authored Empty descendants leaked into static surfaces: "
                    + ", ".join(sorted(set(descendants) & set(surfaces)))
                )
            if set(names) & set(surfaces) or len(surfaces) != len(set(surfaces)):
                raise RuntimeError("register returned overlapping/duplicate surfaces")
            if authored_ids:
                matrix_response = rc.rpc(
                    {"cmd": "get_matrix", "names": sorted(authored_ids)}
                )
                current_root_matrices = matrix_response.get("matrices")
                require_exact_names(
                    "register authored root matrices",
                    current_root_matrices,
                    authored_ids,
                )
                for root_name, matrix in current_root_matrices.items():
                    authored_references[root_name]["current_root_matrix"] = matrix
            prepared = list(names)
        else:
            logical_parts = {}
            prepared = [n for n in names if n in set(prep.get("prepared", []))]
        requested_dumps = prepared + surfaces
        paths = rc.rpc({"cmd": "dump_npz", "names": requested_dumps, "out": str(work)})[
            "paths"
        ]
        if runtime_inventory_profile:
            require_exact_names("register mesh dump", paths, requested_dumps)
            missing_files = sorted(
                name for name, path in paths.items() if not Path(path).is_file()
            )
            if missing_files:
                raise RuntimeError(
                    "register mesh dumps are missing: " + ", ".join(missing_files)
                )
            if not surfaces:
                raise RuntimeError(
                    "GPT-6 actual-support certification requires actual static colliders"
                )
            # Static triangle colliders may be open; validate their mesh inputs
            # without requiring watertight geometry.
            from lib.tools.geometry.surface_validation import audit_surface_meshes

            support_meshes = audit_surface_meshes(paths, surfaces)
            require_exact_names(
                "actual-support collider audit", support_meshes, surfaces
            )
            invalid_surfaces = sorted(
                name
                for name, row in support_meshes.items()
                if row.get("valid") is not True
            )
            if invalid_surfaces:
                raise RuntimeError(
                    "GPT-6 actual-support static colliders are invalid: "
                    + ", ".join(invalid_surfaces)
                )
        isaac = SettleClient(
            work,
            isaac_python or DEFAULT_ISAAC_PYTHON,
            shared_dir=work.parent,  # <scene>/physics — reuse the run's warm boot
        )
        records = _preprocess_records(work)
        rollable = _rollable_flags(work)
        colliders, bindings = _boot_colliders(work, paths, prepared, records)
        if runtime_inventory_profile:
            require_exact_names("dynamic collider", colliders, prepared)
        applied: dict[str, list[str]] = {}
        for n, npz in colliders.items():
            ov = _merge_vlm_physics(
                work,
                n,
                paths[n],
                _add_overrides(
                    records,
                    n,
                    npz,
                    binding=bindings.get(n),
                    strict=runtime_inventory_profile,
                ),
            )
            applied[n] = sorted(ov)
            isaac.rpc(
                {
                    "cmd": "add",
                    "name": n,
                    "npz": npz,
                    **({"rollable": rollable[n]} if n in rollable else {}),
                    **ov,
                }  # fmt: skip
            )
        _log_boot_overrides(work, records, applied, bindings)
        for n in surfaces:
            if n in paths:
                isaac.rpc({"cmd": "add_static", "name": n, "npz": paths[n]})
        certify_request = {"cmd": "certify"}  # capless: one joint free sim, all baked
        if runtime_inventory_profile:
            from isaac.physics_config import EXPORT_VELOCITY_REST_POLICY

            certify_request["support_policy"] = "actual_surfaces_v1"
            certify_request["rest_policy"] = EXPORT_VELOCITY_REST_POLICY
        r = isaac.rpc(certify_request)
        if runtime_inventory_profile:
            _validate_joint_certification(r, prepared, surfaces)
        prep_set = set(prepared)
        baked_deltas = {}
        for n, total in r["total"].items():
            M = _mat(total)
            if n in prep_set and np.abs(M - np.eye(4)).max() > 1e-6:
                rc.rpc({"cmd": "set_matrix", "name": n, "M": M.tolist()})
                baked_deltas[n] = M
        if not runtime_inventory_profile:
            rc.rpc({"cmd": "save", "path": blend})
        # This SECOND diagnostic omits the tabletop-height global proxy and never
        # applies its drift. GPT-6 already omitted it in the baked pass as well.
        from lib.tools.geometry.surface_validation import (
            audit_surface_meshes,
            summarize_surface_validation,
        )

        try:
            actual_request = {"cmd": "validate_surfaces"}
            if runtime_inventory_profile:
                actual_request["rest_policy"] = EXPORT_VELOCITY_REST_POLICY
            actual_response = isaac.rpc(actual_request)
            if runtime_inventory_profile:
                _validate_velocity_rest(actual_response, prepared)
                require_exact_names(
                    "actual-surface dynamic",
                    actual_response.get("drift"),
                    prepared,
                )
                require_exact_names(
                    "actual-surface static",
                    actual_response.get("surfaces"),
                    surfaces,
                )
            actual_surfaces = summarize_surface_validation(
                actual_response,
                names,
                rollable,
                audit_surface_meshes(paths, surfaces),
            )
            if runtime_inventory_profile:
                actual_surfaces["rest_policy"] = actual_response["rest_policy"]
                actual_surfaces["rest"] = actual_response["rest"]
        except Exception as exc:  # noqa: BLE001 - old warm servers must not break export
            if runtime_inventory_profile:
                raise RuntimeError(
                    "GPT-6 actual-surface certification lacked complete coverage"
                ) from exc
            actual_surfaces = {
                "status": "unavailable",
                "baked": False,
                "support_proxy": False,
                "reason": str(exc),
            }
        if runtime_inventory_profile:
            # Do not persist the baked matrices until every dynamic/static body has
            # complete evidence in both physics passes.
            rc.rpc({"cmd": "save", "path": blend})
        (work / "actual_surface_validation.json").write_text(
            json.dumps(actual_surfaces, indent=2)
        )
        print(
            f"[actual-surface-validation] {actual_surfaces['status']} (diagnostic only; not baked)"
        )
    finally:
        if isaac is not None:
            isaac.disconnect()  # run-end shutdown is static_scene.py's job
        rc.close()
    drift = r["drift"]
    converged = bool(r.get("converged", True))
    for n, d in drift.items():
        d["toppled"] = capsized(
            d.get("tilt_deg", 0.0), d.get("dxy", 0.0) * 1000.0, bool(rollable.get(n))
        )
    telemetry: dict[str, dict] | None = {} if runtime_inventory_profile else None
    delivered = delivered_from_meshes(
        md, records, {n: paths[n] for n in prepared if n in paths}, rollable,
        baked_deltas=baked_deltas,
        **(
            {
                "authored_references": authored_references,
                "diagnostics": telemetry,
                "pose_exclusions": pose_exclusions,
            }
            if runtime_inventory_profile
            else {}
        ),
    )  # fmt: skip
    coverage = None
    if runtime_inventory_profile:
        assert telemetry is not None
        require_exact_names("delivered telemetry", telemetry, prepared)
        telemetry_complete = all(
            row.get("status") == "measured"
            or (
                name in pose_exclusions
                and row.get("status") == "excluded"
                and row.get("reason_code") == "initializer_edited"
                and row.get("stage") == "initializer"
            )
            for name, row in telemetry.items()
        )
        coverage = {
            # v2: no top-level status. Object-name coverage is hard-required above
            # (require_exact_names), so the block's presence is the contract.
            "schema_version": 2,
            "support_policy": r["support_policy"],
            "support_proxy": r["support_proxy"],
            "expected_dynamic_objects": names,
            "prepared_dynamic_objects": prepared,
            "collider_dynamic_objects": sorted(colliders),
            "simulated_dynamic_objects": sorted(drift),
            "certification_total_objects": sorted(r["total"]),
            "static_total_transforms_unchanged": True,
            "actual_surface_dynamic_objects": sorted(actual_surfaces.get("drift", {})),
            "static_surfaces": sorted(surfaces),
            "authored_empty_roots": dict(sorted(authored_ids.items())),
            "authored_empty_parts": {
                name: sorted(parts) for name, parts in sorted(logical_parts.items())
            },
            "authored_revision_references": {
                name: dict(reference)
                for name, reference in sorted(authored_references.items())
            },
            "delivered_pose_telemetry": {
                "status": "complete" if telemetry_complete else "incomplete",
                "policy": INITIALIZER_POSE_EXCLUSION_POLICY,
                "required_objects": sorted(set(prepared) - set(pose_exclusions)),
                "excluded_objects": sorted(pose_exclusions),
                "objects": telemetry,
            },
        }
    late_blocks = {
        "composition_certify": drift,
        "composition_certify_converged": converged,
        **(
            {
                "composition_certify_rest_policy": r["rest_policy"],
                "composition_certify_rest": r["rest"],
            }
            if runtime_inventory_profile
            else {}
        ),
        "delivered": delivered,
        "actual_surface_validation": actual_surfaces,
    }
    if coverage is not None:
        late_blocks["composition_certify_coverage"] = coverage
    merge_pose_changes(
        md / "physics" / "pose_changes.json", run_id_for(md),
        late_blocks,
    )  # fmt: skip
    notable = 0
    for n, d in drift.items():
        flag = (
            " TOPPLED (baked)" if d["toppled"]
            else " NOTABLE" if (d["dxy"] > CERT_DXY_CAP_M
                               or d["tilt_deg"] > CERT_TILT_CAP_DEG)
            else ""
        )  # fmt: skip
        notable += bool(flag)
        print(
            f"[composition-certify] {n}: dxy={d['dxy'] * 1000:.1f}mm "
            f"dz={d['dz'] * 1000:+.1f}mm tilt={d['tilt_deg']:.1f}deg{flag}"
        )
    print(
        f"[composition-certify] joint free-settle baked: {len(drift)} objects, "
        f"{notable} notable, converged={converged}"
    )
    caps = sorted(n for n, d in delivered.items() if d["capsized"])
    for n in caps:  # placed -> delivered, the measure no per-stage drift can show
        d = delivered[n]
        print(f"[delivered] {n}: CAPSIZED vs placed — tilt={d['tilt_deg']:.1f}deg "
              f"dxy={d['dxy'] * 1000:.0f}mm dz={d['dz'] * 1000:+.0f}mm")  # fmt: skip
    print(f"[delivered] {len(delivered)} objects measured against their placed pose, "
          f"{len(caps)} capsized")  # fmt: skip
    recovered = sorted(n for n, rr in records.items() if rr.get("settle_recovered"))
    carried = sorted(n for n, rr in records.items() if rr.get("settle_failed"))
    if recovered or carried:
        print(
            "[composition-certify] preprocess settle history: "
            f"{len(recovered)} recovered by a later preprocess drop, "
            f"{len(carried)} entered composition without a demonstrated rest"
        )
    for n in carried:
        d = drift.get(n, {})
        outcome = (
            "late-settled during final certification"
            if converged and n in drift
            else "UNRESOLVED at final certification"
        )
        print(
            f"[composition-certify] {n}: {outcome}; "
            f"dxy={float(d.get('dxy', 0.0)) * 1000.0:.1f}mm "
            f"tilt={float(d.get('tilt_deg', 0.0)):.1f}deg"
        )
    if not converged:
        print(
            "[composition-certify] WARNING: step cap hit mid-motion — the baked "
            "poses are a snapshot, not a rest (recorded, not repaired)"
        )
    return drift
