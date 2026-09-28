"""Root-surface relationship vocabulary (single source of truth).

A relationship is ``{"type": <REL>, "a": "<id>", "b": "<id>"}`` between two root-surface
ids (``"wall#0"``, ``"table#0"``, ...). The MEANING of each type is documented here and
echoed verbatim (via ``relationship_glossary``) into the masking-pipeline VLM prompts;
the per-scene relationships are detected by the VLM (``agentic_mask.propose_objects``),
normalized by ``parse_relationships``, assigned structured authority by
``adjudicate_relationships`` during preprocessing, and handed to the static-scene
initializer as required instructions or advisory visual guidance.
"""

from __future__ import annotations

import re
from typing import Any


def surface_build_name(surface_id: str) -> str:
    """The Blender object name the initializer must build a surface with, derived from its
    scene-graph id (``wall#0`` -> ``wall_0``, ``glasses case#0`` -> ``glasses_case_0``). Same slug
    convention as imported objects' ``obj_<slug>`` minus the ``obj_`` prefix, so the relationship
    rule can locate the built surface by EXACT name (no reliance on a noisy MoGE plane). The
    missing ``obj_`` prefix also keeps the penetration script's object-vs-surface split intact."""
    return (
        re.sub(r"[^a-zA-Z0-9_-]+", "_", surface_id.replace("#", "_"))
        .strip("_")
        .lower()[:48]
        or "surface"
    )


_GROUND_CATEGORIES = frozenset({"floor", "ground"})
_BASE_REQUIRING_SUPPORT_WORDS = frozenset(
    {"table", "tabletop", "desk", "counter", "countertop", "workbench", "workstation"}
)

# Relationship adjudication schema.  A relationship has one atomic source-scene
# verdict.  Runtime obligations compiled from a hard relationship are separate records;
# they must never mutate only one "part" of this source verdict.
RELATIONSHIP_SCHEMA_VERSION = 3
RELATIONSHIP_STATUSES = frozenset(
    {"confirmed", "unverified", "rejected", "conflicting"}
)
RELATIONSHIP_ENFORCEMENTS = frozenset({"hard", "advisory", "none"})
RELATIONSHIP_STATUS_TO_ENFORCEMENT = {
    "confirmed": "hard",
    "unverified": "advisory",
    "conflicting": "advisory",
    "rejected": "none",
}


def relationship_schema_is_valid(rel: dict[str, Any]) -> bool:
    """Whether ``rel`` carries the schema-v3 atomic status/authority mapping."""

    status = str(rel.get("status", "")).strip().lower()
    enforcement = str(rel.get("enforcement", "")).strip().lower()
    return RELATIONSHIP_STATUS_TO_ENFORCEMENT.get(status) == enforcement


def relationship_is_hard(rel: dict[str, Any]) -> bool:
    """Whether a valid schema-v3 relationship may drive required geometry/rules.

    Missing or mismatched fields fail closed.  There is intentionally no legacy
    default-hard path: incompatible cached graphs must be re-preprocessed.
    """

    return relationship_schema_is_valid(rel) and str(rel.get("status")).lower() == "confirmed"


def relationship_is_active(rel: dict[str, Any]) -> bool:
    """Whether a relationship should be shown at all (required or advisory)."""

    return relationship_schema_is_valid(rel) and str(rel.get("enforcement")).lower() != "none"


def floor_under_furniture_support(
    relationships: list[dict[str, str]], main_support_id: str | None
) -> str | None:
    """Return the floor/ground id directly under a furniture main support.

    This is the shared form-reconciliation predicate.  A bare tabletop remains valid when
    there is no ground relationship (including when the initializer later adds an unlisted
    helper floor), but an adjudicated-hard ``under(floor, table)`` relationship means the
    complete furniture must reach that ground through an integrated base.
    """
    main = str(main_support_id or "").strip().lower()
    category = main.split("#", 1)[0]
    words = set(re.findall(r"[a-z0-9]+", category))
    if not main or not words.intersection(_BASE_REQUIRING_SUPPORT_WORDS):
        return None
    for rel in relationships or []:
        if not relationship_is_hard(rel):
            continue
        if str(rel.get("type", "")).strip().lower() != "under":
            continue
        a = str(rel.get("a", "")).strip().lower()
        b = str(rel.get("b", "")).strip().lower()
        if b != main or a == main:
            continue
        if a.split("#", 1)[0] in _GROUND_CATEGORIES:
            return a
    return None


# type -> human-readable meaning. Surfaced verbatim in the env system prompt so the
# agent, the snapper, and the checker share one definition.
RELATIONSHIP_MEANINGS: dict[str, str] = {
    "against": (
        "A's near edge is flush against B's surface (e.g. a table/desk pushed against a "
        "wall, or a cabinet against a wall): A's contacting edge lies IN B's plane — "
        "connected, neither floating away from B nor crossing through it. Whenever a "
        "table/desk/cabinet is pushed up to a wall with NO VISIBLE GAP, say 'against' "
        "— NOT 'perpendicular'."
    ),
    "perpendicular": (
        "A and B meet at a right angle: one HORIZONTAL surface and one VERTICAL surface (e.g. a "
        "table and a wall, or a floor and a wall). Use this ONLY when the two surfaces actually "
        "form an edge together; do NOT use it when one surface merely STANDS ON the other (that "
        "is 'under'), and do NOT use it when one surface sits FLUSH to the other with no gap "
        "(that is 'against')."
    ),
    "corner": (
        "A and B are two walls that share a single VERTICAL edge (a room corner): they "
        "both reach that edge; a slab may extend past it because the other wall hides that "
        "overshoot. This is most often the case when two walls meet at a room corner. Only "
        "ADJACENT walls form a "
        "corner: with THREE walls (left, back, right), the corners are left<->back and "
        "back<->right — the two SIDE walls are PARALLEL to each other and form NO corner. "
        "NEVER emit corner for every pair of walls."
    ),
    "under": (
        "A sits directly beneath B and supports it: A's top meets the BOTTOM of B's complete "
        "supporting structure. For a FLOOR under a table/desk/counter, that means the bottom "
        "of the table's connected legs/pedestal/cabinet — NEVER the underside of a bare "
        "tabletop slab and NEVER raising the floor to tabletop height. When BOTH a floor and a "
        "wall are present, the floor is usually UNDER the wall too (the wall stands on the "
        "floor) — state that relationship as well. Whenever B simply RESTS ON A, use 'under' "
        "— NOT 'perpendicular'."
    ),
}
RELATIONSHIP_TYPES = frozenset(RELATIONSHIP_MEANINGS)


def parse_relationships(raw: Any) -> list[dict[str, str]]:
    """Normalize a raw VLM ``relationships`` list to ``[{type, a, b}]``.

    Drops malformed / unknown-type / self entries. Tolerates key aliases
    (``relation``/``rel`` for type; ``from``/``first`` for a; ``to``/``second`` for b)
    and de-duplicates (treating the pair as unordered for symmetric relations).
    """
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    seen: set[tuple[str, str, str]] = set()
    symmetric = {"perpendicular", "corner"}
    for r in raw:
        if not isinstance(r, dict):
            continue
        t = (
            str(r.get("type") or r.get("relation") or r.get("rel") or "")
            .strip()
            .lower()
        )
        a = str(r.get("a") or r.get("from") or r.get("first") or "").strip().lower()
        b = str(r.get("b") or r.get("to") or r.get("second") or "").strip().lower()
        if t not in RELATIONSHIP_TYPES or not a or not b or a == b:
            continue
        key = (t, *sorted((a, b))) if t in symmetric else (t, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": t, "a": a, "b": b})
    return out


def infer_against(
    rels: list[dict[str, str]],
    id2mask: dict[str, Any],
    world: Any,
    gap_max: float = 0.05,
) -> list[str]:
    """Geometric ``against`` backstop: upgrade ``perpendicular`` (support, wall) pairs
    whose measured gap is under ``gap_max`` — IN PLACE, returning the upgraded pair
    descriptions for logging.

    The proposer VLM nearly always emits ``perpendicular`` for a table touching a wall,
    which carries no yaw pin and no flush constraint; ``against`` pins a rectangular
    support's free yaw to the wall's photo-anchored run and enforces flushness. For each
    ``perpendicular`` pair of one horizontal support (plane normal ~ +Z) and one wall
    (normal ~ horizontal) in the gravity-aligned ``world`` (H,W,3) points: RANSAC the
    wall plane, orient its normal toward the support's centroid, and measure the
    support's near-edge signed distance (2nd percentile — robust to mask-edge noise).
    Under ``gap_max`` (default 5cm: MoGE depth noise at a wall junction is a couple of
    cm, while the built-scene flush tolerance stays the checker's exact 1cm) the pair
    becomes ``{"type": "against", "a": <support>, "b": <wall>}``. Only upgrades pairs
    the proposer already connected; never invents relationships. Masks below
    ``MIN_PLANE_MASK_FRAC`` are identity-only evidence, so their relationships are
    left untouched rather than upgraded from an unreliable plane."""
    import numpy as np

    from lib.tools.geometry.scene_graph import (
        MIN_PLANE_MASK_FRAC,
        PLUMB_SUPPORT_MIN_NZ,
        PLUMB_WALL_MAX_NZ,
        fit_plane,
    )

    def _plane(sid):
        m = id2mask.get(sid)
        if m is None:
            return None
        mask = np.asarray(m, bool)
        if float(mask.mean()) < MIN_PLANE_MASK_FRAC:
            return None
        pts = world[mask & np.isfinite(world).all(axis=2)]
        return fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None

    planes = {}
    upgraded = []
    for r in rels:
        if r.get("type") != "perpendicular":
            continue
        for s in (r["a"], r["b"]):
            if s not in planes:
                planes[s] = _plane(s)
        pa, pb = planes[r["a"]], planes[r["b"]]
        if not pa or not pb:
            continue
        na, nb = (np.asarray(p["normal"], float) for p in (pa, pb))
        # support/wall cuts are the shared plumb_plane bands (scene_graph.py) so this
        # upgrade, the prompt snap, and the checkers agree on wall-vs-support.
        if abs(na[2]) >= PLUMB_SUPPORT_MIN_NZ and abs(nb[2]) <= PLUMB_WALL_MAX_NZ:
            sup, wall, wall_pl, sup_pl = r["a"], r["b"], pb, pa
        elif abs(nb[2]) >= PLUMB_SUPPORT_MIN_NZ and abs(na[2]) <= PLUMB_WALL_MAX_NZ:
            sup, wall, wall_pl, sup_pl = r["b"], r["a"], pa, pb
        else:
            continue  # wall-wall / ambiguous planes: not this rule's business
        n, d = np.asarray(wall_pl["normal"], float), float(wall_pl["d"])
        c_sup = np.asarray(sup_pl["centroid"], float)
        if float(n @ c_sup + d) < 0:  # orient the wall normal toward the support
            n, d = -n, -d
        m = id2mask[sup]
        pts = world[np.asarray(m, bool) & np.isfinite(world).all(axis=2)]
        gap = float(np.percentile(pts @ n + d, 2.0))  # support's near edge -> wall
        if abs(gap) < gap_max:
            r["type"], r["a"], r["b"] = "against", sup, wall
            upgraded.append(f"{sup} against {wall} (gap {gap * 100:.1f}cm)")
    return upgraded


def drop_impossible_corners(
    rels: list[dict[str, str]],
    id2mask: dict[str, Any],
    world: Any,
    max_parallel_cos: float = 0.707,
) -> list[str]:
    """Drop ``corner`` rels between PARALLEL walls — IN PLACE, returning drop
    descriptions for logging.

    With three walls (left, back, right) the VLM tends to emit ALL THREE pairwise
    corners, but the two side walls are parallel opposite walls: their corner is
    geometrically impossible, and the build-time corner rule then demands "make
    them perpendicular" — an unsatisfiable constraint that cannot hold together
    with the two REAL corners (0710_eval1_abc1: the agent burned rounds on exactly
    this fight). For each corner pair, RANSAC both wall planes in the gravity-
    aligned ``world`` and compare the HORIZONTAL components of their normals
    (mod-180 fold): |cos| above ``max_parallel_cos`` (< ~45 deg apart) means
    parallel -> dropped. Near-perpendicular corners are untouched. A mask below
    ``MIN_PLANE_MASK_FRAC`` is not trusted for this plane-based deletion, so the
    proposed corner relationship is preserved."""
    import math

    import numpy as np

    from lib.tools.geometry.scene_graph import MIN_PLANE_MASK_FRAC, fit_plane

    planes: dict = {}

    def _hnormal(sid):
        if sid not in planes:
            m = id2mask.get(sid)
            if m is None:
                planes[sid] = None
            else:
                mask = np.asarray(m, bool)
                if float(mask.mean()) < MIN_PLANE_MASK_FRAC:
                    planes[sid] = None
                    return planes[sid]
                pts = world[mask & np.isfinite(world).all(axis=2)]
                pl = fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None
                n = None
                if pl:
                    nx, ny = float(pl["normal"][0]), float(pl["normal"][1])
                    h = math.hypot(nx, ny)
                    n = (nx / h, ny / h) if h > 1e-6 else None
                planes[sid] = n
        return planes[sid]

    dropped = []
    for r in list(rels):
        if r.get("type") != "corner":
            continue
        na, nb = _hnormal(r["a"]), _hnormal(r["b"])
        if na is None or nb is None:
            continue  # unmeasurable (name-only wall): leave the rel alone
        cosang = abs(na[0] * nb[0] + na[1] * nb[1])
        if cosang > max_parallel_cos:
            rels.remove(r)
            deg = math.degrees(math.acos(min(1.0, cosang)))
            dropped.append(
                f"corner {r['a']} <-> {r['b']} (walls parallel, {deg:.0f} deg apart "
                "— opposite side walls form no corner)"
            )
    return dropped


def adjudicate_relationships(
    rels: list[dict[str, Any]],
    root_ids: set[str] | list[str] | tuple[str, ...],
    id2mask: dict[str, Any] | None,
    world: Any | None,
    *,
    contact_occluder_masks: list[Any] | tuple[Any, ...] | None = None,
    gap_max: float = 0.05,
    gap_reject_min: float = 0.15,
    finite_gap_max: float = 0.05,
    finite_gap_reject_min: float = 0.15,
    under_max_vertical_separation: float = 2.5,
    perpendicular_min_deg: float = 88.0,
    corner_min_deg: float = 85.0,
    corner_reach_max: float = 0.20,
    against_run_conflict_deg: float = 6.0,
    edge_bearing_rectangularity: float = 0.90,
) -> dict[str, Any]:
    """Adjudicate VLM root-surface relationships against measured geometry, in place.

    The VLM remains the *proposer*.  A retained direct semantic ``AGAINST`` becomes
    authoritative when the final masks expose a compatible contact-facing edge; noisy
    monocular depth gaps remain serialized diagnostics and cannot veto that conclusion by
    themselves.  ``UNDER`` is policy-authoritative after endpoint validation.  Every
    source relationship receives one atomic status whose enforcement is derived by the
    schema-v3 mapping; runtime constraint results are a separate concern.

    Direction is normalized whenever it is safe:

    * ``against`` and support/wall ``perpendicular``: support -> wall;
    * ``under``: lower/support -> upper;
    * a wall-wall ``perpendicular`` is normalized to ``corner`` because that is the
      vocabulary's actual wall-intersection relation.

    Returns a compact scene-level summary suitable for ``masks.json`` and
    ``scene_graph.json``.  Every numeric value placed on a relationship is a built-in
    Python scalar/list, so the result is directly JSON serializable.
    """

    import math

    import numpy as np

    from lib.tools.geometry.relationship_contact_evidence import (
        IMAGE_CONTACT_EVIDENCE_VERSION,
        image_contact_compatibility,
    )
    from lib.tools.geometry.scene_graph import (
        MIN_PLANE_MASK_FRAC,
        PLUMB_SUPPORT_MIN_NZ,
        PLUMB_WALL_MAX_NZ,
        fit_plane,
        plane_is_reliable,
        plumb_plane,
    )

    roots = {str(x).strip().lower() for x in root_ids if str(x).strip()}
    masks = id2mask or {}
    world_arr = None if world is None else np.asarray(world)
    plane_cache: dict[str, dict[str, Any] | None] = {}

    def _category(sid: str) -> str:
        return sid.rsplit("#", 1)[0].strip().lower()

    def _category_role(sid: str) -> str:
        cat = _category(sid)
        words = set(re.findall(r"[a-z0-9]+", cat))
        if "wall" in words or "wall" in cat:
            return "wall"
        support_words = (
            _GROUND_CATEGORIES
            | _BASE_REQUIRING_SUPPORT_WORDS
            | {
                "shelf",
                "bench",
                "platform",
                # Furniture whose useful contact surface is commonly proposed by name
                # only.  A missing plane must leave these claims advisory, rather than
                # rejecting the endpoint-role hypothesis outright.
                "cabinet",
                "dresser",
                "sideboard",
                "console",
            }
        )
        return "support" if words.intersection(support_words) else "unknown"

    def _plane(sid: str) -> dict[str, Any] | None:
        if sid in plane_cache:
            return plane_cache[sid]
        mask_raw = masks.get(sid)
        if mask_raw is None or world_arr is None or world_arr.ndim != 3:
            plane_cache[sid] = None
            return None
        mask = np.asarray(mask_raw, bool)
        if mask.shape != world_arr.shape[:2]:
            plane_cache[sid] = None
            return None
        mask_frac = float(mask.mean())
        if mask_frac < MIN_PLANE_MASK_FRAC:
            plane_cache[sid] = None
            return None
        valid = mask & np.isfinite(world_arr).all(axis=2)
        pts = world_arr[valid]
        pl = fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None
        if pl is not None:
            pl = dict(pl)
            pl["mask_frac"] = mask_frac
            # Keep the points private to this pass; never serialize this array.
            pl["_points"] = pts
        if not plane_is_reliable(pl):
            pl = None
        plane_cache[sid] = pl
        return pl

    def _role(sid: str) -> tuple[str, bool]:
        """(role, measured): measured roles override name heuristics."""

        pl = _plane(sid)
        if pl is not None:
            nz = abs(float(np.asarray(pl["normal"], float)[2]))
            if nz >= PLUMB_SUPPORT_MIN_NZ:
                return "support", True
            if nz <= PLUMB_WALL_MAX_NZ:
                return "wall", True
            # The shared plumb policy treats this tilt band exactly like a missing
            # plane: retain the name-derived role, but never grant metric authority.
            return _category_role(sid), False
        return _category_role(sid), False

    def _append_provenance(rel: dict[str, Any], record: dict[str, Any]) -> None:
        prov = rel.setdefault("provenance", [])
        if not isinstance(prov, list):
            prov = rel["provenance"] = []
        if record not in prov:
            prov.append(record)

    def _reason(rel: dict[str, Any], message: str) -> None:
        reasons = rel.setdefault("adjudication_reasons", [])
        if message not in reasons:
            reasons.append(message)

    def _reason_code(rel: dict[str, Any], code: str) -> None:
        codes = rel.setdefault("reason_codes", [])
        if code not in codes:
            codes.append(code)

    def _under_diagnostic(rel: dict[str, Any], message: str) -> None:
        """Record non-authoritative source-geometry evidence for ``UNDER``."""

        diagnostics = rel["measurements"].setdefault("under_geometry_diagnostics", [])
        if message not in diagnostics:
            diagnostics.append(message)

    def _set(rel: dict[str, Any], status: str, enforcement: str, reason: str) -> None:
        expected = RELATIONSHIP_STATUS_TO_ENFORCEMENT.get(status)
        if expected is None or enforcement != expected:
            raise ValueError(
                f"invalid relationship verdict {status!r}/{enforcement!r}; "
                f"expected enforcement {expected!r}"
            )
        rel["status"], rel["enforcement"] = status, enforcement
        _reason(rel, reason)
        _append_provenance(
            rel,
            {
                "source": "geometry_preprocess",
                "action": "adjudicated",
                "status": status,
                "enforcement": enforcement,
                "reason": reason,
            },
        )

    def _semantic_against_evidence(rel: dict[str, Any]) -> dict[str, Any]:
        """Return semantic support without treating a VLM vote as geometry proof."""

        provenance = rel.get("provenance")
        records = provenance if isinstance(provenance, list) else []
        proposer_claimed = any(
            isinstance(record, dict)
            and record.get("source") == "vlm_proposer"
            and record.get("action") == "proposed"
            for record in records
        )
        auditor_actions = []
        for record in records:
            if not isinstance(record, dict) or record.get("source") != "vlm_auditor":
                continue
            action = str(record.get("action") or "")
            supports_against = action in {"confirmed", "proposed"} or (
                action == "corrected_type" and record.get("to") == "against"
            )
            if supports_against:
                auditor_actions.append(action)
        return {
            "direct_against_claim": True,
            "proposer_claimed": proposer_claimed,
            "auditor_retained": bool(auditor_actions),
            "auditor_actions": auditor_actions,
            # A relationship introduced/corrected by the auditor is itself a direct
            # semantic claim even when the proposer used another type.
            "semantic_authoritative_candidate": bool(auditor_actions),
        }

    def _image_contact_evidence(support: str, wall: str) -> dict[str, Any]:
        support_mask = masks.get(support)
        wall_mask = masks.get(wall)
        if support_mask is None or wall_mask is None:
            return {
                "schema_version": IMAGE_CONTACT_EVIDENCE_VERSION,
                "status": "unobservable",
                "reason_code": "missing_support_or_wall_mask",
                "authority": "diagnostic_only",
            }
        return image_contact_compatibility(
            support_mask,
            wall_mask,
            occluder_masks=contact_occluder_masks,
        )

    def _normalize_direction(rel: dict[str, Any], a: str, b: str, why: str) -> None:
        if (rel.get("a"), rel.get("b")) == (a, b):
            return
        old_a, old_b = rel.get("a"), rel.get("b")
        rel["a"], rel["b"] = a, b
        _append_provenance(
            rel,
            {
                "source": "geometry_preprocess",
                "action": "normalized_direction",
                "from": [old_a, old_b],
                "to": [a, b],
                "reason": why,
            },
        )

    def _change_type(rel: dict[str, Any], new_type: str, why: str) -> None:
        old = rel.get("type")
        if old == new_type:
            return
        rel["type"] = new_type
        _append_provenance(
            rel,
            {
                "source": "geometry_preprocess",
                "action": "normalized_type",
                "from": old,
                "to": new_type,
                "reason": why,
            },
        )

    def _support_wall_gap(support: str, wall: str) -> float | None:
        ps, pw = _plane(support), _plane(wall)
        if ps is None or pw is None:
            return None
        pn, kind = plumb_plane(pw["normal"])
        if pn is None or kind != "wall":
            return None
        n = np.asarray(pn, float)
        # The canonical wall contract is the plumbed plane through the measured
        # centroid; prompt, POSE and CONTACT consume this same plane/run.
        d = -float(n @ np.asarray(pw["centroid"], float))
        c_sup = np.asarray(ps["centroid"], float)
        if float(n @ c_sup + d) < 0:
            n, d = -n, -d
        pts = np.asarray(ps["_points"], float)
        return float(np.percentile(pts @ n + d, 2.0)) if len(pts) else None

    def _normal_angle(a: str, b: str) -> float | None:
        pa, pb = _plane(a), _plane(b)
        if pa is None or pb is None:
            return None
        pna, ka = plumb_plane(pa["normal"])
        pnb, kb = plumb_plane(pb["normal"])
        if pna is None or pnb is None or "ambiguous" in {ka, kb}:
            return None
        na, nb = np.asarray(pna, float), np.asarray(pnb, float)
        cosang = min(1.0, abs(float(na @ nb)))
        return float(math.degrees(math.acos(cosang)))

    def _wall_run(sid: str) -> tuple[float, float] | None:
        pl = _plane(sid)
        if pl is None:
            return None
        pn, kind = plumb_plane(pl["normal"])
        if pn is None or kind != "wall":
            return None
        n = np.asarray(pn, float)
        h = float(math.hypot(n[0], n[1]))
        return (-float(n[1]) / h, float(n[0]) / h) if h > 1e-8 else None

    def _interval(
        points: Any, axis: Any, lo: float = 2.0, hi: float = 98.0
    ) -> tuple[float, float] | None:
        pts = np.asarray(points, float)
        vec = np.asarray(axis, float)
        if pts.ndim != 2 or len(pts) == 0 or vec.ndim != 1:
            return None
        values = pts @ vec
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None
        q = np.percentile(values, [lo, hi])
        return float(q[0]), float(q[1])

    def _interval_contact(
        left: tuple[float, float] | None,
        right: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """Return ``(gap, overlap)`` between two finite 1-D intervals."""

        if left is None or right is None:
            return None
        gap = max(left[0] - right[1], right[0] - left[1], 0.0)
        overlap = max(min(left[1], right[1]) - max(left[0], right[0]), 0.0)
        return float(gap), float(overlap)

    def _finite_against_metrics(support: str, wall: str) -> dict[str, Any] | None:
        """Finite contact evidence for a support-to-wall relationship.

        Plane distance alone describes an *infinite* wall.  These intervals also
        establish that the support reaches the visible wall along its run and in Z.
        Coordinates are in the canonical world frame and are deliberately serialized
        so runtime checks and audits can consume the same evidence.
        """

        ps, pw = _plane(support), _plane(wall)
        run = _wall_run(wall)
        if ps is None or pw is None or run is None:
            return None
        run3 = np.asarray([run[0], run[1], 0.0], float)
        zaxis = np.asarray([0.0, 0.0, 1.0], float)
        support_run = _interval(ps["_points"], run3)
        wall_run = _interval(pw["_points"], run3)
        support_z = _interval(ps["_points"], zaxis)
        wall_z = _interval(pw["_points"], zaxis)
        run_contact = _interval_contact(support_run, wall_run)
        vertical_contact = _interval_contact(support_z, wall_z)
        if run_contact is None or vertical_contact is None:
            return None
        return {
            "wall_run_axis_xy": [round(float(run[0]), 6), round(float(run[1]), 6)],
            "support_wall_run_interval_m": [round(x, 5) for x in support_run],
            "wall_run_interval_m": [round(x, 5) for x in wall_run],
            "wall_run_gap_m": round(run_contact[0], 5),
            "wall_run_overlap_m": round(run_contact[1], 5),
            "support_vertical_interval_m": [round(x, 5) for x in support_z],
            "wall_vertical_interval_m": [round(x, 5) for x in wall_z],
            "vertical_gap_m": round(vertical_contact[0], 5),
            "vertical_overlap_m": round(vertical_contact[1], 5),
        }

    def _xy_contact_metrics(a: str, b: str) -> dict[str, Any] | None:
        """Finite robust XY-AABB evidence used by ``under`` adjudication."""

        pa, pb = _plane(a), _plane(b)
        if pa is None or pb is None:
            return None
        intervals_a = [
            _interval(pa["_points"], axis)
            for axis in (
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
            )
        ]
        intervals_b = [
            _interval(pb["_points"], axis)
            for axis in (
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
            )
        ]
        contacts = [
            _interval_contact(ia, ib) for ia, ib in zip(intervals_a, intervals_b)
        ]
        if any(x is None for x in contacts):
            return None
        gaps = [float(x[0]) for x in contacts]
        overlaps = [float(x[1]) for x in contacts]
        return {
            "a_xy_intervals_m": [[round(v, 5) for v in x] for x in intervals_a],
            "b_xy_intervals_m": [[round(v, 5) for v in x] for x in intervals_b],
            "xy_axis_gaps_m": [round(v, 5) for v in gaps],
            "xy_overlap_extents_m": [round(v, 5) for v in overlaps],
            "xy_gap_m": round(max(gaps), 5),
            "xy_intervals_overlap": bool(max(gaps) <= 1e-6),
        }

    def _vertical_contact_metrics(lower: str, upper: str) -> dict[str, Any] | None:
        """Lower plane height versus the finite *bottom* of an upper wall.

        Distance to the wall's whole Z interval is not contact evidence: a floor or
        tabletop cutting through the middle of a wall lies inside that interval and
        would incorrectly produce zero gap.  ``UNDER`` requires the upper body's
        bottom to meet the lower support plane, so retain both the signed bottom gap
        and its absolute contact error.
        """

        pl, pu = _plane(lower), _plane(upper)
        if pl is None or pu is None:
            return None
        lower_z = float(np.asarray(pl["centroid"], float)[2])
        upper_z = _interval(pu["_points"], np.asarray([0.0, 0.0, 1.0]))
        if upper_z is None:
            return None
        signed_gap = float(upper_z[0] - lower_z)
        return {
            "lower_plane_height_m": round(lower_z, 5),
            "upper_vertical_interval_m": [round(x, 5) for x in upper_z],
            "upper_bottom_height_m": round(float(upper_z[0]), 5),
            "vertical_contact_signed_gap_m": round(signed_gap, 5),
            "vertical_contact_gap_m": round(abs(signed_gap), 5),
        }

    def _finite_gap_verdict(
        rel: dict[str, Any], gaps: list[tuple[str, float]], context: str
    ) -> bool:
        """Set a non-hard verdict for uncertain/contradictory finite gaps.

        Returns true only when every named finite gap lies inside the confirmation
        band, allowing the caller to perform any remaining semantic checks before
        granting hard authority.
        """

        rejected = [
            (name, value)
            for name, value in gaps
            if value >= finite_gap_reject_min - 1e-5
        ]
        if rejected:
            detail = ", ".join(f"{name}={value:.3f}m" for name, value in rejected)
            _set(
                rel,
                "rejected",
                "none",
                f"{context} is finitely disjoint ({detail}; rejection threshold "
                f"{finite_gap_reject_min:.3f}m)",
            )
            return False
        uncertain = [
            (name, value) for name, value in gaps if value > finite_gap_max + 1e-5
        ]
        if uncertain:
            detail = ", ".join(f"{name}={value:.3f}m" for name, value in uncertain)
            _set(
                rel,
                "unverified",
                "advisory",
                f"{context} is outside the finite-contact confirmation band "
                f"({detail}; confirmation threshold {finite_gap_max:.3f}m)",
            )
            return False
        return True

    def _support_rectangularity(sid: str) -> float | None:
        """Convex-hull area / minimum-area rectangle area of a measured support mask.

        This is only a preprocessing applicability guard for multi-wall yaw conflicts.
        The runtime A schema remains authoritative because it can inspect the actual
        built top hull.  A disc is ~pi/4; a rectangle/square is ~1.
        """

        raw_mask = masks.get(sid)
        if raw_mask is None:
            return None
        mask = np.asarray(raw_mask, bool)
        if (
            mask.ndim != 2
            or bool(mask[0].any())
            or bool(mask[-1].any())
            or bool(mask[:, 0].any())
            or bool(mask[:, -1].any())
        ):
            # A clipped disc/oval can look box-like.  Preprocessing demotion is
            # irreversible, so defer frame-cut applicability to the built top hull.
            return None
        pl = _plane(sid)
        if pl is None:
            return None
        pts = np.asarray(pl["_points"], float)[:, :2]
        if len(pts) < 3:
            return None
        try:
            from scipy.spatial import ConvexHull

            hull_obj = ConvexHull(pts)
            hull = pts[hull_obj.vertices]
            hull_area = float(hull_obj.volume)  # scipy's 2-D ``volume`` is area
        except Exception:  # noqa: BLE001 - optional geometry evidence
            return None
        if len(hull) < 3 or hull_area <= 1e-10:
            return None
        best = float("inf")
        for i in range(len(hull)):
            edge = hull[(i + 1) % len(hull)] - hull[i]
            n = float(np.linalg.norm(edge))
            if n <= 1e-10:
                continue
            u = edge / n
            v = np.asarray([-u[1], u[0]])
            pu, pv = hull @ u, hull @ v
            best = min(best, float(np.ptp(pu) * np.ptp(pv)))
        return hull_area / best if np.isfinite(best) and best > 1e-10 else None

    def _corner_reach(a: str, b: str) -> dict[str, Any] | None:
        pa, pb = _plane(a), _plane(b)
        if pa is None or pb is None:
            return None
        pna, ka = plumb_plane(pa["normal"])
        pnb, kb = plumb_plane(pb["normal"])
        if pna is None or pnb is None or ka != "wall" or kb != "wall":
            return None
        na, nb = np.asarray(pna, float), np.asarray(pnb, float)
        A = np.asarray([[na[0], na[1]], [nb[0], nb[1]]], float)
        det = float(np.linalg.det(A))
        if abs(det) < 1e-8:
            return None
        # Canonical plumbed planes pass through the measured centroids.
        rhs = np.asarray(
            [
                float(na[:2] @ np.asarray(pa["centroid"], float)[:2]),
                float(nb[:2] @ np.asarray(pb["centroid"], float)[:2]),
            ],
            float,
        )
        xy = np.linalg.solve(A, rhs)
        out = []
        for pl in (pa, pb):
            pts = np.asarray(pl["_points"], float)[:, :2]
            out.append(float(np.percentile(np.linalg.norm(pts - xy, axis=1), 2.0)))
        # Use the same robust source intervals as the other finite relationship
        # measurements.  XY plane intersection alone cannot certify a shared vertical
        # corner edge when the two observed wall slabs occupy disjoint height bands.
        z_intervals = []
        for pl in (pa, pb):
            z = np.asarray(pl["_points"], float)[:, 2]
            z_intervals.append(
                [float(np.percentile(z, 2.0)), float(np.percentile(z, 98.0))]
            )
        z_overlap = min(z_intervals[0][1], z_intervals[1][1]) - max(
            z_intervals[0][0], z_intervals[1][0]
        )
        return {
            "reach_m": [out[0], out[1]],
            "vertical_intervals_m": z_intervals,
            "vertical_overlap_m": max(z_overlap, 0.0),
            "vertical_gap_m": max(-z_overlap, 0.0),
        }

    # First pass: per-claim endpoint, direction, type and geometry adjudication.
    for index, rel in enumerate(rels):
        rel["relationship_id"] = str(rel.get("relationship_id") or f"rel-{index:03d}")
        if not isinstance(rel.get("provenance"), list) or not rel["provenance"]:
            rel["provenance"] = [{"source": "vlm_or_legacy", "action": "proposed"}]
        # Re-adjudication is deterministic and must not retain an obsolete verdict after
        # coplanar wall merging rewrites an endpoint.
        rel["adjudication_reasons"] = []
        rel["reason_codes"] = []
        rel["measurements"] = {}
        rel["evidence"] = {}
        a = str(rel.get("a") or "").strip().lower()
        b = str(rel.get("b") or "").strip().lower()
        rel["a"], rel["b"] = a, b
        t = str(rel.get("type") or "").strip().lower()
        rel["type"] = t
        if t not in RELATIONSHIP_TYPES or not a or not b or a == b:
            _set(rel, "rejected", "none", "malformed or self relationship")
            continue
        missing = [sid for sid in (a, b) if sid not in roots]
        if missing:
            _set(
                rel,
                "rejected",
                "none",
                "endpoint is not a retained root surface: " + ", ".join(missing),
            )
            continue

        ra, ma = _role(a)
        rb, mb = _role(b)
        rel["measurements"].update(
            {"a_role": ra, "b_role": rb, "a_measured": ma, "b_measured": mb}
        )

        if t in {"against", "perpendicular"}:
            # A wall-wall right-angle claim is a CORNER, not a generic perpendicular.
            if t == "perpendicular" and ra == rb == "wall":
                _change_type(rel, "corner", "two wall surfaces use the corner relation")
                t = "corner"
            elif {ra, rb} == {"support", "wall"}:
                support, wall = (a, b) if ra == "support" else (b, a)
                support_measured, wall_measured = (
                    (ma, mb) if ra == "support" else (mb, ma)
                )
                _normalize_direction(
                    rel, support, wall, "support-to-wall canonical order"
                )
                a, b = support, wall
                rel["measurements"].update(
                    {
                        "a_role": "support",
                        "b_role": "wall",
                        "a_measured": support_measured,
                        "b_measured": wall_measured,
                    }
                )
                if t == "against" and _category(support) in _GROUND_CATEGORIES:
                    _reason_code(rel, "invalid_against_ground_wall_roles")
                    _set(
                        rel,
                        "rejected",
                        "none",
                        "floor/ground meets a wall; it is not against it",
                    )
                    continue

                angle = gap = None
                finite = None
                if support_measured and wall_measured:
                    angle = _normal_angle(support, wall)
                    gap = _support_wall_gap(support, wall)
                    finite = _finite_against_metrics(support, wall)
                    rel["measurements"].update(
                        {
                            "normal_angle_deg": (
                                None if angle is None else round(angle, 3)
                            ),
                            "near_edge_gap_m": (
                                None if gap is None else round(gap, 5)
                            ),
                        }
                    )
                    if finite is not None:
                        rel["measurements"].update(finite)

                if t == "against":
                    semantic = _semantic_against_evidence(rel)
                    image_contact = _image_contact_evidence(support, wall)
                    rel["evidence"].update(
                        {
                            "semantic_review": semantic,
                            "image_contact_compatibility": image_contact,
                        }
                    )
                    if (
                        image_contact.get("status") == "contradicted"
                        and image_contact.get("authority") != "authoritative"
                    ):
                        _reason_code(rel, "image_separation_candidate")
                        _reason(
                            rel,
                            "mask boundaries expose a possible separating strip, but "
                            "shadow/baseboard ambiguity keeps it diagnostic",
                        )

                    metric_confirmed = bool(
                        angle is not None
                        and angle >= perpendicular_min_deg
                        and gap is not None
                        and abs(gap) <= gap_max + 1e-5
                        and finite is not None
                        and float(finite["wall_run_gap_m"])
                        <= finite_gap_max + 1e-5
                        and float(finite["vertical_gap_m"])
                        <= finite_gap_max + 1e-5
                    )
                    metric_conflicts = []
                    if gap is not None and abs(gap) > gap_max + 1e-5:
                        metric_conflicts.append(
                            {
                                "kind": "near_edge_gap",
                                "value_m": round(float(gap), 5),
                                "confirmation_max_m": gap_max,
                            }
                        )
                    if finite is not None:
                        for key in ("wall_run_gap_m", "vertical_gap_m"):
                            value = float(finite[key])
                            if value > finite_gap_max + 1e-5:
                                metric_conflicts.append(
                                    {
                                        "kind": key.removesuffix("_m"),
                                        "value_m": round(value, 5),
                                        "confirmation_max_m": finite_gap_max,
                                    }
                                )
                    rel["evidence"]["metric_geometry"] = {
                        "status": (
                            "contact_compatible"
                            if metric_confirmed
                            else "conflicting_diagnostic"
                            if metric_conflicts
                            else "unavailable"
                        ),
                        "role": "confirmation_or_diagnostic",
                        "conflicts": metric_conflicts,
                    }
                    if metric_conflicts:
                        _reason_code(rel, "metric_depth_conflict")
                        _reason(
                            rel,
                            "monocular metric geometry conflicts with flush contact; "
                            "it is diagnostic and cannot veto a retained direct semantic "
                            "AGAINST by itself",
                        )

                    # A reliable role/angle contradiction is semantic, not merely a
                    # noisy depth-gap threshold, so it may reject the whole relation.
                    if angle is not None and angle < perpendicular_min_deg:
                        _reason_code(rel, "incompatible_support_wall_angle")
                        _set(
                            rel,
                            "rejected",
                            "none",
                            "measured surfaces are not support/wall perpendicular",
                        )
                    elif (
                        image_contact.get("status") == "contradicted"
                        and image_contact.get("authority") == "authoritative"
                    ):
                        _reason_code(rel, "visible_contact_separation")
                        _set(
                            rel,
                            "rejected",
                            "none",
                            "a coherent visible separating strip contradicts AGAINST",
                        )
                    elif (
                        semantic["semantic_authoritative_candidate"]
                        and image_contact.get("status") == "compatible"
                    ):
                        _reason_code(rel, "semantic_against_image_compatible")
                        _set(
                            rel,
                            "confirmed",
                            "hard",
                            "the surface auditor retained the direct AGAINST and the "
                            "final masks expose a compatible flush-edge boundary",
                        )
                    elif metric_confirmed:
                        _reason_code(rel, "metric_contact_confirmed")
                        _set(
                            rel,
                            "confirmed",
                            "hard",
                            "measured plane gap, finite wall-run reach, and vertical "
                            "reach agree",
                        )
                    else:
                        _reason_code(rel, "against_contact_unverified")
                        if image_contact.get("status") == "unobservable":
                            reason = (
                                "direct AGAINST contact is not observable in the final "
                                "masks and metric geometry does not confirm it"
                            )
                        elif not semantic["semantic_authoritative_candidate"]:
                            reason = (
                                "image contact is compatible but the direct semantic "
                                "AGAINST was not retained by the surface auditor"
                            )
                        else:
                            reason = "AGAINST contact could not be confirmed"
                        _set(rel, "unverified", "advisory", reason)
                    continue

                # PERPENDICULAR keeps its metric meeting policy.  Geometry may upgrade
                # an already-proposed support/wall PERPENDICULAR to AGAINST, but only
                # inside the original <=5cm metric confirmation band.
                if not (support_measured and wall_measured):
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        "one or both endpoint planes are missing or unreliable",
                    )
                    continue
                if angle is None or angle < perpendicular_min_deg:
                    _set(
                        rel,
                        "rejected",
                        "none",
                        "measured surfaces are not support/wall perpendicular",
                    )
                    continue
                if gap is None:
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        "contact gap could not be measured",
                    )
                    continue
                if abs(gap) >= gap_reject_min - 1e-5:
                    _set(
                        rel,
                        "rejected",
                        "none",
                        f"measured near-edge gap {gap:.3f}m reaches the "
                        f"clear-contradiction threshold {gap_reject_min:.3f}m",
                    )
                    continue
                if abs(gap) > gap_max + 1e-5:
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        f"measured near-edge gap {gap:.3f}m is outside the "
                        f"{gap_max:.3f}m confirmation band but below the "
                        f"{gap_reject_min:.3f}m rejection threshold",
                    )
                    continue
                if finite is None:
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        "finite wall-run or vertical contact could not be measured",
                    )
                    continue
                if not _finite_gap_verdict(
                    rel,
                    [
                        ("wall_run_gap", float(finite["wall_run_gap_m"])),
                        ("vertical_gap", float(finite["vertical_gap_m"])),
                    ],
                    "support and finite wall",
                ):
                    continue
                if _category(support) not in _GROUND_CATEGORIES:
                    _change_type(
                        rel,
                        "against",
                        "measured support edge is flush to wall within 0.05m",
                    )
                _set(
                    rel,
                    "confirmed",
                    "hard",
                    "measured plane gap, finite wall-run reach, and vertical reach agree",
                )
                continue
            elif t != "corner":
                if not (ma and mb):
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        "endpoint roles are not both geometrically measurable as one "
                        "support and one wall",
                    )
                else:
                    _set(
                        rel,
                        "rejected",
                        "none",
                        "relationship endpoints are not one support and one wall",
                    )
                continue

        if t == "corner":
            # Re-read roles because a wall-wall perpendicular may have entered here.
            ra, ma = _role(rel["a"])
            rb, mb = _role(rel["b"])
            rel["measurements"].update(
                {"a_role": ra, "b_role": rb, "a_measured": ma, "b_measured": mb}
            )
            if ra != "wall" or rb != "wall":
                if not (ma and mb):
                    _set(
                        rel,
                        "unverified",
                        "advisory",
                        "corner endpoint roles are not both geometrically measurable",
                    )
                else:
                    _set(rel, "rejected", "none", "corner endpoints are not two walls")
                continue
            if not (ma and mb):
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "one or both wall planes are missing or unreliable",
                )
                continue
            angle = _normal_angle(rel["a"], rel["b"])
            corner_evidence = _corner_reach(rel["a"], rel["b"])
            rel["measurements"]["normal_angle_deg"] = (
                None if angle is None else round(angle, 3)
            )
            if corner_evidence is not None:
                reach = corner_evidence["reach_m"]
                z_intervals = corner_evidence["vertical_intervals_m"]
                z_gap = float(corner_evidence["vertical_gap_m"])
                rel["measurements"].update(
                    corner_reach_m=[round(x, 5) for x in reach],
                    a_vertical_interval_m=[round(x, 5) for x in z_intervals[0]],
                    b_vertical_interval_m=[round(x, 5) for x in z_intervals[1]],
                    vertical_overlap_m=round(
                        float(corner_evidence["vertical_overlap_m"]), 5
                    ),
                    vertical_gap_m=round(z_gap, 5),
                )
            else:
                reach = None
                z_gap = None
            if angle is None or angle < corner_min_deg:
                _set(rel, "rejected", "none", "measured walls are not perpendicular")
            elif reach is not None and max(reach) > corner_reach_max:
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "visible finite wall masks do not establish their plane "
                    f"intersection within {corner_reach_max:.2f}m",
                )
            elif reach is None:
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "finite corner reach could not be measured",
                )
            elif z_gap is None:
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "finite corner vertical overlap could not be measured",
                )
            elif z_gap > finite_gap_max + 1e-5:
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "visible finite wall masks do not establish overlapping "
                    f"vertical corner spans (gap {z_gap:.3f}m)",
                )
            elif z_gap <= 1e-5 and float(corner_evidence["vertical_overlap_m"]) <= 1e-5:
                _set(
                    rel,
                    "unverified",
                    "advisory",
                    "visible vertical spans only touch at a boundary; meaningful "
                    "corner overlap is not established",
                )
            else:
                _set(
                    rel,
                    "confirmed",
                    "hard",
                    "measured wall angle, finite reach, and vertical contact agree",
                )
            continue

        if t == "under":
            ca, cb = _category(a), _category(b)
            # A ground endpoint is safely the lower/supporting endpoint even if its
            # mask is name-only; otherwise use measured roles/heights.
            if cb in _GROUND_CATEGORIES and ca not in _GROUND_CATEGORIES:
                _normalize_direction(
                    rel, b, a, "floor/ground is the lower supporting endpoint"
                )
                a, b, ra, rb, ma, mb = b, a, rb, ra, mb, ma
            elif ca not in _GROUND_CATEGORIES and cb not in _GROUND_CATEGORIES:
                if {ra, rb} == {"support", "wall"} and ra == "wall":
                    _normalize_direction(
                        rel, b, a, "horizontal support is below the wall"
                    )
                    a, b, ra, rb, ma, mb = b, a, rb, ra, mb, ma
                elif ra == rb == "support" and ma and mb:
                    za = float(_plane(a)["centroid"][2])
                    zb = float(_plane(b)["centroid"][2])
                    rel["measurements"]["plane_height_delta_m"] = round(zb - za, 5)
                    if za > zb + 0.02:
                        _normalize_direction(
                            rel, b, a, "lower measured support comes first"
                        )
                        a, b, ra, rb, ma, mb = b, a, rb, ra, mb, ma
            rel["measurements"].update(
                {"a_role": ra, "b_role": rb, "a_measured": ma, "b_measured": mb}
            )
            rel["measurements"]["under_policy_authoritative"] = True
            plausible = ra == "support" and rb in {"support", "wall", "unknown"}
            if not plausible:
                _under_diagnostic(
                    rel,
                    "measured/name-derived endpoint roles do not establish a lower "
                    "supporting surface",
                )
            elif not (ma and mb):
                _under_diagnostic(
                    rel,
                    "vertical support ordering is not fully measurable from the finite masks",
                )
            else:
                xy = _xy_contact_metrics(a, b)
                if xy is None:
                    _under_diagnostic(
                        rel,
                        "finite XY support overlap could not be measured",
                    )
                else:
                    rel["measurements"].update(xy)
                    xy_gap = float(xy["xy_gap_m"])
                    if not bool(xy["xy_intervals_overlap"]):
                        _under_diagnostic(
                            rel,
                            "lower and upper robust XY footprints do not overlap "
                            f"(gap {xy_gap:.3f}m)",
                        )
                if ra == rb == "support":
                    za = float(_plane(a)["centroid"][2])
                    zb = float(_plane(b)["centroid"][2])
                    delta = zb - za
                    rel["measurements"]["plane_height_delta_m"] = round(delta, 5)
                    if delta <= 0.02:
                        _under_diagnostic(
                            rel,
                            "support planes have no clear lower-to-upper order",
                        )
                    elif delta > under_max_vertical_separation:
                        _under_diagnostic(
                            rel,
                            f"lower-to-upper plane separation {delta:.3f}m exceeds "
                            f"the {under_max_vertical_separation:.3f}m diagnostic band",
                        )
                    else:
                        _under_diagnostic(
                            rel,
                            "finite XY overlap and lower-to-upper ordering are plausible, "
                            "but root-surface geometry does not observe the upper "
                            "structure's bottom contact",
                        )
                elif rb == "wall":
                    vertical = _vertical_contact_metrics(a, b)
                    if vertical is None:
                        _under_diagnostic(
                            rel,
                            "finite wall-bottom contact could not be measured",
                        )
                    else:
                        rel["measurements"].update(vertical)
                        vertical_gap = float(vertical["vertical_contact_gap_m"])
                        if vertical_gap > finite_gap_max + 1e-5:
                            _under_diagnostic(
                                rel,
                                "finite lower-support/wall-bottom contact differs by "
                                f"{vertical_gap:.3f}m",
                            )
                else:
                    _under_diagnostic(
                        rel,
                        "upper endpoint role is not geometrically classifiable",
                    )
            _set(
                rel,
                "confirmed",
                "hard",
                "valid UNDER is policy-authoritative; finite-mask geometry is diagnostic only",
            )
            continue

    # Collapse relationships that became duplicates after type/direction normalization.
    # Preserve the duplicate record for audit, but give only one record authority/prompt
    # visibility. Prefer hard over advisory when the same canonical claim appears twice.
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rel in rels:
        if not relationship_is_active(rel):
            continue
        t, a, b = str(rel.get("type")), str(rel.get("a")), str(rel.get("b"))
        if t in {"corner", "perpendicular"}:
            a, b = sorted((a, b))
        key = (t, a, b)
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = rel
            continue
        if relationship_is_hard(rel) and not relationship_is_hard(prior):
            keeper, duplicate = rel, prior
            by_key[key] = rel
        else:
            keeper, duplicate = prior, rel
        _set(
            duplicate,
            "rejected",
            "none",
            f"duplicate of {keeper.get('relationship_id')}",
        )

    # Second pass: measure potential derived-yaw inconsistency without changing source
    # relationship truth.  The constraint compiler/runtime owns joint satisfiability;
    # two confirmed contacts do not become partly false merely because their generated
    # edge-yaw obligations may conflict for an edge-bearing support.
    against_by_support: dict[str, list[dict[str, Any]]] = {}
    for rel in rels:
        if rel.get("type") == "against" and relationship_is_hard(rel):
            against_by_support.setdefault(str(rel.get("a")), []).append(rel)
    constraint_conflict_pairs: list[dict[str, Any]] = []
    potential_conflict_pairs: list[dict[str, Any]] = []
    for support, group in against_by_support.items():
        rectangularity = _support_rectangularity(support)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ri, rj = group[i], group[j]
                ui, uj = _wall_run(str(ri.get("b"))), _wall_run(str(rj.get("b")))
                if ui is None or uj is None:
                    continue
                dot = min(1.0, abs(ui[0] * uj[0] + ui[1] * uj[1]))
                raw = math.degrees(math.acos(dot))  # [0, 90], mod 180
                folded = min(raw, abs(90.0 - raw))  # rectangular edge family, mod 90
                if folded <= against_run_conflict_deg:
                    continue
                pair = {
                    "support": support,
                    "relationship_ids": [ri["relationship_id"], rj["relationship_id"]],
                    "wall_ids": [ri.get("b"), rj.get("b")],
                    "run_disagreement_mod90_deg": round(folded, 3),
                    "support_rectangularity": (
                        None if rectangularity is None else round(rectangularity, 4)
                    ),
                }
                # A disc/axisless support may touch non-orthogonal walls without a yaw
                # contradiction.  Unknown applicability is decided from the built TOP
                # hull; none of these diagnostics mutates the relationship verdict.
                if rectangularity is None:
                    pair["resolution"] = "defer_to_runtime_top_hull"
                    potential_conflict_pairs.append(pair)
                    for rel in (ri, rj):
                        rel["measurements"].setdefault(
                            "potential_against_conflicts", []
                        ).append(pair)
                    continue
                if rectangularity < edge_bearing_rectangularity:
                    pair["resolution"] = "axisless_candidate_no_yaw_conflict"
                    potential_conflict_pairs.append(pair)
                    for rel in (ri, rj):
                        rel["measurements"].setdefault(
                            "potential_against_conflicts", []
                        ).append(pair)
                    continue
                pair["resolution"] = "defer_to_constraint_set_diagnostics"
                constraint_conflict_pairs.append(pair)
                for rel in (ri, rj):
                    rel["measurements"].setdefault(
                        "potential_against_constraint_conflicts", []
                    ).append(pair)

    invalid_verdict_ids = [
        str(rel.get("relationship_id"))
        for rel in rels
        if not relationship_schema_is_valid(rel)
    ]
    if invalid_verdict_ids:
        raise ValueError(
            "relationship adjudication emitted invalid atomic verdict(s): "
            + ", ".join(invalid_verdict_ids)
        )

    counts = {
        key: sum(1 for rel in rels if rel.get("enforcement") == key)
        for key in ("hard", "advisory", "none")
    }
    status_counts = {
        key: sum(1 for rel in rels if rel.get("status") == key)
        for key in sorted(RELATIONSHIP_STATUSES)
    }
    return {
        "schema_version": RELATIONSHIP_SCHEMA_VERSION,
        "counts_by_enforcement": counts,
        "counts_by_status": status_counts,
        "against_constraint_conflicts": constraint_conflict_pairs,
        "potential_against_conflicts": potential_conflict_pairs,
        "parameters": {
            "gap_max_m": gap_max,
            "gap_reject_min_m": gap_reject_min,
            "finite_gap_max_m": finite_gap_max,
            "finite_gap_reject_min_m": finite_gap_reject_min,
            "under_max_vertical_separation_m": under_max_vertical_separation,
            "perpendicular_min_deg": perpendicular_min_deg,
            "corner_min_deg": corner_min_deg,
            "corner_reach_max_m": corner_reach_max,
            "against_run_conflict_deg_mod90": against_run_conflict_deg,
            "edge_bearing_rectangularity": edge_bearing_rectangularity,
        },
    }


def relationship_glossary() -> str:
    """A bulleted ``- <type>: <meaning>`` block for the masking-pipeline VLM prompts."""
    return "\n".join(f"- {t}: {m}" for t, m in RELATIONSHIP_MEANINGS.items())


# Imperative build phrasings (id placeholders filled with agent-facing surface names), used
# in the static-scene initializer's per-scene data so the agent builds surfaces honoring each
# relationship.
_BUILD_TEMPLATES: dict[str, str] = {
    "against": "{a} is AGAINST {b}: put {a}'s near edge flush IN {b}'s plane — connected, "
    "not floating off {b} nor crossing through it.",
    "perpendicular": "{a} and {b} meet at a RIGHT ANGLE (perpendicular, used for one "
    "horizontal surface and one vertical surface, e.g., a wall and a floor).",
    "corner": "{a} and {b} are two walls sharing one VERTICAL CORNER edge — both finite walls "
    "must reach that edge; extending past it is allowed because the overshoot hides behind the "
    "other wall.",
    "under": "{a} is directly UNDER {b}: build {a}'s top meeting the BOTTOM of {b}'s complete "
    "supporting structure. For a floor under a table/desk/counter, the floor meets the bottom "
    "of connected legs/pedestal/cabinet — NEVER the underside of a bare tabletop slab and "
    "NEVER raise the floor to tabletop height. For a floor under a wall, the wall bottom meets "
    "the floor.",
}


def relationship_build_line(rel: dict[str, str], name_a: str, name_b: str) -> str:
    """One imperative build instruction for a relationship, with agent-facing names
    substituted for the raw ids (so the initializer reads names, not ``cat#k``). Falls back
    to the glossary meaning for an unknown type."""
    t = rel.get("type", "")
    tmpl = _BUILD_TEMPLATES.get(t)
    return (
        tmpl.format(a=name_a, b=name_b)
        if tmpl
        else f"{name_a} {t} {name_b}: {RELATIONSHIP_MEANINGS.get(t, '')}".strip()
    )
