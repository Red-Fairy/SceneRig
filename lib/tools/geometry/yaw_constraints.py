"""Resolve immutable source yaw evidence before initializer execution.

The observable-edge extractor and relationship adjudicator answer different
questions.  This module combines their *independent* source evidence without
changing either source verdict:

* image edges and any image-mask PCA share one evidence group;
* every wall measurement shares one evidence group;
* only a retained, non-contradicted advisory AGAINST may contribute a weak wall
  direction;
* hard AGAINST remains a relationship-owned compiled constraint; and
* PERPENDICULAR never pins a horizontal support's footprint yaw.

The resolver is deliberately pure.  Runtime Blender geometry is not an input, so a
later wall edit cannot move a frozen photo/corroborated target.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane

YAW_EVIDENCE_SCHEMA_VERSION = 1
YAW_RESOLVER_VERSION = "source_yaw_resolver_v1"
YAW_CORROBORATION_MAX_DEGREES = 6.0
YAW_CONFLICT_MIN_DEGREES = 15.0
YAW_REQUIRED_TOLERANCE_DEGREES = 12.0

_IMAGE_GROUP = "support_image_geometry"
_WALL_GROUP = "related_wall_geometry"


class YawEvidenceError(ValueError):
    """Source yaw evidence is malformed or cannot be resolved deterministically."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise YawEvidenceError(f"yaw evidence is not canonical JSON: {exc}") from exc


def yaw_observation_digest(observation: dict[str, Any]) -> str:
    if not isinstance(observation, dict):
        raise YawEvidenceError("main-support yaw observation must be an object")
    return hashlib.sha256(_canonical_json(observation)).hexdigest()


def normalize_run_xy(value: Any, *, label: str = "run") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise YawEvidenceError(f"{label} must contain at least two coordinates")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise YawEvidenceError(f"{label} must be numeric") from exc
    length = math.hypot(x, y)
    if not math.isfinite(length) or length < 1e-8:
        raise YawEvidenceError(f"{label} must be a finite non-zero XY direction")
    return [round(x / length, 12), round(y / length, 12)]


def yaw_delta_degrees_mod_90(run_a: Any, run_b: Any) -> float:
    """Signed rotation from line ``run_a`` to line ``run_b`` in [-45, 45)."""

    a = normalize_run_xy(run_a, label="run_a")
    b = normalize_run_xy(run_b, label="run_b")
    angle_a = math.degrees(math.atan2(a[1], a[0]))
    angle_b = math.degrees(math.atan2(b[1], b[0]))
    return (angle_b - angle_a + 45.0) % 90.0 - 45.0


def _mean_run_mod_90(candidates: Iterable[Any]) -> list[float]:
    """Circular mean of unoriented, perpendicular-equivalent edge families."""

    runs = [normalize_run_xy(run) for run in candidates]
    if not runs:
        raise YawEvidenceError("cannot average an empty yaw candidate set")
    sx = sy = 0.0
    for x, y in runs:
        theta = math.atan2(y, x)
        sx += math.cos(4.0 * theta)
        sy += math.sin(4.0 * theta)
    if math.hypot(sx, sy) < 1e-8:
        raise YawEvidenceError("yaw candidates have no stable modulo-90 consensus")
    theta = math.atan2(sy, sx) / 4.0
    return [round(math.cos(theta), 12), round(math.sin(theta), 12)]


def _candidate(
    *,
    evidence_id: str,
    provider: str,
    group: str,
    run: Any,
    strength: str,
    eligible: bool,
    reason_codes: Iterable[str] = (),
    relationship_id: str | None = None,
    surface_id: str | None = None,
    angular_uncertainty_degrees: Any = None,
) -> dict[str, Any]:
    reasons = sorted({str(code) for code in reason_codes if code})
    normalized_run = None
    if eligible:
        try:
            normalized_run = normalize_run_xy(run, label=f"{evidence_id}.run")
        except YawEvidenceError:
            eligible = False
            reasons.append("invalid_or_degenerate_run")
    uncertainty = None
    if angular_uncertainty_degrees is not None:
        try:
            uncertainty = float(angular_uncertainty_degrees)
            if not math.isfinite(uncertainty) or uncertainty < 0.0:
                uncertainty = None
        except (TypeError, ValueError):
            uncertainty = None
    return {
        "evidence_id": str(evidence_id),
        "provider": str(provider),
        "independence_group": str(group),
        "run_xy": normalized_run,
        "strength": "strong" if strength == "strong" else "weak",
        "eligible": bool(eligible),
        "angular_uncertainty_degrees": uncertainty,
        "relationship_id": relationship_id,
        "surface_id": surface_id,
        "reason_codes": sorted(set(reasons)),
    }


def _image_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(observation.get("candidates") or []):
        if not isinstance(raw, dict):
            continue
        provider = str(raw.get("provider") or "observable_top_edges")
        strength = str(raw.get("strength") or "weak").strip().lower()
        reasons = list(raw.get("reason_codes") or [])
        # PCA is retained as a diagnostic weak clue only.  Eccentricity can never
        # turn it into a required source by itself.
        if "pca" in provider.casefold() and strength == "strong":
            strength = "weak"
            reasons.append("pca_cannot_supply_strong_authority")
        result.append(
            _candidate(
                evidence_id=str(raw.get("evidence_id") or f"image-candidate-{index}"),
                provider=provider,
                group=_IMAGE_GROUP,
                run=raw.get("run_world") or raw.get("run_xy"),
                strength=strength,
                eligible=raw.get("eligible") is True,
                reason_codes=reasons,
                angular_uncertainty_degrees=raw.get(
                    "angular_uncertainty_degrees"
                ),
            )
        )
    return result


def _relationship_is_noncontradicted_advisory_against(rel: dict[str, Any]) -> bool:
    if (
        rel.get("type") != "against"
        or rel.get("status") != "unverified"
        or rel.get("enforcement") != "advisory"
    ):
        return False
    reason_codes = {str(code) for code in rel.get("reason_codes") or []}
    if reason_codes & {
        "visible_contact_separation",
        "incompatible_support_wall_angle",
        "invalid_against_ground_wall_roles",
    }:
        return False
    image = ((rel.get("evidence") or {}).get("image_contact_compatibility") or {})
    if str(image.get("status") or "").casefold() == "contradicted":
        return False
    return True


def advisory_against_wall_candidates(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return weak canonical-wall candidates from direct advisory AGAINST only."""

    main_id = str(graph.get("main_support_id") or "")
    nodes = {
        str(node.get("id")): node
        for node in graph.get("nodes", []) or []
        if isinstance(node, dict) and node.get("id")
    }
    result: list[dict[str, Any]] = []
    for rel in graph.get("relationships", []) or []:
        if not isinstance(rel, dict) or not _relationship_is_noncontradicted_advisory_against(rel):
            continue
        a, b = str(rel.get("a") or ""), str(rel.get("b") or "")
        if main_id not in {a, b}:
            continue
        wall_id = b if a == main_id else a
        node = nodes.get(wall_id) or {}
        plane = node.get("plane") or {}
        reasons: list[str] = []
        eligible = True
        normal, kind = plumb_plane(plane.get("normal"))
        if node.get("runtime_added") is True:
            eligible = False
            reasons.append("runtime_added_wall_is_not_source_evidence")
        elif node.get("kind") != "root_surface":
            eligible = False
            reasons.append("wall_node_missing")
        elif kind != "wall" or normal is None:
            eligible = False
            reasons.append("partner_is_not_vertical_wall")
        elif not plane_is_reliable(plane):
            eligible = False
            reasons.append("canonical_wall_plane_unreliable")
        run = None
        if eligible:
            horizontal = math.hypot(float(normal[0]), float(normal[1]))
            if horizontal < 1e-8:
                eligible = False
                reasons.append("canonical_wall_run_degenerate")
            else:
                run = [-float(normal[1]) / horizontal, float(normal[0]) / horizontal]
        result.append(
            _candidate(
                evidence_id=f"advisory-against:{rel.get('relationship_id')}",
                provider="canonical_advisory_against_wall",
                group=_WALL_GROUP,
                run=run,
                strength="weak",
                eligible=eligible,
                reason_codes=reasons,
                relationship_id=str(rel.get("relationship_id") or ""),
                surface_id=wall_id,
            )
        )
    return result


def _representative(
    candidates: list[dict[str, Any]], *, agreement_degrees: float
) -> tuple[dict[str, Any] | None, bool]:
    usable = [candidate for candidate in candidates if candidate.get("eligible")]
    if not usable:
        return None, False
    # A single-reference test admits a fan straddling that reference (for example
    # -5.9°, 0°, +5.9°) even though the group's observable spread is 11.8°.
    # Independence-group agreement is therefore the maximum *pairwise* circular
    # distance, evaluated on the modulo-90 line space.
    conflict = any(
        abs(
            yaw_delta_degrees_mod_90(
                usable[left]["run_xy"], usable[right]["run_xy"]
            )
        )
        > agreement_degrees
        for left in range(len(usable))
        for right in range(left + 1, len(usable))
    )
    if conflict:
        return None, True
    run = _mean_run_mod_90(candidate["run_xy"] for candidate in usable)
    strength = "strong" if any(c.get("strength") == "strong" for c in usable) else "weak"
    return {
        "run_xy": run,
        "strength": strength,
        "candidate_ids": [str(c["evidence_id"]) for c in usable],
        "independence_group": str(usable[0]["independence_group"]),
    }, False


def resolve_source_yaw_evidence(
    graph: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    """Resolve image and advisory-wall evidence into one immutable source decision."""

    main_surface = observation.get("main_surface") or {}
    main_id = str(main_surface.get("id") or "")
    if not main_id or main_id != str(graph.get("main_support_id") or ""):
        raise YawEvidenceError(
            "yaw observation main surface does not match scene graph main_support_id"
        )
    applicability = observation.get("applicability") or {}
    applicability_status = str(applicability.get("status") or "unverified")
    image = _image_candidates(observation)
    walls = advisory_against_wall_candidates(graph)
    candidates = image + walls
    seen: set[str] = set()
    for candidate in candidates:
        evidence_id = str(candidate["evidence_id"])
        if evidence_id in seen:
            raise YawEvidenceError(f"duplicate source yaw evidence id {evidence_id!r}")
        seen.add(evidence_id)

    def _decision(
        authority: str,
        status: str,
        *,
        run: Any = None,
        ids: Iterable[str] = (),
        reasons: Iterable[str] = (),
    ) -> dict[str, Any]:
        return {
            "authority": authority,
            "status": status,
            "target_run_xy": (
                normalize_run_xy(run, label="resolved target run")
                if run is not None
                else None
            ),
            "candidate_ids": [str(value) for value in ids],
            "reason_codes": sorted({str(value) for value in reasons if value}),
            "tolerance_degrees_mod_90": (
                YAW_REQUIRED_TOLERANCE_DEGREES if authority == "required" else None
            ),
        }

    if applicability_status == "not_applicable":
        decision = _decision(
            "not_applicable",
            "not_applicable",
            reasons=(applicability.get("reason_codes") or ["source_yaw_not_applicable"]),
        )
    elif applicability_status != "applicable":
        decision = _decision(
            "unverified",
            "unverified",
            reasons=(applicability.get("reason_codes") or ["source_yaw_applicability_unverified"]),
        )
    else:
        observation_decision = observation.get("decision") or {}
        observation_conflicting = (
            str(observation_decision.get("status") or "") == "conflicting"
        )
        image_rep, image_conflict = _representative(
            image, agreement_degrees=YAW_CORROBORATION_MAX_DEGREES
        )
        if image_rep and not (
            observation_decision.get("status") == "confirmed"
            and observation_decision.get("authority") == "required"
        ):
            image_rep["strength"] = "weak"
        wall_rep, wall_conflict = _representative(
            walls, agreement_degrees=YAW_CORROBORATION_MAX_DEGREES
        )
        if observation_conflicting or image_conflict:
            decision = _decision(
                "conflicting",
                "conflicting",
                reasons=("conflicting_image_yaw_candidates",),
            )
        elif (
            image_rep
            and image_rep["strength"] == "strong"
            and observation_decision.get("status") == "confirmed"
            and observation_decision.get("authority") == "required"
        ):
            extractor_target = observation_decision.get("target_run_world")
            decision = _decision(
                "required",
                "confirmed",
                run=extractor_target or image_rep["run_xy"],
                ids=image_rep["candidate_ids"],
                reasons=("two_observable_edge_families",),
            )
        elif wall_conflict:
            decision = _decision(
                "conflicting",
                "conflicting",
                ids=(image_rep or {}).get("candidate_ids", ()),
                reasons=("related_wall_yaw_candidates_disagree",),
            )
        elif image_rep and wall_rep:
            delta = abs(
                yaw_delta_degrees_mod_90(image_rep["run_xy"], wall_rep["run_xy"])
            )
            ids = image_rep["candidate_ids"] + wall_rep["candidate_ids"]
            if delta <= YAW_CORROBORATION_MAX_DEGREES:
                decision = _decision(
                    "required",
                    "confirmed",
                    run=_mean_run_mod_90(
                        [image_rep["run_xy"], wall_rep["run_xy"]]
                    ),
                    ids=ids,
                    reasons=("independent_weak_sources_corroborate",),
                )
            elif delta <= YAW_CONFLICT_MIN_DEGREES:
                decision = _decision(
                    "unverified",
                    "unverified",
                    ids=ids,
                    reasons=("weak_sources_outside_corroboration_band",),
                )
            else:
                decision = _decision(
                    "conflicting",
                    "conflicting",
                    ids=ids,
                    reasons=("independent_yaw_sources_conflict",),
                )
        elif image_rep or wall_rep:
            representative = image_rep or wall_rep
            decision = _decision(
                "advisory",
                "advisory",
                run=representative["run_xy"],
                ids=representative["candidate_ids"],
                reasons=("single_independent_yaw_group",),
            )
        else:
            decision = _decision(
                "unverified",
                "unverified",
                reasons=("no_usable_source_yaw_candidate",),
            )

    decision_id = f"source-yaw:{main_id}"
    return {
        "schema_version": YAW_EVIDENCE_SCHEMA_VERSION,
        "resolver_version": YAW_RESOLVER_VERSION,
        "decision_id": decision_id,
        "main_surface_id": main_id,
        "source_candidates": candidates,
        "source_decision": decision,
    }
