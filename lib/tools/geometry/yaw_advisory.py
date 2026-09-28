"""Bounded contracts and deterministic checks for residual yaw advisories.

This module deliberately does not decide source-evidence authority and does not
compile constraints.  It handles the smaller problem left *after* the initializer
ladder has passed: issuing opaque, state-bound candidate identifiers and validating
an agent-selected correspondence without trusting model-provided coordinates,
angles, tolerances, or conclusions.

The helpers are Blender-free so the executor, generator, root handoff, and focused
tests all use the same strict vocabulary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

SCHEMA_VERSION = 1
ADVISORY_STATES = frozenset({"required", "manual_review", "not_required"})
RESOLUTION_STATUSES = frozenset(
    {"verified_match", "verified_mismatch", "unverified", "stale", "not_applicable"}
)
# A corner-based method is intentionally absent until the backend measures the
# submitted corner displacement. Merely validating a corner ID is not evidence.
ALLOWED_METHODS = frozenset({"two_edge_families"})
REVIEW_DECISIONS = frozenset({"acceptable", "mismatch", "unverified"})
MAX_CANDIDATES = 12
MAX_REASON_CODES = 12
MAX_ID_LENGTH = 160
# Manual-review readings are short observational prose, not identifiers.  Keep a
# separate generous-but-bounded limit so useful pixel/edge evidence is not rejected
# merely because it needs more detail than an opaque ID.
MAX_REVIEW_READING_LENGTH = 512


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > MAX_ID_LENGTH:
        raise ValueError(f"{field} is too long")
    return normalized


def _review_reading(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > MAX_REVIEW_READING_LENGTH:
        raise ValueError(f"{field} exceeds {MAX_REVIEW_READING_LENGTH} characters")
    return normalized


def _finite_pair(value: Any, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain two numbers")
    pair = [float(value[0]), float(value[1])]
    if not all(math.isfinite(component) for component in pair):
        raise ValueError(f"{field} must be finite")
    return pair


def _unit_run(value: Any, field: str = "run_xy") -> list[float]:
    run = _finite_pair(value, field)
    norm = math.hypot(run[0], run[1])
    if norm <= 1e-9:
        raise ValueError(f"{field} must be non-zero")
    # Schema normalization is also used before hashing an issued request.  Dividing
    # an already-unit diagonal twice can change its final float bit, making the mint
    # and immediate resolve bindings disagree even though no scene input changed.
    # Preserve an already-normalized representation so normalize(normalize(x)) is
    # byte-stable while retaining ordinary normalization for real non-unit inputs.
    if abs(norm - 1.0) <= 1e-12:
        return run
    return [run[0] / norm, run[1] / norm]


def _reason_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("reason_codes must be an array")
    result: list[str] = []
    for raw in value[:MAX_REASON_CODES]:
        code = _identifier(raw, "reason_code")
        if code not in result:
            result.append(code)
    return result


def _candidate(value: Any, *, public: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("candidate must be an object")
    candidate_id = _identifier(
        value.get("candidate_id") or value.get("id") or value.get("evidence_id"),
        "candidate_id",
    )
    kind = str(value.get("kind") or value.get("type") or "edge").strip().lower()
    if kind not in {"edge", "corner"}:
        raise ValueError("candidate.kind must be edge or corner")
    result: dict[str, Any] = {"candidate_id": candidate_id, "kind": kind}
    source = value.get("source")
    if isinstance(source, str) and source.strip():
        result["source"] = source.strip()[:80]
    family = value.get("family")
    if isinstance(family, str) and family.strip():
        result["family"] = family.strip()[:80]
    if public:
        return result

    run = value.get("run_xy")
    if run is not None:
        result["run_xy"] = _unit_run(run)
    endpoints = value.get("endpoints_px")
    if endpoints is not None:
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            raise ValueError("candidate.endpoints_px must contain two points")
        result["endpoints_px"] = [
            _finite_pair(endpoints[0], "endpoints_px[0]"),
            _finite_pair(endpoints[1], "endpoints_px[1]"),
        ]
    endpoints_world = value.get("endpoints_world_xy")
    if endpoints_world is not None:
        if not isinstance(endpoints_world, (list, tuple)) or len(endpoints_world) != 2:
            raise ValueError("candidate.endpoints_world_xy must contain two points")
        result["endpoints_world_xy"] = [
            _finite_pair(endpoints_world[0], "endpoints_world_xy[0]"),
            _finite_pair(endpoints_world[1], "endpoints_world_xy[1]"),
        ]
    point = value.get("point_px")
    if point is not None:
        result["point_px"] = _finite_pair(point, "point_px")
    point_world = value.get("point_world_xy")
    if point_world is not None:
        result["point_world_xy"] = _finite_pair(point_world, "point_world_xy")
    return result


def normalize_candidates(value: Any, *, public: bool = False) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("candidates must be an array")
    candidates = [_candidate(raw, public=public) for raw in value[:MAX_CANDIDATES]]
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    return candidates


def normalize_advisory_requirement(
    value: Any, *, public: bool = False
) -> dict[str, Any]:
    """Validate and bound one executor-issued residual-advisory request.

    ``public=True`` removes the opaque observation token and all measurements while
    retaining candidate identifiers.  Root uses that form for the verifier handoff.
    """

    if not isinstance(value, dict):
        raise ValueError("advisory_requirement must be an object")
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported advisory_requirement schema_version")
    state = str(value.get("state") or "").strip().lower()
    if state not in ADVISORY_STATES:
        raise ValueError("invalid advisory_requirement state")
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "state": state}
    result["reason_codes"] = _reason_codes(value.get("reason_codes"))

    if state == "not_required":
        return result

    result["advisory_id"] = _identifier(value.get("advisory_id"), "advisory_id")
    if not public:
        result["observation_token"] = _identifier(
            value.get("observation_token"), "observation_token"
        )
    methods = value.get("allowed_methods") or []
    if not isinstance(methods, list):
        raise ValueError("allowed_methods must be an array")
    normalized_methods: list[str] = []
    for method in methods:
        name = str(method).strip()
        if name not in ALLOWED_METHODS:
            raise ValueError(f"unsupported advisory method {name!r}")
        if name not in normalized_methods:
            normalized_methods.append(name)
    result["allowed_methods"] = normalized_methods
    result["target_candidates"] = normalize_candidates(
        value.get("target_candidates"), public=public
    )
    result["built_candidates"] = normalize_candidates(
        value.get("built_candidates"), public=public
    )
    if state == "required" and not normalized_methods:
        raise ValueError("required advisory has no machine-verifiable method")
    return result


def normalize_resolution_submission(value: Any) -> dict[str, Any]:
    """Accept identifiers only; reject model-supplied coordinates/conclusions."""

    if not isinstance(value, dict):
        raise ValueError("yaw advisory resolution must be an object")
    allowed_keys = {
        "advisory_id",
        "observation_token",
        "method",
        "edge_matches",
    }
    extras = sorted(set(value) - allowed_keys)
    if extras:
        raise ValueError(
            "yaw advisory resolution accepts candidate ids only; unexpected fields: "
            + ", ".join(extras)
        )
    method = str(value.get("method") or "").strip()
    if method not in ALLOWED_METHODS:
        raise ValueError("invalid yaw advisory resolution method")
    matches = value.get("edge_matches")
    if not isinstance(matches, list) or not (1 <= len(matches) <= 2):
        raise ValueError("edge_matches must contain one or two matches")
    clean_matches: list[dict[str, str]] = []
    for raw in matches:
        if not isinstance(raw, dict) or set(raw) != {
            "target_edge_id",
            "built_edge_id",
        }:
            raise ValueError("each edge match must contain only two candidate ids")
        clean_matches.append(
            {
                "target_edge_id": _identifier(
                    raw.get("target_edge_id"), "target_edge_id"
                ),
                "built_edge_id": _identifier(raw.get("built_edge_id"), "built_edge_id"),
            }
        )
    if len({m["target_edge_id"] for m in clean_matches}) != len(clean_matches):
        raise ValueError("target edge ids must be unique")
    if len({m["built_edge_id"] for m in clean_matches}) != len(clean_matches):
        raise ValueError("built edge ids must be unique")

    result: dict[str, Any] = {
        "advisory_id": _identifier(value.get("advisory_id"), "advisory_id"),
        "observation_token": _identifier(
            value.get("observation_token"), "observation_token"
        ),
        "method": method,
        "edge_matches": clean_matches,
    }
    if method == "two_edge_families" and len(clean_matches) != 2:
        raise ValueError("two_edge_families requires exactly two edge matches")
    return result


def normalize_yaw_resolution(value: Any, *, public: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("yaw_resolution must be an object")
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported yaw_resolution schema_version")
    status = str(value.get("status") or "").strip().lower()
    if status not in RESOLUTION_STATUSES:
        raise ValueError("invalid yaw_resolution status")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "advisory_id": _identifier(value.get("advisory_id"), "advisory_id"),
        "status": status,
        "reason_codes": _reason_codes(value.get("reason_codes")),
    }
    if not public and value.get("observation_token") is not None:
        result["observation_token"] = _identifier(
            value.get("observation_token"), "observation_token"
        )
    method = value.get("method")
    if method is not None:
        method = str(method).strip()
        if method not in ALLOWED_METHODS:
            raise ValueError("invalid yaw_resolution method")
        result["method"] = method
    matches = value.get("validated_matches")
    if isinstance(matches, list):
        result["validated_matches"] = [
            {
                "target_edge_id": _identifier(
                    m.get("target_edge_id"), "target_edge_id"
                ),
                "built_edge_id": _identifier(m.get("built_edge_id"), "built_edge_id"),
            }
            for m in matches[:2]
            if isinstance(m, dict)
        ]
    measurements = value.get("measurements")
    if isinstance(measurements, dict):
        compact: dict[str, Any] = {}
        for key in (
            "max_delta_degrees_mod_90",
            "mean_delta_degrees_mod_90",
            "pair_disagreement_degrees",
            "tolerance_degrees",
        ):
            raw = measurements.get(key)
            if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                compact[key] = round(float(raw), 6)
        if compact:
            result["measurements"] = compact
    return result


def normalize_manual_review(value: Any, expected_advisory_id: str) -> dict[str, Any]:
    """Validate the verifier's mandatory structured manual-review row."""

    if not isinstance(value, dict):
        raise ValueError("yaw_advisory_review must be an object")
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported yaw_advisory_review schema_version")
    advisory_id = _identifier(value.get("advisory_id"), "advisory_id")
    if advisory_id != expected_advisory_id:
        raise ValueError("yaw_advisory_review advisory_id does not match the request")
    decision = str(value.get("decision") or "").strip().lower()
    if decision not in REVIEW_DECISIONS:
        raise ValueError("invalid yaw_advisory_review decision")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence or len(evidence) > 4:
        raise ValueError("yaw_advisory_review evidence must contain 1-4 rows")
    clean_evidence: list[dict[str, str]] = []
    for row in evidence:
        if not isinstance(row, dict):
            raise ValueError("yaw_advisory_review evidence row must be an object")
        if set(row) != {"anchor_type", "target_reading", "render_reading"}:
            raise ValueError(
                "yaw_advisory_review evidence row contains untrusted fields"
            )
        anchor_type = str(row.get("anchor_type") or "").strip().lower()
        if anchor_type not in {"edge", "corner", "mask_boundary", "object_gap"}:
            raise ValueError("invalid yaw advisory anchor_type")
        target = _review_reading(row.get("target_reading"), "target_reading")
        render = _review_reading(row.get("render_reading"), "render_reading")
        clean_evidence.append(
            {
                "anchor_type": anchor_type,
                "target_reading": target,
                "render_reading": render,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "advisory_id": advisory_id,
        "decision": decision,
        "evidence": clean_evidence,
        "reason_codes": _reason_codes(value.get("reason_codes")),
    }


def built_candidates_from_top_hull(hull: Any) -> list[dict[str, Any]]:
    """Issue deterministic ids for exact built top-hull edges and corners."""

    if not isinstance(hull, list) or len(hull) < 3:
        return []
    try:
        points = [_finite_pair(point, "top_hull point") for point in hull[:64]]
    except (TypeError, ValueError):
        return []
    candidates: list[dict[str, Any]] = []
    edge_rows: list[tuple[float, int, list[float], list[list[float]]]] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        edge_rows.append((length, index, [dx / length, dy / length], [start, end]))
    # Long physical sides are useful; tiny tessellation chords are not. Preserve the
    # original hull index in the id so ordering/selection is deterministic.
    for _length, index, run, endpoints in sorted(edge_rows, reverse=True)[:8]:
        candidates.append(
            {
                "candidate_id": f"built-edge-{index}",
                "kind": "edge",
                "source": "exact_built_top_hull",
                "run_xy": run,
                "endpoints_world_xy": endpoints,
            }
        )
    for index, point in enumerate(points[:4]):
        candidates.append(
            {
                "candidate_id": f"built-corner-{index}",
                "kind": "corner",
                "source": "exact_built_top_hull",
                "point_world_xy": point,
            }
        )
    return candidates[:MAX_CANDIDATES]


def project_built_candidates(
    candidates: list[dict[str, Any]],
    camera: Any,
    *,
    top_z: float = 0.0,
) -> list[dict[str, Any]]:
    """Attach source-view pixels to exact built top-hull candidates.

    Projection is backend-owned and calibrated. Candidates that do not project in
    front of the camera are omitted rather than handed to the model as misleading IDs.
    """
    if not isinstance(camera, dict):
        return []
    try:
        plane_z = float(top_z)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(plane_z):
        return []
    from lib.tools.geometry.moge_camera import project_world_to_pixel

    projected: list[dict[str, Any]] = []
    for raw in candidates[:MAX_CANDIDATES]:
        candidate = dict(raw)
        try:
            if candidate.get("kind") == "edge":
                endpoints = candidate.get("endpoints_world_xy")
                if not isinstance(endpoints, list) or len(endpoints) != 2:
                    continue
                pixels = [
                    project_world_to_pixel([point[0], point[1], plane_z], camera)
                    for point in endpoints
                ]
                if any(
                    not all(math.isfinite(float(v)) for v in pixel[:2])
                    or float(pixel[2]) <= 0
                    for pixel in pixels
                ):
                    continue
                candidate["endpoints_px"] = [
                    [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]
                    for pixel in pixels
                ]
            elif candidate.get("kind") == "corner":
                point = candidate.get("point_world_xy")
                if not isinstance(point, list) or len(point) != 2:
                    continue
                pixel = project_world_to_pixel([point[0], point[1], plane_z], camera)
                if (
                    not all(math.isfinite(float(v)) for v in pixel[:2])
                    or float(pixel[2]) <= 0
                ):
                    continue
                candidate["point_px"] = [
                    round(float(pixel[0]), 3),
                    round(float(pixel[1]), 3),
                ]
            projected.append(candidate)
        except (TypeError, ValueError, KeyError):
            continue
    return projected


def source_candidates_from_observation(observation: Any) -> list[dict[str, Any]]:
    """Issue eligible physical segment IDs from a strictly validated artifact.

    A resolved/folded orientation candidate is not two physical edges. Machine
    correspondence therefore consumes the artifact's individual observable segments,
    retaining their calibrated world runs and source-pixel endpoints.
    """
    if not isinstance(observation, dict):
        return []
    if observation.get("extractor_version") == "observable_top_edges_v1":
        return []
    candidates: list[dict[str, Any]] = []
    for raw in observation.get("segments") or []:
        if not isinstance(raw, dict) or raw.get("eligible") is not True:
            continue
        run = raw.get("world_run")
        endpoints = raw.get("endpoints_px")
        if run is None or endpoints is None:
            continue
        try:
            candidate = _candidate(
                {
                    "candidate_id": raw.get("segment_id"),
                    "kind": "edge",
                    "source": "observable_top_edge_segment",
                    "family": str(raw.get("family") or "physical_segment"),
                    "run_xy": list(run)[:2],
                    "endpoints_px": endpoints,
                }
            )
        except (TypeError, ValueError):
            continue
        candidates.append(candidate)
    return candidates[:MAX_CANDIDATES]


def source_candidates_from_yaw_evidence(yaw: Any) -> list[dict[str, Any]]:
    """Read only typed source candidates; never infer coordinates from prose."""

    if not isinstance(yaw, dict):
        return []
    raw = yaw.get("source_candidates")
    if not isinstance(raw, list):
        raw = yaw.get("anchor_candidates")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw[:MAX_CANDIDATES]):
        if not isinstance(candidate, dict):
            continue
        normalized = dict(candidate)
        normalized.setdefault(
            "candidate_id",
            candidate.get("evidence_id") or candidate.get("id") or f"source-{index}",
        )
        normalized.setdefault("kind", "edge")
        try:
            clean = _candidate(normalized)
        except ValueError:
            continue
        # A direction must come from the calibrated source extractor. Pixel endpoints
        # alone cannot be compared with a world-space built edge.
        if clean.get("kind") == "edge" and "run_xy" not in clean:
            continue
        result.append(clean)
    ids = set()
    unique: list[dict[str, Any]] = []
    for candidate in result:
        if candidate["candidate_id"] in ids:
            continue
        ids.add(candidate["candidate_id"])
        unique.append(candidate)
    return unique


def _wrap_signed_degrees(value: float, *, period: float) -> float:
    """Canonical signed circular delta in ``[-period/2, period/2)``."""

    wrapped = (float(value) + period / 2.0) % period - period / 2.0
    return 0.0 if abs(wrapped) < 1e-12 else wrapped


def _signed_line_delta_degrees(
    target: Iterable[float], built: Iterable[float]
) -> float:
    """Signed built-minus-target yaw on the modulo-90 edge-family circle."""

    target_run = _unit_run(list(target), "target")
    built_run = _unit_run(list(built), "built")
    target_angle = math.degrees(math.atan2(target_run[1], target_run[0]))
    built_angle = math.degrees(math.atan2(built_run[1], built_run[0]))
    return _wrap_signed_degrees(built_angle - target_angle, period=90.0)


def _circular_distance_degrees(a: float, b: float, *, period: float) -> float:
    return abs(_wrap_signed_degrees(float(a) - float(b), period=period))


def _nonparallel(a: Iterable[float], b: Iterable[float]) -> bool:
    ar = _unit_run(list(a), "a")
    br = _unit_run(list(b), "b")
    angle = math.degrees(
        math.acos(max(-1.0, min(1.0, abs(ar[0] * br[0] + ar[1] * br[1]))))
    )
    return angle >= 20.0


def machine_methods(
    target_candidates: list[dict[str, Any]], built_candidates: list[dict[str, Any]]
) -> list[str]:
    target_edges = [
        c for c in target_candidates if c.get("kind") == "edge" and c.get("run_xy")
    ]
    built_edges = [
        c for c in built_candidates if c.get("kind") == "edge" and c.get("run_xy")
    ]
    if any(
        _nonparallel(a["run_xy"], b["run_xy"])
        for i, a in enumerate(target_edges)
        for b in target_edges[i + 1 :]
    ) and any(
        _nonparallel(a["run_xy"], b["run_xy"])
        for i, a in enumerate(built_edges)
        for b in built_edges[i + 1 :]
    ):
        return ["two_edge_families"]
    # A corner-based method stays intentionally unadvertised until the evaluator checks
    # calibrated corner displacement (not merely that the submitted IDs exist).
    return []


def evaluate_submission(
    requirement: dict[str, Any],
    submission: dict[str, Any],
    *,
    tolerance_degrees: float = 12.0,
) -> dict[str, Any]:
    """Validate issued ids and calculate the verdict from backend-owned directions."""

    req = normalize_advisory_requirement(requirement)
    sub = normalize_resolution_submission(submission)
    base = {
        "schema_version": SCHEMA_VERSION,
        "advisory_id": sub["advisory_id"],
        "observation_token": sub["observation_token"],
        "method": sub["method"],
    }
    if req["state"] == "not_required":
        return {
            **base,
            "status": "not_applicable",
            "reason_codes": ["advisory_not_required"],
        }
    if sub["advisory_id"] != req.get("advisory_id") or sub[
        "observation_token"
    ] != req.get("observation_token"):
        return {
            **base,
            "status": "stale",
            "reason_codes": ["request_identity_mismatch"],
        }
    if sub["method"] not in req.get("allowed_methods", []):
        return {**base, "status": "unverified", "reason_codes": ["method_not_issued"]}
    targets = {c["candidate_id"]: c for c in req.get("target_candidates", [])}
    built = {c["candidate_id"]: c for c in req.get("built_candidates", [])}
    deltas: list[float] = []
    validated: list[dict[str, str]] = []
    target_runs: list[list[float]] = []
    built_runs: list[list[float]] = []
    for match in sub["edge_matches"]:
        target = targets.get(match["target_edge_id"])
        current = built.get(match["built_edge_id"])
        if (
            not target
            or not current
            or not target.get("run_xy")
            or not current.get("run_xy")
        ):
            return {
                **base,
                "status": "unverified",
                "reason_codes": ["candidate_id_or_run_invalid"],
            }
        target_runs.append(target["run_xy"])
        built_runs.append(current["run_xy"])
        deltas.append(_signed_line_delta_degrees(target["run_xy"], current["run_xy"]))
        validated.append(dict(match))
    if sub["method"] == "two_edge_families" and (
        not _nonparallel(target_runs[0], target_runs[1])
        or not _nonparallel(built_runs[0], built_runs[1])
    ):
        return {
            **base,
            "status": "unverified",
            "reason_codes": ["edge_pairs_not_independent"],
        }
    absolute_deltas = [abs(delta) for delta in deltas]
    maximum = max(absolute_deltas)
    mean = sum(absolute_deltas) / len(absolute_deltas)
    disagreement = max(
        _circular_distance_degrees(left, right, period=90.0)
        for index, left in enumerate(deltas)
        for right in deltas[index + 1 :]
    )
    measurements = {
        "max_delta_degrees_mod_90": maximum,
        "mean_delta_degrees_mod_90": mean,
        "pair_disagreement_degrees": disagreement,
        "tolerance_degrees": float(tolerance_degrees),
    }
    if disagreement > 6.0:
        status = "unverified"
        reasons = ["matched_edge_deltas_disagree"]
    elif maximum <= float(tolerance_degrees):
        status = "verified_match"
        reasons = ["backend_edge_correspondence_within_tolerance"]
    else:
        status = "verified_mismatch"
        reasons = ["backend_edge_correspondence_outside_tolerance"]
    return {
        **base,
        "status": status,
        "validated_matches": validated,
        "measurements": measurements,
        "reason_codes": reasons,
    }
