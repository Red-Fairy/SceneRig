"""Compile adjudicated source relationships into initializer obligations.

Relationships answer a source-scene question (for example, whether ``table#0`` is
AGAINST ``wall#0``).  Constraints answer a different question: what the generated
Blender scene must satisfy when that relationship is authoritative.  Keeping the two
records separate prevents a failed build from mutating the source relationship verdict
and gives the prompt and rules gate one immutable contract to consume.

This module deliberately has no legacy adapter.  A caller must load a schema-v2
artifact whose digests match both the active scene graph and immutable source-yaw
observation, or regenerate preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
from lib.tools.geometry.surface_relations import RELATIONSHIP_SCHEMA_VERSION
from lib.tools.geometry.yaw_constraints import (
    YAW_REQUIRED_TOLERANCE_DEGREES,
    resolve_source_yaw_evidence,
    yaw_observation_digest,
)

INITIALIZER_CONSTRAINTS_FILENAME = "initializer_constraints.json"
CONSTRAINT_SCHEMA_VERSION = 2
CONSTRAINT_COMPILER_VERSION = "initializer_constraints_v2"

_STATUS_ENFORCEMENT = {
    "confirmed": "hard",
    "unverified": "advisory",
    "conflicting": "advisory",
    "rejected": "none",
}
_STAGE_ORDER = {"STRUCTURE": 0, "POSE": 1, "CONTACT": 2}
_SUPPORTED_RELATIONSHIPS = {"against", "under", "corner", "perpendicular"}


class ConstraintArtifactError(RuntimeError):
    """The initializer constraint artifact is absent, stale, or malformed."""


def _canonical_json(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConstraintArtifactError(
            f"scene graph/constraint payload is not canonical JSON: {exc}"
        ) from exc
    return payload.encode("utf-8")


def scene_graph_digest(graph: dict[str, Any]) -> str:
    """Digest the complete source graph used by the constraint compiler.

    Runtime root transactions mutate the graph and therefore must atomically recompile
    this artifact.  Hashing the complete graph makes an uncompiled new hard relationship
    impossible to hide behind a projection that happened not to include it.
    """

    if not isinstance(graph, dict):
        raise ConstraintArtifactError("scene graph must be a JSON object")
    return hashlib.sha256(_canonical_json(graph)).hexdigest()


def _nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict) or not node.get("id"):
            continue
        node_id = str(node["id"])
        if node_id in nodes:
            raise ConstraintArtifactError(f"duplicate scene-graph node id {node_id!r}")
        nodes[node_id] = node
    return nodes


def _wall_binding(node: dict[str, Any] | None) -> str:
    """Select an immutable canonical wall when reliable, otherwise live geometry."""

    plane = (node or {}).get("plane")
    normal, kind = plumb_plane((plane or {}).get("normal"))
    if normal is not None and kind == "wall" and plane_is_reliable(plane):
        return "canonical_scene_graph"
    return "current_built_root"


def _canonical_corner_xy(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    *,
    relationship_id: str,
    endpoint_ids: tuple[str, str],
) -> list[float]:
    """Return the XY intersection of two reliable canonical vertical wall planes."""

    rows: list[tuple[list[float], list[float]]] = []
    unavailable: list[str] = []
    for node_id, node in zip(endpoint_ids, (a, b)):
        plane = (node or {}).get("plane")
        normal, kind = plumb_plane((plane or {}).get("normal"))
        center = (node or {}).get("world_center")
        missing: list[str] = []
        if normal is None or kind != "wall" or not plane_is_reliable(plane):
            missing.append("reliable canonical vertical plane")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            missing.append("world_center")
        if missing:
            unavailable.append(f"{node_id}: {', '.join(missing)}")
            continue
        try:
            nxy = [float(normal[0]), float(normal[1])]
            cxy = [float(center[0]), float(center[1])]
        except (TypeError, ValueError) as exc:
            raise ConstraintArtifactError(
                f"hard CORNER relationship {relationship_id!r} endpoint {node_id!r} "
                "has a non-numeric canonical wall normal/world_center"
            ) from exc
        if not all(math.isfinite(value) for value in (*nxy, *cxy)):
            raise ConstraintArtifactError(
                f"hard CORNER relationship {relationship_id!r} endpoint {node_id!r} "
                "has a non-finite canonical wall normal/world_center"
            )
        rows.append((nxy, cxy))

    if unavailable:
        a_id, b_id = endpoint_ids
        raise ConstraintArtifactError(
            f"hard CORNER relationship {relationship_id!r} between {a_id!r} and "
            f"{b_id!r} cannot compile its canonical plane-intersection corner; "
            "missing or unreliable measurements: " + "; ".join(unavailable)
        )

    (na, ca), (nb, cb) = rows
    det = na[0] * nb[1] - na[1] * nb[0]
    if abs(det) < 1e-8:
        a_id, b_id = endpoint_ids
        raise ConstraintArtifactError(
            f"hard CORNER relationship {relationship_id!r} between {a_id!r} and "
            f"{b_id!r} has canonical wall planes without a stable intersection"
        )
    ra = na[0] * ca[0] + na[1] * ca[1]
    rb = nb[0] * cb[0] + nb[1] * cb[1]
    x = (ra * nb[1] - na[1] * rb) / det
    y = (na[0] * rb - ra * nb[0]) / det
    return [round(float(x), 6), round(float(y), 6)]


def _base_constraint(
    relationship: dict[str, Any],
    *,
    suffix: str,
    kind: str,
    stage: str,
) -> dict[str, Any]:
    relationship_id = str(relationship["relationship_id"])
    return {
        "constraint_id": f"{relationship_id}:{suffix}",
        "kind": kind,
        "source_relationship_id": relationship_id,
        "source_yaw_evidence_id": None,
        "stage": stage,
        "authority": "required",
        "targets": [str(relationship["a"]), str(relationship["b"])],
        "reference": {},
        "applicability": {},
        "parameters": {},
        "effects": {},
    }


def _validated_yaw_observation(observation: Any) -> dict[str, Any]:
    """Validate the extractor artifact without assuming its validator return style."""

    if not isinstance(observation, dict):
        raise ConstraintArtifactError(
            "main-support yaw observation is required; rerun preprocessing"
        )
    try:
        from lib.tools.geometry.main_support_yaw_observation import (
            validate_main_support_yaw_observation,
        )

        validated = validate_main_support_yaw_observation(observation)
    except Exception as exc:
        raise ConstraintArtifactError(
            f"invalid main-support yaw observation: {exc}; rerun preprocessing"
        ) from exc
    return validated if isinstance(validated, dict) else observation


def _compile_source_yaw_constraint(
    resolved: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compile a frozen independent source target, never a relationship verdict."""

    decision = resolved.get("source_decision") or {}
    if decision.get("authority") != "required":
        return []
    main_id = str(resolved.get("main_surface_id") or "")
    evidence_id = str(resolved.get("decision_id") or "")
    if not main_id or main_id not in nodes:
        raise ConstraintArtifactError(
            "required source-yaw decision targets a missing main support"
        )
    run = decision.get("target_run_xy")
    if not isinstance(run, list) or len(run) != 2:
        raise ConstraintArtifactError(
            "required source-yaw decision has no frozen target run"
        )
    return [
        {
            "constraint_id": f"{evidence_id}:edge-yaw",
            "kind": "main_support_edge_yaw",
            "source_relationship_id": None,
            "source_yaw_evidence_id": evidence_id,
            "stage": "POSE",
            "authority": "required",
            "targets": [main_id],
            "reference": {
                "run_binding": "frozen_source_consensus",
                "run_xy": list(run),
            },
            "applicability": {
                "condition": "edge_bearing_top",
                "target_surface_id": main_id,
            },
            "parameters": {"max_delta_degrees_mod_90": YAW_REQUIRED_TOLERANCE_DEGREES},
            "effects": {},
        }
    ]


def _compile_against(
    relationship: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    support_id, wall_id = str(relationship["a"]), str(relationship["b"])
    binding = _wall_binding(nodes.get(wall_id))

    yaw = _base_constraint(
        relationship,
        suffix="edge-yaw",
        kind="edge_parallel_to_surface",
        stage="POSE",
    )
    yaw["reference"] = {
        "surface_id": wall_id,
        "run_binding": binding,
    }
    yaw["applicability"] = {
        "condition": "edge_bearing_top",
        "target_surface_id": support_id,
    }
    yaw["parameters"] = {"max_delta_degrees_mod_90": 3.0}

    flush = _base_constraint(
        relationship,
        suffix="flush",
        kind="finite_against",
        stage="CONTACT",
    )
    flush["reference"] = {
        "surface_id": wall_id,
        "plane_binding": binding,
        "finite_extent_binding": "current_built_root",
    }
    flush["parameters"] = {
        "max_float_gap_m": 0.01,
        "max_penetration_m": 0.01,
        "max_finite_run_gap_m": 0.05,
        "max_vertical_gap_m": 0.05,
    }
    flush["effects"] = {"owns_contact_pair": [support_id, wall_id]}
    return [yaw, flush]


def _is_floor(node: dict[str, Any] | None, node_id: str) -> bool:
    category = str((node or {}).get("category") or node_id.split("#", 1)[0])
    return category.strip().casefold() in {"floor", "ground"}


def _compile_under(
    relationship: dict[str, Any],
    graph: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lower_id, upper_id = str(relationship["a"]), str(relationship["b"])
    constraints: list[dict[str, Any]] = []
    if upper_id == str(graph.get("main_support_id") or "") and _is_floor(
        nodes.get(lower_id), lower_id
    ):
        form = _base_constraint(
            relationship,
            suffix="complete-base",
            kind="main_support_form",
            stage="STRUCTURE",
        )
        form["reference"] = {
            "lower_surface_id": lower_id,
            "binding": "current_built_root",
        }
        form["parameters"] = {
            "form": "complete_furniture",
            "require_connected_base_to_floor": True,
            "forbid_bare_floating_top": True,
        }
        constraints.append(form)

    contact = _base_constraint(
        relationship,
        suffix="contact",
        kind="finite_under",
        stage="CONTACT",
    )
    contact["reference"] = {
        "lower_surface_id": lower_id,
        "upper_surface_id": upper_id,
        "lower_binding": "current_built_root",
        "upper_binding": (
            "complete_grouped_body"
            if upper_id == str(graph.get("main_support_id") or "")
            else "current_built_root"
        ),
    }
    contact["parameters"] = {
        "max_vertical_gap_m": 0.05,
        "require_xy_overlap": True,
    }
    contact["effects"] = {"owns_contact_pair": [lower_id, upper_id]}
    constraints.append(contact)
    return constraints


def _compile_corner(
    relationship: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    a_id, b_id = str(relationship["a"]), str(relationship["b"])
    corner = _base_constraint(
        relationship,
        suffix="shared-edge",
        kind="finite_wall_corner",
        stage="CONTACT",
    )
    corner["reference"] = {
        "binding": "canonical_scene_graph_corner",
        "corner_xy_m": _canonical_corner_xy(
            nodes.get(a_id),
            nodes.get(b_id),
            relationship_id=str(relationship["relationship_id"]),
            endpoint_ids=(a_id, b_id),
        ),
        "finite_extent_binding": "current_built_root",
    }
    corner["parameters"] = {
        "minimum_normal_angle_degrees": 85.0,
        "max_built_reach_error_m": 0.05,
        "max_vertical_gap_m": 0.05,
        "require_positive_vertical_overlap": True,
    }
    return [corner]


def _compile_perpendicular(
    relationship: dict[str, Any],
) -> list[dict[str, Any]]:
    meeting = _base_constraint(
        relationship,
        suffix="finite-meeting",
        kind="finite_perpendicular",
        stage="CONTACT",
    )
    meeting["reference"] = {"binding": "current_built_roots"}
    meeting["parameters"] = {
        "minimum_normal_angle_degrees": 88.0,
        "max_finite_contact_gap_m": 0.05,
    }
    return [meeting]


def _validated_relationships(graph: dict[str, Any]) -> list[dict[str, Any]]:
    relationships = graph.get("relationships")
    if relationships is None:
        raise ConstraintArtifactError("scene graph has no relationships array")
    if not isinstance(relationships, list):
        raise ConstraintArtifactError("scene graph relationships must be an array")

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            raise ConstraintArtifactError(f"relationship {index} must be an object")
        missing = [
            key
            for key in ("relationship_id", "type", "a", "b", "status", "enforcement")
            if not relationship.get(key)
        ]
        if missing:
            raise ConstraintArtifactError(
                f"relationship {index} is missing required fields {missing}"
            )
        relationship_id = str(relationship["relationship_id"])
        if relationship_id in seen:
            raise ConstraintArtifactError(
                f"duplicate relationship id {relationship_id!r}"
            )
        seen.add(relationship_id)
        relationship_type = str(relationship["type"]).strip().casefold()
        if relationship_type not in _SUPPORTED_RELATIONSHIPS:
            raise ConstraintArtifactError(
                f"relationship {relationship_id!r} has unsupported type "
                f"{relationship_type!r}"
            )
        status = str(relationship["status"]).strip().casefold()
        enforcement = str(relationship["enforcement"]).strip().casefold()
        if (
            relationship["type"] != relationship_type
            or relationship["status"] != status
            or relationship["enforcement"] != enforcement
        ):
            raise ConstraintArtifactError(
                f"relationship {relationship_id!r} uses non-canonical enum spelling; "
                "type/status/enforcement must be lowercase schema values"
            )
        expected = _STATUS_ENFORCEMENT.get(status)
        if expected is None or enforcement != expected:
            raise ConstraintArtifactError(
                f"relationship {relationship_id!r} has invalid atomic verdict "
                f"{status!r}/{enforcement!r}; expected enforcement {expected!r}"
            )
        result.append(relationship)
    return result


def compile_relationship_constraints(
    graph: dict[str, Any], *, yaw_observation: dict[str, Any]
) -> dict[str, Any]:
    """Compile relationship and independent-yaw obligations deterministically."""

    adjudication = graph.get("relationship_adjudication")
    if (
        not isinstance(adjudication, dict)
        or adjudication.get("schema_version") != RELATIONSHIP_SCHEMA_VERSION
    ):
        raise ConstraintArtifactError(
            "scene graph does not carry current relationship adjudication schema "
            f"v{RELATIONSHIP_SCHEMA_VERSION}; rerun preprocessing (legacy relationship "
            "schemas are unsupported)"
        )
    observation = _validated_yaw_observation(yaw_observation)
    try:
        resolved_yaw = resolve_source_yaw_evidence(graph, observation)
        observation_sha256 = yaw_observation_digest(observation)
    except Exception as exc:
        raise ConstraintArtifactError(
            f"cannot resolve main-support source yaw: {exc}; rerun preprocessing"
        ) from exc
    nodes = _nodes_by_id(graph)
    constraints: list[dict[str, Any]] = []
    for relationship in _validated_relationships(graph):
        if (
            relationship["status"] != "confirmed"
            or relationship["enforcement"] != "hard"
        ):
            continue
        a_id, b_id = str(relationship["a"]), str(relationship["b"])
        missing = [node_id for node_id in (a_id, b_id) if node_id not in nodes]
        if missing:
            raise ConstraintArtifactError(
                f"hard relationship {relationship['relationship_id']!r} names missing "
                f"scene-graph endpoint(s): {missing}"
            )
        relationship_type = str(relationship["type"]).strip().casefold()
        if relationship_type == "against":
            constraints.extend(_compile_against(relationship, nodes))
        elif relationship_type == "under":
            constraints.extend(_compile_under(relationship, graph, nodes))
        elif relationship_type == "corner":
            constraints.extend(_compile_corner(relationship, nodes))
        elif relationship_type == "perpendicular":
            constraints.extend(_compile_perpendicular(relationship))

    constraints.extend(_compile_source_yaw_constraint(resolved_yaw, nodes))

    constraints.sort(
        key=lambda row: (
            str(row.get("source_relationship_id") or row.get("source_yaw_evidence_id")),
            _STAGE_ORDER[str(row["stage"])],
            str(row["constraint_id"]),
        )
    )
    return {
        "schema_version": CONSTRAINT_SCHEMA_VERSION,
        "compiler_version": CONSTRAINT_COMPILER_VERSION,
        "scene_graph_sha256": scene_graph_digest(graph),
        "source_yaw_observation_sha256": observation_sha256,
        "source_yaw_evidence": resolved_yaw,
        "constraints": constraints,
    }


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ConstraintArtifactError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConstraintArtifactError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ConstraintArtifactError(
            f"{label} must be finite and at least {minimum}, got {value!r}"
        )
    return result


def _validate_constraint_semantics(constraint: dict[str, Any]) -> None:
    """Validate schema-v2 provenance and kind-specific runtime bindings."""

    constraint_id = str(constraint["constraint_id"])
    relationship_source = constraint.get("source_relationship_id")
    yaw_source = constraint.get("source_yaw_evidence_id")
    if bool(relationship_source) == bool(yaw_source):
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} must name exactly one of "
            "source_relationship_id or source_yaw_evidence_id"
        )
    source_id = str(relationship_source or yaw_source)
    if not constraint_id.startswith(f"{source_id}:"):
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} is not namespaced by its source evidence"
        )
    targets = constraint["targets"]
    if not all(isinstance(target, str) and target for target in targets):
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} targets must be nonempty strings"
        )
    if len(targets) == 2 and targets[0] == targets[1]:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} cannot target the same endpoint twice"
        )
    for field in ("reference", "applicability", "parameters", "effects"):
        if not isinstance(constraint[field], dict):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} field {field!r} must be an object"
            )

    kind = constraint["kind"]
    expected_stage = {
        "main_support_form": "STRUCTURE",
        "edge_parallel_to_surface": "POSE",
        "main_support_edge_yaw": "POSE",
        "finite_against": "CONTACT",
        "finite_under": "CONTACT",
        "finite_wall_corner": "CONTACT",
        "finite_perpendicular": "CONTACT",
    }.get(kind)
    if expected_stage is None:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} has unsupported kind {kind!r}"
        )
    if constraint["stage"] != expected_stage:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} kind {kind!r} requires stage "
            f"{expected_stage!r}"
        )

    reference = constraint["reference"]
    applicability = constraint["applicability"]
    parameters = constraint["parameters"]
    effects = constraint["effects"]
    if kind == "main_support_edge_yaw":
        if relationship_source or not yaw_source or len(targets) != 1:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has invalid independent-yaw provenance"
            )
        run = reference.get("run_xy")
        if (
            reference.get("run_binding") != "frozen_source_consensus"
            or not isinstance(run, list)
            or len(run) != 2
        ):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid frozen yaw reference"
            )
        x = _finite_number(
            run[0], label=f"{constraint_id}.run_xy[0]", minimum=-math.inf
        )
        y = _finite_number(
            run[1], label=f"{constraint_id}.run_xy[1]", minimum=-math.inf
        )
        if math.hypot(x, y) < 0.999 or math.hypot(x, y) > 1.001:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} frozen yaw run must be unit length"
            )
        if applicability != {
            "condition": "edge_bearing_top",
            "target_surface_id": targets[0],
        }:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has invalid independent-yaw applicability"
            )
        _finite_number(
            parameters.get("max_delta_degrees_mod_90"),
            label=f"{constraint_id}.max_delta_degrees_mod_90",
        )
        if effects:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must not own contact"
            )
        return

    if not relationship_source or yaw_source:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} must be relationship-owned"
        )
    if kind == "edge_parallel_to_surface":
        if reference.get("surface_id") != targets[1] or reference.get(
            "run_binding"
        ) not in {"canonical_scene_graph", "current_built_root"}:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid wall-run reference"
            )
        if applicability != {
            "condition": "edge_bearing_top",
            "target_surface_id": targets[0],
        }:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has invalid edge-yaw applicability"
            )
        _finite_number(
            parameters.get("max_delta_degrees_mod_90"),
            label=f"{constraint_id}.max_delta_degrees_mod_90",
        )
        if effects:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must not own contact"
            )
        return

    if kind == "finite_against":
        binding = reference.get("plane_binding")
        if (
            reference.get("surface_id") != targets[1]
            or binding not in {"canonical_scene_graph", "current_built_root"}
            or reference.get("finite_extent_binding") != "current_built_root"
        ):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid AGAINST reference"
            )
        for name in (
            "max_float_gap_m",
            "max_penetration_m",
            "max_finite_run_gap_m",
            "max_vertical_gap_m",
        ):
            _finite_number(parameters.get(name), label=f"{constraint_id}.{name}")
        if effects != {"owns_contact_pair": targets}:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must own exactly its AGAINST pair"
            )
        return

    if kind == "main_support_form":
        if reference != {
            "lower_surface_id": targets[0],
            "binding": "current_built_root",
        }:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid floor reference"
            )
        expected_parameters = {
            "form": "complete_furniture",
            "require_connected_base_to_floor": True,
            "forbid_bare_floating_top": True,
        }
        if parameters != expected_parameters or applicability or effects:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has invalid complete-base semantics"
            )
        return

    if kind == "finite_under":
        if (
            reference.get("lower_surface_id") != targets[0]
            or reference.get("upper_surface_id") != targets[1]
            or reference.get("lower_binding") != "current_built_root"
            or reference.get("upper_binding")
            not in {"current_built_root", "complete_grouped_body"}
        ):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid UNDER reference"
            )
        _finite_number(
            parameters.get("max_vertical_gap_m"),
            label=f"{constraint_id}.max_vertical_gap_m",
        )
        if parameters.get("require_xy_overlap") is not True:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must require finite XY overlap"
            )
        if effects != {"owns_contact_pair": targets}:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must own exactly its UNDER pair"
            )
        return

    if kind == "finite_wall_corner":
        xy = reference.get("corner_xy_m")
        if (
            reference.get("binding") != "canonical_scene_graph_corner"
            or reference.get("finite_extent_binding") != "current_built_root"
            or not isinstance(xy, list)
            or len(xy) != 2
        ):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has an invalid canonical corner reference"
            )
        for index, value in enumerate(xy):
            _finite_number(
                value, label=f"{constraint_id}.corner_xy_m[{index}]", minimum=-math.inf
            )
        for name in (
            "minimum_normal_angle_degrees",
            "max_built_reach_error_m",
            "max_vertical_gap_m",
        ):
            _finite_number(parameters.get(name), label=f"{constraint_id}.{name}")
        if parameters.get("require_positive_vertical_overlap") is not True:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must require positive vertical overlap"
            )
        if applicability or effects:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must not carry applicability/contact ownership"
            )
        return

    if reference != {"binding": "current_built_roots"}:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} has an invalid PERPENDICULAR reference"
        )
    for name in ("minimum_normal_angle_degrees", "max_finite_contact_gap_m"):
        _finite_number(parameters.get(name), label=f"{constraint_id}.{name}")
    if applicability or effects:
        raise ConstraintArtifactError(
            f"constraint {constraint_id!r} must not carry applicability/contact ownership"
        )


def _validate_artifact_shape(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ConstraintArtifactError(
            "initializer constraint artifact must be an object"
        )
    if artifact.get("schema_version") != CONSTRAINT_SCHEMA_VERSION:
        raise ConstraintArtifactError(
            "unsupported initializer constraint schema version: "
            f"{artifact.get('schema_version')!r}; rerun preprocessing"
        )
    if artifact.get("compiler_version") != CONSTRAINT_COMPILER_VERSION:
        raise ConstraintArtifactError(
            "unsupported initializer constraint compiler version: "
            f"{artifact.get('compiler_version')!r}; rerun preprocessing"
        )
    digest = artifact.get("scene_graph_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ConstraintArtifactError(
            "initializer constraint artifact has no valid scene-graph digest"
        )
    yaw_digest = artifact.get("source_yaw_observation_sha256")
    if not isinstance(yaw_digest, str) or len(yaw_digest) != 64:
        raise ConstraintArtifactError(
            "initializer constraint artifact has no valid source-yaw digest"
        )
    resolved_yaw = artifact.get("source_yaw_evidence")
    if (
        not isinstance(resolved_yaw, dict)
        or resolved_yaw.get("schema_version") != 1
        or not isinstance(resolved_yaw.get("source_candidates"), list)
        or not isinstance(resolved_yaw.get("source_decision"), dict)
    ):
        raise ConstraintArtifactError(
            "initializer constraint artifact has no valid resolved source-yaw evidence"
        )
    constraints = artifact.get("constraints")
    if not isinstance(constraints, list):
        raise ConstraintArtifactError(
            "initializer constraint artifact has no constraint list"
        )
    seen: set[str] = set()
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise ConstraintArtifactError(f"constraint {index} must be an object")
        missing = [
            key
            for key in (
                "constraint_id",
                "kind",
                "source_relationship_id",
                "source_yaw_evidence_id",
                "stage",
                "authority",
                "targets",
                "reference",
                "applicability",
                "parameters",
                "effects",
            )
            if key not in constraint
        ]
        if missing:
            raise ConstraintArtifactError(
                f"constraint {index} is missing required fields {missing}"
            )
        constraint_id = str(constraint["constraint_id"])
        if constraint_id in seen:
            raise ConstraintArtifactError(f"duplicate constraint id {constraint_id!r}")
        seen.add(constraint_id)
        if constraint["stage"] not in _STAGE_ORDER:
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} has invalid stage {constraint['stage']!r}"
            )
        if constraint["authority"] != "required":
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} is not a required obligation"
            )
        expected_target_count = (
            1 if constraint.get("kind") == "main_support_edge_yaw" else 2
        )
        if (
            not isinstance(constraint["targets"], list)
            or len(constraint["targets"]) != expected_target_count
        ):
            raise ConstraintArtifactError(
                f"constraint {constraint_id!r} must have exactly "
                f"{expected_target_count} target(s)"
            )
        _validate_constraint_semantics(constraint)
    expected_order = sorted(
        constraints,
        key=lambda row: (
            str(row.get("source_relationship_id") or row.get("source_yaw_evidence_id")),
            _STAGE_ORDER[str(row["stage"])],
            str(row["constraint_id"]),
        ),
    )
    if constraints != expected_order:
        raise ConstraintArtifactError(
            "initializer constraints are not in deterministic relationship/stage order"
        )
    return artifact


def write_initializer_constraints(
    scene_dir: str | os.PathLike[str],
    graph: dict[str, Any],
    *,
    yaw_observation: dict[str, Any] | None = None,
) -> Path:
    """Atomically compile and persist ``initializer_constraints.json``."""

    root = Path(scene_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / INITIALIZER_CONSTRAINTS_FILENAME
    if yaw_observation is None:
        try:
            from lib.tools.geometry.main_support_yaw_observation import (
                load_main_support_yaw_observation,
            )

            yaw_observation = load_main_support_yaw_observation(root)
        except Exception as exc:
            raise ConstraintArtifactError(
                "main_support_yaw_observation.json is required before compiling "
                f"initializer constraints: {exc}; rerun preprocessing"
            ) from exc
    artifact = compile_relationship_constraints(graph, yaw_observation=yaw_observation)
    _validate_artifact_shape(artifact)
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=root, prefix=f".{path.name}.", delete=False
        ) as stream:
            tmp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    return path


def load_initializer_constraints(
    scene_dir: str | os.PathLike[str],
    *,
    scene_graph: dict[str, Any] | None = None,
    yaw_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a current artifact and reject missing, stale, or edited contracts."""

    root = Path(scene_dir)
    path = root / INITIALIZER_CONSTRAINTS_FILENAME
    if not path.is_file():
        raise ConstraintArtifactError(
            f"missing {path}; rerun preprocessing (legacy relationship caches are unsupported)"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstraintArtifactError(f"cannot read {path}: {exc}") from exc
    artifact = _validate_artifact_shape(artifact)
    if yaw_observation is None:
        try:
            from lib.tools.geometry.main_support_yaw_observation import (
                load_main_support_yaw_observation,
            )

            yaw_observation = load_main_support_yaw_observation(root)
        except Exception as exc:
            raise ConstraintArtifactError(
                "main_support_yaw_observation.json is required to validate "
                f"initializer constraints: {exc}; rerun preprocessing"
            ) from exc
    yaw_observation = _validated_yaw_observation(yaw_observation)
    actual_yaw_digest = yaw_observation_digest(yaw_observation)
    if artifact["source_yaw_observation_sha256"] != actual_yaw_digest:
        raise ConstraintArtifactError(
            "initializer constraint artifact is stale for the immutable source-yaw "
            "observation; rerun preprocessing"
        )
    if scene_graph is not None:
        actual_digest = scene_graph_digest(scene_graph)
        if artifact["scene_graph_sha256"] != actual_digest:
            raise ConstraintArtifactError(
                "initializer constraint artifact is stale for the active scene graph "
                f"({artifact['scene_graph_sha256']} != {actual_digest}); rerun "
                "preprocessing or recompile the committed runtime-root transaction"
            )
        expected = compile_relationship_constraints(
            scene_graph, yaw_observation=yaw_observation
        )
        if _canonical_json(artifact) != _canonical_json(expected):
            raise ConstraintArtifactError(
                "initializer constraint artifact does not match the deterministic "
                "relationship compiler; rerun preprocessing"
            )
    return artifact
