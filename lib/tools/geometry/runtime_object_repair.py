"""Persistent runtime-object inventory helpers for the GPT-6 repair harness.

The preprocessing mask registry remains immutable.  Objects added after preprocessing
and runtime mesh revisions of source objects are instead described by
``runtime_objects/inventory.json``.  This module owns that small overlay's schema and
provides pure staging helpers; committing the returned graph, placement, overlay,
Blender scene, and physics artifacts is the caller's transaction boundary.

Only committed objects belong in the overlay.  Failed and rejected proposals belong
in the mutation journal, not in active inventory.  Keeping those concerns separate
makes the effective scene inventory deterministic after a restart.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RUNTIME_INVENTORY_SCHEMA_VERSION = 1
RUNTIME_INVENTORY_RELATIVE_PATH = Path("runtime_objects") / "inventory.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_STATUSES = {"committed", "superseded"}
_PHYSICAL_MATERIAL_STATUSES = {"estimated", "fallback"}
_OBJECT_ORIGINS = {"runtime_added", "source_revised"}
_MASK_POLICIES = {"tracked", "excluded"}
_TOMBSTONE_ORIGINS = {"source_removed", "runtime_removed"}


class RuntimeObjectRepairError(ValueError):
    """A runtime-object overlay or staged mutation is structurally invalid."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_inventory_path(scene_dir: str | Path) -> Path:
    """Return the canonical backend-owned runtime inventory path."""

    return Path(scene_dir) / RUNTIME_INVENTORY_RELATIVE_PATH


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_content(item) for item in value)
    return value is not None


def _valid_transaction_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_revision(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _object_id(category: Any, instance: Any) -> str | None:
    if not isinstance(category, str) or not category.strip():
        return None
    if not isinstance(instance, int) or isinstance(instance, bool) or instance < 0:
        return None
    return f"{category}#{instance}"


def _record_errors(raw: Any, index: int) -> tuple[str | None, list[str]]:
    origin = f"runtime_inventory.objects[{index}]"
    if not isinstance(raw, Mapping):
        return None, [f"{origin} is not an object"]

    errors: list[str] = []
    category = raw.get("category")
    instance = raw.get("instance")
    derived_id = _object_id(category, instance)
    record_id = raw.get("id")
    if derived_id is None:
        errors.append(f"{origin} has an invalid category/instance identity")
    if not isinstance(record_id, str) or not record_id.strip():
        errors.append(f"{origin} has no nonempty id")
        record_id = None
    elif derived_id is not None and record_id != derived_id:
        errors.append(
            f"{origin} id {record_id!r} does not match category/instance "
            f"identity {derived_id!r}"
        )
    if raw.get("kind") != "object":
        errors.append(f"{origin} kind must be 'object'")
    object_origin = raw.get("origin")
    if object_origin not in _OBJECT_ORIGINS:
        errors.append(f"{origin} origin must be one of {sorted(_OBJECT_ORIGINS)!r}")
    if raw.get("status") != "committed":
        errors.append(f"{origin} status must be 'committed'")
    transaction_id = raw.get("transaction_id")
    if not _valid_transaction_id(transaction_id):
        errors.append(f"{origin} has no positive committed transaction_id")

    mask_policy = raw.get("mask_policy", "tracked")
    if mask_policy not in _MASK_POLICIES:
        errors.append(f"{origin} mask_policy must be one of {sorted(_MASK_POLICIES)!r}")
    mask_path = raw.get("mask_path")
    mask_sha256 = raw.get("mask_sha256")
    scoring_mask_path = raw.get("scoring_mask_path", mask_path)
    scoring_mask_sha256 = raw.get("scoring_mask_sha256", mask_sha256)
    if mask_policy == "excluded":
        for field, value in (
            ("mask_path", mask_path),
            ("mask_sha256", mask_sha256),
            ("scoring_mask_path", scoring_mask_path),
            ("scoring_mask_sha256", scoring_mask_sha256),
        ):
            if field not in raw:
                errors.append(
                    f"{origin} excluded mask policy requires explicit {field}=null"
                )
            if value is not None:
                errors.append(
                    f"{origin} excluded mask policy requires {field}=null"
                )
        if object_origin == "source_revised":
            source_mask_path = raw.get("source_mask_path")
            source_mask_sha256 = raw.get("source_mask_sha256")
            if not isinstance(source_mask_path, str) or not source_mask_path.strip():
                errors.append(f"{origin} has no nonempty source_mask_path provenance")
            if not _SHA256_RE.fullmatch(str(source_mask_sha256 or "")):
                errors.append(f"{origin} has no valid source_mask_sha256 provenance")
    else:
        if not isinstance(mask_path, str) or not mask_path.strip():
            errors.append(f"{origin} has no nonempty mask_path")
        if not _SHA256_RE.fullmatch(str(mask_sha256 or "")):
            errors.append(f"{origin} has no valid mask_sha256 digest")
        if not isinstance(scoring_mask_path, str) or not scoring_mask_path.strip():
            errors.append(f"{origin} has no nonempty scoring_mask_path")
        if not _SHA256_RE.fullmatch(str(scoring_mask_sha256 or "")):
            errors.append(f"{origin} has no valid scoring_mask_sha256 digest")
        if scoring_mask_path == mask_path and scoring_mask_sha256 != mask_sha256:
            errors.append(
                f"{origin} uses one mask path with conflicting content digests"
            )
    evidence = raw.get("evidence")
    if not isinstance(evidence, Mapping) or not _has_content(evidence):
        errors.append(f"{origin} has no nonempty evidence object")

    graph_node = raw.get("graph_node")
    if not isinstance(graph_node, Mapping):
        errors.append(f"{origin} has no graph_node snapshot")
    else:
        graph_expectations = {
            "id": record_id,
            "category": category,
            "kind": "object",
            "mask_path": mask_path,
        }
        if mask_policy == "excluded" and "mask_path" not in graph_node:
            errors.append(
                f"{origin}.graph_node excluded mask_path must be explicitly null"
            )
        if object_origin == "runtime_added":
            graph_expectations.update(
                runtime_added=True,
                runtime_transaction_id=transaction_id,
            )
        elif graph_node.get("runtime_added") is True:
            errors.append(f"{origin}.graph_node source object is marked runtime_added")
        for field, expected in graph_expectations.items():
            if graph_node.get(field) != expected:
                errors.append(
                    f"{origin}.graph_node {field}={graph_node.get(field)!r}; "
                    f"expected {expected!r}"
                )
        parent = graph_node.get("parent")
        if not isinstance(parent, str) or not parent.strip():
            errors.append(f"{origin}.graph_node has no parent")
        if graph_node.get("support") != parent:
            errors.append(f"{origin}.graph_node parent/support differ")
        if not isinstance(graph_node.get("children"), list):
            errors.append(f"{origin}.graph_node children must be a list")

    placement = raw.get("placement")
    if not isinstance(placement, Mapping):
        errors.append(f"{origin} has no placement snapshot")
    else:
        placement_expectations = {
            "category": category,
            "instance": instance,
        }
        if object_origin == "runtime_added":
            placement_expectations.update(
                runtime_added=True,
                runtime_transaction_id=transaction_id,
            )
        elif placement.get("runtime_added") is True:
            errors.append(f"{origin}.placement source object is marked runtime_added")
        if mask_policy == "excluded" and placement.get("mask_path") is not None:
            errors.append(f"{origin}.placement excluded mask_path must be null")
        for field, expected in placement_expectations.items():
            if placement.get(field) != expected:
                errors.append(
                    f"{origin}.placement {field}={placement.get(field)!r}; "
                    f"expected {expected!r}"
                )

    physical_material = raw.get("physical_material")
    if not isinstance(physical_material, Mapping):
        errors.append(f"{origin} has no physical_material record")
    else:
        status = physical_material.get("status")
        if status not in _PHYSICAL_MATERIAL_STATUSES:
            errors.append(
                f"{origin}.physical_material status must be 'estimated' or 'fallback'"
            )
        if not _has_content(physical_material.get("provenance")):
            errors.append(f"{origin}.physical_material has no provenance")
        if (
            not isinstance(physical_material.get("material"), str)
            or not str(physical_material.get("material")).strip()
        ):
            errors.append(f"{origin}.physical_material has no material label")
        mass = physical_material.get("mass_kg")
        if (
            not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not math.isfinite(float(mass))
            or float(mass) <= 0.0
        ):
            errors.append(f"{origin}.physical_material has no positive finite mass_kg")
        mass_range = physical_material.get("mass_range_kg")
        if (
            not isinstance(mass_range, (list, tuple))
            or len(mass_range) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in mass_range
            )
        ):
            errors.append(
                f"{origin}.physical_material has no positive finite mass_range_kg"
            )
        elif (
            float(mass_range[0]) > float(mass_range[1])
            or isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or not float(mass_range[0]) <= float(mass) <= float(mass_range[1])
        ):
            errors.append(
                f"{origin}.physical_material mass_range_kg does not contain mass_kg"
            )
        for field in ("friction", "density_kgm3"):
            value = physical_material.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                errors.append(
                    f"{origin}.physical_material has no positive finite {field}"
                )
        if (
            not isinstance(physical_material.get("density_gate"), str)
            or not str(physical_material.get("density_gate")).strip()
        ):
            errors.append(f"{origin}.physical_material has no density_gate")

    visual_material = raw.get("visual_material")
    if not isinstance(visual_material, Mapping):
        errors.append(f"{origin} has no visual_material record")
    else:
        if visual_material.get("status") != "embedded":
            errors.append(f"{origin}.visual_material status must be 'embedded'")
        if not _has_content(visual_material.get("provenance")):
            errors.append(f"{origin}.visual_material has no provenance")
        if not _SHA256_RE.fullmatch(str(visual_material.get("sha256") or "")):
            errors.append(f"{origin}.visual_material has no valid sha256 digest")

    revisions = raw.get("mesh_revisions")
    current_revision = raw.get("current_mesh_revision")
    by_revision: dict[int, Mapping[str, Any]] = {}
    if not isinstance(revisions, list) or not revisions:
        errors.append(f"{origin} has no mesh_revisions")
    else:
        for revision_index, revision in enumerate(revisions):
            revision_origin = f"{origin}.mesh_revisions[{revision_index}]"
            if not isinstance(revision, Mapping):
                errors.append(f"{revision_origin} is not an object")
                continue
            number = revision.get("revision")
            if not _valid_revision(number):
                errors.append(f"{revision_origin} has no positive revision number")
                continue
            if number in by_revision:
                errors.append(f"{origin} has duplicate mesh revision {number}")
            else:
                by_revision[number] = revision
            revision_status = revision.get("status")
            provenance = revision.get("provenance")
            if revision_status not in _REVISION_STATUSES:
                errors.append(
                    f"{revision_origin} status must be one of "
                    f"{sorted(_REVISION_STATUSES)!r}"
                )
            if not _has_content(provenance):
                errors.append(f"{revision_origin} has no provenance")
            source_baseline = (
                revision.get("transaction_id") is None
                and revision_status == "superseded"
                and isinstance(provenance, Mapping)
                and provenance.get("source") == "preprocess"
            )
            if (
                not _valid_transaction_id(revision.get("transaction_id"))
                and not source_baseline
            ):
                errors.append(f"{revision_origin} has no positive transaction_id")
            for field in ("mesh_name", "mesh_glb"):
                if (
                    not isinstance(revision.get(field), str)
                    or not str(revision.get(field)).strip()
                ):
                    errors.append(f"{revision_origin} has no nonempty {field}")
            if not _SHA256_RE.fullmatch(str(revision.get("sha256") or "")):
                errors.append(f"{revision_origin} has no valid sha256 digest")

    if not _valid_revision(current_revision):
        errors.append(f"{origin} has no positive current_mesh_revision")
    elif current_revision not in by_revision:
        errors.append(
            f"{origin} current_mesh_revision {current_revision} is not present"
        )
    else:
        current = by_revision[current_revision]
        committed_revisions = [
            number
            for number, revision in by_revision.items()
            if revision.get("status") == "committed"
        ]
        if committed_revisions != [current_revision]:
            errors.append(
                f"{origin} must have exactly one committed current mesh revision; "
                f"found {sorted(committed_revisions)!r}"
            )
        if current.get("status") != "committed":
            errors.append(f"{origin} current mesh revision is not committed")
        if not _valid_transaction_id(current.get("transaction_id")):
            errors.append(f"{origin} current mesh revision has no transaction_id")
        if isinstance(visual_material, Mapping) and visual_material.get(
            "sha256"
        ) != current.get("sha256"):
            errors.append(
                f"{origin}.visual_material sha256 does not match the current "
                "committed mesh revision"
            )
        if isinstance(placement, Mapping):
            for field in ("mesh_name", "mesh_glb"):
                if placement.get(field) != current.get(field):
                    errors.append(
                        f"{origin}.placement {field} does not match current mesh "
                        "revision"
                    )

    return record_id if isinstance(record_id, str) else None, errors


def _tombstone_errors(raw: Any, index: int) -> tuple[str | None, list[str]]:
    origin = f"runtime_inventory.tombstones[{index}]"
    if not isinstance(raw, Mapping):
        return None, [f"{origin} is not an object"]

    errors: list[str] = []
    category = raw.get("category")
    instance = raw.get("instance")
    derived_id = _object_id(category, instance)
    record_id = raw.get("id")
    if derived_id is None:
        errors.append(f"{origin} has an invalid category/instance identity")
    if not isinstance(record_id, str) or not record_id.strip():
        errors.append(f"{origin} has no nonempty id")
        record_id = None
    elif derived_id is not None and record_id != derived_id:
        errors.append(
            f"{origin} id {record_id!r} does not match category/instance "
            f"identity {derived_id!r}"
        )
    if raw.get("kind") != "object":
        errors.append(f"{origin} kind must be 'object'")
    if raw.get("origin") not in _TOMBSTONE_ORIGINS:
        errors.append(
            f"{origin} origin must be one of {sorted(_TOMBSTONE_ORIGINS)!r}"
        )
    if raw.get("status") != "removed":
        errors.append(f"{origin} status must be 'removed'")
    if not _valid_transaction_id(raw.get("transaction_id")):
        errors.append(f"{origin} has no positive transaction_id")
    evidence = raw.get("evidence")
    if not isinstance(evidence, Mapping) or not _has_content(evidence):
        errors.append(f"{origin} has no nonempty evidence object")

    graph_node = raw.get("graph_node")
    if not isinstance(graph_node, Mapping):
        errors.append(f"{origin} has no graph_node snapshot")
    else:
        for field, expected in (
            ("id", record_id),
            ("category", category),
            ("kind", "object"),
        ):
            if graph_node.get(field) != expected:
                errors.append(
                    f"{origin}.graph_node {field}={graph_node.get(field)!r}; "
                    f"expected {expected!r}"
                )
    placement = raw.get("placement")
    if not isinstance(placement, Mapping):
        errors.append(f"{origin} has no placement snapshot")
    else:
        for field, expected in (("category", category), ("instance", instance)):
            if placement.get(field) != expected:
                errors.append(
                    f"{origin}.placement {field}={placement.get(field)!r}; "
                    f"expected {expected!r}"
                )
        if not isinstance(placement.get("mesh_glb"), str) or not str(
            placement.get("mesh_glb")
        ).strip():
            errors.append(f"{origin}.placement has no nonempty mesh_glb")
    if not _SHA256_RE.fullmatch(str(raw.get("mesh_sha256") or "")):
        errors.append(f"{origin} has no valid mesh_sha256 digest")

    prior = raw.get("prior_runtime_record")
    if prior is not None:
        prior_id, prior_errors = _record_errors(prior, index)
        errors.extend(
            error.replace(
                f"runtime_inventory.objects[{index}]", f"{origin}.prior_runtime_record"
            )
            for error in prior_errors
        )
        if prior_id != record_id:
            errors.append(f"{origin}.prior_runtime_record identity does not match")
        if isinstance(graph_node, Mapping) and prior.get("graph_node") != graph_node:
            errors.append(f"{origin}.prior_runtime_record graph snapshot differs")
        if isinstance(placement, Mapping) and prior.get("placement") != placement:
            errors.append(f"{origin}.prior_runtime_record placement snapshot differs")
        revisions = prior.get("mesh_revisions")
        current_number = prior.get("current_mesh_revision")
        if isinstance(revisions, list):
            current = next(
                (
                    revision
                    for revision in revisions
                    if isinstance(revision, Mapping)
                    and revision.get("revision") == current_number
                ),
                None,
            )
            if isinstance(current, Mapping) and current.get("sha256") != raw.get(
                "mesh_sha256"
            ):
                errors.append(
                    f"{origin}.prior_runtime_record mesh digest differs from tombstone"
                )
    tombstone_origin = raw.get("origin")
    if tombstone_origin == "runtime_removed":
        if not isinstance(prior, Mapping) or prior.get("origin") != "runtime_added":
            errors.append(
                f"{origin} runtime removal requires prior runtime_added record"
            )
        if isinstance(graph_node, Mapping) and graph_node.get("runtime_added") is not True:
            errors.append(f"{origin}.graph_node is not marked runtime_added")
        if isinstance(placement, Mapping) and placement.get("runtime_added") is not True:
            errors.append(f"{origin}.placement is not marked runtime_added")
    elif tombstone_origin == "source_removed":
        if isinstance(prior, Mapping) and prior.get("origin") != "source_revised":
            errors.append(
                f"{origin} source removal prior record must be source_revised"
            )
        if isinstance(graph_node, Mapping) and graph_node.get("runtime_added") is True:
            errors.append(f"{origin}.graph_node source is marked runtime_added")
        if isinstance(placement, Mapping) and placement.get("runtime_added") is True:
            errors.append(f"{origin}.placement source is marked runtime_added")
    return record_id if isinstance(record_id, str) else None, errors


def runtime_inventory_errors(payload: Any) -> list[str]:
    """Return actionable structural errors for a runtime overlay payload."""

    if not isinstance(payload, Mapping):
        return ["runtime inventory must be a JSON object"]
    errors: list[str] = []
    if payload.get("schema_version") != RUNTIME_INVENTORY_SCHEMA_VERSION:
        errors.append(
            "runtime inventory schema_version must be "
            f"{RUNTIME_INVENTORY_SCHEMA_VERSION}"
        )
    objects = payload.get("objects")
    if not isinstance(objects, list):
        errors.append("runtime inventory objects must be a list")
        return errors
    tombstones = payload.get("tombstones", [])
    if not isinstance(tombstones, list):
        errors.append("runtime inventory tombstones must be a list")
        return errors

    seen_ids: set[str] = set()
    seen_mesh_names: dict[str, str] = {}
    for index, raw in enumerate(objects):
        record_id, record_errors = _record_errors(raw, index)
        errors.extend(record_errors)
        if record_id is None:
            continue
        if record_id in seen_ids:
            errors.append(f"duplicate runtime object id {record_id!r}")
        seen_ids.add(record_id)
        if not isinstance(raw, Mapping):
            continue
        placement = raw.get("placement")
        if not isinstance(placement, Mapping):
            continue
        mesh_name = placement.get("mesh_name")
        if not isinstance(mesh_name, str) or not mesh_name.strip():
            continue
        owner = seen_mesh_names.get(mesh_name)
        if owner is not None:
            errors.append(
                f"runtime mesh_name {mesh_name!r} is shared by {owner} and {record_id}"
            )
        else:
            seen_mesh_names[mesh_name] = record_id
    seen_tombstone_ids: set[str] = set()
    for index, raw in enumerate(tombstones):
        record_id, record_errors = _tombstone_errors(raw, index)
        errors.extend(record_errors)
        if record_id is None:
            continue
        if record_id in seen_tombstone_ids:
            errors.append(f"duplicate runtime tombstone id {record_id!r}")
        seen_tombstone_ids.add(record_id)
        if record_id in seen_ids:
            errors.append(
                f"runtime object id {record_id!r} is both active and tombstoned"
            )
    return errors


def validate_runtime_inventory_schema(
    payload: Any,
) -> dict[str, Mapping[str, Any]]:
    """Validate an overlay and return committed records keyed by graph ID."""

    errors = runtime_inventory_errors(payload)
    if errors:
        raise RuntimeObjectRepairError(
            "runtime object inventory is invalid:\n  - " + "\n  - ".join(errors)
        )
    assert isinstance(payload, Mapping)
    objects = payload["objects"]
    assert isinstance(objects, list)
    return {str(record["id"]): record for record in objects}


def runtime_inventory_tombstones(
    payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate an overlay and return durable removals keyed by graph ID."""

    validate_runtime_inventory_schema(payload)
    tombstones = payload.get("tombstones", [])
    assert isinstance(tombstones, list)
    return {str(record["id"]): record for record in tombstones}


def empty_runtime_inventory() -> dict[str, Any]:
    """Build an empty runtime overlay."""

    payload = {
        "schema_version": RUNTIME_INVENTORY_SCHEMA_VERSION,
        "objects": [],
        "tombstones": [],
    }
    validate_runtime_inventory_schema(payload)
    return payload


def new_runtime_inventory(scene_dir: str | Path) -> dict[str, Any]:
    """Create an in-memory empty overlay for an already-preprocessed scene."""

    masks_path = Path(scene_dir) / "masks" / "masks.json"
    if not masks_path.is_file():
        raise RuntimeObjectRepairError(f"source masks are missing: {masks_path}")
    return empty_runtime_inventory()


def load_runtime_inventory(scene_dir: str | Path) -> dict[str, Any] | None:
    ""

    path = runtime_inventory_path(scene_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeObjectRepairError(
            f"runtime object inventory JSON is unreadable: {path} ({exc})"
        ) from exc
    validate_runtime_inventory_schema(payload)
    assert isinstance(payload, dict)
    return payload


def write_runtime_inventory(scene_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a structurally valid active overlay.

    This helper intentionally writes only the overlay.  A caller mutating graph,
    placement, Blender, or physics state must include it in that caller's wider
    transaction protocol.
    """

    validate_runtime_inventory_schema(payload)
    destination = runtime_inventory_path(scene_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def procedural_capture_of(record) -> dict | None:
    ""
    placement = (record or {}).get("placement") if isinstance(record, dict) else None
    capture = (placement or {}).get("procedural_capture") if isinstance(placement, dict) else None
    if isinstance(capture, dict) and capture.get("source_schema") == "gpt6_blender_empty_tree_v1":
        return capture
    return None


def build_runtime_object_record(
    *,
    transaction_id: int,
    category: str,
    instance: int,
    mask_path: str | None,
    evidence: Mapping[str, Any],
    graph_node: Mapping[str, Any],
    placement: Mapping[str, Any],
    mesh_sha256: str,
    physical_material: Mapping[str, Any],
    visual_material: Mapping[str, Any],
    mask_sha256: str | None,
    mesh_revision: int = 1,
    revision_provenance: Mapping[str, Any] | None = None,
    mask_policy: str | None = None,
    origin: str = "runtime_added",
    source_mask_path: str | None = None,
    source_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Build and validate one committed runtime-object record.

    ``mask_policy='excluded'`` is the only valid representation for a directly
    authored object which has no backend-produced segmentation.  Its effective mask
    fields remain null; source replacements may separately retain immutable source
    mask provenance.  The batch stager can infer ``source_revised`` for a replacement
    and supply that provenance from the pre-mutation source snapshot.
    """

    record_id = _object_id(category, instance)
    if record_id is None:
        raise RuntimeObjectRepairError("category/instance cannot form an object id")
    resolved_mask_policy = mask_policy or (
        "excluded" if mask_path in (None, "") else "tracked"
    )
    record: dict[str, Any] = {
        "id": record_id,
        "category": category,
        "instance": instance,
        "kind": "object",
        "origin": origin,
        "status": "committed",
        "transaction_id": transaction_id,
        "mask_path": mask_path,
        "scoring_mask_path": mask_path,
        "mask_sha256": mask_sha256,
        "scoring_mask_sha256": mask_sha256,
        "mask_policy": resolved_mask_policy,
        "evidence": copy.deepcopy(dict(evidence)),
        "graph_node": copy.deepcopy(dict(graph_node)),
        "placement": copy.deepcopy(dict(placement)),
        "current_mesh_revision": mesh_revision,
        "mesh_revisions": [
            {
                "revision": mesh_revision,
                "transaction_id": transaction_id,
                "status": "committed",
                "mesh_name": placement.get("mesh_name"),
                "mesh_glb": placement.get("mesh_glb"),
                "sha256": mesh_sha256,
                "provenance": copy.deepcopy(
                    dict(revision_provenance or {"source": "runtime_addition"})
                ),
            }
        ],
        "physical_material": copy.deepcopy(dict(physical_material)),
        "visual_material": copy.deepcopy(dict(visual_material)),
    }
    if source_mask_path is not None:
        record["source_mask_path"] = source_mask_path
    if source_mask_sha256 is not None:
        record["source_mask_sha256"] = source_mask_sha256
    _, errors = _record_errors(record, 0)
    if errors:
        raise RuntimeObjectRepairError(
            "runtime object record is invalid:\n  - " + "\n  - ".join(errors)
        )
    return record


def _load_masks(scene: Path) -> Mapping[str, Any]:
    path = scene / "masks" / "masks.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeObjectRepairError(
            f"source masks JSON is unreadable: {path} ({exc})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeObjectRepairError(f"source masks are not a JSON object: {path}")
    return payload


def _overlay_for_staging(
    scene: Path, runtime_inventory: Mapping[str, Any] | None
) -> dict[str, Any]:
    if runtime_inventory is None:
        loaded = load_runtime_inventory(scene)
        if loaded is not None:
            return copy.deepcopy(loaded)
        return new_runtime_inventory(scene)
    validate_runtime_inventory_schema(runtime_inventory)
    return copy.deepcopy(dict(runtime_inventory))


def _resolve_existing_artifact(scene: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeObjectRepairError(f"{field} has no nonempty artifact path")
    path = Path(value)
    candidates = (
        [path.resolve()]
        if path.is_absolute()
        else [
            (scene / path).resolve(),
            (Path(__file__).resolve().parents[3] / path).resolve(),
            path.resolve(),
        ]
    )
    resolved = next(
        (candidate for candidate in candidates if candidate.is_file()), None
    )
    if resolved is None:
        raise RuntimeObjectRepairError(f"{field} artifact is missing: {candidates[0]}")
    if resolved.stat().st_size <= 0:
        raise RuntimeObjectRepairError(f"{field} artifact is empty: {resolved}")
    return resolved


def _validate_staged_bundle(
    scene: Path,
    masks: Mapping[str, Any],
    runtime_inventory: Mapping[str, Any],
    graph: Mapping[str, Any],
    placement: Mapping[str, Any],
) -> None:
    from lib.tools.geometry.inventory_contract import (
        validate_object_materialization,
        validate_runtime_object_artifacts,
        validate_runtime_object_bindings,
        validate_runtime_source_masks,
        validate_scene_graph_inventory,
    )

    validate_runtime_source_masks(runtime_inventory, scene / "masks" / "masks.json")
    validate_scene_graph_inventory(masks, graph, runtime_inventory=runtime_inventory)
    validate_object_materialization(graph, placement, artifact_root=scene)
    validate_runtime_object_bindings(runtime_inventory, graph, placement)
    validate_runtime_object_artifacts(runtime_inventory, scene)


def stage_runtime_object_batch(
    scene_dir: str | Path,
    *,
    graph: Mapping[str, Any],
    placement: Mapping[str, Any],
    transaction_id: int,
    additions: Sequence[Mapping[str, Any]],
    removed_object_ids: Sequence[str],
    runtime_inventory: Mapping[str, Any] | None = None,
    removal_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Stage one atomic declared add/remove/replace inventory transition.

    A name present in both ``additions`` and ``removed_object_ids`` is a physical-
    identity-preserving replacement.  Replacements retain mesh revision history and
    never create a tombstone.  Pure removals create durable tombstones containing the
    exact pre-mutation graph/placement snapshots (and prior overlay record, when one
    exists); referenced mesh files are intentionally never deleted here.

    This helper is specifically for directly authored initializer geometry, so every
    added/replacement record must use ``mask_policy='excluded'`` and null mask fields.
    It authenticates the input bundle, mutates only deep copies, then validates the
    final bundle, allowing a batch to add a support before its child or remove several
    mutually dependent objects.
    """

    if not _valid_transaction_id(transaction_id):
        raise RuntimeObjectRepairError("object batch needs a positive transaction_id")
    if isinstance(additions, (str, bytes)) or not isinstance(additions, Sequence):
        raise RuntimeObjectRepairError("object batch additions must be a sequence")
    if isinstance(removed_object_ids, (str, bytes)) or not isinstance(
        removed_object_ids, Sequence
    ):
        raise RuntimeObjectRepairError(
            "object batch removed_object_ids must be a sequence"
        )

    scene = Path(scene_dir).resolve()
    masks = _load_masks(scene)
    staged_overlay = _overlay_for_staging(scene, runtime_inventory)
    staged_graph = copy.deepcopy(dict(graph))
    staged_placement = copy.deepcopy(dict(placement))
    # Authenticate every snapshot that can become deletion or revision provenance
    # before the mutation has a chance to hide a stale baseline inconsistency.
    _validate_staged_bundle(
        scene, masks, staged_overlay, staged_graph, staged_placement
    )
    records_before = validate_runtime_inventory_schema(staged_overlay)
    tombstones_before = runtime_inventory_tombstones(staged_overlay)

    nodes = staged_graph.get("nodes")
    rows = staged_placement.get("objects")
    relationships = staged_graph.get("relationships", [])
    if not isinstance(nodes, list):
        raise RuntimeObjectRepairError("scene graph nodes must be a list")
    if not isinstance(rows, list):
        raise RuntimeObjectRepairError("placement.objects must be a list")
    if not isinstance(relationships, list):
        raise RuntimeObjectRepairError("scene graph relationships must be a list")

    def index_nodes() -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                raise RuntimeObjectRepairError("scene graph node is not an object")
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                raise RuntimeObjectRepairError("scene graph node has no id")
            if node_id in indexed:
                raise RuntimeObjectRepairError(f"duplicate graph id: {node_id}")
            indexed[node_id] = node
        return indexed

    def index_rows() -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeObjectRepairError("placement row is not an object")
            row_id = _object_id(row.get("category"), row.get("instance"))
            if row_id is None:
                raise RuntimeObjectRepairError(
                    "placement row has invalid category/instance"
                )
            if row_id in indexed:
                raise RuntimeObjectRepairError(f"duplicate placement id: {row_id}")
            indexed[row_id] = row
        return indexed

    node_by_id = index_nodes()
    row_by_id = index_rows()
    addition_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(additions):
        if not isinstance(raw, Mapping):
            raise RuntimeObjectRepairError(f"object batch additions[{index}] is invalid")
        candidate = copy.deepcopy(dict(raw))
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise RuntimeObjectRepairError(
                f"object batch additions[{index}] has no id"
            )
        if candidate_id in addition_by_id:
            raise RuntimeObjectRepairError(
                f"duplicate object batch addition: {candidate_id}"
            )
        if candidate.get("mask_policy", "tracked") != "excluded":
            raise RuntimeObjectRepairError(
                f"directly authored object {candidate_id} must use "
                "mask_policy='excluded'"
            )
        if candidate_id in tombstones_before:
            raise RuntimeObjectRepairError(
                f"tombstoned object identity cannot be reused: {candidate_id}"
            )
        addition_by_id[candidate_id] = candidate

    removed_ids: list[str] = []
    removed_seen: set[str] = set()
    for index, value in enumerate(removed_object_ids):
        if not isinstance(value, str) or not value.strip():
            raise RuntimeObjectRepairError(
                f"object batch removed_object_ids[{index}] is not a nonempty id"
            )
        if value in removed_seen:
            raise RuntimeObjectRepairError(f"duplicate object batch removal: {value}")
        removed_seen.add(value)
        removed_ids.append(value)

    replacement_ids = set(addition_by_id) & removed_seen
    pure_removal_ids = removed_seen - replacement_ids
    pure_addition_ids = set(addition_by_id) - replacement_ids
    collisions = sorted(pure_addition_ids & set(node_by_id))
    if collisions:
        raise RuntimeObjectRepairError(
            "object batch additions collide with active identities: "
            + ", ".join(collisions)
        )

    removed_snapshots: dict[str, dict[str, Any]] = {}
    for object_id in removed_ids:
        node = node_by_id.get(object_id)
        row = row_by_id.get(object_id)
        if node is None or node.get("kind") != "object":
            raise RuntimeObjectRepairError(
                f"removed object {object_id} needs exactly one active object graph node"
            )
        if row is None:
            raise RuntimeObjectRepairError(
                f"removed object {object_id} needs exactly one placement row"
            )
        removed_snapshots[object_id] = {
            "graph_node": copy.deepcopy(node),
            "placement": copy.deepcopy(row),
            "runtime_record": copy.deepcopy(records_before.get(object_id)),
        }

    active_records: dict[str, dict[str, Any]] = {
        record_id: copy.deepcopy(dict(record))
        for record_id, record in records_before.items()
    }
    tombstones: list[dict[str, Any]] = [
        copy.deepcopy(dict(record))
        for record in staged_overlay.get("tombstones", [])
        if isinstance(record, Mapping)
    ]
    for object_id in pure_removal_ids:
        snapshot = removed_snapshots[object_id]
        old_node = snapshot["graph_node"]
        old_row = snapshot["placement"]
        old_record = snapshot["runtime_record"]
        mesh = _resolve_existing_artifact(
            scene, old_row.get("mesh_glb"), f"removed object {object_id} mesh_glb"
        )
        category = old_row.get("category")
        instance = old_row.get("instance")
        evidence = (
            removal_evidence.get(object_id)
            if isinstance(removal_evidence, Mapping)
            else None
        )
        if not isinstance(evidence, Mapping) or not _has_content(evidence):
            evidence = {
                "source": "declared_runtime_mutation",
                "operation": "remove_object",
            }
        tombstone: dict[str, Any] = {
            "id": object_id,
            "category": category,
            "instance": instance,
            "kind": "object",
            "origin": (
                "runtime_removed"
                if old_node.get("runtime_added") is True
                else "source_removed"
            ),
            "status": "removed",
            "transaction_id": transaction_id,
            "evidence": copy.deepcopy(dict(evidence)),
            "graph_node": old_node,
            "placement": old_row,
            "mesh_sha256": sha256_file(mesh),
        }
        if old_record is not None:
            tombstone["prior_runtime_record"] = old_record
        tombstones.append(tombstone)
        active_records.pop(object_id, None)

    # Temporarily remove every declared identity; replacements are reinserted below.
    nodes[:] = [node for node in nodes if node.get("id") not in removed_seen]
    rows[:] = [
        row
        for row in rows
        if _object_id(row.get("category"), row.get("instance")) not in removed_seen
    ]
    staged_graph["relationships"] = [
        relationship
        for relationship in relationships
        if not (
            isinstance(relationship, Mapping)
            and pure_removal_ids
            & {relationship.get("a"), relationship.get("b")}
        )
    ]

    for object_id, candidate in addition_by_id.items():
        graph_node = candidate.get("graph_node")
        placement_row = candidate.get("placement")
        if not isinstance(graph_node, dict) or not isinstance(placement_row, dict):
            raise RuntimeObjectRepairError(
                f"object batch addition {object_id} needs graph/placement snapshots"
            )
        candidate["mask_policy"] = "excluded"
        for field in (
            "mask_path",
            "mask_sha256",
            "scoring_mask_path",
            "scoring_mask_sha256",
        ):
            candidate[field] = None
        graph_node["mask_path"] = None
        placement_row.pop("mask_path", None)

        if object_id in replacement_ids:
            snapshot = removed_snapshots[object_id]
            old_node = snapshot["graph_node"]
            old_row = snapshot["placement"]
            old_record = snapshot["runtime_record"]
            if (
                candidate.get("category") != old_row.get("category")
                or candidate.get("instance") != old_row.get("instance")
            ):
                raise RuntimeObjectRepairError(
                    f"replacement changes physical identity {object_id}"
                )
            if placement_row.get("mesh_name") != old_row.get("mesh_name"):
                raise RuntimeObjectRepairError(
                    f"replacement {object_id} changes canonical mesh_name"
                )

            if old_node.get("runtime_added") is True:
                if old_record is None:
                    raise RuntimeObjectRepairError(
                        f"runtime replacement {object_id} has no prior overlay record"
                    )
                candidate["origin"] = old_record.get("origin")
                candidate["transaction_id"] = old_record.get("transaction_id")
                graph_node["runtime_added"] = True
                graph_node["runtime_transaction_id"] = old_node.get(
                    "runtime_transaction_id"
                )
                placement_row["runtime_added"] = True
                placement_row["runtime_transaction_id"] = old_row.get(
                    "runtime_transaction_id"
                )
                for field in ("source_mask_path", "source_mask_sha256"):
                    if field in old_record:
                        candidate[field] = old_record[field]
                history = copy.deepcopy(old_record["mesh_revisions"])
            else:
                candidate["origin"] = "source_revised"
                graph_node.pop("runtime_added", None)
                graph_node.pop("runtime_transaction_id", None)
                placement_row.pop("runtime_added", None)
                placement_row.pop("runtime_transaction_id", None)
                source_mask_path = (
                    old_record.get("source_mask_path")
                    if isinstance(old_record, Mapping)
                    and old_record.get("source_mask_path")
                    else old_node.get("mask_path")
                )
                source_mask = _resolve_existing_artifact(
                    scene,
                    source_mask_path,
                    f"source replacement {object_id} mask_path",
                )
                candidate["source_mask_path"] = source_mask_path
                candidate["source_mask_sha256"] = sha256_file(source_mask)
                if old_record is None:
                    old_mesh = _resolve_existing_artifact(
                        scene,
                        old_row.get("mesh_glb"),
                        f"source replacement {object_id} mesh_glb",
                    )
                    history = [
                        {
                            "revision": 1,
                            "transaction_id": None,
                            "status": "superseded",
                            "mesh_name": old_row.get("mesh_name"),
                            "mesh_glb": old_row.get("mesh_glb"),
                            "sha256": sha256_file(old_mesh),
                            "provenance": {"source": "preprocess"},
                        }
                    ]
                else:
                    candidate["transaction_id"] = old_record.get("transaction_id")
                    history = copy.deepcopy(old_record["mesh_revisions"])
                    for field in ("source_mask_path", "source_mask_sha256"):
                        if field in old_record:
                            candidate[field] = old_record[field]

            for revision in history:
                if isinstance(revision, dict) and revision.get("status") == "committed":
                    revision["status"] = "superseded"
            incoming_revisions = candidate.get("mesh_revisions")
            incoming_current = candidate.get("current_mesh_revision")
            if not isinstance(incoming_revisions, list):
                raise RuntimeObjectRepairError(
                    f"replacement {object_id} has no mesh revision"
                )
            incoming = next(
                (
                    revision
                    for revision in incoming_revisions
                    if isinstance(revision, Mapping)
                    and revision.get("revision") == incoming_current
                ),
                None,
            )
            if incoming is None:
                raise RuntimeObjectRepairError(
                    f"replacement {object_id} has no current mesh revision"
                )
            new_revision = copy.deepcopy(dict(incoming))
            next_revision = max(int(row["revision"]) for row in history) + 1
            new_revision.update(
                revision=next_revision,
                transaction_id=transaction_id,
                status="committed",
            )
            candidate["mesh_revisions"] = [*history, new_revision]
            candidate["current_mesh_revision"] = next_revision
        else:
            candidate["origin"] = "runtime_added"
            candidate["transaction_id"] = transaction_id
            candidate.pop("source_mask_path", None)
            candidate.pop("source_mask_sha256", None)
            graph_node["runtime_added"] = True
            graph_node["runtime_transaction_id"] = transaction_id
            placement_row["runtime_added"] = True
            placement_row["runtime_transaction_id"] = transaction_id
            revisions = candidate.get("mesh_revisions")
            current_number = candidate.get("current_mesh_revision")
            if isinstance(revisions, list):
                current = next(
                    (
                        revision
                        for revision in revisions
                        if isinstance(revision, dict)
                        and revision.get("revision") == current_number
                    ),
                    None,
                )
                if current is not None:
                    current["transaction_id"] = transaction_id

        _, candidate_errors = _record_errors(candidate, 0)
        if candidate_errors:
            raise RuntimeObjectRepairError(
                f"object batch addition {object_id} is invalid:\n  - "
                + "\n  - ".join(candidate_errors)
            )
        active_records[object_id] = candidate
        nodes.append(copy.deepcopy(graph_node))
        rows.append(copy.deepcopy(placement_row))

    # Rebuild the inverse child lists from final parent links.  This is both order
    # independent and prevents a removed child from lingering in a source parent.
    final_nodes = index_nodes()
    for node in nodes:
        node["children"] = []
    for node_id, node in final_nodes.items():
        if node.get("kind") == "root_surface":
            continue
        parent_id = node.get("parent")
        parent = final_nodes.get(parent_id)
        if parent is None:
            raise RuntimeObjectRepairError(
                f"object batch leaves {node_id} with missing parent {parent_id!r}"
            )
        parent["children"].append(node_id)
    for node in nodes:
        node["children"].sort()

    # Graph snapshots include children, so a support's overlay record must advance
    # when another object is added to or removed from it in this same transaction.
    for record_id, record in active_records.items():
        live_node = final_nodes.get(record_id)
        if live_node is not None:
            record["graph_node"] = copy.deepcopy(live_node)
        live_row = next(
            (
                row
                for row in rows
                if _object_id(row.get("category"), row.get("instance")) == record_id
            ),
            None,
        )
        if live_row is not None:
            record["placement"] = copy.deepcopy(live_row)

    staged_overlay["objects"] = [
        active_records[record_id] for record_id in sorted(active_records)
    ]
    staged_overlay["tombstones"] = sorted(tombstones, key=lambda row: str(row["id"]))
    validate_runtime_inventory_schema(staged_overlay)
    _validate_staged_bundle(
        scene, masks, staged_overlay, staged_graph, staged_placement
    )
    return {
        "runtime_inventory": staged_overlay,
        "scene_graph": staged_graph,
        "placement": staged_placement,
    }


def stage_runtime_mesh_revision(
    scene_dir: str | Path,
    *,
    graph: Mapping[str, Any],
    placement: Mapping[str, Any],
    object_id: str,
    transaction_id: int,
    mesh_glb: str,
    mesh_sha256: str,
    mesh_name: str | None = None,
    replacement_placement: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    revision_provenance: Mapping[str, Any] | None = None,
    physical_material: Mapping[str, Any] | None = None,
    visual_material: Mapping[str, Any] | None = None,
    mask_policy: str | None = None,
    runtime_inventory: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a validated mesh replacement for a source or runtime-added object.

    A first revision of a preprocessing-origin object creates a ``source_revised``
    overlay record while leaving the source mask/graph identity intact.  That first
    revision requires evidence plus explicit visual and physical material provenance.
    """

    if not _valid_transaction_id(transaction_id):
        raise RuntimeObjectRepairError("mesh revision needs a positive transaction_id")
    scene = Path(scene_dir).resolve()
    masks = _load_masks(scene)
    staged_overlay = _overlay_for_staging(scene, runtime_inventory)
    staged_graph = copy.deepcopy(dict(graph))
    staged_placement = copy.deepcopy(dict(placement))
    records = validate_runtime_inventory_schema(staged_overlay)
    record = records.get(object_id)
    rows = staged_placement.get("objects")
    if not isinstance(rows, list):
        raise RuntimeObjectRepairError("placement.objects must be a list")
    matching_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _object_id(row.get("category"), row.get("instance")) == object_id
    ]
    if len(matching_rows) != 1:
        raise RuntimeObjectRepairError(
            f"object {object_id} needs exactly one placement row"
        )
    placement_row = matching_rows[0]
    baseline_placement = copy.deepcopy(placement_row)
    nodes = staged_graph.get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeObjectRepairError("scene graph nodes must be a list")
    matching_nodes = [
        node for node in nodes if isinstance(node, dict) and node.get("id") == object_id
    ]
    if len(matching_nodes) != 1 or matching_nodes[0].get("kind") != "object":
        raise RuntimeObjectRepairError(
            f"object {object_id} needs exactly one object-kind graph node"
        )
    graph_node = matching_nodes[0]
    resolved_mask_policy = (
        mask_policy
        if mask_policy is not None
        else (record.get("mask_policy", "tracked") if record is not None else "tracked")
    )
    if resolved_mask_policy not in _MASK_POLICIES:
        raise RuntimeObjectRepairError(
            f"mesh revision mask_policy must be one of {sorted(_MASK_POLICIES)!r}"
        )

    canonical_mesh_name = str(baseline_placement.get("mesh_name") or "")
    if not canonical_mesh_name:
        raise RuntimeObjectRepairError(f"object {object_id} placement has no mesh_name")
    if mesh_name is not None and mesh_name != canonical_mesh_name:
        raise RuntimeObjectRepairError(
            f"mesh revision cannot rename {canonical_mesh_name!r} to {mesh_name!r}"
        )
    resolved_mesh_name = canonical_mesh_name
    replacement = copy.deepcopy(
        dict(replacement_placement)
        if replacement_placement is not None
        else baseline_placement
    )
    if (
        _object_id(replacement.get("category"), replacement.get("instance"))
        != object_id
    ):
        raise RuntimeObjectRepairError(
            f"replacement placement identity does not match {object_id}"
        )
    replacement["mesh_name"] = resolved_mesh_name
    replacement["mesh_glb"] = mesh_glb

    if record is None:
        if graph_node.get("runtime_added") is True:
            raise RuntimeObjectRepairError(
                f"runtime-added object {object_id} has no committed overlay record"
            )
        if not isinstance(evidence, Mapping) or not _has_content(evidence):
            raise RuntimeObjectRepairError(
                "first source-object mesh revision requires nonempty evidence"
            )
        if physical_material is None or visual_material is None:
            raise RuntimeObjectRepairError(
                "first source-object mesh revision requires physical_material and "
                "visual_material provenance"
            )
        category = baseline_placement.get("category")
        instance = baseline_placement.get("instance")
        if _object_id(category, instance) != object_id:
            raise RuntimeObjectRepairError(
                f"placement identity does not match source object {object_id}"
            )
        old_mesh = _resolve_existing_artifact(
            scene,
            baseline_placement.get("mesh_glb"),
            f"source object {object_id} mesh_glb",
        )
        mask_path = graph_node.get("mask_path")
        source_mask = _resolve_existing_artifact(
            scene, mask_path, f"source object {object_id} mask_path"
        )
        if resolved_mask_policy == "excluded":
            scoring_mask_path = None
            scoring_mask_sha256 = None
            replacement.pop("mask_path", None)
            graph_node["mask_path"] = None
        else:
            scoring_mask_path = replacement.get("mask_path") or mask_path
            scoring_mask = _resolve_existing_artifact(
                scene,
                scoring_mask_path,
                f"source object {object_id} scoring_mask_path",
            )
            if scoring_mask == source_mask:
                # Canonicalize equivalent absolute/relative spellings so reusing the
                # immutable source mask does not look like runtime segmentation.
                scoring_mask_path = mask_path
                replacement["mask_path"] = mask_path
            scoring_mask_sha256 = sha256_file(scoring_mask)
        record = {
            "id": object_id,
            "category": category,
            "instance": instance,
            "kind": "object",
            "origin": "source_revised",
            "status": "committed",
            "transaction_id": transaction_id,
            "mask_policy": resolved_mask_policy,
            "mask_path": mask_path if resolved_mask_policy == "tracked" else None,
            "scoring_mask_path": scoring_mask_path,
            "mask_sha256": (
                sha256_file(source_mask)
                if resolved_mask_policy == "tracked"
                else None
            ),
            "scoring_mask_sha256": scoring_mask_sha256,
            "source_mask_path": mask_path,
            "source_mask_sha256": sha256_file(source_mask),
            "evidence": copy.deepcopy(dict(evidence)),
            "graph_node": copy.deepcopy(graph_node),
            "placement": copy.deepcopy(replacement),
            "current_mesh_revision": 2,
            "mesh_revisions": [
                {
                    "revision": 1,
                    "transaction_id": None,
                    "status": "superseded",
                    "mesh_name": baseline_placement.get("mesh_name"),
                    "mesh_glb": baseline_placement.get("mesh_glb"),
                    "sha256": sha256_file(old_mesh),
                    "provenance": {"source": "preprocess"},
                },
                {
                    "revision": 2,
                    "transaction_id": transaction_id,
                    "status": "committed",
                    "mesh_name": resolved_mesh_name,
                    "mesh_glb": mesh_glb,
                    "sha256": mesh_sha256,
                    "provenance": copy.deepcopy(
                        dict(revision_provenance or {"source": "runtime_mesh_edit"})
                    ),
                },
            ],
            "physical_material": copy.deepcopy(dict(physical_material)),
            "visual_material": copy.deepcopy(dict(visual_material)),
        }
        staged_overlay.setdefault("objects", []).append(record)
    else:
        if not isinstance(record, dict):
            raise RuntimeObjectRepairError(
                f"runtime overlay record {object_id} is not mutable"
            )
        revisions = record["mesh_revisions"]
        assert isinstance(revisions, list)
        current_number = record["current_mesh_revision"]
        current = next(
            revision
            for revision in revisions
            if isinstance(revision, dict) and revision.get("revision") == current_number
        )
        current["status"] = "superseded"
        next_revision = max(int(revision["revision"]) for revision in revisions) + 1
        overlay_mesh_name = str(current["mesh_name"])
        if overlay_mesh_name != canonical_mesh_name:
            raise RuntimeObjectRepairError(
                f"runtime overlay mesh name {overlay_mesh_name!r} differs from "
                f"canonical placement {canonical_mesh_name!r}"
            )
        revisions.append(
            {
                "revision": next_revision,
                "transaction_id": transaction_id,
                "status": "committed",
                "mesh_name": resolved_mesh_name,
                "mesh_glb": mesh_glb,
                "sha256": mesh_sha256,
                "provenance": copy.deepcopy(
                    dict(revision_provenance or {"source": "runtime_mesh_edit"})
                ),
            }
        )
        record["current_mesh_revision"] = next_revision
        embedded_placement = record["placement"]
        assert isinstance(embedded_placement, dict)
        embedded_placement.clear()
        embedded_placement.update(copy.deepcopy(replacement))
        if resolved_mask_policy == "excluded":
            if record.get("origin") == "source_revised" and not record.get(
                "source_mask_path"
            ):
                source_mask_path = record.get("mask_path")
                source_mask = _resolve_existing_artifact(
                    scene,
                    source_mask_path,
                    f"source object {object_id} mask_path",
                )
                record["source_mask_path"] = source_mask_path
                record["source_mask_sha256"] = sha256_file(source_mask)
            record["mask_policy"] = "excluded"
            for field in (
                "mask_path",
                "mask_sha256",
                "scoring_mask_path",
                "scoring_mask_sha256",
            ):
                record[field] = None
            replacement.pop("mask_path", None)
            embedded_placement.clear()
            embedded_placement.update(copy.deepcopy(replacement))
            embedded_node = record["graph_node"]
            assert isinstance(embedded_node, dict)
            embedded_node["mask_path"] = None
            graph_node["mask_path"] = None
        else:
            scoring_mask_path = replacement.get("mask_path") or record.get("mask_path")
            scoring_mask = _resolve_existing_artifact(
                scene,
                scoring_mask_path,
                f"runtime object {object_id} scoring_mask_path",
            )
            if record.get("origin") == "source_revised":
                source_mask_path = record.get("source_mask_path") or record.get(
                    "mask_path"
                )
                source_mask = _resolve_existing_artifact(
                    scene,
                    source_mask_path,
                    f"source object {object_id} mask_path",
                )
                if scoring_mask == source_mask:
                    scoring_mask_path = source_mask_path
                    replacement["mask_path"] = scoring_mask_path
                    embedded_placement.clear()
                    embedded_placement.update(copy.deepcopy(replacement))
            record["mask_policy"] = "tracked"
            record["scoring_mask_path"] = scoring_mask_path
            record["scoring_mask_sha256"] = sha256_file(scoring_mask)
            if record.get("origin") == "runtime_added":
                # Runtime re-segmentation advances graph/scoring evidence together.
                record["mask_path"] = scoring_mask_path
                record["mask_sha256"] = record["scoring_mask_sha256"]
                embedded_node = record["graph_node"]
                assert isinstance(embedded_node, dict)
                embedded_node["mask_path"] = scoring_mask_path
                graph_node["mask_path"] = scoring_mask_path
        if evidence is not None:
            record["evidence"] = copy.deepcopy(dict(evidence))
        if physical_material is not None:
            record["physical_material"] = copy.deepcopy(dict(physical_material))
        if visual_material is not None:
            record["visual_material"] = copy.deepcopy(dict(visual_material))

    placement_row.clear()
    placement_row.update(replacement)

    validate_runtime_inventory_schema(staged_overlay)
    _validate_staged_bundle(
        scene, masks, staged_overlay, staged_graph, staged_placement
    )
    return {
        "runtime_inventory": staged_overlay,
        "scene_graph": staged_graph,
        "placement": staged_placement,
    }


__all__ = [
    "RUNTIME_INVENTORY_RELATIVE_PATH",
    "RUNTIME_INVENTORY_SCHEMA_VERSION",
    "RuntimeObjectRepairError",
    "build_runtime_object_record",
    "empty_runtime_inventory",
    "load_runtime_inventory",
    "new_runtime_inventory",
    "runtime_inventory_errors",
    "runtime_inventory_path",
    "runtime_inventory_tombstones",
    "sha256_file",
    "stage_runtime_mesh_revision",
    "stage_runtime_object_batch",
    "validate_runtime_inventory_schema",
    "write_runtime_inventory",
]
