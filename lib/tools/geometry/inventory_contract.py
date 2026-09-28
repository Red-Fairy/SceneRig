"""Lossless scene-inventory validation at pipeline artifact boundaries.

``masks/masks.json`` owns the retained source inventory and remains immutable.
``scene_graph.json`` may enrich those records with geometry and may acquire explicitly
marked runtime root surfaces.  A GPT-6 harness may additionally layer committed
objects from the backend-owned ``runtime_objects/inventory.json`` overlay, but it must
not silently lose, replace, or reinterpret a retained source record.  Placement and
Blender materialization are later representations of the resolved non-root inventory.

This module deliberately uses direct semantic comparisons.  File hashes continue to
belong to the preprocess manifest and compiled-constraint provenance; a digest would
only make inventory mismatches less actionable here.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lib.tools.geometry.runtime_object_repair import (
    RuntimeObjectRepairError,
    load_runtime_inventory,
    runtime_inventory_tombstones,
    validate_runtime_inventory_schema,
)


class InventoryContractError(RuntimeError):
    """A retained entity diverged between authoritative pipeline artifacts."""


def _fail(boundary: str, errors: Iterable[str]) -> None:
    rows = [str(error) for error in errors if str(error)]
    if rows:
        raise InventoryContractError(
            f"{boundary} inventory contract failed:\n  - " + "\n  - ".join(rows)
        )


def _record_id(record: Mapping[str, Any], origin: str) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    category = record.get("category")
    instance = record.get("instance")
    if not isinstance(category, str) or not category.strip():
        errors.append(f"{origin} has no nonempty string category")
    if not isinstance(instance, int) or isinstance(instance, bool) or instance < 0:
        errors.append(
            f"{origin} has invalid instance {instance!r}; expected integer >= 0"
        )
    if errors:
        return None, errors
    return f"{category}#{instance}", []


def authoritative_inventory(masks: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the final retained masks inventory keyed by exact scene-graph id.

    Records already moved to ``dropped`` are intentionally absent from ``instances``
    and therefore do not participate.  Name-only verifier-added roots are represented
    by ``unmasked_root_surfaces`` and receive their implicit graph values here.
    """

    errors: list[str] = []
    inventory: dict[str, dict[str, Any]] = {}
    instances = masks.get("instances")
    if not isinstance(instances, list):
        _fail("masks", ["instances must be a list"])
    unmasked = masks.get("unmasked_root_surfaces", [])
    if not isinstance(unmasked, list):
        _fail("masks", ["unmasked_root_surfaces must be a list"])

    for section, rows in (
        ("instances", instances),
        ("unmasked_root_surfaces", unmasked),
    ):
        for index, raw in enumerate(rows):
            origin = f"masks.{section}[{index}]"
            if not isinstance(raw, Mapping):
                errors.append(f"{origin} is not an object")
                continue
            record_id, id_errors = _record_id(raw, origin)
            errors.extend(id_errors)
            if record_id is None:
                continue
            if record_id in inventory:
                errors.append(
                    f"duplicate retained id {record_id!r} (collision at {origin})"
                )
                continue
            kind = (
                "root_surface"
                if section == "unmasked_root_surfaces"
                else raw.get("kind")
            )
            if kind not in {"object", "root_surface"}:
                errors.append(
                    f"{origin} ({record_id}) has invalid kind {kind!r}; expected "
                    "'object' or 'root_surface'"
                )
                continue
            if section == "unmasked_root_surfaces" and raw.get("kind") not in (
                None,
                "root_surface",
            ):
                errors.append(
                    f"{origin} ({record_id}) is an unmasked root but declares "
                    f"kind={raw.get('kind')!r}"
                )
                continue
            inventory[record_id] = {
                "id": record_id,
                "category": raw.get("category"),
                "instance": raw.get("instance"),
                "kind": kind,
                "support": raw.get("support"),
                "mask_path": raw.get("mask_path"),
                "source_section": section,
            }
    _fail("masks", errors)
    return inventory


def _runtime_records(
    runtime_inventory: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if runtime_inventory is None:
        return {}
    try:
        return validate_runtime_inventory_schema(runtime_inventory)
    except RuntimeObjectRepairError as exc:
        raise InventoryContractError(str(exc)) from exc


def _runtime_tombstones(
    runtime_inventory: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if runtime_inventory is None:
        return {}
    try:
        return runtime_inventory_tombstones(runtime_inventory)
    except RuntimeObjectRepairError as exc:
        raise InventoryContractError(str(exc)) from exc


def resolved_inventory(
    masks: Mapping[str, Any],
    runtime_inventory: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return source masks plus committed runtime objects keyed by graph ID.

    The function never mutates or synthesizes entries in ``masks``.  Runtime-added
    records must be new identities; source-revision records may overlay only their
    source identity's materialized mesh fields.
    """

    inventory = authoritative_inventory(masks)
    source_inventory = copy.deepcopy(inventory)
    errors: list[str] = []
    for record_id, tombstone in _runtime_tombstones(runtime_inventory).items():
        origin = tombstone.get("origin")
        source = source_inventory.get(record_id)
        if origin == "source_removed":
            if source is None:
                errors.append(
                    f"source-removed tombstone {record_id} has no source identity"
                )
                continue
            snapshot = tombstone.get("graph_node")
            assert isinstance(snapshot, Mapping)
            prior = tombstone.get("prior_runtime_record")
            prior_excluded = (
                isinstance(prior, Mapping)
                and prior.get("origin") == "source_revised"
                and prior.get("mask_policy") == "excluded"
            )
            for field in ("category", "kind", "support"):
                if snapshot.get(field) != source.get(field):
                    errors.append(
                        f"source-removed tombstone {record_id} changes immutable "
                        f"{field}: source={source.get(field)!r}, "
                        f"snapshot={snapshot.get(field)!r}"
                    )
            if prior_excluded:
                if snapshot.get("mask_path") is not None or prior.get(
                    "source_mask_path"
                ) != source.get("mask_path"):
                    errors.append(
                        f"source-removed tombstone {record_id} has invalid excluded "
                        "source-mask provenance"
                    )
            elif snapshot.get("mask_path") != source.get("mask_path"):
                errors.append(
                    f"source-removed tombstone {record_id} changes immutable "
                    f"mask_path: source={source.get('mask_path')!r}, "
                    f"snapshot={snapshot.get('mask_path')!r}"
                )
            inventory.pop(record_id, None)
        elif record_id in source_inventory:
            errors.append(
                f"runtime-removed tombstone {record_id} collides with source inventory"
            )
    for record_id, record in _runtime_records(runtime_inventory).items():
        graph_node = record["graph_node"]
        placement = record["placement"]
        assert isinstance(graph_node, Mapping)
        assert isinstance(placement, Mapping)
        origin = record.get("origin")
        if origin == "runtime_added" and record_id in inventory:
            errors.append(
                f"runtime object {record_id} collides with retained source inventory"
            )
            continue
        if origin == "source_revised" and record_id not in inventory:
            errors.append(
                f"source-revised object {record_id} has no retained source identity"
            )
            continue
        resolved = {
            "id": record_id,
            "category": record.get("category"),
            "instance": record.get("instance"),
            "kind": "object",
            "support": graph_node.get("support"),
            "mask_path": record.get("mask_path"),
            "mesh_name": placement.get("mesh_name"),
            "mesh_glb": placement.get("mesh_glb"),
            "runtime_transaction_id": record.get("transaction_id"),
            "runtime_origin": origin,
        }
        if origin == "runtime_added":
            resolved["source_section"] = "runtime_objects"
            inventory[record_id] = resolved
        else:
            source = inventory[record_id]
            for field in ("category", "instance", "kind", "support"):
                if source.get(field) != resolved.get(field):
                    errors.append(
                        f"source-revised object {record_id} changes immutable {field}: "
                        f"source={source.get(field)!r}, overlay={resolved.get(field)!r}"
                    )
            mask_policy = record.get("mask_policy", "tracked")
            if mask_policy == "tracked" and source.get("mask_path") != resolved.get(
                "mask_path"
            ):
                errors.append(
                    f"source-revised object {record_id} changes immutable mask_path: "
                    f"source={source.get('mask_path')!r}, "
                    f"overlay={resolved.get('mask_path')!r}"
                )
            if mask_policy == "excluded" and record.get(
                "source_mask_path"
            ) != source.get("mask_path"):
                errors.append(
                    f"source-revised object {record_id} has wrong source mask "
                    f"provenance: source={source.get('mask_path')!r}, "
                    f"overlay={record.get('source_mask_path')!r}"
                )
            source.update(
                mesh_name=resolved["mesh_name"],
                mesh_glb=resolved["mesh_glb"],
                mask_path=resolved["mask_path"],
                mask_policy=mask_policy,
                runtime_transaction_id=resolved["runtime_transaction_id"],
                runtime_origin=origin,
            )
    _fail("resolved masks/runtime inventory", errors)
    return inventory


def validate_runtime_source_masks(
    runtime_inventory: Mapping[str, Any] | None,
    masks_path: str | Path,
) -> None:
    """Require the source masks a runtime overlay refers to."""

    if runtime_inventory is None:
        return
    _runtime_records(runtime_inventory)
    path = Path(masks_path)
    if not path.is_file():
        _fail("runtime overlay -> source masks", [f"source masks are missing: {path}"])


def _indexed_graph_nodes(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    errors: list[str] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _fail("scene graph", ["nodes must be a list"])
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(nodes):
        if not isinstance(raw, dict):
            errors.append(f"graph.nodes[{index}] is not an object")
            continue
        node_id = raw.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"graph.nodes[{index}] has no nonempty string id")
            continue
        if node_id in indexed:
            errors.append(f"duplicate graph node id {node_id!r}")
            continue
        indexed[node_id] = raw
    _fail("scene graph", errors)
    return indexed


def _validate_relationships(
    masks: Mapping[str, Any],
    graph: Mapping[str, Any],
    node_ids: set[str],
    removed_node_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    mask_relationships = masks.get("relationships", [])
    graph_relationships = graph.get("relationships", [])
    if not isinstance(mask_relationships, list):
        return ["masks.relationships must be a list"]
    if not isinstance(graph_relationships, list):
        return ["graph.relationships must be a list"]

    removed = removed_node_ids or set()
    expected_source_relationships = [
        row
        for row in mask_relationships
        if not (
            isinstance(row, Mapping)
            and removed & {row.get("a"), row.get("b")}
        )
    ]
    source_relationships = [
        row
        for row in graph_relationships
        if not isinstance(row, dict) or row.get("runtime_added") is not True
    ]
    if source_relationships != expected_source_relationships:
        errors.append(
            "source relationships differ from final masks.relationships "
            f"(active masks={len(expected_source_relationships)}, "
            f"graph={len(source_relationships)})"
        )

    for index, relationship in enumerate(graph_relationships):
        if not isinstance(relationship, dict):
            errors.append(f"graph.relationships[{index}] is not an object")
            continue
        enforcement = str(relationship.get("enforcement") or "").strip().lower()
        if enforcement not in {"hard", "advisory"}:
            continue
        relationship_id = relationship.get("relationship_id") or f"index {index}"
        for endpoint in ("a", "b"):
            value = relationship.get(endpoint)
            if value not in node_ids:
                errors.append(
                    f"active relationship {relationship_id!r} ({enforcement}) has "
                    f"missing endpoint {endpoint}={value!r}"
                )
    return errors


def validate_scene_graph_inventory(
    masks: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    allow_runtime_additions: bool = True,
    runtime_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ""

    expected = authoritative_inventory(masks)
    runtime_records = _runtime_records(runtime_inventory)
    runtime_tombstones = _runtime_tombstones(runtime_inventory)
    # Validate source/runtime identity disjointness even if the graph happens to hide
    # the colliding runtime record.
    resolved_inventory(masks, runtime_inventory)
    nodes = _indexed_graph_nodes(graph)
    errors: list[str] = []

    source_ids = {
        node_id
        for node_id, node in nodes.items()
        if node.get("runtime_added") is not True
    }
    runtime_ids = set(nodes) - source_ids
    runtime_record_ids = set(runtime_records)
    runtime_added_record_ids = {
        record_id
        for record_id, record in runtime_records.items()
        if record.get("origin") == "runtime_added"
    }
    runtime_revised_record_ids = runtime_record_ids - runtime_added_record_ids
    removed_source_ids = {
        record_id
        for record_id, tombstone in runtime_tombstones.items()
        if tombstone.get("origin") == "source_removed"
    }
    expected_ids = set(expected) - removed_source_ids
    tombstone_ids = set(runtime_tombstones)
    present_tombstones = sorted(tombstone_ids & set(nodes))
    if present_tombstones:
        errors.append(
            "tombstoned objects remain in scene graph: "
            + ", ".join(present_tombstones)
        )
    missing = sorted(expected_ids - source_ids)
    unexpected = sorted(source_ids - expected_ids)
    if missing:
        errors.append(f"missing graph nodes: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected source graph nodes: {', '.join(unexpected)}")
    if runtime_ids and not allow_runtime_additions:
        errors.append(
            "runtime-added graph nodes are not allowed at this boundary: "
            + ", ".join(sorted(runtime_ids))
        )
    missing_runtime_nodes = sorted(runtime_added_record_ids - runtime_ids)
    if missing_runtime_nodes:
        errors.append(
            "committed runtime objects missing graph nodes: "
            + ", ".join(missing_runtime_nodes)
        )

    for node_id in sorted(expected_ids & source_ids):
        wanted = expected[node_id]
        actual = nodes[node_id]
        revised = runtime_records.get(node_id)
        compared_fields = ["category", "kind", "support"]
        if revised is None or revised.get("mask_policy", "tracked") == "tracked":
            compared_fields.append("mask_path")
        for field in compared_fields:
            if actual.get(field) != wanted.get(field):
                errors.append(
                    f"{field} mismatch for {node_id}: "
                    f"masks={wanted.get(field)!r}, graph={actual.get(field)!r}"
                )
        if revised is not None:
            if revised.get("origin") != "source_revised":
                errors.append(
                    f"source identity {node_id} has non-source overlay origin "
                    f"{revised.get('origin')!r}"
                )
            snapshot = revised.get("graph_node")
            assert isinstance(snapshot, Mapping)
            for field in (
                "id",
                "category",
                "kind",
                "support",
                "parent",
                "mask_path",
                "runtime_added",
                "runtime_transaction_id",
            ):
                if actual.get(field) != snapshot.get(field):
                    errors.append(
                        f"source-revised graph snapshot mismatch for {node_id}.{field}: "
                        f"overlay={snapshot.get(field)!r}, graph={actual.get(field)!r}"
                    )

    for node_id in sorted(runtime_ids):
        node = nodes[node_id]
        if node_id in expected_ids:
            errors.append(
                f"source identity {node_id} is incorrectly marked runtime_added"
            )
        kind = node.get("kind")
        if kind == "root_surface":
            if node_id in runtime_records:
                errors.append(
                    f"runtime overlay object {node_id} materialized as a root surface"
                )
            if node.get("parent") is not None:
                errors.append(f"runtime-added root {node_id} must be parentless")
            continue
        if kind != "object":
            errors.append(
                f"runtime-added node {node_id} has kind={node.get('kind')!r}; "
                "expected a root surface or committed overlay object"
            )
            continue
        record = runtime_records.get(node_id)
        if record is None:
            errors.append(
                f"runtime-added object {node_id} has no committed runtime overlay record"
            )
            continue
        if record.get("origin") != "runtime_added":
            errors.append(
                f"runtime-added graph object {node_id} has overlay origin "
                f"{record.get('origin')!r}"
            )
        snapshot = record.get("graph_node")
        assert isinstance(snapshot, Mapping)
        for field in (
            "id",
            "category",
            "kind",
            "support",
            "parent",
            "mask_path",
            "runtime_added",
            "runtime_transaction_id",
        ):
            if node.get(field) != snapshot.get(field):
                errors.append(
                    f"runtime graph snapshot mismatch for {node_id}.{field}: "
                    f"overlay={snapshot.get(field)!r}, graph={node.get(field)!r}"
                )

    from lib.tools.geometry.surface_relations import surface_build_name

    root_build_names: dict[str, str] = {}
    build_name_owner: dict[str, str] = {}
    for node_id, node in nodes.items():
        if node.get("kind") != "root_surface":
            continue
        raw_name = node.get("build_name")
        name = (
            str(raw_name).strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else surface_build_name(node_id)
        )
        owner = build_name_owner.get(name)
        if owner is not None:
            errors.append(
                f"root build name {name!r} is shared by graph ids {owner!r} "
                f"and {node_id!r}"
            )
        else:
            build_name_owner[name] = node_id
        root_build_names[node_id] = name

    # One hierarchy: support == parent, children is its exact inverse, roots are
    # root-surface identities only.  Geometry fields are intentionally irrelevant.
    expected_children = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        kind = node.get("kind")
        if kind not in {"object", "root_surface"}:
            errors.append(f"graph node {node_id} has invalid kind {kind!r}")
            continue
        parent = node.get("parent")
        support = node.get("support")
        if kind == "root_surface":
            # Root ``support`` is retained semantic metadata (for example a wall or
            # furniture root may declare floor#0). It is compared to masks above but
            # does not turn one root surface into another's topology child.
            if parent is not None:
                errors.append(f"root surface {node_id} must be parentless")
        else:
            if not isinstance(parent, str) or not parent:
                errors.append(f"object {node_id} has no exact parent id")
                continue
            if support != parent:
                errors.append(
                    f"object {node_id} has parent/support mismatch: "
                    f"parent={parent!r}, support={support!r}"
                )
            if parent not in nodes:
                errors.append(f"object {node_id} has dangling parent {parent!r}")
            elif parent == node_id:
                errors.append(f"object {node_id} cannot parent itself")
            else:
                expected_children[parent].append(node_id)

    for node_id, node in nodes.items():
        children = node.get("children")
        if not isinstance(children, list):
            errors.append(f"graph node {node_id} children must be a list")
            continue
        normalized = [str(child) for child in children]
        if len(normalized) != len(set(normalized)):
            errors.append(f"graph node {node_id} has duplicate children")
        expected_for_node = sorted(expected_children[node_id])
        if sorted(normalized) != expected_for_node:
            errors.append(
                f"children mismatch for {node_id}: "
                f"expected={expected_for_node!r}, graph={sorted(normalized)!r}"
            )

    # Detect cycles independently of the roots list so a closed object cycle cannot
    # evade the parentless-object check.
    for start, node in nodes.items():
        if node.get("kind") == "root_surface":
            continue
        seen: list[str] = []
        current = start
        while current in nodes and nodes[current].get("kind") != "root_surface":
            if current in seen:
                cycle = seen[seen.index(current) :] + [current]
                errors.append("support cycle: " + " -> ".join(cycle))
                break
            seen.append(current)
            parent = nodes[current].get("parent")
            if not isinstance(parent, str):
                break
            current = parent

    roots = graph.get("roots")
    if not isinstance(roots, list):
        errors.append("graph.roots must be a list")
    else:
        root_values = [str(root) for root in roots]
        if len(root_values) != len(set(root_values)):
            errors.append("graph.roots contains duplicate ids")
        expected_roots = sorted(
            node_id
            for node_id, node in nodes.items()
            if node.get("kind") == "root_surface"
        )
        if sorted(root_values) != expected_roots:
            errors.append(
                f"graph.roots mismatch: expected={expected_roots!r}, "
                f"graph={sorted(root_values)!r}"
            )

    errors.extend(
        _validate_relationships(
            masks, graph, set(nodes), removed_node_ids=tombstone_ids
        )
    )
    _fail("masks -> scene graph", errors)
    return {
        "source_node_ids": sorted(expected_ids),
        "tombstoned_object_ids": sorted(tombstone_ids),
        "removed_source_node_ids": sorted(removed_source_ids),
        "runtime_root_ids": sorted(
            node_id
            for node_id in runtime_ids
            if nodes[node_id].get("kind") == "root_surface"
        ),
        "runtime_object_ids": sorted(runtime_record_ids),
        "runtime_added_object_ids": sorted(runtime_added_record_ids),
        "runtime_revised_object_ids": sorted(runtime_revised_record_ids),
        "resolved_node_ids": sorted(set(nodes)),
        "root_ids": sorted(
            node_id
            for node_id, node in nodes.items()
            if node.get("kind") == "root_surface"
        ),
        "root_build_names": {
            node_id: root_build_names[node_id] for node_id in sorted(root_build_names)
        },
        "object_ids": sorted(
            node_id
            for node_id, node in nodes.items()
            if node.get("kind") != "root_surface"
        ),
    }


def _resolve_artifact_path(value: str, artifact_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    # Placement historically stores both scene-relative paths (``meshes/x.glb``)
    # and repository-relative paths (``output/<run>/<scene>/meshes/x.glb``). Honor
    # both existing contracts deterministically rather than blindly prefixing the
    # scene root and producing ``<scene>/output/<run>/...``.
    candidates = []
    if artifact_root is not None:
        candidates.append((artifact_root / path).resolve())
    project_root = Path(__file__).resolve().parents[3]
    candidates.extend(((project_root / path).resolve(), path.resolve()))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else path.resolve()


def validate_placement_inventory(
    graph: Mapping[str, Any],
    placement: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate exact graph-object identity before expensive mesh reconstruction."""

    nodes = _indexed_graph_nodes(graph)
    graph_objects = {
        node_id: node
        for node_id, node in nodes.items()
        if node.get("kind") != "root_surface"
    }
    rows: Any = (
        placement.get("objects") if isinstance(placement, Mapping) else placement
    )
    if not isinstance(rows, (list, tuple)):
        _fail("scene graph -> placement", ["placement.objects must be a list"])
    errors: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        origin = f"placement.objects[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{origin} is not an object")
            continue
        object_id, id_errors = _record_id(raw, origin)
        errors.extend(id_errors)
        if object_id is None:
            continue
        if object_id in by_id:
            errors.append(f"duplicate placement row for {object_id}")
            continue
        by_id[object_id] = raw
        graph_node = graph_objects.get(object_id)
        if graph_node is not None and raw.get("category") != graph_node.get("category"):
            errors.append(
                f"category mismatch for {object_id}: "
                f"graph={graph_node.get('category')!r}, "
                f"placement={raw.get('category')!r}"
            )

    expected_ids = set(graph_objects)
    placement_ids = set(by_id)
    missing = sorted(expected_ids - placement_ids)
    unexpected = sorted(placement_ids - expected_ids)
    if missing:
        errors.append(f"missing placement rows: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected placement rows: {', '.join(unexpected)}")
    _fail("scene graph -> placement", errors)
    return by_id


def validate_retained_mask_artifacts(
    masks: Mapping[str, Any], artifact_root: Path
) -> None:
    """Distinguish intentional maskless roots from broken declared masks."""

    inventory = authoritative_inventory(masks)
    errors: list[str] = []
    for record_id, record in inventory.items():
        value = record.get("mask_path")
        if record.get("source_section") == "unmasked_root_surfaces":
            if value not in (None, ""):
                errors.append(
                    f"intentional unmasked root {record_id} unexpectedly declares "
                    f"mask_path={value!r}"
                )
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"retained masked instance {record_id} has no mask_path; intentional "
                "maskless roots must be in unmasked_root_surfaces"
            )
            continue
        path = _resolve_artifact_path(value, artifact_root)
        if not path.is_file():
            errors.append(f"retained instance {record_id} mask is missing: {path}")
            continue
        if path.stat().st_size <= 0:
            errors.append(f"retained instance {record_id} mask is empty: {path}")
            continue
        try:
            import numpy as np

            mask = np.load(path, mmap_mode="r", allow_pickle=False)
            if mask.ndim != 2 or mask.size == 0:
                errors.append(
                    f"retained instance {record_id} mask has invalid shape "
                    f"{tuple(mask.shape)!r}: {path}"
                )
        except Exception as exc:  # noqa: BLE001 - artifact error is the diagnostic
            errors.append(
                f"retained instance {record_id} mask is unreadable/corrupt: "
                f"{path} ({type(exc).__name__}: {exc})"
            )
    _fail("retained mask artifacts", errors)


def _require_runtime_owned_path(
    value: Any,
    *,
    record_id: str,
    field: str,
    artifact_root: Path,
    errors: list[str],
    require_runtime_owned: bool | None = True,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"runtime object {record_id} has no nonempty {field}")
        return None
    path = _resolve_artifact_path(value, artifact_root)
    if require_runtime_owned is None:
        return path
    required_root = (
        (artifact_root / "runtime_objects").resolve()
        if require_runtime_owned
        else artifact_root.resolve()
    )
    try:
        path.relative_to(required_root)
    except ValueError:
        if require_runtime_owned:
            errors.append(
                f"runtime object {record_id} {field} is outside backend-owned "
                f"runtime_objects/: {path}"
            )
        else:
            errors.append(
                f"runtime object {record_id} {field} is outside the scene: {path}"
            )
        return None
    return path


def validate_runtime_object_artifacts(
    runtime_inventory: Mapping[str, Any] | None,
    artifact_root: str | Path,
) -> None:
    """Validate that an overlay's masks and current mesh revision exist and load."""

    if runtime_inventory is None:
        return
    records = _runtime_records(runtime_inventory)
    root = Path(artifact_root).resolve()
    errors: list[str] = []
    for record_id, record in records.items():
        mask_fields: list[tuple[str, Any]] = []
        if record.get("mask_policy", "tracked") == "tracked":
            mask_fields.append(("mask_path", record.get("mask_path")))
            scoring_mask = record.get("scoring_mask_path", record.get("mask_path"))
            if scoring_mask != record.get("mask_path"):
                mask_fields.append(("scoring_mask_path", scoring_mask))
        elif record.get("origin") == "source_revised":
            # The source mask is provenance only (never scoring input) after a direct
            # reconstruction.
            mask_fields.append(("source_mask_path", record.get("source_mask_path")))
        for field, value in mask_fields:
            mask_path = _require_runtime_owned_path(
                value,
                record_id=record_id,
                field=field,
                artifact_root=root,
                errors=errors,
                require_runtime_owned=(
                    True
                    if field == "scoring_mask_path"
                    or record.get("origin") == "runtime_added"
                    else None
                ),
            )
            if mask_path is None:
                continue
            if not mask_path.is_file():
                errors.append(
                    f"runtime object {record_id} {field} is missing: {mask_path}"
                )
            elif mask_path.stat().st_size <= 0:
                errors.append(
                    f"runtime object {record_id} {field} is empty: {mask_path}"
                )
            else:
                try:
                    import numpy as np

                    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                    if mask.ndim != 2 or mask.size == 0:
                        errors.append(
                            f"runtime object {record_id} {field} has invalid shape "
                            f"{tuple(mask.shape)!r}: {mask_path}"
                        )
                except Exception as exc:  # noqa: BLE001 - artifact diagnostic
                    errors.append(
                        f"runtime object {record_id} {field} is unreadable/corrupt: "
                        f"{mask_path} ({type(exc).__name__}: {exc})"
                    )

        revisions = record["mesh_revisions"]
        assert isinstance(revisions, list)
        current_number = record["current_mesh_revision"]
        current = next(
            revision
            for revision in revisions
            if isinstance(revision, Mapping)
            and revision.get("revision") == current_number
        )
        mesh_path = _require_runtime_owned_path(
            current.get("mesh_glb"),
            record_id=record_id,
            field="current mesh_glb",
            artifact_root=root,
            errors=errors,
        )
        if mesh_path is None:
            continue
        if mesh_path.suffix.lower() != ".glb":
            errors.append(
                f"runtime object {record_id} current mesh is not a GLB: {mesh_path}"
            )
            continue
        if not mesh_path.is_file():
            errors.append(
                f"runtime object {record_id} current mesh_glb is missing: {mesh_path}"
            )
            continue
        if mesh_path.stat().st_size <= 0:
            errors.append(
                f"runtime object {record_id} current mesh_glb is empty: {mesh_path}"
            )
    for record_id, tombstone in _runtime_tombstones(runtime_inventory).items():
        placement = tombstone.get("placement")
        assert isinstance(placement, Mapping)
        mesh_path = _require_runtime_owned_path(
            placement.get("mesh_glb"),
            record_id=record_id,
            field="tombstone mesh_glb",
            artifact_root=root,
            errors=errors,
            require_runtime_owned=None,
        )
        if mesh_path is None:
            continue
        if mesh_path.suffix.lower() != ".glb":
            errors.append(
                f"runtime object {record_id} tombstone mesh is not a GLB: {mesh_path}"
            )
        elif not mesh_path.is_file():
            errors.append(
                f"runtime object {record_id} tombstone mesh is missing: {mesh_path}"
            )
        elif mesh_path.stat().st_size <= 0:
            errors.append(
                f"runtime object {record_id} tombstone mesh is empty: {mesh_path}"
            )
    _fail("runtime object artifacts", errors)


def validate_runtime_object_bindings(
    runtime_inventory: Mapping[str, Any] | None,
    graph: Mapping[str, Any],
    placement: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    """Validate overlay snapshots against the current graph and placement rows."""

    if runtime_inventory is None:
        return
    records = _runtime_records(runtime_inventory)
    tombstones = _runtime_tombstones(runtime_inventory)
    nodes = _indexed_graph_nodes(graph)
    rows = validate_placement_inventory(graph, placement)
    errors: list[str] = []
    for record_id, record in records.items():
        node = nodes.get(record_id)
        if node is None:
            errors.append(f"runtime object {record_id} has no graph node")
        else:
            snapshot = record["graph_node"]
            assert isinstance(snapshot, Mapping)
            for field in (
                "id",
                "category",
                "kind",
                "support",
                "parent",
                "mask_path",
                "runtime_added",
                "runtime_transaction_id",
            ):
                if node.get(field) != snapshot.get(field):
                    errors.append(
                        f"runtime object {record_id} graph {field} differs from "
                        "overlay snapshot"
                    )

        row = rows.get(record_id)
        if row is None:
            errors.append(f"runtime object {record_id} has no placement row")
            continue
        snapshot = record["placement"]
        assert isinstance(snapshot, Mapping)
        for field in (
            "category",
            "instance",
            "mesh_name",
            "mesh_glb",
            "runtime_added",
            "runtime_transaction_id",
        ):
            if row.get(field) != snapshot.get(field):
                errors.append(
                    f"runtime object {record_id} placement {field} differs from "
                    "overlay snapshot"
                )
        if dict(row) != dict(snapshot):
            differing = sorted(
                key
                for key in set(row) | set(snapshot)
                if row.get(key) != snapshot.get(key)
            )
            errors.append(
                f"runtime object {record_id} full placement snapshot is stale; "
                f"differing fields: {', '.join(differing)}"
            )
    for record_id in tombstones:
        if record_id in nodes:
            errors.append(f"tombstoned object {record_id} still has a graph node")
        if record_id in rows:
            errors.append(f"tombstoned object {record_id} still has a placement row")
    _fail("runtime overlay -> graph/placement", errors)


def validate_runtime_physics_materials(
    runtime_inventory: Mapping[str, Any] | None,
    placement: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    artifact_root: str | Path,
) -> None:
    """Validate canonical USD-physics inputs and bind logged runtime materials.

    Runtime inventory is audit provenance; ``physics/physics_vlm.json`` is what the
    Isaac translator actually consumes.  The canonical placement/physics identity and
    JSON-shape checks apply even when no runtime overlay exists: a pose-only GPT-6 run
    must not be finalized with inputs that the downstream translator cannot consume.
    When a runtime overlay does exist, its logged material must additionally agree with
    that canonical physics record.
    """

    records = _runtime_records(runtime_inventory)
    root = Path(artifact_root).resolve()
    estimates_path = root / "physics" / "physics_vlm.json"
    manifest_path = root / "physics" / "physics_estimate_manifest.json"
    pose_changes_path = root / "physics" / "pose_changes.json"
    blend_base_path = root / "physics" / "blend_base.json"
    errors: list[str] = []

    def load_mapping(path: Path, label: str) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{label} is unreadable: {path} ({exc})")
            return {}
        if not isinstance(payload, Mapping):
            errors.append(f"{label} is not a JSON object: {path}")
            return {}
        return payload

    estimates = load_mapping(estimates_path, "runtime physics estimates")
    manifest = load_mapping(manifest_path, "physics estimate manifest")
    pose_changes = load_mapping(pose_changes_path, "physics pose changes")
    blend_base = load_mapping(blend_base_path, "physics Blend base")
    manifest_records = manifest.get("objects") if isinstance(manifest, Mapping) else {}
    if not isinstance(manifest_records, Mapping):
        errors.append("physics estimate manifest objects is not a mapping")
        manifest_records = {}

    rows = placement.get("objects") if isinstance(placement, Mapping) else placement
    if not isinstance(rows, (list, tuple)):
        errors.append("placement objects is not a list")
        rows = []
    expected_names = {
        str(row.get("mesh_name"))
        for row in rows
        if isinstance(row, Mapping) and row.get("mesh_name")
    }
    if set(manifest_records) != expected_names:
        missing = sorted(expected_names - set(manifest_records))
        extra = sorted(set(manifest_records) - expected_names)
        if missing:
            errors.append(
                "physics estimate manifest is missing placement objects: "
                + ", ".join(missing)
            )
        if extra:
            errors.append(
                "physics estimate manifest has stale objects: " + ", ".join(extra)
            )
    if manifest.get("expected_count") != len(expected_names):
        errors.append(
            "physics estimate manifest expected_count does not match placement "
            f"inventory ({manifest.get('expected_count')!r} != {len(expected_names)})"
        )

    stale_estimates = sorted(set(estimates) - expected_names)
    if stale_estimates:
        errors.append(
            "runtime physics estimates contain stale placement objects: "
            + ", ".join(stale_estimates)
        )
    for mesh_name in sorted(expected_names):
        manifest_record = manifest_records.get(mesh_name)
        if not isinstance(manifest_record, Mapping):
            errors.append(
                f"physics estimate manifest record for {mesh_name!r} is not an object"
            )
            continue
        manifest_status = manifest_record.get("status")
        if manifest_status not in {
            "estimated",
            "cached",
            "failed_fallback",
            "disabled_fallback",
        }:
            errors.append(
                f"physics estimate manifest record for {mesh_name!r} has invalid "
                f"status {manifest_status!r}"
            )
        estimate = estimates.get(mesh_name)
        if manifest_status in {"estimated", "cached"} and not isinstance(
            estimate, Mapping
        ):
            errors.append(
                "physics estimate manifest marks "
                f"{mesh_name} as {manifest_status} but physics_vlm "
                "has no canonical object entry"
            )
        elif mesh_name in estimates and not isinstance(estimate, Mapping):
            errors.append(f"physics_vlm[{mesh_name!r}] is not a canonical object entry")
        if not isinstance(estimate, Mapping):
            continue

        for field in ("material", "mass_source", "density_gate"):
            value = estimate.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"physics_vlm[{mesh_name!r}] has no nonempty {field}")
        mass = estimate.get("mass_kg")
        if (
            not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not math.isfinite(float(mass))
            or float(mass) <= 0.0
        ):
            errors.append(f"physics_vlm[{mesh_name!r}] has no positive finite mass_kg")
        mass_range = estimate.get("mass_range_kg")
        valid_mass_range = (
            isinstance(mass_range, (list, tuple))
            and len(mass_range) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in mass_range
            )
        )
        if not valid_mass_range:
            errors.append(
                f"physics_vlm[{mesh_name!r}] has no positive finite mass_range_kg"
            )
        elif (
            float(mass_range[0]) > float(mass_range[1])
            or not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not float(mass_range[0]) <= float(mass) <= float(mass_range[1])
        ):
            errors.append(
                f"physics_vlm[{mesh_name!r}] mass_range_kg does not contain mass_kg"
            )
        friction = estimate.get("friction")
        if (
            not isinstance(friction, (int, float))
            or isinstance(friction, bool)
            or not math.isfinite(float(friction))
            or float(friction) < 0.0
        ):
            errors.append(
                f"physics_vlm[{mesh_name!r}] has no nonnegative finite friction"
            )
        extents = estimate.get("est_obb_extents_m")
        if extents is not None and (
            not isinstance(extents, (list, tuple))
            or len(extents) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in extents
            )
        ):
            errors.append(f"physics_vlm[{mesh_name!r}] has invalid est_obb_extents_m")
        volume = estimate.get("est_volume_m3")
        if volume is not None and (
            not isinstance(volume, (int, float))
            or isinstance(volume, bool)
            or not math.isfinite(float(volume))
            or float(volume) <= 0.0
        ):
            errors.append(f"physics_vlm[{mesh_name!r}] has invalid est_volume_m3")

    pose_objects = pose_changes.get("objects")
    if not isinstance(pose_objects, Mapping):
        errors.append("physics pose changes objects is not a mapping")
        pose_objects = {}
    pose_names = set(pose_objects)
    # pose_changes contains preprocess ladder history, not a required row for every
    # current identity. Runtime additions have no such history. Unknown/stale rows are
    # unsafe because blend_to_isaac would try to seed overrides for the wrong identity.
    extra = sorted(pose_names - expected_names)
    if extra:
        errors.append(
            "physics pose changes have stale objects: " + ", ".join(extra)
        )
    for mesh_name, pose_record in pose_objects.items():
        if not isinstance(pose_record, Mapping):
            errors.append(f"pose_changes.objects[{mesh_name!r}] is not an object")
            continue
        overrides = pose_record.get("physics_overrides")
        if overrides is not None and not isinstance(overrides, Mapping):
            errors.append(
                f"pose_changes.objects[{mesh_name!r}].physics_overrides is not an object"
            )

    base_names = set(blend_base)
    if base_names != expected_names:
        missing = sorted(expected_names - base_names)
        extra = sorted(base_names - expected_names)
        if missing:
            errors.append(
                "physics Blend base is missing placement objects: " + ", ".join(missing)
            )
        if extra:
            errors.append("physics Blend base has stale objects: " + ", ".join(extra))
    for mesh_name, matrix in blend_base.items():
        valid_matrix = (
            isinstance(matrix, (list, tuple))
            and len(matrix) == 4
            and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in matrix)
        )
        if valid_matrix:
            values = [value for row in matrix for value in row]
            valid_matrix = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            )
        if not valid_matrix:
            errors.append(
                f"physics Blend base matrix for {mesh_name!r} is not finite 4x4"
            )

    for record_id, record in records.items():
        snapshot = record.get("placement")
        logged = record.get("physical_material")
        assert isinstance(snapshot, Mapping)
        assert isinstance(logged, Mapping)
        mesh_name = str(snapshot.get("mesh_name") or "")
        canonical = estimates.get(mesh_name)
        if not isinstance(canonical, Mapping):
            errors.append(
                f"runtime object {record_id} has no canonical physics_vlm entry "
                f"for {mesh_name!r}"
            )
            continue
        for field in (
            "material",
            "mass_kg",
            "mass_range_kg",
            "friction",
            "density_kgm3",
            "density_gate",
        ):
            if logged.get(field) != canonical.get(field):
                errors.append(
                    f"runtime object {record_id} physical_material.{field} differs "
                    f"from physics_vlm[{mesh_name!r}]"
                )
        manifest_record = manifest_records.get(mesh_name)
        if not isinstance(manifest_record, Mapping):
            continue
        allowed_statuses = (
            {"estimated", "cached"}
            if logged.get("status") == "estimated"
            else {"failed_fallback"}
        )
        if manifest_record.get("status") not in allowed_statuses:
            errors.append(
                f"runtime object {record_id} material status {logged.get('status')!r} "
                "does not match physics estimate manifest status "
                f"{manifest_record.get('status')!r}"
            )
    _fail("runtime material -> USD physics", errors)


def validate_object_materialization(
    graph: Mapping[str, Any],
    placement: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, str]:
    """Validate graph objects -> placement rows -> nonempty mesh artifacts.

    Returns ``{scene_graph_id: mesh_name}``, which downstream Blender gates use as
    their exact expected imported-object inventory.
    """

    nodes = _indexed_graph_nodes(graph)
    graph_objects = {
        node_id: node
        for node_id, node in nodes.items()
        if node.get("kind") != "root_surface"
    }
    rows: Any = (
        placement.get("objects") if isinstance(placement, Mapping) else placement
    )
    by_id = validate_placement_inventory(graph, placement)
    assert isinstance(rows, (list, tuple))
    root = Path(artifact_root).resolve() if artifact_root is not None else None
    errors: list[str] = []
    mesh_name_to_id: dict[str, str] = {}
    for index, raw in enumerate(rows):
        origin = f"placement.objects[{index}]"
        assert isinstance(raw, Mapping)
        object_id, id_errors = _record_id(raw, origin)
        assert not id_errors
        if object_id is None:
            continue

        mesh_name = raw.get("mesh_name")
        if not isinstance(mesh_name, str) or not mesh_name.strip():
            errors.append(f"placement {object_id} has no nonempty mesh_name")
        elif mesh_name in mesh_name_to_id:
            errors.append(
                f"mesh_name {mesh_name!r} is shared by {mesh_name_to_id[mesh_name]} "
                f"and {object_id}"
            )
        else:
            mesh_name_to_id[mesh_name] = object_id

        mesh_glb = raw.get("mesh_glb")
        if not isinstance(mesh_glb, str) or not mesh_glb.strip():
            errors.append(f"placement {object_id} has no mesh_glb")
        else:
            path = _resolve_artifact_path(mesh_glb, root)
            if not path.is_file():
                errors.append(f"placement {object_id} mesh_glb is missing: {path}")
            elif path.stat().st_size <= 0:
                errors.append(f"placement {object_id} mesh_glb is empty: {path}")

    _fail("scene graph -> placement/mesh", errors)
    return {
        object_id: str(by_id[object_id]["mesh_name"])
        for object_id in sorted(graph_objects)
    }


def validate_blender_object_names(
    expected_by_id: Mapping[str, str], actual_pipeline_names: Iterable[str]
) -> None:
    ""

    expected_names = set(expected_by_id.values())
    actual_names = {str(name) for name in actual_pipeline_names}
    errors: list[str] = []
    missing = sorted(expected_names - actual_names)

    def is_member(name: str) -> bool:
        return any(
            name == root or re.fullmatch(re.escape(root) + r"_[1-9][0-9]*", name)
            for root in expected_names
        )

    unexpected = sorted(name for name in actual_names if not is_member(name))
    if missing:
        errors.append(f"missing Blender objects: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected Blender obj_* objects: {', '.join(unexpected)}")
    _fail("placement -> Blender", errors)


def validate_scene_artifacts(
    scene_dir: str | Path,
    *,
    allow_runtime_additions: bool = True,
    allow_runtime_inventory: bool | None = None,
    validate_runtime_physics: bool = False,
) -> dict[str, Any]:
    ""

    if allow_runtime_inventory is None:
        allow_runtime_inventory = validate_runtime_physics

    scene = Path(scene_dir)
    paths = {
        "masks": scene / "masks" / "masks.json",
        "graph": scene / "scene_graph.json",
        "placement": scene / "placement.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    _fail(
        "scene artifacts", [f"required artifact is missing: {path}" for path in missing]
    )
    try:
        masks = json.loads(paths["masks"].read_text())
        graph = json.loads(paths["graph"].read_text())
        placement = json.loads(paths["placement"].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryContractError(
            f"scene artifact JSON is unreadable: {exc}"
        ) from exc
    runtime_inventory_file = scene / "runtime_objects" / "inventory.json"
    if not allow_runtime_inventory and runtime_inventory_file.exists():
        raise InventoryContractError(
            "runtime object inventory is not allowed at this pipeline boundary: "
            f"{runtime_inventory_file}"
        )
    try:
        runtime_inventory = (
            load_runtime_inventory(scene) if allow_runtime_inventory else None
        )
    except RuntimeObjectRepairError as exc:
        raise InventoryContractError(str(exc)) from exc
    validate_retained_mask_artifacts(masks, scene.resolve())
    validate_runtime_source_masks(runtime_inventory, paths["masks"])
    graph_report = validate_scene_graph_inventory(
        masks,
        graph,
        allow_runtime_additions=allow_runtime_additions,
        runtime_inventory=runtime_inventory,
    )
    object_mesh_names = validate_object_materialization(
        graph, placement, artifact_root=scene
    )
    validate_runtime_object_bindings(runtime_inventory, graph, placement)
    validate_runtime_object_artifacts(runtime_inventory, scene)
    if validate_runtime_physics:
        validate_runtime_physics_materials(runtime_inventory, placement, scene)
    return {
        **graph_report,
        "runtime_inventory_present": runtime_inventory is not None,
        "object_mesh_names": object_mesh_names,
        "expected_blender_object_names": sorted(object_mesh_names.values()),
    }


__all__ = [
    "InventoryContractError",
    "authoritative_inventory",
    "resolved_inventory",
    "validate_blender_object_names",
    "validate_object_materialization",
    "validate_placement_inventory",
    "validate_retained_mask_artifacts",
    "validate_runtime_object_artifacts",
    "validate_runtime_object_bindings",
    "validate_runtime_physics_materials",
    "validate_runtime_source_masks",
    "validate_scene_artifacts",
    "validate_scene_graph_inventory",
]
