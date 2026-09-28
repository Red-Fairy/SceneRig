"""Declarations and observed deltas for code-authored initializer transactions.

This module does not execute Blender or simulate physics. The executor owns the
atomic scene/artifact transaction; the Blender wrapper enforces the declared tree
changes while permitting whole-object pose changes.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from lib.tools.geometry.register import mesh_name_for

# Relative spread of per-axis scale ratios above which a transform edit is a SHAPE
# edit. Blender's float32 scales perturb ratios at ~1e-7; real rescales in the
# 2026-09-14 benchmark were 5-150%.
ISOTROPIC_SCALE_TOLERANCE = 0.005


def anisotropic_scale_change(before_scale: Any, after_scale: Any) -> str | None:
    """Describe a non-uniform scale change between two integrity signatures.

    Returns ``None`` when either scale is absent/malformed or the change is uniform
    (including no change). Physics never changes scale, so this applies to every
    pre-existing object regardless of who moved it.
    """
    if not (
        isinstance(before_scale, (list, tuple))
        and isinstance(after_scale, (list, tuple))
        and len(before_scale) == 3
        and len(after_scale) == 3
    ):
        return None
    try:
        ratios = [float(a) / float(b) for a, b in zip(after_scale, before_scale)]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(r) for r in ratios):
        return None
    if any(r <= 0.0 for r in ratios):
        # audit F-M7: a reflection (negative scale) is neither a rigid pose nor a uniform
        # rescale — it used to pass as "uniform" because the check returned None here
        return "mirrored " + "x".join(f"{r:.3f}" for r in ratios)
    if (max(ratios) - min(ratios)) / max(ratios) <= ISOTROPIC_SCALE_TOLERANCE:
        return None
    return "x".join(f"{r:.3f}" for r in ratios)


def validate_object_addition_scope(
    *,
    graph: dict[str, Any],
    added_objects: list[dict[str, Any]],
    removed_objects: list[str] | None = None,
) -> None:
    """Require newly admitted objects to belong to the main-support subtree.

    Resolve the proposed final graph, including same-call support additions. This
    admission rule is independent of physical coverage against the immediate
    support. Existing off-main objects may be replaced on their unchanged support
    chain; their prior admission does not authorize new background objects.

    Raises:
        ValueError: If a proposed support is missing, cyclic, ambiguous, or outside
            the retained main-support scene.
    """
    if not added_objects:
        return
    original = {node["id"]: node for node in graph["nodes"]}
    main_id = graph.get("main_support_id")
    if main_id not in original:
        raise ValueError(f"Object admission needs a valid main_support_id: {main_id!r}")
    removed = set(removed_objects or [])
    final = {key: value for key, value in original.items() if key not in removed}
    for addition in added_objects:
        final[addition["object_id"]] = {
            "id": addition["object_id"],
            "kind": "object",
            "support": addition["support"],
            "parent": addition["support"],
        }

    def support_of(node: dict[str, Any]) -> str | None:
        support, parent = node.get("support"), node.get("parent")
        if support and parent and support != parent:
            raise ValueError(f"Conflicting support/parent for {node['id']!r}")
        return support or parent

    for addition in added_objects:
        object_id = addition["object_id"]
        current = object_id
        visited: set[str] = set()
        unchanged_chain = object_id in original and object_id in removed
        while current != main_id:
            if current in visited:
                raise ValueError(
                    f"Cyclic support chain for addition {object_id!r}: {current!r}"
                )
            visited.add(current)
            node = final.get(current)
            if node is None:
                raise ValueError(
                    f"Missing support {current!r} for addition {object_id!r}"
                )
            if node.get("kind") != "object":
                if unchanged_chain and current in original:
                    break
                raise ValueError(
                    f"Addition {object_id!r} is outside the main-support scene: "
                    f"its support chain reaches {current!r}, not {main_id!r}. "
                    "Add only missing objects on the main support or its object stacks; "
                    "background furniture and objects on other supports are excluded."
                )
            support = support_of(node)
            old = original.get(current)
            unchanged_chain = bool(
                unchanged_chain and old is not None and support_of(old) == support
            )
            if not isinstance(support, str) or not support:
                raise ValueError(
                    f"Missing support for {current!r} in addition {object_id!r}"
                )
            current = support


def normalize_object_declarations(
    added_objects: list[dict[str, Any]] | None,
    removed_objects: list[str] | None,
    *,
    graph: dict[str, Any],
    placement: dict[str, Any],
) -> dict[str, Any]:
    """Resolve exact graph identities and validate the proposed final inventory.

    An identity present in both lists denotes replacement. New object names are
    deterministic and are injected into agent code as ``ADDED_OBJECT_NAMES``.
    Root surfaces retain their separate registration tools.
    """
    if added_objects is not None and not isinstance(added_objects, list):
        raise ValueError("added_objects must be a list of object declarations")
    if removed_objects is not None and not isinstance(removed_objects, list):
        raise ValueError("removed_objects must be a list of exact object identities")
    nodes = {row["id"]: row for row in graph["nodes"]}
    rows = {
        f"{row['category']}#{int(row['instance'])}": row for row in placement["objects"]
    }
    aliases = {row["mesh_name"]: object_id for object_id, row in rows.items()}
    removed = []
    for value in removed_objects or []:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid removal identity: {value!r}")
        object_id = aliases.get(value, value)
        if object_id not in rows or nodes.get(object_id, {}).get("kind") != "object":
            raise ValueError(f"removal target is not a current object: {value!r}")
        if object_id in removed:
            raise ValueError(f"duplicate removal declaration: {object_id!r}")
        removed.append(object_id)

    added = []
    used_ids: set[str] = set()
    used_names: set[str] = set()
    for raw in added_objects or []:
        if not isinstance(raw, dict):
            raise ValueError(f"object declaration must be a mapping: {raw!r}")
        unknown = set(raw) - {
            "object_id",
            "description",
            "support",
            "physical_material_hint",
        }
        if unknown:
            raise ValueError(f"unknown object declaration fields: {sorted(unknown)!r}")
        object_id = raw.get("object_id")
        if not isinstance(object_id, str) or object_id.count("#") != 1:
            raise ValueError(f"object_id must be category#N: {object_id!r}")
        category, instance_text = object_id.split("#")
        if (
            not category
            or category != category.strip().lower()
            or len(category) > 80
            or not instance_text.isascii()
            or not instance_text.isdigit()
            or str(int(instance_text)) != instance_text
        ):
            raise ValueError(f"object_id must be a canonical category#N: {object_id!r}")
        instance = int(instance_text)
        mesh_name = mesh_name_for(category, instance)
        if object_id in used_ids or mesh_name in used_names:
            raise ValueError(f"duplicate/colliding addition declaration: {object_id!r}")
        if object_id in nodes and object_id not in removed:
            raise ValueError(
                f"existing object {object_id!r} must also appear in removed_objects "
                "to replace its mesh"
            )
        if mesh_name in aliases and aliases[mesh_name] not in removed:
            raise ValueError(
                f"addition collides with current Blender name: {mesh_name!r}"
            )
        for key in ("description", "support"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"{object_id!r} needs a nonempty {key}")
        used_ids.add(object_id)
        used_names.add(mesh_name)
        added.append(
            {
                **copy.deepcopy(raw),
                "category": category,
                "instance": instance,
                "mesh_name": mesh_name,
            }
        )
    final_ids = (set(nodes) - set(removed)) | used_ids
    support_names = {**aliases, **{row["mesh_name"]: row["object_id"] for row in added}}
    support_names.update(
        (node.get("build_name") or object_id.replace("#", "_"), object_id)
        for object_id, node in nodes.items()
        if node.get("kind") == "root_surface"
    )
    for row in added:
        if row["support"] not in final_ids or row["support"] == row["object_id"]:
            available = sorted(final_ids - {row["object_id"]})
            matched = support_names.get(row["support"])
            hint = (
                f"matching graph ID for this Blender name: {matched!r}"
                if matched in available
                else f"available final graph IDs (up to 12): {available[:12]!r}"
            )
            raise ValueError(
                f"{row['object_id']!r} has no valid final support: {row['support']!r}; "
                f"use an exact scene-graph ID; {hint}. Support-chain scope still applies."
            )
    return {
        "added_objects": added,
        "removed_objects": removed,
        "added_names": {row["object_id"]: row["mesh_name"] for row in added},
        "removed_names": {
            object_id: rows[object_id]["mesh_name"] for object_id in removed
        },
    }


def observed_object_changes(
    pre: dict[str, Any],
    post: dict[str, Any],
    *,
    added_names: list[str],
    removed_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate exact inventory delta and bind every observed object-pose change.

    Geometry authorization is checked inside Blender before save, so a pose-only
    transaction does not need an agent-supplied list of every object it moved.
    """
    before = pre.get("object_integrity") or {}
    after = post.get("object_integrity") or {}
    expected = (set(before) - set(removed_names)) | set(added_names)
    if set(after) != expected:
        raise RuntimeError(
            "initializer object delta differs from declarations: "
            f"missing={sorted(expected - set(after))!r}, "
            f"unexpected={sorted(set(after) - expected)!r}"
        )
    # Shape protection (RC1, 2026-09-14): an existing object's transform may change
    # rigidly or by a uniform scale. A non-uniform scale stretches the mesh (drawer
    # 1.19x1.08x1.50, spoon 1x1.40x1) and must be a declared replacement instead.
    for name in sorted(set(before) & set(after)):
        if name in added_names:
            continue  # a declared replacement may carry any scale
        anisotropy = anisotropic_scale_change(
            (before.get(name) or {}).get("scale"), (after.get(name) or {}).get("scale")
        )
        if anisotropy is not None:
            raise RuntimeError(
                f"initializer edit changed the shape of {name!r}: non-uniform scale "
                f"{anisotropy}. Existing objects may be translated, rotated, or "
                "uniformly rescaled; a different shape needs a declared replacement "
                "under the repair contract."
            )
    touched = {}
    for name in sorted(set(before) | set(after)):
        if before.get(name) == after.get(name) and name not in added_names:
            continue
        current = after.get(name)
        if current is not None:
            matrix = current.get("matrix")
            if (
                not isinstance(matrix, list)
                or len(matrix) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
                or any(
                    not isinstance(value, (float, int))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for row in matrix
                    for value in row
                )
            ):
                raise RuntimeError(f"initializer object has invalid pose: {name!r}")
        touched[name] = {
            "before_signature": copy.deepcopy(before.get(name)),
            "after_signature": copy.deepcopy(current),
            "before_matrix": copy.deepcopy((before.get(name) or {}).get("matrix")),
            "after_matrix": copy.deepcopy((current or {}).get("matrix")),
        }
    return touched
