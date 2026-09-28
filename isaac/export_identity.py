"""Stable pipeline-object identity across Blender's USD export boundary.

Pipeline mesh names are valid Blender names but not necessarily valid USD identifiers
(``obj_o-ring_0`` becomes ``obj_o_ring_0``), and Blender may append ``_<n>`` when two
exported prim names collide.  This module is the one place that resolves those dialects.

Fresh Isaac exports write the resolved records to ``object_identity.json``.  Downstream
steps consume that manifest instead of trying to recover object ownership from prim-path
strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

MANIFEST_VERSION = 1
_INVALID_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


class ExportIdentityError(ValueError):
    """The pipeline objects cannot be mapped one-to-one to exported USD roots."""


def usd_identifier(name: str) -> str:
    """Return the USD-safe identifier Blender produces for pipeline-name characters.

    Pipeline slugs are ASCII and always begin with ``obj_``; the leading-character guard
    keeps this helper correct for standalone tests and future callers as well.
    """
    out = _INVALID_IDENTIFIER.sub("_", str(name))
    if not out:
        return "_"
    if out[0].isdigit():
        out = "_" + out
    return out


def resolve_export_names(
    export_names: Iterable[str], pipeline_names: Iterable[str]
) -> dict[str, str]:
    """Resolve pipeline names to unique exported USD names.

    Exact matches are reserved first. Otherwise the USD-sanitized base and Blender dedup
    forms (``<sanitized>_<n>``) are considered together and must yield exactly one
    candidate. Any unaccounted canonical sibling, canonical-name collision, missing object,
    or reused export name is an error: guessing would attach one object's physics to another.
    """
    exported = [str(n) for n in export_names]
    pipeline = [str(n) for n in pipeline_names]
    if len(exported) != len(set(exported)):
        raise ExportIdentityError("duplicate exported USD object-root names")
    if len(pipeline) != len(set(pipeline)):
        raise ExportIdentityError("duplicate pipeline mesh names")

    by_safe: dict[str, list[str]] = {}
    for name in pipeline:
        by_safe.setdefault(usd_identifier(name), []).append(name)
    collisions = {safe: names for safe, names in by_safe.items() if len(names) > 1}
    if collisions:
        detail = ", ".join(
            f"{safe} <- {sorted(names)}" for safe, names in sorted(collisions.items())
        )
        raise ExportIdentityError(f"pipeline names collide after USD sanitization: {detail}")

    available = set(exported)
    resolved: dict[str, str] = {}

    # Reserve all exact matches first so a sanitized spelling can never steal a real name.
    for name in pipeline:
        if name in available:
            resolved[name] = name
            available.remove(name)

    # An exact name is not sufficient when an otherwise-unclaimed exported sibling shares
    # its canonical family: exporter ordering alone cannot tell us which source object got
    # the base spelling. Exact siblings that are themselves pipeline names were reserved
    # above and are safe.
    for name in pipeline:
        if name not in resolved:
            continue
        siblings = sorted(
            candidate
            for candidate in available
            if re.fullmatch(re.escape(name) + r"_\d+", candidate)
        )
        if siblings:
            raise ExportIdentityError(
                f"pipeline object {name!r} has ambiguous exported USD roots: "
                f"{[name, *siblings]}"
            )

    for name in pipeline:
        if name in resolved:
            continue
        safe = usd_identifier(name)
        candidates = sorted(
            candidate
            for candidate in available
            if candidate == safe
            or re.fullmatch(re.escape(safe) + r"_\d+", candidate)
        )
        if len(candidates) == 1:
            resolved[name] = candidates[0]
            available.remove(candidates[0])
            continue
        if not candidates:
            raise ExportIdentityError(
                f"pipeline object {name!r} has no exported USD root "
                f"(expected {safe!r} or a unique dedup suffix)"
            )
        raise ExportIdentityError(
            f"pipeline object {name!r} has ambiguous exported USD roots: {candidates}"
        )
    return resolved


def build_identity_manifest(
    pipeline_names: Iterable[str], root_prims: Iterable[tuple[str, str]]
) -> dict:
    """Build a versioned manifest from ``(USD name, absolute prim path)`` roots."""
    roots = [(str(name), str(path)) for name, path in root_prims]
    by_name = {name: path for name, path in roots}
    if len(by_name) != len(roots):
        raise ExportIdentityError("duplicate exported USD object-root names")
    names = [str(n) for n in pipeline_names]
    resolved = resolve_export_names(by_name, names)
    return {
        "version": MANIFEST_VERSION,
        "objects": [
            {
                "pipeline_name": name,
                "usd_name": resolved[name],
                "prim_path": by_name[resolved[name]],
            }
            for name in names
        ],
    }


def validate_identity_manifest(data: dict) -> dict:
    """Validate and return an identity manifest loaded from JSON."""
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        raise ExportIdentityError(
            f"object identity manifest must have version {MANIFEST_VERSION}"
        )
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ExportIdentityError("object identity manifest has no objects")
    required = {"pipeline_name", "usd_name", "prim_path"}
    for i, record in enumerate(objects):
        if not isinstance(record, dict) or not required <= record.keys():
            raise ExportIdentityError(f"invalid object identity record at index {i}")
        if not str(record["prim_path"]).startswith("/"):
            raise ExportIdentityError(
                f"object identity prim path is not absolute: {record['prim_path']!r}"
            )
    for key in required:
        values = [str(record[key]) for record in objects]
        if len(values) != len(set(values)):
            raise ExportIdentityError(f"duplicate {key} in object identity manifest")

    # A dynamic object root below another dynamic root creates nested rigid bodies and
    # makes collider ownership ambiguous.  Reject it at every manifest-consumption seam.
    paths = sorted(str(record["prim_path"]).rstrip("/") for record in objects)
    for i, parent in enumerate(paths):
        for child in paths[i + 1 :]:
            if child.startswith(parent + "/"):
                raise ExportIdentityError(
                    f"nested pipeline object roots are forbidden: {child} under {parent}"
                )
    return data


def load_identity_manifest(path: str | Path) -> dict:
    return validate_identity_manifest(json.loads(Path(path).read_text()))


def write_identity_manifest(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(validate_identity_manifest(data), indent=1))
