"""Fail-closed helpers for transaction-owned procedural object capture.

The functions in this module do not import Blender.  They validate the small piece of
model-authored Python and build a *trusted* follow-up script that an executor can run in
Blender after its ordinary mutation guard has accepted that code.  The follow-up script
owns names, export paths, capture metadata, and saving the candidate Blend.

Every accepted capture is re-imported by a factory-empty Blender child process.  The
record is emitted only after that isolated round trip preserves the union's world
bounds, single-mesh asset structure, surviving material identities, and packed texture
identities.

The live representation is one canonical EMPTY root with an editable MESH descendant
tree. Canonicalization changes object/datablock names and adds descriptive part labels,
while preserving every parent link and transform channel. Authored mesh datablocks,
material slots, modifiers, and the target's world pose remain the live source of truth.
A shared centered root-local
exact Boolean builds both the exported asset and runtime collider arrays. The temporary
union is deleted before saving; a plain mesh join is never treated as a geometric union.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROCEDURAL_CAPTURE_SCHEMA_VERSION = 1
MAX_AUTHORED_CODE_BYTES = 256_000
MAX_PART_LABEL_BYTES = 128

_TARGET_NAME_RE = re.compile(r"^obj_[A-Za-z0-9_-]+$")
_TARGET_ID_RE = re.compile(r"^[^#\r\n]+#[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IMPORT_ROOTS = frozenset(
    {
        "bmesh",
        "bpy",
        "collections",
        "colorsys",
        "itertools",
        "math",
        "mathutils",
        "numpy",
        "random",
    }
)
# Pure helper modules the agents keep reaching for (world_to_camera_view, view3d
# ray helpers). Exact module names only: bpy_extras.image_utils / io_utils touch files.
_SAFE_IMPORT_MODULES = frozenset(
    {"bpy_extras.object_utils", "bpy_extras.view3d_utils"}
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
)
_FORBIDDEN_BPY_CALL_PREFIXES = (
    "bpy.data.libraries.load",
    "bpy.data.libraries.write",
    "bpy.ops.export_scene",
    "bpy.ops.import_scene",
    "bpy.ops.render",
    "bpy.ops.script",
)
_FORBIDDEN_BPY_CALLS = frozenset(
    {
        "bpy.ops.wm.open_mainfile",
        "bpy.ops.wm.quit_blender",
        "bpy.ops.wm.read_factory_settings",
        "bpy.ops.wm.read_homefile",
        "bpy.ops.wm.recover_auto_save",
        "bpy.ops.wm.recover_last_session",
        "bpy.ops.wm.save_as_mainfile",
        "bpy.ops.wm.save_mainfile",
    }
)
_PLACEMENT_EVIDENCE_FIELDS = frozenset(
    {
        "depth",
        "description",
        "mask_path",
        "redetect",
        "rollable",
        "screen_bbox",
    }
)


class ProceduralObjectError(ValueError):
    """Procedural code, capture metadata, or placement input is invalid."""


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


class _AuthoredCodeValidator(ast.NodeVisitor):
    """Reject obvious attempts to cross the executor-owned transaction boundary."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.partition(".")[0]
            if root not in _SAFE_IMPORT_ROOTS and alias.name not in _SAFE_IMPORT_MODULES:
                self.errors.append(f"import of {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        root = (node.module or "").partition(".")[0]
        if node.level or (
            root not in _SAFE_IMPORT_ROOTS and node.module not in _SAFE_IMPORT_MODULES
        ):
            label = "." * node.level + (node.module or "")
            self.errors.append(f"import from {label!r} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in {"TARGET_OBJECT_ID", "TARGET_OBJECT_NAME"} and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            self.errors.append(f"backend binding {node.id} may not be assigned")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        for name in node.names:
            if name in {"TARGET_OBJECT_ID", "TARGET_OBJECT_NAME"}:
                self.errors.append(f"backend binding {name} may not be declared global")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        for name in node.names:
            if name in {"TARGET_OBJECT_ID", "TARGET_OBJECT_NAME"}:
                self.errors.append(
                    f"backend binding {name} may not be declared nonlocal"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            self.errors.append(f"call to {node.func.id}() is not allowed")
        path = _attribute_path(node.func)
        if path is not None and (
            path in _FORBIDDEN_BPY_CALLS
            or any(
                path.startswith(prefix + ".") or path == prefix
                for prefix in _FORBIDDEN_BPY_CALL_PREFIXES
            )
        ):
            self.errors.append(f"call to {path}() is backend-owned")
        self.generic_visit(node)


def validate_authored_code(code: str) -> str:
    """Validate and return one procedural Blender program.

    This is intentionally a first-line validation layer, not a Python sandbox.  The
    executor must still run the code in its isolated candidate Blend and enforce its
    object/artifact integrity snapshots.  Here we reject syntax errors, oversized input,
    backend binding reassignment, filesystem/process imports, and direct Blender
    load/save/export/render calls so ordinary mistakes fail before a subprocess starts.
    """

    if not isinstance(code, str) or not code.strip():
        raise ProceduralObjectError("authored code must be a nonempty string")
    if "\x00" in code:
        raise ProceduralObjectError("authored code contains a NUL byte")
    if len(code.encode("utf-8")) > MAX_AUTHORED_CODE_BYTES:
        raise ProceduralObjectError(
            f"authored code exceeds {MAX_AUTHORED_CODE_BYTES} UTF-8 bytes"
        )
    try:
        tree = ast.parse(code, filename="<procedural-object>", mode="exec")
    except SyntaxError as exc:
        detail = exc.msg or "invalid syntax"
        raise ProceduralObjectError(
            f"authored code has invalid Python syntax at line {exc.lineno}: {detail}"
        ) from exc
    validator = _AuthoredCodeValidator()
    validator.visit(tree)
    if validator.errors:
        unique = list(dict.fromkeys(validator.errors))
        raise ProceduralObjectError(
            "authored code crosses the backend-owned boundary: " + "; ".join(unique)
        )
    return code


def _validate_target_binding(
    target_object_name: str, target_object_id: str
) -> tuple[str, str]:
    if not isinstance(target_object_name, str) or not _TARGET_NAME_RE.fullmatch(
        target_object_name
    ):
        raise ProceduralObjectError(
            "target_object_name must be a canonical obj_* name containing only "
            "letters, digits, '_' or '-'"
        )
    if len(target_object_name) > 63:
        raise ProceduralObjectError("target_object_name exceeds Blender's 63-byte name")
    if not isinstance(target_object_id, str) or not _TARGET_ID_RE.fullmatch(
        target_object_id
    ):
        raise ProceduralObjectError(
            "target_object_id must be an exact nonempty category#instance identity"
        )
    return target_object_name, target_object_id


def build_target_prelude(target_object_name: str, target_object_id: str) -> str:
    """Return backend-owned Python globals prepended to model-authored code."""

    target_object_name, target_object_id = _validate_target_binding(
        target_object_name, target_object_id
    )
    return (
        "# Backend-owned procedural object identity. Do not reassign.\n"
        f"TARGET_OBJECT_NAME = {json.dumps(target_object_name)}\n"
        f"TARGET_OBJECT_ID = {json.dumps(target_object_id)}\n"
    )


def _absolute_output_path(value: str | Path, *, suffix: str, label: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ProceduralObjectError(f"{label} must be a filesystem path") from exc
    if not raw or "\x00" in raw:
        raise ProceduralObjectError(f"{label} must be a nonempty path without NUL")
    path = Path(raw)
    if not path.is_absolute():
        raise ProceduralObjectError(f"{label} must be an absolute transaction path")
    if path.suffix.lower() != suffix:
        raise ProceduralObjectError(f"{label} must end in {suffix}")
    return str(path)


_CAPTURE_SCRIPT = r'''
import hashlib as _grase_hashlib
import importlib.util as _grase_importlib_util
import json as _grase_json
import math as _grase_math
import os as _grase_os
import subprocess as _grase_subprocess

import bmesh as _grase_bmesh
import bpy as _grase_bpy
from mathutils import Matrix as _grase_Matrix

_GRASE_TARGET_NAME = __GRASE_TARGET_NAME__
_GRASE_TARGET_ID = __GRASE_TARGET_ID__
_GRASE_GLB_PATH = __GRASE_GLB_PATH__
_GRASE_CAPTURE_PATH = __GRASE_CAPTURE_PATH__
_GRASE_DIAGNOSTICS_PATH = __GRASE_DIAGNOSTICS_PATH__
_GRASE_BLEND_PATH = __GRASE_BLEND_PATH__
_GRASE_CAPTURE_SCHEMA = __GRASE_CAPTURE_SCHEMA__
_GRASE_AUTHORED_GEOMETRY_PATH = __GRASE_AUTHORED_GEOMETRY_PATH__
_GRASE_MAX_PART_LABEL_BYTES = __GRASE_MAX_PART_LABEL_BYTES__

# Trusted adjacent source, independent of Blender's cwd/PYTHONPATH/package cache.
# [MLR-02.08] optional Blender helper: load only inside this authored capture.
_grase_geometry_spec = _grase_importlib_util.spec_from_file_location(
    "_grase_capture_authored_root_local_geometry", _GRASE_AUTHORED_GEOMETRY_PATH
)
if _grase_geometry_spec is None or _grase_geometry_spec.loader is None:
    raise RuntimeError("could not load authored capture geometry helper")
_grase_authored_geometry = _grase_importlib_util.module_from_spec(_grase_geometry_spec)
_grase_geometry_spec.loader.exec_module(_grase_authored_geometry)


def _grase_fail(message):
    raise RuntimeError("procedural object capture rejected: " + str(message))


def _grase_matrix_rows(matrix):
    rows = [[float(value) for value in row] for row in matrix]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        _grase_fail("target matrix is not 4x4")
    if not all(_grase_math.isfinite(value) for row in rows for value in row):
        _grase_fail("target matrix contains a non-finite value")
    return rows


def _grase_plain(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if _grase_math.isfinite(value) else repr(value)
    try:
        return [_grase_plain(item) for item in value]
    except TypeError:
        return repr(value)


def _grase_image_fact(image):
    packed_sha256 = None
    packed = getattr(image, "packed_file", None)
    if packed is not None:
        try:
            packed_sha256 = _grase_hashlib.sha256(bytes(packed.data)).hexdigest()
        except Exception:
            packed_sha256 = None
    return {
        "name": image.name,
        "source": str(image.source),
        "filepath": str(image.filepath),
        "file_format": str(image.file_format),
        "size": [int(image.size[0]), int(image.size[1])],
        "colorspace": str(image.colorspace_settings.name),
        "packed": packed is not None,
        "packed_sha256": packed_sha256,
    }


def _grase_material_fact(material):
    payload = {
        "name": material.name,
        "use_nodes": bool(material.use_nodes),
        "diffuse_color": _grase_plain(material.diffuse_color),
        "nodes": [],
        "links": [],
        "images": [],
    }
    images = {}
    if material.use_nodes and material.node_tree is not None:
        for node in sorted(material.node_tree.nodes, key=lambda item: (item.bl_idname, item.name)):
            inputs = []
            for socket in node.inputs:
                if hasattr(socket, "default_value"):
                    inputs.append([socket.name, _grase_plain(socket.default_value)])
            image = getattr(node, "image", None)
            if image is not None:
                images[image.name] = _grase_image_fact(image)
            payload["nodes"].append(
                {
                    "name": node.name,
                    "type": node.bl_idname,
                    "inputs": inputs,
                    "image": image.name if image is not None else None,
                }
            )
        payload["links"] = sorted(
            [
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            ]
            for link in material.node_tree.links
        )
    payload["images"] = [images[name] for name in sorted(images)]
    encoded = _grase_json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = _grase_hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def _grase_image_inventory(rows):
    """Count each image identity once while retaining contradictory dimensions."""
    images, conflicts = {}, []
    for row in rows:
        name, size = row["name"], list(row["size"])
        if name in images and images[name] != size:
            conflicts.append({"name": name, "sizes": [images[name], size]})
        else:
            images[name] = size
    return images, conflicts


def _grase_compare_image_inventories(source_rows, imported_rows):
    """Keep exact names, allowing only an unambiguous byte-identical export rename."""
    expected, source_conflicts = _grase_image_inventory(source_rows)
    actual, imported_conflicts = _grase_image_inventory(imported_rows)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = [
        {"name": name, "expected_size": expected[name], "actual_size": actual[name]}
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    ]
    inventories, content_hashes = [], []
    for rows in (source_rows, imported_rows):
        facts, hashes = {}, {}
        for row in rows:
            name, size, digest = row["name"], list(row["size"]), row.get("packed_sha256")
            facts[(name, tuple(size), digest)] = {
                "name": name, "size": size, "packed_sha256": digest,
            }
            hashes.setdefault(name, set()).add(digest)
        inventories.append(sorted(facts.values(), key=lambda row: (
            row["name"], row["size"], row["packed_sha256"] or "",
        )))
        content_hashes.append({
            name: next(iter(values)) if len(values) == 1 else None
            for name, values in hashes.items()
        })
    source_hashes, imported_hashes = content_hashes
    aliases, alias_rejection = [], None
    if (missing or unexpected) and not (changed or source_conflicts or imported_conflicts):
        if len(expected) != len(actual) or len(missing) != len(unexpected):
            alias_rejection = "unique image cardinalities differ"
        elif not all(isinstance(value, str) and value for hashes in content_hashes for value in hashes.values()):
            alias_rejection = "packed image hashes are missing or conflicting"
        elif any(source_hashes[name] != imported_hashes[name] for name in set(expected) & set(actual)):
            alias_rejection = "common-name image bytes changed"
        else:
            remaining = set(unexpected)
            for name in missing:
                matches = [
                    other for other in sorted(remaining)
                    if expected[name] == actual[other]
                    and source_hashes[name] == imported_hashes[other]
                ]
                if len(matches) != 1:
                    aliases = []
                    alias_rejection = "renamed image lacks a unique byte-identical match"
                    break
                other = matches[0]
                remaining.remove(other)
                aliases.append({
                    "source_name": name, "imported_name": other,
                    "size": expected[name], "packed_sha256": source_hashes[name],
                })
    names_match = not (missing or unexpected) or bool(aliases)
    return {
        "status": "passed" if names_match and not (changed or source_conflicts or imported_conflicts) else "failed",
        "expected_images": inventories[0],
        "actual_images": inventories[1],
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
        "source_conflicts": source_conflicts,
        "imported_conflicts": imported_conflicts,
        "accepted_aliases": aliases,
        "alias_rejection": alias_rejection,
    }


def _grase_object_path(obj, root):
    names = [obj.name]
    parent = obj.parent
    while parent is not None and parent is not root:
        names.append(parent.name)
        parent = parent.parent
    if obj is not root and parent is not root:
        _grase_fail("target descendants do not form one rooted tree")
    return tuple(reversed(names))


_grase_target = _grase_bpy.data.objects.get(_GRASE_TARGET_NAME)
if _grase_target is None:
    _grase_fail("canonical target object is missing")
if _grase_target.name != _GRASE_TARGET_NAME:
    _grase_fail("canonical target object was renamed")
if _grase_target.type != "EMPTY" or _grase_target.data is not None:
    _grase_fail("canonical target root must be an EMPTY with no object data")
if _grase_target.parent is not None:
    _grase_fail("canonical target root must be parentless")
if _grase_target.library is not None:
    _grase_fail("linked canonical target roots are not exportable")
_grase_matrix_rows(_grase_target.matrix_world)
_grase_root_determinant = float(_grase_target.matrix_world.to_3x3().determinant())
if not _grase_math.isfinite(_grase_root_determinant) or _grase_root_determinant <= 1.0e-12:
    _grase_fail("canonical target root requires a finite, non-mirrored world transform")
if _grase_target.constraints:
    _grase_fail("canonical target root may not retain constraints")
if _grase_target.animation_data is not None:
    _grase_fail("canonical target root may not retain animation or drivers")
if _grase_target.hide_render or _grase_target.hide_viewport or _grase_target.hide_get():
    _grase_fail("canonical target root must be visible for deterministic export")

_grase_parts = list(_grase_target.children_recursive)
if not _grase_parts:
    _grase_fail("canonical target EMPTY must have at least one MESH child")
if len(set(_grase_parts)) != len(_grase_parts):
    _grase_fail("target subtree contains duplicate descendant references")
for _grase_part in _grase_parts:
    if _grase_part.type != "MESH":
        _grase_fail(
            "target subtree may contain only MESH objects; found "
            + _grase_part.type
            + " "
            + _grase_part.name
        )
    if _grase_part.data is None or _grase_part.data.users != 1:
        _grase_fail("each target part must own one unshared mesh datablock")
    if _grase_part.library is not None or _grase_part.data.library is not None:
        _grase_fail("linked target parts are not exportable")
    if _grase_part.constraints:
        _grase_fail("target parts may not retain constraints")
    if _grase_part.animation_data is not None:
        _grase_fail("target parts may not retain animation or drivers")
    if _grase_part.hide_render or _grase_part.hide_viewport or _grase_part.hide_get():
        _grase_fail("target parts must be visible for deterministic export")
    matrix = _grase_matrix_rows(_grase_part.matrix_world)
    determinant = float(_grase_part.matrix_world.to_3x3().determinant())
    if not _grase_math.isfinite(determinant) or determinant <= 1.0e-12:
        _grase_fail("target parts require a finite, non-mirrored world transform")

_grase_bound = [
    obj
    for obj in _grase_bpy.data.objects
    if obj.get("grase_graph_id") == _GRASE_TARGET_ID
]
if any(obj is not _grase_target for obj in _grase_bound):
    _grase_fail("graph identity is already bound to another Blender object")
_grase_existing_id = _grase_target.get("grase_graph_id")
if _grase_existing_id not in (None, "", _GRASE_TARGET_ID):
    _grase_fail("canonical target root is bound to a different graph identity")
if any(obj.get("grase_graph_id") not in (None, "") for obj in _grase_parts):
    _grase_fail("only the canonical target root may carry a graph identity")

# Establish a repeatable order before changing hierarchy or names.  Paths make the
# first canonicalization deterministic; canonical part names keep later captures stable.
_grase_descendants = sorted(
    _grase_parts,
    key=lambda obj: (_grase_object_path(obj, _grase_target), obj.name),
)
_grase_world_before = {
    obj: obj.matrix_world.copy() for obj in [_grase_target] + _grase_parts
}
_grase_data_before = {obj: obj.data for obj in _grase_parts}
_grase_part_labels = {}
_grase_authored_names = {}
_grase_seen_part_labels = set()
for obj in _grase_descendants:
    label = obj.get("grase_part_label", obj.name)
    authored_name = obj.get("grase_authored_name", obj.name)
    for field, value in (("grase_part_label", label), ("grase_authored_name", authored_name)):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.encode("utf-8")) > _GRASE_MAX_PART_LABEL_BYTES
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            _grase_fail(
                "part " + repr(obj.name) + " " + field
                + " must be a nonempty string of at most "
                + str(_GRASE_MAX_PART_LABEL_BYTES) + " UTF-8 bytes without control characters"
            )
    if label in _grase_seen_part_labels:
        _grase_fail(
            "duplicate grase_part_label " + repr(label)
            + "; assign a unique semantic label to each part in the target tree"
        )
    _grase_part_labels[obj] = label
    _grase_authored_names[obj] = authored_name
    _grase_seen_part_labels.add(label)
_grase_slug = _GRASE_TARGET_NAME[4:44]
_grase_names = {
    obj: "_grase_" + _grase_slug + "_part_" + f"{index:03d}"
    for index, obj in enumerate(_grase_descendants, 1)
}
_grase_mesh_names = {
    obj: "mesh_" + _grase_slug + "_part_" + f"{index:03d}"
    for index, obj in enumerate(_grase_descendants, 1)
}
for obj, name in _grase_names.items():
    owner = _grase_bpy.data.objects.get(name)
    if owner is not None and owner not in _grase_parts:
        _grase_fail("canonical part name is already owned outside the target tree: " + name)
for obj, name in _grase_mesh_names.items():
    owner = _grase_bpy.data.meshes.get(name)
    if owner is not None and owner not in _grase_data_before.values():
        _grase_fail("canonical mesh name is already owned outside the target tree: " + name)

# Temporary names prevent Blender's automatic .001 suffixes when the desired canonical
# name is currently held by a different member of this same tree. Keep the authored
# hierarchy and raw transform channels exact: the shared helper evaluates nested parts
# directly and never needs a flatten/reparent/world-matrix round trip.
for index, obj in enumerate(_grase_descendants, 1):
    obj.name = "_grase_capture_tmp_object_" + f"{index:03d}"
for index, obj in enumerate(_grase_descendants, 1):
    obj.data.name = "_grase_capture_tmp_mesh_" + f"{index:03d}"
for obj in _grase_descendants:
    obj.name = _grase_names[obj]
for obj in _grase_descendants:
    obj.data.name = _grase_mesh_names[obj]
    # Preserve human-authored part identity across canonical renaming and reloads.
    # These descriptive properties do not authorize edits or alter exported geometry.
    obj["grase_part_label"] = _grase_part_labels[obj]
    obj["grase_authored_name"] = _grase_authored_names[obj]
_grase_target["grase_graph_id"] = _GRASE_TARGET_ID
_grase_bpy.context.view_layer.update()

for obj in [_grase_target] + _grase_parts:
    if obj is _grase_target:
        before = _grase_matrix_rows(_grase_world_before[obj])
        after = _grase_matrix_rows(obj.matrix_world)
        error = max(
            abs(after[row][column] - before[row][column])
            for row in range(4)
            for column in range(4)
        )
        if error > 1.0e-7:
            _grase_fail("canonicalization changed the target root's world pose")
        continue
    if obj.data is not _grase_data_before[obj]:
        _grase_fail("canonicalization replaced live mesh data")
    before = _grase_matrix_rows(_grase_world_before[obj])
    after = _grase_matrix_rows(obj.matrix_world)
    error = max(
        abs(after[row][column] - before[row][column])
        for row in range(4)
        for column in range(4)
    )
    if error > 1.0e-7:
        _grase_fail("canonicalization changed a target part's world pose")

_grase_depsgraph = _grase_bpy.context.evaluated_depsgraph_get()
_grase_points = []
_grase_part_rows = []
_grase_materials = {}
_grase_has_null_material = False
for obj in _grase_descendants:
    evaluated = obj.evaluated_get(_grase_depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=_grase_depsgraph)
    try:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
            _grase_fail(
                "every evaluated target part must contain vertices and polygons; authored part "
                + repr(_GRASE_TARGET_ID + "/" + (_grase_part_labels[obj] or obj.name))
            )
        for vertex in mesh.vertices:
            point = evaluated.matrix_world @ vertex.co
            values = [float(point[axis]) for axis in range(3)]
            if not all(_grase_math.isfinite(value) for value in values):
                _grase_fail("evaluated target geometry contains a non-finite coordinate")
            _grase_points.append(values)
        slots = []
        for slot_index, slot in enumerate(obj.material_slots):
            material = slot.material
            if material is None:
                _grase_has_null_material = True
            slots.append(
                {
                    "slot": slot_index,
                    "material": material.name if material is not None else None,
                }
            )
            if material is not None:
                for node in material.node_tree.nodes if material.use_nodes and material.node_tree else []:
                    image = getattr(node, "image", None)
                    if image is not None and getattr(image, "packed_file", None) is None:
                        _grase_fail(
                            "material "
                            + material.name
                            + " has an external or unpacked texture dependency: "
                            + image.name
                        )
                _grase_materials[material.name] = _grase_material_fact(material)
        _grase_part_rows.append(
            {
                "name": obj.name,
                "label": _grase_part_labels[obj],
                "authored_name": _grase_authored_names[obj],
                "mesh_data_name": obj.data.name,
                "vertex_count": len(mesh.vertices),
                "polygon_count": len(mesh.polygons),
                "material_slots": slots,
                "modifier_types": [modifier.type for modifier in obj.modifiers],
                "matrix_world": _grase_matrix_rows(obj.matrix_world),
            }
        )
    finally:
        evaluated.to_mesh_clear()

if not _grase_points:
    _grase_fail("target tree has no evaluated geometry")
_grase_minimum = [min(point[axis] for point in _grase_points) for axis in range(3)]
_grase_maximum = [max(point[axis] for point in _grase_points) for axis in range(3)]
_grase_size = [
    _grase_maximum[axis] - _grase_minimum[axis] for axis in range(3)
]
if any(not _grase_math.isfinite(value) or value <= 1.0e-9 for value in _grase_size):
    _grase_fail("target world bounds must have positive finite size on every axis")
_grase_center = [
    (_grase_minimum[axis] + _grase_maximum[axis]) * 0.5 for axis in range(3)
]


def _grase_solid_stats(mesh, label):
    """Return topological solid evidence without repairing authored geometry."""
    bm = _grase_bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        if not bm.verts or not bm.faces:
            _grase_fail(label + " contains no volumetric mesh")
        non_manifold = [edge for edge in bm.edges if not edge.is_manifold]
        if non_manifold:
            _grase_fail(
                label
                + " is not a closed manifold solid ("
                + str(len(non_manifold))
                + " non-manifold/boundary edges); close the authored primitive geometry"
            )
        pending = set(bm.faces)
        components = 0
        while pending:
            components += 1
            stack = [pending.pop()]
            while stack:
                face = stack.pop()
                for edge in face.edges:
                    for neighbor in edge.link_faces:
                        if neighbor in pending:
                            pending.remove(neighbor)
                            stack.append(neighbor)
        volume = abs(float(bm.calc_volume(signed=True)))
        if not _grase_math.isfinite(volume) or volume <= 1.0e-12:
            _grase_fail(label + " has no positive finite enclosed volume")
        return {
            "component_count": components,
            "manifold": True,
            "volume": volume,
            "vertex_count": len(bm.verts),
            "polygon_count": len(bm.faces),
        }
    finally:
        bm.free()


# Build one disposable centered root-local exact union. Capture export and runtime
# registration intentionally share these exact bytes; only their output adapters differ.
# The helper snapshots every evaluated source before linking temporary objects and never
# mutates the canonical Empty or its editable descendant hierarchy.
_grase_glb_parent = _grase_os.path.dirname(_GRASE_GLB_PATH)
_grase_capture_parent = _grase_os.path.dirname(_GRASE_CAPTURE_PATH)
_grase_os.makedirs(_grase_glb_parent, exist_ok=True)
_grase_os.makedirs(_grase_capture_parent, exist_ok=True)
if _grase_os.path.exists(_GRASE_GLB_PATH):
    _grase_os.unlink(_GRASE_GLB_PATH)

try:
    with _grase_authored_geometry.authored_union(
        _grase_target,
        _grase_descendants,
        bpy=_grase_bpy,
        bmesh=_grase_bmesh,
    ) as _grase_union_payload:
        _grase_union, _grase_canonical_to_world = (
            _grase_authored_geometry.prepare_capture_object(
                _grase_union_payload, Matrix=_grase_Matrix
            )
        )
        _grase_bpy.context.view_layer.update()
        _grase_union.data.name = "mesh_" + _grase_slug + "_connected_union"
        _grase_solid = _grase_union_payload["solid_stats"]
        _grase_union_stats = {
            "component_count": int(_grase_solid["components"]),
            "manifold": True,
            "volume": float(_grase_solid["world_volume"]),
            "vertex_count": int(
                _grase_union_payload["canonical_stats"]["vertices"]
            ),
            "polygon_count": int(
                _grase_union_payload["canonical_stats"]["triangles"]
            ),
        }
        _grase_union_minimum = list(_grase_union_payload["world_bounds"]["min"])
        _grase_union_maximum = list(_grase_union_payload["world_bounds"]["max"])
        # a seam-weld retry offsets the operand copies by up to bounds_slack_m (0.02 mm x parts)
        _grase_bounds_slack = float(_grase_union_payload.get("bounds_slack_m", 0.0) or 0.0)
        for axis in range(3):
            tolerance = max(1.0e-5, abs(_grase_size[axis]) * 1.0e-5) + _grase_bounds_slack
            if (
                abs(_grase_union_minimum[axis] - _grase_minimum[axis]) > tolerance
                or abs(_grase_union_maximum[axis] - _grase_maximum[axis]) > tolerance
            ):
                _grase_fail(
                    "exact Boolean union changed the authored target's world bounds"
                )

        _grase_union_material_names = sorted(
            name for name in _grase_union_payload["material_names"] if name
        )
        _grase_union_material_slot_count = len(_grase_union.data.materials)
        _grase_union_geometry_payload = {
            "vertices": [
                [round(float(value), 9) for value in vertex]
                for vertex in _grase_union_payload["world_vertices"]
            ],
            "polygons": [
                [int(value) for value in face]
                for face in _grase_union_payload["faces"]
            ],
        }
        _grase_union_geometry_sha256 = _grase_hashlib.sha256(
            _grase_json.dumps(
                _grase_union_geometry_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        _grase_selected_before = list(_grase_bpy.context.selected_objects)
        _grase_active_before = _grase_bpy.context.view_layer.objects.active
        try:
            _grase_bpy.ops.object.select_all(action="DESELECT")
            _grase_union.select_set(True)
            _grase_bpy.context.view_layer.objects.active = _grase_union
            result = _grase_bpy.ops.export_scene.gltf(
                filepath=_GRASE_GLB_PATH,
                export_format="GLB",
                use_selection=True,
                export_apply=True,
                # GLB is a Y-up format. Keeping the standard conversion lets a fresh
                # Blender importer reconstruct the exact authored Z-up world pose.
                export_yup=True,
                export_materials="EXPORT",
            )
            if "FINISHED" not in result:
                _grase_fail("Blender GLB exporter did not finish")
        finally:
            _grase_bpy.ops.object.select_all(action="DESELECT")
            for obj in _grase_selected_before:
                if obj.name in _grase_bpy.context.view_layer.objects:
                    obj.select_set(True)
            if (
                _grase_active_before is not None
                and _grase_active_before.name in _grase_bpy.context.view_layer.objects
            ):
                _grase_bpy.context.view_layer.objects.active = _grase_active_before
except RuntimeError as exc:
    if "union is disconnected" in str(exc):
        _grase_fail(
            "exact Boolean union for "
            + _GRASE_TARGET_NAME
            + " has disconnected solid components; connect the authored primitives "
            + "with real overlapping structural geometry"
        )
    raise
if not _grase_os.path.isfile(_GRASE_GLB_PATH):
    _grase_fail("GLB exporter did not create the requested artifact")
_grase_glb_size = _grase_os.path.getsize(_GRASE_GLB_PATH)
if _grase_glb_size <= 20:
    _grase_fail("GLB artifact is empty or truncated")
with open(_GRASE_GLB_PATH, "rb") as stream:
    if stream.read(4) != b"glTF":
        _grase_fail("exported artifact has no GLB magic")
    stream.seek(0)
    digest = _grase_hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
_grase_glb_sha256 = digest.hexdigest()

_grase_material_rows = [_grase_materials[name] for name in sorted(_grase_materials)]
_grase_material_json = _grase_json.dumps(
    _grase_material_rows, sort_keys=True, separators=(",", ":")
)

# Import the just-written GLB in a factory-empty Blender child process.  A new process is
# intentional: importing into a temporary collection in this process can silently reuse
# live materials/images by datablock name, which is not evidence that the GLB is durable.
_grase_roundtrip_script = r"""
import json
import math
import os
import sys

import bpy


def matrix_rows(matrix):
    return [[float(value) for value in row] for row in matrix]


def image_fact(image):
    import hashlib
    packed = getattr(image, "packed_file", None)
    return {
        "name": image.name,
        "size": [int(image.size[0]), int(image.size[1])],
        "packed": packed is not None,
        "packed_sha256": hashlib.sha256(bytes(packed.data)).hexdigest() if packed else None,
    }


start = sys.argv.index("--") + 1
glb_path, output_path = sys.argv[start : start + 2]
bpy.ops.wm.read_factory_settings(use_empty=True)
result = bpy.ops.import_scene.gltf(filepath=glb_path)
if "FINISHED" not in result:
    raise RuntimeError("isolated GLB importer did not finish")
objects = list(bpy.context.scene.objects)
meshes = [obj for obj in objects if obj.type == "MESH"]
non_mesh = [obj.name + ":" + obj.type for obj in objects if obj.type != "MESH"]
if non_mesh:
    raise RuntimeError("isolated GLB contains non-MESH nodes: " + ", ".join(non_mesh))
if not meshes:
    raise RuntimeError("isolated GLB contains no meshes")

depsgraph = bpy.context.evaluated_depsgraph_get()
points = []
mesh_rows = []
materials = {}
images = {}
material_slot_count = 0
for obj in meshes:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        if mesh is None or not mesh.vertices or not mesh.polygons:
            raise RuntimeError("isolated GLB contains empty mesh " + obj.name)
        for vertex in mesh.vertices:
            point = evaluated.matrix_world @ vertex.co
            values = [float(point[axis]) for axis in range(3)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("isolated GLB contains non-finite geometry")
            points.append(values)
        slot_names = []
        for slot in obj.material_slots:
            material_slot_count += 1
            material = slot.material
            slot_names.append(material.name if material is not None else None)
            if material is None:
                continue
            materials[material.name] = True
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    image = getattr(node, "image", None)
                    if image is not None:
                        images[image.name] = image_fact(image)
        mesh_rows.append(
            {
                "name": obj.name,
                "vertex_count": len(mesh.vertices),
                "polygon_count": len(mesh.polygons),
                "material_slots": slot_names,
                "matrix_world": matrix_rows(obj.matrix_world),
            }
        )
    finally:
        evaluated.to_mesh_clear()

minimum = [min(point[axis] for point in points) for axis in range(3)]
maximum = [max(point[axis] for point in points) for axis in range(3)]
payload = {
    "mesh_count": len(meshes),
    "meshes": sorted(mesh_rows, key=lambda row: row["name"]),
    "world_bounds": {
        "min": minimum,
        "max": maximum,
        "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
    },
    "material_names": sorted(materials),
    "material_slot_count": material_slot_count,
    "images": [images[name] for name in sorted(images)],
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    (os.fsync(stream.fileno()) if os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
os.replace(temporary, output_path)
print("GRASE_PROCEDURAL_ROUNDTRIP_OK")
"""
_grase_roundtrip_script_path = _GRASE_CAPTURE_PATH + ".roundtrip.py"
_grase_roundtrip_result_path = _GRASE_CAPTURE_PATH + ".roundtrip.json"
try:
    with open(_grase_roundtrip_script_path, "w", encoding="utf-8") as stream:
        stream.write(_grase_roundtrip_script)
        stream.flush()
        (_grase_os.fsync(stream.fileno()) if _grase_os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
    completed = _grase_subprocess.run(
        [
            _grase_bpy.app.binary_path,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            _grase_roundtrip_script_path,
            "--",
            _GRASE_GLB_PATH,
            _grase_roundtrip_result_path,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0 or "GRASE_PROCEDURAL_ROUNDTRIP_OK" not in completed.stdout:
        detail = (completed.stdout + "\n" + completed.stderr)[-4000:]
        _grase_fail("isolated GLB round trip failed: " + detail)
    with open(_grase_roundtrip_result_path, encoding="utf-8") as stream:
        _grase_roundtrip = _grase_json.load(stream)
finally:
    for path in (_grase_roundtrip_script_path, _grase_roundtrip_result_path):
        try:
            _grase_os.unlink(path)
        except FileNotFoundError:
            pass

if _grase_roundtrip.get("mesh_count") != 1:
    _grase_fail("isolated GLB round trip did not preserve one union asset mesh")
_grase_rt_bounds = _grase_roundtrip.get("world_bounds")
if not isinstance(_grase_rt_bounds, dict):
    _grase_fail("isolated GLB round trip has no world bounds")
for key, expected in (
    ("min", _grase_minimum),
    ("max", _grase_maximum),
    ("center", _grase_center),
    ("size", _grase_size),
):
    actual = _grase_rt_bounds.get(key)
    if not isinstance(actual, list) or len(actual) != 3:
        _grase_fail("isolated GLB round trip has malformed " + key)
    for axis in range(3):
        # min/max/center move by at most the seam-retry operand offset, size by twice that
        tolerance = max(1.0e-5, abs(float(expected[axis])) * 1.0e-5) + 2.0 * float(
            _grase_union_payload.get("bounds_slack_m", 0.0) or 0.0
        )
        if abs(float(actual[axis]) - float(expected[axis])) > tolerance:
            _grase_fail("isolated GLB round trip changed world bounds")

if _grase_roundtrip.get("material_names") != _grase_union_material_names:
    _grase_fail("isolated GLB round trip changed material identities")
_grase_source_images = [
    image
    for material in _grase_material_rows
    if material["name"] in _grase_union_material_names
    for image in material["images"]
]
_grase_image_comparison = _grase_compare_image_inventories(
    _grase_source_images, _grase_roundtrip.get("images", [])
)
_grase_image_comparison.update(
    schema_version=1, target_object_id=_GRASE_TARGET_ID,
    source_reference_count=len(_grase_source_images),
)
# The caller places this small record outside rejected assets and recovery snapshots.
# Persist operands before raising: rollback may remove the candidate Blend and GLB.
_grase_diagnostic_parent = _grase_os.path.dirname(_GRASE_DIAGNOSTICS_PATH)
_grase_os.makedirs(_grase_diagnostic_parent, exist_ok=True)
with open(_GRASE_DIAGNOSTICS_PATH + ".tmp", "w", encoding="utf-8") as stream:
    _grase_json.dump(_grase_image_comparison, stream, indent=2, sort_keys=True)
    stream.flush()
    (_grase_os.fsync(stream.fileno()) if _grase_os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
_grase_os.replace(_GRASE_DIAGNOSTICS_PATH + ".tmp", _GRASE_DIAGNOSTICS_PATH)
_grase_diagnostic_fd = _grase_os.open(
    _grase_diagnostic_parent, _grase_os.O_RDONLY | getattr(_grase_os, "O_DIRECTORY", 0)
)
try:
    (_grase_os.fsync(_grase_diagnostic_fd) if _grase_os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
finally:
    _grase_os.close(_grase_diagnostic_fd)
if _grase_image_comparison["status"] != "passed":
    _grase_difference = {
        key: _grase_image_comparison[key][:3]
        for key in ("missing", "unexpected", "changed", "source_conflicts", "imported_conflicts")
        if _grase_image_comparison[key]
    }
    _grase_fail(
        "isolated GLB round trip changed embedded texture identities or sizes: "
        + _grase_json.dumps(_grase_difference, sort_keys=True)[:1600]
        + "; full comparison: " + _GRASE_DIAGNOSTICS_PATH
    )

_grase_capture = {
    "schema_version": _GRASE_CAPTURE_SCHEMA,
    "source_schema": "gpt6_blender_empty_tree_v1",
    "target_object_name": _GRASE_TARGET_NAME,
    "target_object_id": _GRASE_TARGET_ID,
    "mesh_name": _GRASE_TARGET_NAME,
    "root_object_type": "EMPTY",
    "root_matrix_world": _grase_matrix_rows(_grase_target.matrix_world),
    "world_bounds": {
        "min": _grase_minimum,
        "max": _grase_maximum,
        "center": _grase_center,
        "size": _grase_size,
    },
    "part_count": len(_grase_part_rows),
    "parts": _grase_part_rows,
    "part_name_map": [
        {
            "authored_name": row["authored_name"],
            "canonical_name": row["name"],
            "label": row["label"],
        }
        for row in _grase_part_rows
    ],
    "connected_union": {
        "method": "blender_boolean_exact_v1",
        "source_part_count": len(_grase_part_rows),
        "asset_mesh_count": 1,
        "logical_rigid_body_count": 1,
        "component_count": _grase_union_stats["component_count"],
        "manifold": _grase_union_stats["manifold"],
        "volume": _grase_union_stats["volume"],
        "vertex_count": _grase_union_stats["vertex_count"],
        "polygon_count": _grase_union_stats["polygon_count"],
        "material_slot_count": _grase_union_material_slot_count,
        "material_names": _grase_union_material_names,
        "geometry_sha256": _grase_union_geometry_sha256,
        "frame_version": _grase_union_payload["frame_version"],
        "cleanup": dict(_grase_union_payload["cleanup_evidence"]),
        "root_local_center": [
            float(value) for value in _grase_union_payload["root_local_center"]
        ],
        "canonical_to_world": [
            [float(value) for value in row]
            for row in _grase_canonical_to_world
        ],
    },
    "materials": _grase_material_rows,
    "material_sha256": _grase_hashlib.sha256(
        _grase_material_json.encode("utf-8")
    ).hexdigest(),
    "glb_path": _GRASE_GLB_PATH,
    "glb_size": _grase_glb_size,
    "glb_sha256": _grase_glb_sha256,
    "roundtrip_verified": True,
    "roundtrip": _grase_roundtrip,
    "export": {
        "format": "GLB",
        "selection_only": True,
        "apply_modifiers": True,
        "export_yup": True,
        "live_hierarchy": "parentless_empty_root_with_mesh_descendants",
        "asset_geometry": "single_connected_boolean_union_mesh",
    },
}

if _GRASE_BLEND_PATH is None:
    if not _grase_bpy.data.filepath:
        _grase_fail("no blend output path was provided and the current Blend has no path")
    _grase_blend_result = _grase_bpy.ops.wm.save_as_mainfile(
        filepath=_grase_bpy.data.filepath
    )
else:
    _grase_blend_parent = _grase_os.path.dirname(_GRASE_BLEND_PATH)
    _grase_os.makedirs(_grase_blend_parent, exist_ok=True)
    _grase_blend_result = _grase_bpy.ops.wm.save_as_mainfile(filepath=_GRASE_BLEND_PATH)
if "FINISHED" not in _grase_blend_result:
    _grase_fail("Blender did not save the candidate Blend")

_grase_capture_tmp = _GRASE_CAPTURE_PATH + ".tmp"
with open(_grase_capture_tmp, "w", encoding="utf-8") as stream:
    _grase_json.dump(_grase_capture, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    (_grase_os.fsync(stream.fileno()) if _grase_os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
_grase_os.replace(_grase_capture_tmp, _GRASE_CAPTURE_PATH)
_grase_directory_fd = _grase_os.open(
    _grase_capture_parent, _grase_os.O_RDONLY | getattr(_grase_os, "O_DIRECTORY", 0)
)
try:
    (_grase_os.fsync(_grase_directory_fd) if _grase_os.environ.get("GRASE_DURABLE_FSYNC") == "1" else None)
finally:
    _grase_os.close(_grase_directory_fd)

print(
    "GRASE_PROCEDURAL_CAPTURE_OK "
    + _GRASE_TARGET_ID
    + " "
    + _grase_glb_sha256
)
'''


def build_procedural_capture_script(
    *,
    target_object_name: str,
    target_object_id: str,
    mesh_glb_path: str | Path,
    capture_json_path: str | Path,
    blend_output_path: str | Path | None = None,
    diagnostics_json_path: str | Path | None = None,
) -> str:
    """Build trusted Blender code that canonicalizes, round-trips, captures, and saves.

    ``mesh_glb_path`` and ``capture_json_path`` must be transaction-owned absolute
    paths.  When ``blend_output_path`` is omitted Blender saves its current mainfile;
    callers handling an untrusted mutation should normally supply a transaction-local
    candidate path explicitly. ``diagnostics_json_path`` retains the image comparison;
    use a transaction path outside assets/snapshots so rollback pruning preserves it.
    Its standalone default is a sidecar beside ``capture_json_path``.
    """

    target_object_name, target_object_id = _validate_target_binding(
        target_object_name, target_object_id
    )
    glb_path = _absolute_output_path(
        mesh_glb_path, suffix=".glb", label="mesh_glb_path"
    )
    capture_path = _absolute_output_path(
        capture_json_path, suffix=".json", label="capture_json_path"
    )
    diagnostics_path = _absolute_output_path(
        capture_path + ".image_validation.json"
        if diagnostics_json_path is None
        else diagnostics_json_path,
        suffix=".json",
        label="diagnostics_json_path",
    )
    blend_path = (
        _absolute_output_path(
            blend_output_path, suffix=".blend", label="blend_output_path"
        )
        if blend_output_path is not None
        else None
    )
    outputs = [
        glb_path,
        capture_path,
        diagnostics_path,
        *([blend_path] if blend_path else []),
    ]
    if len(outputs) != len(set(outputs)):
        raise ProceduralObjectError("procedural capture output paths must be distinct")
    replacements = {
        "__GRASE_TARGET_NAME__": json.dumps(target_object_name),
        "__GRASE_TARGET_ID__": json.dumps(target_object_id),
        "__GRASE_GLB_PATH__": json.dumps(glb_path),
        "__GRASE_CAPTURE_PATH__": json.dumps(capture_path),
        "__GRASE_DIAGNOSTICS_PATH__": json.dumps(diagnostics_path),
        "__GRASE_BLEND_PATH__": json.dumps(blend_path),
        "__GRASE_CAPTURE_SCHEMA__": str(PROCEDURAL_CAPTURE_SCHEMA_VERSION),
        "__GRASE_MAX_PART_LABEL_BYTES__": str(MAX_PART_LABEL_BYTES),
        "__GRASE_AUTHORED_GEOMETRY_PATH__": json.dumps(
            str(Path(__file__).resolve().with_name("authored_root_local_geometry.py"))
        ),
    }
    script = textwrap.dedent(_CAPTURE_SCRIPT).lstrip()
    for marker, value in replacements.items():
        script = script.replace(marker, value)
    if "__GRASE_" in script:
        raise AssertionError("procedural capture script contains an unresolved marker")
    return script


def _capture_part_name_map(capture: Mapping[str, Any]) -> list[dict[str, str]]:
    ""
    parts = capture.get("parts")
    mapping = capture.get("part_name_map")
    if "part_name_map" not in capture and (
        not isinstance(parts, list)
        or not any(
            isinstance(part, Mapping) and ("label" in part or "authored_name" in part)
            for part in parts
        )
    ):
        return []
    if (
        not isinstance(parts, list)
        or not isinstance(mapping, list)
        or len(mapping) != len(parts)
    ):
        raise ProceduralObjectError(
            "capture part_name_map must match every captured part"
        )
    names: set[str] = set()
    labels: set[str] = set()
    rows: list[dict[str, str]] = []
    for part, entry in zip(parts, mapping, strict=True):
        if not isinstance(part, Mapping) or not isinstance(entry, Mapping):
            raise ProceduralObjectError(
                "capture part_name_map entries must be mappings"
            )
        if set(entry) != {"authored_name", "canonical_name", "label"}:
            raise ProceduralObjectError("capture part_name_map has invalid fields")
        for field in ("authored_name", "label"):
            value = entry[field]
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > MAX_PART_LABEL_BYTES
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in value
                )
            ):
                raise ProceduralObjectError(
                    f"capture part {field} must be a nonempty string of at most "
                    f"{MAX_PART_LABEL_BYTES} UTF-8 bytes without control characters"
                )
            if part.get(field) != value:
                raise ProceduralObjectError(
                    f"capture part_name_map {field} disagrees with parts"
                )
        name = entry["canonical_name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"_grase_[A-Za-z0-9_-]+_part_[0-9]+", name) is None
            or len(name.encode("utf-8")) > 63
            or name != part.get("name")
            or name in names
        ):
            raise ProceduralObjectError(
                "capture part_name_map has an invalid canonical name"
            )
        if entry["label"] in labels:
            raise ProceduralObjectError("capture part_name_map labels must be unique")
        names.add(name)
        labels.add(entry["label"])
        rows.append(dict(entry))
    return rows


def format_part_name_map(capture: Mapping[str, Any], *, max_parts: int = 12) -> str:
    ""
    if isinstance(max_parts, bool) or not isinstance(max_parts, int) or max_parts <= 0:
        raise ValueError("max_parts must be a positive integer")
    rows = _capture_part_name_map(capture)
    if not rows:
        return ""
    lines = ["Captured part names (authored Blender names were canonicalized):", ""]
    for row in rows[:max_parts]:
        lines.append(
            f"- label {json.dumps(row['label'], ensure_ascii=False)}: "
            f"{row['canonical_name']} "
            f"(authored {json.dumps(row['authored_name'], ensure_ascii=False)})"
        )
    if len(rows) > max_parts:
        lines.append(
            f"- {len(rows) - max_parts} additional parts omitted from this preview."
        )
    lines.extend(
        [
            "",
            "For later edits, resolve this root's children_recursive by "
            "part.get('grase_part_label'), or use the current canonical names; "
            "do not look up stale authored Blender names.",
        ]
    )
    return "\n".join(lines)


def _finite_vector(value: Any, *, size: int, label: str) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != size
        or any(isinstance(item, bool) for item in value)
    ):
        raise ProceduralObjectError(f"{label} must contain exactly {size} numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ProceduralObjectError(
            f"{label} must contain exactly {size} numbers"
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise ProceduralObjectError(f"{label} contains a non-finite number")
    return result


def _validated_capture(
    capture: Mapping[str, Any], *, expected_object_id: str
) -> dict[str, Any]:
    if not isinstance(capture, Mapping):
        raise ProceduralObjectError("capture must be a mapping")
    value = copy.deepcopy(dict(capture))
    if value.get("schema_version") != PROCEDURAL_CAPTURE_SCHEMA_VERSION:
        raise ProceduralObjectError("unsupported procedural capture schema_version")
    if value.get("source_schema") != "gpt6_blender_empty_tree_v1":
        raise ProceduralObjectError("unsupported procedural capture source_schema")
    if value.get("target_object_id") != expected_object_id:
        raise ProceduralObjectError(
            "capture target_object_id does not match category/instance"
        )
    target_name = value.get("target_object_name")
    if value.get("mesh_name") != target_name:
        raise ProceduralObjectError("capture target_object_name and mesh_name differ")
    if value.get("root_object_type") != "EMPTY":
        raise ProceduralObjectError("capture canonical root is not an EMPTY")
    _validate_target_binding(str(target_name or ""), expected_object_id)

    rows = value.get("root_matrix_world")
    if not isinstance(rows, (list, tuple)) or len(rows) != 4:
        raise ProceduralObjectError("capture root_matrix_world must be 4x4")
    matrix = [
        _finite_vector(row, size=4, label="capture root_matrix_world row")
        for row in rows
    ]
    if any(
        abs(matrix[3][index] - expected) > 1.0e-7
        for index, expected in enumerate((0, 0, 0, 1))
    ):
        raise ProceduralObjectError("capture root_matrix_world is not affine")

    bounds = value.get("world_bounds")
    if not isinstance(bounds, Mapping):
        raise ProceduralObjectError("capture has no world_bounds mapping")
    minimum = _finite_vector(bounds.get("min"), size=3, label="world_bounds.min")
    maximum = _finite_vector(bounds.get("max"), size=3, label="world_bounds.max")
    center = _finite_vector(bounds.get("center"), size=3, label="world_bounds.center")
    extent = _finite_vector(bounds.get("size"), size=3, label="world_bounds.size")
    for axis in range(3):
        derived_size = maximum[axis] - minimum[axis]
        derived_center = (maximum[axis] + minimum[axis]) * 0.5
        tolerance = max(1.0e-7, abs(derived_size) * 1.0e-7)
        if derived_size <= 1.0e-9 or abs(extent[axis] - derived_size) > tolerance:
            raise ProceduralObjectError("capture world_bounds size is inconsistent")
        if abs(center[axis] - derived_center) > tolerance:
            raise ProceduralObjectError("capture world_bounds center is inconsistent")

    if not isinstance(value.get("part_count"), int) or value["part_count"] <= 0:
        raise ProceduralObjectError("capture part_count must be positive")
    parts = value.get("parts")
    if not isinstance(parts, list) or len(parts) != value["part_count"]:
        raise ProceduralObjectError("capture parts do not match part_count")
    _capture_part_name_map(value)
    union = value.get("connected_union")
    if not isinstance(union, Mapping):
        raise ProceduralObjectError("capture has no connected_union evidence")
    if union.get("method") != "blender_boolean_exact_v1":
        raise ProceduralObjectError("capture used an unsupported union method")
    if union.get("source_part_count") != value["part_count"]:
        raise ProceduralObjectError("capture union source_part_count is inconsistent")
    if union.get("asset_mesh_count") != 1:
        raise ProceduralObjectError("capture union must contain exactly one asset mesh")
    if union.get("logical_rigid_body_count") != 1:
        raise ProceduralObjectError("capture union must bind one logical rigid body")
    if union.get("component_count") != 1 or union.get("manifold") is not True:
        raise ProceduralObjectError("capture union is not one connected manifold solid")
    try:
        union_volume = float(union.get("volume"))
    except (TypeError, ValueError) as exc:
        raise ProceduralObjectError("capture union volume is invalid") from exc
    if not math.isfinite(union_volume) or union_volume <= 1.0e-12:
        raise ProceduralObjectError("capture union volume is invalid")
    for field in ("vertex_count", "polygon_count"):
        if (
            not isinstance(union.get(field), int)
            or isinstance(union.get(field), bool)
            or union[field] <= 0
        ):
            raise ProceduralObjectError(f"capture union {field} is invalid")
    if not _SHA256_RE.fullmatch(str(union.get("geometry_sha256") or "")):
        raise ProceduralObjectError("capture union has no valid geometry_sha256")
    if not isinstance(value.get("materials"), list):
        raise ProceduralObjectError("capture materials must be a list")
    if not _SHA256_RE.fullmatch(str(value.get("material_sha256") or "")):
        raise ProceduralObjectError("capture has no valid material_sha256")
    if not _SHA256_RE.fullmatch(str(value.get("glb_sha256") or "")):
        raise ProceduralObjectError("capture has no valid glb_sha256")
    if not isinstance(value.get("glb_size"), int) or value["glb_size"] <= 20:
        raise ProceduralObjectError("capture glb_size is invalid")
    if not isinstance(value.get("glb_path"), str) or not value["glb_path"]:
        raise ProceduralObjectError("capture has no glb_path")
    if value.get("roundtrip_verified") is not True:
        raise ProceduralObjectError("capture is not roundtrip_verified")
    if not isinstance(value.get("roundtrip"), Mapping):
        raise ProceduralObjectError("capture has no roundtrip evidence")
    value["root_matrix_world"] = matrix
    value["world_bounds"] = {
        "min": minimum,
        "max": maximum,
        "center": center,
        "size": extent,
    }
    return value


def _placement_evidence(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {}
    if not isinstance(source, Mapping):
        raise ProceduralObjectError("placement evidence must be a mapping")
    return {
        key: copy.deepcopy(source[key])
        for key in _PLACEMENT_EVIDENCE_FIELDS
        if key in source
    }


def build_procedural_placement_row(
    *,
    category: str,
    instance: int,
    support: str,
    mask: str | Path | Mapping[str, Any] | None,
    capture: Mapping[str, Any],
    mesh_glb: str | Path,
    old_row: Mapping[str, Any] | None = None,
    transaction_id: int | None = None,
) -> dict[str, Any]:
    """Build a placement row bound to evaluated procedural geometry and its GLB.

    ``mask`` may be absent, the path alone, or a fresh mask-derived placement row.
    Only mask/evidence fields survive from that row and ``old_row``; stale
    reconstruction, asset-bank, normalization, OBB, and same-size metadata cannot leak
    into a new procedural revision. Runtime identity flags are preserved only from
    ``old_row``.
    """

    if not isinstance(category, str) or not category.strip() or "#" in category:
        raise ProceduralObjectError("category must be a nonempty label without '#'")
    if not isinstance(instance, int) or isinstance(instance, bool) or instance < 0:
        raise ProceduralObjectError("instance must be a nonnegative integer")
    if not isinstance(support, str) or not support.strip():
        raise ProceduralObjectError("support must be a nonempty exact graph id")
    if transaction_id is not None and (
        not isinstance(transaction_id, int)
        or isinstance(transaction_id, bool)
        or transaction_id <= 0
    ):
        raise ProceduralObjectError("transaction_id must be a positive integer")
    object_id = f"{category}#{instance}"
    capture_row = _validated_capture(capture, expected_object_id=object_id)

    row = _placement_evidence(old_row)
    if mask is None:
        pass
    elif isinstance(mask, Mapping):
        mask_identity = (
            mask.get("category"),
            mask.get("instance"),
        )
        if mask_identity != (None, None) and mask_identity != (category, instance):
            raise ProceduralObjectError(
                "mask-derived placement identity does not match category/instance"
            )
        row.update(_placement_evidence(mask))
    else:
        try:
            mask_path = os.fspath(mask)
        except TypeError as exc:
            raise ProceduralObjectError(
                "mask must be a path or placement mapping"
            ) from exc
        if not mask_path or "\x00" in mask_path:
            raise ProceduralObjectError("mask path must be nonempty and contain no NUL")
        row["mask_path"] = mask_path
    if "mask_path" in row and (
        not isinstance(row["mask_path"], str) or not row["mask_path"].strip()
    ):
        raise ProceduralObjectError("placement mask_path must be nonempty when present")

    if old_row is not None:
        if not isinstance(old_row, Mapping):
            raise ProceduralObjectError("old_row must be a placement mapping")
        old_identity = (old_row.get("category"), old_row.get("instance"))
        if old_identity != (None, None) and old_identity != (category, instance):
            raise ProceduralObjectError(
                "old placement identity does not match category/instance"
            )
        if old_row.get("runtime_added") is True:
            original_txid = old_row.get("runtime_transaction_id")
            if (
                not isinstance(original_txid, int)
                or isinstance(original_txid, bool)
                or original_txid <= 0
            ):
                raise ProceduralObjectError(
                    "runtime-added old placement has no positive transaction id"
                )
            row["runtime_added"] = True
            row["runtime_transaction_id"] = original_txid

    glb_value = os.fspath(mesh_glb)
    if not glb_value or "\x00" in glb_value:
        raise ProceduralObjectError("mesh_glb must be a nonempty path")
    if capture_row["glb_path"] != glb_value:
        raise ProceduralObjectError("mesh_glb does not match the captured GLB path")

    bounds = capture_row["world_bounds"]
    row.update(
        {
            "category": category,
            "instance": instance,
            "same_size": False,
            "center": copy.deepcopy(bounds["center"]),
            "size": copy.deepcopy(bounds["size"]),
            "obb_size": None,
            "obb_yaw": None,
            "mesh_name": capture_row["mesh_name"],
            "mesh_glb": glb_value,
            "support": support,
            "procedural_capture": {
                "schema_version": capture_row["schema_version"],
                "source_schema": capture_row["source_schema"],
                "root_object_type": capture_row["root_object_type"],
                "root_matrix_world": copy.deepcopy(capture_row["root_matrix_world"]),
                "world_bounds": copy.deepcopy(bounds),
                "part_count": capture_row["part_count"],
                **(
                    {"part_name_map": copy.deepcopy(capture_row["part_name_map"])}
                    if "part_name_map" in capture_row
                    else {}
                ),
                "connected_union": copy.deepcopy(capture_row["connected_union"]),
                "material_sha256": capture_row["material_sha256"],
                "glb_sha256": capture_row["glb_sha256"],
                "roundtrip_verified": True,
            },
        }
    )
    if transaction_id is not None:
        if old_row is None:
            row["runtime_added"] = True
            row["runtime_transaction_id"] = transaction_id
        else:
            row["runtime_mesh_revision_transaction_id"] = transaction_id
    return row


__all__ = [
    "MAX_AUTHORED_CODE_BYTES",
    "MAX_PART_LABEL_BYTES",
    "PROCEDURAL_CAPTURE_SCHEMA_VERSION",
    "ProceduralObjectError",
    "build_procedural_capture_script",
    "build_procedural_placement_row",
    "build_target_prelude",
    "format_part_name_map",
    "validate_authored_code",
]
