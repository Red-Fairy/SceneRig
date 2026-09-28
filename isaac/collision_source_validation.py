"""Read-only binding of a paired visual/collision USD export before collider cooking."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

try:  # Tightly coupled Isaac scripts also run directly in the standalone USD venv.
    from .export_identity import build_identity_manifest, load_identity_manifest
except ImportError:
    from export_identity import build_identity_manifest, load_identity_manifest

# Authored part children in the collision USD (blender_collision_source /
# usd_dump_objects.PART_MARKER): cook inputs beside the union body, never the body.
PART_MARKER = "_collision_part_"

_POLICY = "current_pose_authored_union_v1"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read collision export contract {str(path)!r}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Collision export contract must be an object: {str(path)!r}")
    return value


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError(
            f"Cannot hash collision export artifact {str(path)!r}: {exc}"
        ) from exc


def _stage_geometry(
    path: Path, names: list[str], *, collision: bool
) -> tuple[dict, dict]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open paired USD {str(path)!r}")
    if (
        UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError(f"Paired USD must use Z-up meters: {str(path)!r}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"Paired USD has no default prim: {str(path)!r}")
    roots = [
        (prim.GetName(), str(prim.GetPath()))
        for prim in root.GetChildren()
        if prim.GetName().startswith("obj_")
    ]
    manifest = build_identity_manifest(names, roots)
    identities = {row["pipeline_name"]: row for row in manifest["objects"]}
    mapped = {row["prim_path"] for row in identities.values()}
    if mapped != {path for _name, path in roots}:
        raise ValueError(f"Paired USD has unmatched object roots: {str(path)!r}")
    expected_usd_names = {row["usd_name"] for row in identities.values()}
    cache = UsdGeom.XformCache()
    geometry = {}
    for name, record in identities.items():
        body = stage.GetPrimAtPath(record["prim_path"])
        minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
        vertex_count = triangle_count = part_count = 0
        for prim in Usd.PrimRange(body):
            if prim != body and prim.GetName() in expected_usd_names:
                raise ValueError(
                    f"Paired USD has nested object root {str(prim.GetPath())!r}"
                )
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                raise ValueError(f"Paired USD object {name!r} has malformed points")
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            world = points @ matrix[:3, :3].T + matrix[:3, 3]
            if not np.isfinite(world).all():
                raise ValueError(f"Paired USD object {name!r} has nonfinite geometry")
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            if (
                counts.ndim != 1
                or counts.dtype.kind not in "iu"
                or indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or not len(counts)
                or np.any(counts < 3)
                or np.sum(counts, dtype=np.int64) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(points))
            ):
                raise ValueError(f"Paired USD object {name!r} has malformed topology")
            if collision and (np.any(counts != 3) or mesh.GetHoleIndicesAttr().Get()):
                raise ValueError(
                    f"Collision source {name!r} must contain explicit triangles without holes"
                )
            if collision and PART_MARKER in prim.GetName():
                # authored part child: cook input, not the union body. Counted only —
                # the seam retry offsets the union from its parts by up to 0.06 mm,
                # far beyond the 1e-6 record tolerance below.
                part_count += 1
                continue
            minimum = np.minimum(minimum, world.min(0))
            maximum = np.maximum(maximum, world.max(0))
            vertex_count += len(points)
            triangle_count += int(np.sum(counts - 2))
        if not vertex_count or not triangle_count:
            raise ValueError(f"Paired USD object {name!r} has no mesh geometry")
        geometry[name] = (minimum, maximum, vertex_count, triangle_count, part_count)
    return identities, geometry


def validate_collision_source(exp: Path) -> dict:
    """Validate hashes, identities, methods, counts, and world bounds without writes.

    Different visual/union topology is intentional; the stages must nevertheless
    refer to the same objects in the same current world locations.

    Raises:
        ValueError: A file, hash, identity, geometry, or provenance binding is invalid.
    """
    scene = Path(exp).resolve() / "scene"
    folder = scene / "isaac"
    request = _read_json(folder / "collision_export_request.json")
    manifest = _read_json(folder / "scene_collision.json")
    for label, record in (("request", request), ("manifest", manifest)):
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("policy") != _POLICY
        ):
            raise ValueError(f"Unsupported collision export {label} schema/policy")
    names = request.get("expected_objects")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not name.startswith("obj_") for name in names
        )
        or names != sorted(set(names))
    ):
        raise ValueError(
            "Collision export expected_objects must be sorted unique object names"
        )
    authored = request.get("authored_empty_ids")
    if (
        not isinstance(authored, dict)
        or not set(authored).issubset(names)
        or any(not isinstance(value, str) or not value for value in authored.values())
        or len(set(authored.values())) != len(authored)
    ):
        raise ValueError("Collision export authored_empty_ids is invalid")

    expected_hashes = (
        (scene / "final/final.blend", request.get("blend_sha256"), "request Blend"),
        (
            scene / "placement.json",
            request.get("placement_sha256"),
            "request placement",
        ),
        (folder / "scene_visual.usdc", manifest.get("visual_sha256"), "visual USD"),
        (
            folder / "scene_collision.usdc",
            manifest.get("collision_sha256"),
            "collision USD",
        ),
    )
    for path, expected, label in expected_hashes:
        if _sha256(path) != expected:
            raise ValueError(f"Collision export {label} SHA-256 mismatch")
    if manifest.get("source_blend_sha256") != request["blend_sha256"]:
        raise ValueError("Collision manifest source Blend SHA-256 mismatch")
    placement = _read_json(scene / "placement.json").get("objects")
    if (
        not isinstance(placement, list)
        or any(
            not isinstance(row, dict) or not isinstance(row.get("mesh_name"), str)
            for row in placement
        )
        or sorted(row["mesh_name"] for row in placement) != names
    ):
        raise ValueError("Collision export request and active placement names differ")

    records = manifest.get("objects")
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("pipeline_name"), str)
        for row in records
    ):
        raise ValueError("Collision manifest object records are invalid")
    if sorted(row["pipeline_name"] for row in records) != names:
        raise ValueError("Collision manifest and request object names differ")
    declared_identity = load_identity_manifest(folder / "object_identity.json")
    declared = {row["pipeline_name"]: row for row in declared_identity["objects"]}
    visual_identity, visual = _stage_geometry(
        folder / "scene_visual.usdc", names, collision=False
    )
    collision_identity, collision = _stage_geometry(
        folder / "scene_collision.usdc", names, collision=True
    )
    if declared != visual_identity or declared != collision_identity:
        raise ValueError("Visual, collision, and recorded object identities differ")

    for record in records:
        name = record["pipeline_name"]
        expected_method = (
            "shared_authored_boolean_union"
            if name in authored
            else "evaluated_blender_triangles"
        )
        if record.get("method") != expected_method:
            raise ValueError(
                f"Collision source method disagrees with authored identity: {name!r}"
            )
        low, high, vertices, triangles, part_count = collision[name]
        allowed = {0, record.get("source_part_count")} if name in authored else {0}
        if part_count not in allowed:
            raise ValueError(
                f"Collision source {name!r} authored part children disagree with "
                "source_part_count"
            )
        for field, actual in (
            ("vertex_count", vertices),
            ("triangle_count", triangles),
        ):
            if type(record.get(field)) is not int or record[field] != actual:
                raise ValueError(
                    f"Collision source {name!r} {field} differs from actual USD"
                )
        if (
            type(record.get("source_part_count")) is not int
            or record["source_part_count"] <= 0
        ):
            raise ValueError(f"Collision source {name!r} has invalid source_part_count")
        span = float(np.max(high - low))
        tolerance = max(1e-4, span * 1e-5)
        visual_low, visual_high, _vertices, _triangles, _parts = visual[name]
        if (
            np.max(np.abs(np.concatenate((visual_low - low, visual_high - high))))
            > tolerance
        ):
            raise ValueError(f"Visual/collision world bounds disagree for {name!r}")
        precision_tolerance = max(1e-6, span * 1e-6)
        error = record.get("storage_error_m")
        if (
            type(error) not in (int, float)
            or not np.isfinite(error)
            or not 0 <= error <= precision_tolerance
        ):
            raise ValueError(f"Collision source {name!r} has invalid storage_error_m")
        for field, actual in (("world_min", low), ("world_max", high)):
            try:
                bound = np.asarray(record.get(field), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Collision source {name!r} has invalid {field}"
                ) from exc
            if (
                bound.shape != (3,)
                or not np.isfinite(bound).all()
                or np.max(np.abs(bound - actual)) > precision_tolerance
            ):
                raise ValueError(
                    f"Collision source {name!r} {field} differs from actual USD"
                )
    return {"policy": _POLICY, "objects": names, "object_count": len(names)}


if __name__ == "__main__":
    report = validate_collision_source(Path(sys.argv[1]))
    # Machine-readable success protocol consumed by blend_to_isaac.run.
    sys.stdout.write(f"ISAAC_COLLISION_VALIDATED objects={report['object_count']}\n")
