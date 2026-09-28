"""Build a disposable collision-only Blender scene from current evaluated geometry.

The loaded visual scene is never saved. Authored roots use exactly the harness's
Boolean-union helper; ordinary object meshes retain evaluated Blender triangles.
The separate USD is a collision input, not a replacement for visual materials.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def prepare_collision_source(request: dict, *, bpy, bmesh) -> list[dict]:
    """Switch to a disposable scene after collecting every authorized world mesh.

    Args:
        request: Driver-authenticated object identities and delivered Blend hash.
        bpy: Blender's runtime module.
        bmesh: Blender's evaluated mesh module.

    Returns:
        Per-object collision-source geometry/provenance records.

    Raises:
        RuntimeError: An identity, geometry, or delivered-input binding is invalid.
    """
    if (
        request.get("schema_version") != 1
        or request.get("policy") != "current_pose_authored_union_v1"
    ):
        raise RuntimeError("Unsupported collision export request")
    names = request.get("expected_objects")
    authored = request.get("authored_empty_ids")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not name.startswith("obj_") for name in names
        )
        or len(set(names)) != len(names)
        or not isinstance(authored, dict)
        or not set(authored).issubset(names)
    ):
        raise RuntimeError("Invalid collision export identities")
    with Path(bpy.data.filepath).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != request.get("blend_sha256"):
        raise RuntimeError("Loaded Blender file differs from collision export request")

    helper = None
    if authored:
        source = (
            Path(__file__).resolve().parents[1]
            / "lib/tools/blender/authored_root_local_geometry.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_grase_usd_authored_union", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load shared authored geometry helper: {source!s}"
            )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

    source_scene = bpy.context.scene
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    snapshots, records = [], []
    for name in names:
        root = source_scene.objects.get(name)
        if root is None:
            raise RuntimeError(f"Missing collision-source object: {name!r}")
        descendants = list(root.children_recursive)
        parts = sorted(
            [obj for obj in [root, *descendants] if obj.type == "MESH"],
            key=lambda obj: obj.name,
        )
        method = "evaluated_blender_triangles"
        part_meshes: list[tuple[np.ndarray, np.ndarray]] = []
        if name in authored:
            graph_id = authored[name]
            if (
                root.type != "EMPTY"
                or root.parent is not None
                or not isinstance(graph_id, str)
                or root.get("grase_graph_id") != graph_id
                or not parts
                or len(parts) != len(descendants)
                or any(obj.get("grase_graph_id") not in (None, "") for obj in parts)
                or [
                    obj
                    for obj in source_scene.objects
                    if obj.get("grase_graph_id") == graph_id
                ]
                != [root]
            ):
                raise RuntimeError(f"Invalid authored collision root: {name!r}")
            with helper.authored_union(root, parts, bpy=bpy, bmesh=bmesh) as payload:
                vertices, faces = helper.register_arrays(payload)
                part_meshes = helper.part_arrays(payload)
            method = "shared_authored_boolean_union"
        else:
            if root.type == "EMPTY" or root.get("grase_graph_id"):
                raise RuntimeError(f"Unauthenticated authored collision root: {name!r}")
            vertices_list, faces_list, offset = [], [], 0
            for part in parts:
                evaluated = part.evaluated_get(deps)
                mesh = evaluated.to_mesh()
                try:
                    mesh.calc_loop_triangles()
                    points = np.asarray(
                        [tuple(v.co) for v in mesh.vertices], dtype=np.float64
                    )
                    matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
                    vertices_list.append(points @ matrix[:3, :3].T + matrix[:3, 3])
                    faces_list.append(
                        np.asarray(
                            [tuple(t.vertices) for t in mesh.loop_triangles],
                            dtype=np.int64,
                        )
                        + offset
                    )
                    offset += len(points)
                finally:
                    evaluated.to_mesh_clear()
            if not vertices_list:
                raise RuntimeError(f"Collision object has no mesh: {name!r}")
            vertices, faces = np.vstack(vertices_list), np.vstack(faces_list)
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise RuntimeError(f"Collision object has invalid geometry: {name!r}")
        snapshots.append((name, vertices, faces, part_meshes))
        records.append(
            {
                "pipeline_name": name,
                "method": method,
                "source_part_count": len(parts),
                "vertex_count": len(vertices),
                "triangle_count": len(faces),
                "world_min": vertices.min(0).tolist(),
                "world_max": vertices.max(0).tolist(),
            }
        )

    # Snapshot every body before changing any scene/name; dependencies remain intact
    # throughout evaluation. Global Blender names must be released for matching USD
    # identities, but the original hierarchy and mesh datablocks are not destroyed.
    collision_scene = bpy.data.scenes.new("grase_collision_source")
    collision_scene.unit_settings.system = source_scene.unit_settings.system
    collision_scene.unit_settings.scale_length = source_scene.unit_settings.scale_length
    for name, _vertices, _faces, _parts in snapshots:
        source_scene.objects[name].name = "_grase_export_visual_" + name
    for (name, vertices, faces, part_meshes), record in zip(snapshots, records):
        root = bpy.data.objects.new(name, None)
        collision_scene.collection.objects.link(root)
        root.location = ((vertices.min(0) + vertices.max(0)) / 2).tolist()
        # Subtract the ACTUAL float32 Blender origin so large world translations do
        # not round the small local geometry. USD composes these in double precision.
        origin = np.asarray(root.location, dtype=np.float64)
        local = vertices - origin
        mesh = bpy.data.meshes.new(name + "_collision_source")
        mesh.from_pydata(local.tolist(), [], faces.tolist())
        child = bpy.data.objects.new(name + "_collision_source", mesh)
        collision_scene.collection.objects.link(child)
        child.parent = root
        stored = (
            np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float64) + origin
        )
        error = float(np.max(np.linalg.norm(stored - vertices, axis=1)))
        tolerance = max(1e-6, float(np.max(np.ptp(vertices, axis=0))) * 1e-6)
        if error > tolerance:
            raise RuntimeError(
                f"Collision-source precision loss for {name!r}: {error} > {tolerance}"
            )
        record["storage_error_m"] = error
        # Authored objects also publish each evaluated part as its own child mesh
        # (``<name>_collision_part_NNN``): usd_dump_objects keeps them out of the union
        # body and the cook hulls them one by one (collision.decompose_authored_parts).
        for index, (part_vertices, part_faces) in enumerate(part_meshes):
            part_name = f"{name}_collision_part_{index:03d}"
            part_mesh = bpy.data.meshes.new(part_name)
            part_mesh.from_pydata(
                (part_vertices - origin).tolist(), [], part_faces.tolist()
            )
            part_child = bpy.data.objects.new(part_name, part_mesh)
            collision_scene.collection.objects.link(part_child)
            part_child.parent = root
    bpy.context.window.scene = collision_scene
    bpy.context.view_layer.update()
    return records
