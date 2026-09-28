"""UNADMITTED v7: decoded root-local normal carrier through native Boolean interpolation.

Embeddable source: no project imports, paths, file/process IO, caches, modifier
whitelist, or operators. Callers retain their existing identity/transaction rules.
Every call evaluates current source geometry before introducing temporary objects.
The Blender/BMesh path and capture adapter require separate real-scene/material/
non-BEVEL/GLB roundtrip admission; pure tests do not certify generic correctness.
"""

import hashlib
import json
from contextlib import contextmanager
from fractions import Fraction

import numpy as np

FRAME_VERSION = "authored_root_local_centered_proposal_v7"


def require(condition, message):
    if not condition:
        raise RuntimeError("authored root-local proposal: " + message)


def affine(value, *, positive=True):
    value = np.asarray(value, dtype=np.float64)
    require(value.shape == (4, 4) and np.isfinite(value).all(), "invalid matrix")
    require(np.array_equal(value[3], [0, 0, 0, 1]), "non-affine matrix")
    determinant = float(np.linalg.det(value[:3, :3]))
    require(np.isfinite(determinant) and determinant != 0, "singular matrix")
    if positive:
        require(determinant > 0, "mirrored output frame")
    return value.copy()


def points_in_frame(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points), "invalid points")
    require(np.isfinite(points).all(), "non-finite points")
    matrix = affine(matrix)
    result = points @ matrix[:3, :3].T + matrix[:3, 3]
    require(np.isfinite(result).all(), "non-finite transformed points")
    return result


def select_part_mapping(
    root_world,
    evaluated_world,
    *,
    direct_object,
    raw_mapping_eligible=True,
    parent_inverse=None,
    basis=None,
):
    """Stable raw fast path, evaluated-snapshot fallback; neither is a cache key."""
    root_world, evaluated_world = affine(root_world), affine(evaluated_world)
    reason = "not_direct_object_parent"
    if direct_object and not raw_mapping_eligible:
        reason = "evaluated_transform_dependency"
    if direct_object and raw_mapping_eligible:
        parent_inverse = affine(parent_inverse, positive=False)
        basis = affine(basis, positive=False)
        raw = parent_inverse @ basis
        reconstructed = root_world @ raw
        rounding = (
            8
            * np.finfo(np.float32).eps
            * np.maximum(1, np.abs(root_world) @ np.abs(parent_inverse) @ np.abs(basis))
        )
        error = float(np.max(np.abs(reconstructed - evaluated_world)))
        if np.all(np.abs(reconstructed - evaluated_world) <= rounding):
            return affine(raw), {
                "mode": "raw_parent_inverse_basis",
                "world_matrix_max_error": error,
            }
        reason = "raw_mapping_disagrees_with_current_evaluation"
    # Other currently accepted transform semantics remain evaluated, not banned.
    # This path deliberately does NOT promise pose-invariant topology.
    local = affine(np.linalg.solve(root_world, evaluated_world))
    reconstructed = root_world @ local
    rounding = (
        32
        * np.finfo(np.float64).eps
        * np.maximum(1, np.abs(root_world) @ np.abs(local) + np.abs(evaluated_world))
    )
    require(
        np.all(np.abs(reconstructed - evaluated_world) <= rounding),
        "evaluated fallback is numerically unresolved",
    )
    return local, {
        "mode": "evaluated_world_snapshot_solve",
        "reason": reason,
        "world_matrix_max_error": float(
            np.max(np.abs(reconstructed - evaluated_world))
        ),
        "pose_invariant_geometry_claimed": False,
    }


def centered_parts(parts):
    require(bool(parts), "empty part list")
    lo = np.min([part.min(0) for part in parts], axis=0)
    hi = np.max([part.max(0) for part in parts], axis=0)
    require(
        np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo),
        "invalid volumetric bounds",
    )
    center = lo + 0.5 * (hi - lo)
    return center, [part - center for part in parts]


def transform_normals(normals, local_linear):
    normals = np.asarray(normals, dtype=np.float64)
    require(
        normals.ndim == 2 and normals.shape[1] == 3 and np.isfinite(normals).all(),
        "invalid normals",
    )
    # Row normals use inverse(linear), equivalent to inverse-transpose columns.
    result = np.linalg.solve(np.asarray(local_linear, dtype=np.float64).T, normals.T).T
    lengths = np.linalg.norm(result, axis=1)
    require(
        np.isfinite(lengths).all() and np.all(lengths > 0),
        "invalid transformed normals",
    )
    return result / lengths[:, None]


def triangle_stats(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    require(
        vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(),
        "invalid triangle vertices",
    )
    require(
        faces.ndim == 2
        and faces.shape[1] == 3
        and len(faces)
        and np.issubdtype(faces.dtype, np.integer),
        "invalid triangles",
    )
    require(
        faces.min() >= 0 and faces.max() < len(vertices), "invalid triangle indices"
    )
    tri = vertices[faces]
    areas2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    require(
        np.isfinite(areas2).all() and np.all(areas2 > 0),
        "zero/non-finite triangle area",
    )
    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "minimum_double_area": float(areas2.min()),
    }


def _persistent_material(material, *, bpy):
    """Keep evaluated slot selection, but never retain a COW ID in a Main mesh."""
    if material is None:
        return None
    original = material.original
    require(
        original is not None and not original.is_evaluated,
        "material has no non-evaluated original ID",
    )
    pointer = original.as_pointer()
    require(
        pointer != 0
        and any(candidate.as_pointer() == pointer for candidate in bpy.data.materials),
        "material original is not a stable Main material ID",
    )
    return original


def _attribute_schema(mesh):
    return sorted(
        (attribute.name, attribute.domain, attribute.data_type)
        for attribute in mesh.attributes
    )


def _preservation_evidence(mesh, expected_schema, material_names, *, custom_normals):
    """Fail closed on detectable loss; does not prove interpolated attribute values."""
    schema = _attribute_schema(mesh)
    require(
        set(expected_schema).issubset(set(schema)),
        "union lost or changed an evaluated source data layer",
    )
    actual_materials = [
        material.name if material else "" for material in mesh.materials
    ]
    require(
        actual_materials == material_names, "union changed the INDEX material palette"
    )
    require(
        all(
            0 <= polygon.material_index < len(material_names)
            for polygon in mesh.polygons
        ),
        "union has an invalid material assignment",
    )
    require(
        not custom_normals or mesh.has_custom_normals,
        "union dropped evaluated custom normals",
    )
    return {
        "attribute_schema": schema,
        "material_names": actual_materials,
        "custom_normals": bool(mesh.has_custom_normals),
        "attribute_value_roundtrip_validated": False,
    }


def _solid_stats(mesh, bmesh, root_determinant, *, connected, part_name=None):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        require(
            bm.verts and bm.faces and all(edge.is_manifold for edge in bm.edges),
            "mesh is not a closed manifold"
            + (f"; authored part {part_name!r}" if part_name else ""),
        )
        require(
            all(np.isfinite(tuple(vertex.co)).all() for vertex in bm.verts),
            "non-finite mesh",
        )
        unseen, components = set(bm.faces), 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                for edge in stack.pop().edges:
                    for face in edge.link_faces:
                        if face in unseen:
                            unseen.remove(face)
                            stack.append(face)
        local_volume = abs(float(bm.calc_volume(signed=True)))
        world_volume = local_volume * abs(float(root_determinant))
        require(
            np.isfinite(world_volume) and world_volume > 1e-12,
            "no positive finite world volume",
        )
        require(not connected or components == 1, "union is disconnected")
        return {
            "local_volume": local_volume,
            "world_volume": world_volume,
            "components": components,
        }
    finally:
        bm.free()


def cleanup_plan(vertices, edges, root_linear):
    """Map only bounded groups of short mesh edges to existing vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    root_linear = np.asarray(root_linear, dtype=np.float64)
    require(
        vertices.ndim == 2
        and vertices.shape[1] == 3
        and len(vertices)
        and np.isfinite(vertices).all(),
        "invalid cleanup vertices",
    )
    require(
        root_linear.shape == (3, 3) and np.isfinite(root_linear).all(),
        "invalid cleanup root transform",
    )
    extent = float(np.ptp(vertices, axis=0).max())
    stretch = float(np.linalg.svd(root_linear, compute_uv=False).max())
    require(extent > 0 and stretch > 0, "invalid cleanup scale")
    tolerance = min(float(np.finfo(np.float32).eps) * extent, 1e-6 / stretch)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    short_edges = edges[lengths <= tolerance]
    neighbors = {}
    for a, b in short_edges:
        a, b = int(a), int(b)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    remaining, targets = set(neighbors), {}
    while remaining:
        seed = min(remaining)
        group, stack = {seed}, [seed]
        remaining.remove(seed)
        while stack:
            for neighbor in neighbors[stack.pop()] & remaining:
                remaining.remove(neighbor)
                group.add(neighbor)
                stack.append(neighbor)
        indices = sorted(group)
        points = vertices[indices]
        for index in range(len(points) - 1):
            diameter = float(
                np.linalg.norm(points[index + 1 :] - points[index], axis=1).max()
            )
            require(
                diameter <= tolerance,
                f"Boolean seam group at vertex {seed} spans {diameter:.9g} local "
                f"units, exceeding cleanup tolerance {tolerance:.9g}; "
                "repair the authored thin geometry (no weld applied)",
            )
        targets.update((index, seed) for index in indices[1:])
    displacement = (
        vertices[list(targets.values())] - vertices[list(targets)]
        if targets
        else np.zeros((1, 3))
    )
    return targets, {
        "version": "short_edge_weld_v1",
        "tolerance_local": tolerance,
        "tolerance_world_bound_m": tolerance * stretch,
        "candidate_edge_count": len(short_edges),
        "merged_vertex_count": len(targets),
        "max_vertex_displacement_local": float(
            np.linalg.norm(displacement, axis=1).max()
        ),
        "max_vertex_displacement_m": float(
            np.linalg.norm(displacement @ root_linear.T, axis=1).max()
        ),
    }


def _cleanup_union_seams(mesh, bmesh, root_linear):
    """Weld numerical seams on the disposable final union, before triangulation."""
    vertices = np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float64)
    targets, evidence = cleanup_plan(
        vertices, [tuple(e.vertices) for e in mesh.edges], root_linear
    )
    evidence["collapsed_face_count"] = 0
    if not targets:
        return evidence
    schema = _attribute_schema(mesh)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        original_vertices = list(bm.verts)
        indices = {vertex: index for index, vertex in enumerate(original_vertices)}
        vertex_selection = [bool(vertex.select) for vertex in mesh.vertices]
        edge_selection = {
            edge: bool(mesh.edges[i].select) for i, edge in enumerate(bm.edges)
        }
        face_selection = {
            face: bool(mesh.polygons[i].select) for i, face in enumerate(bm.faces)
        }
        original_faces = list(bm.faces)
        surviving, face_keys = [], set()
        for face in original_faces:
            mapped = [targets.get(indices[v], indices[v]) for v in face.verts]
            reduced = [
                value for i, value in enumerate(mapped) if value != mapped[i - 1]
            ]
            if len(set(reduced)) < 3:
                continue
            require(
                len(set(reduced)) == len(reduced),
                "Boolean seam weld would pinch a polygon; repair the authored intersections",
            )
            key = tuple(sorted(reduced))
            require(
                key not in face_keys,
                "Boolean seam weld would duplicate a polygon; repair the authored intersections",
            )
            face_keys.add(key)
            surviving.append(face)
            # Welding may retain either corner beside a collapsed edge. Copy the
            # chosen existing corner's data first, keeping UVs/colors/normals
            # independent of BMesh's edge direction and distinct between faces.
            corners = {}
            for loop in sorted(face.loops, key=lambda loop: indices[loop.vert]):
                index = indices[loop.vert]
                corners.setdefault(targets.get(index, index), loop)
            for loop in face.loops:
                index = indices[loop.vert]
                source = corners[targets.get(index, index)]
                if source is not loop:
                    loop.copy_from(source)
        bmesh.ops.weld_verts(
            bm,
            targetmap={
                original_vertices[a]: original_vertices[b] for a, b in targets.items()
            },
        )
        require(
            len(bm.verts) == len(original_vertices) - len(targets)
            and len(bm.faces) == len(surviving)
            and all(face.is_valid for face in surviving),
            "Boolean seam weld changed geometry beyond its collapsed vertices/faces",
        )
        require(
            all(
                v in indices and tuple(v.co) == tuple(vertices[indices[v]])
                for v in bm.verts
            ),
            "Boolean seam weld moved a representative vertex",
        )
        bm.normal_update()
        # Blender can omit all-false/zero built-in layers during BMesh conversion.
        # Preserve their presence and the surviving elements' values explicitly.
        builtin_values = {
            ".select_vert": (
                "POINT",
                "BOOLEAN",
                [vertex_selection[indices[v]] for v in bm.verts],
            ),
            ".select_edge": (
                "EDGE",
                "BOOLEAN",
                [edge_selection.get(e, False) for e in bm.edges],
            ),
            ".select_poly": ("FACE", "BOOLEAN", [face_selection[f] for f in bm.faces]),
            "material_index": ("FACE", "INT", [f.material_index for f in bm.faces]),
            "sharp_face": ("FACE", "BOOLEAN", [not f.smooth for f in bm.faces]),
        }
        bm.to_mesh(mesh)
    finally:
        bm.free()
    for name, (domain, kind, values) in builtin_values.items():
        attribute = mesh.attributes.get(name)
        if (name, domain, kind) not in schema:
            if attribute is not None:
                mesh.attributes.remove(attribute)
            continue
        if attribute is None:
            attribute = mesh.attributes.new(name, kind, domain)
        require(len(attribute.data) == len(values), "cleanup attribute count changed")
        for item, value in zip(attribute.data, values):
            item.value = value
    mesh.update()
    require(
        set(schema).issubset(set(_attribute_schema(mesh))),
        "Boolean seam cleanup lost or changed a source data layer",
    )
    evidence["collapsed_face_count"] = len(original_faces) - len(mesh.polygons)
    return evidence


def _retriangulation_altitudes(vertices, faces):
    points = vertices[faces]
    longest = np.linalg.norm(points - np.roll(points, 1, axis=1), axis=2).max(axis=1)
    area = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    return np.divide(area, longest, out=np.zeros_like(area), where=longest > 0)


def _coplanar_flip(pair):
    """Return a directed quad and alternate triangles, without moving vertices."""
    shared = set(pair[0]) & set(pair[1])
    if len(shared) != 2 or len(set(pair.ravel())) != 4:
        return None
    boundary = {}
    for face in pair:
        for a, b in zip(face, np.roll(face, -1)):
            if {a, b} != shared:
                if int(a) in boundary:
                    return None
                boundary[int(a)] = int(b)
    if len(boundary) != 4:
        return None
    quad = [min(boundary)]
    for _ in range(3):
        if quad[-1] not in boundary:
            return None
        quad.append(boundary[quad[-1]])
    if len(set(quad)) != 4 or boundary.get(quad[-1]) != quad[0]:
        return None
    a, b, c, d = quad
    if shared == {a, c}:
        return quad, np.array([[a, b, d], [b, c, d]]), tuple(sorted((b, d)))
    if shared == {b, d}:
        return quad, np.array([[a, b, c], [a, c, d]]), tuple(sorted((a, c)))
    return None


def retriangulation_plan(
    vertices, triangles, polygon_ids, root_linear, *, accept_flip=None
):
    """Flip only improving diagonals inside exactly coplanar, convex source faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3).copy()
    polygon_ids = np.asarray(polygon_ids, dtype=np.int64)
    stretch = float(
        np.linalg.svd(np.asarray(root_linear, dtype=np.float64), compute_uv=False).max()
    )
    tolerance = min(
        float(np.finfo(np.float32).eps) * float(np.ptp(vertices, axis=0).max()),
        1e-6 / stretch,
    )
    initial_count = int(
        (_retriangulation_altitudes(vertices, faces) <= tolerance).sum()
    )
    flips = []
    for _ in range(initial_count):
        owners = {}
        for index, face in enumerate(faces):
            for a, b in zip(face, np.roll(face, -1)):
                owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
        bad = set(
            np.flatnonzero(_retriangulation_altitudes(vertices, faces) <= tolerance)
        )
        candidates = sorted(
            {
                tuple(sorted(pair))
                for pair in owners.values()
                if len(pair) == 2
                and bad.intersection(pair)
                and polygon_ids[pair[0]] == polygon_ids[pair[1]]
            }
        )
        for i, j in candidates:
            pair = faces[[i, j]]
            proposal = _coplanar_flip(pair)
            if proposal is None:
                continue
            quad, replacement, new_edge = proposal
            if new_edge in owners:
                continue
            old_alt = _retriangulation_altitudes(vertices, pair)
            new_alt = _retriangulation_altitudes(vertices, replacement)
            if (
                new_alt.min() <= old_alt.min()
                or (new_alt <= tolerance).sum() >= (old_alt <= tolerance).sum()
            ):
                continue
            # Exact predicates touch only the four candidate vertices.
            points = {
                k: np.array([Fraction(float(v)) for v in vertices[k]], dtype=object)
                for k in quad
            }
            first = [points[int(k)] for k in pair[0]]
            normal = np.cross(first[1] - first[0], first[2] - first[0])
            if any(np.dot(normal, points[k] - first[0]) != 0 for k in quad):
                continue
            if any(
                np.dot(
                    np.cross(
                        points[quad[(k + 1) % 4]] - points[quad[k]],
                        points[quad[(k + 2) % 4]] - points[quad[(k + 1) % 4]],
                    ),
                    normal,
                )
                <= 0
                for k in range(4)
            ):
                continue
            polygon_id = int(polygon_ids[i])
            if accept_flip is not None and not accept_flip(polygon_id, tuple(quad)):
                continue
            faces[[i, j]] = replacement
            flips.append(
                {
                    "faces": [i, j],
                    "polygon_id": polygon_id,
                    "quad_boundary": quad,
                    "old_triangles": pair.tolist(),
                    "new_triangles": replacement.tolist(),
                    "old_diagonal": sorted(int(k) for k in set(pair[0]) & set(pair[1])),
                    "new_diagonal": list(new_edge),
                    "old_min_altitude_local": float(old_alt.min()),
                    "new_min_altitude_local": float(new_alt.min()),
                }
            )
            break
        else:
            break
    return faces, {
        "version": "coplanar_internal_diagonal_v1",
        "flipped_edge_count": len(flips),
        "candidate_sliver_count": initial_count,
        "remaining_sliver_count": int(
            (_retriangulation_altitudes(vertices, faces) <= tolerance).sum()
        ),
        "tolerance_local": tolerance,
        "tolerance_world_bound_m": tolerance * stretch,
        "max_vertex_displacement_local": 0.0,
        "max_vertex_displacement_m": 0.0,
        "flips": flips,
    }


def _refine_triangle_plans(mesh, plans, root_linear):
    """Improve generated diagonals without crossing a source polygon or data seam."""
    vertices = np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float64)
    polygon_ids = [index for index, triangles in plans.items() for _ in triangles]
    triangles = [triangle for group in plans.values() for triangle in group]
    fields = [
        attribute
        for attribute in mesh.attributes
        if attribute.domain in {"POINT", "CORNER"}
        and attribute.name
        not in {
            "position",
            ".corner_vert",
            ".corner_edge",
            ".select_vert",
            "custom_normal",
        }
    ]
    continuous = {
        "FLOAT": "value",
        "FLOAT2": "vector",
        "FLOAT_VECTOR": "vector",
        "FLOAT_COLOR": "color",
        "BYTE_COLOR": "color",
    }
    discrete = {"INT": "value", "INT8": "value", "BOOLEAN": "value"}
    vetoes, max_roundoff = 0, 0.0

    def accept_flip(polygon_index, quad):
        nonlocal vetoes, max_roundoff
        quad = list(quad)
        polygon = mesh.polygons[polygon_index]
        loops = {
            mesh.loops[index].vertex_index: index for index in polygon.loop_indices
        }
        if len(loops) != len(polygon.vertices):
            vetoes += 1
            return False
        corner_indices = [loops[index] for index in quad]
        points = vertices[quad]
        triples = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        samples = points[triples]
        areas = np.linalg.norm(
            np.cross(samples[:, 1] - samples[:, 0], samples[:, 2] - samples[:, 0]),
            axis=1,
        )
        opposite = int(areas.argmax())
        basis = triples[opposite]
        origin = points[basis[0]]
        weights2 = np.linalg.lstsq(
            (points[basis[1:]] - origin).T, points[opposite] - origin, rcond=None
        )[0]
        weights = np.r_[1 - weights2.sum(), weights2]
        candidate_roundoff = 0.0

        def affine_values(values):
            nonlocal candidate_roundoff
            values = np.asarray(values, dtype=np.float64).reshape(4, -1)
            if not np.isfinite(values).all():
                return False
            residual = np.abs(weights @ values[basis] - values[opposite])
            scale = np.max(np.abs(values), axis=0)
            unit = float(np.finfo(np.float32).eps) * scale
            if np.any(residual > 4 * unit):
                return False
            ratio = np.divide(
                residual, unit, out=np.zeros_like(residual), where=unit > 0
            )
            candidate_roundoff = max(candidate_roundoff, float(ratio.max()))
            return True

        for attribute in fields:
            indices = quad if attribute.domain == "POINT" else corner_indices
            property_name = continuous.get(attribute.data_type) or discrete.get(
                attribute.data_type
            )
            if property_name is None:
                vetoes += 1
                return False
            values = [
                getattr(attribute.data[index], property_name) for index in indices
            ]
            compatible = (
                affine_values(values)
                if attribute.data_type in continuous
                else all(value == values[0] for value in values[1:])
            )
            if not compatible:
                vetoes += 1
                return False
        if not affine_values(
            [tuple(mesh.corner_normals[index].vector) for index in corner_indices]
        ):
            vetoes += 1
            return False
        max_roundoff = max(max_roundoff, candidate_roundoff)
        return True

    refined, evidence = retriangulation_plan(
        vertices,
        triangles,
        polygon_ids,
        np.eye(3) if root_linear is None else root_linear,
        accept_flip=accept_flip,
    )
    result = {index: [] for index in plans}
    for index, triangle in zip(polygon_ids, refined):
        result[index].append(tuple(int(vertex) for vertex in triangle))
    evidence["attribute_veto_count"] = vetoes
    evidence["max_interpolation_error_float32_eps"] = max_roundoff
    return result, evidence


def _explicit_triangles(mesh, bmesh, *, root_linear=None, part_labels=None):
    """Materialize Blender's centered loop triangles while copying face/loop data.

    No TriangulateModifier.keep_custom_normals assumption: Blender 4.2 lacks it.
    A temporary vector loop layer carries decoded custom normals across new loops.
    Real Blender UV/attribute/custom-normal roundtrip tests are still mandatory.
    """
    mesh.calc_loop_triangles()
    plans = {}
    triangle_sources = {}
    attributes = getattr(mesh, "attributes", None)
    source_parts = attributes.get(PART_ATTRIBUTE) if attributes is not None else None
    for triangle in mesh.loop_triangles:
        key = tuple(sorted(triangle.vertices))
        if key in triangle_sources:
            where = ""
            if source_parts is not None and part_labels:
                a = int(source_parts.data[triangle_sources[key]].value)
                b = int(source_parts.data[triangle.polygon_index].value)
                names = sorted({part_labels[a], part_labels[b]}) if max(a, b) < len(part_labels) else []
                if len(names) == 2:
                    where = f" where part {names[0]!r} meets part {names[1]!r}"
                elif len(names) == 1:
                    where = f" inside part {names[0]!r}"
            require(
                False,
                f"mesh {mesh.name!r} has duplicate tessellated triangle vertices {key}"
                f"{where} (source polygons {triangle_sources.get(key)} and "
                f"{triangle.polygon_index}): their surfaces graze along a near-tangent "
                "intersection — deepen the overlap between these parts or shrink the "
                "bevel/rim so they intersect with real volume (no triangles were removed)",
            )
        triangle_sources[key] = triangle.polygon_index
        plans.setdefault(triangle.polygon_index, []).append(tuple(triangle.vertices))
    if source_parts is not None:
        mesh.attributes.remove(source_parts)  # diagnostic only; never exported
    plans, retriangulation = _refine_triangle_plans(mesh, plans, root_linear)
    schema = _attribute_schema(mesh)
    material = mesh.attributes.get("material_index")
    source_material_indices = None
    if material is not None:
        require(
            material.domain == "FACE"
            and material.data_type == "INT"
            and len(material.data) == len(mesh.polygons),
            "invalid source material-index layer",
        )
        source_material_indices = [int(value.value) for value in material.data]
    # Preserve sharp_face presence separately from its per-face shading values.
    sharp_face = mesh.attributes.get("sharp_face")
    source_sharp_faces = None
    if sharp_face is not None:
        require(
            sharp_face.domain == "FACE"
            and sharp_face.data_type == "BOOLEAN"
            and len(sharp_face.data) == len(mesh.polygons),
            "invalid source sharp-face layer",
        )
        source_sharp_faces = [bool(value.value) for value in sharp_face.data]
    source_vertices = [tuple(vertex.co) for vertex in mesh.vertices]
    source_edges = [tuple(edge.vertices) for edge in mesh.edges]
    source_selections = {}
    for name, domain, count in (
        (".select_vert", "POINT", len(source_vertices)),
        (".select_edge", "EDGE", len(source_edges)),
        (".select_poly", "FACE", len(mesh.polygons)),
    ):
        selection = mesh.attributes.get(name)
        values = None
        if selection is not None:
            require(
                selection.domain == domain
                and selection.data_type == "BOOLEAN"
                and len(selection.data) == count,
                "invalid source selection layer",
            )
            values = [bool(value.value) for value in selection.data]
        source_selections[name] = (domain, values)
    custom = (
        np.asarray(
            [tuple(normal.vector) for normal in mesh.corner_normals], dtype=np.float64
        )
        if mesh.has_custom_normals
        else None
    )
    transfer = "_grase_root_local_normal_transport"
    while mesh.attributes.get(transfer) is not None:
        transfer += "_"
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        original_vertices, original_edges = list(bm.verts), list(bm.edges)
        vertex_sources = {
            id(vertex): (vertex, index)
            for index, vertex in enumerate(original_vertices)
        }
        edge_sources = {
            id(edge): (edge, index) for index, edge in enumerate(original_edges)
        }
        require(
            len(original_vertices) == len(source_vertices)
            and all(
                vertex.index == index and tuple(vertex.co) == source_vertices[index]
                for index, vertex in enumerate(original_vertices)
            ),
            "BMesh source-vertex correspondence changed",
        )
        require(
            len(original_edges) == len(source_edges)
            and all(
                edge.index == index
                and sorted(vertex_sources[id(vertex)][1] for vertex in edge.verts)
                == sorted(source_edges[index])
                for index, edge in enumerate(original_edges)
            ),
            "BMesh source-edge correspondence changed",
        )
        originals = list(bm.faces)
        face_sources = {id(face): (face, index) for index, face in enumerate(originals)}
        require(
            len(originals) == len(mesh.polygons), "BMesh polygon correspondence changed"
        )
        normal_layer = (
            bm.loops.layers.float_vector.new(transfer) if custom is not None else None
        )
        for index, face in enumerate(originals):
            polygon = mesh.polygons[index]
            require(
                tuple(v.index for v in face.verts) == tuple(polygon.vertices),
                "BMesh corner correspondence changed",
            )
            if normal_layer is not None:
                for loop, source_index in zip(face.loops, polygon.loop_indices):
                    loop[normal_layer] = custom[source_index]
        for index, face in enumerate(originals):
            if len(face.verts) == 3:
                continue
            loops = {loop.vert.index: loop for loop in face.loops}
            for indices in plans[index]:
                triangle = bm.faces.new([bm.verts[i] for i in indices], face)
                face_sources[id(triangle)] = (triangle, index)
                for loop in triangle.loops:
                    loop.copy_from(loops[loop.vert.index])
            bm.faces.remove(face)
        bm.normal_update()
        output_vertices, output_edges = list(bm.verts), list(bm.edges)
        require(
            {id(vertex) for vertex in output_vertices} == set(vertex_sources)
            and len(output_vertices) == len(vertex_sources),
            "triangulation added or removed a source vertex",
        )
        require(
            set(edge_sources).issubset({id(edge) for edge in output_edges}),
            "triangulation removed a source edge",
        )
        require(
            all(
                sorted(vertex_sources[id(vertex)][1] for vertex in edge.verts)
                == sorted(source_edges[index])
                for edge, index in edge_sources.values()
            ),
            "triangulation rewired a source edge",
        )
        vertex_indices = {
            id(vertex): index for index, vertex in enumerate(output_vertices)
        }
        vertex_plan = [
            (tuple(vertex.co), vertex_sources[id(vertex)][1])
            for vertex in output_vertices
        ]
        require(
            all(co == source_vertices[index] for co, index in vertex_plan),
            "triangulation moved a source vertex",
        )
        edge_plan = [
            (
                tuple(vertex_indices[id(vertex)] for vertex in edge.verts),
                edge_sources[id(edge)][1] if id(edge) in edge_sources else None,
            )
            for edge in output_edges
        ]
        selection_plan = [
            (
                tuple(vertex_indices[id(vertex)] for vertex in face.verts),
                face_sources[id(face)][1],
            )
            for face in bm.faces
        ]
        bm.to_mesh(mesh)
    finally:
        bm.free()
    require(
        len(vertex_plan) == len(mesh.vertices)
        and all(
            tuple(vertex.co) == co
            for vertex, (co, _) in zip(mesh.vertices, vertex_plan)
        ),
        "output source-vertex correspondence changed",
    )
    require(
        len(edge_plan) == len(mesh.edges)
        and all(
            tuple(edge.vertices) == vertices
            for edge, (vertices, _) in zip(mesh.edges, edge_plan)
        ),
        "output source-edge correspondence changed",
    )
    require(
        len(selection_plan) == len(mesh.polygons)
        and all(
            tuple(polygon.vertices) == vertices
            for polygon, (vertices, _) in zip(mesh.polygons, selection_plan)
        ),
        "triangle source-face correspondence changed",
    )
    domain_plans = {
        "POINT": [index for _, index in vertex_plan],
        "EDGE": [index for _, index in edge_plan],
        "FACE": [index for _, index in selection_plan],
    }
    for name, (domain, source_values) in source_selections.items():
        selection = mesh.attributes.get(name)
        if selection is not None:
            require(
                selection.domain == domain
                and selection.data_type == "BOOLEAN"
                and len(selection.data) == len(domain_plans[domain]),
                "invalid output selection layer",
            )
        if source_values is None:
            if selection is not None:
                mesh.attributes.remove(selection)
            require(mesh.attributes.get(name) is None, "selection absence changed")
            continue
        if selection is None:
            selection = mesh.attributes.new(name, "BOOLEAN", domain)
        require(
            selection.domain == domain
            and selection.data_type == "BOOLEAN"
            and len(selection.data) == len(domain_plans[domain]),
            "invalid restored selection layer",
        )
        expected_selection = [
            False if index is None else source_values[index]
            for index in domain_plans[domain]
        ]
        for value, selected in zip(selection.data, expected_selection):
            value.value = selected
        require(
            [bool(value.value) for value in selection.data] == expected_selection,
            "explicit triangulation changed source selection values",
        )
    material = mesh.attributes.get("material_index")
    if material is not None:
        require(
            material.domain == "FACE"
            and material.data_type == "INT"
            and len(material.data) == len(selection_plan),
            "invalid output material-index layer",
        )
    if source_material_indices is None:
        if material is not None:
            mesh.attributes.remove(material)
        require(
            mesh.attributes.get("material_index") is None,
            "material-index absence changed",
        )
    else:
        if material is None:
            material = mesh.attributes.new("material_index", "INT", "FACE")
        require(
            material.domain == "FACE"
            and material.data_type == "INT"
            and len(material.data) == len(selection_plan),
            "invalid restored material-index layer",
        )
        expected_material_indices = [
            source_material_indices[index] for _, index in selection_plan
        ]
        for value, index in zip(material.data, expected_material_indices):
            value.value = index
        require(
            [int(value.value) for value in material.data] == expected_material_indices,
            "explicit triangulation changed source material indices",
        )
    # BMesh may omit an all-false sharp_face layer; preserve the source contract.
    sharp_face = mesh.attributes.get("sharp_face")
    if sharp_face is not None:
        require(
            sharp_face.domain == "FACE"
            and sharp_face.data_type == "BOOLEAN"
            and len(sharp_face.data) == len(selection_plan),
            "invalid output sharp-face layer",
        )
    if source_sharp_faces is None:
        if sharp_face is not None:
            mesh.attributes.remove(sharp_face)
        require(mesh.attributes.get("sharp_face") is None, "sharp-face absence changed")
    else:
        if sharp_face is None:
            sharp_face = mesh.attributes.new("sharp_face", "BOOLEAN", "FACE")
        require(
            sharp_face.domain == "FACE"
            and sharp_face.data_type == "BOOLEAN"
            and len(sharp_face.data) == len(selection_plan),
            "invalid restored sharp-face layer",
        )
        expected_sharp_faces = [
            source_sharp_faces[index] for _, index in selection_plan
        ]
        for value, sharp in zip(sharp_face.data, expected_sharp_faces):
            value.value = sharp
        require(
            [bool(value.value) for value in sharp_face.data] == expected_sharp_faces,
            "explicit triangulation changed source sharp-face values",
        )
    if custom is not None:
        attribute = mesh.attributes.get(transfer)
        require(
            attribute is not None and len(attribute.data) == len(mesh.loops),
            "custom-normal transport disappeared",
        )
        normals = [tuple(value.vector) for value in attribute.data]
        mesh.attributes.remove(attribute)
        mesh.normals_split_custom_set(normals)
    mesh.update()
    require(
        set(schema).issubset(set(_attribute_schema(mesh))),
        "explicit triangulation lost or changed a source data layer",
    )
    require(
        all(len(polygon.vertices) == 3 for polygon in mesh.polygons),
        "export mesh is not explicitly triangulated",
    )
    return retriangulation


def _source_witness(root, parts):
    # In-memory no-mutation evidence only; intentionally NOT a geometry cache key.
    rows = []
    for obj in [root, *parts]:
        row = {
            "name": obj.name,
            "pointer": obj.as_pointer(),
            "parent": obj.parent.name if obj.parent else None,
            "world": np.asarray(obj.matrix_world).tolist(),
            "basis": np.asarray(obj.matrix_basis).tolist(),
            "parent_inverse": np.asarray(obj.matrix_parent_inverse).tolist(),
        }
        if obj.type == "MESH":
            row.update(
                mesh_pointer=obj.data.as_pointer(),
                vertices=[tuple(v.co) for v in obj.data.vertices],
                polygons=[tuple(p.vertices) for p in obj.data.polygons],
                schema=_attribute_schema(obj.data),
                materials=[
                    slot.material.as_pointer() if slot.material else None
                    for slot in obj.material_slots
                ],
                material_indices=[p.material_index for p in obj.data.polygons],
                modifiers=[
                    (modifier.as_pointer(), modifier.name, modifier.type)
                    for modifier in obj.modifiers
                ],
            )
        rows.append(row)
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _plan_normal_transport(snapshots):
    """Mixed operands preserve captured shading directions, not auto/custom tags."""
    if not any(row["normals"] is not None for row in snapshots):
        return None
    occupied = {name for row in snapshots for name, _, _ in row["schema"]}
    base = "_grase_root_local_boolean_normal_transport"
    name, suffix = base, 0
    while name in occupied:
        suffix += 1
        name = base + "_" + str(suffix)
    for row in snapshots:
        normals = row["normals"]
        if normals is None:
            normals = np.asarray(
                [tuple(normal.vector) for normal in row["mesh"].corner_normals],
                dtype=np.float64,
            )
        require(
            len(normals) == len(row["mesh"].loops),
            "normal carrier source corner count changed",
        )
        row["transport_normals"] = transform_normals(normals, row["local"][:3, :3])
    return name


def _read_normal_transport(mesh, name):
    """Never retain an RNA layer handle across custom-data rebuilding operations."""
    attribute = mesh.attributes.get(name)
    require(
        attribute is not None
        and attribute.name == name
        and attribute.domain == "CORNER"
        and attribute.data_type == "FLOAT_VECTOR"
        and len(attribute.data) == len(mesh.loops),
        "normal carrier missing or changed schema/count",
    )
    return np.asarray(
        [tuple(value.vector) for value in attribute.data], dtype=np.float64
    )


def _write_normal_transport(mesh, name, normals):
    require(mesh.attributes.get(name) is None, "normal carrier name collision")
    attribute = mesh.attributes.new(name, "FLOAT_VECTOR", "CORNER")
    require(
        attribute.name == name
        and attribute.domain == "CORNER"
        and attribute.data_type == "FLOAT_VECTOR"
        and len(attribute.data) == len(mesh.loops) == len(normals),
        "normal carrier creation changed schema/count",
    )
    values = np.asarray(normals, dtype=np.float32)
    require(
        values.shape == (len(mesh.loops), 3) and np.isfinite(values).all(),
        "invalid normal carrier source values",
    )
    attribute.data.foreach_set("vector", values.ravel())
    require(
        np.array_equal(_read_normal_transport(mesh, name), values),
        "normal carrier write changed values",
    )


def _apply_normal_transport(mesh, name, *, remove):
    values = _read_normal_transport(mesh, name)
    normals = transform_normals(values, np.eye(3))
    mesh.normals_split_custom_set(normals.tolist())
    # The setter can reallocate corner CustomData; reacquire by our exact name.
    require(
        np.array_equal(_read_normal_transport(mesh, name), values),
        "custom-normal application changed its carrier",
    )
    if remove:
        mesh.attributes.remove(mesh.attributes.get(name))
        require(mesh.attributes.get(name) is None, "normal carrier survived cleanup")


PART_ATTRIBUTE = "grase_src_part"  # FACE INT stamped on each operand; survives the exact Boolean
SEAM_RETRY_OFFSET_M = 2e-5  # second union attempt: operand copies shift 0.02-0.06 mm (see seam_retry_offset)
_SEAM_RETRY_DIRECTIONS = np.asarray(
    [[1, 0.7, 0.3], [-0.6, 1, 0.4], [0.3, -0.5, 1], [-1, -0.4, 0.7],
     [0.8, 0.2, -1], [0.1, 1, -0.6], [-0.7, 0.6, -0.9], [1, -1, 0.2]],
    dtype=np.float64,
)
_SEAM_RETRY_DIRECTIONS /= np.linalg.norm(_SEAM_RETRY_DIRECTIONS, axis=1, keepdims=True)
SEAM_RETRY_MAX_OFFSET_M = 3 * SEAM_RETRY_OFFSET_M


def seam_retry_offset(index):
    """Bounded, deterministic operand shift: 8 skew directions x 3 magnitudes (0.02/0.04/0.06 mm),
    so any two operands closer than 24 apart shift DIFFERENTLY (that asymmetry is what breaks a
    near-tangent seam) while no operand moves more than SEAM_RETRY_MAX_OFFSET_M."""
    direction = _SEAM_RETRY_DIRECTIONS[index % len(_SEAM_RETRY_DIRECTIONS)]
    return direction * (SEAM_RETRY_OFFSET_M * (1 + index % 3))
CONTACT_PLANE_TOL_M = 5e-5
GRAZE_DEPTH_M = 1.5e-3


def part_contacts(parts):
    """Classify how authored parts meet, from per-part (label, face_centers, face_normals) in ONE frame.

    Returns (flush, grazing): pairs whose faces are coplanar within CONTACT_PLANE_TOL_M (they share
    a face and cannot union into one solid — the 0916 glasses bridge/rack dividers) and pairs whose
    bounding boxes overlap by less than GRAZE_DEPTH_M on some axis without coplanar faces (near-tangent
    rims/bevels, the seam-weld and duplicate-triangle cases). Pure numpy so it is unit-testable."""
    rows = []
    for label, centers, normals in parts:
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        rows.append((str(label), centers, normals, centers.min(0), centers.max(0)))
    flush, grazing = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            la, ca, na, loa, hia = rows[i]
            lb, cb, nb, lob, hib = rows[j]
            depth = np.minimum(hia, hib) - np.maximum(loa, lob)
            if (depth < -CONTACT_PLANE_TOL_M).any():
                continue
            parallel = np.abs(na @ nb.T) > 1.0 - 1e-4
            if parallel.any():
                # distance of every B face centre from every A face plane, only where parallel
                offsets = np.einsum("abk,ak->ab", cb[None, :, :] - ca[:, None, :], na)
                coplanar = parallel & (np.abs(offsets) < CONTACT_PLANE_TOL_M)
                if coplanar.any():
                    flush.append((la, lb, int(coplanar.sum())))
                    continue
            if float(depth.min()) < GRAZE_DEPTH_M:
                grazing.append((la, lb, float(depth.min())))
    return flush, grazing


def contact_note(parts, *, limit=6):
    """Agent-facing sentence naming the part pairs that only touch or graze ('' when none)."""
    flush, grazing = part_contacts(parts)
    notes = []
    if flush:
        pairs = ", ".join(f"{a} x {b}" for a, b, _ in flush[:limit])
        notes.append(
            f"parts that share a face with no volume overlap (flush contact): {pairs} — "
            "make each pair overlap by a few millimetres of real volume"
        )
    if grazing:
        pairs = ", ".join(f"{a} x {b} ({d * 1000:.1f} mm)" for a, b, d in grazing[:limit])
        notes.append(
            f"parts that only graze (overlap under {GRAZE_DEPTH_M * 1000:.1f} mm): {pairs} — "
            "deepen the overlap or shrink the rim/bevel"
        )
    return ("; " + "; ".join(notes)) if notes else ""


def _mesh_part_contacts(snapshots):
    """part_contacts() input from the operand meshes (all in the centered root-local frame).

    Centres/normals come from the vertex data (Newell), not from polygon properties whose
    caches can lag a foreach_set on the vertices."""
    rows = []
    for row in snapshots:
        mesh = row["mesh"]
        if mesh is None or not len(mesh.polygons):
            continue
        coords = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        starts = np.empty(len(mesh.polygons), dtype=np.int64)
        totals = np.empty(len(mesh.polygons), dtype=np.int64)
        mesh.polygons.foreach_get("loop_start", starts)
        mesh.polygons.foreach_get("loop_total", totals)
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        centers, normals = [], []
        for start, total in zip(starts, totals):
            pts = coords[loop_vertices[start : start + total]]
            nxt = np.roll(pts, -1, axis=0)
            normal = np.sum(np.cross(pts, nxt), axis=0)  # Newell
            length = np.linalg.norm(normal)
            if length <= 0.0:
                continue
            centers.append(pts.mean(0))
            normals.append(normal / length)
        if centers:
            rows.append((_part_label(row), np.asarray(centers), np.asarray(normals)))
    return rows


def _part_label(row):
    part = row["part"]
    return str(part.get("grase_part_label") or part.name)


@contextmanager
def authored_union(root, parts, *, bpy, bmesh):
    """Yield one disposable explicitly triangulated union for thin output adapters.

    Caller must authenticate canonical root/part identities and apply existing
    capture/transaction policy. No new restrictions on accepted modifier types,
    constraints, shape keys, linked/shared data, or parenting are introduced here.
    All actual Blender behavior remains proposal-stage until separately admitted.
    """
    parts = sorted(parts, key=lambda part: part.name)
    require(
        root.type == "EMPTY" and root.parent is None and parts, "invalid authored root"
    )
    require(
        len(set(parts)) == len(parts) and all(part.type == "MESH" for part in parts),
        "invalid part identities",
    )
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    world = affine(root.evaluated_get(deps).matrix_world)
    determinant = float(np.linalg.det(world[:3, :3]))
    before = _source_witness(root, parts)
    prefix = "_grase_authored_root_local_" + root.name
    require(bpy.data.collections.get(prefix) is None, "reserved collection exists")
    collection, meshes, snapshots = None, [], []
    try:
        # Evaluate/snapshot EVERY source before linking ANY temporary object.
        for index, part in enumerate(parts):
            evaluated = part.evaluated_get(deps)
            evaluated_world = affine(evaluated.matrix_world)
            direct = part.parent is root and part.parent_type == "OBJECT"
            local, provenance = select_part_mapping(
                world,
                evaluated_world,
                direct_object=direct,
                raw_mapping_eligible=not (
                    part.constraints
                    or part.animation_data is not None
                    or part.rigid_body is not None
                    or part.rigid_body_constraint is not None
                ),
                parent_inverse=part.matrix_parent_inverse if direct else None,
                basis=part.matrix_basis if direct else None,
            )
            name = prefix + f"_part_{index:03d}"
            require(bpy.data.meshes.get(name) is None, "reserved mesh exists")
            mesh = bpy.data.meshes.new_from_object(
                evaluated, preserve_all_data_layers=True, depsgraph=deps
            )
            require(mesh is not None, "evaluated mesh copy failed")
            mesh.name = name
            meshes.append(name)
            vertices = np.asarray(
                [tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64
            )
            mesh.calc_loop_triangles()
            triangles = np.asarray(
                [tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int64
            ).reshape(-1, 3)
            normals = (
                np.asarray(
                    [tuple(normal.vector) for normal in mesh.corner_normals],
                    dtype=np.float64,
                )
                if mesh.has_custom_normals
                else None
            )
            slots = [
                _persistent_material(slot.material, bpy=bpy)
                for slot in evaluated.material_slots
            ]
            polygon_materials = [
                slots[p.material_index] if p.material_index < len(slots) else None
                for p in mesh.polygons
            ]
            snapshots.append(
                {
                    "part": part,
                    "mesh": mesh,
                    "local": local,
                    "mapping": provenance,
                    "local_points": points_in_frame(vertices, local),
                    "world_points": points_in_frame(vertices, evaluated_world),
                    "triangles": triangles,
                    "normals": normals,
                    "polygon_materials": polygon_materials,
                    "material_slots": slots,
                    "evaluated_modifiers": [
                        modifier.type for modifier in part.modifiers
                    ],
                    "schema": _attribute_schema(mesh),
                }
            )
        center, centered = centered_parts([row["local_points"] for row in snapshots])
        palette = {
            material.name if material else "": material
            for row in snapshots
            for material in row["material_slots"] + row["polygon_materials"]
        }
        material_names = sorted(palette)
        # An unmaterialed mesh still needs one explicit empty slot for INDEX mode.
        if not material_names:
            palette[""] = None
            material_names = [""]
        expected_schema = set().union(*(set(row["schema"]) for row in snapshots))
        expected_custom_normals = any(row["normals"] is not None for row in snapshots)
        normal_transport = _plan_normal_transport(snapshots)
        collection = bpy.data.collections.new(prefix)
        bpy.context.scene.collection.children.link(collection)
        for index, (row, vertices) in enumerate(zip(snapshots, centered)):
            mesh = row["mesh"]
            mesh.vertices.foreach_set(
                "co", np.asarray(vertices, dtype=np.float32).ravel()
            )
            mesh.materials.clear()
            for name in material_names:
                mesh.materials.append(palette[name])
            for polygon, material in zip(mesh.polygons, row["polygon_materials"]):
                polygon.material_index = material_names.index(
                    material.name if material else ""
                )
            mesh.update()
            if row["normals"] is not None:
                mesh.normals_split_custom_set(
                    transform_normals(row["normals"], row["local"][:3, :3]).tolist()
                )
            if normal_transport is not None:
                _write_normal_transport(
                    mesh, normal_transport, row["transport_normals"]
                )
            stats = _solid_stats(
                mesh,
                bmesh,
                determinant,
                connected=False,
                part_name=(
                    f"{root.get('grase_graph_id') or root.name}/"
                    f"{row['part'].get('grase_part_label') or row['part'].name}"
                ),
            )
            row["local_volume"] = stats["local_volume"]
        part_labels = [_part_label(row) for row in snapshots]

        def attempt(offset_m, tag):
            """Chain the exact Boolean over fresh operands; offset_m > 0 shifts operand i by
            seam_retry_offset(i) (the disposable copies only — the agent's parts are
            untouched). The asymmetric shift is what breaks near-tangent rims the seam weld
            cannot collapse (a symmetric dilation keeps the tangency and failed all three
            0916 cases); flush contacts may come apart and are then reported by name."""
            operands = []
            for index, row in enumerate(snapshots):
                mesh = row["mesh"]
                if offset_m:
                    mesh = mesh.copy()
                    mesh.name = prefix + f"_{tag}_part_{index:03d}"
                    meshes.append(mesh.name)
                    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
                    mesh.vertices.foreach_get("co", coords)
                    coords = coords.reshape(-1, 3) + seam_retry_offset(index) * (offset_m / SEAM_RETRY_OFFSET_M)
                    mesh.vertices.foreach_set("co", coords.astype(np.float32).ravel())
                    mesh.update()
                if mesh.attributes.get(PART_ATTRIBUTE) is None:
                    attribute = mesh.attributes.new(PART_ATTRIBUTE, "INT", "FACE")
                    attribute.data.foreach_set("value", [index] * len(mesh.polygons))
                name = prefix + f"_{tag}_operand_{index:03d}"
                require(bpy.data.objects.get(name) is None, "reserved operand exists")
                operand = bpy.data.objects.new(name, mesh)
                collection.objects.link(operand)
                operands.append((operand, row["local_volume"]))
            operands.sort(key=lambda item: (-item[1], item[0].name))
            union = operands[0][0]
            for index, (operand, _) in enumerate(operands[1:]):
                modifier = union.modifiers.new(
                    name=prefix + f"_boolean_{index}", type="BOOLEAN"
                )
                modifier.operation, modifier.operand_type, modifier.object = (
                    "UNION",
                    "OBJECT",
                    operand,
                )
                modifier.solver, modifier.material_mode, modifier.use_self = (
                    "EXACT",
                    "INDEX",
                    False,
                )
                bpy.context.view_layer.update()
                current_deps = bpy.context.evaluated_depsgraph_get()
                name = prefix + f"_{tag}_union_{index:03d}"
                require(bpy.data.meshes.get(name) is None, "reserved union mesh exists")
                mesh = bpy.data.meshes.new_from_object(
                    union.evaluated_get(current_deps),
                    preserve_all_data_layers=True,
                    depsgraph=current_deps,
                )
                require(mesh is not None, "Boolean mesh copy failed")
                mesh.name = name
                meshes.append(name)
                require(
                    mesh.vertices and mesh.polygons, "Boolean produced empty geometry"
                )
                union.modifiers.remove(modifier)
                union.data = mesh
                bpy.data.objects.remove(operand, do_unlink=True)
                bpy.context.view_layer.update()
            evidence = _cleanup_union_seams(union.data, bmesh, world[:3, :3])
            return union, evidence

        bounds_slack, contacts, retried, first_seam_error = 0.0, "", False, None
        try:
            union, cleanup_evidence = attempt(0.0, "first")
        except RuntimeError as first_error:
            if "Boolean seam" not in str(first_error):
                raise
            retried, first_seam_error = True, first_error
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            contacts = contact_note(_mesh_part_contacts(snapshots))
            try:
                union, cleanup_evidence = attempt(SEAM_RETRY_OFFSET_M, "retry")
                _solid_stats(union.data, bmesh, determinant, connected=True)
            except RuntimeError as retry_error:
                # Parts that only touch come apart under the offset: report the ORIGINAL
                # failure with the pairs named, never the retry's own symptom.
                raise RuntimeError(str(first_error) + contacts) from retry_error
            bounds_slack = SEAM_RETRY_MAX_OFFSET_M
            cleanup_evidence["seam_weld_retry"] = {
                "operand_offset_m": SEAM_RETRY_OFFSET_M,
                "max_operand_offset_m": SEAM_RETRY_MAX_OFFSET_M,
                "first_error": str(first_error)[:300],
            }
        def finish():
            _solid_stats(union.data, bmesh, determinant, connected=True)
            if normal_transport is not None:
                _apply_normal_transport(union.data, normal_transport, remove=False)
            _preservation_evidence(
                union.data,
                expected_schema,
                material_names,
                custom_normals=expected_custom_normals,
            )
            cleanup_evidence["retriangulation"] = _explicit_triangles(
                union.data, bmesh, root_linear=world[:3, :3], part_labels=part_labels
            )
            if normal_transport is not None:
                _apply_normal_transport(union.data, normal_transport, remove=True)
            preservation = _preservation_evidence(
                union.data,
                expected_schema,
                material_names,
                custom_normals=expected_custom_normals,
            )
            stats = _solid_stats(union.data, bmesh, determinant, connected=True)
            canonical = np.asarray(
                [tuple(v.co) for v in union.data.vertices], dtype=np.float64
            )
            faces = np.asarray(
                [tuple(p.vertices) for p in union.data.polygons], dtype=np.int64
            )
            world_vertices = points_in_frame(canonical + center, world)
            canonical_stats, world_stats = (
                triangle_stats(canonical, faces),
                triangle_stats(world_vertices, faces),
            )
            lo = np.min([row["world_points"].min(0) for row in snapshots], axis=0)
            hi = np.max([row["world_points"].max(0) for row in snapshots], axis=0)
            error = np.maximum(
                np.abs(world_vertices.min(0) - lo), np.abs(world_vertices.max(0) - hi)
            )
            require(
                np.all(error <= np.maximum(1e-5, (hi - lo) * 1e-5) + bounds_slack),
                "union world bounds changed",
            )
            require(
                _source_witness(root, parts) == before,
                "source changed before output adapter",
            )
            return (
                cleanup_evidence, preservation, stats, canonical, faces, world_vertices,
                canonical_stats, world_stats, error,
            )

        try:
            (
                cleanup_evidence, preservation, stats, canonical, faces, world_vertices,
                canonical_stats, world_stats, error,
            ) = finish()
        except RuntimeError as exc:
            if not retried:
                raise
            raise RuntimeError(str(first_seam_error) + contacts) from exc
        yield {
            "temporary_object": union,
            "canonical_vertices": canonical,
            "faces": faces,
            "root_local_center": center,
            "root_world_matrix": world,
            "world_vertices": world_vertices,
            "frame_version": FRAME_VERSION,
            "mapping_provenance": {
                row["part"].name: row["mapping"] for row in snapshots
            },
            "evaluated_modifier_types": {
                row["part"].name: row["evaluated_modifiers"] for row in snapshots
            },
            "source_attribute_schemas": {
                row["part"].name: row["schema"] for row in snapshots
            },
            "part_world_points": [row["world_points"] for row in snapshots],
            "part_triangles": [row["triangles"] for row in snapshots],
            "part_labels": part_labels,
            "material_names": material_names,
            "triangle_material_indices": np.asarray(
                [p.material_index for p in union.data.polygons], dtype=np.int64
            ),
            "preservation_evidence": preservation,
            "cleanup_evidence": cleanup_evidence,
            "canonical_stats": canonical_stats,
            "world_stats": world_stats,
            "solid_stats": stats,
            "world_bounds": {
                "min": world_vertices.min(0).tolist(),
                "max": world_vertices.max(0).tolist(),
            },
            "bounds_error_m": error,
            "bounds_slack_m": bounds_slack,
            "contact_note": contacts,
            "source_witness": before,
        }
    finally:
        if collection is not None:
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(collection)
        for name in meshes:
            mesh = bpy.data.meshes.get(name)
            if mesh is not None:
                require(mesh.users == 0, "temporary mesh acquired an external user")
                bpy.data.meshes.remove(mesh)
        require(
            _source_witness(root, parts) == before,
            "source changed after output/cleanup",
        )


def register_arrays(payload):
    """Authored-only register adapter: no cast, repair, transform fit, or disk IO."""
    return payload["world_vertices"].copy(), payload["faces"].copy()


def part_arrays(payload):
    """Evaluated world-frame ``[(vertices, triangles), ...]`` per source part, in the
    union's operand order. The collider cook hulls each part separately instead of
    re-slicing the fused union (``collision.decompose_authored_parts``)."""
    return [
        (points.copy(), triangles.copy())
        for points, triangles in zip(
            payload["part_world_points"], payload["part_triangles"]
        )
    ]


def prepare_capture_object(payload, *, Matrix):
    """Caller may export this TEMPORARY object inside the context; never save live parts.

    Centered vertices stay local. The Blender object's float32 matrix and its GLB
    roundtrip must be separately validated against the float64 frame metadata.
    """
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = payload["root_local_center"]
    matrix = payload["root_world_matrix"] @ translation
    payload["temporary_object"].matrix_world = Matrix(matrix.tolist())
    return payload["temporary_object"], matrix
