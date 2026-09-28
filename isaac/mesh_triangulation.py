"""Deterministic triangulation of simple planar USD polygon faces.

Triangle input is retained verbatim, including degenerates handled by downstream
collider cleanup. N-gons must describe a simple planar boundary: silently fanning
a concave face fills its notches and can change both collision geometry and mass.
"""

from __future__ import annotations

import numpy as np

_AREA_TOLERANCE = 1e-12
_PLANAR_TOLERANCE = 1e-6
_MAX_POLYGON_VERTICES = 1024


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _check_simple_boundary(polygon: np.ndarray, label: str) -> None:
    """Reject intersecting or touching nonadjacent polygon edges."""
    size = len(polygon)
    following = np.roll(polygon, -1, axis=0)
    for first in range(size - 2):
        others = np.arange(first + 2, size)
        if first == 0:
            others = others[:-1]  # First/last edges share the closing vertex.
        a, b = polygon[first], following[first]
        c, d = polygon[others], following[others]
        ab_c, ab_d = _cross_2d(b - a, c - a), _cross_2d(b - a, d - a)
        cd_a, cd_b = _cross_2d(d - c, a - c), _cross_2d(d - c, b - c)
        eps = _AREA_TOLERANCE
        crosses_ab = ((ab_c <= eps) & (ab_d >= -eps)) | ((ab_d <= eps) & (ab_c >= -eps))
        crosses_cd = ((cd_a <= eps) & (cd_b >= -eps)) | ((cd_b <= eps) & (cd_a >= -eps))
        boxes_overlap = np.all(
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + eps,
            axis=1,
        )
        if np.any(crosses_ab & crosses_cd & boxes_overlap):
            raise ValueError(f"{label}: self-intersecting or touching polygon boundary")


def _triangulate_polygon(
    points: np.ndarray, indices: np.ndarray, label: str
) -> np.ndarray:
    size = len(indices)
    if size > _MAX_POLYGON_VERTICES:
        raise ValueError(
            f"{label}: polygon has {size} vertices; maximum is {_MAX_POLYGON_VERTICES}"
        )
    if len(np.unique(indices)) != size:
        raise ValueError(f"{label}: polygon repeats a vertex index")

    polygon = points[indices]
    scale = float(np.max(np.ptp(polygon, axis=0)))
    if scale <= 0:
        raise ValueError(f"{label}: degenerate polygon has zero extent")
    polygon = (polygon - polygon[0]) / scale
    if len(np.unique(polygon, axis=0)) != size:
        raise ValueError(f"{label}: polygon repeats a vertex position")
    normal = np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= _AREA_TOLERANCE:
        raise ValueError(
            f"{label}: degenerate or self-intersecting polygon has zero area"
        )
    normal /= normal_length
    deviation = float(np.max(np.abs(polygon @ normal)))
    if deviation > _PLANAR_TOLERANCE:
        raise ValueError(
            f"{label}: nonplanar polygon (relative deviation {deviation:.9g} "
            f"> {_PLANAR_TOLERANCE})"
        )
    # Dropping the dominant normal axis is well conditioned. Ear orientation is
    # measured in this projection, but output indices retain the original winding.
    projected = np.delete(polygon, int(np.argmax(np.abs(normal))), axis=1)
    _check_simple_boundary(projected, label)
    signed_area = float(np.sum(_cross_2d(projected, np.roll(projected, -1, axis=0))))
    winding = 1.0 if signed_area > 0 else -1.0
    remaining = list(range(size))
    triangles = []
    while len(remaining) > 3:
        for cursor, current in enumerate(remaining):
            previous = remaining[cursor - 1]
            following = remaining[(cursor + 1) % len(remaining)]
            a, b, c = projected[[previous, current, following]]
            if winding * _cross_2d(b - a, c - a) <= _AREA_TOLERANCE:
                continue
            others = [i for i in remaining if i not in (previous, current, following)]
            candidates = projected[others]
            inside = (
                (winding * _cross_2d(b - a, candidates - a) >= -_AREA_TOLERANCE)
                & (winding * _cross_2d(c - b, candidates - b) >= -_AREA_TOLERANCE)
                & (winding * _cross_2d(a - c, candidates - c) >= -_AREA_TOLERANCE)
            )
            # A vertex on an ear diagonal also blocks clipping: otherwise the
            # remaining boundary can contain an overlapping/zero-area triangle.
            if np.any(inside):
                continue
            triangles.append((previous, current, following))
            del remaining[cursor]
            break
        else:
            raise ValueError(
                f"{label}: polygon cannot be triangulated without degeneracy"
            )
    a, b, c = projected[remaining]
    if winding * _cross_2d(b - a, c - a) <= _AREA_TOLERANCE:
        raise ValueError(f"{label}: final polygon triangle is degenerate")
    triangles.append(tuple(remaining))
    local_indices = np.asarray(triangles, dtype=np.int64)
    local = projected[local_indices]
    triangle_area = np.sum(
        _cross_2d(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    )
    if not np.isclose(triangle_area, signed_area, rtol=1e-10, atol=_AREA_TOLERANCE):
        raise ValueError(f"{label}: triangulation does not preserve polygon area")
    return indices[local_indices]


def triangulate_mesh_faces(
    points: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    *,
    mesh_name: str = "mesh",
) -> np.ndarray:
    """Return triangle indices without adding vertices or changing face winding.

    Existing triangles are not geometrically cleaned or rejected: tiny/zero-area
    SAM3D triangles remain the responsibility of collider cleanup. Simple planar
    n-gons are ear-clipped; unsupported polygons fail before a dump is published.

    Raises:
        ValueError: For malformed topology, nonfinite points, or an unsupported
            n-gon (nonplanar, degenerate, self-intersecting, or over 1024 vertices).
    """
    vertices = np.asarray(points, dtype=np.float64)
    counts = np.asarray(face_vertex_counts)
    indices = np.asarray(face_vertex_indices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"{mesh_name}: points must have nonempty (N, 3) shape")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{mesh_name}: points contain nonfinite coordinates")
    for field, values in (("faceVertexCounts", counts), ("faceVertexIndices", indices)):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(
                f"{mesh_name}: {field} must be a one-dimensional integer array"
            )
    if not len(counts) or np.any(counts < 3):
        raise ValueError(
            f"{mesh_name}: every face must contain at least three vertices"
        )
    if np.any(counts > len(indices)) or int(np.sum(counts, dtype=np.int64)) != len(
        indices
    ):
        raise ValueError(f"{mesh_name}: face counts do not match the index buffer")
    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError(f"{mesh_name}: face vertex index is outside the point array")
    indices = indices.astype(np.int64, copy=False)
    if np.all(counts == 3):
        return indices.reshape(-1, 3).copy()

    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    pieces = []
    for face, count in enumerate(counts):
        face_indices = indices[offsets[face] : offsets[face + 1]]
        if count == 3:
            pieces.append(face_indices.reshape(1, 3))
        else:
            pieces.append(
                _triangulate_polygon(vertices, face_indices, f"{mesh_name} face {face}")
            )
    return np.concatenate(pieces)
