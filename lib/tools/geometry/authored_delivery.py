"""Scalable authenticated authored-object delivered-pose correspondence.

A vectorized proposed pairing is accepted only after complete triangle/corner
verification. Exact graph matching is limited to the unresolved ambiguous remainder.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import deque
from collections.abc import Mapping

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORNER_PERMUTATIONS = tuple(itertools.permutations(range(3)))
_MIN_AUTHENTICATED_TOLERANCE_M = 1e-12
_AUTHENTICATED_TOLERANCE_CAP_M = 1e-5
_MAX_AUTHENTICATED_WORKING_BYTES = 3 * 1024**3
_ESTIMATED_PEAK_BYTES_PER_FACE = 768
_MAX_AUTHENTICATED_FACES = (
    _MAX_AUTHENTICATED_WORKING_BYTES // _ESTIMATED_PEAK_BYTES_PER_FACE
)
_DIRECT_EXACT_FALLBACK_FACES = 256
_MAX_EXACT_FALLBACK_FACES = 4_096
_MAX_EXACT_CANDIDATE_EDGES = 100_000
_VERIFY_CHUNK_FACES = 65_536


class AuthoredTelemetryError(ValueError):
    """Fail-closed authored telemetry error with a machine-readable reason code."""

    def __init__(self, message: str, *, code: str = "invalid_authored_telemetry"):
        super().__init__(message)
        self.code = code


def _require(
    condition: bool,
    message: str,
    *,
    code: str = "invalid_authored_telemetry",
) -> None:
    if not condition:
        raise AuthoredTelemetryError(message, code=code)


def _canonical_mesh_name(object_id: str) -> str:
    category, separator, instance = object_id.rpartition("#")
    _require(
        bool(separator and category and instance.isdigit()),
        f"invalid authored object identity {object_id!r}",
        code="stale_binding",
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", category).strip("_").lower()[:48]
    return f"obj_{slug or 'object'}_{int(instance)}"


def _affine_matrix(value: object, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AuthoredTelemetryError(
            f"{label} is not a numeric 4x4 matrix", code="malformed_matrix"
        ) from exc
    _require(matrix.shape == (4, 4), f"{label} is not a 4x4 matrix", code="malformed_matrix")
    _require(
        np.isfinite(matrix).all(),
        f"{label} contains non-finite values",
        code="malformed_matrix",
    )
    _require(
        np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9, rtol=0.0),
        f"{label} is not affine",
        code="malformed_matrix",
    )
    determinant = float(np.linalg.det(matrix[:3, :3]))
    _require(
        math.isfinite(determinant) and determinant > 1e-12,
        f"{label} is singular or reflected",
        code="unsupported_transform",
    )
    return matrix


def _positive_similarity(value: object, label: str) -> tuple[np.ndarray, float, float]:
    matrix = _affine_matrix(value, label)
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    _require(
        math.isfinite(scale) and scale > 1e-12,
        f"{label} has invalid scale",
        code="unsupported_transform",
    )
    rotation = linear / scale
    residual = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    rotation_determinant = float(np.linalg.det(rotation))
    _require(
        residual <= 1e-5 and abs(rotation_determinant - 1.0) <= 1e-5,
        f"{label} contains unsupported shear or non-uniform scale",
        code="unsupported_transform",
    )
    return matrix, scale, residual


def _triangles(value: object, label: str) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        _require(
            len(value) <= 3 * _MAX_AUTHENTICATED_FACES,
            f"{label} exceeds authenticated telemetry face cap",
            code="resource_face_cap",
        )
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise AuthoredTelemetryError(
            f"{label} is not numeric triangle geometry", code="malformed_geometry"
        ) from exc
    if raw.ndim == 2 and raw.shape[1:] == (3,):
        _require(
            len(raw) > 0 and len(raw) % 3 == 0,
            f"{label} does not contain complete triangles",
            code="malformed_geometry",
        )
        face_count = len(raw) // 3
    else:
        face_count = len(raw) if raw.ndim else 0
    _require(
        (raw.ndim == 2 and raw.shape[1:] == (3,))
        or (raw.ndim == 3 and raw.shape[1:] == (3, 3)),
        f"{label} must have shape [faces, 3, 3]",
        code="malformed_geometry",
    )
    _require(face_count > 0, f"{label} is empty", code="malformed_geometry")
    _require(
        face_count <= _MAX_AUTHENTICATED_FACES,
        f"{label} exceeds authenticated telemetry face cap "
        f"({face_count} > {_MAX_AUTHENTICATED_FACES}; "
        f"budget {_MAX_AUTHENTICATED_WORKING_BYTES} bytes)",
        code="resource_face_cap",
    )
    try:
        triangles = raw.astype(np.float64, copy=False).reshape(-1, 3, 3)
    except (TypeError, ValueError) as exc:
        raise AuthoredTelemetryError(
            f"{label} is not numeric triangle geometry", code="malformed_geometry"
        ) from exc
    _require(
        np.isfinite(triangles).all(),
        f"{label} contains non-finite geometry",
        code="malformed_geometry",
    )
    return triangles


def _apply(matrix: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    transformed = triangles @ matrix[:3, :3].T + matrix[:3, 3]
    _require(
        np.isfinite(transformed).all(),
        "transformed authored geometry is non-finite",
        code="malformed_geometry",
    )
    return transformed


def _triangle_error(left: np.ndarray, right: np.ndarray) -> float:
    return min(
        float(np.max(np.abs(left - right[list(permutation)])))
        for permutation in _CORNER_PERMUTATIONS
    )


def _triangle_anchors(triangles: np.ndarray) -> np.ndarray:
    """Return permutation-invariant componentwise min/max anchors in chunks."""

    anchors = np.empty((len(triangles), 6), dtype=np.float64)
    for start in range(0, len(triangles), _VERIFY_CHUNK_FACES):
        stop = min(start + _VERIFY_CHUNK_FACES, len(triangles))
        chunk = triangles[start:stop]
        anchors[start:stop, :3] = np.min(chunk, axis=1)
        anchors[start:stop, 3:] = np.max(chunk, axis=1)
    return anchors


def _canonical_corner_signatures(triangles: np.ndarray) -> np.ndarray:
    signatures = np.empty((len(triangles), 9), dtype=np.float64)
    for start in range(0, len(triangles), _VERIFY_CHUNK_FACES):
        stop = min(start + _VERIFY_CHUNK_FACES, len(triangles))
        chunk = triangles[start:stop]
        order = np.lexsort(
            (chunk[:, :, 2], chunk[:, :, 1], chunk[:, :, 0]), axis=1
        )
        signatures[start:stop] = np.take_along_axis(
            chunk, order[:, :, None], axis=1
        ).reshape(-1, 9)
    return signatures


def _anchor_signature_order(
    anchors: np.ndarray, signatures: np.ndarray
) -> np.ndarray:
    keys = tuple(signatures[:, axis] for axis in range(8, -1, -1)) + tuple(
        anchors[:, axis] for axis in range(5, -1, -1)
    )
    return np.lexsort(keys)


def _max_exact_anchor_run(anchors: np.ndarray) -> int:
    if len(anchors) <= 1:
        return len(anchors)
    order = np.lexsort(tuple(anchors[:, axis] for axis in range(5, -1, -1)))
    ordered = anchors[order]
    boundaries = np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1
    lengths = np.diff(np.concatenate(([0], boundaries, [len(ordered)])))
    return int(np.max(lengths))


def _verified_pairing_error(
    source: np.ndarray,
    target: np.ndarray,
    source_order: np.ndarray,
    target_order: np.ndarray,
    tolerance: float,
) -> float | None:
    """Fully verify a proposed one-to-one pairing in bounded vectorized chunks."""

    _require(
        source_order.shape == target_order.shape,
        "proposed triangle pairing has unequal sides",
        code="geometry_mismatch",
    )
    worst = 0.0
    for start in range(0, len(source_order), _VERIFY_CHUNK_FACES):
        stop = min(start + _VERIFY_CHUNK_FACES, len(source_order))
        left = source[source_order[start:stop]]
        right = target[target_order[start:stop]]
        best = np.full(len(left), np.inf, dtype=np.float64)
        for permutation in _CORNER_PERMUTATIONS:
            error = np.max(
                np.abs(left - right[:, list(permutation), :]), axis=(1, 2)
            )
            np.minimum(best, error, out=best)
        if not np.isfinite(best).all() or np.any(best > tolerance):
            return None
        if len(best):
            worst = max(worst, float(np.max(best)))
    return worst


def _sorted_fast_pairing(
    source: np.ndarray,
    target: np.ndarray,
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    tolerance: float,
) -> dict[str, float | int | str] | None:
    """Propose an invariant-anchor sort pairing, then prove every triangle."""

    source_order = _anchor_signature_order(
        source_anchors, _canonical_corner_signatures(source)
    )
    target_order = _anchor_signature_order(
        target_anchors, _canonical_corner_signatures(target)
    )
    error = _verified_pairing_error(
        source, target, source_order, target_order, tolerance
    )
    if error is None:
        return None
    return {
        "triangle_count": int(len(source)),
        "max_corner_error_m": error,
        "matching_path": "sorted_verified",
        "fallback_triangle_count": 0,
        "fallback_candidate_edges": 0,
    }


def _exact_ambiguous_bijection(
    source: np.ndarray,
    target: np.ndarray,
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    tolerance: float,
) -> tuple[float, int]:
    """Exact bounded matching for the unresolved anchor-ambiguous remainder."""

    count = len(source)
    _require(
        count <= _MAX_EXACT_FALLBACK_FACES,
        "authored geometry ambiguity exceeds exact fallback face cap "
        f"({count} > {_MAX_EXACT_FALLBACK_FACES})",
        code="resource_fallback_face_cap",
    )
    cell_size = 2.0 * tolerance
    cell_origin = np.minimum(
        np.min(source_anchors[:, :3], axis=0),
        np.min(target_anchors[:, :3], axis=0),
    )

    def cell(point: np.ndarray) -> tuple[int, int, int]:
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (point - cell_origin) / cell_size
        _require(
            np.isfinite(scaled).all(),
            "authored triangle anchor range cannot be indexed safely",
            code="malformed_geometry",
        )
        return tuple(math.floor(float(value)) for value in scaled)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, anchor in enumerate(target_anchors):
        buckets.setdefault(cell(anchor[:3]), []).append(index)

    # Build only anchor candidate indices first. This lets the resource cap fire
    # before expensive corner comparisons or a dense error graph is allocated.
    candidate_rows: list[list[int]] = []
    candidate_edges = 0
    anchor_tolerance = np.nextafter(tolerance, math.inf)
    for anchor in source_anchors:
        origin = cell(anchor[:3])
        candidates: list[int] = []
        for offset in itertools.product((-1, 0, 1), repeat=3):
            key = tuple(origin[axis] + offset[axis] for axis in range(3))
            candidates.extend(buckets.get(key, ()))
        candidates = sorted(
            {
                index
                for index in candidates
                if float(np.max(np.abs(anchor - target_anchors[index])))
                <= anchor_tolerance
            }
        )
        _require(
            bool(candidates),
            "authored geometry mismatch: a triangle anchor has no candidate",
            code="geometry_mismatch",
        )
        candidate_edges += len(candidates)
        _require(
            candidate_edges <= _MAX_EXACT_CANDIDATE_EDGES,
            "authored geometry ambiguity exceeds exact fallback candidate-edge cap "
            f"({candidate_edges} > {_MAX_EXACT_CANDIDATE_EDGES})",
            code="resource_candidate_edge_cap",
        )
        candidate_rows.append(candidates)

    adjacency: list[list[int]] = []
    errors: dict[tuple[int, int], float] = {}
    for source_index, candidates in enumerate(candidate_rows):
        row = []
        for target_index in candidates:
            error = _triangle_error(source[source_index], target[target_index])
            if error <= tolerance:
                row.append(target_index)
                errors[source_index, target_index] = error
        _require(
            bool(row),
            "authored geometry mismatch: a reference triangle has no exact match",
            code="geometry_mismatch",
        )
        adjacency.append(sorted(row, key=lambda item: (errors[source_index, item], item)))

    left_to_right = [-1] * count
    right_to_left = [-1] * count
    for start in sorted(range(count), key=lambda index: len(adjacency[index])):
        queue: deque[int] = deque([start])
        visited_left = {start}
        visited_right: set[int] = set()
        parent_right: dict[int, int] = {}
        free_right: int | None = None
        while queue and free_right is None:
            left = queue.popleft()
            for right in adjacency[left]:
                if right in visited_right:
                    continue
                visited_right.add(right)
                parent_right[right] = left
                matched_left = right_to_left[right]
                if matched_left == -1:
                    free_right = right
                    break
                if matched_left not in visited_left:
                    visited_left.add(matched_left)
                    queue.append(matched_left)
        _require(
            free_right is not None,
            "authored geometry mismatch: triangles have no one-to-one bijection",
            code="geometry_mismatch",
        )
        right = free_right
        while right is not None:
            left = parent_right[right]
            previous_right = left_to_right[left]
            left_to_right[left] = right
            right_to_left[right] = left
            right = previous_right if previous_right != -1 else None

    _require(
        all(index >= 0 for index in left_to_right)
        and all(index >= 0 for index in right_to_left),
        "authored geometry mismatch: triangle bijection is incomplete",
        code="geometry_mismatch",
    )
    max_error = max(errors[left, right] for left, right in enumerate(left_to_right))
    return float(max_error), candidate_edges


def _ambiguity_fallback(
    source: np.ndarray,
    target: np.ndarray,
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    tolerance: float,
) -> dict[str, float | int | str]:
    """Lock mutual-unique anchor pairs and exactly match only the remainder."""

    if len(source) <= _DIRECT_EXACT_FALLBACK_FACES:
        max_error, edges = _exact_ambiguous_bijection(
            source,
            target,
            source_anchors,
            target_anchors,
            tolerance,
        )
        return {
            "triangle_count": int(len(source)),
            "max_corner_error_m": max_error,
            "matching_path": "exact_fallback_small",
            "fallback_triangle_count": int(len(source)),
            "fallback_candidate_edges": edges,
        }

    maximum_exact_run = max(
        _max_exact_anchor_run(source_anchors),
        _max_exact_anchor_run(target_anchors),
    )
    _require(
        maximum_exact_run <= _MAX_EXACT_FALLBACK_FACES,
        "authored geometry exact-anchor density exceeds exact fallback face cap "
        f"({maximum_exact_run} > {_MAX_EXACT_FALLBACK_FACES})",
        code="resource_fallback_face_cap",
    )

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        _require(
            len(source) <= _MAX_EXACT_FALLBACK_FACES,
            "vectorized anchor matcher unavailable and full exact fallback exceeds cap",
            code="resource_fast_matcher_unavailable",
        )
        max_error, edges = _exact_ambiguous_bijection(
            source,
            target,
            source_anchors,
            target_anchors,
            tolerance,
        )
        return {
            "triangle_count": int(len(source)),
            "max_corner_error_m": max_error,
            "matching_path": "exact_fallback_without_scipy",
            "fallback_triangle_count": int(len(source)),
            "fallback_candidate_edges": edges,
        }

    upper_bound = np.nextafter(tolerance, math.inf)
    try:
        target_tree = cKDTree(target_anchors)
        source_distances, source_neighbors = target_tree.query(
            source_anchors,
            k=2,
            p=math.inf,
            distance_upper_bound=upper_bound,
            workers=1,
        )
        source_tree = cKDTree(source_anchors)
        target_distances, target_neighbors = source_tree.query(
            target_anchors,
            k=2,
            p=math.inf,
            distance_upper_bound=upper_bound,
            workers=1,
        )
    except MemoryError as exc:
        raise AuthoredTelemetryError(
            "vectorized anchor matcher exhausted its bounded memory budget",
            code="resource_memory",
        ) from exc

    source_distances = np.atleast_2d(source_distances)
    source_neighbors = np.atleast_2d(source_neighbors)
    target_distances = np.atleast_2d(target_distances)
    target_neighbors = np.atleast_2d(target_neighbors)
    # np.atleast_2d turns a one-face (2,) result into (1, 2), as intended.
    _require(
        np.isfinite(source_distances[:, 0]).all()
        and np.isfinite(target_distances[:, 0]).all(),
        "authored geometry mismatch: at least one triangle anchor has no match",
        code="geometry_mismatch",
    )
    source_one = ~np.isfinite(source_distances[:, 1])
    target_one = ~np.isfinite(target_distances[:, 1])
    proposed_target = source_neighbors[:, 0].astype(np.int64, copy=False)
    source_indices = np.arange(len(source), dtype=np.int64)
    locked_mask = (
        source_one
        & target_one[proposed_target]
        & (target_neighbors[proposed_target, 0] == source_indices)
    )
    locked_source = source_indices[locked_mask]
    locked_target = proposed_target[locked_mask]
    locked_error = _verified_pairing_error(
        source, target, locked_source, locked_target, tolerance
    )
    _require(
        locked_error is not None,
        "authored geometry mismatch: a mutual-unique triangle does not match",
        code="geometry_mismatch",
    )

    remaining_source = source_indices[~locked_mask]
    target_locked = np.zeros(len(target), dtype=bool)
    target_locked[locked_target] = True
    remaining_target = source_indices[~target_locked]
    _require(
        len(remaining_source) == len(remaining_target),
        "authored geometry mismatch: unique pairing left unequal ambiguous sides",
        code="geometry_mismatch",
    )
    if not len(remaining_source):
        return {
            "triangle_count": int(len(source)),
            "max_corner_error_m": float(locked_error or 0.0),
            "matching_path": "mutual_unique_verified",
            "fallback_triangle_count": 0,
            "fallback_candidate_edges": 0,
        }

    _require(
        len(remaining_source) <= _MAX_EXACT_FALLBACK_FACES,
        "authored geometry ambiguity exceeds exact fallback face cap "
        f"({len(remaining_source)} > {_MAX_EXACT_FALLBACK_FACES})",
        code="resource_fallback_face_cap",
    )
    fallback_error, edges = _exact_ambiguous_bijection(
        source[remaining_source],
        target[remaining_target],
        source_anchors[remaining_source],
        target_anchors[remaining_target],
        tolerance,
    )
    return {
        "triangle_count": int(len(source)),
        "max_corner_error_m": max(float(locked_error or 0.0), fallback_error),
        "matching_path": "mutual_unique_plus_exact_fallback",
        "fallback_triangle_count": int(len(remaining_source)),
        "fallback_candidate_edges": edges,
    }


def triangle_bijection(
    reference: object,
    current: object,
    *,
    tolerance_m: float = _AUTHENTICATED_TOLERANCE_CAP_M,
) -> dict[str, float | int | str]:
    """Prove an order-independent triangle bijection under bounded resources."""

    _require(
        not isinstance(tolerance_m, bool)
        and isinstance(tolerance_m, (int, float))
        and math.isfinite(float(tolerance_m))
        and float(tolerance_m) > 0.0,
        "triangle tolerance must be finite and positive",
        code="malformed_tolerance",
    )
    tolerance = float(tolerance_m)
    _require(
        tolerance >= _MIN_AUTHENTICATED_TOLERANCE_M,
        "triangle tolerance is below the supported 1e-12 meter floor",
        code="malformed_tolerance",
    )
    _require(
        tolerance <= _AUTHENTICATED_TOLERANCE_CAP_M,
        "triangle tolerance exceeds the authenticated 1e-5 meter cap",
        code="weakened_tolerance",
    )
    source = _triangles(reference, "reference triangles")
    target = _triangles(current, "current triangles")
    _require(
        source.shape == target.shape,
        "authored geometry topology changed: triangle counts differ",
        code="topology_mismatch",
    )
    _require(
        len(source) <= _MAX_AUTHENTICATED_FACES,
        "authored geometry exceeds authenticated telemetry face cap "
        f"({len(source)} > {_MAX_AUTHENTICATED_FACES})",
        code="resource_face_cap",
    )
    try:
        source_anchors = _triangle_anchors(source)
        target_anchors = _triangle_anchors(target)
        _require(
            np.isfinite(source_anchors).all() and np.isfinite(target_anchors).all(),
            "authored triangle anchors are non-finite",
            code="malformed_geometry",
        )
        fast = _sorted_fast_pairing(
            source, target, source_anchors, target_anchors, tolerance
        )
        if fast is not None:
            return fast
        return _ambiguity_fallback(
            source, target, source_anchors, target_anchors, tolerance
        )
    except MemoryError as exc:
        raise AuthoredTelemetryError(
            "authored telemetry exhausted its bounded memory budget",
            code="resource_memory",
        ) from exc


def _validate_binding(
    binding: Mapping[str, object],
    *,
    object_name: str,
    actual_reference_sha256: str,
) -> tuple[str, int, int]:
    _require(
        isinstance(binding, Mapping),
        "authored revision binding is not a mapping",
        code="stale_binding",
    )
    object_id = binding.get("object_id")
    _require(
        isinstance(object_id, str)
        and object_id
        and binding.get("mesh_name") == object_name
        and _canonical_mesh_name(object_id) == object_name,
        "authored revision has a stale graph/object binding",
        code="stale_binding",
    )
    _require(
        binding.get("reference_pose") == "authored_revision",
        "authored revision has an invalid reference pose",
        code="stale_binding",
    )
    digests = [
        actual_reference_sha256,
        binding.get("sha256"),
        binding.get("revision_sha256"),
        binding.get("capture_glb_sha256"),
    ]
    _require(
        all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in digests),
        "authored revision has an invalid SHA256 binding",
        code="stale_binding",
    )
    _require(
        len(set(digests)) == 1,
        "authored revision GLB/capture digest binding is stale",
        code="stale_binding",
    )
    revision = binding.get("revision")
    current_revision = binding.get("current_revision")
    _require(
        isinstance(revision, int)
        and not isinstance(revision, bool)
        and revision >= 1
        and isinstance(current_revision, int)
        and not isinstance(current_revision, bool)
        and current_revision == revision,
        "authored revision number binding is stale",
        code="stale_binding",
    )
    transaction_id = binding.get("transaction_id")
    placement_transaction_id = binding.get("placement_transaction_id")
    record_transaction_id = binding.get("record_transaction_id")
    _require(
        isinstance(transaction_id, int)
        and not isinstance(transaction_id, bool)
        and transaction_id >= 1
        and isinstance(placement_transaction_id, int)
        and not isinstance(placement_transaction_id, bool)
        and placement_transaction_id == transaction_id
        and isinstance(record_transaction_id, int)
        and not isinstance(record_transaction_id, bool)
        and record_transaction_id >= 1,
        "authored revision transaction binding is stale",
        code="stale_binding",
    )
    return object_id, revision, transaction_id


def measure_authored_delivery(
    *,
    object_name: str,
    binding: Mapping[str, object],
    actual_reference_sha256: str,
    reference_triangles: object,
    current_triangles: object,
    capture_root_matrix: object,
    current_root_matrix: object,
    baked_delta: object,
    tolerance_m: float = _AUTHENTICATED_TOLERANCE_CAP_M,
) -> dict[str, object]:
    """Authenticate geometry and derive capture-to-delivered world motion."""

    object_id, revision, transaction_id = _validate_binding(
        binding,
        object_name=object_name,
        actual_reference_sha256=actual_reference_sha256,
    )
    capture = _affine_matrix(capture_root_matrix, "capture root matrix")
    current = _affine_matrix(current_root_matrix, "current root matrix")
    bake, bake_scale, bake_residual = _positive_similarity(
        baked_delta, "certification baked delta"
    )
    try:
        current_from_capture = current @ np.linalg.inv(capture)
    except np.linalg.LinAlgError as exc:
        raise AuthoredTelemetryError(
            "capture root matrix is not invertible", code="malformed_matrix"
        ) from exc
    current_from_capture, pose_scale, pose_residual = _positive_similarity(
        current_from_capture, "current-from-capture root transform"
    )
    source = _triangles(reference_triangles, "reference triangles")
    target = _triangles(current_triangles, "current triangles")
    correspondence = triangle_bijection(
        _apply(current_from_capture, source),
        target,
        tolerance_m=tolerance_m,
    )
    delivered = bake @ current_from_capture
    delivered, delivered_scale, delivered_residual = _positive_similarity(
        delivered, "capture-to-delivered transform"
    )
    return {
        "delta": delivered,
        "current_from_capture": current_from_capture,
        "object_id": object_id,
        "mesh_name": object_name,
        "revision": revision,
        "transaction_id": transaction_id,
        "reference_pose": "authored_revision",
        **correspondence,
        "pose_uniform_scale": pose_scale,
        "baked_uniform_scale": bake_scale,
        "delivered_uniform_scale": delivered_scale,
        "pose_similarity_residual": pose_residual,
        "baked_similarity_residual": bake_residual,
        "delivered_similarity_residual": delivered_residual,
    }


__all__ = [
    "AuthoredTelemetryError",
    "measure_authored_delivery",
    "triangle_bijection",
]
