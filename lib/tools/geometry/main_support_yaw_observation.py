"""Immutable source-image yaw evidence for the initializer's main support.

The old runtime yaw hint fitted PCA to the complete support mask.  That mask can
contain legs, bases, crop-created contours, and holes made by foreground objects.
This module instead measures straight, observable boundary segments of the support's
canonical top plane.  It deliberately does *not* inspect the generated Blender scene.

The resulting ``main_support_yaw_observation.json`` is a required, digest-bound
preprocessing artifact.  A two-family, corner-supported observation may be strong;
one family and whole-mask PCA are advisory only.  PCA and observable edges carry the
same independence group so a downstream resolver cannot count the same mask twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from lib.tools.geometry import moge_camera as mc

MAIN_SUPPORT_YAW_OBSERVATION_FILENAME = "main_support_yaw_observation.json"
MAIN_SUPPORT_YAW_OBSERVATION_SCHEMA_VERSION = 1
MAIN_SUPPORT_YAW_EXTRACTOR_VERSION = "observable_top_edges_v3_footprint"

SUPPORT_IMAGE_GEOMETRY_GROUP = "support_image_geometry"

# These are intentionally extractor thresholds, not runtime tolerances.  They are
# stamped into the artifact binding so changing one invalidates staged preprocessing.
EXTRACTOR_PARAMETERS: dict[str, float | int] = {
    "top_plane_min_tolerance_m": 0.025,
    "top_plane_max_tolerance_m": 0.060,
    "top_plane_quantile": 0.25,
    "top_plane_quantile_multiplier": 2.5,
    "occluder_dilation_px": 3,
    "foreground_clearance_min_m": 0.025,
    "frame_clearance_px": 2,
    "frame_clearance_min_dimension_fraction": 0.010,
    "minimum_segment_frame_diag_fraction": 0.060,
    "strong_segment_frame_diag_fraction": 0.080,
    "strong_segment_top_bbox_diag_fraction": 0.150,
    "maximum_fit_rmse_frame_diag_fraction": 0.002,
    "minimum_boundary_support_fraction": 0.80,
    "minimum_top_face_support_fraction": 0.80,
    "minimum_occluder_clear_fraction": 0.80,
    "minimum_world_incidence": 0.05,
    "maximum_strong_angular_uncertainty_degrees": 3.0,
    "maximum_weak_angular_uncertainty_degrees": 8.0,
    "family_parallel_tolerance_degrees": 8.0,
    "weak_family_consensus_degrees": 6.0,
    "pair_orthogonality_tolerance_degrees": 8.0,
    "pair_folded_disagreement_degrees": 3.0,
    "pair_corner_top_bbox_diag_fraction": 0.020,
    "axisymmetric_circle_residual_fraction": 0.05,
    "axisymmetric_max_angular_gap_degrees": 60.0,
    # Occluder-silhouette test (2026-09-17): a walked boundary run is an object's rim,
    # not a table edge, when the band just BEYOND it (away from the top face) is above
    # the plane AND its world XY lies inside the visible top face's convex footprint.
    # A wall flush behind a true edge is above the plane but outside the footprint;
    # the 2 cm inset keeps a flush wall base from counting as inside.
    "silhouette_probe_min_px": 6,
    "silhouette_probe_max_px": 24,
    "silhouette_probe_steps": 5,
    "silhouette_footprint_inset_m": 0.02,
    "maximum_silhouette_fraction": 0.30,
    "maximum_segments": 24,
    "maximum_candidates": 12,
}

_DECISION_PAIRS = {
    ("confirmed", "required"),
    ("unverified", "advisory"),
    ("conflicting", "advisory"),
    ("not_applicable", "none"),
}


class YawObservationArtifactError(RuntimeError):
    """The source yaw artifact is missing, malformed, or stale."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise YawObservationArtifactError(
            f"yaw observation contains non-canonical JSON: {exc}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(_canonical_json(list(arr.shape)))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _instance_id(record: dict[str, Any]) -> str:
    return f"{record.get('category')}#{record.get('instance')}"


def _build_name(surface_id: str) -> str:
    # Root surfaces use these short names in Blender and in the initializer contract.
    return surface_id.replace("#", "_").replace(" ", "_")


def _unit_xy(vector: Iterable[float]) -> np.ndarray | None:
    value = np.asarray(list(vector), dtype=np.float64)
    if value.shape[0] < 2 or not np.isfinite(value[:2]).all():
        return None
    norm = float(np.linalg.norm(value[:2]))
    if norm < 1e-9:
        return None
    run = value[:2] / norm
    # A line is unoriented.  Pick one sign for deterministic JSON/digests.
    if run[0] < -1e-12 or (abs(run[0]) <= 1e-12 and run[1] < 0):
        run = -run
    return run


def _line_angle_degrees(run: Iterable[float]) -> float:
    value = _unit_xy(run)
    if value is None:
        return math.nan
    return math.degrees(math.atan2(float(value[1]), float(value[0]))) % 180.0


def _fold_angle_degrees(run: Iterable[float]) -> float:
    return _line_angle_degrees(run) % 90.0


def _angle_delta_mod_period(a: float, b: float, period: float) -> float:
    return abs((float(a) - float(b) + period / 2.0) % period - period / 2.0)


def _mean_folded_angle(angles: list[float]) -> float:
    # Doubled once for unoriented lines, and again because the target is modulo 90.
    radians = np.radians(np.asarray(angles, dtype=np.float64) * 4.0)
    vector = np.array([np.cos(radians).sum(), np.sin(radians).sum()])
    if float(np.linalg.norm(vector)) < 1e-9:
        return min(float(angle) % 90.0 for angle in angles)
    return (math.degrees(math.atan2(float(vector[1]), float(vector[0]))) / 4.0) % 90.0


def _run_from_folded_angle(angle: float) -> list[float]:
    radians = math.radians(float(angle) % 90.0)
    return [round(math.cos(radians), 8), round(math.sin(radians), 8), 0.0]


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    value = np.asarray(mask).astype(bool)
    if value.shape == (height, width):
        return value
    resized = cv2.resize(
        value.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def _top_face_mask(
    main_mask: np.ndarray,
    world_points: np.ndarray,
    *,
    top_plane_z_m: float,
) -> tuple[np.ndarray, float]:
    finite = np.isfinite(world_points).all(axis=2)
    selected = main_mask & finite
    if int(selected.sum()) < 20:
        return np.zeros_like(main_mask), float(
            EXTRACTOR_PARAMETERS["top_plane_min_tolerance_m"]
        )
    distances = np.abs(world_points[..., 2][selected] - float(top_plane_z_m))
    quantile = float(
        np.quantile(distances, float(EXTRACTOR_PARAMETERS["top_plane_quantile"]))
    )
    tolerance = max(
        float(EXTRACTOR_PARAMETERS["top_plane_min_tolerance_m"]),
        quantile * float(EXTRACTOR_PARAMETERS["top_plane_quantile_multiplier"]),
    )
    tolerance = min(tolerance, float(EXTRACTOR_PARAMETERS["top_plane_max_tolerance_m"]))
    result = selected & (
        np.abs(world_points[..., 2] - float(top_plane_z_m)) <= tolerance
    )
    # Isolated depth speckles manufacture tiny polygon edges.  Keep components that are
    # at least 0.1% of the image; never close/fill holes, because those are visibility
    # evidence and must remain available to the occluder filter.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        result.astype(np.uint8), connectivity=8
    )
    minimum = max(12, int(result.size * 0.001))
    clean = np.zeros_like(result)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            clean |= labels == label
    return clean, tolerance


def _mask_bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0.0
    return math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def _arc_between(contour: np.ndarray, start: int, stop: int) -> np.ndarray:
    if stop > start:
        return contour[start : stop + 1]
    return np.concatenate((contour[start:], contour[: stop + 1]), axis=0)


def _nearest_mask_fraction(
    points_xy: np.ndarray, distance_to_true: np.ndarray, maximum_distance: float
) -> float:
    if len(points_xy) == 0:
        return 0.0
    h, w = distance_to_true.shape
    x = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, h - 1)
    return float(np.mean(distance_to_true[y, x] <= maximum_distance))


def _line_samples(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(length)) + 1)
    t = np.linspace(0.0, 1.0, count)[:, None]
    return a[None, :] * (1.0 - t) + b[None, :] * t


def _top_side_support(
    samples: np.ndarray, top_mask: np.ndarray, normal: np.ndarray
) -> float:
    h, w = top_mask.shape
    scores: list[float] = []
    for sign in (-1.0, 1.0):
        supported = np.zeros(len(samples), dtype=bool)
        for offset in (1.0, 2.0, 3.0):
            probe = samples + sign * offset * normal[None, :]
            x = np.rint(probe[:, 0]).astype(int)
            y = np.rint(probe[:, 1]).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            supported[inside] |= top_mask[y[inside], x[inside]]
        scores.append(float(np.mean(supported)))
    return max(scores)


def _world_segment(
    a: np.ndarray,
    b: np.ndarray,
    camera: dict[str, Any],
    *,
    pixel_sigma: float,
    top_plane_z_m: float,
) -> tuple[list[float] | None, float | None, float | None]:
    try:
        wa, _, ia = mc.intersect_pixel_ray_with_plane(
            float(a[0]),
            float(a[1]),
            camera,
            plane_point=(0.0, 0.0, float(top_plane_z_m)),
        )
        wb, _, ib = mc.intersect_pixel_ray_with_plane(
            float(b[0]),
            float(b[1]),
            camera,
            plane_point=(0.0, 0.0, float(top_plane_z_m)),
        )
    except ValueError:
        return None, None, None
    base = _unit_xy(wb[:2] - wa[:2])
    if base is None:
        return None, None, None
    base_angle = _line_angle_degrees(base)
    pixel_run = b - a
    pixel_length = float(np.linalg.norm(pixel_run))
    normal = (
        np.array([-pixel_run[1], pixel_run[0]], dtype=np.float64) / pixel_length
        if pixel_length > 1e-9
        else np.array([0.0, 1.0])
    )
    perturbed: list[float] = []
    sigma = max(1.0, float(pixel_sigma))
    for sa in (-sigma, sigma):
        for sb in (-sigma, sigma):
            try:
                pa, _, _ = mc.intersect_pixel_ray_with_plane(
                    *(a + sa * normal),
                    camera,
                    plane_point=(0.0, 0.0, float(top_plane_z_m)),
                )
                pb, _, _ = mc.intersect_pixel_ray_with_plane(
                    *(b + sb * normal),
                    camera,
                    plane_point=(0.0, 0.0, float(top_plane_z_m)),
                )
            except ValueError:
                continue
            run = _unit_xy(pb[:2] - pa[:2])
            if run is not None:
                perturbed.append(_line_angle_degrees(run))
    uncertainty = max(
        [_angle_delta_mod_period(angle, base_angle, 180.0) for angle in perturbed]
        or [180.0]
    )
    return (
        [round(float(base[0]), 8), round(float(base[1]), 8), 0.0],
        float(uncertainty),
        min(float(ia), float(ib)),
    )


def _footprint_hull(top_mask: np.ndarray, world_points: np.ndarray) -> np.ndarray | None:
    """Convex hull (world XY, float32 Nx2) of the visible top face; None if degenerate."""
    finite = np.isfinite(world_points).all(axis=2)
    ys, xs = np.where(top_mask & finite)
    if len(xs) < 3:
        return None
    if len(xs) > 20000:
        step = int(math.ceil(len(xs) / 20000.0))
        ys, xs = ys[::step], xs[::step]
    xy = world_points[ys, xs, :2].astype(np.float32)
    hull = cv2.convexHull(xy).reshape(-1, 2)
    return hull if len(hull) >= 3 else None


def _silhouette_fraction(
    samples: np.ndarray,
    normal: np.ndarray,
    *,
    top_mask: np.ndarray,
    world_points: np.ndarray,
    footprint_hull: np.ndarray | None,
    top_plane_z_m: float,
    plane_clearance_m: float,
) -> float:
    """Fraction of segment samples whose beyond-edge band holds geometry standing ON the table.

    The inward side is the one the top face supports at 1-3 px.  Probes then step
    outward ``silhouette_probe_min_px..max_px``; a sample counts when any probe lands on
    finite geometry higher than ``top_plane_z_m + plane_clearance_m`` whose world XY is at
    least ``silhouette_footprint_inset_m`` inside the footprint hull.  Out-of-frame probes
    never count, so a genuine edge at the frame border is not penalised.
    """
    if footprint_hull is None or len(samples) == 0:
        return 0.0
    h, w = top_mask.shape
    support: list[float] = []
    for sign in (-1.0, 1.0):
        hit = np.zeros(len(samples), dtype=bool)
        for offset in (1.0, 2.0, 3.0):
            probe = samples + sign * offset * normal[None, :]
            x = np.rint(probe[:, 0]).astype(int)
            y = np.rint(probe[:, 1]).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            hit[inside] |= top_mask[y[inside], x[inside]]
        support.append(float(np.mean(hit)))
    inward = -1.0 if support[0] >= support[1] else 1.0
    offsets = np.linspace(
        float(EXTRACTOR_PARAMETERS["silhouette_probe_min_px"]),
        float(EXTRACTOR_PARAMETERS["silhouette_probe_max_px"]),
        int(EXTRACTOR_PARAMETERS["silhouette_probe_steps"]),
    )
    inset = float(EXTRACTOR_PARAMETERS["silhouette_footprint_inset_m"])
    z_min = float(top_plane_z_m) + float(plane_clearance_m)
    hit = np.zeros(len(samples), dtype=bool)
    for offset in offsets:
        probe = samples - inward * float(offset) * normal[None, :]
        x = np.rint(probe[:, 0]).astype(int)
        y = np.rint(probe[:, 1]).astype(int)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        pw = np.full((len(samples), 3), np.nan)
        pw[inside] = world_points[y[inside], x[inside]]
        above = np.isfinite(pw).all(axis=1) & (pw[:, 2] > z_min)
        for index in np.where(above & ~hit)[0]:
            if (
                cv2.pointPolygonTest(
                    footprint_hull, (float(pw[index, 0]), float(pw[index, 1])), True
                )
                >= inset
            ):
                hit[index] = True
    return float(np.mean(hit))


def _fit_segment(
    arc: np.ndarray,
    *,
    top_mask: np.ndarray,
    boundary_distance: np.ndarray,
    occluder_distance: np.ndarray,
    camera: dict[str, Any],
    frame_diag: float,
    top_bbox_diag: float,
    top_plane_z_m: float,
    world_points: np.ndarray,
    footprint_hull: np.ndarray | None,
    plane_clearance_m: float,
) -> dict[str, Any] | None:
    points = np.asarray(arc, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    projection = (points - center) @ direction
    a = center + float(projection.min()) * direction
    b = center + float(projection.max()) * direction
    if tuple(a) > tuple(b):
        a, b = b, a
    pixel_run = b - a
    length = float(np.linalg.norm(pixel_run))
    if length < 2.0:
        return None
    normal = np.array([-pixel_run[1], pixel_run[0]], dtype=np.float64) / length
    residual = np.abs((points - center) @ normal)
    rmse = float(math.sqrt(float(np.mean(residual**2))))
    samples = _line_samples(a, b)
    boundary_support = _nearest_mask_fraction(samples, boundary_distance, 2.0)
    top_support = _top_side_support(samples, top_mask, normal)
    silhouette_fraction = _silhouette_fraction(
        samples,
        normal,
        top_mask=top_mask,
        world_points=world_points,
        footprint_hull=footprint_hull,
        top_plane_z_m=top_plane_z_m,
        plane_clearance_m=plane_clearance_m,
    )

    h, w = top_mask.shape
    sx = np.rint(samples[:, 0]).astype(int)
    sy = np.rint(samples[:, 1]).astype(int)
    in_frame = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    clearance = max(
        int(EXTRACTOR_PARAMETERS["frame_clearance_px"]),
        int(
            math.ceil(
                min(h, w)
                * float(EXTRACTOR_PARAMETERS["frame_clearance_min_dimension_fraction"])
            )
        ),
    )
    frame_clear = bool(
        np.all(in_frame)
        and np.all(sx >= clearance)
        and np.all(sx < w - clearance)
        and np.all(sy >= clearance)
        and np.all(sy < h - clearance)
    )
    occluder_clear = np.zeros(len(samples), dtype=bool)
    occluder_clear[in_frame] = occluder_distance[sy[in_frame], sx[in_frame]] > float(
        EXTRACTOR_PARAMETERS["occluder_dilation_px"]
    )
    occluder_clear_fraction = float(np.mean(occluder_clear))
    world_run, angular_uncertainty, incidence = _world_segment(
        a,
        b,
        camera,
        pixel_sigma=max(rmse, 1.0),
        top_plane_z_m=top_plane_z_m,
    )

    length_frame = length / max(frame_diag, 1.0)
    length_bbox = length / max(top_bbox_diag, 1.0)
    reasons: list[str] = []
    if length_frame < float(
        EXTRACTOR_PARAMETERS["minimum_segment_frame_diag_fraction"]
    ):
        reasons.append("segment_too_short")
    if rmse > max(
        0.75,
        float(EXTRACTOR_PARAMETERS["maximum_fit_rmse_frame_diag_fraction"])
        * frame_diag,
    ):
        reasons.append("line_fit_residual_high")
    if boundary_support < float(
        EXTRACTOR_PARAMETERS["minimum_boundary_support_fraction"]
    ):
        reasons.append("boundary_support_low")
    if top_support < float(EXTRACTOR_PARAMETERS["minimum_top_face_support_fraction"]):
        reasons.append("top_face_support_low")
    if occluder_clear_fraction < float(
        EXTRACTOR_PARAMETERS["minimum_occluder_clear_fraction"]
    ):
        reasons.append("occluder_contaminated")
    if silhouette_fraction > float(EXTRACTOR_PARAMETERS["maximum_silhouette_fraction"]):
        reasons.append("occluder_silhouette")
    if not frame_clear:
        reasons.append("segment_touches_frame")
    if world_run is None or angular_uncertainty is None or incidence is None:
        reasons.append("world_unprojection_failed")
    elif incidence < float(EXTRACTOR_PARAMETERS["minimum_world_incidence"]):
        reasons.append("grazing_camera_ray")
    elif angular_uncertainty > float(
        EXTRACTOR_PARAMETERS["maximum_weak_angular_uncertainty_degrees"]
    ):
        reasons.append("angular_uncertainty_high")
    eligible = not reasons
    strong_eligible = bool(
        eligible
        and length_frame
        >= float(EXTRACTOR_PARAMETERS["strong_segment_frame_diag_fraction"])
        and length_bbox
        >= float(EXTRACTOR_PARAMETERS["strong_segment_top_bbox_diag_fraction"])
        and angular_uncertainty is not None
        and angular_uncertainty
        <= float(EXTRACTOR_PARAMETERS["maximum_strong_angular_uncertainty_degrees"])
    )
    strong_reasons: list[str] = []
    if eligible and not strong_eligible:
        if length_frame < float(
            EXTRACTOR_PARAMETERS["strong_segment_frame_diag_fraction"]
        ) or length_bbox < float(
            EXTRACTOR_PARAMETERS["strong_segment_top_bbox_diag_fraction"]
        ):
            strong_reasons.append("insufficient_strong_span")
        if angular_uncertainty is not None and angular_uncertainty > float(
            EXTRACTOR_PARAMETERS["maximum_strong_angular_uncertainty_degrees"]
        ):
            strong_reasons.append("insufficient_strong_angular_stability")

    return {
        "endpoints_px": [
            [round(float(a[0]), 3), round(float(a[1]), 3)],
            [round(float(b[0]), 3), round(float(b[1]), 3)],
        ],
        "endpoints_normalized": [
            [
                round(float(a[0]) / max(w - 1, 1), 6),
                round(float(a[1]) / max(h - 1, 1), 6),
            ],
            [
                round(float(b[0]) / max(w - 1, 1), 6),
                round(float(b[1]) / max(h - 1, 1), 6),
            ],
        ],
        "world_run": world_run,
        "length_frame_diag": round(length_frame, 6),
        "length_top_bbox_diag": round(length_bbox, 6),
        "fit_rmse_px": round(rmse, 4),
        "boundary_support_fraction": round(boundary_support, 4),
        "top_face_support_fraction": round(top_support, 4),
        "occluder_clear_fraction": round(occluder_clear_fraction, 4),
        "occluder_silhouette_fraction": round(silhouette_fraction, 4),
        "frame_clear": frame_clear,
        "angular_uncertainty_degrees": (
            round(float(angular_uncertainty), 4)
            if angular_uncertainty is not None
            else None
        ),
        "minimum_plane_incidence": round(float(incidence), 6)
        if incidence is not None
        else None,
        "eligible": eligible,
        "strong_eligible": strong_eligible,
        "reason_codes": reasons + strong_reasons,
    }


def _extract_segments(
    top_mask: np.ndarray,
    occluder_mask: np.ndarray,
    camera: dict[str, Any],
    *,
    top_plane_z_m: float,
    world_points: np.ndarray,
    plane_clearance_m: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    h, w = top_mask.shape
    footprint_hull = _footprint_hull(top_mask, world_points)
    frame_diag = math.hypot(w, h)
    bbox_diag = _mask_bbox_diag(top_mask)
    boundary = top_mask & ~binary_erosion(top_mask, structure=np.ones((3, 3), bool))
    boundary_distance = distance_transform_edt(~boundary)
    occluder_distance = distance_transform_edt(~occluder_mask)
    contours, _ = cv2.findContours(
        top_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    segments: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    for contour_cv in contours:
        contour = contour_cv.reshape(-1, 2)
        if len(contour) < 8:
            continue
        epsilon = max(1.25, 0.004 * bbox_diag)
        approx = cv2.approxPolyDP(contour_cv, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        indices: list[int] = []
        cursor = 0
        for vertex in approx:
            matches = np.where(np.all(contour == vertex, axis=1))[0]
            after = matches[matches >= cursor]
            index = int(after[0] if len(after) else matches[0])
            indices.append(index)
            cursor = index + 1
        for pos, start in enumerate(indices):
            stop = indices[(pos + 1) % len(indices)]
            arc = _arc_between(contour, start, stop)
            segment = _fit_segment(
                arc,
                top_mask=top_mask,
                boundary_distance=boundary_distance,
                occluder_distance=occluder_distance,
                camera=camera,
                frame_diag=frame_diag,
                top_bbox_diag=bbox_diag,
                top_plane_z_m=top_plane_z_m,
                world_points=world_points,
                footprint_hull=footprint_hull,
                plane_clearance_m=plane_clearance_m,
            )
            if segment is None:
                rejected["degenerate_segment"] += 1
                continue
            if not segment["eligible"]:
                rejected.update(segment["reason_codes"])
            segments.append(segment)

    # Keep the artifact bounded, preferring usable long segments.  Re-number only after
    # sorting so candidate IDs are deterministic across OpenCV contour enumeration.
    segments.sort(
        key=lambda row: (
            not bool(row["strong_eligible"]),
            not bool(row["eligible"]),
            -float(row["length_frame_diag"]),
            row["endpoints_px"],
        )
    )
    segments = segments[: int(EXTRACTOR_PARAMETERS["maximum_segments"])]
    for index, segment in enumerate(segments):
        segment["segment_id"] = f"source-edge-{index:02d}"
    _assign_segment_families(segments)
    return segments, rejected


def _assign_segment_families(segments: list[dict[str, Any]]) -> None:
    """Assign deterministic physical (modulo-180) direction families in place.

    These family IDs are consumed only as correspondence topology.  They deliberately
    do not fold perpendicular directions together modulo 90: the machine verifier must
    be able to prove that two submitted edges are nonparallel physical families.
    """

    usable = [
        index
        for index, segment in enumerate(segments)
        if segment.get("world_run") is not None
    ]
    adjacency = {index: {index} for index in usable}
    for pos, left in enumerate(usable):
        angle_left = _line_angle_degrees(segments[left]["world_run"])
        for right in usable[pos + 1 :]:
            angle_right = _line_angle_degrees(segments[right]["world_run"])
            if _angle_delta_mod_period(angle_left, angle_right, 180.0) <= float(
                EXTRACTOR_PARAMETERS["family_parallel_tolerance_degrees"]
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[int]] = []
    unseen = set(usable)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        unseen -= component
        components.append(sorted(component))
    components.sort(
        key=lambda component: min(
            _line_angle_degrees(segments[index]["world_run"]) for index in component
        )
    )
    for segment in segments:
        segment["family"] = "unqualified"
    for family_index, component in enumerate(components):
        for segment_index in component:
            segments[segment_index]["family"] = f"edge_family_{family_index}"


def _line_intersection(a: dict[str, Any], b: dict[str, Any]) -> np.ndarray | None:
    pa = np.asarray(a["endpoints_px"][0], dtype=np.float64)
    qa = np.asarray(a["endpoints_px"][1], dtype=np.float64)
    pb = np.asarray(b["endpoints_px"][0], dtype=np.float64)
    qb = np.asarray(b["endpoints_px"][1], dtype=np.float64)
    ra, rb = qa - pa, qb - pb
    determinant = float(ra[0] * rb[1] - ra[1] * rb[0])
    if abs(determinant) < 1e-8:
        return None
    delta = pb - pa
    t = float((delta[0] * rb[1] - delta[1] * rb[0]) / determinant)
    return pa + t * ra


def _qualified_pairs(
    segments: list[dict[str, Any]], top_mask: np.ndarray
) -> list[dict[str, Any]]:
    boundary = top_mask & ~binary_erosion(top_mask, structure=np.ones((3, 3), bool))
    distance = distance_transform_edt(~boundary)
    h, w = top_mask.shape
    bbox_diag = _mask_bbox_diag(top_mask)
    corner_limit = max(
        3.0,
        bbox_diag * float(EXTRACTOR_PARAMETERS["pair_corner_top_bbox_diag_fraction"]),
    )
    result: list[dict[str, Any]] = []
    strong = [segment for segment in segments if segment["strong_eligible"]]
    for index, a in enumerate(strong):
        for b in strong[index + 1 :]:
            if a["family"] == b["family"]:
                continue
            angle_a = _line_angle_degrees(a["world_run"])
            angle_b = _line_angle_degrees(b["world_run"])
            angle = _angle_delta_mod_period(angle_a, angle_b, 180.0)
            acute = min(angle, 180.0 - angle)
            orthogonality_error = abs(90.0 - acute)
            folded_disagreement = _angle_delta_mod_period(
                _fold_angle_degrees(a["world_run"]),
                _fold_angle_degrees(b["world_run"]),
                90.0,
            )
            if orthogonality_error > float(
                EXTRACTOR_PARAMETERS["pair_orthogonality_tolerance_degrees"]
            ) or folded_disagreement > float(
                EXTRACTOR_PARAMETERS["pair_folded_disagreement_degrees"]
            ):
                continue
            corner = _line_intersection(a, b)
            if corner is None:
                continue
            endpoints_a = np.asarray(a["endpoints_px"], dtype=np.float64)
            endpoints_b = np.asarray(b["endpoints_px"], dtype=np.float64)
            endpoint_distance = max(
                float(np.min(np.linalg.norm(endpoints_a - corner, axis=1))),
                float(np.min(np.linalg.norm(endpoints_b - corner, axis=1))),
            )
            ix, iy = int(round(float(corner[0]))), int(round(float(corner[1])))
            if 0 <= ix < w and 0 <= iy < h:
                mask_distance = float(distance[iy, ix])
            else:
                mask_distance = math.inf
            if endpoint_distance > corner_limit or mask_distance > corner_limit:
                continue
            target_angle = _mean_folded_angle(
                [
                    _fold_angle_degrees(a["world_run"]),
                    _fold_angle_degrees(b["world_run"]),
                ]
            )
            uncertainty = max(
                float(a["angular_uncertainty_degrees"]),
                float(b["angular_uncertainty_degrees"]),
                folded_disagreement,
            )
            result.append(
                {
                    "segment_ids": [a["segment_id"], b["segment_id"]],
                    "world_angle_degrees": round(acute, 4),
                    "orthogonality_error_degrees": round(orthogonality_error, 4),
                    "folded_run_disagreement_degrees": round(folded_disagreement, 4),
                    "corner_px": [
                        round(float(corner[0]), 3),
                        round(float(corner[1]), 3),
                    ],
                    "corner_endpoint_distance_px": round(endpoint_distance, 4),
                    "corner_mask_distance_px": round(mask_distance, 4),
                    "target_folded_angle_degrees": round(target_angle, 6),
                    "angular_uncertainty_degrees": round(uncertainty, 4),
                    "score": round(
                        float(a["length_frame_diag"])
                        + float(b["length_frame_diag"])
                        - 0.01 * uncertainty,
                        6,
                    ),
                }
            )
    result.sort(key=lambda row: (-float(row["score"]), row["segment_ids"]))
    return result


def _axisymmetric_top(
    top_mask: np.ndarray,
    occluder_mask: np.ndarray,
    camera: dict[str, Any],
    *,
    top_plane_z_m: float,
) -> dict[str, Any]:
    contours, _ = cv2.findContours(
        top_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return {"confirmed": False, "reason_codes": ["no_top_contour"]}
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    h, w = top_mask.shape
    clearance = max(
        int(EXTRACTOR_PARAMETERS["frame_clearance_px"]),
        int(
            math.ceil(
                min(h, w)
                * float(EXTRACTOR_PARAMETERS["frame_clearance_min_dimension_fraction"])
            )
        ),
    )
    frame_clear = bool(
        np.all(contour[:, 0] >= clearance)
        and np.all(contour[:, 0] < w - clearance)
        and np.all(contour[:, 1] >= clearance)
        and np.all(contour[:, 1] < h - clearance)
    )
    occ_distance = distance_transform_edt(~occluder_mask)
    cx = np.clip(contour[:, 0], 0, w - 1)
    cy = np.clip(contour[:, 1], 0, h - 1)
    occluder_clear = float(
        np.mean(
            occ_distance[cy, cx] > float(EXTRACTOR_PARAMETERS["occluder_dilation_px"])
        )
    )
    if not frame_clear or occluder_clear < 0.95 or len(contour) < 24:
        return {
            "confirmed": False,
            "frame_clear": frame_clear,
            "occluder_clear_fraction": round(occluder_clear, 4),
            "reason_codes": ["incomplete_contour"],
        }
    stride = max(1, len(contour) // 512)
    world: list[np.ndarray] = []
    for x, y in contour[::stride]:
        try:
            point, _, incidence = mc.intersect_pixel_ray_with_plane(
                float(x),
                float(y),
                camera,
                plane_point=(0.0, 0.0, float(top_plane_z_m)),
            )
        except ValueError:
            continue
        if incidence >= float(EXTRACTOR_PARAMETERS["minimum_world_incidence"]):
            world.append(point[:2])
    if len(world) < 16:
        return {"confirmed": False, "reason_codes": ["contour_unprojection_failed"]}
    xy = np.asarray(world, dtype=np.float64)
    # Algebraic least-squares circle in the canonical support plane.
    design = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    target = np.sum(xy**2, axis=1)
    center_x, center_y, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    radius_sq = float(constant + center_x**2 + center_y**2)
    if radius_sq <= 1e-10:
        return {"confirmed": False, "reason_codes": ["degenerate_circle_fit"]}
    radius = math.sqrt(radius_sq)
    radii = np.linalg.norm(xy - np.array([center_x, center_y]), axis=1)
    residual = float(math.sqrt(float(np.mean((radii - radius) ** 2))) / radius)
    angles = np.sort(
        np.mod(
            np.degrees(np.arctan2(xy[:, 1] - center_y, xy[:, 0] - center_x)),
            360.0,
        )
    )
    gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
    maximum_gap = float(gaps.max())
    confirmed = bool(
        residual <= float(EXTRACTOR_PARAMETERS["axisymmetric_circle_residual_fraction"])
        and maximum_gap
        <= float(EXTRACTOR_PARAMETERS["axisymmetric_max_angular_gap_degrees"])
    )
    return {
        "confirmed": confirmed,
        "world_center_xy": [round(float(center_x), 6), round(float(center_y), 6)],
        "world_radius_m": round(radius, 6),
        "relative_radial_rmse": round(residual, 6),
        "maximum_angular_gap_degrees": round(maximum_gap, 4),
        "frame_clear": frame_clear,
        "occluder_clear_fraction": round(occluder_clear, 4),
        "reason_codes": [] if confirmed else ["contour_not_axisymmetric"],
    }


def _mask_pca(top_mask: np.ndarray, world_points: np.ndarray) -> dict[str, Any]:
    selected = top_mask & np.isfinite(world_points).all(axis=2)
    xy = np.asarray(world_points[selected, :2], dtype=np.float64)
    if len(xy) < 20:
        return {
            "available": False,
            "run_world": None,
            "eccentricity": None,
            "reason_codes": ["insufficient_top_points"],
        }
    covariance = np.cov(xy - np.median(xy, axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major, minor = float(values[order[0]]), max(float(values[order[1]]), 1e-12)
    run = _unit_xy(vectors[:, order[0]])
    eccentricity = math.sqrt(max(major, 0.0) / minor)
    return {
        "available": run is not None,
        "run_world": (
            [round(float(run[0]), 8), round(float(run[1]), 8), 0.0]
            if run is not None
            else None
        ),
        "eccentricity": round(eccentricity, 6),
        "reason_codes": ["diagnostic_only_blanket_pca"],
    }


def _empty_observation(
    *,
    source_binding: dict[str, Any],
    main_surface: dict[str, Any],
    applicability_status: str,
    reason_codes: list[str],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    not_applicable = applicability_status == "not_applicable"
    return {
        "schema_version": MAIN_SUPPORT_YAW_OBSERVATION_SCHEMA_VERSION,
        "extractor_version": MAIN_SUPPORT_YAW_EXTRACTOR_VERSION,
        "source_binding": source_binding,
        "main_surface": main_surface,
        "applicability": {
            "status": applicability_status,
            "reason_codes": list(reason_codes),
        },
        "candidates": [],
        "decision": {
            "status": "not_applicable" if not_applicable else "unverified",
            "authority": "none" if not_applicable else "advisory",
            "target_run_world": None,
            "participating_evidence_ids": [],
            "reason_codes": list(reason_codes),
        },
        "segments": [],
        "pair": None,
        "diagnostics": diagnostics or {},
    }


def observe_main_support_yaw(
    main_mask: np.ndarray,
    world_points: np.ndarray,
    camera: dict[str, Any],
    *,
    main_surface_id: str,
    semantic_class: str,
    build_name: str | None = None,
    scene_form: str = "closeup:table",
    occluder_mask: np.ndarray | None = None,
    source_binding: dict[str, Any] | None = None,
    top_plane_z_m: float = 0.0,
) -> dict[str, Any]:
    """Measure immutable source yaw evidence from masks and canonical world points."""

    world = np.asarray(world_points, dtype=np.float64)
    if world.ndim != 3 or world.shape[2] != 3:
        raise ValueError("world_points must have shape (H, W, 3)")
    h, w = world.shape[:2]
    mask = _resize_mask(np.asarray(main_mask), h, w)
    occluder = (
        np.zeros((h, w), dtype=bool)
        if occluder_mask is None
        else _resize_mask(np.asarray(occluder_mask), h, w)
    )
    main_surface = {
        "id": str(main_surface_id),
        "build_name": str(build_name or _build_name(str(main_surface_id))),
        "semantic_class": str(semantic_class),
    }
    binding = dict(
        source_binding
        or {
            "main_surface_id": str(main_surface_id),
            "main_build_name": str(build_name or _build_name(str(main_surface_id))),
            "semantic_class": str(semantic_class),
            "scene_form": str(scene_form),
            "input_sha256": None,
            "main_mask_sha256": _sha256_array(mask.astype(np.uint8)),
            "occluder_set_sha256": _sha256_array(occluder.astype(np.uint8)),
            "pointmap_sha256": _sha256_array(world),
            "camera_sha256": _sha256_json(camera),
            "extractor_parameters_sha256": _sha256_json(EXTRACTOR_PARAMETERS),
        }
    )
    category = str(semantic_class).strip().casefold()
    if category in {"floor", "ground"} or str(scene_form).strip().casefold() == "room":
        return _empty_observation(
            source_binding=binding,
            main_surface=main_surface,
            applicability_status="not_applicable",
            reason_codes=["room_floor_has_no_footprint_yaw"],
            diagnostics={
                "main_mask_pixel_count": int(mask.sum()),
                "occluder_pixel_count": int(occluder.sum()),
            },
        )
    top_mask, plane_tolerance = _top_face_mask(mask, world, top_plane_z_m=top_plane_z_m)
    # Visibility is independent of reconstruction inventory: ignored robots still
    # occlude the tabletop. Positive height above its measured plane identifies
    # foreground depth boundaries, whereas the floor/background below it must remain
    # available as evidence of a genuine outer table edge. Never fill these holes.
    depth_occluder = np.isfinite(world).all(axis=2) & (
        world[..., 2]
        > float(top_plane_z_m)
        + max(
            plane_tolerance, float(EXTRACTOR_PARAMETERS["foreground_clearance_min_m"])
        )
    )
    occluder = occluder | depth_occluder
    diagnostics: dict[str, Any] = {
        "main_mask_pixel_count": int(mask.sum()),
        "top_face_pixel_count": int(top_mask.sum()),
        "occluder_pixel_count": int(occluder.sum()),
        "visibility_only_depth_pixel_count": int(depth_occluder.sum()),
        "top_plane_z_m": round(float(top_plane_z_m), 6),
        "top_plane_tolerance_m": round(float(plane_tolerance), 6),
    }
    if int(top_mask.sum()) < 20:
        return _empty_observation(
            source_binding=binding,
            main_surface=main_surface,
            applicability_status="unverified",
            reason_codes=["insufficient_visible_top_face"],
            diagnostics=diagnostics,
        )

    # Only a complete, unoccluded canonical circle makes yaw inapplicable.  Failure to
    # observe straight edges is never treated as proof of a disc.
    axisymmetric = _axisymmetric_top(
        top_mask, occluder, camera, top_plane_z_m=top_plane_z_m
    )
    diagnostics["axisymmetric_fit"] = axisymmetric
    pca = _mask_pca(top_mask, world)
    diagnostics["mask_pca"] = pca
    if axisymmetric.get("confirmed"):
        return _empty_observation(
            source_binding=binding,
            main_surface=main_surface,
            applicability_status="not_applicable",
            reason_codes=["complete_axisymmetric_top_contour"],
            diagnostics=diagnostics,
        )

    segments, rejected = _extract_segments(
        top_mask,
        occluder,
        camera,
        top_plane_z_m=top_plane_z_m,
        world_points=world,
        plane_clearance_m=max(
            plane_tolerance, float(EXTRACTOR_PARAMETERS["foreground_clearance_min_m"])
        ),
    )
    diagnostics["rejected_segment_reason_counts"] = dict(sorted(rejected.items()))
    pairs = _qualified_pairs(segments, top_mask)
    candidates: list[dict[str, Any]] = []
    for index, pair in enumerate(
        pairs[: int(EXTRACTOR_PARAMETERS["maximum_candidates"])]
    ):
        candidates.append(
            {
                "evidence_id": f"photo-edge-pair-{index}",
                "provider": "observable_top_edges",
                "independence_group": SUPPORT_IMAGE_GEOMETRY_GROUP,
                "run_world": _run_from_folded_angle(
                    float(pair["target_folded_angle_degrees"])
                ),
                "strength": "strong",
                "eligible": True,
                "angular_uncertainty_degrees": pair["angular_uncertainty_degrees"],
                "segment_ids": pair["segment_ids"],
                "reason_codes": ["two_nonparallel_corner_supported_edges"],
            }
        )

    eligible_segments = [segment for segment in segments if segment["eligible"]]
    if not pairs and eligible_segments:
        # Preserve one representative from every independently observed physical
        # direction family.  Selecting only the longest segment here would hide
        # incompatible contour evidence and could let an arbitrary short edge become
        # hard merely because an independent wall candidate happened to agree with it.
        # Families that are parallel modulo 180 already share one deterministic ID.
        family_representatives: dict[str, dict[str, Any]] = {}
        for segment in eligible_segments:
            family_representatives.setdefault(str(segment["family"]), segment)
        for index, family in enumerate(
            sorted(family_representatives)[
                : int(EXTRACTOR_PARAMETERS["maximum_candidates"])
            ]
        ):
            best = family_representatives[family]
            candidates.append(
                {
                    "evidence_id": f"photo-edge-family-{index}",
                    "provider": "observable_top_edges",
                    "independence_group": SUPPORT_IMAGE_GEOMETRY_GROUP,
                    "run_world": best["world_run"],
                    "strength": "weak",
                    "eligible": True,
                    "angular_uncertainty_degrees": best["angular_uncertainty_degrees"],
                    "segment_ids": [best["segment_id"]],
                    "reason_codes": ["single_observable_edge_family"],
                }
            )

    # Whole-mask PCA is retained for diagnostics/weak corroboration only.  When an
    # observable edge exists it is shadowed; both still declare the same independence
    # group so no consumer can manufacture two votes from one mask.
    if pca.get("available") and float(pca.get("eccentricity") or 0.0) >= 1.15:
        has_edge = any(
            candidate["provider"] == "observable_top_edges" for candidate in candidates
        )
        candidates.append(
            {
                "evidence_id": "photo-mask-pca",
                "provider": "mask_pca",
                "independence_group": SUPPORT_IMAGE_GEOMETRY_GROUP,
                "run_world": pca["run_world"],
                "strength": "weak",
                "eligible": not has_edge,
                "angular_uncertainty_degrees": None,
                "segment_ids": [],
                "reason_codes": [
                    "diagnostic_only_blanket_pca",
                    *(["shadowed_by_observable_edges"] if has_edge else []),
                ],
            }
        )

    decision: dict[str, Any]
    selected_pair: dict[str, Any] | None = None
    if pairs:
        pair_angles = [float(pair["target_folded_angle_degrees"]) for pair in pairs]
        center = _mean_folded_angle(pair_angles)
        spread = max(
            _angle_delta_mod_period(angle, center, 90.0) for angle in pair_angles
        )
        if spread > float(EXTRACTOR_PARAMETERS["pair_folded_disagreement_degrees"]):
            decision = {
                "status": "conflicting",
                "authority": "advisory",
                "target_run_world": None,
                "participating_evidence_ids": [
                    candidate["evidence_id"]
                    for candidate in candidates
                    if candidate["strength"] == "strong"
                ],
                "reason_codes": ["qualified_edge_pairs_disagree"],
            }
        else:
            selected_pair = pairs[0]
            participating = [
                candidate["evidence_id"]
                for candidate in candidates
                if candidate["strength"] == "strong" and candidate["eligible"]
            ]
            decision = {
                "status": "confirmed",
                "authority": "required",
                "target_run_world": _run_from_folded_angle(center),
                "participating_evidence_ids": participating,
                "reason_codes": ["two_nonparallel_corner_supported_edges"],
            }
    else:
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        image_edges = [
            candidate
            for candidate in eligible
            if candidate["provider"] == "observable_top_edges"
        ]
        if image_edges:
            angles = [
                _fold_angle_degrees(candidate["run_world"]) for candidate in image_edges
            ]
            center = _mean_folded_angle(angles)
            spread = max(
                _angle_delta_mod_period(angle, center, 90.0) for angle in angles
            )
            participating = [candidate["evidence_id"] for candidate in image_edges]
            if spread > float(EXTRACTOR_PARAMETERS["weak_family_consensus_degrees"]):
                decision = {
                    "status": "conflicting",
                    "authority": "advisory",
                    "target_run_world": None,
                    "participating_evidence_ids": participating,
                    "reason_codes": ["observable_edge_families_disagree"],
                }
            else:
                decision = {
                    "status": "unverified",
                    "authority": "advisory",
                    "target_run_world": _run_from_folded_angle(center),
                    "participating_evidence_ids": participating,
                    "reason_codes": [
                        "single_observable_edge_family"
                        if len(image_edges) == 1
                        else "incomplete_observable_edge_families_agree"
                    ],
                }
        else:
            target = eligible[0]["run_world"] if eligible else None
            decision = {
                "status": "unverified",
                "authority": "advisory",
                "target_run_world": target,
                "participating_evidence_ids": (
                    [eligible[0]["evidence_id"]] if eligible else []
                ),
                "reason_codes": [
                    "blanket_pca_is_weak_only"
                    if eligible
                    else "no_qualified_orientation_evidence"
                ],
            }

    artifact = {
        "schema_version": MAIN_SUPPORT_YAW_OBSERVATION_SCHEMA_VERSION,
        "extractor_version": MAIN_SUPPORT_YAW_EXTRACTOR_VERSION,
        "source_binding": binding,
        "main_surface": main_surface,
        "applicability": {"status": "applicable", "reason_codes": []},
        "candidates": candidates[: int(EXTRACTOR_PARAMETERS["maximum_candidates"])],
        "decision": decision,
        "segments": segments,
        "pair": selected_pair,
        "diagnostics": diagnostics,
    }
    return validate_main_support_yaw_observation(artifact)


def _resolve_record_mask(record: dict[str, Any], root: Path) -> Path:
    raw = record.get("mask_path")
    if not raw:
        raise YawObservationArtifactError(
            f"source instance {_instance_id(record)!r} has no mask_path"
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        # Staged caches may have rebased their root while preserving just the basename.
        local = root / "masks" / Path(str(raw)).name
        if local.is_file():
            path = local
    if not path.is_file():
        raise YawObservationArtifactError(f"source mask is missing: {path}")
    return path


def _load_declared_main_mask(path: Path, main_id: str) -> np.ndarray:
    """Load one authoritative main mask with a pipeline-level diagnostic.

    The yaw artifact is built before the runner's final inventory boundary, so a
    corrupt declared mask must not leak a raw NumPy exception to preprocessing.
    Intentional maskless roots never call this helper.
    """

    try:
        mask = np.asarray(np.load(path, allow_pickle=False))
    except Exception as exc:  # noqa: BLE001 - artifact error is the diagnostic
        raise YawObservationArtifactError(
            f"main support {main_id!r} declared mask is unreadable/corrupt: "
            f"{path} ({type(exc).__name__}: {exc})"
        ) from exc
    if mask.ndim != 2 or mask.size == 0:
        raise YawObservationArtifactError(
            f"main support {main_id!r} declared mask has invalid shape "
            f"{tuple(mask.shape)!r}: {path}"
        )
    return mask


def _occluder_binding(
    records: list[dict[str, Any]], root: Path, height: int, width: int
) -> tuple[np.ndarray, str]:
    union = np.zeros((height, width), dtype=bool)
    rows: list[dict[str, str]] = []
    for record in records:
        if record.get("kind") == "root_surface":
            continue
        try:
            path = _resolve_record_mask(record, root)
            mask = _resize_mask(np.load(path), height, width)
        except (OSError, ValueError, YawObservationArtifactError):
            continue
        union |= mask
        rows.append(
            {
                "id": _instance_id(record),
                "mask_sha256": _sha256_array(mask.astype(np.uint8)),
            }
        )
    rows.sort(key=lambda row: row["id"])
    return union, _sha256_json(rows)


def _source_inputs(
    root: Path,
    *,
    graph: dict[str, Any] | None = None,
    masks_payload: dict[str, Any] | None = None,
    world_points: np.ndarray | None = None,
    camera_config: dict[str, Any] | None = None,
    input_path: str | os.PathLike[str] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any] | None,
    np.ndarray | None,
    dict[str, Any],
]:
    graph_payload = graph or json.loads((root / "scene_graph.json").read_text())
    masks = masks_payload or json.loads((root / "masks" / "masks.json").read_text())
    main_id = str(graph_payload.get("main_support_id") or "")
    records = [
        record for record in masks.get("instances", []) if isinstance(record, dict)
    ]
    # Optional supplied masks for excluded objects are visibility evidence only;
    # they never enter main-support selection, placement, or mesh generation.
    visibility_records = records + [
        record
        for record in masks.get("visibility_only_instances", [])
        if isinstance(record, dict)
    ]
    unmasked_roots = [
        record
        for record in masks.get("unmasked_root_surfaces", [])
        if isinstance(record, dict)
    ]
    main_records = [
        (record, intentionally_unmasked)
        for source, intentionally_unmasked in (
            (records, False),
            (unmasked_roots, True),
        )
        for record in source
        if _instance_id(record).casefold() == main_id.casefold()
    ]
    if len(main_records) != 1:
        raise YawObservationArtifactError(
            "main support must name exactly one retained source inventory record: "
            f"{main_id!r} matched {len(main_records)}"
        )
    main_record, intentionally_unmasked = main_records[0]
    # A record in ``unmasked_root_surfaces`` is an intentional semantic-only root,
    # unlike an ``instances`` record whose declared mask is unexpectedly absent.  The
    # latter remains an artifact-integrity error; the former produces a valid no-evidence
    # observation below and must never fabricate a source yaw target.
    main_path = (
        None if intentionally_unmasked else _resolve_record_mask(main_record, root)
    )
    routing = masks.get("routing") or {}
    scene_form = str(routing.get("form") or "closeup:table")

    points_path = root / "moge" / "points.npy"
    world = world_points
    camera = camera_config
    mj: dict[str, Any] = {}
    moge_path = root / "moge" / "moge.json"
    if moge_path.is_file():
        mj = json.loads(moge_path.read_text())
    if world is None and points_path.is_file():
        gravity = mj.get("gravity") or {}
        raw_r, raw_t = gravity.get("R"), gravity.get("T")
        if raw_r is not None and raw_t is not None:
            world = mc.moge_points_to_world(
                np.load(points_path),
                np.asarray(raw_r, dtype=np.float64),
                np.asarray(raw_t, dtype=np.float64),
            )
    if camera is None and mj:
        gravity = mj.get("gravity") or {}
        raw_r, raw_t = gravity.get("R"), gravity.get("T")
        if raw_r is not None and raw_t is not None:
            camera = mc.camera_config_from_moge(
                mj,
                R=np.asarray(raw_r, dtype=np.float64),
                T=np.asarray(raw_t, dtype=np.float64),
            )

    main_mask: np.ndarray | None = None
    occluder: np.ndarray | None = None
    if intentionally_unmasked:
        # No pixels participate in yaw extraction, so unrelated object masks cannot
        # become orientation evidence.  Keep the existing digest-shaped bindings (and
        # schema) by hashing explicit canonical sentinels; strict reload recomputes the
        # same values.
        occluder_digest = _sha256_json(
            {"status": "not_observed_for_intentionally_unmasked_main"}
        )
    elif world is not None:
        h, w = world.shape[:2]
        assert main_path is not None
        main_mask = _resize_mask(_load_declared_main_mask(main_path, main_id), h, w)
        occluder, occluder_digest = _occluder_binding(visibility_records, root, h, w)
    else:
        assert main_path is not None
        raw_mask = _load_declared_main_mask(main_path, main_id).astype(bool)
        main_mask = raw_mask
        h, w = raw_mask.shape
        occluder, occluder_digest = _occluder_binding(visibility_records, root, h, w)

    source_image = Path(input_path) if input_path is not None else root / "input.png"
    binding = {
        "main_surface_id": main_id,
        "main_build_name": _build_name(main_id),
        "semantic_class": str(main_record.get("category") or "surface"),
        "scene_form": scene_form,
        "input_sha256": _sha256_file(source_image) if source_image.is_file() else None,
        "main_mask_sha256": (
            _sha256_array(main_mask.astype(np.uint8))
            if main_mask is not None
            else _sha256_json(
                {
                    "status": "intentionally_unmasked",
                    "main_surface_id": main_id,
                }
            )
        ),
        "occluder_set_sha256": occluder_digest,
        "pointmap_sha256": _sha256_file(points_path) if points_path.is_file() else None,
        "camera_sha256": _sha256_json(camera) if camera is not None else None,
        "extractor_parameters_sha256": _sha256_json(EXTRACTOR_PARAMETERS),
    }
    return graph_payload, main_record, main_mask, world, camera, occluder, binding


def build_main_support_yaw_observation(
    scene_dir: str | os.PathLike[str],
    *,
    graph: dict[str, Any] | None = None,
    masks_payload: dict[str, Any] | None = None,
    world_points: np.ndarray | None = None,
    camera_config: dict[str, Any] | None = None,
    input_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build (but do not write) a yaw artifact from a completed preprocess source set."""

    root = Path(scene_dir)
    (
        graph_payload,
        main_record,
        main_mask,
        world,
        camera,
        occluder,
        binding,
    ) = _source_inputs(
        root,
        graph=graph,
        masks_payload=masks_payload,
        world_points=world_points,
        camera_config=camera_config,
        input_path=input_path,
    )
    surface = {
        "id": str(graph_payload.get("main_support_id")),
        "build_name": str(binding["main_build_name"]),
        "semantic_class": str(binding["semantic_class"]),
    }
    routing = (
        (masks_payload or {}).get("routing") if masks_payload is not None else None
    )
    if routing is None:
        masks = json.loads((root / "masks" / "masks.json").read_text())
        routing = masks.get("routing") or {}
    scene_form = str((routing or {}).get("form") or "closeup:table")
    if scene_form != binding["scene_form"]:
        raise YawObservationArtifactError(
            "source routing form changed while building yaw observation"
        )
    if main_mask is None:
        return validate_main_support_yaw_observation(
            _empty_observation(
                source_binding=binding,
                main_surface=surface,
                applicability_status="unverified",
                reason_codes=["main_support_intentionally_unmasked"],
                diagnostics={"main_mask_pixel_count": 0},
            )
        )
    if world is None or camera is None:
        return validate_main_support_yaw_observation(
            _empty_observation(
                source_binding=binding,
                main_surface=surface,
                applicability_status="unverified",
                reason_codes=["canonical_camera_or_world_points_unavailable"],
                diagnostics={"main_mask_pixel_count": int(main_mask.sum())},
            )
        )
    return observe_main_support_yaw(
        main_mask,
        world,
        camera,
        main_surface_id=surface["id"],
        semantic_class=surface["semantic_class"],
        build_name=surface["build_name"],
        scene_form=scene_form,
        occluder_mask=occluder,
        source_binding=binding,
        top_plane_z_m=0.0,
    )


def _finite_run(value: Any, *, label: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise YawObservationArtifactError(f"{label} must be a three-vector")
    try:
        run = np.asarray([float(component) for component in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise YawObservationArtifactError(f"{label} must be numeric") from exc
    if not np.isfinite(run).all() or abs(float(run[2])) > 1e-6:
        raise YawObservationArtifactError(f"{label} must be a finite horizontal run")
    if abs(float(np.linalg.norm(run[:2])) - 1.0) > 1e-4:
        raise YawObservationArtifactError(f"{label} must be unit length")


def _finite_metric(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    maximum: float = math.inf,
) -> float:
    if isinstance(value, bool):
        raise YawObservationArtifactError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise YawObservationArtifactError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise YawObservationArtifactError(
            f"{label} must be in [{minimum}, {maximum}], got {value!r}"
        )
    return result


def _finite_point(value: Any, *, label: str, normalized: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise YawObservationArtifactError(f"{label} must be a two-vector")
    minimum, maximum = (-0.05, 1.05) if normalized else (-1e7, 1e7)
    for index, coordinate in enumerate(value):
        _finite_metric(
            coordinate,
            label=f"{label}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )


def validate_main_support_yaw_observation(artifact: Any) -> dict[str, Any]:
    """Strictly validate schema-v1; there is intentionally no legacy adapter."""

    if not isinstance(artifact, dict):
        raise YawObservationArtifactError("yaw observation artifact must be an object")
    if artifact.get("schema_version") != MAIN_SUPPORT_YAW_OBSERVATION_SCHEMA_VERSION:
        raise YawObservationArtifactError(
            "unsupported main-support yaw observation schema version: "
            f"{artifact.get('schema_version')!r}; rerun preprocessing"
        )
    # v2 artifacts (2026-08-20 .. 09-17) stay loadable so finished runs can still seed
    # agent-only reruns; they simply lack the v3 silhouette field. v1 is upgraded at a
    # restart boundary by ``refresh_legacy_yaw_observation``.
    if artifact.get("extractor_version") not in {
        MAIN_SUPPORT_YAW_EXTRACTOR_VERSION,
        "observable_top_edges_v2_visibility",
        "observable_top_edges_v1",
    }:
        raise YawObservationArtifactError(
            "unsupported main-support yaw extractor version: "
            f"{artifact.get('extractor_version')!r}; rerun preprocessing"
        )
    for key in (
        "source_binding",
        "main_surface",
        "applicability",
        "candidates",
        "decision",
        "segments",
        "pair",
        "diagnostics",
    ):
        if key not in artifact:
            raise YawObservationArtifactError(f"yaw observation is missing {key!r}")
    binding = artifact["source_binding"]
    if not isinstance(binding, dict):
        raise YawObservationArtifactError("source_binding must be an object")
    for key in (
        "main_surface_id",
        "main_build_name",
        "semantic_class",
        "scene_form",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or not value:
            raise YawObservationArtifactError(f"source_binding.{key} is missing")
    for key in (
        "main_mask_sha256",
        "occluder_set_sha256",
        "extractor_parameters_sha256",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise YawObservationArtifactError(
                f"source_binding.{key} is not a lowercase SHA-256 digest"
            )
    # The parameter digest pins CURRENT-version artifacts only: an accepted legacy
    # version was built with its own (older) parameter set by definition, and staying
    # loadable is what lets ``refresh_legacy_yaw_observation`` upgrade it.
    if artifact.get(
        "extractor_version"
    ) == MAIN_SUPPORT_YAW_EXTRACTOR_VERSION and binding[
        "extractor_parameters_sha256"
    ] != _sha256_json(EXTRACTOR_PARAMETERS):
        raise YawObservationArtifactError(
            "yaw observation extractor parameters are stale; rerun preprocessing"
        )
    for key in ("input_sha256", "pointmap_sha256", "camera_sha256"):
        value = binding.get(key)
        if value is not None and (
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise YawObservationArtifactError(f"source_binding.{key} is not a digest")
    surface = artifact["main_surface"]
    if not isinstance(surface, dict) or any(
        not isinstance(surface.get(key), str) or not surface.get(key)
        for key in ("id", "build_name", "semantic_class")
    ):
        raise YawObservationArtifactError("main_surface identity is malformed")
    if (
        surface["id"] != binding["main_surface_id"]
        or surface["build_name"] != binding["main_build_name"]
        or surface["semantic_class"] != binding["semantic_class"]
    ):
        raise YawObservationArtifactError("main_surface disagrees with source_binding")
    applicability = artifact["applicability"]
    if not isinstance(applicability, dict) or applicability.get("status") not in {
        "applicable",
        "not_applicable",
        "unverified",
    }:
        raise YawObservationArtifactError("yaw applicability is malformed")
    if not isinstance(applicability.get("reason_codes"), list):
        raise YawObservationArtifactError(
            "yaw applicability reason_codes must be a list"
        )
    decision = artifact["decision"]
    if (
        not isinstance(decision, dict)
        or (decision.get("status"), decision.get("authority")) not in _DECISION_PAIRS
    ):
        raise YawObservationArtifactError("yaw decision status/authority is invalid")
    if not isinstance(
        decision.get("participating_evidence_ids"), list
    ) or not isinstance(decision.get("reason_codes"), list):
        raise YawObservationArtifactError("yaw decision provenance is malformed")
    target = decision.get("target_run_world")
    if target is not None:
        _finite_run(target, label="decision.target_run_world")
    if decision["authority"] == "required" and target is None:
        raise YawObservationArtifactError("required yaw decision has no target run")
    if decision["authority"] == "none" and target is not None:
        raise YawObservationArtifactError("not-applicable yaw cannot have a target run")

    candidates = artifact["candidates"]
    segments = artifact["segments"]
    if not isinstance(candidates, list) or len(candidates) > int(
        EXTRACTOR_PARAMETERS["maximum_candidates"]
    ):
        raise YawObservationArtifactError("yaw candidates are not a bounded list")
    if not isinstance(segments, list) or len(segments) > int(
        EXTRACTOR_PARAMETERS["maximum_segments"]
    ):
        raise YawObservationArtifactError("yaw segments are not a bounded list")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise YawObservationArtifactError("yaw candidate must be an object")
        evidence_id = candidate.get("evidence_id")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in candidate_ids
        ):
            raise YawObservationArtifactError(
                "yaw candidate IDs must be unique strings"
            )
        candidate_ids.add(evidence_id)
        if candidate.get("provider") not in {"observable_top_edges", "mask_pca"}:
            raise YawObservationArtifactError(
                f"yaw candidate {evidence_id!r} has an unknown provider"
            )
        if candidate.get("independence_group") != SUPPORT_IMAGE_GEOMETRY_GROUP:
            raise YawObservationArtifactError(
                f"yaw candidate {evidence_id!r} has an invalid independence group"
            )
        if candidate.get("strength") not in {"strong", "weak"} or not isinstance(
            candidate.get("eligible"), bool
        ):
            raise YawObservationArtifactError(
                f"yaw candidate {evidence_id!r} has invalid strength/eligibility"
            )
        _finite_run(
            candidate.get("run_world"), label=f"candidate {evidence_id}.run_world"
        )
        if candidate["provider"] == "mask_pca" and candidate["strength"] != "weak":
            raise YawObservationArtifactError("blanket PCA can only be weak evidence")
    participating = decision["participating_evidence_ids"]
    if any(evidence_id not in candidate_ids for evidence_id in participating):
        raise YawObservationArtifactError(
            "yaw decision names a missing evidence candidate"
        )
    if decision["authority"] == "required":
        if not any(
            candidate["evidence_id"] in participating
            and candidate["provider"] == "observable_top_edges"
            and candidate["strength"] == "strong"
            and candidate["eligible"]
            for candidate in candidates
        ):
            raise YawObservationArtifactError(
                "required source yaw lacks a strong observable-edge witness"
            )
    segment_ids: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise YawObservationArtifactError("yaw segment must be an object")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not re.fullmatch(
            r"source-edge-[0-9]{2}", segment_id
        ):
            raise YawObservationArtifactError("yaw segment identity is malformed")
        if segment_id in segment_ids:
            raise YawObservationArtifactError("yaw segment IDs must be unique")
        segment_ids.add(segment_id)
        family = segment.get("family")
        if not isinstance(family, str) or not (
            family == "unqualified" or re.fullmatch(r"edge_family_[0-9]+", family)
        ):
            raise YawObservationArtifactError(
                f"segment {segment_id!r} has an invalid physical family"
            )
        for flag in ("eligible", "strong_eligible", "frame_clear"):
            if not isinstance(segment.get(flag), bool):
                raise YawObservationArtifactError(
                    f"segment {segment_id!r}.{flag} must be boolean"
                )
        if segment["strong_eligible"] and not segment["eligible"]:
            raise YawObservationArtifactError(
                f"segment {segment_id!r} cannot be strong but ineligible"
            )
        endpoints = segment.get("endpoints_px")
        normalized_endpoints = segment.get("endpoints_normalized")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise YawObservationArtifactError(
                f"segment {segment_id!r}.endpoints_px must contain two points"
            )
        if not isinstance(normalized_endpoints, list) or len(normalized_endpoints) != 2:
            raise YawObservationArtifactError(
                f"segment {segment_id!r}.endpoints_normalized must contain two points"
            )
        for index in range(2):
            _finite_point(
                endpoints[index], label=f"segment {segment_id}.endpoints_px[{index}]"
            )
            _finite_point(
                normalized_endpoints[index],
                label=f"segment {segment_id}.endpoints_normalized[{index}]",
                normalized=True,
            )
        if (
            np.linalg.norm(
                np.asarray(endpoints[1], dtype=np.float64)
                - np.asarray(endpoints[0], dtype=np.float64)
            )
            < 1e-6
        ):
            raise YawObservationArtifactError(
                f"segment {segment_id!r} endpoints are degenerate"
            )
        for field, maximum in (
            ("length_frame_diag", 5.0),
            ("length_top_bbox_diag", 5.0),
            ("fit_rmse_px", 1e6),
        ):
            _finite_metric(
                segment.get(field),
                label=f"segment {segment_id}.{field}",
                maximum=maximum,
            )
        unit_fields = [
            "boundary_support_fraction",
            "top_face_support_fraction",
            "occluder_clear_fraction",
        ]
        if (
            artifact.get("extractor_version") == MAIN_SUPPORT_YAW_EXTRACTOR_VERSION
            or "occluder_silhouette_fraction" in segment
        ):
            unit_fields.append("occluder_silhouette_fraction")
        for field in unit_fields:
            _finite_metric(
                segment.get(field),
                label=f"segment {segment_id}.{field}",
                maximum=1.0,
            )
        uncertainty = segment.get("angular_uncertainty_degrees")
        incidence = segment.get("minimum_plane_incidence")
        if uncertainty is not None:
            _finite_metric(
                uncertainty,
                label=f"segment {segment_id}.angular_uncertainty_degrees",
                maximum=180.0,
            )
        if incidence is not None:
            _finite_metric(
                incidence,
                label=f"segment {segment_id}.minimum_plane_incidence",
                maximum=1.0,
            )
        reasons = segment.get("reason_codes")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise YawObservationArtifactError(
                f"segment {segment_id!r}.reason_codes must be nonempty strings"
            )
        if segment.get("world_run") is not None:
            _finite_run(segment["world_run"], label=f"segment {segment_id}.world_run")
        if segment["eligible"] and (
            segment.get("world_run") is None
            or uncertainty is None
            or incidence is None
            or family == "unqualified"
        ):
            raise YawObservationArtifactError(
                f"eligible segment {segment_id!r} lacks verifiable geometry"
            )
    for candidate in candidates:
        referenced_segments = candidate.get("segment_ids")
        if not isinstance(referenced_segments, list) or any(
            not isinstance(segment_id, str) or segment_id not in segment_ids
            for segment_id in referenced_segments
        ):
            raise YawObservationArtifactError(
                f"yaw candidate {candidate['evidence_id']!r} has invalid segment_ids"
            )
    if not isinstance(artifact["diagnostics"], dict):
        raise YawObservationArtifactError("yaw diagnostics must be an object")
    # Force a final no-NaN/canonical-JSON pass.
    _canonical_json(artifact)
    return artifact


def write_main_support_yaw_observation(
    scene_dir: str | os.PathLike[str], artifact: dict[str, Any]
) -> Path:
    """Atomically write the strict source yaw artifact."""

    root = Path(scene_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MAIN_SUPPORT_YAW_OBSERVATION_FILENAME
    validated = validate_main_support_yaw_observation(artifact)
    payload = json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=root, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path


def refresh_legacy_yaw_observation(scene_dir: str | os.PathLike[str]) -> bool:
    """Upgrade known legacy source evidence at a restart boundary, without VLM calls.

    Recompile the dependent initializer contract as well. Runtime readers never
    rewrite frozen evidence; unknown/corrupt schemas retain their normal failure.
    """
    root = Path(scene_dir)
    path = root / MAIN_SUPPORT_YAW_OBSERVATION_FILENAME
    artifact = json.loads(path.read_text())
    if artifact.get("extractor_version") == MAIN_SUPPORT_YAW_EXTRACTOR_VERSION:
        return False
    # Any accepted legacy version (v1, v2) is rebuilt from its immutable source inputs;
    # unknown/corrupt versions still fail inside validate.
    validate_main_support_yaw_observation(artifact)
    rebuilt = build_main_support_yaw_observation(root)
    # Circular dependency: relationship constraints themselves consume yaw artifacts.
    from lib.tools.geometry.relationship_constraints import (
        write_initializer_constraints,
    )

    graph = json.loads((root / "scene_graph.json").read_text())
    write_initializer_constraints(root, graph, yaw_observation=rebuilt)
    # Publish the version marker LAST. If constraint compilation/writing fails, the
    # old observation still triggers a retry instead of masking a partial upgrade.
    write_main_support_yaw_observation(root, rebuilt)
    return True


def load_main_support_yaw_observation(
    scene_dir_or_path: str | os.PathLike[str],
    *,
    verify_source_bindings: bool = False,
) -> dict[str, Any]:
    """Load schema-v1 and optionally verify every immutable source binding.

    Runtime graph transactions normally need schema validation only: graph revisions are
    mutable, while this observation is deliberately frozen.  Cache reuse and resettle
    pass ``verify_source_bindings=True`` so staged masks/camera/point maps cannot be mixed.
    """

    supplied = Path(scene_dir_or_path)
    path = (
        supplied / MAIN_SUPPORT_YAW_OBSERVATION_FILENAME
        if supplied.is_dir()
        else supplied
    )
    root = supplied if supplied.is_dir() else path.parent
    if not path.is_file():
        raise YawObservationArtifactError(
            f"missing {path}; rerun preprocessing (legacy yaw caches are unsupported)"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise YawObservationArtifactError(f"cannot read {path}: {exc}") from exc
    artifact = validate_main_support_yaw_observation(artifact)
    if verify_source_bindings:
        try:
            *_, expected = _source_inputs(root)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise YawObservationArtifactError(
                f"cannot verify yaw observation source bindings: {exc}"
            ) from exc
        actual = artifact["source_binding"]
        if actual != expected:
            changed = sorted(
                key
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise YawObservationArtifactError(
                "main-support yaw observation is stale for its immutable source inputs "
                f"(changed bindings: {changed}); rerun preprocessing"
            )
    return artifact
