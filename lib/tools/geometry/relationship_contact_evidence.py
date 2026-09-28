"""Conservative image-space compatibility evidence for root-surface contact.

This module deliberately does *not* claim to recover 3-D contact from a single image.
It answers the narrower question needed by relationship adjudication: do the final
support and wall masks expose a coherent boundary that is compatible with the VLM's
``AGAINST`` claim, expose a sustained visible separation, or leave the alleged contact
region unobservable?

Every threshold is normalized by the shorter image dimension and every quantity used
by the decision is returned in the JSON-serializable evidence record.  A curved
silhouette where one mask merely occludes the other is ``unobservable``, not proof of
contact and not a contradiction.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

IMAGE_CONTACT_EVIDENCE_VERSION = 1
IMAGE_CONTACT_COMPATIBILITIES = frozenset(
    {"compatible", "contradicted", "unobservable"}
)


def _clean_mask(mask: Any) -> Any | None:
    """Return a bool mask with tiny detached components removed."""

    import numpy as np
    from scipy import ndimage

    arr = np.asarray(mask, bool)
    if arr.ndim != 2 or not bool(arr.any()):
        return None
    labels, count = ndimage.label(arr, structure=np.ones((3, 3), bool))
    if count <= 1:
        return arr
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    # Root masks may be split around foreground objects.  Keep meaningful pieces but
    # discard SAM flecks that otherwise manufacture one or two adjacent pixels.
    minimum = max(16, int(math.ceil(largest * 0.002)))
    keep = sizes >= minimum
    keep[0] = False
    cleaned = keep[labels]
    return cleaned if bool(cleaned.any()) else None


def _boundary(mask: Any) -> Any:
    import numpy as np
    from scipy import ndimage

    return np.asarray(mask, bool) & ~ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), bool), border_value=0
    )


def _line_metrics(y: Any, x: Any) -> dict[str, Any] | None:
    """PCA line metrics for one connected pixel chain."""

    import numpy as np

    if len(x) < 2:
        return None
    pts = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    center = pts.mean(axis=0)
    centered = pts - center
    covariance = centered.T @ centered / len(pts)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    minor = max(float(values[order[0]]), 0.0)
    direction = vectors[:, order[-1]]
    projections = centered @ direction
    span = float(projections.max() - projections.min())
    rms = math.sqrt(minor)
    return {
        "pixel_count": int(len(pts)),
        "span_px": round(span, 3),
        "line_rms_px": round(rms, 3),
        "line_rms_over_span": round(rms / max(span, 1e-9), 6),
        "direction_xy": [round(float(direction[0]), 6), round(float(direction[1]), 6)],
        "center_xy_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
    }


def _components(binary: Any) -> list[tuple[Any, Any]]:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(binary, structure=np.ones((3, 3), bool))
    components: list[tuple[Any, Any]] = []
    for index in range(1, count + 1):
        y, x = np.where(labels == index)
        if len(x) >= 2:
            components.append((y, x))
    components.sort(key=lambda pair: len(pair[0]), reverse=True)
    return components


def _best_straight_chain(
    binary: Any,
    *,
    minimum_span_px: float,
    maximum_line_rms_px: float,
    image_shape: tuple[int, int],
    frame_margin_px: int,
) -> dict[str, Any] | None:
    # Use the dominant adjacent component.  Selecting a short straight sub-component
    # from a much longer curved silhouette would incorrectly turn an occlusion arc into
    # a flush edge (the IMG_8210 round-table regression).
    best = None
    for y, x in _components(binary):
        metrics = _line_metrics(y, x)
        if metrics is None:
            continue
        height, width = image_shape
        on_frame = (
            (y < frame_margin_px)
            | (y >= height - frame_margin_px)
            | (x < frame_margin_px)
            | (x >= width - frame_margin_px)
        )
        frame_fraction = float(on_frame.mean())
        metrics["frame_border_fraction"] = round(frame_fraction, 6)
        metrics["materially_on_frame_border"] = bool(frame_fraction >= 0.50)
        metrics["meets_minimum_span"] = bool(metrics["span_px"] >= minimum_span_px)
        metrics["meets_line_residual"] = bool(
            metrics["line_rms_px"] <= maximum_line_rms_px
        )
        if best is None or (metrics["pixel_count"], metrics["span_px"]) > (
            best["pixel_count"],
            best["span_px"],
        ):
            best = metrics
    return best


def _corridor_is_clear(
    support_y: Any,
    support_x: Any,
    wall_y: Any,
    wall_x: Any,
    occluder: Any | None,
) -> tuple[bool, float]:
    """Sample nearest-boundary segments and report their occluded fraction."""

    import numpy as np

    if occluder is None or len(support_x) == 0:
        return True, 0.0
    occ = np.asarray(occluder, bool)
    hit, total = 0, 0
    # At most 512 evenly-spaced correspondences keep this deterministic and cheap.
    indices = np.linspace(0, len(support_x) - 1, min(512, len(support_x))).astype(int)
    for index in indices:
        y0, x0 = int(support_y[index]), int(support_x[index])
        y1, x1 = int(wall_y[index]), int(wall_x[index])
        steps = max(abs(y1 - y0), abs(x1 - x0), 1)
        yy = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        xx = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        total += 1
        hit += int(bool(occ[yy, xx].any()))
    fraction = float(hit / total) if total else 0.0
    return fraction <= 0.10, fraction


def image_contact_compatibility(
    support_mask: Any,
    wall_mask: Any,
    *,
    occluder_masks: Iterable[Any] | None = None,
    adjacency_radius_fraction: float = 0.007,
    minimum_run_fraction: float = 0.05,
    maximum_line_rms_fraction: float = 0.015,
    separation_search_fraction: float = 0.08,
) -> dict[str, Any]:
    """Measure conservative mask evidence for a claimed support ``AGAINST`` wall.

    ``compatible`` means a long, nearly straight support-boundary chain lies next to
    the wall mask.  It means that contact is visually *compatible*, never that 3-D
    contact was proven.  ``contradicted`` identifies two nearly parallel, straight
    boundary chains separated by a sustained image strip, but the record remains
    diagnostic because shadows/baseboards can produce the same mask topology.
    Everything clipped, curved, fragmented, or occluded is ``unobservable``.
    """

    import numpy as np
    from scipy import ndimage

    evidence: dict[str, Any] = {
        "schema_version": IMAGE_CONTACT_EVIDENCE_VERSION,
        "status": "unobservable",
        "reason_code": "invalid_or_empty_masks",
        "authority": "diagnostic_only",
    }
    support = _clean_mask(support_mask)
    wall = _clean_mask(wall_mask)
    if support is None or wall is None or support.shape != wall.shape:
        return evidence

    height, width = support.shape
    scale = float(min(height, width))
    adjacency_radius = max(2, int(round(adjacency_radius_fraction * scale)))
    minimum_span = max(24.0, minimum_run_fraction * scale)
    maximum_line_rms = max(2.0, maximum_line_rms_fraction * minimum_span)
    separation_search_radius = max(
        adjacency_radius + 3, int(round(separation_search_fraction * scale))
    )
    frame_margin = max(2, adjacency_radius)

    support_boundary = _boundary(support)
    wall_boundary = _boundary(wall)
    if not bool(support_boundary.any()) or not bool(wall_boundary.any()):
        return evidence

    wall_distance, nearest_wall = ndimage.distance_transform_edt(
        ~wall_boundary, return_indices=True
    )
    support_distances = wall_distance[support_boundary]
    evidence.update(
        {
            "mask_shape_hw": [int(height), int(width)],
            "policy": {
                "adjacency_radius_fraction": adjacency_radius_fraction,
                "adjacency_radius_px": adjacency_radius,
                "minimum_run_fraction": minimum_run_fraction,
                "minimum_run_span_px": round(minimum_span, 3),
                "maximum_line_rms_fraction": maximum_line_rms_fraction,
                "maximum_line_rms_px": round(maximum_line_rms, 3),
                "separation_search_fraction": separation_search_fraction,
                "separation_search_radius_px": separation_search_radius,
                "frame_margin_px": frame_margin,
            },
            "support_boundary_pixels": int(support_boundary.sum()),
            "wall_boundary_pixels": int(wall_boundary.sum()),
            "minimum_boundary_gap_px": round(float(support_distances.min()), 3),
            "p02_boundary_gap_px": round(
                float(np.percentile(support_distances, 2.0)), 3
            ),
        }
    )

    occluder = None
    for raw in occluder_masks or ():
        arr = np.asarray(raw, bool)
        if arr.shape != support.shape:
            continue
        occluder = arr.copy() if occluder is None else (occluder | arr)
    if occluder is not None:
        occluder_distance = ndimage.distance_transform_edt(~occluder)
        occluder_near = occluder_distance <= 2 * adjacency_radius
    else:
        occluder_near = np.zeros_like(support, bool)

    raw_near = support_boundary & (wall_distance <= adjacency_radius)
    near = raw_near & ~occluder_near
    near_best = _best_straight_chain(
        near,
        minimum_span_px=minimum_span,
        maximum_line_rms_px=maximum_line_rms,
        image_shape=(height, width),
        frame_margin_px=frame_margin,
    )
    evidence["occluder_masks_available"] = bool(occluder is not None)
    evidence["adjacent_boundary_pixels_before_occlusion_filter"] = int(raw_near.sum())
    evidence["adjacent_boundary_pixels_removed_as_occluded"] = int(
        (raw_near & occluder_near).sum()
    )
    evidence["adjacent_boundary_pixels"] = int(near.sum())
    evidence["best_adjacent_run"] = near_best
    if (
        near_best is not None
        and near_best["meets_minimum_span"]
        and near_best["meets_line_residual"]
        and not near_best["materially_on_frame_border"]
    ):
        evidence.update(
            status="compatible",
            reason_code="coherent_straight_adjacent_boundary",
            authority="semantic_corroboration",
        )
        return evidence

    # A contradiction requires a visible *strip*, not merely absence of adjacency.
    # Find the closest coherent support chain in a bounded search radius, match it to
    # its nearest wall pixels, and require both chains to be straight/parallel with a
    # stable nonzero gap.  Far-away or curved silhouettes remain unobservable.
    within_search = (
        support_boundary
        & (wall_distance <= separation_search_radius)
        & ~occluder_near
    )
    candidates = []

    for y, x in _components(within_search):
        gap = wall_distance[y, x]
        # Remove pixels already in the adjacency band.  A visible separating strip
        # must stay nonzero along the candidate chain.
        select = gap > adjacency_radius + 1
        y, x, gap = y[select], x[select], gap[select]
        if len(x) < 2:
            continue
        on_frame = (
            (y < frame_margin)
            | (y >= height - frame_margin)
            | (x < frame_margin)
            | (x >= width - frame_margin)
        )
        frame_fraction = float(on_frame.mean())
        support_line = _line_metrics(y, x)
        if support_line is None or support_line["span_px"] < minimum_span:
            continue
        wall_y, wall_x = nearest_wall[0, y, x], nearest_wall[1, y, x]
        wall_line = _line_metrics(wall_y, wall_x)
        if wall_line is None or wall_line["span_px"] < minimum_span:
            continue
        support_rms_limit = max(
            2.0, maximum_line_rms_fraction * support_line["span_px"]
        )
        wall_rms_limit = max(2.0, maximum_line_rms_fraction * wall_line["span_px"])
        dot = abs(
            support_line["direction_xy"][0] * wall_line["direction_xy"][0]
            + support_line["direction_xy"][1] * wall_line["direction_xy"][1]
        )
        median_gap = float(np.median(gap))
        gap_spread = float(np.percentile(gap, 90) - np.percentile(gap, 10))
        corridor_clear, occluded_fraction = _corridor_is_clear(
            y, x, wall_y, wall_x, occluder
        )
        record = {
            "support_run": support_line,
            "wall_run": wall_line,
            "direction_abs_dot": round(float(dot), 6),
            "median_gap_px": round(median_gap, 3),
            "p90_minus_p10_gap_px": round(gap_spread, 3),
            "corridor_occluded_fraction": round(occluded_fraction, 6),
            "corridor_clear": bool(corridor_clear),
            "frame_border_fraction": round(frame_fraction, 6),
        }
        candidates.append(record)
        if (
            support_line["line_rms_px"] <= support_rms_limit
            and wall_line["line_rms_px"] <= wall_rms_limit
            and dot >= math.cos(math.radians(10.0))
            and gap_spread <= max(3.0, 0.25 * median_gap)
            and corridor_clear
            and frame_fraction < 0.50
        ):
            evidence["best_separation_run"] = record
            evidence.update(
                status="contradicted",
                reason_code="coherent_separating_strip_candidate",
                authority="diagnostic_only",
            )
            return evidence

    if candidates:
        evidence["best_separation_run"] = max(
            candidates,
            key=lambda item: item["support_run"]["span_px"],
        )
    if near_best is not None and near_best.get("materially_on_frame_border"):
        evidence["reason_code"] = "candidate_contact_boundary_is_frame_clipped"
    elif near_best is not None and near_best["span_px"] >= minimum_span:
        evidence["reason_code"] = "adjacent_boundary_is_not_a_flush_edge"
    elif bool(within_search.any()):
        evidence["reason_code"] = "contact_boundary_fragmented_or_occluded"
    else:
        evidence["reason_code"] = "no_observable_contact_boundary"
    return evidence


__all__ = [
    "IMAGE_CONTACT_COMPATIBILITIES",
    "IMAGE_CONTACT_EVIDENCE_VERSION",
    "image_contact_compatibility",
]
