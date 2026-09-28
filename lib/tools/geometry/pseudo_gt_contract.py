"""Portable completeness contract for pseudo-GT novel-view bundles.

The producer writes ``cameras.json`` before SHARP/GPT has finished, so manifest
existence alone is not a completion marker.  This module is deliberately
producer-independent: preprocessing uses it to qualify an explicitly reused
bundle, while Blender's executor uses it to avoid exposing partial bundles.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

PSEUDO_GT_VIEWS_AZ_EL: list[tuple[float, float]] = [
    (0.0, 0.0),
    (30.0, 0.0),
    (-30.0, 0.0),
    (0.0, 15.0),
    (0.0, -15.0),
]

_ROTATION_TOL_DEG = 0.6
_CAM_TRANSLATION_TOL_M = 0.015
_LOCATION_TOL_M = 0.015
_LOOK_AT_TOL_M = 0.025
_LENS_TOL_MM = 0.01
_SCALAR_ABS_TOL = 1e-7


class PseudoGTContractError(RuntimeError):
    """A pseudo-GT folder is partial, malformed, or incompatible with this run."""


def view_tag(az: float, el: float) -> str:
    """Filesystem-safe tag for a view, e.g. ``(-30, 15) -> az-30_el15``."""

    return f"az{int(round(az))}_el{int(round(el))}"


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PseudoGTContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PseudoGTContractError(f"{label} must be a finite number")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PseudoGTContractError(f"{label} must contain {length} finite numbers")
    if len(value) != length:
        raise PseudoGTContractError(f"{label} must contain {length} finite numbers")
    return [
        _finite_scalar(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _finite_matrix4(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PseudoGTContractError(f"{label} must be a finite 4x4 matrix")
    if len(value) != 4:
        raise PseudoGTContractError(f"{label} must be a finite 4x4 matrix")
    return [
        _finite_vector(row, 4, f"{label}[{index}]") for index, row in enumerate(value)
    ]


def _validate_rigid_matrix(matrix: Sequence[Sequence[float]], label: str) -> None:
    rotation = [row[:3] for row in matrix[:3]]
    columns = [[rotation[row][column] for row in range(3)] for column in range(3)]
    for index, column in enumerate(columns):
        if not math.isclose(sum(x * x for x in column), 1.0, abs_tol=1e-5):
            raise PseudoGTContractError(f"{label} rotation is not orthonormal")
        for other in columns[:index]:
            if not math.isclose(
                sum(x * y for x, y in zip(column, other)), 0.0, abs_tol=1e-5
            ):
                raise PseudoGTContractError(f"{label} rotation is not orthonormal")
    if any(
        not math.isclose(actual, expected, abs_tol=1e-7)
        for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise PseudoGTContractError(f"{label} must be a rigid homogeneous transform")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PseudoGTContractError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PseudoGTContractError(f"{label} must be a positive integer") from exc
    if result <= 0 or float(result) != _finite_scalar(value, label):
        raise PseudoGTContractError(f"{label} must be a positive integer")
    return result


def _verify_image(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PseudoGTContractError(f"{label} is missing or empty: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - Pillow has several decode exceptions
        raise PseudoGTContractError(f"{label} is not a readable image: {path}") from exc


def _assert_source_match(source_image: Path, cached_source: Path) -> None:
    _verify_image(source_image, "current canonical input")
    try:
        with Image.open(source_image) as current_image:
            current = current_image.convert("RGB")
        with Image.open(cached_source) as cached_image:
            cached = cached_image.convert("RGB")
    except Exception as exc:  # pragma: no cover - _verify_image already gives detail
        raise PseudoGTContractError("unable to compare source-view images") from exc
    if current.size != cached.size or ImageChops.difference(current, cached).getbbox():
        raise PseudoGTContractError(
            "pseudo-GT source view does not pixel-match this run's canonical input"
        )


def _close_scalar(actual: Any, expected: Any, label: str, *, abs_tol: float) -> None:
    a_value = _finite_scalar(actual, label)
    e_value = _finite_scalar(expected, label)
    if not math.isclose(a_value, e_value, rel_tol=1e-7, abs_tol=abs_tol):
        raise PseudoGTContractError(f"{label} differs from this run's camera")


def _vector_close(
    actual: Any, expected: Any, length: int, label: str, *, distance_tol: float
) -> None:
    actual_vector = _finite_vector(actual, length, label)
    expected_vector = _finite_vector(expected, length, f"expected.{label}")
    if math.dist(actual_vector, expected_vector) > distance_tol:
        raise PseudoGTContractError(f"{label} differs from this run's camera")


def _camera_matrix_close(actual: Any, expected: Any, label: str) -> None:
    actual_matrix = _finite_matrix4(actual, label)
    expected_matrix = _finite_matrix4(expected, f"expected.{label}")
    _validate_rigid_matrix(actual_matrix, label)
    _validate_rigid_matrix(expected_matrix, f"expected.{label}")
    _vector_close(
        [row[3] for row in actual_matrix[:3]],
        [row[3] for row in expected_matrix[:3]],
        3,
        f"{label}.translation",
        distance_tol=_CAM_TRANSLATION_TOL_M,
    )
    trace_relative = sum(
        actual_matrix[row][column] * expected_matrix[row][column]
        for row in range(3)
        for column in range(3)
    )
    angle_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, (trace_relative - 1.0) / 2.0)))
    )
    if angle_deg > _ROTATION_TOL_DEG:
        raise PseudoGTContractError(f"{label}.rotation differs from this run's camera")


def _validate_expected_cameras(
    views: Sequence[Mapping[str, Any]], expected_views: Sequence[Mapping[str, Any]]
) -> None:
    expected_by_tag = {str(view.get("tag")): view for view in expected_views}
    required_tags = {
        view_tag(azimuth, elevation) for azimuth, elevation in PSEUDO_GT_VIEWS_AZ_EL
    }
    if set(expected_by_tag) != required_tags:
        raise PseudoGTContractError(
            "current run did not produce the canonical five pseudo-GT cameras"
        )
    for view in views:
        tag = str(view["tag"])
        expected = expected_by_tag[tag]
        for key in ("az", "el"):
            _close_scalar(
                view[key],
                expected[key],
                f"{tag}.{key}",
                abs_tol=_SCALAR_ABS_TOL,
            )
        _close_scalar(
            view["lens"],
            expected["lens"],
            f"{tag}.lens",
            abs_tol=_LENS_TOL_MM,
        )
        _vector_close(
            view["location"],
            expected["location"],
            3,
            f"{tag}.location",
            distance_tol=_LOCATION_TOL_M,
        )
        _vector_close(
            view["look_at"],
            expected["look_at"],
            3,
            f"{tag}.look_at",
            distance_tol=_LOOK_AT_TOL_M,
        )
        _camera_matrix_close(
            view["cam2world_moge"],
            expected["cam2world_moge"],
            f"{tag}.cam2world_moge",
        )
        for key in ("res_x", "res_y"):
            if _positive_int(view[key], f"{tag}.{key}") != _positive_int(
                expected[key], f"expected.{tag}.{key}"
            ):
                raise PseudoGTContractError(
                    f"{tag}.{key} differs from this run's camera"
                )


def load_complete_pseudo_gt(
    pseudo_gt_dir: str | Path,
    *,
    source_image: str | Path | None = None,
    expected_views: Sequence[Mapping[str, Any]] | None = None,
    rebase_paths: bool = False,
) -> dict[str, Any]:
    """Load, validate, and optionally make a pseudo-GT bundle self-contained.

    Completeness means the exact canonical five camera rows and one readable local
    ``<tag>/completed.png`` per row.  Recorded image paths are intentionally ignored:
    copied runs commonly retain paths into the source run.  Returned rows always point
    at their destination-local image; ``rebase_paths=True`` persists that normalization
    atomically to ``cameras.json``.
    """

    root = Path(pseudo_gt_dir)
    if root.is_symlink():
        raise PseudoGTContractError(
            f"pseudo-GT directory must not be a symlink: {root}"
        )
    resolved_root = root.resolve()
    manifest = root / "cameras.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise PseudoGTContractError(f"pseudo-GT cameras.json is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PseudoGTContractError(
            f"pseudo-GT cameras.json is unreadable: {exc}"
        ) from exc
    rows = payload.get("views") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise PseudoGTContractError("pseudo-GT cameras.json must contain a views list")

    required = {
        (float(azimuth), float(elevation)): view_tag(azimuth, elevation)
        for azimuth, elevation in PSEUDO_GT_VIEWS_AZ_EL
    }
    normalized: list[dict[str, Any]] = []
    seen_poses: set[tuple[float, float]] = set()
    seen_tags: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PseudoGTContractError(f"pseudo-GT view {index} must be an object")
        view = dict(raw)
        azimuth = _finite_scalar(view.get("az"), f"views[{index}].az")
        elevation = _finite_scalar(view.get("el"), f"views[{index}].el")
        pose = (azimuth, elevation)
        tag = str(view.get("tag") or "")
        if pose not in required or tag != required[pose]:
            raise PseudoGTContractError(
                f"pseudo-GT view {index} is not a canonical (az, el, tag) row"
            )
        if pose in seen_poses or tag in seen_tags:
            raise PseudoGTContractError(f"pseudo-GT view is duplicated: {tag}")
        seen_poses.add(pose)
        seen_tags.add(tag)

        view["location"] = _finite_vector(view.get("location"), 3, f"{tag}.location")
        view["look_at"] = _finite_vector(view.get("look_at"), 3, f"{tag}.look_at")
        lens = _finite_scalar(view.get("lens"), f"{tag}.lens")
        if lens <= 0:
            raise PseudoGTContractError(f"{tag}.lens must be positive")
        view["lens"] = lens
        view["res_x"] = _positive_int(view.get("res_x"), f"{tag}.res_x")
        view["res_y"] = _positive_int(view.get("res_y"), f"{tag}.res_y")
        view["cam2world_moge"] = _finite_matrix4(
            view.get("cam2world_moge"), f"{tag}.cam2world_moge"
        )
        _validate_rigid_matrix(view["cam2world_moge"], f"{tag}.cam2world_moge")
        if not isinstance(view.get("pseudo_gt"), str) or not view["pseudo_gt"]:
            raise PseudoGTContractError(f"{tag}.pseudo_gt is missing")
        if "pseudo_gt_trusted" in view and not isinstance(
            view["pseudo_gt_trusted"], bool
        ):
            raise PseudoGTContractError(f"{tag}.pseudo_gt_trusted must be boolean")

        local_view_dir = root / tag
        local_completed = local_view_dir / "completed.png"
        if local_view_dir.is_symlink() or local_completed.is_symlink():
            raise PseudoGTContractError(
                f"{tag} completion must be a destination-local regular file"
            )
        completed = local_completed.resolve()
        try:
            completed.relative_to(resolved_root)
        except ValueError as exc:
            raise PseudoGTContractError(
                f"{tag} completion escapes the pseudo-GT directory"
            ) from exc
        _verify_image(completed, f"{tag} completion")
        view["pseudo_gt"] = str(completed)

        view_dir = root / tag
        if (view_dir / "gpt_fallback.txt").is_file():
            view["pseudo_gt_trusted"] = False
            view["pseudo_gt_untrusted_reason"] = "generation_failed"
        elif (view_dir / "drift_flag.txt").is_file():
            view["pseudo_gt_trusted"] = False
            view["pseudo_gt_untrusted_reason"] = "excessive_observed_content_drift"
        elif "pseudo_gt_trusted" not in view:
            view["pseudo_gt_trusted"] = True
        normalized.append(view)

    required_tags = set(required.values())
    if seen_tags != required_tags or len(normalized) != len(required_tags):
        missing = sorted(required_tags - seen_tags)
        extra = sorted(seen_tags - required_tags)
        raise PseudoGTContractError(
            f"pseudo-GT view set is incomplete (missing={missing}, extra={extra})"
        )

    if source_image is not None:
        _assert_source_match(
            Path(source_image), (root / view_tag(0.0, 0.0) / "completed.png").resolve()
        )
    if expected_views is not None:
        _validate_expected_cameras(normalized, expected_views)

    normalized_payload = dict(payload)
    normalized_payload["views"] = normalized
    if rebase_paths and normalized_payload != payload:
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(normalized_payload, indent=2) + "\n")
        os.replace(temporary, manifest)
    return normalized_payload
