"""Test-only factory for strict source-yaw artifacts.

Production never synthesizes missing observations.  Tests that construct scene graphs
by hand must opt in to this explicit no-evidence artifact, mirroring preprocessing's
required-file contract without weakening the loader.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from lib.tools.geometry.main_support_yaw_observation import (
    EXTRACTOR_PARAMETERS,
    write_main_support_yaw_observation,
)


def unverified_yaw_observation(graph: dict[str, Any]) -> dict[str, Any]:
    main_id = str(graph.get("main_support_id") or "table#0")
    build_name = main_id.replace("#", "_")
    semantic_class = main_id.split("#", 1)[0]
    parameters_sha256 = hashlib.sha256(
        json.dumps(
            EXTRACTOR_PARAMETERS,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "extractor_version": "observable_top_edges_v1",
        "source_binding": {
            "main_surface_id": main_id,
            "main_build_name": build_name,
            "semantic_class": semantic_class,
            "scene_form": "room" if semantic_class in {"floor", "ground"} else "closeup:table",
            "main_mask_sha256": hashlib.sha256(b"test-mask").hexdigest(),
            "occluder_set_sha256": hashlib.sha256(
                b"empty-test-occluders"
            ).hexdigest(),
            "extractor_parameters_sha256": parameters_sha256,
            "input_sha256": None,
            "pointmap_sha256": None,
            "camera_sha256": None,
        },
        "main_surface": {
            "id": main_id,
            "build_name": build_name,
            "semantic_class": semantic_class,
        },
        "applicability": {"status": "applicable", "reason_codes": []},
        "candidates": [],
        "decision": {
            "status": "unverified",
            "authority": "advisory",
            "target_run_world": None,
            "participating_evidence_ids": [],
            "reason_codes": ["test_no_yaw_evidence"],
        },
        "segments": [],
        "pair": None,
        "diagnostics": {},
    }


def write_unverified_yaw_observation(
    scene_dir: str | Path, graph: dict[str, Any]
) -> dict[str, Any]:
    artifact = unverified_yaw_observation(graph)
    write_main_support_yaw_observation(scene_dir, artifact)
    return artifact


def strong_yaw_observation(
    graph: dict[str, Any], run_xy: tuple[float, float] = (1.0, 0.0)
) -> dict[str, Any]:
    artifact = unverified_yaw_observation(graph)
    length = math.hypot(float(run_xy[0]), float(run_xy[1]))
    run = [float(run_xy[0]) / length, float(run_xy[1]) / length, 0.0]
    perpendicular = [-run[1], run[0], 0.0]

    def segment(index: int, world_run: list[float], family: str) -> dict[str, Any]:
        offset = float(index) * 20.0
        return {
            "segment_id": f"source-edge-{index:02d}",
            "family": family,
            "eligible": True,
            "strong_eligible": True,
            "frame_clear": True,
            "endpoints_px": [[10.0 + offset, 10.0], [50.0 + offset, 10.0]],
            "endpoints_normalized": [
                [0.1 + index * 0.1, 0.1],
                [0.4 + index * 0.1, 0.1],
            ],
            "world_run": world_run,
            "length_frame_diag": 0.2,
            "length_top_bbox_diag": 0.5,
            "fit_rmse_px": 0.5,
            "boundary_support_fraction": 0.95,
            "top_face_support_fraction": 0.95,
            "occluder_clear_fraction": 0.95,
            "angular_uncertainty_degrees": 2.0,
            "minimum_plane_incidence": 0.5,
            "reason_codes": [],
        }

    artifact["segments"] = [
        segment(0, run, "edge_family_0"),
        segment(1, perpendicular, "edge_family_1"),
    ]
    artifact["candidates"] = [
        {
            "evidence_id": "observable-edge-pair-0",
            "provider": "observable_top_edges",
            "independence_group": "support_image_geometry",
            "run_world": run,
            "strength": "strong",
            "eligible": True,
            "angular_uncertainty_degrees": 2.0,
            "segment_ids": ["source-edge-00", "source-edge-01"],
            "reason_codes": [],
        }
    ]
    artifact["decision"] = {
        "status": "confirmed",
        "authority": "required",
        "target_run_world": run,
        "participating_evidence_ids": ["observable-edge-pair-0"],
        "reason_codes": ["two_observable_edge_families"],
    }
    artifact["pair"] = {
        "segment_ids": ["source-edge-00", "source-edge-01"]
    }
    return artifact
