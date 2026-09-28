"""MoGE-2 geometry refresh for edited images."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

def refine_depth(
    scene_dir: str,
    edited_image: str,
    hole_mask: str,
    tag: str,
    timeout: float = 600.0,
    *,
    reference_depth: str | None = None,
    intrinsics_norm: list[list[float]] | None = None,
) -> Optional[dict]:
    del timeout
    import numpy as np
    from PIL import Image

    from lib.tools.geometry.depth_geometry import unproject_depth
    from lib.tools.geometry.moge_estimate import estimate

    scene = Path(scene_dir)
    source = json.load(open(scene / "moge" / "moge.json"))
    out_dir = scene / "masks" / "edited"
    work_dir = out_dir / f"{tag}_moge2"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = Image.open(edited_image).size
    kn = np.asarray(intrinsics_norm or source["intrinsics_norm"], dtype=float)
    fov_x = math.degrees(2.0 * math.atan(0.5 / float(kn[0, 0])))
    result = estimate(edited_image, str(work_dir), fov_x_deg=fov_x)
    predicted = np.load(result.depth_npy).astype(np.float32)
    reference = np.load(reference_depth or scene / "moge" / "depth.npy").astype(np.float32)
    hole = np.load(hole_mask).astype(bool)
    known = (~hole) & np.isfinite(reference) & np.isfinite(predicted)
    known &= (reference > 0) & (predicted > 0)
    if known.any():
        predicted *= float(np.median(reference[known] / predicted[known]))

    k_px = np.array(
        [
            [kn[0, 0] * width, 0.0, kn[0, 2] * width],
            [0.0, kn[1, 1] * height, kn[1, 2] * height],
            [0.0, 0.0, 1.0],
        ]
    )
    points = unproject_depth(predicted, k_px)
    out_depth = out_dir / f"{tag}_depth.npy"
    out_points = out_dir / f"{tag}_points.npy"
    np.save(out_depth, predicted)
    np.save(out_points, points.astype(np.float32))
    return {
        "ok": True,
        "backend": "moge2",
        "out_depth": str(out_depth),
        "out_points": str(out_points),
    }
