"""Ground-truth depth ingestion — a MoGE-2 drop-in backend for known depth.

Takes a metric Z-depth map (``.npy``, distance along the optical axis — NOT ray
length) aligned pixel-per-pixel with the input RGB (e.g. the robolab top
camera's aligned depth) and writes the SAME artifact contract as
:mod:`moge_estimate` under ``<out_dir>/moge/`` (depth.npy, points.npy,
normal.npy, mask.npy, depth_viz.png, moge.json), so every downstream consumer
(segmentation sizing, SAM3D conditioning, placement, ICP, register, novel
views) runs on the provided GT geometry.

Intrinsics resolution order:
  1. ``--fov-x-deg`` (square pixels, centered principal point),
  2. ``intrinsics.json`` next to the depth file (or ``--intrinsics``): pixel
     units ``{fx, fy, cx, cy, w, h}`` at the depth map's resolution,
  3. MoGE-2 run once for its predicted FOV ONLY (its depth is discarded) —
     the only case where a monocular model still loads.

Usable both as a library (``estimate``) and a CLI:
    python -m lib.tools.geometry.gt_depth_estimate --image <img> --out-dir <dir> --depth <npy>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from lib.tools.geometry.depth_geometry import normals_from_points, unproject_depth
from lib.tools.geometry.moge_estimate import (
    MoGeResult,
    _depth_to_png,
    fov_from_normalized_focal,
)


def _resize_depth(m: np.ndarray, w: int, h: int) -> np.ndarray:
    """NEAREST resize — metric depth must not blend across object boundaries."""
    import cv2

    if m.shape[:2] == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def _k_norm_from_moge(image_path: str, device: Optional[str] = None) -> np.ndarray:
    """Run MoGE-2 for its predicted intrinsics ONLY (normalized 3x3); depth discarded."""
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image

    from lib.tools.geometry.moge_estimate import DEFAULT_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(DEFAULT_MODEL).to(device).eval()
    im = Image.open(image_path).convert("RGB")
    arr = torch.tensor(
        np.array(im) / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(arr)
    return out["intrinsics"].cpu().numpy().astype(np.float64)


def estimate(
    image_path: str,
    out_dir: str,
    depth_path: str,
    intrinsics_path: Optional[str] = None,
    fov_x_deg: Optional[float] = None,
    device: Optional[str] = None,
) -> MoGeResult:
    """Ingest GT depth for ``image_path`` and write MoGE-contract outputs under ``out_dir``/moge/."""
    from PIL import Image

    moge_dir = Path(out_dir) / "moge"
    moge_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    depth0 = np.load(depth_path).astype(np.float32)
    if depth0.ndim != 2:
        raise ValueError(f"GT depth must be (H, W), got shape {depth0.shape}")
    if abs((depth0.shape[1] / depth0.shape[0]) / (w / h) - 1.0) > 0.01:
        raise ValueError(
            f"GT depth aspect {depth0.shape[1]}x{depth0.shape[0]} does not match "
            f"image {w}x{h} — the depth must be pixel-aligned with the RGB"
        )

    if fov_x_deg is not None:
        fxn = 0.5 / math.tan(math.radians(float(fov_x_deg)) / 2.0)
        k_norm = np.array(  # square pixels, principal point centered
            [[fxn, 0.0, 0.5], [0.0, fxn * w / h, 0.5], [0.0, 0.0, 1.0]]
        )
        intr_source = "fov_x_deg"
    else:
        cand = (
            Path(intrinsics_path)
            if intrinsics_path
            else Path(depth_path).parent / "intrinsics.json"
        )
        if cand.exists():
            intr = json.load(open(cand))
            w0 = int(intr.get("w", depth0.shape[1]))
            h0 = int(intr.get("h", depth0.shape[0]))
            if (h0, w0) != depth0.shape:
                raise ValueError(
                    f"intrinsics.json is for {w0}x{h0} but the depth map is "
                    f"{depth0.shape[1]}x{depth0.shape[0]}"
                )
            k_norm = np.array(
                [
                    [intr["fx"] / w0, 0.0, intr["cx"] / w0],
                    [0.0, intr["fy"] / h0, intr["cy"] / h0],
                    [0.0, 0.0, 1.0],
                ]
            )
            intr_source = f"file:{cand}"
        elif intrinsics_path:
            raise FileNotFoundError(f"intrinsics file not found: {intrinsics_path}")
        else:
            k_norm = _k_norm_from_moge(image_path, device=device)
            intr_source = "moge2-fov"

    depth = _resize_depth(depth0, w, h)
    mask = np.isfinite(depth) & (depth > 0)
    k_px = k_norm.copy()
    k_px[0] *= w
    k_px[1] *= h
    points = unproject_depth(depth, k_px)
    normal = normals_from_points(points)
    points[~mask] = np.inf  # MoGE marks invalid points non-finite; keep that contract

    fov_x, fov_y = fov_from_normalized_focal(k_norm[0, 0], k_norm[1, 1])

    depth_npy = moge_dir / "depth.npy"
    points_npy = moge_dir / "points.npy"
    normal_npy = moge_dir / "normal.npy"
    mask_npy = moge_dir / "mask.npy"
    viz_png = moge_dir / "depth_viz.png"
    np.save(depth_npy, depth)
    np.save(points_npy, points)
    np.save(normal_npy, normal)
    np.save(mask_npy, mask.astype(np.uint8))
    _depth_to_png(np.where(mask, depth, np.inf), viz_png)

    finite = depth[mask]
    d_min = float(finite.min()) if finite.size else 0.0
    d_med = float(np.median(finite)) if finite.size else 0.0
    d_max = float(finite.max()) if finite.size else 0.0

    result = MoGeResult(
        image_width=w,
        image_height=h,
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
        intrinsics_norm=k_norm.tolist(),
        depth_npy=str(depth_npy),
        points_npy=str(points_npy),
        normal_npy=str(normal_npy),
        mask_npy=str(mask_npy),
        depth_viz_png=str(viz_png),
        depth_min=d_min,
        depth_median=d_med,
        depth_max=d_max,
    )
    payload = {
        **result.to_dict(),
        "backend": "gt",
        "depth_source": str(depth_path),
        "intrinsics_source": intr_source,
    }
    with open(moge_dir / "moge.json", "w") as f:
        json.dump(payload, f, indent=2)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="GT depth ingestion (MoGE contract)")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--depth", required=True, help="Metric Z-depth .npy aligned with --image")
    p.add_argument(
        "--intrinsics",
        default=None,
        help="Pixel-unit intrinsics json {fx,fy,cx,cy,w,h}; default: intrinsics.json "
        "next to --depth",
    )
    p.add_argument(
        "--fov-x-deg",
        type=float,
        default=None,
        help="Horizontal FOV (deg) override — wins over the intrinsics file.",
    )
    p.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES"))
    args = p.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    res = estimate(
        args.image,
        args.out_dir,
        args.depth,
        intrinsics_path=args.intrinsics,
        fov_x_deg=args.fov_x_deg,
    )
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
