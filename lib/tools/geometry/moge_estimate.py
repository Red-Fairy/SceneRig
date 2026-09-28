"""MoGE-2 monocular geometry estimation for the initializer.

Runs MoGE-2 once on the target image and writes camera intrinsics (FOV), a
metric depth map, a camera-space point map, normals and a mask to
``<out_dir>/moge/``. The pipeline uses this to (a) fix the Blender camera to the
source view and (b) place objects at metrically-correct depth.

For a single image there is no absolute world pose: the camera is the canonical
reference frame and MoGE gives the intrinsics (FOV) plus per-pixel 3D points in
that camera frame. The point map is in OpenCV camera convention (+X right, +Y
down, +Z forward into the scene); ``moge_camera`` converts it to the GRASE Z-up
world (camera at origin looking -Y) and derives the Blender camera lens.

Usable both as a library (``estimate``) and a CLI:
    python -m lib.tools.geometry.moge_estimate --image <img> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


@dataclass(kw_only=True, slots=True)
class MoGeResult:
    """Paths + camera summary written for a run."""

    image_width: int
    image_height: int
    fov_x_deg: float
    fov_y_deg: float
    intrinsics_norm: list[list[float]]  # 3x3, normalized (image spans [0,1])
    depth_npy: str
    points_npy: str
    normal_npy: Optional[str]
    mask_npy: Optional[str]
    depth_viz_png: str
    depth_min: float
    depth_median: float
    depth_max: float

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "intrinsics_norm": self.intrinsics_norm,
            "depth_npy": self.depth_npy,
            "points_npy": self.points_npy,
            "normal_npy": self.normal_npy,
            "mask_npy": self.mask_npy,
            "depth_viz_png": self.depth_viz_png,
            "depth_min": self.depth_min,
            "depth_median": self.depth_median,
            "depth_max": self.depth_max,
        }


def fov_from_normalized_focal(fx: float, fy: float) -> tuple[float, float]:
    """Horizontal/vertical FOV (degrees) from MoGE normalized focal lengths.

    MoGE intrinsics are normalized so the image spans [0, 1] with the principal
    point at 0.5; the half-extent is therefore 0.5, so tan(fov/2) = 0.5 / f.
    """
    fov_x = math.degrees(2.0 * math.atan(0.5 / float(fx)))
    fov_y = math.degrees(2.0 * math.atan(0.5 / float(fy)))
    return fov_x, fov_y


def query_point(points: np.ndarray, u: float, v: float) -> Optional[list[float]]:
    """Camera-space 3D point at normalized image coord (u, v) in [0, 1].

    u is horizontal (0=left, 1=right), v is vertical (0=top, 1=bottom). Returns
    None if the sampled point is masked/non-finite. Samples a small median
    window for robustness.
    """
    h, w = points.shape[:2]
    px = int(round(min(max(u, 0.0), 1.0) * (w - 1)))
    py = int(round(min(max(v, 0.0), 1.0) * (h - 1)))
    r = max(1, min(h, w) // 100)
    patch = points[
        max(0, py - r) : min(h, py + r + 1), max(0, px - r) : min(w, px + r + 1)
    ].reshape(-1, 3)
    finite = patch[np.isfinite(patch).all(axis=1)]
    if finite.size == 0:
        return None
    return [float(x) for x in np.median(finite, axis=0)]


def _depth_to_png(depth: np.ndarray, dst: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image

    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        Image.new("RGB", (depth.shape[1], depth.shape[0])).save(dst)
        return
    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm = np.where(np.isfinite(depth), norm, 0.0)
    rgb = (cm.turbo(1.0 - norm)[..., :3] * 255).astype("uint8")  # near=warm
    Image.fromarray(rgb).save(dst)


def estimate(
    image_path: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    fov_x_deg: Optional[float] = None,
) -> MoGeResult:
    """Run MoGE-2 on ``image_path`` and write outputs under ``out_dir``/moge/.

    When ``fov_x_deg`` is given, MoGE uses that horizontal FOV instead of
    predicting it — useful to pin a regenerated/edited image to the source
    camera's intrinsics so depths are directly comparable.
    """
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image

    moge_dir = Path(out_dir) / "moge"
    moge_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(model_name).to(device).eval()

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    arr = torch.tensor(
        np.array(im) / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(arr, fov_x=fov_x_deg)

    k = out["intrinsics"].cpu().numpy().astype(float)
    fov_x, fov_y = fov_from_normalized_focal(k[0, 0], k[1, 1])
    depth = out["depth"].cpu().numpy().astype(np.float32)
    points = out["points"].cpu().numpy().astype(np.float32)
    normal = out["normal"].cpu().numpy().astype(np.float32) if "normal" in out else None
    mask = out["mask"].cpu().numpy() if "mask" in out else None

    depth_npy = moge_dir / "depth.npy"
    points_npy = moge_dir / "points.npy"
    viz_png = moge_dir / "depth_viz.png"
    np.save(depth_npy, depth)
    np.save(points_npy, points)
    normal_npy = None
    if normal is not None:
        normal_npy = moge_dir / "normal.npy"
        np.save(normal_npy, normal)
    mask_npy = None
    if mask is not None:
        mask_npy = moge_dir / "mask.npy"
        np.save(mask_npy, np.asarray(mask).astype(np.uint8))
    _depth_to_png(depth, viz_png)

    finite = depth[np.isfinite(depth)]
    d_min = float(finite.min()) if finite.size else 0.0
    d_med = float(np.median(finite)) if finite.size else 0.0
    d_max = float(finite.max()) if finite.size else 0.0

    result = MoGeResult(
        image_width=w,
        image_height=h,
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
        intrinsics_norm=k.tolist(),
        depth_npy=str(depth_npy),
        points_npy=str(points_npy),
        normal_npy=str(normal_npy) if normal_npy else None,
        mask_npy=str(mask_npy) if mask_npy else None,
        depth_viz_png=str(viz_png),
        depth_min=d_min,
        depth_median=d_med,
        depth_max=d_max,
    )
    with open(moge_dir / "moge.json", "w") as f:
        json.dump(
            {**result.to_dict(), "backend": "moge2", "model": model_name}, f, indent=2
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="MoGE-2 geometry estimation")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES"))
    p.add_argument(
        "--fov-x-deg",
        type=float,
        default=None,
        help="Pin horizontal FOV (deg) instead of letting MoGE predict it.",
    )
    args = p.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    res = estimate(
        args.image, args.out_dir, model_name=args.model, fov_x_deg=args.fov_x_deg
    )
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
