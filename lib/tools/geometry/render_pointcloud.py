"""Render the MoGE point cloud from a new camera (NumPy z-buffer splat).

MoGE writes a per-pixel camera-space point map ``points.npy`` (H, W, 3) in the
OpenCV camera convention (+X right, +Y down, +Z forward) with the source camera
at the origin. This module treats that source-camera frame as the world frame
and re-renders the cloud from an arbitrary new camera given as a 4x4
camera-to-world pose. The source image supplies per-point colour (the point map
itself stores no colour).

Because a single-view cloud has no geometry behind what the source camera saw,
a novel view leaves disocclusion holes. The renderer returns both the splatted
RGB image and a ``filled`` mask; the holes are exactly the region a generative
model is asked to complete (see ``gpt_inpaint``). With an identity pose the
render reproduces the source view (holes only where MoGE marked pixels invalid),
which is the renderer's sanity check.

Usable as a library (``render_view``) and a CLI::

    python -m lib.tools.geometry.render_pointcloud \
        --moge-dir <out_dir>/moge --image <source.png> --out-dir <dir> \
        --pose 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(kw_only=True, slots=True)
class RenderResult:
    """Paths written for a novel-view render."""

    render_png: str
    mask_png: str  # RGBA, transparent over holes (OpenAI edit convention)
    mask_vis_png: str  # grayscale, white = hole (debugging)
    out_width: int
    out_height: int
    filled_fraction: float  # share of output pixels that received a point


def intrinsics_pixels(
    moge_json: dict, out_w: int, out_h: int
) -> tuple[float, float, float, float]:
    """Pixel pinhole intrinsics (fx, fy, cx, cy) for an ``out_w`` x ``out_h`` render.

    MoGE stores ``intrinsics_norm`` normalized so the image spans [0, 1]; scaling
    by the output size keeps the field of view fixed at any resolution.
    """
    k = np.asarray(moge_json["intrinsics_norm"], dtype=np.float64)
    fx = float(k[0, 0]) * out_w
    fy = float(k[1, 1]) * out_h
    cx = float(k[0, 2]) * out_w
    cy = float(k[1, 2]) * out_h
    return fx, fy, cx, cy


def splat_render(
    points: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam2world: np.ndarray,
    out_h: int,
    out_w: int,
    point_radius: int = 1,
    bg: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Splat a camera-space point cloud into a new view via a painter z-buffer.

    ``points`` (H, W, 3) world-frame points (== MoGE source-camera OpenCV frame),
    ``colors`` (H, W, 3) uint8, ``valid`` (H, W) bool. ``cam2world`` is the new
    camera's 4x4 pose in that frame. Returns ``(rgb uint8 (out_h,out_w,3),
    filled bool (out_h,out_w))``. Nearest point wins per pixel; each point is
    drawn as a ``(2*point_radius+1)`` square splat.
    """
    world2cam = np.linalg.inv(np.asarray(cam2world, dtype=np.float64))
    r_mat, t_vec = world2cam[:3, :3], world2cam[:3, 3]

    p = points.reshape(-1, 3)[valid.reshape(-1)]
    c = colors.reshape(-1, 3)[valid.reshape(-1)]
    finite = np.isfinite(p).all(axis=1)
    p, c = p[finite], c[finite]

    pc = p @ r_mat.T + t_vec
    z = pc[:, 2]
    front = z > 1e-6
    pc, c, z = pc[front], c[front], z[front]

    u = fx * pc[:, 0] / pc[:, 2] + cx
    v = fy * pc[:, 1] / pc[:, 2] + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)
    ui, vi, c, z = ui[inb], vi[inb], c[inb], z[inb]

    # Painter's algorithm: draw far points first so nearer points overwrite them.
    order = np.argsort(-z)
    ui, vi, c = ui[order], vi[order], c[order]

    img = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    filled = np.zeros((out_h, out_w), dtype=bool)
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            yy, xx = vi + dy, ui + dx
            m = (yy >= 0) & (yy < out_h) & (xx >= 0) & (xx < out_w)
            img[yy[m], xx[m]] = c[m]
            filled[yy[m], xx[m]] = True
    return img, filled


def hole_mask_rgba(filled: np.ndarray) -> np.ndarray:
    """RGBA mask (H, W, 4) for OpenAI image edit: holes are transparent (a=0).

    The OpenAI edit endpoint regenerates the fully-transparent regions and keeps
    the opaque ones, so transparent == "complete this".
    """
    h, w = filled.shape
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    mask[..., :3] = 255
    mask[..., 3] = np.where(filled, 255, 0).astype(np.uint8)
    return mask


def load_inputs(
    moge_dir: str, image_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load points + colours + validity from a MoGE output dir and a source image.

    Returns ``(points (H,W,3), colors (H,W,3) uint8, valid (H,W) bool, moge_json)``,
    with the source image resized to the point-map resolution.
    """
    from PIL import Image

    moge_dir = Path(moge_dir)
    with open(moge_dir / "moge.json") as f:
        moge_json = json.load(f)
    points = np.load(moge_dir / "points.npy").astype(np.float32)
    h, w = points.shape[:2]

    img = Image.open(image_path).convert("RGB").resize((w, h), Image.BILINEAR)
    colors = np.asarray(img, dtype=np.uint8)

    mask_path = moge_dir / "mask.npy"
    if mask_path.exists():
        valid = np.load(mask_path).astype(bool)
    else:
        valid = np.isfinite(points).all(axis=2)
    return points, colors, valid, moge_json


def _parse_pose(pose: Optional[str], pose_file: Optional[str]) -> np.ndarray:
    """Resolve a 4x4 camera-to-world pose from a CLI string or a .npy/.json file."""
    if pose_file:
        if pose_file.endswith(".npy"):
            m = np.load(pose_file)
        else:
            with open(pose_file) as f:
                m = np.asarray(json.load(f), dtype=np.float64)
        return np.asarray(m, dtype=np.float64).reshape(4, 4)
    if pose:
        vals = [float(x) for x in pose.replace(",", " ").split()]
        if len(vals) != 16:
            raise ValueError(f"--pose needs 16 numbers, got {len(vals)}")
        return np.asarray(vals, dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


def render_view(
    moge_dir: str,
    image_path: str,
    out_dir: str,
    cam2world: np.ndarray,
    out_w: Optional[int] = None,
    out_h: Optional[int] = None,
    point_radius: int = 1,
) -> RenderResult:
    """Render the MoGE cloud from ``cam2world`` and write render + hole masks."""
    from PIL import Image

    points, colors, valid, moge_json = load_inputs(moge_dir, image_path)
    h, w = points.shape[:2]
    out_w, out_h = out_w or w, out_h or h
    fx, fy, cx, cy = intrinsics_pixels(moge_json, out_w, out_h)

    rgb, filled = splat_render(
        points,
        colors,
        valid,
        fx,
        fy,
        cx,
        cy,
        cam2world,
        out_h,
        out_w,
        point_radius=point_radius,
    )
    mask_rgba = hole_mask_rgba(filled)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    render_png = out / "render.png"
    mask_png = out / "hole_mask.png"
    mask_vis_png = out / "hole_mask_vis.png"
    Image.fromarray(rgb).save(render_png)
    Image.fromarray(mask_rgba, mode="RGBA").save(mask_png)
    Image.fromarray(((~filled) * 255).astype(np.uint8)).save(mask_vis_png)

    return RenderResult(
        render_png=str(render_png),
        mask_png=str(mask_png),
        mask_vis_png=str(mask_vis_png),
        out_width=out_w,
        out_height=out_h,
        filled_fraction=float(filled.mean()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Render MoGE point cloud from a new camera")
    p.add_argument("--moge-dir", required=True, help="dir with points.npy + moge.json")
    p.add_argument("--image", required=True, help="source image (per-point colour)")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--pose",
        default=None,
        help="16 comma/space-separated floats, row-major cam2world",
    )
    p.add_argument("--pose-file", default=None, help=".npy or .json 4x4 cam2world")
    p.add_argument("--out-width", type=int, default=None)
    p.add_argument("--out-height", type=int, default=None)
    p.add_argument("--point-radius", type=int, default=1)
    args = p.parse_args()

    cam2world = _parse_pose(args.pose, args.pose_file)
    res = render_view(
        args.moge_dir,
        args.image,
        args.out_dir,
        cam2world,
        out_w=args.out_width,
        out_h=args.out_height,
        point_radius=args.point_radius,
    )
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
