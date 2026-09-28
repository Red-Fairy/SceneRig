"""Small depth-map geometry helpers shared by GT depth and resegmentation."""

from __future__ import annotations

import numpy as np


def unproject_depth(depth: np.ndarray, k_px: np.ndarray) -> np.ndarray:
    """Depth (H, W) + pixel intrinsics (3x3) -> OpenCV camera-frame points."""

    h, w = depth.shape
    fx, fy = float(k_px[0, 0]), float(k_px[1, 1])
    cx, cy = float(k_px[0, 2]), float(k_px[1, 2])
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)


def normals_from_points(points: np.ndarray) -> np.ndarray:
    """Per-pixel unit normals (H, W, 3) from a camera-frame point map."""

    p = points.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        du = np.gradient(p, axis=1)
        dv = np.gradient(p, axis=0)
        n = np.cross(du, dv)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(norm > 1e-12, n / norm, np.nan)
        flip = np.sum(n * p, axis=-1, keepdims=True) > 0
        n = np.where(flip, -n, n)
    n[~np.isfinite(p).all(axis=-1)] = np.nan
    return n.astype(np.float32)
