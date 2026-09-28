"""Objective for per-object pose registration (pure, unit-tested).

PoseSession isolates an object with the object visible and ancestors, occluders, and
support children acting as silhouette holdouts. It renders through the reference-view
camera and scores how well the object matches the reference by combining:

  score = w_iou * silhouette_IoU(render, SAM3 mask)
        + w_dino * appearance_similarity(render_crop, reference_crop)
        - w_depth * |object_depth - MoGE_depth| / MoGE_depth
        - w_size * |sqrt(render_area / mask_area) - 1|

IoU and the explicit size loss drive in-plane translation and scale; the depth
regularizer breaks the depth<->scale projection degeneracy (both change apparent size
from one view) by pulling depth toward the metric MoGE prior. The optional DINO function
keeps this module model-free for tests, but the live PoseSession caller does not pass it
here: rotation search adds appearance separately as ``score + LAMBDA_FEAT * similarity``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def coverage(rgba: np.ndarray) -> np.ndarray:
    """Per-pixel foreground COVERAGE in [0, 1] from a rendered RGBA image's alpha.
    Antialiased alpha IS subpixel coverage, so summing it estimates the silhouette
    area without the thresholding bias that deletes thin structures: a ~1px-wide
    spoon handle renders as sub-0.5-alpha pixels that a hard threshold erases
    entirely (0720_orinit2_abc2 spoon#1: 132 of ~170 true pixels survived
    ``alpha > 0.5`` at 512-res — size_loss ~2x overstated)."""
    a = np.asarray(rgba)
    if a.ndim == 3 and a.shape[2] >= 4:
        chan = a[..., 3]
    else:  # no alpha -> treat non-black pixels as foreground
        chan = a[..., :3].max(axis=2) if a.ndim == 3 else a
    chan = chan.astype(np.float64)
    if chan.max() > 1.0:
        chan = chan / 255.0
    return np.clip(chan, 0.0, 1.0)


def silhouette(rgba: np.ndarray, alpha_thresh: float = 0.5) -> np.ndarray:
    """Binary foreground mask from a rendered RGBA image (alpha channel)."""
    return coverage(rgba) > alpha_thresh


def _resize_bool(m: np.ndarray, h: int, w: int) -> np.ndarray:
    if m.shape == (h, w):
        return m
    from PIL import Image

    return np.asarray(Image.fromarray(m.astype("uint8") * 255).resize((w, h))) > 127


def _soft_mask(m: np.ndarray, h: int, w: int) -> np.ndarray:
    """The GT mask as float coverage in [0, 1] at (h, w) — bilinear when resized,
    so a downsampled thin mask keeps its fractional boundary area instead of the
    ``> 127`` cliff (the mask-side twin of ``coverage``)."""
    m = (np.asarray(m) > 0).astype(np.float64)
    if m.shape == (h, w):
        return m
    from PIL import Image

    im = Image.fromarray((m * 255).astype("uint8")).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (b is resized to a's shape if needed)."""
    a = np.asarray(a) > 0
    b = _resize_bool(np.asarray(b) > 0, *a.shape)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def depth_penalty(depth: Optional[float], moge_depth: Optional[float]) -> float:
    """Relative depth deviation from the MoGE prior, in [0, ~1]."""
    if depth is None or not moge_depth:
        return 0.0
    return abs(depth - moge_depth) / max(abs(moge_depth), 1e-6)


def _shift2d(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer-shift a 2D array with zero fill (no wraparound).

    Lives HERE, not in register.py, so both can use one implementation: register.py already
    imports this module as ``ro``, and the reverse import would be a cycle. dtype-agnostic —
    the float coverage/mask arrays below and register.py's bool silhouettes both work.
    """
    out = np.zeros_like(a)
    h, w = a.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def centered_iou(cov: np.ndarray, m: np.ndarray) -> float:
    ""
    cs, ms = float(cov.sum()), float(m.sum())
    if cs <= 0.0 or ms <= 0.0:
        return 0.0
    h, w = cov.shape
    ry, rx = np.arange(h), np.arange(w)
    dy = int(round(float(cov.sum(1) @ ry) / cs - float(m.sum(1) @ ry) / ms))
    dx = int(round(float(cov.sum(0) @ rx) / cs - float(m.sum(0) @ rx) / ms))
    s = _shift2d(m, dy, dx)
    union = float(np.maximum(cov, s).sum())
    return float(np.minimum(cov, s).sum()) / union if union else 0.0


def objective(
    render_rgba: np.ndarray,
    sam3_mask: np.ndarray,
    *,
    depth: Optional[float] = None,
    moge_depth: Optional[float] = None,
    render_crop: Optional[np.ndarray] = None,
    ref_crop: Optional[np.ndarray] = None,
    dino: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    w_iou: float = 1.0,
    w_dino: float = 0.5,
    w_depth: float = 0.4,
    w_size: float = 0.3,
) -> dict[str, float]:
    """Combined registration score (higher = better). Returns the parts too.

    ``w_size`` weights a SILHOUETTE-SIZE loss: |sqrt(A_render/A_mask) - 1|, the linear
    size ratio between the rendered silhouette and the target mask. IoU is a poor SIZE
    signal when the reconstructed shape is imperfect (a too-small spoon scaled up barely
    changes IoU, so the scale search left it small); the size loss reads the size error
    directly (34% too small -> 0.34). It only actively moves selection where the
    projected size changes (scale — where it rides the centered selection score, see
    ``centered_iou`` — and the depth/y axis) — for xy/rotation the area is ~constant so
    it cancels in the gain. Tuned to 0.3 on abc2 (0717): grows an under-sized spoon
    1.0->~1.3 while leaving well-sized objects at 1.0 up to w=0.8.

    IoU and the area terms are SOFT (coverage-weighted): the render's antialiased
    alpha and a bilinear-resized mask are compared via sum(min)/sum(max) instead of
    thresholded counts. On binary inputs this equals the hard IoU exactly; for thin
    or small objects it removes the 0.5-alpha threshold bias (which halved a thin
    spoon's measured area) and the ~1/mask_px quantization of the accept signal."""
    cov = coverage(render_rgba)
    m = _soft_mask(sam3_mask, *cov.shape)
    union = float(np.maximum(cov, m).sum())
    iou = float(np.minimum(cov, m).sum()) / union if union else 0.0
    sim = (
        dino(render_crop, ref_crop)
        if dino is not None and render_crop is not None and ref_crop is not None
        else 0.0
    )
    pen = depth_penalty(depth, moge_depth)
    a_ren, a_msk = float(cov.sum()), float(m.sum())
    ratio = (a_ren / a_msk) ** 0.5 if a_msk > 0 and a_ren > 0 else 1.0
    size_loss = abs(ratio - 1.0)
    score = w_iou * iou + w_dino * sim - w_depth * pen - w_size * size_loss
    return {
        "score": score, "iou": iou, "dino": float(sim), "depth_penalty": pen,
        "size_loss": size_loss, "size_ratio": ratio,
        # translation-invariant twin of `iou` — rotation AND scale SELECT on it (phase
        # C1 + the scale cutover, register.optimize_axis; scale swaps it for the iou
        # term of `score`, keeping the size/depth terms); diagnostic on the translation
        # axes. Deliberately NOT folded into `score` here: that would change every axis.
        "centered_iou": centered_iou(cov, m),
    }  # fmt: skip
