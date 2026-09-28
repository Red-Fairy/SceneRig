""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lib.tools.geometry.agentic_mask import _dilate1

PAD_FRAC = 0.25
HOLE_DILATE_PX = 8

OUTPAINT_NEG = "floating text, caption, watermark, user interface panel"


def outpaint_prompt(side: str, target: str) -> str:
    ""
    category = target.split("#")[0]
    return (
        f"Extend the scene to the {side}. Continue the existing support and background "
        f"surfaces and the {side} side "
        f"of the {category} that is cut off by the {side} edge (its frame, body and any "
        "stand or base). Keep the style, lighting and perspective consistent; do not add "
        "any new object or duplicate anything. Do not invent captions, watermarks, "
        "unrelated readable text, or new user-interface elements. If visible content "
        "belongs to the cut target and naturally continues across the boundary, preserve "
        "that continuation."
    )


def _shift(arr: np.ndarray, side: str, D: int, fill) -> np.ndarray:
    """Pad-shift ``arr`` (H,W[,C]) by ``D`` px, opening a blank strip on ``side`` (same
    size). Kept for the geometry tests; the live path uses native outpaint instead."""
    W = arr.shape[1]
    out = np.full_like(arr, fill)
    if side == "left":
        out[:, D:] = arr[:, : W - D]
    else:
        out[:, : W - D] = arr[:, D:]
    return out


def _strip(H: int, W: int, side: str, D: int) -> np.ndarray:
    s = np.zeros((H, W), bool)
    if side == "left":
        s[:, :D] = True
    else:
        s[:, W - D :] = True
    return s


def border_cut_side(mask: np.ndarray, margin: int = 3) -> Optional[str]:
    ""
    left = bool(mask[:, :margin].any())
    right = bool(mask[:, -margin:].any())
    if left == right:  # neither, or both edges -> not a single-side cut
        return None
    return "left" if left else "right"


def edge_contact_ratio(mask: np.ndarray, side: str, margin: int = 3) -> float:
    """Fraction of the mask's bbox HEIGHT that touches the ``side`` frame edge.

    High-band magnitude signal for the border gate. The middle band is ambiguous;
    there only the VLM's
    beyond_frame_fraction separates (uncut <= 0.05 vs cut >= 0.10) — so this
    ratio must never be used alone; it OR-gates with the VLM floor."""
    col = mask[:, :margin].any(axis=1) if side == "left" else mask[:, -margin:].any(axis=1)
    rows = np.where(mask.any(axis=1))[0]
    if not len(rows):
        return 0.0
    return float(col.sum() / (rows.max() - rows.min() + 1))


def _ensure_canvas(path: str, w: int, h: int, label: str) -> None:
    """Force an edited image back to its expected canvas size (in place).

    The LanPaint/Qwen workers run at their own canonical resolution and may return
    e.g. 1402x1122 for a 1138x912 request. Every array downstream of an edit here
    (occ holes, padded depth, target masks, content offsets) assumes the canvas is
    exactly what was requested — the mismatch broadcast-crashed border completion on
    gpt1-family scenes on every attempt since 0718 (15 logged failures, >50 % of
    fires), silently discarding the full Qwen edit each time. Resizing back at the
    boundary restores the contract at the cost of one LANCZOS resample."""
    from PIL import Image

    im = Image.open(path)
    if im.size != (w, h):
        print(
            f"[border_complete] {label}: edit returned {im.size}, "
            f"resizing to {(w, h)} to restore the canvas contract"
        )
        im.resize((w, h), Image.LANCZOS).save(path)


def _remove_occluders(scene, out_dir, slug, occluders, masks, desc, qwen):
    """LanPaint inpaint removing ``occluders`` on input.png; returns (deoccluded_png,
    occ_hole HxW bool) or (input.png, empty hole) when there is nothing to remove."""
    from PIL import Image

    img = Image.open(scene / "input.png").convert("RGB")
    w, h = img.size
    occ = np.zeros((h, w), bool)
    for o in occluders:
        if o in masks:
            occ |= masks[o]
    if not occ.any():
        return str(scene / "input.png"), occ
    for _ in range(HOLE_DILATE_PX):
        occ = _dilate1(occ)
    em = out_dir / f"{slug}_border_rm_mask.png"
    Image.fromarray(np.where(occ, 0, 255).astype("uint8"), "L").save(em)
    from lib.tools.geometry.generative_resegment import (  # lazy: parent imports us
        build_removal_prompt,
    )

    prompt = build_removal_prompt(occluders, desc)
    out = out_dir / f"{slug}_border_deoccluded.png"
    resp = qwen.inpaint(str(scene / "input.png"), str(em), prompt, str(out))
    if not resp or not resp.get("ok"):
        print(f"[border_complete] {slug}: occluder removal failed; outpainting original")
        return str(scene / "input.png"), np.zeros((h, w), bool)
    _ensure_canvas(str(out), w, h, slug)  # inpaint may come back at worker resolution
    return str(out), occ


def border_complete(
    scene_dir: str,
    target: str,
    side: str,
    occluders: list[str],
    masks: dict[str, np.ndarray],
    desc: dict[str, str],
    qwen: Any,
    sam3: Any = None,
    gpu: Optional[str] = None,
) -> Optional[dict]:
    """Remove occluders (LanPaint inpaint) then extend the cut side (native outpaint) ->
    MoGE-2 depth refresh (shifted principal point at the new width) -> unproject ->
    re-segment. Returns a redetect record with ``points_npy`` + a ``border`` marker, or
    None on failure. ``masks``/``desc`` are keyed by rid."""
    from PIL import Image

    from lib.tools.geometry.agentic_mask import Sam3Server
    from lib.tools.geometry.depth_geometry import unproject_depth
    from lib.tools.geometry.depth_refine import refine_depth
    from lib.utils._path import SAM3_PY

    scene = Path(scene_dir)
    out_dir = scene / "masks" / "edited"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = target.replace("#", "").replace(" ", "_")

    img = np.asarray(Image.open(scene / "input.png").convert("RGB"))
    H, W = img.shape[:2]
    PAD = max(16, int(round(PAD_FRAC * W / 16)) * 16)
    moge = json.load(open(scene / "moge" / "moge.json"))
    Kn = np.asarray(moge["intrinsics_norm"], float)
    fx, fy = Kn[0, 0] * W, Kn[1, 1] * H
    cx0, cy = Kn[0, 2] * W, Kn[1, 2] * H
    depth = np.load(scene / "moge" / "depth.npy").astype(np.float32)

    # 1) remove occluders (separate inpaint pass; outpaint API forbids a mask)
    base_image, occ_hole = _remove_occluders(scene, out_dir, slug, occluders, masks, desc, qwen)

    # 2) native outpaint on the cut side
    pad_spec = f"{'l' if side == 'left' else 'r'}{PAD}"
    prompt = outpaint_prompt(side, target)
    final_path = out_dir / f"{slug}_border.png"
    resp = qwen.outpaint(base_image, pad_spec, prompt, str(final_path), neg=OUTPAINT_NEG)
    if not resp or not resp.get("ok"):
        print(f"[border_complete] {target}: outpaint failed ({(resp or {}).get('error')})")
        return None
    # Expected canvas is EXACTLY source + the one-side pad. Do not trust the worker's
    # reported dims — normalize the file and derive (Wn, Hn) from the request, so the
    # depth pad / hole / offset geometry below is consistent by construction.
    Wn, Hn = W + PAD, H
    _ensure_canvas(str(final_path), Wn, Hn, target)

    # 3) content offset in the expanded canvas + shifted principal point
    xoff = PAD if side == "left" else 0
    cx_new = cx0 + (PAD if side == "left" else 0)

    # padded depth (Wn x Hn): original depth at [xoff:xoff+W], everything else = hole.
    sh_depth = np.zeros((Hn, Wn), np.float32)
    sh_depth[:H, xoff : xoff + W] = depth
    hole = np.ones((Hn, Wn), bool)
    hole[:H, xoff : xoff + W] = False
    if occ_hole.any():  # removed occluders' original depth belongs to them -> re-complete
        hole[:H, xoff : xoff + W] |= occ_hole
    for _ in range(HOLE_DILATE_PX):
        hole = _dilate1(hole)

    Kn_sh = np.array(
        [[fx / Wn, 0, cx_new / Wn], [0, fy / Hn, cy / Hn], [0, 0, 1]], float
    )
    sh_depth_p = out_dir / f"{slug}_border_depth.npy"
    hole_p = out_dir / f"{slug}_border_hole.npy"
    np.save(sh_depth_p, sh_depth)
    np.save(hole_p, hole.astype(np.uint8))
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    refreshed = refine_depth(
        scene_dir,
        str(final_path),
        str(hole_p),
        f"{slug}_border_refined",
        reference_depth=str(sh_depth_p),
        intrinsics_norm=Kn_sh.tolist(),
    )
    if not refreshed:
        print(f"[border_complete] {target}: MoGE-2 refresh failed")
        return None

    # 4) unproject refreshed depth with the shifted pixel-K -> points in the ORIGINAL frame
    refined = np.load(refreshed["out_depth"])
    k_px = np.array([[fx, 0, cx_new], [0, fy, cy], [0, 0, 1]], float)
    points = unproject_depth(refined, k_px)
    points_npy = out_dir / f"{slug}_border_points.npy"
    np.save(points_npy, points.astype(np.float32))

    # 5) re-segment the completed object on the outpainted canvas
    own = None
    if sam3 is None:
        own = sam3 = Sam3Server(SAM3_PY, log_path=str(out_dir / "sam3_border.log"))
    cat = target.split("#")[0]
    # original partial mask placed at the content offset, for the overlap gate
    tgt_full = np.zeros((Hn, Wn), bool)
    tgt_full[:H, xoff : xoff + W] = masks[target]
    completed_p = scene / "masks" / f"{slug}_border.npy"
    try:
        all_m, scores = sam3.segment_all(str(final_path), cat, str(completed_p))
    finally:
        if own is not None:
            own.close()
    best, best_a, best_s = None, 0, 0.0
    if all_m is not None and len(all_m):
        for m, s in zip(all_m, scores):
            m = m > 0
            if (m & tgt_full).sum() > 0 and m.sum() > best_a:
                best, best_a, best_s = m, int(m.sum()), float(s)
    if best is None or best.sum() <= int(masks[target].sum()):
        print(f"[border_complete] {target}: re-segmentation found no gain; skipping")
        return None
    np.save(completed_p, best)

    return {
        "mask_path": str(completed_p),
        "edited_image": str(final_path),
        "points_npy": str(points_npy),
        "seg_prompt": cat,
        "score": best_s,
        "removed": occluders,
        "prompt": prompt,
        "vital_part": "",
        "border": {
            "side": side, "pad": int(PAD), "width": int(Wn), "height": int(Hn),
            "xoff": int(xoff), "cx_norm": float(cx_new / Wn),
        },
    }
