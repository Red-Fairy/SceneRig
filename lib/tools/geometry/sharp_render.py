"""SHARP raw renders for the pseudo-GT cameras — runs under the SHARP venv (SHARP_PY).

One process per scene: predict the 3DGS once from the source image (MoGE's focal at
predict AND render time, so geometry and cameras agree), then render every novel
camera. Per view it writes

  render_raw.png  un-premultiplied render, NOTHING masked — the gpt-polish input.
                  gsplat composites over black, so colour comes back premultiplied
                  by alpha; divide it out in LINEAR space (the renderer already
                  does exactly this for depth).
  hole_mask.png   L, white=keep / black=hole (alpha < 0.8, dilated long_side//100).
                  Not consumed by the maskless gpt call — it defines the observed
                  region for the drift check in build_pseudo_gt.

Thresholds are the 2026-07-28 sweep values: 0.9+ punches holes through solid
surfaces because fully-covered pixels sit at alpha 0.99-0.999.

    <SHARP_PY> lib/tools/geometry/sharp_render.py \
        --image <scene>/input.png --moge-json <scene>/moge/moge.json \
        --cameras-json <pseudo_gt>/cameras.json --out-dir <pseudo_gt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import color_space as cs
from sharp.utils import gsplat as sg
from sharp.utils import io as sio

HOLE_ALPHA = 0.8


def main() -> None:
    p = argparse.ArgumentParser(description="SHARP raw renders for pseudo-GT cameras")
    p.add_argument("--image", required=True)
    p.add_argument("--moge-json", required=True)
    p.add_argument("--cameras-json", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)

    image, _icc, _f_exif = sio.load_rgb(Path(args.image))
    h, w = image.shape[:2]
    fov_x = float(json.loads(Path(args.moge_json).read_text())["fov_x_deg"])
    f_px = (w / 2.0) / float(np.tan(np.radians(fov_x) / 2.0))
    gaussians = predict_image(predictor, image, f_px, device).to(device)

    intr = torch.tensor(
        [[f_px, 0, (w - 1) / 2.0], [0, f_px, (h - 1) / 2.0], [0, 0, 1]],
        device=device, dtype=torch.float32,
    )
    # predict_image returns linearRGB Gaussians; the renderer converts after blending.
    renderer = sg.GSplatRenderer(color_space="linearRGB")
    dilate = max(w, h) // 100

    cams = json.loads(Path(args.cameras_json).read_text())["views"]
    for cam in cams:
        if float(cam["az"]) == 0.0 and float(cam["el"]) == 0.0:
            continue  # the reference view is the source photo, no render needed
        extr = np.linalg.inv(np.asarray(cam["cam2world_moge"], dtype=np.float64))
        out = renderer(
            gaussians,
            extrinsics=torch.tensor(extr, device=device, dtype=torch.float32)[None],
            intrinsics=intr[None],
            image_width=w,
            image_height=h,
        )
        alpha = out.alpha[0, 0]
        lin = cs.sRGB2linearRGB(out.color[0]) / alpha.clamp(min=1e-4)[None]
        color = cs.linearRGB2sRGB(lin.clamp(0, 1))
        hole = (alpha < HOLE_ALPHA)[None, None].float()
        if dilate > 0:
            hole = torch.nn.functional.max_pool2d(hole, 2 * dilate + 1, stride=1, padding=dilate)
        hole = hole[0, 0].bool()

        vdir = Path(args.out_dir) / cam["tag"]
        vdir.mkdir(parents=True, exist_ok=True)
        rgb = (color.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(rgb).save(vdir / "render_raw.png")
        keep = ((~hole).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(keep, mode="L").save(vdir / "hole_mask.png")
        print(f"[sharp_render] {cam['tag']}: hole {hole.float().mean().item():.1%}", flush=True)
    print("[sharp_render] done", flush=True)


if __name__ == "__main__":
    main()
