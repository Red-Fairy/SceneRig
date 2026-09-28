"""One-shot LingBot-Depth worker (runs under the isolated lingbot venv, LINGBOT_PY).

Refines/completes a metric depth map whose EDIT HOLE has been invalidated: the masked
LanPaint+Qwen removal regenerates pixels inside the hole, so the original MoGE depth
there belongs to the removed occluders — this worker feeds the edited RGB + the
hole-zeroed depth + the MoGE intrinsics to LingBot (masked depth modeling) and writes
the completed depth and camera-space point map.

    LINGBOT_PY lib/tools/geometry/lingbot_worker.py <request.json>

request.json: {"image": <edited png>, "depth": <moge depth.npy>,
               "hole": <hole npy/png with alpha==0 marking the hole>,
               "intrinsics_norm": [[...3x3...]],
               "out_depth": <npy>, "out_points": <npy>}
Prints one JSON line: {"ok": true, ...} | {"ok": false, "error": ...}.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch


def _load_hole(path: str, shape) -> np.ndarray:
    """Boolean hole mask from a .npy or the RGBA hole PNG (alpha==0 = hole)."""
    if path.endswith(".npy"):
        m = np.load(path) > 0
    else:
        from PIL import Image

        m = np.asarray(Image.open(path).convert("RGBA"))[..., 3] == 0
    if m.shape != shape:
        from PIL import Image

        m = (
            np.asarray(
                Image.fromarray(m.astype("uint8") * 255).resize((shape[1], shape[0]))
            )
            > 127
        )
    return m


def main() -> None:
    from mdm.model.v2 import MDMModel
    from PIL import Image

    req = json.load(open(sys.argv[1]))
    img = np.asarray(Image.open(req["image"]).convert("RGB"))
    depth = np.load(req["depth"]).astype(np.float32)
    if depth.shape != img.shape[:2]:  # align depth to the (input-res) edited image
        depth = np.asarray(
            Image.fromarray(depth).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        )
    hole = _load_hole(req["hole"], depth.shape)
    depth_orig = depth.copy()  # aligned original, for the agreement check below
    depth = depth.copy()
    depth[hole] = 0.0  # invalid -> LingBot completes it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDMModel.from_pretrained(
        req.get("model", "robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    ).to(device)
    model.eval()
    image_t = (
        torch.tensor(img / 255.0, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    depth_t = torch.tensor(depth, dtype=torch.float32, device=device).unsqueeze(0)
    K = torch.tensor(
        np.asarray(req["intrinsics_norm"], dtype=np.float32), device=device
    ).unsqueeze(0)
    with torch.no_grad():
        out = model.infer(image_t, depth_in=depth_t, intrinsics=K, apply_mask=False)
    refined = out["depth"].squeeze(0).float().cpu().numpy()
    points = out["points"].squeeze(0).float().cpu().numpy()
    np.save(req["out_depth"], refined.astype(np.float32))
    np.save(req["out_points"], points.astype(np.float32))
    # agreement over the untouched region = sanity signal for the caller
    keep = ~hole & np.isfinite(refined) & (depth_orig > 0)
    rel = np.abs(refined[keep] - depth_orig[keep]) / np.maximum(depth_orig[keep], 1e-6)
    print(
        json.dumps(
            {
                "ok": True,
                "out_depth": req["out_depth"],
                "out_points": req["out_points"],
                "keep_median_rel_err": float(np.median(rel)) if keep.any() else None,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - report over the protocol
        print(json.dumps({"ok": False, "error": str(e)[:300]}))
        sys.exit(1)
