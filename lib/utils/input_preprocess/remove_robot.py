"""Remove robot arms/hands/grippers from an input image (+ optional depth refine).

Pipeline: SAM3 text-prompted detection of robot parts -> dilate a few px past the
soft mask edge -> LanPaint + Qwen-Image-Edit-2509 masked inpaint. Optionally:

- ``--mask``: extra repaint region unioned with the robot mask (black=edit/repaint,
  white=keep — same convention as the LanPaint edit mask). Resized to the image
  size if needed.
- ``--depth``: same-size metric depth ``.npy`` (meters). The combined repaint hole
  is zeroed out and LingBot-Depth completes it against the repainted RGB, writing
  a refined ``depth.npy`` (+ ``points.npy`` camera-space point map).

Runs under the HOST venv; the isolated SAM3 / lanpaint-qwen / lingbot venvs are
spawned as subprocess workers (see lib/utils/_path.py):

    PYTHONPATH=. .venv/bin/python lib/utils/input_preprocess/remove_robot.py \
        input.png -o out_dir [--mask extra.png] [--depth depth.npy]

Outputs in out_dir: ``image.png`` (repainted), ``depth.npy``/``points.npy`` (if
--depth), plus intermediates (``repaint_mask.png`` white=keep/black=edit,
``hole.png`` RGBA alpha hole, ``sam3/`` per-concept mask stacks, worker logs).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from lib.tools.geometry.agentic_mask import REPO_ROOT, Sam3Server, _dilate1
from lib.tools.geometry.qwen_edit import QwenEditServer
from lib.utils._path import SAM3_PY

LINGBOT_WORKER = os.path.join(REPO_ROOT, "lib", "tools", "geometry", "lingbot_worker.py")

# Simple common noun phrases (SAM3 open-vocab works best with these, not brand names).
# NOTE: "robotic arm/hand" detect where "robot arm/hand/gripper" return NOTHING on the
# same image (YAM lab-rig top view: robotic arm 0.78/0.84, robotic hand 0.65/0.71);
# "robot arm" is kept as a fallback for other viewpoints.
ROBOT_CONCEPTS = ("robotic arm", "robotic hand", "mechanical arm", "robot arm")
DILATE_PX = 12  # match generative_resegment.HOLE_DILATE_PX: a few px past the soft edge
MIN_SCORE = 0.5
# Normalized pinhole placeholder for the LingBot worker; intrinsics only affect the
# auxiliary point-map output, never the refined depth.
DEFAULT_INTRINSICS = ((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0))


def load_repaint_mask(path: str, size: tuple[int, int]) -> np.ndarray:
    """Bool (H, W) repaint mask: black(<128)=edit, white=keep; ``size``=(W, H)."""
    m = Image.open(path)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.asarray(m.convert("L")) < 128


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    for _ in range(px):
        mask = _dilate1(mask)
    return mask


def build_prompt(names: list[str], extra: str = "", surface: str = "") -> str:
    """Removal prompt per the proven generative_resegment template. The Qwen-Edit
    backbone is conditioned on the FULL original image and cannot see the mask (the
    mask only bounds which pixels LanPaint lets change), so removal is driven by
    NAMING every target: an unnamed extra-masked fruit pile is regenerated verbatim
    even when fully masked, and named objects are removed with just a partial mask
    (raw.png fruit-pile test). ``extra`` names the extra-masked content; naming the
    concrete underlying ``surface`` (e.g. "light wooden table") pushes harder toward
    an empty fill than the generic "scene behind them"."""
    parts = list(names) + ([extra] if extra else [])
    if not parts:
        parts = ["objects in the masked areas"]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    fill = f"empty {surface}, completely bare," if surface else "empty scene behind them,"
    return (
        f"Remove the {joined} from the image. The areas they covered must show only "
        f"the {fill} with no trace left. Keep everything else in the image unchanged."
    )


def detect_robot(
    image_path: str,
    out_dir: Path,
    concepts: tuple[str, ...] = ROBOT_CONCEPTS,
    min_score: float = MIN_SCORE,
) -> tuple[np.ndarray, list[str]]:
    """SAM3 union mask over ``concepts`` -> (bool (H, W), concepts that detected)."""
    w, h = Image.open(image_path).size
    union = np.zeros((h, w), bool)
    found: list[str] = []
    sam3 = Sam3Server(SAM3_PY, log_path=str(out_dir / "sam3.log"))
    try:
        for concept in concepts:
            slug = concept.replace(" ", "_")
            masks, scores = sam3.segment_all(
                image_path, concept, str(out_dir / "sam3" / f"{slug}.npy")
            )
            if masks is None:
                continue
            kept = [m for m, s in zip(masks, scores) if s >= min_score]
            if kept:
                found.append(concept)
                for m in kept:
                    union |= m > 0
    finally:
        sam3.close()
    return union, found


def refine_depth(
    image_png: str,
    depth_npy: str,
    hole_png: str,
    out_depth: str,
    out_points: str,
    intrinsics_norm=DEFAULT_INTRINSICS,
    timeout: float = 600.0,
) -> Optional[dict]:
    """One-shot LingBot-Depth completion (scene-dir-free twin of depth_refine.refine_depth)."""
    from lib.utils._path import LINGBOT_PY

    req = {
        "image": image_png,
        "depth": depth_npy,
        "hole": hole_png,
        "intrinsics_norm": [list(r) for r in intrinsics_norm],
        "out_depth": out_depth,
        "out_points": out_points,
    }
    req_path = Path(out_depth).with_suffix(".lingbot_req.json")
    req_path.write_text(json.dumps(req))
    proc = subprocess.run(
        [LINGBOT_PY, LINGBOT_WORKER, str(req_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in reversed(proc.stdout.strip().splitlines() or [""]):
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            continue
        if resp.get("ok"):
            return resp
        print(f"[remove_robot] lingbot error: {resp.get('error')}")
        return None
    print(f"[remove_robot] lingbot no response (rc={proc.returncode}): {proc.stderr[-300:]}")
    return None


def _pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = [line.split(",") for line in out.strip().splitlines()]
    return min(rows, key=lambda r: int(r[1]))[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="input RGB image")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--mask", help="extra repaint mask (black=edit/repaint, white=keep)")
    ap.add_argument(
        "--mask-desc",
        default="",
        help="what the extra mask covers (e.g. 'the pile of plastic toy fruits on the "
        "left'); the edit model cannot see the mask, so UNNAMED masked objects are "
        "regenerated verbatim — always pass this with --mask",
    )
    ap.add_argument("--depth", help="metric depth .npy (H, W) in meters, same size as image")
    ap.add_argument("--concepts", default=",".join(ROBOT_CONCEPTS), help="comma-separated SAM3 prompts")
    ap.add_argument("--dilate", type=int, default=DILATE_PX)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--prompt", help="override the auto-built removal prompt")
    ap.add_argument(
        "--surface",
        default="",
        help="concrete underlying surface for the fill (e.g. 'light wooden table'); "
        "strongly recommended with --mask to avoid regenerating the masked objects",
    )
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", help="CUDA device index for all workers (default: freest)")
    ap.add_argument(
        "--intrinsics",
        help="normalized fx,fy,cx,cy for the LingBot point map (depth output is unaffected)",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or _pick_gpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = str(Path(args.image).resolve())
    img = Image.open(image_path).convert("RGB")

    concepts = tuple(c.strip() for c in args.concepts.split(",") if c.strip())
    robot, found = detect_robot(image_path, out_dir, concepts, args.min_score)
    print(f"[remove_robot] SAM3 detected: {found or 'nothing'}")
    hole = dilate(robot, args.dilate)
    if args.mask:
        hole |= load_repaint_mask(args.mask, img.size)

    out_image = out_dir / "image.png"
    if not hole.any():
        print("[remove_robot] empty repaint region; copying inputs unchanged")
        img.save(out_image)
        if args.depth:
            shutil.copy(args.depth, out_dir / "depth.npy")
        return

    # LanPaint edit mask: white(255)=keep, black(0)=regenerate. Same region saved as
    # an RGBA alpha hole (alpha==0 = repaint) for LingBot depth invalidation.
    edit_mask = out_dir / "repaint_mask.png"
    Image.fromarray(np.where(hole, 0, 255).astype("uint8"), "L").save(edit_mask)
    rgba = np.asarray(img.convert("RGBA")).copy()
    rgba[..., 3] = np.where(hole, 0, 255)
    hole_png = out_dir / "hole.png"
    Image.fromarray(rgba, "RGBA").save(hole_png)

    extra = ""
    if args.mask:
        extra = args.mask_desc or "every other object in the masked areas"
    prompt = args.prompt or build_prompt(found, extra=extra, surface=args.surface)
    print(f"[remove_robot] inpaint prompt: {prompt}")
    qwen = QwenEditServer(log_path=str(out_dir / "qwen.log"))
    try:
        resp = qwen.inpaint(
            image_path, str(edit_mask), prompt, str(out_image),
            guidance=args.guidance, steps=args.steps, seed=args.seed,
        )
    finally:
        qwen.close()
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"qwen inpaint failed: {(resp or {}).get('error')}")
    print(f"[remove_robot] wrote {out_image}")

    if args.depth:
        intr = DEFAULT_INTRINSICS
        if args.intrinsics:
            fx, fy, cx, cy = (float(x) for x in args.intrinsics.split(","))
            intr = ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0))
        resp = refine_depth(
            str(out_image), str(Path(args.depth).resolve()), str(hole_png),
            str(out_dir / "depth.npy"), str(out_dir / "points.npy"), intr,
        )
        if not resp:
            raise RuntimeError("lingbot depth refinement failed")
        print(
            f"[remove_robot] wrote {resp['out_depth']} "
            f"(keep_median_rel_err={resp.get('keep_median_rel_err')})"
        )


if __name__ == "__main__":
    main()
