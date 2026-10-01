"""SAM3 Worker for Text-Prompted Image Segmentation.

This script uses SAM3 (Segment Anything Model 3) with text prompts to segment
specific objects from images based on natural language descriptions.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3"))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def sam3_checkpoint() -> str | None:
    explicit = os.environ.get("SAM3_CHECKPOINT")
    if explicit:
        return explicit
    default = os.path.join(ROOT, "utils", "third_party", "sam3", "checkpoints", "sam3.pt")
    if os.path.isfile(default) and os.path.getsize(default) > 0:
        return default
    return None


def main() -> None:
    """Run SAM3 segmentation on an image with a text prompt.

    Loads an image, uses SAM3 to segment the object specified by the text
    prompt, and saves the resulting binary mask as a numpy array.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument(
        "--object", required=True, help="Text description of object to segment"
    )
    p.add_argument("--out", required=True, help="Path for output mask npy file")
    args = p.parse_args()

    model = build_sam3_image_model(checkpoint_path=sam3_checkpoint())
    proc = Sam3Processor(model)
    img = Image.open(args.image).convert("RGB")
    # SAM3 weights are meant to run under bf16 autocast (its detector/predictor
    # paths wrap forward in torch.autocast); the image processor does not, so
    # wrap here — otherwise the ViTDet backbone hits a bf16/float32 mat dtype mismatch.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = proc.set_image(img)
        out = proc.set_text_prompt(state=state, prompt=args.object)
    masks = out["masks"]
    if masks is None or len(masks) == 0:
        # SAM3 is open-vocabulary; an unusual/branded/over-specific prompt can
        # match nothing. Fail with a clear message instead of an IndexError so
        # the bridge/agent can fall back (e.g. retry with a simpler noun phrase).
        sys.stderr.write(
            f"SAM3 found no object matching prompt {args.object!r} in the image. "
            "Use a simple, common noun phrase (e.g. 'robot arm', not a brand/model name).\n"
        )
        sys.exit(2)
    mask = masks[0]

    # Convert mask to numpy array if needed
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    elif hasattr(mask, "numpy"):
        mask = mask.numpy()

    # Remove batch dimension if present
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # Binarize and scale to 0-255
    mask = (mask > 0.5).astype("uint8") * 255
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, mask)


if __name__ == "__main__":
    main()
