"""Complete a novel-view render with GPT-Image-2 (OpenAI image edit / inpaint).

Takes the splatted render from ``render_pointcloud`` and its hole mask (RGBA,
transparent over disocclusions) and asks GPT-Image-2 to fill the transparent
region so the visible content, perspective and lighting stay consistent. The
OpenAI image-edit endpoint is asked to regenerate fully-transparent mask pixels and
keep the opaque ones. In practice the keep region can still drift (see the
``POLISH_PROMPT`` measurement below).

Usable as a library (``complete``) and a CLI::

    python -m lib.tools.geometry.gpt_inpaint \
        --render <dir>/render.png --mask <dir>/hole_mask.png --out <dir>/completed.png
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from lib.utils.common import build_client

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_PROMPT = (
    "Fill in the transparent regions of this photo to produce a complete, "
    "photorealistic image that is geometrically and stylistically consistent "
    "with the visible content, perspective, materials and lighting. Preserve the "
    "identity, geometry, positions, composition and appearance of already-visible "
    "content; limit changes to the transparent regions."
)


POLISH_PROMPT = (
    "This photo has hazy semi-transparent smears near object edges and some "
    "black empty regions. Produce the scene as if captured "
    "cleanly: fill the black regions and replace the smears with sharp, "
    "consistent scene content. Preserve the identity, geometry, positions, "
    "composition and appearance of already-clear content; limit changes to the "
    "black regions, smears and visible rendering artifacts."
)


def save_b64_png(b64: str, out_path: str) -> str:
    """Decode a base64 PNG payload to ``out_path``."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return out_path


def complete(
    render_path: str,
    mask_path: str,
    out_path: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    size: str = "auto",
) -> str:
    """Inpaint the masked holes of ``render_path`` with GPT-Image-2; write ``out_path``.

    ``mask_path`` is an RGBA PNG whose transparent pixels mark the region to
    regenerate (as produced by ``render_pointcloud.hole_mask_rgba``).
    """
    client = build_client(model)
    with open(render_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        resp = client.images.edit(
            model=model,
            image=img_f,
            mask=mask_f,
            prompt=prompt,
            size=size,
        )
    return save_b64_png(resp.data[0].b64_json, out_path)


def polish(
    image_path: str,
    out_path: str,
    prompt: str = POLISH_PROMPT,
    model: str = DEFAULT_MODEL,
    size: str = "auto",
) -> str:
    """Maskless GPT-Image edit of ``image_path`` (SHARP raw render); write ``out_path``."""
    client = build_client(model)
    with open(image_path, "rb") as img_f:
        resp = client.images.edit(model=model, image=img_f, prompt=prompt, size=size)
    usage = getattr(resp, "usage", None)
    if usage is not None:  # exact per-call token cost, for the wiring cost ledger
        print(f"[gpt_inpaint] polish usage: {usage}", flush=True)
    return save_b64_png(resp.data[0].b64_json, out_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Complete a novel-view render with GPT-Image-2"
    )
    p.add_argument("--render", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--size", default="auto")
    args = p.parse_args()
    out = complete(args.render, args.mask, args.out, args.prompt, args.model, args.size)
    print(out)


if __name__ == "__main__":
    main()
