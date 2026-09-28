"""Persistent MolmoPoint-8B pointing server.

Loads MolmoPoint-8B once, then answers many "point to <object>" requests over a
newline-delimited JSON protocol on stdin/stdout. For each request it returns one
2D point per detected instance (so "the rocks" -> several points), in BOTH pixel
and normalized [0,1] coordinates of the original image.

Runs in the dependency-isolated Molmo venv (transformers==4.57.1); see
lib/utils/_path.py. Everything except the JSON protocol goes to stderr so stdout
stays pure JSON.

Protocol (one JSON object per line):
  request  -> {"image": <path>, "object": <name>}
             {"cmd": "ping"} | {"cmd": "shutdown"}
  response <- {"ok": true, "points": [[u, v], ...], "points_px": [[x, y], ...],
               "width": W, "height": H}
             {"ok": false, "error": <msg>}
             {"ready": true}                                  (after model load)
"""

import json
import os
import sys

import torch
from PIL import Image

_real_stdout = sys.stdout
sys.stdout = sys.stderr  # keep stdout pure JSON; model/library chatter -> stderr

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

MODEL = os.environ.get("MOLMO_MODEL", "allenai/MolmoPoint-8B")


def _respond(obj):
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def main():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, trust_remote_code=True, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, trust_remote_code=True, padding_side="left"
    )
    # Keyed by (path, mtime_ns, size), not path alone — same stale-cache landmine as
    # sam3_server's state cache (fixed 2026-07-30): a fixed per-instance path
    # rewritten with new pixels must never be answered from the old image.
    image_cache: dict[tuple, tuple] = {}  # (path, mtime_ns, size) -> (PIL, W, H)
    _respond({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "error": "bad json"})
            continue
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd == "ping":
            _respond({"ok": True, "pong": True})
            continue
        try:
            img_path = req["image"]
            obj = req["object"]
            st = os.stat(img_path)
            key = (img_path, st.st_mtime_ns, st.st_size)
            cached = image_cache.get(key)
            if cached is None:
                im = Image.open(img_path).convert("RGB")
                cached = (im, im.size[0], im.size[1])
                image_cache[key] = cached
            im, w, h = cached

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Point to {obj}"},
                        {"type": "image", "image": im},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                return_pointing_metadata=True,
            )
            meta = inputs.pop("metadata")
            dev = {
                k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(
                    **dev,
                    logits_processor=model.build_logit_processor_from_inputs(dev),
                    max_new_tokens=200,
                )
            text = processor.tokenizer.decode(
                out[0, dev["input_ids"].shape[1] :], skip_special_tokens=False
            )
            pts = model.extract_image_points(
                text,
                meta["token_pooling"],
                meta["subpatch_mapping"],
                meta["image_sizes"],
            )
            points_px = [[float(x), float(y)] for _, _, x, y in pts]
            points = [[x / w, y / h] for x, y in points_px]
            _respond(
                {
                    "ok": True,
                    "points": points,
                    "points_px": points_px,
                    "width": w,
                    "height": h,
                }
            )
        except Exception as exc:  # noqa: BLE001 - never kill the server on one request
            _respond({"ok": False, "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
