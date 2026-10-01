"""Persistent SAM3 text-prompted segmentation server.

Loads the SAM3 model once, then answers many segmentation requests over a
newline-delimited JSON protocol on stdin/stdout. The image backbone is the
expensive part, so the encoded image `state` is cached per (path, mtime, size) and only
the (cheap) text prompt + grounding is re-run per object — making N objects on
one image far faster than spawning sam3_worker N times.

Protocol (one JSON object per line):
  request  -> {"image": <path>, "object": <prompt>, "out": <mask.npy path>}
             {"image": <path>, "points": [[u,v],...], "labels": [...],
              "all_candidates": <bool>, "out": <mask.npy path>}
             {"cmd": "ping"} | {"cmd": "shutdown"}
  response <- {"ok": true, "out": <path>, "n": N, "scores": [...]} (text masks)
             {"ok": true, "out": <path>, "n": N, "iou": F,
              "ious": [...]}                                    (point masks)
             {"ok": false, "reason": "no_detection"|"error", "error": <msg>}
             {"ready": true}                                     (after model load)

Runs in the dependency-isolated SAM3 venv (see lib/utils/_path.py).
Everything except the protocol is routed to stderr so stdout stays pure JSON.
"""

import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

# Route any library/model chatter to stderr; keep a private real stdout for the
# JSON protocol so it is never polluted.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3"))

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


def _respond(obj):
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def _masks_to_array(masks):
    """SAM3 masks (N,1,H,W)/(N,H,W)/(H,W) -> uint8 0/255 stack (N,H,W).

    Cast to float first: SAM3 runs under bf16 autocast and numpy can't convert
    bfloat16 ('Got unsupported ScalarType BFloat16').
    """
    if hasattr(masks, "cpu"):
        arr = masks.float().cpu().numpy()
    else:
        arr = np.asarray(masks, dtype=np.float32)
    if arr.ndim == 2:  # (H,W) single
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[1] == 1:  # (N,1,H,W)
        arr = arr[:, 0]
    return (arr > 0.5).astype("uint8") * 255


def _save_first_mask(masks, out_path):
    arr = _masks_to_array(masks)[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)


def _save_all_masks(masks, scores, out_path):
    """Save every instance mask as a (N,H,W) stack; return (n, scores list)."""
    arr = _masks_to_array(masks)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    sc = []
    if scores is not None:
        sc = [
            float(s)
            for s in (
                scores.float().cpu().numpy() if hasattr(scores, "cpu") else scores
            )
        ]
    return arr.shape[0], sc


@torch.no_grad()
def _inject_shared_features(predictor, state):
    """Wire the SHARED-backbone image features into the interactive predictor.

    SAM3 shares ONE vision backbone between the concept detector and the tracker
    (the checkpoint has no ``tracker.backbone`` weights), so the interactive
    predictor's own ``model.backbone`` is None and its ``set_image()`` crashes in
    ``forward_image``. ``Sam3Processor.set_image`` already ran the shared backbone and
    stashed ``sam2_backbone_out`` (with conv_s0/conv_s1 applied) in ``state``; that is
    exactly what the predictor's ``forward_image`` would have returned. So we replicate
    the tail of the predictor's ``set_image`` using those precomputed features instead
    of recomputing them through the (missing) backbone."""
    predictor.reset_predictor()
    predictor._orig_hw = [(state["original_height"], state["original_width"])]
    backbone_out = state["backbone_out"]["sam2_backbone_out"]
    _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
    # no_mem_embed is added to the lowest-res feature map (same as set_image does).
    vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *fs)
        for feat, fs in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
    ][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._is_image_set = True


def _segment_points(predictor, state, points, labels, out_path, all_candidates=False):
    ""
    w, h = state["original_width"], state["original_height"]
    coords = np.array(
        [[float(u) * w, float(v) * h] for u, v in points], dtype=np.float32
    )
    labs = np.array(labels, dtype=np.int32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _inject_shared_features(predictor, state)
        masks, ious, _ = predictor.predict(
            point_coords=coords,
            point_labels=labs,
            multimask_output=(len(points) == 1),
            normalize_coords=True,
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if all_candidates:
        raw = np.asarray(
            masks.float().cpu().numpy() if hasattr(masks, "cpu") else masks,
            dtype=np.float32,
        )
        if raw.ndim == 2:
            raw = raw[None]
        if raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0]
        np.save(out_path, (raw > 0).astype("uint8") * 255)  # same >0 rule as argmax path
        return [float(x) for x in np.asarray(ious, dtype=np.float32).reshape(-1)]
    bi = int(np.argmax(ious))
    arr = (np.asarray(masks[bi]) > 0).astype("uint8") * 255
    np.save(out_path, arr)
    return float(ious[bi])


def sam3_checkpoint():
    """Local weight file, then the Hugging Face download inside ``build_sam3_image_model``."""
    explicit = os.environ.get("SAM3_CHECKPOINT")
    if explicit:
        return explicit
    default = os.path.join(ROOT, "utils", "third_party", "sam3", "checkpoints", "sam3.pt")
    if os.path.isfile(default) and os.path.getsize(default) > 0:
        return default
    return None


def main():
    # enable_inst_interactivity exposes the SAM2-style point predictor alongside the
    # concept/text detector (they share the backbone).
    model = build_sam3_image_model(
        enable_inst_interactivity=True,
        checkpoint_path=sam3_checkpoint(),
    )
    proc = Sam3Processor(model)
    predictor = model.inst_interactive_predictor
    torch.set_grad_enabled(False)  # F8: inference-only; no autograd graph on cached states
    image_states: "OrderedDict[tuple, object]" = OrderedDict()
    STATE_CACHE_MAX = 8

    def _state_key(path):
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)

    def _store_state(key, state):
        image_states[key] = state
        image_states.move_to_end(key)
        while len(image_states) > STATE_CACHE_MAX:
            image_states.popitem(last=False)  # evict LRU -> frees its GPU features

    _respond({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "reason": "error", "error": "bad json"})
            continue
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd == "ping":
            _respond({"ok": True, "pong": True})
            continue
        try:
            img_path = req["image"]
            out_path = req["out"]
            points = req.get("points")  # interactive point seg: [[u,v],...] normalized
            if points is not None:
                labels = req.get("labels") or [1] * len(points)
                # Run the SHARED backbone once via the processor (the interactive
                # predictor has no backbone of its own), then point-segment.
                key = _state_key(img_path)
                state = image_states.get(key)
                if state is not None:
                    image_states.move_to_end(key)  # mark MRU
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        state = proc.set_image(Image.open(img_path).convert("RGB"))
                    _store_state(key, state)
                if req.get("all_candidates"):
                    ious = _segment_points(
                        predictor, state, points, labels, out_path,
                        all_candidates=True,
                    )
                    _respond({
                        "ok": True, "out": out_path, "n": len(ious),
                        "ious": ious, "iou": (max(ious) if ious else 0.0),
                    })
                    continue
                iou = _segment_points(predictor, state, points, labels, out_path)
                _respond({"ok": True, "out": out_path, "n": 1, "iou": iou})
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key = _state_key(img_path)
                state = image_states.get(key)
                if state is not None:
                    image_states.move_to_end(key)  # mark MRU
                else:
                    img = Image.open(img_path).convert("RGB")
                    state = proc.set_image(img)
                    _store_state(key, state)
                out = proc.set_text_prompt(state=state, prompt=req["object"])
            masks = out["masks"]
            if masks is None or len(masks) == 0:
                _respond({"ok": False, "reason": "no_detection"})
                continue
            if req.get("all_instances", True):
                # Text prompt: return EVERY instance mask + scores (stacked).
                n, scores = _save_all_masks(masks, out.get("scores"), out_path)
                _respond({"ok": True, "out": out_path, "n": n, "scores": scores})
            else:
                _save_first_mask(masks, out_path)
                _respond({"ok": True, "out": out_path, "n": 1})
        except Exception as exc:  # noqa: BLE001 - never kill the server on one bad req
            _respond({"ok": False, "reason": "error", "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
