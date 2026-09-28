"""DINOv2 patch-similarity metric for pose registration (rotation/facing term).

Silhouette IoU is blind to a ~180-degree yaw on near-symmetric objects (spoons,
forks): the outline barely changes but the visible face does. ``patch_sim``
scores APPEARANCE agreement between two same-size crops (render vs photo) as the
mean per-patch cosine similarity of DINOv2 patch tokens, optionally restricted to
the patches covered by a mask.

The model runs in a persistent SPAWNED WORKER SUBPROCESS, not in-process:
importing torch after an in-process CoACD decomposition HANGS the process
(OpenMP runtime clash, reproduced 2026-07-14 — same family as the known
scipy-after-CoACD segfault), and this module is imported inside the composition
MCP server, which does run CoACD. The worker speaks JSON-lines over its OWN
stdin/stdout pipes; the parent's stdout (the MCP JSON-RPC transport) is never
written to — all diagnostics go to stderr.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import subprocess
import sys
from typing import Optional

# Weight of the DINO term added to ROTATION candidates' selection score in
# ``PoseSession.optimize_axis``. The current value is 0.2; set it to 0 for IoU-only
# selection. Selection also falls back to IoU-only when the DINO worker is unavailable.
# Tuned on the
# 0714_flipfirst_abc1 / 0714_e2e_abc1 offline replay (the replay script is gone;
# re-tuning today would use the recorded benchmark register.json traces): at
# 0.2 no known-correct object's winning candidate changes (true for the whole
# {0.05..0.4} sweep) while near-tied rotations get real orientation pressure.
# NOTE: no sweep value can outweigh a full 180-reversal's IoU gap (~0.15 on the
# flipfirst spoon; would need lambda ~3) — that correction belongs to the
# orientation HINT + rotate_180 path, not this term.
LAMBDA_FEAT = 0.2
# ``orientation_hint`` flags a suspected 180-reversal when the flipped render's
# photo-similarity beats the current one by at least this margin, on the
# masked-white crop (isolated object on white) at the stable pose.
HINT_MARGIN = 0.02  # re-tuned for the position-invariant bbox-stretch crop
# (_orient_object_crop; 2026-07-20). Across abc1/abc2/abc3/gpt1 the true reversals
# separate at >= +0.026 (spoon +0.31, fork +0.21, knife +0.10, notebook +0.040,
# book#0 +0.11, book#1 +0.026) from the clear keeps <= +0.003 (round plate/donut,
# cup/saucer, mics/sensors, mug/clock/pen); 0.02 sits in the gap and keeps book#1.
# The bbox-stretch normalises round objects to ~0, so the donut/plate false
# positives the earlier square-pad crop produced are gone (near-symmetric croissants
# can still flag ~+0.05, but a 180-yaw of a near-symmetric object is a no-op).
# The inverse-direction margin: ``sim0 - sim180`` at or above this reports the
# orientation as "SAME (verified)" in the investigate hints (exec.py) — strong
# evidence AGAINST flipping. Scores between the two margins are neither flagged
# nor verified. Same crop/score scale as HINT_MARGIN: retune the two together.
VERIFIED_SAME_MARGIN = 0.10
# The VERIFIED band of the reversal flag: ``sim180 - sim0`` at or above this
# reports "REVERSED (verified)" in the investigate hints (exec.py) and lets the
# reversal own the whole hint chain; flagged margins in [HINT_MARGIN, this) are
# surfaced as "REVERSED (suspected)" — judge-the-crops-first, no chain
# suppression. Deliberately NOT symmetric with VERIFIED_SAME_MARGIN (0.10): on
# the 0821 benchmark hints.jsonl both confirmed-TRUE flips sit under 0.10 —
# robolab_breakfast_table milk carton#0 at +0.054 and robolab_food_packing_dense
# bin#0 at +0.065 — while the one confirmed FALSE positive (misc_online6 book#0,
# flip applied then undone at IoU 0.28->0.09) sits at +0.023. 0.05 is the
# largest round bar that keeps both true positives verified and demotes the
# false positive to "suspected". Low-margin TRUE reversals exist (down to
# +0.026, see the HINT_MARGIN note) — exactly why "suspected" still surfaces
# them. Same crop/score scale as the two margins above: retune all three
# together.
REVERSED_VERIFIED_MARGIN = 0.05

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_worker: Optional[subprocess.Popen] = None
_ok: Optional[bool] = None  # None = not tried yet


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def available() -> bool:
    """Spawn + warm the worker on first call; False (cached) on any failure so
    callers degrade to IoU-only scoring."""
    global _worker, _ok
    if _ok is not None:
        return _ok
    try:
        _worker = subprocess.Popen(
            [sys.executable, "-u", "-m", "lib.tools.geometry.feature_metric"],
            cwd=_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        atexit.register(_shutdown)
        line = _worker.stdout.readline()
        _ok = bool(line) and json.loads(line).get("ready", False)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"[feature-metric] worker unavailable: {exc}", file=sys.stderr)
        _ok = False
    if not _ok:
        _shutdown()
    return _ok


warmup = available  # alias: PoseSession warms the worker at construction time


def worker_dead() -> bool:
    """True once the worker failed to spawn or died mid-session. Pure read of the
    cached state — no spawn side effect — so hint emitters can distinguish
    "backend offline" (infrastructure failure) from "no verdict" (ambiguity)."""
    return _ok is False


def _shutdown() -> None:
    global _worker
    if _worker is not None:
        try:
            _worker.kill()
        except Exception:  # noqa: BLE001
            pass
        _worker = None


def patch_sim(img_a, img_b, mask=None) -> float:
    """Mean per-patch DINOv2 cosine similarity of two PIL images (same size),
    over the patches covered by ``mask`` (PIL/np bool, same size; None = all).
    Raises RuntimeError when the worker is unavailable/dead."""
    global _ok
    if not available():
        raise RuntimeError("DINO feature worker unavailable")
    from PIL import Image

    if mask is not None and not isinstance(mask, Image.Image):
        import numpy as np

        mask = Image.fromarray((np.asarray(mask) > 0).astype("uint8") * 255)
    req = {
        "a": _png_b64(img_a.convert("RGB")),
        "b": _png_b64(img_b.convert("RGB")),
        "mask": _png_b64(mask.convert("L")) if mask is not None else None,
    }
    _worker.stdin.write(json.dumps(req) + "\n")
    _worker.stdin.flush()
    line = _worker.stdout.readline()
    if not line:
        _ok = False
        _shutdown()
        raise RuntimeError("DINO feature worker died")
    resp = json.loads(line)
    if "error" in resp:
        raise RuntimeError(f"DINO feature worker error: {resp['error']}")
    return float(resp["sim"])


# --------------------------------------------------------------------------- #
# Worker process                                                              #
# --------------------------------------------------------------------------- #
def _worker_main() -> None:  # pragma: no cover - subprocess entry
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):  # hub load prints; keep pipe clean
        import numpy as np
        import torch
        from PIL import Image

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def decode(b64: str) -> "Image.Image":
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    def tokens(img, tw: int, th: int):
        x = np.asarray(img.convert("RGB").resize((tw, th)), dtype=np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
        t = (t - mean) / std
        with torch.no_grad():
            tok = model.forward_features(t)["x_norm_patchtokens"][0]
        return torch.nn.functional.normalize(tok, dim=-1)

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            a, b = decode(req["a"]), decode(req["b"])
            w, h = a.size
            s = 518.0 / max(w, h)  # long side ~518, both dims multiples of 14
            tw, th = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
            cos = (tokens(a, tw, th) * tokens(b, tw, th)).sum(-1)  # (n_patches,)
            hp, wp = th // 14, tw // 14
            sel = None
            if req.get("mask"):
                m = np.asarray(decode(req["mask"]).convert("L").resize((wp, hp)))
                sel = torch.from_numpy(m > 64).to(dev).reshape(-1)
                if not bool(sel.any()):
                    sel = None
            sim = float((cos[sel] if sel is not None else cos).mean())
            out = {"sim": sim}
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            out = {"error": str(exc)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    _worker_main()
