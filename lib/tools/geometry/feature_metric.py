""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import subprocess
import sys
from typing import Optional

LAMBDA_FEAT = 0.2
# ``orientation_hint`` flags a suspected 180-reversal when the flipped render's
# photo-similarity beats the current one by at least this margin, on the
# masked-white crop (isolated object on white) at the stable pose.
HINT_MARGIN = 0.02  # re-tuned for the position-invariant bbox-stretch crop
VERIFIED_SAME_MARGIN = 0.10
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
