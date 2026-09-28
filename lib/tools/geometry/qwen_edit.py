"""Host-side client for the persistent LanPaint + Qwen-Image-Edit-2509 worker.

Spawns ``lanpaint_worker.py`` under ``LANPAINT_QWEN_PY`` (isolated venv), loads Qwen
ONCE, and serves many edits. Two ops:
  - ``inpaint(image, mask, prompt, out)``  masked removal/fill (mask: white=keep, black=edit)
  - ``outpaint(image, pad, prompt, out)``  native canvas extension (pad e.g. "l192")

Mirrors the Sam3Server lifecycle (create once per preprocess pass, ``close()`` at the end).
"""

from __future__ import annotations

from typing import Optional

from lib.tools.geometry.agentic_mask import _JsonServer
from lib.utils._path import LANPAINT_QWEN_PY

_WORKER = "lib/tools/geometry/lanpaint_worker.py"


class QwenEditServer(_JsonServer):
    """Persistent LanPaint+Qwen inpaint/outpaint server."""

    def __init__(self, log_path: Optional[str] = None, ready_timeout: float = 600.0):
        super().__init__(LANPAINT_QWEN_PY, _WORKER, log_path, ready_timeout)

    def inpaint(
        self, image: str, mask: str, prompt: str, out: str,
        neg: str = "", guidance: float = 4.0, steps: int = 20, seed: int = 0,
    ) -> Optional[dict]:
        return self._rpc({
            "op": "inpaint", "image": image, "mask": mask, "prompt": prompt,
            "out": out, "neg": neg, "guidance": guidance, "steps": steps, "seed": seed,
        })

    def outpaint(
        self, image: str, pad: str, prompt: str, out: str,
        neg: str = "", guidance: float = 4.0, steps: int = 20, seed: int = 0,
    ) -> Optional[dict]:
        return self._rpc({
            "op": "outpaint", "image": image, "pad": pad, "prompt": prompt,
            "out": out, "neg": neg, "guidance": guidance, "steps": steps, "seed": seed,
        })
