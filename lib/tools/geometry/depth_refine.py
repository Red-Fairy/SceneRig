"""Host-side wrapper for LingBot-Depth refinement (isolated env, one-shot worker).

After a masked occluder removal, the edit hole's original MoGE depth belongs to the
REMOVED objects; the redetected target's newly revealed parts (e.g. a monitor's stand)
need depth for placement/ICP. ``refine_depth`` runs ``lingbot_worker.py`` under
``LINGBOT_PY``: edited RGB + hole-invalidated MoGE depth + normalized intrinsics ->
completed metric depth + camera-space point map (same MoGE camera convention, so
``moge_points_to_world`` applies unchanged).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
WORKER = os.path.join(REPO_ROOT, "lib", "tools", "geometry", "lingbot_worker.py")


def refine_depth(
    scene_dir: str,
    edited_image: str,
    hole_mask: str,
    tag: str,
    timeout: float = 600.0,
) -> Optional[dict]:
    """Refined depth/points for one edited image -> ``masks/edited/<tag>_{depth,points}.npy``.

    Returns the worker's response dict (with ``out_points``) or None on failure."""
    from lib.utils._path import LINGBOT_PY

    scene = Path(scene_dir)
    moge = json.load(open(scene / "moge" / "moge.json"))
    out_dir = scene / "masks" / "edited"
    req = {
        "image": edited_image,
        "depth": str(scene / "moge" / "depth.npy"),
        "hole": hole_mask,
        "intrinsics_norm": moge["intrinsics_norm"],
        "out_depth": str(out_dir / f"{tag}_depth.npy"),
        "out_points": str(out_dir / f"{tag}_points.npy"),
    }
    req_path = out_dir / f"{tag}_lingbot_req.json"
    with open(req_path, "w") as f:
        json.dump(req, f)
    proc = subprocess.run(
        [LINGBOT_PY, WORKER, str(req_path)],
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
        print(f"[depth_refine] worker error: {resp.get('error')}")
        return None
    print(f"[depth_refine] no response (rc={proc.returncode}): {proc.stderr[-300:]}")
    return None
