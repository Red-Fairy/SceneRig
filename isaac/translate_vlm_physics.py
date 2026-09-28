"""Translate the preprocess VLM physics estimate into the isaac-side contract.

    <repo>/.venv/bin/python isaac/translate_vlm_physics.py <exp_dir>

scene/physics/physics_vlm.json (pipeline names, estimation-time size anchors) ->
scene/isaac/physics_vlm.json (USD prim names, mass rescaled to the export
geometry). Run with the repo venv (trimesh for the OBB extents). No VLM calls.
Prints VLM_TRANSLATE_OK objects=<n>; n=0 means nothing to translate (callers
fall back to heuristic defaults).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.tools.geometry.physics_estimate import translate_to_isaac  # noqa: E402

if __name__ == "__main__":
    exp = Path(sys.argv[1])
    exp = exp if exp.is_absolute() else REPO / exp
    n = translate_to_isaac(exp)
    print(f"VLM_TRANSLATE_OK objects={n}")
