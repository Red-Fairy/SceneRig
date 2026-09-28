"""Render a scene blend from its LOCKED reference (source-view) camera.

One background Blender launch, no scene state of its own — the camera is whatever the
blend already has. Split out of the retired ``register_legacy`` module (2026-07-27): the
one-shot register stage went away with ``--legacy-register``, but this helper is on the
ACTIVE path as the verifier fallback render.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

_REF_RENDER_SCRIPT = r"""
import sys
import bpy
blend, out, engine = sys.argv[-3], sys.argv[-2], sys.argv[-1]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
if sc.camera is None:
    print("NO_CAMERA"); sys.exit(0)
# SAM3D meshes are inward-wound open shells; render them solid (not see-through).
for m in bpy.data.materials:
    m.use_backface_culling = False
try:
    sc.render.engine = engine
except Exception:
    sc.render.engine = "CYCLES"
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("REF_RENDER_OK", out)
"""

REF_RENDER_DEFAULT_ENGINE = "BLENDER_EEVEE_NEXT"


def render_reference_view(
    blend_path: str,
    out_png: str,
    blender_cmd: str,
    engine: str = REF_RENDER_DEFAULT_ENGINE,
) -> bool:
    """Render the scene from its LOCKED reference (source-view) camera to ``out_png``.

    The sole production caller is the verifier fallback
    (``root._render_reference_view_for``), which passes the stage's own engine. The
    EEVEE value is only this standalone helper's signature default. The lighting
    verifier judges exposure, shadow softness, highlight clipping and colour temperature —
    an EEVEE stand-in for a CYCLES stage
    would have it grading a different image than the one being delivered.

    Falls back to CYCLES if Blender rejects the engine name. Best-effort throughout: a
    render failure must not abort the pipeline."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_REF_RENDER_SCRIPT)
        script = f.name
    try:
        subprocess.run(
            [
                blender_cmd,
                "--background",
                "--python",
                script,
                "--",
                blend_path,
                out_png,
                engine,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(out_png)
    finally:
        os.unlink(script)
