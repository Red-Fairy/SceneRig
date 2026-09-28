#!/bin/bash
# prewarm_cycles_kernels.sh <gpu_index>: load (JIT-compile if absent) Blender's Cycles CUDA
# render + denoising kernels for this GPU/driver/Blender BEFORE any stage timer runs.
#
# Why (2026-09-15): Blender 4.5 ships no precompiled CUDA kernel for the H100, so the
# driver JIT-compiles a 51 MB kernel the first time a process needs it — 371-390 s
# measured. The result lives in the driver cache (CUDA_CACHE_PATH, shared on /fsx), but
# that cache sat at 383 MB against the driver's default cap and other JIT users (PhysX)
# evicted the kernel during the day, so whichever lane rendered next would have paid the
# 6.5 min inside the agent stage — eight cold lanes all at once. run_e2e.sh raises the cap
# and calls this once per lane: ~2 s when warm; the flock makes cold lanes compile once and
# the rest wait. Always exits 0 (a failed prewarm degrades to the old in-stage compile).
GPU="${1:-0}"
GRASE="$(cd "$(dirname "$0")/.." && pwd)"
BLENDER="${GRASE_BLENDER_COMMAND:-$GRASE/lib/utils/third_party/blender-4.5/blender}"
CACHE="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
mkdir -p "$CACHE" 2>/dev/null || true
note() { echo "[prewarm_cycles] $*" >&2; }
[ -x "$BLENDER" ] || { note "blender not found at $BLENDER; skipping"; exit 0; }
PROBE="$(mktemp -d)/probe.py"
cat > "$PROBE" <<'PY'
import bpy, time
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type in ("CUDA", "OPTIX")
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"
sc.cycles.samples = 1
sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = 16
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = bpy.app.tempdir + "/grase_cycles_prewarm.png"
t = time.time()
bpy.ops.render.render(write_still=True)
print("PREWARM_CYCLES_OK %.1f" % (time.time() - t))
PY
t0=$(date +%s)
(
  flock -w 900 9 || note "lock wait exceeded; compiling in parallel"
  CUDA_VISIBLE_DEVICES="$GPU" timeout 900 "$BLENDER" -b --factory-startup --python "$PROBE" 2>&1 \
    | grep -E "PREWARM_CYCLES_OK|Error|Loading .* kernels" | tail -3 | sed 's/^/[prewarm_cycles] /' >&2 || true
) 9>"$CACHE/.grase_cycles_prewarm.lock"
note "done in $(( $(date +%s) - t0 ))s on GPU $GPU (cache $CACHE)"
rm -rf "$(dirname "$PROBE")"
exit 0
