#!/bin/bash
# Two-tier HF weight cache: durable source on /fsx (Lustre), hot tier on the
# node-local NVMe. Mirrors /fsx -> NVMe (idempotent rsync, flock-serialized
# across lanes) and PRINTS ON STDOUT the HF_HOME the caller should use — the
# NVMe mirror on success, the /fsx source on any skip or failure. All
# diagnostics go to stderr. Always exits 0: a broken prewarm must degrade to a
# slow (cold-Lustre) boot, never a dead run.
#
# Why (audits/RUNTIME_PERF_AUDIT_2026_07_28.md + CHANGELOG 2026-07-29): the 0727
# fix moved HF_HOME from the wipeable NVMe to /fsx, which ended the ~100 GB
# cold-redownload incidents but made every batch's first wave pay concurrent
# cold-Lustre weight reads — Molmo 8-shard load 6 s warm vs 250-370 s cold,
# SAM3D boot 66 -> 270 s, ~495 s wait on the Qwen-Image-Edit worker (the 0728
# resegment blow-up). The two-tier scheme keeps both properties: a pod-restart
# wipe costs one re-mirror (~95 GB, minutes, once per node), and warm boots read
# NVMe. Staggered-lane loads of 46-89 s prove warm reads are all we need.
#
# Contract with callers (run_e2e.sh):
#     HF_HOME="$(bash scripts/prewarm_models.sh || true)"
#     export HF_HOME="${HF_HOME:-/fsx/rundongluo/.cache/huggingface}"
# Callers keep owning `unset HUGGINGFACE_HUB_CACHE` (it overrides HF_HOME) and
# the HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE exports, as before.
#
# Behaviors:
#   - HF_HOME already set       -> respected verbatim, no mirror (explicit override).
#   - HF_HUB_OFFLINE=0 (fetch)  -> prints the /fsx source: downloads MUST land on
#     the durable store, never the wipeable NVMe — pointing a fetch run at the
#     NVMe is exactly how the 0727 incident happened. The next offline launch
#     mirrors the new model down.
#   - NVMe root missing or not a mountpoint -> prints the /fsx source. The
#     mountpoint check only applies to the DEFAULT root; an explicit
#     GRASE_HF_NVME_ROOT is trusted (test hook, other pod layouts).
#   - Concurrent callers (8 lanes, overlapping batches) serialize on a
#     node-local flock: the first mirrors, the rest wait then no-op. Lock wait
#     capped at 1200 s; on timeout the caller falls back to /fsx.
#   - Cross-batch note: `--delete` keeps the mirror exact. If a model is
#     removed/updated on /fsx while an older batch is mid-run, that batch's
#     already-open files stay readable (ext4 unlink semantics); a not-yet-opened
#     model would fail like any cache invalidation. Accepted as rare.
#   - Separately backgrounds a page-cache warm of the /fsx python envs (sam3d
#     micromamba env + third_party venvs): their cold .so imports cost ~137 s at
#     0729 and the NVMe HF cache cannot help them — venvs are path-dependent, so
#     they are page-cache-warmed in place, never copied (seig-rename incident).
set -uo pipefail

SRC="${GRASE_HF_SRC:-/fsx/rundongluo/.cache/huggingface}"
NVME_ROOT_DEFAULT="/mnt/localdisk/scratch_space"
NVME_ROOT="${GRASE_HF_NVME_ROOT:-$NVME_ROOT_DEFAULT}"
DST="${GRASE_HF_NVME:-$NVME_ROOT/hf-cache-grase/huggingface}"
GRASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

note() { echo "[prewarm_models] $*" >&2; }

# ---- background page-cache warm of the /fsx python envs (node-scoped, one
# warmer per node via non-blocking flock; stdout must stay closed or a caller's
# $(...) capture would block on us). GRASE_SKIP_ENV_WARM=1 disables (tests). ----
warm_envs() {
  local d
  for d in \
    /fsx/rundongluo/.local/micromamba/envs/sam3d-objects \
    "$GRASE/lib/utils/third_party/molmo/.venv" \
    "$GRASE/lib/utils/third_party/sam3/.venv" \
    "$GRASE/lib/utils/third_party/lingbot-depth/.venv" \
    "$GRASE/lib/utils/third_party/lanpaint-qwen/.venv" \
    "$GRASE/lib/utils/third_party/sharp/.venv" \
    "$GRASE/lib/utils/third_party/moge/.venv" \
    "$GRASE/lib/utils/third_party/sam3d" \
    "$GRASE/.venv"; do
    [ -d "$d" ] || continue
    find "$d" -type f \( -name '*.so*' -o -name '*.pyc' -o -name '*.pt' \
      -o -name '*.pth' -o -name '*.safetensors' -o -name '*.ckpt' -o -name '*.bin' \) \
      -print0 2>/dev/null | xargs -0 -r cat > /dev/null 2>&1
  done
}
if [ "${GRASE_SKIP_ENV_WARM:-0}" != "1" ]; then
  WARM_LOCK_DIR="$NVME_ROOT"
  { [ -d "$WARM_LOCK_DIR" ] && [ -w "$WARM_LOCK_DIR" ]; } || WARM_LOCK_DIR="/tmp"
  ( flock -n 8 || exit 0; warm_envs ) 8>"$WARM_LOCK_DIR/.grase_env_warm.lock" \
    > /dev/null 2>&1 &
fi

# ---- pick the HF_HOME to print ----
if [ -n "${HF_HOME:-}" ]; then
  note "HF_HOME preset -> $HF_HOME (no mirror)"
  echo "$HF_HOME"; exit 0
fi

if [ "${HF_HUB_OFFLINE:-1}" = "0" ]; then
  note "HF_HUB_OFFLINE=0 fetch run -> $SRC (downloads must land on /fsx)"
  echo "$SRC"; exit 0
fi

if [ ! -d "$SRC" ]; then
  note "source $SRC missing -> printing it anyway (nothing to mirror)"
  echo "$SRC"; exit 0
fi

# Guard against writing 95 GB to the container root fs when the ephemeral
# volume is not mounted; explicit GRASE_HF_NVME_ROOT skips the mount check.
if [ -z "${GRASE_HF_NVME_ROOT:-}" ] && ! findmnt -rn "$NVME_ROOT" >/dev/null 2>&1; then
  note "$NVME_ROOT not a mountpoint -> /fsx"
  echo "$SRC"; exit 0
fi
if [ ! -d "$NVME_ROOT" ] || ! mkdir -p "$DST" 2>/dev/null || [ ! -w "$DST" ]; then
  note "NVMe unavailable at $DST -> /fsx"
  echo "$SRC"; exit 0
fi

exec 9>"$DST.lock"
if ! flock -w 1200 9; then
  note "mirror lock timeout (another sync stuck?) -> /fsx"
  echo "$SRC"; exit 0
fi

t0=$SECONDS
# rsync lives on the pod's ephemeral root fs and vanishes on restart (rc=127 on
# the 0729_ppfix batch). Best-effort reinstall; any failure falls through to the
# rsync call below, whose existing failure path degrades to the /fsx source.
if ! command -v rsync >/dev/null 2>&1; then
  note "rsync missing — installing via apt-get"
  (export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq rsync) 1>&2 \
    || note "apt-get install rsync failed; falling back to /fsx source"
fi
if rsync -a --delete "$SRC/" "$DST/" 1>&2; then
  note "mirror ok in $((SECONDS - t0))s -> $DST"
  echo "$DST"
else
  note "rsync failed (rc=$?) after $((SECONDS - t0))s -> /fsx"
  echo "$SRC"
fi
exit 0
