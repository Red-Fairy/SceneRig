#!/bin/bash
# Pull the Isaac Sim install into THIS NODE's page cache before any lane boots Kit.
#
# Why (measured 2026-07-28, audits/RUNTIME_PERF_AUDIT_2026_07_28.md §9.4):
#   Isaac cold boot ~225 s on a fully idle box, ~67 s once warm. The cost is Lustre
#   per-file metadata latency across a 25 GB / 63,414-file install -- stat 1.76 ms cold
#   vs 0.069 ms warm (25x), open+read 2.93 ms vs 0.73 ms -- NOT bandwidth, which runs
#   320-450 MB/s. So the "warm" case is nothing but the kernel page cache, and we can
#   populate it deliberately instead of paying a 300 s boot timeout to populate it by
#   accident (which is exactly what 0728_idle_9096 did, twice).
#
# This COPIES NOTHING. It reads the files and discards the bytes; Isaac still boots from
# the same /fsx paths. Deliberately NOT an NVMe copy: the instance store is wiped on pod
# restart (that is what cost the 0727 batch ~55 % wall-clock), and RAM beats NVMe anyway.
#
# Sizing: node has ~2 TB RAM, install is 25 GB => 1.25 % of memory, no eviction pressure.
set -uo pipefail

ISAAC_ROOT="${GRASE_ISAAC_ROOT:-/fsx/rundongluo/isaac/venv/lib/python3.11/site-packages/isaacsim}"
# /tmp is node-local overlayfs, so the marker has the SAME scope and lifetime as the page
# cache it describes. A marker on /fsx would make a second node skip a warm-up it never
# did; and a pod restart wipes /tmp at exactly the moment the cache is dropped. Both right.
MARKER="${GRASE_PREWARM_MARKER:-/tmp/.grase_isaac_prewarm.done}"
LOCK="${MARKER}.lock"
MAX_AGE="${GRASE_PREWARM_MAX_AGE:-21600}"   # 6 h
PARALLEL="${GRASE_PREWARM_PARALLEL:-32}"    # latency-bound, so parallelism is what matters,
                                            # not bandwidth. Measured 2026-07-28: the full
                                            # 25 GB install warms in 59 s at -P 32 (nearly all
                                            # of it sys time -- kernel I/O, ~0.6 s user).
TIMEOUT="${GRASE_PREWARM_TIMEOUT:-300}"

if [ "${GRASE_PREWARM_ISAAC:-1}" = "0" ]; then
    exit 0
fi
if [ ! -d "$ISAAC_ROOT" ]; then
    echo "[prewarm] no Isaac install at $ISAAC_ROOT; skipping" >&2
    exit 0
fi

# Serialize across lanes: 8 lanes each launching a 25 GB parallel read would multiply the
# very Lustre metadata contention this exists to remove. First lane warms, rest fall
# through to the freshness check below and return immediately.
# NB: no redirection on this exec other than fd 9 -- `exec` with a redirection and no
# command applies it to the CURRENT shell, so a stray `2>/dev/null` here would silently
# mute the whole script's stderr (including the log line below).
exec 9>"$LOCK" || exit 0
flock 9 || exit 0

if [ -f "$MARKER" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$MARKER" 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$MAX_AGE" ]; then
        exit 0
    fi
fi

t0=$(date +%s)
timeout "$TIMEOUT" bash -c '
    find "$1" -type f -print0 2>/dev/null \
        | xargs -0 -P "$2" -n 128 cat 2>/dev/null > /dev/null
' _ "$ISAAC_ROOT" "$PARALLEL" || true
echo "[prewarm] isaac page cache warmed in $(( $(date +%s) - t0 ))s ($ISAAC_ROOT)" >&2
touch "$MARKER"
