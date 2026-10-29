#!/usr/bin/env bash
# night_seq_v9/v10 entry: ARM (NO_TOPO|NO_QSQ|NO_E84) PHASE (cold|warm).
# Smoke uses the same CLI argv on CPU with --dry-run: PLAN_ONLY, GPU gates N/A.
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export PATH=<USER_HOME>/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
exec taskset -c 12,28 nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout 1860 python "$HERE/launch.py" "$@"
