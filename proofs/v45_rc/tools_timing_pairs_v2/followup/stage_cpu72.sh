#!/usr/bin/env bash
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/lib.sh
exec env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- taskset -c 12,28 nice -n 19 ionice -c3 timeout 3600 "$W45R_PY" -B "$W45R_TOOLS/reference_io.py" prepare --manifest "${1:?manifest}" --execute
