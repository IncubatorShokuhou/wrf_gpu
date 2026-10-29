#!/usr/bin/env bash
# Synchronous CPU post-GPU phase. Caller v9 keeps its exclusion flag; all model children must be reaped.
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
MAN=${1:?manifest} WINDOW=${2:?GA_START_UTC} DEADLINE=${3:-2026-10-29T05:30:00Z}
R=<USER_HOME>/wrf_gpu2_lanes/v45-rc
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NOW=$(date -u +%s); END=$(date -u -d "$DEADLINE" +%s)
(( END - NOW >= 35*60 )) || { echo 'POST NOSTART: need full 35 minutes before 05:30; raw retained'; exit 4; }
exec env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- taskset -c 12,28 nice -n 19 timeout -k 15 2100 <USER_HOME>/miniconda3/bin/python3 -B "$SCRIPT_DIR/controller.py" --manifest "$MAN" --window-start "$WINDOW" --deadline "$DEADLINE" --apply
