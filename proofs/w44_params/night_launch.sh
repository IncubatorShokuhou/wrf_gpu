#!/usr/bin/env bash
# The published entry point supervises the entire arm, including nested timeouts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} == --dry-run ]]; then
  exec env JAX_PLATFORMS=cpu python "$HERE/supervise.py" --whole-arm --seconds 60 -- \
    bash "$HERE/gpu_session.sh" --dry-run
fi
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && end - now >= 1200 )) || { echo 'insufficient authorized window'; exit 5; }
budget=$((end-now))
(( budget <= 2700 )) || budget=2700
exec env JAX_PLATFORMS=cpu python "$HERE/supervise.py" --whole-arm \
  --seconds "$budget" --deadline-epoch "$end" -- bash "$HERE/gpu_session.sh"
