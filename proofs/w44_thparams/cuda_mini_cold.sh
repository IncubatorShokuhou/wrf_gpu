#!/usr/bin/env bash
# CUDA mini, cold half (manager 21:48Z Option A): the SAME four v5 preparation entries
# and night_check.sh with THPARAMS_CUDA_MINI=1, i.e. the 5-column Thompson MODEL SWAP
# (interpret=False; d02 = its own 6-column program) instead of the PROD domain programs -- NOT a grid crop.
# GA_START_UTC/GA_END_UTC required (manager window). GPU cap 4 x 50 s = 200 s (expected ~2 min);
# night_check then runs inventory (<= 20 s) + E41 (<= 80 s) on the CPU after lock release and RECORDS the
# E41 verdict of the real mini cubins. Hard outer cap 290 s for the whole half (plan cap 5 min).
# THPARAMS_LAUNCH_SMOKE=1 = CPU probe.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export THPARAMS_CUDA_MINI=1
if [[ ${THPARAMS_MINI_HALF:-0} != 1 ]]; then
  THPARAMS_MINI_HALF=1 exec python "$HERE/outer_supervise.py" --seconds 290 -- bash "$HERE/cuda_mini_cold.sh"
fi
for arm in off_d01 off_d02 on_d01 on_d02; do
  bash "$HERE/night_prepare_$arm.sh"
done
exec bash "$HERE/night_check.sh"
