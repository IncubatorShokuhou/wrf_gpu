#!/usr/bin/env bash
# CUDA mini, warm half (manager 21:48Z Option A): the SAME v5 warm entry with
# THPARAMS_CUDA_MINI=1 -- 5-column Thompson MODEL SWAP, not a grid crop. Requires
# cuda_mini_cold.sh (prepared.json; its recorded E41 verdict is reported). GPU caps 80 s measure +
# 110 s nsys profile (expected ~1 min); post export/metrics on the CPU after lock release.
# night_warm.sh applies the hard 290 s outer cap (plan cap 5 min). THPARAMS_LAUNCH_SMOKE=1 = CPU probe.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export THPARAMS_CUDA_MINI=1
exec bash "$HERE/night_warm.sh"
