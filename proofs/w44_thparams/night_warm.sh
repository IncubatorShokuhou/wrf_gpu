#!/usr/bin/env bash
# GA_START_UTC/GA_END_UTC only; reports (never requires) the cold half's recorded E41 verdict.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# v5 hard outer caps (subreaper teardown of every descendant): production GPU 300 + 300 s, CPU export
# 120 + metrics 60 s -> 900 s (plan cap 15 min); CUDA mini GPU 80 + 110 s, CPU 60 + 30 s -> 290 s (plan cap 5).
seconds=900; if [[ ${THPARAMS_CUDA_MINI:-0} == 1 ]]; then seconds=290; fi
exec python "$HERE/outer_supervise.py" --seconds "$seconds" -- bash "$HERE/night_launch.sh" warm_all
