#!/usr/bin/env bash
# Manager-sequenced 2-minute W4 admission probe. No GPU waiter or fallback compile.
set -euo pipefail
export JAX_PLATFORMS=cpu
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=<USER_HOME>/wrf_gpu2_lanes/w44-precompile
MAIN=<USER_HOME>/src/wrf_gpu2
PY=<USER_HOME>/miniconda3/bin/python
[[ $# == 0 || ( $# == 1 && "$1" == --cpu-dry ) ]] || exit 2
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
START=$(date -u -d "${GA_START_UTC:?manager real UTC window required}" +%s)
END=$(date -u -d "${GA_END_UTC:?manager real UTC window required}" +%s)
offset=$(chronyc tracking | awk '/^System time/{v=$4; if($6=="fast") print -v; else print v}')
[[ -n "$offset" ]] || exit 2
NOW=$(JAX_PLATFORMS=cpu "$PY" -c 'import sys,time; print(int(time.time()+float(sys.argv[1])))' "$offset")
if [[ "${1:-}" == --cpu-dry ]]; then NOW=${W44_DRY_NOW:-$NOW}; fi
(( NOW >= START )) || exit 4
(( END - NOW >= 120 )) || exit 6
[[ -s "$BASE/prod_d01/program.xlaexec" && -s "$HERE/gpu_admission.py" ]] || exit 2
if [[ "${1:-}" == --cpu-dry ]]; then echo 'CPU_DRY_PASS no GPU/lock/backend'; exit 0; fi
[[ $# == 0 ]] || exit 2
# The manager allocates a short RAM slot before the GPU lock; never queue on RAM
# while holding the GPU. The official wrapper retains the lock until work exits.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout --signal=TERM --kill-after=10s 110s \
  bash "$MAIN/scripts/with_ram_slot.sh" --label w44-precompile-admission --max 6G --timeout 10 -- \
  bash "$MAIN/scripts/with_gpu_lock.sh" --timeout 1 --label w44-precompile -- \
  env JAX_PLATFORMS=gpu GPUWRF_JAX_CACHE=0 GPUWRF_XLA_AUTOTUNE_CACHE=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 \
  timeout --signal=TERM --kill-after=5s 75s "$PY" "$HERE/gpu_admission.py" \
    --source "$BASE/snapshot" --artifact "$BASE/prod_d01" --out "$BASE/GPU_ADMISSION.json"
