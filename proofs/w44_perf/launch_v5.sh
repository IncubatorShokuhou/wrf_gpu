#!/usr/bin/env bash
# One builder for production and CUDA-mini. Only case/output namespaces differ.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
PY=<USER_HOME>/miniconda3/bin/python
PHASE=${1:?repeatcompare or hostprofile}; SIZE=${2:?prod or cuda_mini}; shift 2
CPU=0; if [[ ${1:-} == --cpu-probe ]]; then CPU=1; shift; fi
[[ $# == 0 && $PHASE =~ ^(repeatcompare|hostprofile)$ && $SIZE =~ ^(prod|cuda_mini)$ ]] || exit 2
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
unset W44_LAUNCH_SMOKE W44_SMOKE_ID W44_CAPTURE_ROOTS W44_CAPTURE_START W44_INIT_OBSERVER W44_HOST_PROFILE
export W44_CPU_PROBE=$CPU W44_SIZE=$SIZE
export W44_INPUT_CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
[[ $SIZE != cuda_mini ]] || export W44_INPUT_CASE=$L/cuda_mini_case_v1
CAP=600; [[ $PHASE != hostprofile ]] || CAP=180
[[ $SIZE != cuda_mini ]] || CAP=300
if [[ ${W44_V5_INSIDE:-0} != 1 ]]; then
  RID="${SIZE}_${PHASE}_$(date -u +%s)_$$"
  export W44_RUN_ROOT=$L/launch_v5_runs/$RID
  mkdir -p "$W44_RUN_ROOT"
  ROOT_FREE=$(df --output=avail -B1 "$L" | tail -1 | tr -d ' ')
  DATA_FREE=$(df --output=avail -B1 <DATA_ROOT> | tail -1 | tr -d ' ')
  ((ROOT_FREE>=20*1024**3 && DATA_FREE>=365*1024**3)) || exit 95
  [[ "$(sha256sum "$HERE/../../CUDA_MINI_MANIFEST.json" | awk '{print $1}')" == "${W44_SEAL_SHA256:?}" ]] || exit 4
  "$PY" "$HERE/verify_launch_v5.py"
  export W44_V5_INSIDE=1
  # This controller owns RAM scope, lock, nsys and all descendants until reaped.
  BUDGET=$((CAP-10)); [[ $SIZE != cuda_mini ]] || BUDGET=270
  RC=0
  "$PY" "$HERE/launch_supervisor.py" --whole-arm --seconds "$BUDGET" -- \
    systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
    bash "$0" "$PHASE" "$SIZE" $([[ $CPU != 1 ]] || printf '%s' --cpu-probe) || RC=$?
  # Supervisor has reaped the entire tree before any cache is removed.
  JAX_PLATFORMS=cpu "$PY" "$HERE/cuda_cleanup.py" "$W44_RUN_ROOT"
  exit "$RC"
fi
[[ ${W44_RUN_ROOT:-} == "$L"/launch_v5_runs/* ]] || exit 2
if [[ $CPU == 1 ]]; then
  export W44_LAUNCH_SMOKE=1 W44_SMOKE_ID=${W44_RUN_ROOT##*/}
fi
# Inner invocation reuses the parent's namespace; no second setsid/slot.
if [[ $CPU == 0 && ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  NOW=$(bash "$HERE/real_utc.sh")
  START=$(date -u -d "${GA_START_UTC:?}" +%s)
  END=$(date -u -d "${GA_END_UTC:?}" +%s)
  (( NOW>=START && END-NOW>=CAP+60 )) || exit 6
  exec bash <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --timeout 1 --label w44-perf -- \
    bash "$0" "$PHASE" "$SIZE"
fi
if [[ $CPU == 0 ]]; then
  [[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 && ${GPUWRF_GPU_LOCK_LABEL:-} == w44-perf ]] || exit 2
fi
if [[ $PHASE == repeatcompare ]]; then
  bash "$HERE/repeat_session.sh"
else
  bash "$HERE/host_session.sh"
fi
# These execute after the measurement/QUIET phase, with explicit CPU imports.
export JAX_PLATFORMS=cpu PYTHONPATH="$HERE/../../src"
"$PY" "$HERE/cuda_postflight.py" --phase "$PHASE" --root "$W44_RUN_ROOT" --cpu-probe "$CPU"
