#!/usr/bin/env bash
# Manager sequencer entry: no GPU work until explicit GO for the immutable spec.
set -euo pipefail
LANE_DIR=<USER_HOME>/wrf_gpu2_lanes/w44-branch
SPEC=${BRANCH_GATE_SPEC:-$LANE_DIR/GPU_SPEC_V2.json}
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(JAX_PLATFORMS=cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$SPEC")
if [[ ${1:-} == --dry-run ]]; then
  bash -n "$0"
  JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 60 python "$HERE/gpu_gate.py" "$SPEC" --dry-run
  exit
fi
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && end - now >= 1800 )) || { echo 'outside manager 30-minute window'; exit 5; }
SPEC_SHA=$(sha256sum "$SPEC" | cut -d ' ' -f1)
rg -Fq "manager -> w44-branch: GO GPU $SPEC_SHA" <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md || {
  echo 'manager GO for this exact spec is missing'; exit 6;
}
if [[ ${1:-} != --locked ]]; then
  (( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 200 ))
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  rc=0
  "$TREE/scripts/with_gpu_lock.sh" --label w44-branch --timeout 1 -- timeout --signal=TERM --kill-after=30 1740 bash "$0" --locked || rc=$?
  printf 'GPU END rc=%s\n' "$rc"
  # Cold native cache must be scanned even after a partial/failed scientific arm.
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET; cache retained for E41'; exit 3; fi
  CACHE=$(JAX_PLATFORMS=cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache"])' "$SPEC")
  RUN_ROOT=$(JAX_PLATFORMS=cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$SPEC")
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 "$TREE/scripts/with_ram_slot.sh" --label w44-branch-e41 --max 6G --timeout 600 -- timeout 600 python "$TREE/scripts/wave_cubin_gate.py" "$CACHE/jax" "$LANE_DIR/GPU_E41" --run-proof "$RUN_ROOT/control_run.json" > "$LANE_DIR/GPU_E41.log" 2>&1 || {
    echo 'E41 incomplete/failed; cache retained'; exit 4;
  }
  if (( rc == 0 )); then
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET; verified outputs retained until cleanup'; exit 3; fi
    JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 180 python "$HERE/gpu_gate.py" "$SPEC" --postprocess
  fi
  exit "$rc"
fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]]
CACHE=$(JAX_PLATFORMS=cpu python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache"])' "$SPEC")
mkdir -p "$CACHE/jax" "$CACHE/triton" "$CACHE/cuda" "$CACHE/autotune" "$CACHE/tmp"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0
export PYTHONPATH="$TREE/src" GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/cuda" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TMPDIR="$CACHE/tmp"
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30 OMP_NUM_THREADS=4
unset XLA_FLAGS CUDA_ROOT
python "$HERE/gpu_gate.py" "$SPEC"
