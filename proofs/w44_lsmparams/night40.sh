#!/usr/bin/env bash
# One manager-sequencer step: BOTH W2 families, 40-minute total cap.
set -euo pipefail
umask 077
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
LANE=<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams
CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
if [[ ${1:-} == --dry-run ]]; then
  bash -n "$0"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  for family in lsm jarvis; do
    JAX_PLATFORMS=cpu timeout --foreground 30 python "$HERE/root_gpu.py" --input-dir "$CASE" \
      --out "$LANE/night40_dry/$family" --phase prime --family "$family" --dry-run
  done
  JAX_PLATFORMS=cpu timeout --foreground 20 python "$HERE/supervise.py" --selftest
  JAX_PLATFORMS=cpu timeout --foreground 15 python "$HERE/profile_metrics.py" --selftest
  exit
fi
if [[ ${1:-} != --_body ]]; then
  [[ $# == 0 ]] || exit 126
  # 10 seconds for sequencer dispatch and 20 for final teardown, inside 40 min.
  JAX_PLATFORMS=cpu python "$HERE/night_guard.py" --need 2390
  export W2_NIGHT_ADMITTED=1
  # The controller forwards caller-group cancellation to one payload group.
  # No nested setsid; all GNU timers below use --foreground.
  exec env JAX_PLATFORMS=cpu python -u "$HERE/supervise.py" --whole-step --seconds 2370 -- bash "$0" --_body
fi
[[ ${W2_NIGHT_ADMITTED:-0} == 1 && ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]] || exit 126
JAX_PLATFORMS=cpu python "$HERE/night_guard.py" --need 1
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 200 ))
lock_headroom=${GPUWRF_LOCK_MIN_FREE_MIB:-24000}
for parameter_env in ${!GPUWRF_@}; do unset "$parameter_env"; done
export GPUWRF_LOCK_MIN_FREE_MIB="$lock_headroom" GPUWRF_GPU_ARM_CPUS=3,19,13,29
export GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0
export GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_RRTMG_MP_RE=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.25
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$TREE/src" OMP_NUM_THREADS=4
unset XLA_FLAGS JAX_PLATFORM_NAME XLA_PYTHON_CLIENT_MEM_FRACTION
status=0
drop_family_cache() {
  # All GPU processes have exited, E41 has consumed the native blobs, and
  # warm/profile receipts are durable. Nsight export needs only its own trace.
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET: cache cleanup pending'; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout --foreground 30 python - "$family" <<'PY'
import shutil, sys
if sys.argv[1] == "lsm":
    shutil.rmtree("<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams/gpu/night40/cache/lsm")
elif sys.argv[1] == "jarvis":
    shutil.rmtree("<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams/gpu/night40/cache/jarvis")
else:
    raise ValueError("unknown cache family")
PY
}
for family in lsm jarvis; do
  RUN=$LANE/gpu/night40/$family
  CACHE=$LANE/gpu/night40/cache/$family
  [[ ! -e $RUN/prime.log ]] || { echo 'refuse an already used arm directory'; exit 6; }
  mkdir -p "$CACHE/jax" "$CACHE/autotune" "$CACHE/triton" "$CACHE/cuda" "$RUN/tmp" "$RUN/profile"
  export GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
  export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TRITON_CACHE_DIR="$CACHE/triton"
  export CUDA_CACHE_PATH="$CACHE/cuda" TMPDIR="$RUN/tmp"
  JAX_PLATFORMS=cpu python "$HERE/night_guard.py" --need 1190
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  rc=0
  "$TREE/scripts/with_gpu_lock.sh" --label w44-lsmparams --timeout 1 -- \
    env JAX_PLATFORMS=cuda timeout --foreground --signal=TERM 735 python "$HERE/supervise.py" --seconds 720 -- \
    python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$RUN" --family "$family" --phase prime \
    > "$RUN/prime.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$RUN/prime.rc"
  # Release the official lock before CPU E41. Scan real cache even if prime
  # failed; preserve every cold artifact until the scan/consumer disposition.
  scan_rc=0
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET: E41 pending, cache retained'; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 \
    timeout --foreground 190 "$TREE/scripts/with_ram_slot.sh" --label w44-lsmparams-e41 --max 12G --timeout 10 -- \
    timeout --foreground 175 python "$TREE/scripts/wave_cubin_gate.py" "$CACHE/jax" "$RUN/e41" \
    > "$RUN/e41.log" 2>&1 || scan_rc=$?
  printf '%s\n' "$scan_rc" > "$RUN/e41.rc"
  if (( rc != 0 || scan_rc != 0 )); then
    # An unfinished E41 is still a dependent consumer: retain only that cache
    # for the CPU continuation, and do not claim the gate passed.
    (( scan_rc != 0 )) || drop_family_cache
    status=7
    continue
  fi
  JAX_PLATFORMS=cpu python "$HERE/night_guard.py" --need 240
  "$TREE/scripts/with_gpu_lock.sh" --label w44-lsmparams --timeout 1 -- \
    env JAX_PLATFORMS=cuda timeout --foreground --signal=TERM 115 python "$HERE/supervise.py" --seconds 100 -- \
    python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$RUN" --family "$family" --phase measure \
    > "$RUN/measure.log" 2>&1
  "$TREE/scripts/with_gpu_lock.sh" --label w44-lsmparams --timeout 1 -- \
    env JAX_PLATFORMS=cuda timeout --foreground --signal=TERM 115 python "$HERE/supervise.py" --seconds 100 -- \
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o "$RUN/profile/roots" \
    python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$RUN" --family "$family" --phase measure --profile --reps 3 \
    > "$RUN/profile.log" 2>&1
  drop_family_cache
done
echo "Both-family night step finished rc=$status; inspect per-family prime/E41/warm/profile receipts."
exit "$status"
