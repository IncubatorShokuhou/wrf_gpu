#!/usr/bin/env bash
# Manager invokes the frozen snapshot only. Prime, release/scan, then warm/profile.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
LANE=<USER_HOME>/wrf_gpu2_lanes/w44-params
FAMILY=${PARAMS_FAMILY:-diffusion}
case "$FAMILY" in diffusion|mynn) ;; *) echo "unknown parameter family"; exit 2;; esac
RUN=$LANE/gpu/$FAMILY
CACHE=$LANE/gpu/cache
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
if [[ ${1:-} == --dry-run ]]; then
  bash -n "$0"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/root_gpu.py" \
    --input-dir "$CASE" --out "$LANE/gpu_dry/$FAMILY" --phase prime --family "$FAMILY" --dry-run
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 15 python "$HERE/supervise.py" --selftest
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 15 python "$HERE/profile_metrics.py" --selftest
  exit
fi
[[ ${PARAMS_ARM_SUPERVISOR_PID:-} == "$PPID" ]] || { echo 'use night_launch.sh: whole-arm supervisor required'; exit 9; }
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && end - now >= 1200 )) || { echo 'insufficient authorized window'; exit 5; }
RUN=$LANE/gpu/$FAMILY/$start
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
[[ ! -e $RUN/prime.json ]] || { echo 'refuse an already used arm directory'; exit 6; }
(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 200 ))
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]] || { echo 'launcher must acquire its own official lock'; exit 8; }
lock_headroom=${GPUWRF_LOCK_MIN_FREE_MIB:-24000}
for parameter_env in ${!GPUWRF_@}; do unset "$parameter_env"; done
export GPUWRF_LOCK_MIN_FREE_MIB="$lock_headroom" GPUWRF_GPU_ARM_CPUS=3,19,13,29
mkdir -p "$CACHE/jax" "$CACHE/autotune" "$CACHE/triton" "$CACHE/cuda" "$RUN/tmp" "$RUN/profile"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0
export GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TRITON_CACHE_DIR="$CACHE/triton"
export CUDA_CACHE_PATH="$CACHE/cuda" TMPDIR="$RUN/tmp" PYTHONPATH="$TREE/src" OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.25
export GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_RRTMG_MP_RE=1
unset XLA_FLAGS
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
rc=0
"$TREE/scripts/with_gpu_lock.sh" --label w44-params --timeout 1 -- \
  timeout --signal=TERM 930 python "$HERE/supervise.py" --seconds 900 --deadline-epoch "$end" -- python "$HERE/root_gpu.py" \
  --input-dir "$CASE" --out "$RUN" --family "$FAMILY" --phase prime > "$RUN/prime.log" 2>&1 || rc=$?
# The GPU lock is released before CPU E41. Always inspect actual cold artifacts,
# including a failed numerical arm; never discard the cache before this gate.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET: E41 pending, cache retained'; exit 3; fi
scan_rc=0
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 \
  "$TREE/scripts/with_ram_slot.sh" --label w44-params-e41 --max 12G --timeout 600 -- \
  timeout 600 python "$TREE/scripts/wave_cubin_gate.py" "$CACHE/jax" "$RUN/e41" \
  > "$RUN/e41.log" 2>&1 || scan_rc=$?
(( rc == 0 && scan_rc == 0 )) || { echo "prime=$rc E41=$scan_rc; stop, receipts retained"; exit 7; }
now=$(JAX_PLATFORMS=cpu "$REAL")
(( end - now >= 360 )) || { echo 'warm/profile need another manager window'; exit 5; }
"$TREE/scripts/with_gpu_lock.sh" --label w44-params --timeout 1 -- \
  timeout --signal=TERM 170 python "$HERE/supervise.py" --seconds 140 --deadline-epoch "$end" -- python "$HERE/root_gpu.py" \
  --input-dir "$CASE" --out "$RUN" --family "$FAMILY" --phase measure > "$RUN/measure.log" 2>&1
"$TREE/scripts/with_gpu_lock.sh" --label w44-params --timeout 1 -- \
  timeout --signal=TERM 180 python "$HERE/supervise.py" --seconds 150 --deadline-epoch "$end" -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false -o "$RUN/profile/roots" python "$HERE/root_gpu.py" \
  --input-dir "$CASE" --out "$RUN" --family "$FAMILY" --phase measure --profile --reps 3 > "$RUN/profile.log" 2>&1
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 \
  "$TREE/scripts/with_ram_slot.sh" --label w44-params-profile --max 8G --timeout 120 -- \
  timeout 120 nsys export --type sqlite --force-overwrite=false --output "$RUN/profile/roots.sqlite" "$RUN/profile/roots.nsys-rep"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 \
  timeout 60 python "$HERE/profile_metrics.py" "$RUN/profile/roots.sqlite" "$RUN/profile/metrics.json"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 \
  timeout 60 python "$HERE/program_metrics.py" "$RUN/measure.json" "$RUN/e41" "$RUN/program_metrics.json"
echo 'Root/E41/profile receipts complete; review the recorded numeric and cost gates before a KEEP decision. Cache retained for remaining families/review.'
