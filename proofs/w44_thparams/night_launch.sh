#!/usr/bin/env bash
# Frozen manager night launcher. Cold preparation and warm measurement are
# separate arms; CPU E41/export runs after releasing the official GPU lock.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
LANE=<USER_HOME>/wrf_gpu2_lanes/w44-thparams
# v5: every phase in its OWN user systemd scope (MemoryMax, no swap; GPU 16G, CPU 6-12G), never the
# shared RAM-slot FIFO (E233); the recorded E41 verdict is reported, never a precondition (E230).
# Run dirs/caches on the <DATA_ROOT> lane scratch (root fs is at 97 %).
SCR=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams
CACHE=$SCR/gpu_v5/cache
RUN=$SCR/gpu_v5
smoke=${THPARAMS_LAUNCH_SMOKE:-0}
if [[ $smoke == 1 ]]; then RUN=$SCR/launch_smoke_v5; CACHE=$RUN/cache; fi
# CUDA mini (manager 21:48Z): same launcher/env/lock/supervisor/E41/checker with the
# 5-column Thompson MODEL SWAP (not a grid crop); smoke=1 + mini=1 = its CPU probe.
mini=${THPARAMS_CUDA_MINI:-0}
[[ $mini == 0 || $mini == 1 ]]
MINI_ARGS=(); MEASURE_CAP=300; PROFILE_CAP=300; EXPORT_CAP=120; METRICS_CAP=60
# Mini warm half fits its 5-min plan cap: GPU 80 + 110 s, CPU export 60 + metrics 30 s.
if [[ $mini == 1 ]]; then MINI_ARGS=(--mini); MEASURE_CAP=80; PROFILE_CAP=110; EXPORT_CAP=60; METRICS_CAP=30; RUN=$SCR/cuda_mini_v5; CACHE=$RUN/cache; fi
if [[ $mini == 1 && $smoke == 1 ]]; then RUN=$SCR/cuda_mini_probe_v5; CACHE=$RUN/cache; fi
CASE=${THPARAMS_CASE:-<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725}
FIXTURE=$TREE/tests/v025/b_thompson/fixtures/mixed_phase_columns.npz
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
stage=${1:?use dry-run, prime, measure, or post}
case "$stage" in dry-run|prime|measure|post|warm_all) ;; *) exit 2;; esac
[[ $stage != prime ]] || { echo 'Use four night_prepare_{off,on}_{d01,d02}.sh arms, then night_check.sh'; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu python3 -I "$HERE/check_alignment.py" "$CASE/namelist.input" 3  # E225: hours*3600 % every dt == 0
if [[ $stage == warm_all ]]; then
  bash "$0" measure
  if [[ $smoke == 1 ]]; then exit; fi
  exec bash "$0" post
fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]] || { echo 'launcher acquires its own lock'; exit 2; }
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
headroom=${GPUWRF_LOCK_MIN_FREE_MIB:-24000}
for parameter_env in ${!GPUWRF_@}; do unset "$parameter_env"; done
export GPUWRF_LOCK_MIN_FREE_MIB="$headroom" GPUWRF_GPU_ARM_CPUS=3,19,13,29
export GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0
export GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/cuda"
export PYTHONPATH="$TREE/src" OMP_NUM_THREADS=4 GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_RRTMG_MP_RE=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.25
unset XLA_FLAGS
if [[ $stage == dry-run ]]; then
  bash -n "$0"
  # Fresh stamped output per dry-run (gpu_component opens its receipt exclusively; a fixed
  # $LANE/launcher_dry path made every repeat dry-run refuse, SM06 00:24Z).
  DRY=$RUN/launcher_dry/$(date -u +%Y%m%dT%H%M%S%N)
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/gpu_component.py" --fixture "$FIXTURE" --out "$DRY/component" --phase prime --dry-run
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$DRY/root" --phase prime --dry-run
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 15 python "$HERE/supervise.py" --selftest
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 15 python "$HERE/profile_metrics.py" --selftest
  exit
fi
if [[ $stage == post ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 -- timeout $EXPORT_CAP nsys export --type sqlite --output "$RUN/profile/root.sqlite" "$RUN/profile/root.nsys-rep"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 -- timeout $METRICS_CAP python "$HERE/profile_metrics.py" "$RUN/profile/root.sqlite" "$RUN/profile/root_metrics.json"
  echo 'Export done. Preserve receipts, then remove exact owned caches/raw traces after all comparisons.'
  exit
fi
if [[ $smoke != 1 ]]; then
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${GA_START_UTC:?manager night window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager night window required}" +%s)
(( now >= start && now < end )) || { echo 'outside authorized window'; exit 5; }
(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 365 ))
# E231: refuse before any lock unless the user manager grants a memory-capped scope.
systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 -- true || { echo 'user scope refused (E231)'; exit 7; }
# Cap each complete GPU phase at 15 min and respect the manager window.
deadline=$((now + 900)); (( deadline <= end )) || deadline=$end
fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu mkdir -p "$CACHE/jax" "$CACHE/autotune" "$CACHE/triton" "$CACHE/cuda" "$RUN/tmp" "$RUN/profile" "$RUN/root"
export JAX_PLATFORMS=cuda TMPDIR="$RUN/tmp"

gpu() {
  local requested=$1 label=$2 remaining seconds; shift 2
  if [[ $smoke == 1 ]]; then
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- timeout 300 python "$HERE/outer_supervise.py" --seconds 280 -- "$@" --cpu-smoke > "$RUN/$label-smoke.log" 2>&1
    return
  fi
  remaining=$((deadline - $(JAX_PLATFORMS=cpu "$REAL") - 15))
  (( remaining >= 30 )) || { echo 'next GPU phase needs another authorized window'; return 5; }
  seconds=$requested; (( seconds <= remaining )) || seconds=$remaining
  systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 -- "$TREE/scripts/with_gpu_lock.sh" --label w44-thparams --timeout 1 -- \
    timeout --signal=TERM --kill-after=10 $((seconds+5)) python "$HERE/supervise.py" --seconds "$seconds" -- "$@" > "$RUN/$label.log" 2>&1
}

  # The cold half (four preparations + night_check) must have finished; its E41 verdict is reported only.
  [[ -e $RUN/prepared.json ]] || { echo "cold half incomplete: no $RUN/prepared.json"; exit 5; }
  if [[ -e $RUN/e41/verdict.json ]]; then echo "E41 verdict of the cold half:"; cat "$RUN/e41/verdict.json"
  else echo 'E41 verdict not recorded (recover it on the CPU from the retained cache, E220)'; fi
  gpu $MEASURE_CAP root_measure python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$RUN/root" --phase measure "${MINI_ARGS[@]}"
  if [[ $smoke == 1 ]]; then
    JAX_PLATFORMS=cpu python - "$RUN/root/measure_cpu_smoke.json" "$mini" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='PASS' and r['cpu_smoke'] and r['platform']=='cpu' and not r['gpu_executed']
assert r.get('mini',False)==(sys.argv[2]=='1')
assert r['changed_active'] and r['environment']['JAX_PLATFORMS']=='cpu'
print('LAUNCH SMOKE PASS w44-thparams warm',sys.argv[1])
PY
    exit
  fi
  gpu $PROFILE_CAP root_profile nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o "$RUN/profile/root" python "$HERE/root_gpu.py" --input-dir "$CASE" --out "$RUN/root" --phase measure --profile --reps 3 "${MINI_ARGS[@]}"
  echo 'GPU measurements done. CPU WRF scoring and profile export follow; no final verdict inferred.'
