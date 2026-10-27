#!/usr/bin/env bash
# Immutable once posted. Usage: gpu_prototype.sh SNAPSHOT RUN_DIR [--dry-run]
# No child with a CUDA pool is spawned: one measured Python in the official lock.
set -euo pipefail
SNAP=$(realpath "$1")
RUN=$2
CASE=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
if [[ ${3:-} == --dry-run ]]; then
  [[ -f $SNAP/proofs/w44_budget/coupled_probe.py && -f $CASE/namelist.input ]]
  printf 'source=%s\ncase=%s\nrun=%s\nfirst-step then 3h ON/OFF; official GPU lock; timeout1200; E41 after process\n' "$SNAP" "$CASE" "$RUN"
  exit 0
fi
[[ $RUN == <USER_HOME>/wrf_gpu2_lanes/w44-budget/* ]]
[[ $(git -C "$SNAP" status --porcelain --untracked-files=no) == '' ]]
[[ ! -e $RUN ]] || { echo 'Run directory already exists'; exit 3; }
# The manager provides a real-UTC window when dispatching the immutable plan.
: "${W44_GPU_NOT_BEFORE:?manager real-UTC start required}"
: "${W44_GPU_DEADLINE:?manager real-UTC end required}"
NOW=$(JAX_PLATFORMS=cpu bash "$RU")
START=$(date -u -d "$W44_GPU_NOT_BEFORE" +%s)
END=$(date -u -d "$W44_GPU_DEADLINE" +%s)
(( NOW >= START && NOW + 300 < END )) || { echo 'Outside manager window'; exit 4; }
mkdir -p "$RUN"
JAX_PLATFORMS=cpu bash "$RU" --iso > "$RUN/start_real_utc.txt"
git -C "$SNAP" rev-parse HEAD HEAD:src/gpuwrf > "$RUN/source.txt"
sha256sum "$SNAP/proofs/w44_budget/coupled_probe.py" > "$RUN/runner.sha256"
export SNAP RUN CASE RU END
RC=0
"$SNAP/scripts/with_gpu_lock.sh" --label w44-budget --timeout 600 -- bash -c '
  set -euo pipefail
  NOW=$(JAX_PLATFORMS=cpu bash "$RU")
  REM=$(( END - NOW - 60 ))
  (( REM >= 300 )) || exit 4
  (( REM <= 1200 )) || REM=1200
  # System clock is slewing; apply the established 0.9 duration margin.
  CAP=$(( REM * 9 / 10 ))
  mkdir -p "$RUN/cache/jax" "$RUN/cache/triton" "$RUN/cache/cuda" "$RUN/tmp"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RUN/gpu_start.txt"
  journalctl -k --no-pager 2>/dev/null | rg -i "hibernat" | tail -n 8 > "$RUN/hibernation.txt" || true
  env -u JAX_PLATFORMS GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 \
    GPUWRF_JAX_CACHE_DIR="$RUN/cache/jax" JAX_COMPILATION_CACHE_DIR="$RUN/cache/jax" \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$RUN/cache/jax/xla_gpu_per_fusion_autotune_cache_dir" \
    TRITON_CACHE_DIR="$RUN/cache/triton" CUDA_CACHE_PATH="$RUN/cache/cuda" \
    TMPDIR="$RUN/tmp" XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=.22 \
    GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF PYTHONPATH="$SNAP/src" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout -k 20 "$CAP" python -u "$SNAP/proofs/w44_budget/coupled_probe.py" \
      --input-dir "$CASE" --out "$RUN/probe" --max-dom 3 > "$RUN/probe.log" 2>&1
' || RC=$?
# Always scan actual new stepping executables, including the STOP_NONBYTE case.
if [[ -e /tmp/wrf_gpu2_quiet ]]; then echo 'E41 pending: QUIET'; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 \
  "$SNAP/scripts/with_ram_slot.sh" --label w44-budget --max 12G --timeout 300 -- \
  timeout 300 python "$SNAP/scripts/wave_cubin_gate.py" "$RUN/cache/jax" "$RUN/e41" \
  > "$RUN/e41.log" 2>&1 || E41_RC=$?
printf '%s\n' "$RC" > "$RUN/probe.rc"
printf '%s\n' "${E41_RC:-0}" > "$RUN/e41.rc"
JAX_PLATFORMS=cpu bash "$RU" --iso > "$RUN/end_real_utc.txt"
exit "$RC"
