#!/usr/bin/env bash
# Probe the normal JAX cache using exact original PIN filenames and bytes.
set -uo pipefail
test "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 || exit 2
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FINALVAL_PARALLEL=3
export JAX_LOG_COMPILES=1 JAX_EXPLAIN_CACHE_MISSES=1
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu python proofs/v040/final_validation/gpu_authorization.py || exit 3
export FINALVAL_DEADLINE_EPOCH=$(( $(date -u +%s) + 1800 ))
cutoff=$(date -u -d '2026-10-15 19:10:00 UTC' +%s)
(( FINALVAL_DEADLINE_EPOCH > cutoff )) && FINALVAL_DEADLINE_EPOCH=$cutoff
(( FINALVAL_DEADLINE_EPOCH - $(date -u +%s) >= 600 )) || exit 3
started=$(date -u +%FT%TZ)
setsid bash proofs/v040/final_validation/gpu_inside.sh c45 & job=$!
trap 'kill -TERM -- "-$job" 2>/dev/null || true' EXIT TERM INT
log=<USER_HOME>/wrf_gpu2_lanes/v-prep/c45/gpu.log
while kill -0 "$job" 2>/dev/null; do
  if rg -q 'PERSISTENT COMPILATION CACHE MISS.*advance_chunk_fori' "$log" 2>/dev/null; then
    echo 'READINESS_ABORT: normal main-program JAX cache miss; no repeated cold compile'
    kill -TERM -- "-$job" 2>/dev/null || true
    wait "$job" || true
    if [ ! -e <USER_HOME>/wrf_gpu2_lanes/v-prep/c45/gpu_launch.json ]; then
      JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/v040/final_validation/record_gpu_arm.py c45 "$started" 143 1770 0.15
    fi
    exit 3
  fi
  sleep 1
done
wait "$job"
