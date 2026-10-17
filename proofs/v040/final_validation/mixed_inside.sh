#!/usr/bin/env bash
# One bounded lock: same-source cached Cu7 plus two fresh non-hail programs.
set -uo pipefail
test "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 || exit 2
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FINALVAL_PARALLEL=3
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu python <DATA_ROOT>/wrf_gpu2/scratch/v-prep/gpu16-tree/proofs/v040/final_validation/gpu_authorization.py || exit 3
export FINALVAL_DEADLINE_EPOCH=$(( $(date -u +%s) + 1800 ))
cutoff=$(date -u -d '2026-10-15 19:10:00 UTC' +%s)
(( FINALVAL_DEADLINE_EPOCH > cutoff )) && FINALVAL_DEADLINE_EPOCH=$cutoff
(( FINALVAL_DEADLINE_EPOCH - $(date -u +%s) >= 600 )) || exit 3
free -g
(
  cd <DATA_ROOT>/wrf_gpu2/scratch/v-prep/gpu15-tree || exit 2
  export PYTHONPATH="$PWD/src"
  bash proofs/v040/final_validation/gpu_inside.sh c29
) & old_pid=$!
(
  cd <DATA_ROOT>/wrf_gpu2/scratch/v-prep/gpu16-tree || exit 2
  export PYTHONPATH="$PWD/src"
  bash proofs/v040/final_validation/gpu_inside.sh c45 c89
) & new_pid=$!
result=0
wait "$old_pid" || result=1
wait "$new_pid" || result=1
exit "$result"
