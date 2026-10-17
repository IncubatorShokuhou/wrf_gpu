#!/usr/bin/env bash
set -uo pipefail
test "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 || exit 2
root=<USER_HOME>/wrf_gpu2_lanes/v-prep
parallel=${GPUWRF_FINALVAL_PARALLEL:-3}
[[ "$parallel" = 3 || "$parallel" = 4 ]] || exit 2
(( $# <= parallel )) || exit 2
# One lock acquisition is one <=30min cohort, including timeout/receipt grace.
cohort_deadline=$(( $(date -u +%s) + 1800 ))
free -g
if [[ "$parallel" = 4 ]]; then
  # Owner13:13Z: four ordinary cases only if measured peaks plus 1GiB
  # headroom each fit the actual free device memory at lock acquisition.
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
  (( free_mib >= 4 * (5900 + 1024) )) || exit 3
fi
launch() {
  local id=$1
  [[ $id =~ ^c[0-9][0-9]$ ]] || return 2
  local case_dir=$root/$id
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python "$(dirname "$0")/gpu_authorization.py" || return 3
  # Recheck after lock acquisition: queueing must not cross the last start.
  local now effective_deadline last_start hard_end
  now=$(date -u +%s)
  effective_deadline=$FINALVAL_DEADLINE_EPOCH
  (( effective_deadline > cohort_deadline )) && effective_deadline=$cohort_deadline
  last_start=$(date -u -d "2026-10-15 18:15:00 UTC" +%s)
  hard_end=$(date -u -d "2026-10-15 18:45:00 UTC" +%s)
  (( now <= last_start )) || return 3
  (( effective_deadline > hard_end )) && effective_deadline=$hard_end
  (( effective_deadline - now >= 600 )) || return 3
  local nested arm_budget fraction
  nested=$(JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python -c 'import json,sys; print(int(len(json.load(open(sys.argv[1]))["domains"])>1))' "$case_dir/inputs.json")
  arm_budget=1770
  fraction=0.15
  if [[ "$id" = c28 || "$id" = c65 || "$id" = c58 || "$id" = c59 || "$id" = c60 ]]; then fraction=0.30; fi
  if [[ "$nested" = 1 ]]; then
    fraction=0.30
  fi
  # Leave time for timeout's kill grace and the launch journal before the
  # enclosing absolute-deadline supervisor terminates this shell.
  if (( arm_budget > effective_deadline - now - 30 )); then
    arm_budget=$((effective_deadline - now - 30))
  fi
  test ! -e "$case_dir/gpu_run.json" || return 2
  test ! -e "$case_dir/gpu_launch.json" || return 2
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/v040/final_validation/spill_paths.py "$id" || return 3
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/v040/final_validation/before_arm.py \
    "$id" "$(JAX_PLATFORMS=cpu git rev-parse HEAD)" || return 2
  mkdir -p "$case_dir/cache/jax" "$case_dir/cache/autotune" "$case_dir/cache/triton" "$case_dir/cache/cuda" "$case_dir/tmp"
  # Charge host setup to the cohort too, not just the GPU child's lifetime.
  now=$(date -u +%s)
  (( effective_deadline - now > 30 )) || return 3
  (( arm_budget > effective_deadline - now - 30 )) && arm_budget=$((effective_deadline - now - 30))
  local arm_started arm_rc
  arm_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  env JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 \
    GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 GPUWRF_JAX_CACHE=1 \
    GPUWRF_JAX_CACHE_DIR="$case_dir/cache/jax" JAX_COMPILATION_CACHE_DIR="$case_dir/cache/jax" \
    GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$case_dir/cache/autotune" \
    TRITON_CACHE_DIR="$case_dir/cache/triton" CUDA_CACHE_PATH="$case_dir/cache/cuda" TMPDIR="$case_dir/tmp" \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION="$fraction" \
    GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=1 \
    timeout --signal=TERM --kill-after=15 "$arm_budget" python proofs/v040/final_validation/gpu_case.py "$case_dir" \
    > "$case_dir/gpu.log" 2>&1
  arm_rc=$?
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 timeout 15 python proofs/v040/final_validation/record_gpu_arm.py \
    "$id" "$arm_started" "$arm_rc" "$arm_budget" "$fraction"
  return "$arm_rc"
}
# Three ordinary Swiss processes by default; four under the measured preflight.
# Nested runs are submitted ALONE, under the same30min cohort cap.
# A separate cold-setup exception needs a new manager allocation.
result=0
while (( $# )); do
  pids=()
  for ((slot=1; slot<=parallel; slot++)); do
    (( $# )) || break
    launch "$1" & pids+=("$!")
    shift
  done
  for pid in "${pids[@]}"; do wait "$pid" || result=1; done
done
exit "$result"
