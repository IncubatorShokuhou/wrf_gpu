#!/usr/bin/env bash
# Standalone cold three-domain smoke. All model source is a detached snapshot.
set -uo pipefail
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python "$(dirname "$0")/gpu_authorization.py" || exit 3
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
(( $# == 2 )) || exit 2
source_tree=$1
deadline=$(date -u -d "$2" +%s)
now=$(date -u +%s)
cutoff=$(date -u -d "2026-10-15 19:10:00 UTC" +%s)
(( deadline > now && deadline <= cutoff && deadline-now <= 5400 )) || exit 3
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
case_dir=<USER_HOME>/wrf_gpu2_lanes/v-prep/WN3-smoke
helper=$(cd "$(dirname "$0")" && pwd)/wn3_gpu_smoke.py
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 0 nice -n 19 python "$(dirname "$helper")/spill_paths.py" WN3-smoke || exit 3
export PYTHONPATH="$source_tree/src"
"$source_tree/scripts/with_gpu_lock.sh" --label v-prep --timeout "$((deadline-now))" -- \
  bash -c 'now=$(date -u +%s); (( "$1"-now >= 1800 )) || exit 3;
    arm_budget=$(( "$1"-now-30 )); ((arm_budget>1770)) && arm_budget=1770;
    mkdir -p "$2/cache/jax" "$2/cache/autotune" "$2/cache/triton" "$2/cache/cuda" "$2/tmp";
    # Released W3R byte regression uses its ordinary census-OFF program.
    # The composite fidelity arms separately keep their full census ON.
    export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_RRTMG_MP_RE=1 GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_JAX_CACHE=1;
    export GPUWRF_JAX_CACHE_DIR="$2/cache/jax" JAX_COMPILATION_CACHE_DIR="$2/cache/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$2/cache/autotune" TRITON_CACHE_DIR="$2/cache/triton" CUDA_CACHE_PATH="$2/cache/cuda" TMPDIR="$2/tmp";
    export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30 GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=4;
    timeout --signal=TERM --kill-after=15 "$arm_budget" python "$3" "$4" > "$2/gpu.log" 2>&1' \
    wn3-smoke "$deadline" "$case_dir" "$helper" "$source_tree"
