#!/bin/bash
# night_seq_v9 exact-call entry. The manager supplies an immutable GRAUPEL_TREE.
set -euo pipefail
umask 077
tree=${GRAUPEL_TREE:?immutable archive path required}
lane=<USER_HOME>/wrf_gpu2_lanes/v45-graupel
py=<USER_HOME>/miniconda3/bin/python
mode=gpu
if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 ]]; then mode=smoke; fi
out=$lane/GPU_component_$mode
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]
for var in $(compgen -e | grep -E 'TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$' || true); do unset "$var"; done
export PYTHONPATH="$tree/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GPUWRF_FAST_DEFAULTS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUWRF_JAX_CACHE_DIR="$out/bootstrap_cache" TRITON_CACHE_DIR="$out/triton"
export CUDA_CACHE_PATH="$out/cuda" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$out/autotune"
if [[ $mode == smoke ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu mkdir -p "$out"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 \
    taskset -c 12,28 nice -n 19 flock "$out/private.lock" timeout --kill-after=10 300 \
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 "$py" "$tree/proofs/v45_graupel/qualify.py" --mode smoke --out "$out"
else
  real=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
  now=$(JAX_PLATFORMS=cpu "$real")
  start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
  end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
  (( now >= start && now + 310 <= end )) || { echo 'outside/insufficient authorized window'; exit 5; }
  export GPUWRF_GPU_ARM_CPUS=3,19,13,29
  JAX_PLATFORMS=cpu mkdir -p "$out"
  JAX_PLATFORMS=cpu systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 \
    "$tree/scripts/with_gpu_lock.sh" --label v45-graupel --timeout 1 -- \
    timeout --kill-after=10 300 env JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=1 \
    "$py" "$tree/proofs/v45_graupel/qualify.py" --mode gpu --out "$out"
fi
