#!/usr/bin/env bash
# Component-only plan entry. Manager supplies immutable source; each PDF owns its cache.
set -euo pipefail
phase=${1:?expected operators}; test "$phase" = operators
source_root=${MO_SOURCE_ROOT:?immutable archive required}
output_root=${MO_OUTPUT_ROOT:?owned lane output required}
python=<USER_HOME>/miniconda3/bin/python
backend=gpu
if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 ]]; then backend=cpu; fi
export JAX_PLATFORMS=$backend GPUWRF_FAST_DEFAULTS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$source_root/src
export XLA_FLAGS=--xla_cpu_max_isa=AVX
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p "$output_root"
run_root=$output_root/cache_runs/${backend}_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$run_root"
cache_enabled=1; [[ $backend == cpu ]] && cache_enabled=0
for option in 0 1; do
  arm_root=$run_root/pdf$option
  mkdir -p "$arm_root"/{jax,triton,cuda,xdg,tmp}
  command=(timeout 540 "$python" "$source_root/proofs/v45_mynnopts/gpu_operators.py"
         --backend "$backend" --option "$option" --cache-root "$arm_root"
         --out "$output_root/operators_${backend}_pdf${option}.json")
  clean_env=(env -i PATH="$PATH" HOME="$HOME" USER="${USER:-enric}"
           XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-}"
           LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" JAX_PLATFORMS="$backend" GPUWRF_FAST_DEFAULTS=1
           GPUWRF_JAX_CACHE="$cache_enabled" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source_root/src"
           JAX_COMPILATION_CACHE_DIR="$arm_root/jax" GPUWRF_JAX_CACHE_DIR="$arm_root/jax"
           TRITON_CACHE_DIR="$arm_root/triton" CUDA_CACHE_PATH="$arm_root/cuda"
           XDG_CACHE_HOME="$arm_root/xdg" TMPDIR="$arm_root/tmp" CUDA_CACHE_MAXSIZE=67108864
           GPUWRF_AOT_KEEP_TRACE_METADATA=1
           XLA_FLAGS=--xla_cpu_max_isa=AVX GPUWRF_GPU_LOCK_HELD="${GPUWRF_GPU_LOCK_HELD:-0}")
  if [[ $backend == cpu ]]; then
    "${clean_env[@]}" flock "$output_root/cpu_private.lock" "${command[@]}"
  else
    "${clean_env[@]}" <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label v45-mynnopts -- "${command[@]}"
  fi
done
