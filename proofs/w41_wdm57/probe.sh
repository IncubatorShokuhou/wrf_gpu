#!/usr/bin/env bash
# CPU-only full actual c10/c18 compilation with per-ptxas wall/RSS sampling.
set -euo pipefail
case "$1/$2" in before/c10|before/c18|after/c10|after/c18) ;; *) exit 2;; esac
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/baseline
[[ $1 == after ]] && tree=$lane/candidate
label=$1-$2
mkdir -p "$lane/$label-work"/{tmp,jax,triton,cuda,autotune,dump}
exec env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES= \
  GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 \
  GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 PYTHONDONTWRITEBYTECODE=1 \
  JAX_TRACEBACK_FILTERING=off TMPDIR="$lane/$label-work/tmp" \
  JAX_COMPILATION_CACHE_DIR="$lane/$label-work/jax" \
  TRITON_CACHE_DIR="$lane/$label-work/triton" CUDA_CACHE_PATH="$lane/$label-work/cuda" \
  GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$lane/$label-work/autotune" \
  XLA_FLAGS="--xla_dump_to=$lane/$label-work/dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*advance_chunk.*" \
  taskset -c 24,25,28,29 nice -n 19 python \
  <USER_HOME>/src/wrf_gpu2_wt/w41-wdm57/proofs/w41_wdm57/deviceless.py \
  --source "$tree" --case "<USER_HOME>/wrf_gpu2_lanes/v-prep/$2" \
  --out "$lane/$label" --target "<USER_HOME>/wrf_gpu2_lanes/v-prep/$2/target_config.pbtxt" \
  --dump "$lane/$label-work/dump" --ptxas-limit-s 300
