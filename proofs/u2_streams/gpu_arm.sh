#!/usr/bin/env bash
set -euo pipefail
umask 077
# Manager grants <=15min only; queue until18:45Z, finish <=19:15Z.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if JAX_PLATFORMS=cpu git symbolic-ref --quiet HEAD >/dev/null; then echo 'Use detached immutable snapshot'; exit 2; fi
if ! JAX_PLATFORMS=cpu git diff --quiet; then echo 'Dirty snapshot'; exit 2; fi
export STREAM_SOURCE_COMMIT="$(JAX_PLATFORMS=cpu git rev-parse HEAD)"
export STREAM_GPU_PREFIX=<DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
export PYTHONPATH="$PWD/src"
export GPUWRF_JAX_CACHE=1
export GPUWRF_CENSUS=1
export GPUWRF_CENSUS_SLIM=0
export GPUWRF_JAX_CACHE_DIR=<DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_cache
export JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_cache
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR=<DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_autotune
export TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_tmp
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$TMPDIR"
queue_seconds=$(( $(date -u -d '2026-10-13 18:45:00Z' +%s) - $(date -u +%s) ))
if (( queue_seconds <= 0 )); then echo 'Defer GPU arm to tomorrow11:30Z'; exit 3; fi
env -i HOME="$HOME" USER="${USER:-enric}" PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
 PYTHONPATH="$PYTHONPATH" OMP_NUM_THREADS=4 GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 \
 GPUWRF_JAX_CACHE=1 GPUWRF_JAX_CACHE_DIR="$GPUWRF_JAX_CACHE_DIR" \
 JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$GPUWRF_XLA_AUTOTUNE_CACHE_DIR" \
 TMPDIR="$TMPDIR" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false STREAM_SOURCE_COMMIT="$STREAM_SOURCE_COMMIT" \
 scripts/with_gpu_lock.sh --timeout "$queue_seconds" --label u2-streams -- \
 timeout 900 bash -c 'if (( $(date -u +%s) >= $(date -u -d "2026-10-13 18:45:00Z" +%s) )); then echo "Late lock; defer"; exit 3; fi; exec nsys profile --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -o <DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_trace python proofs/u2_streams/gpu_arm.py <DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU'
# Cache retained until complete E41+transfer audit; no premature cleanup.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8 nice -n 19 timeout 900 python \
 scripts/wave_cubin_gate.py <DATA_ROOT>/wrf_gpu2/scratch/u2-streams/US_GPU_cache <USER_HOME>/wrf_gpu2_lanes/u2-streams/US_GPU_cubin
