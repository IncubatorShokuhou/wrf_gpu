#!/bin/bash
# ONE GPU arm per lock session (manager rule) for a w41-stoch case:
#   scripts/with_gpu_lock.sh --label w41-stoch -- proofs/w41_stoch/gpu_arm.sh <commit> <case>
# Immutable clean detached snapshot, release-default CLI (no flag file), 24 h, asserted GPU (gpu_case.py receipt).
# Environment = the v0.4.0 final-validation c00 arm (gpu_inside.sh): census ON, capped prealloc 0.15, private caches.
# E41 + D6 + pattern scoring run afterwards OUTSIDE the lock (gpu_post.sh). Cold estimate ~9-13 GPU-min (c00 521 s).
set -u
COMMIT=$1; c=$2
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/cases
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-stoch
REPO=<USER_HOME>/src/wrf_gpu2_wt/w41-stoch
SNAP=<DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/snap_$COMMIT
[ "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 ] || { echo "run under scripts/with_gpu_lock.sh"; exit 3; }
[ -d "$SNAP" ] || { echo "no snapshot $SNAP (create it on CPU first)"; exit 4; }
cd "$SNAP" || exit 4
[ -z "$(git status --porcelain)" ] || { echo "dirty snapshot"; exit 4; }
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "snapshot HEAD mismatch"; exit 4; }
C=$ROOT/$c
[ -d "$C/input" ] || { echo "$c: no input"; exit 4; }
[ -e "$C/gpu" ] && { echo "$c: gpu dir exists, refusing to overwrite (E42)"; exit 4; }
K=$LANE/cache_$c
mkdir -p "$K/jax" "$K/autotune" "$K/triton" "$K/cuda" "$K/tmp"
t0=$(date +%s)
echo "$(date -u +%FT%TZ) w41-stoch: GPU START $c (commit $COMMIT)" >> "$LANE/gpu_status"
env JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 \
  GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 GPUWRF_JAX_CACHE=1 \
  GPUWRF_JAX_CACHE_DIR="$K/jax" JAX_COMPILATION_CACHE_DIR="$K/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$K/autotune" \
  TRITON_CACHE_DIR="$K/triton" CUDA_CACHE_PATH="$K/cuda" TMPDIR="$K/tmp" \
  XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.15 \
  GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=15 "${ARM_TIMEOUT:-1800}" python proofs/w41_stoch/gpu_case.py "$C" 24 > "$C/gpu.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) w41-stoch: GPU END $c rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ls "$C"/gpu/wrfout_* 2>/dev/null | wc -l)" | tee -a "$LANE/gpu_status"
exit $rc
