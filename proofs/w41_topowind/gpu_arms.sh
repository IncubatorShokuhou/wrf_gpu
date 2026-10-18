#!/bin/bash
# ONE GPU arm per lock session (manager rule) for a topo_wind composite case (lane w41-topowind):
#   scripts/with_gpu_lock.sh --label w41-topowind -- proofs/w41_topowind/gpu_arms.sh <commit> <case>
# Cases (CPU-WRF references in <case>/cpu): c02tw1 (YSU tw=1), c07tw1 (Shin-Hong tw=1), c02tw2 (YSU tw=2),
# wn3tw (WN3 0227 3-nest YSU tw=2,1,1). Immutable detached snapshot, release-default CLI (no flag file),
# 24 h, asserted GPU (gpu_case.py receipt). E41 + D6 run afterwards OUTSIDE the lock: gpu_post.sh.
# Estimated GPU minutes (cold): Swiss ~17 each (v040 C07 24 h cold 994 s), WN3 ~25.
set -u
COMMIT=$1; c=$2
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/cases
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-topowind
REPO=<USER_HOME>/src/wrf_gpu2_wt/w41-topowind
SNAP=$LANE/snap_$COMMIT
[ "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 ] || { echo "run under scripts/with_gpu_lock.sh"; exit 3; }
if [ ! -d "$SNAP" ]; then
  git -C "$REPO" worktree add --detach "$SNAP" "$COMMIT" || exit 4
  ln -s <USER_HOME>/src/wrf_pristine "$SNAP/data/wrf_pristine"
fi
cd "$SNAP" || exit 4
[ -z "$(git status --porcelain)" ] || { echo "dirty snapshot"; exit 4; }
C=$ROOT/$c
[ -d "$C/input" ] || { echo "$c: no input"; exit 4; }
[ -e "$C/gpu" ] && { echo "$c: gpu dir exists, refusing to overwrite (E42)"; exit 4; }
export GPUWRF_JAX_CACHE_DIR=$LANE/cache_$c JAX_COMPILATION_CACHE_DIR=$LANE/cache_$c GPUWRF_CENSUS=0
t0=$(date +%s)
echo "$(date -u +%FT%TZ) $c GPU START commit=$COMMIT" >> "$LANE/gpu_status"
# Own host-memory cap inside the lock (manager 11:37Z: never with_ram_slot around/inside the GPU lock).
systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- \
  timeout "${ARM_TIMEOUT:-3000}" python proofs/w41_topowind/gpu_case.py "$C" "${HOURS:-24}" > "$C/gpu.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) $c GPU END rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ls "$C"/gpu/wrfout_* 2>/dev/null | wc -l)" | tee -a "$LANE/gpu_status"
exit $rc
