#!/bin/bash
# ONE GPU lock session (manager 22:30:53Z / 22:32:49Z, <= 60 GPU-min) for lane w41-shinhongtke:
#   scripts/with_gpu_lock.sh --label w41-shinhongtke -- proofs/w41_shinhongtke/gpu_session.sh <commit>
# Arms in order (release-default CLI, asserted GPU, immutable detached snapshot, own 20G memcg each,
# ONE shared JAX/autotune cache): c03 MYJ 24 h, c06 BouLac 24 h (cold TKE seed changes initial data),
# c07sd1 Shin-Hong shinhong_tke_diag=1 24 h, c28 CAM-UW 6 h (history-only change) if >= 14 min remain.
# E41 + D6 + TKE compare afterwards OUTSIDE the lock: post.sh <commit> <case>.
set -u
COMMIT=$1; BUDGET_S=${BUDGET_S:-3540}
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/w41-shinhongtke/cases
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-shinhongtke
SNAP=$LANE/snap_$COMMIT
[ "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 ] || { echo "run under scripts/with_gpu_lock.sh"; exit 3; }
[ -e /tmp/wrf_gpu2_run_approvals/w41-shinhongtke.lease ] || { echo "no manager lease"; exit 3; }
[ -d "$SNAP" ] || { echo "create the snapshot outside the lock first"; exit 4; }
cd "$SNAP" || exit 4
[ -z "$(git status --porcelain)" ] || { echo "dirty snapshot"; exit 4; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$COMMIT")" ] || { echo "snapshot != commit"; exit 4; }
export GPUWRF_JAX_CACHE_DIR=$LANE/cache_w41sht JAX_COMPILATION_CACHE_DIR=$LANE/cache_w41sht GPUWRF_CENSUS=0
start=$(date +%s)
for spec in c03:24:1500 c06:24:1000 c07sd1:24:1400 c28:6:900; do
  c=${spec%%:*}; rest=${spec#*:}; hours=${rest%%:*}; cap=${rest#*:}
  left=$(( BUDGET_S - ($(date +%s) - start) ))
  if [ "$left" -lt 840 ] || { [ "$c" != c28 ] && [ "$left" -lt 600 ]; }; then
    echo "$(date -u +%FT%TZ) $c SKIPPED budget left=${left}s" | tee -a "$LANE/gpu_status"; continue
  fi
  [ "$cap" -gt "$left" ] && cap=$left
  C=$ROOT/$c
  [ -d "$C/input" ] || { echo "$c: no input"; continue; }
  [ -e "$C/gpu" ] && { echo "$c: gpu dir exists (E42)"; continue; }
  t0=$(date +%s)
  echo "$(date -u +%FT%TZ) $c GPU START commit=$COMMIT hours=$hours cap=${cap}s" >> "$LANE/gpu_status"
  systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- \
    timeout "$cap" python proofs/w41_shinhongtke/gpu_case.py "$C" "$hours" > "$C/gpu.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $c GPU END rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ls "$C"/gpu/wrfout_* 2>/dev/null | wc -l)" | tee -a "$LANE/gpu_status"
done
echo "$(date -u +%FT%TZ) session GPU END total=$(( $(date +%s) - start ))s" | tee -a "$LANE/gpu_status"
