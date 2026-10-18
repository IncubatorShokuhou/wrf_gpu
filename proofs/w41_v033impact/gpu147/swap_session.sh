#!/usr/bin/env bash
# w41-v033impact #147b — IMMUTABLE GPU session: SAME executable (lifted fused d02/d03 program, v0.3.3 86e6303702), 0707 carry,
# 200 chained parent substeps; variants own / own repeat / foreign(0227) TMN / foreign(0227) aux4 series + shdfac. No d01
# force-down (sensitivity test, not a forecast). Answers at cell level which outputs a foreign baked constant reaches.
set -u
export LC_ALL=C
L=<USER_HOME>/wrf_gpu2_lanes/w41-v033impact
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact
SNAP=$L/snap_v033
G=$S/gpu147/swap
DEADLINE=$(date -u -d "2026-10-18T12:50:00Z" +%s)
mkdir -p "$G"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$G/session.log"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-}" == 1 ]] || { log "REFUSE: lock"; exit 3; }
[[ "$(git -C "$SNAP" rev-parse HEAD)" == 86e63037020392df38c3f600a9d903e258e3e9d4 && -z "$(git -C "$SNAP" status --porcelain)" ]] || { log "REFUSE: snapshot"; exit 3; }
(( $(date +%s) + 1500 <= DEADLINE )) || { log "SKIP: would pass 12:50Z"; exit 9; }
log "swap start label=${GPUWRF_GPU_LOCK_LABEL:-?}"
cd "$SNAP" || exit 3
env GPUWRF_JAX_CACHE=0 timeout -k 60 1500 python "$L/gpu_swap.py" "$SNAP" "$G" "$S/cases/g2_20260707" \
  --foreign="$L/probe1/leaves_wn3_0227.npz" --steps=200 > "$G/swap.log" 2>&1
log "swap end rc=$?"
