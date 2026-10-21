#!/usr/bin/env bash
# w43-teacher GPU waiter (manager GPU SLOTS 21:30:44Z): sleep until REAL UTC GA_START_UTC (+5 s) via real_utc.sh, then ONE
# scripts/with_gpu_lock.sh --label w43-teacher session of gpu_arms.sh batch1. Lock-wait timeout = (GA_END - 45 min - real_now) x 0.9
# on the system clock, so the session can never start with < 45 min left (gpu_arms.sh re-checks: rc4/rc6).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
export GA_START_UTC=2026-10-21T09:20:00Z GA_END_UTC=2026-10-21T11:30:00Z
S=$(date -u -d "$GA_START_UTC" +%s); E=$(date -u -d "$GA_END_UTC" +%s)
L=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/waiter.log
log(){ echo "system $(date -u +%FT%TZ) real $("$REAL" --iso) $*" >> "$L"; }
rn(){ local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] && echo "$e" || echo 0; }
log "armed pid $$"
while (( $(rn) < S + 5 )); do sleep 20; done
W=$(( (E - 2700 - $(rn)) * 9 / 10 )); (( W > 60 )) || { log "no time left (W=$W)"; exit 6; }
log "launch, lock wait ${W}s"
<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w43-teacher --timeout "$W" -- bash "$HERE/gpu_arms.sh" "$HERE/batch1_cases.txt" >> "$L" 2>&1
log "done rc=$?"
