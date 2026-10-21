#!/usr/bin/env bash
# w43-release TOP-UP waiter (copy of waiter.sh; runs topup_session.sh, log waiter_topup.log)
# w43-release GPU waiter (V043 pattern, cf. w43-teacher waiter.sh): sleep until REAL UTC GA_START_UTC (+5 s) via real_utc.sh,
# then ONE scripts/with_gpu_lock.sh --label w43-release session of gpu_session.sh (frozen tools dir = this dir).
# Lock-wait timeout = (GA_END - GA_MIN_S - real_now) x 0.9 on the system clock (gpu_session.sh re-checks: rc4/rc6).
# usage: setsid bash waiter.sh GA_START_UTC GA_END_UTC   (posts nothing itself; gpu_session.sh posts GPU START/END)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
export GA_START_UTC=$1 GA_END_UTC=$2 GA_MIN_S=${GA_MIN_S:-1800}
S=$(date -u -d "$GA_START_UTC" +%s); E=$(date -u -d "$GA_END_UTC" +%s)
L=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/waiter_topup.log
log(){ echo "system $(date -u +%FT%TZ) real $("$REAL" --iso) $*" >> "$L"; }
rn(){ local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] && echo "$e" || echo 0; }
log "armed pid $$ window $GA_START_UTC .. $GA_END_UTC tools $HERE"
while (( $(rn) < S + 5 )); do sleep 20; done
W=$(( (E - GA_MIN_S - $(rn)) * 9 / 10 )); (( W > 60 )) || { log "no time left (W=$W)"; exit 6; }
log "launch, lock wait ${W}s"
<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w43-release --timeout "$W" -- bash "$HERE/topup_session.sh" >> "$L" 2>&1
log "done rc=$?"
