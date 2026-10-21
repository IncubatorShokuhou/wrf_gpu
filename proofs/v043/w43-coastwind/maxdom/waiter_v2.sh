#!/bin/bash
# w43-coastwind GPU waiter (real UTC): start at 2026-10-21T08:30:05Z real, lock wait until 08:48:00Z real at most, then gpu_maxdom_v2.sh.
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
S=$(date -u -d 2026-10-21T08:30:05Z +%s); LOCK_BY=$(date -u -d 2026-10-21T08:48:00Z +%s)
while true; do e=$($RU); [[ $e =~ ^[0-9]+$ ]] && (( e >= S )) && break; sleep 20; done
e=$($RU); [[ $e =~ ^[0-9]+$ ]] || exit 7
W=$(( (LOCK_BY - e) * 9 / 10 )); (( W > 0 )) || { echo "lock deadline passed (real $e)"; exit 6; }
cd <DATA_ROOT>/wrf_gpu2/worktrees/w43-coastwind
GA_START_UTC=2026-10-21T08:30:00Z GA_END_UTC=2026-10-21T09:18:00Z exec <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w43-coastwind --timeout $W -- bash analysis/maxdom/gpu_maxdom_v2.sh
