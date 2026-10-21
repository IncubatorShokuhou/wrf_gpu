#!/usr/bin/env bash
# w43-release: start disk_relief_v2.py (frozen tools_topup) once the RW3T top-up session's own bash process exists.
# The session PID is taken from /proc/<pid>/cmdline == "bash <tools_topup>/topup_session.sh" (not with_gpu_lock.sh, not this
# script: E141). Gives up at 18:05Z real UTC (slot end + 5 min) if the session never started.
set -u
T=<USER_HOME>/wrf_gpu2_lanes/w43-release/tools_topup; L=<USER_HOME>/wrf_gpu2_lanes/w43-release
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh; GIVEUP=$(date -u -d 2026-10-21T18:05:00Z +%s)
log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/relief_starter_topup.log"; }
log "armed pid $$"
while :; do
  for p in $(pgrep -x bash); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    if [[ "$c" == "bash $T/topup_session.sh " ]]; then
      log "session pid $p -> disk_relief_v2"
      exec env JAX_PLATFORMS=cpu taskset -c 24,28 nice -n 19 ionice -c3 python3 "$T/disk_relief_v2.py" "$p" >> "$L/disk_relief_topup.out" 2>&1
    fi
  done
  now=$("$REAL"); [[ $now =~ ^[0-9]+$ ]] && (( now > GIVEUP )) && { log "no session by 18:05Z, exit"; exit 0; }
  sleep 10
done
