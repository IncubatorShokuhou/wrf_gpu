#!/usr/bin/env bash
# w41-bundle041: start session.sh under the official GPU lock once w41-bdyrec posts its B1 'GPU END' (manager 11:15:41Z:
# anchor on the sender at line start). Only MAILBOX lines appended AFTER arming count (E77/E141: never our own lines).
# usage: trigger.sh           (logs to $S/trigger.log; gives up with a NOT STARTED line at BS_GIVEUP_UTC, default 17:00Z)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh; LEASE=/tmp/wrf_gpu2_run_approvals/w41-bundle041.lease
N0=$(wc -l < "$M"); GIVEUP=$(date -u -d "${BS_GIVEUP_UTC:-2026-10-18T17:00:00Z}" +%s)
RE='^20[0-9T:.+Z-]+ w41-bdyrec( -> [^:]*)?: GPU END'
log() { echo "$(date -u +%FT%TZ) $*" >> "$S/trigger.log"; }
post() { printf '%s w41-bundle041: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$M"; }
log "armed (tools $HERE), MAILBOX lines > $N0, pattern /$RE/, give up $(date -u -d @$GIVEUP +%FT%TZ)"
while :; do
  hit=$(tail -n +"$((N0 + 1))" "$M" | /usr/bin/grep -E -m1 "$RE")
  [ -n "$hit" ] && break
  (( $(date +%s) < GIVEUP )) || { log "give up"; post "NOT STARTED: no w41-bdyrec 'GPU END' by $(date -u -d @$GIVEUP +%TZ); session not started (manager re-slots)"; exit 9; }
  sleep 30
done
log "trigger line: ${hit:0:160}"
sleep 20
exp=$(sed -n 's/^expires-utc: //p' "$LEASE" 2>/dev/null)
[[ -n "$exp" ]] && (( $(date -u -d "$exp" +%s) - $(date +%s) >= 3600 )) || { log "lease missing/short ($exp)"; post "NOT STARTED: lease $LEASE missing or < 60 min left ($exp)"; exit 8; }
while [ -e /tmp/wrf_gpu2_quiet ]; do log "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; sleep 30; done
[[ $(sha256sum "$LOCK" | awk '{print $1}') = $(awk '{print $1}' <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sha256) ]] || { log "lock script sha mismatch"; exit 126; }
log "taking the lock"
"$LOCK" --label w41-bundle041 --timeout 3600 -- bash "$HERE/session.sh"
rc=$?
log "session rc=$rc"
(( rc == 124 )) && post "NOT STARTED: GPU lock not acquired within 60 min (rc 124)"
(( rc >= 2 && rc <= 6 )) && post "session refused/aborted early rc=$rc (see $S/session.log)"
exit $rc
