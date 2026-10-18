#!/usr/bin/env bash
# w41-bundle041 ATTRIB2 re-arm (manager GO 2026-10-18T16:17:11Z; first root attrib2 burnt by the owner e2 sweep, TERM 16:26:15Z).
# Every 10 s: e2_quiet.py (e2-generator/ngspice CPU <= 20 % over 5 s) AND cores_free.py 3,13,19,29 (20 %, 5 s) AND no
# /tmp/wrf_gpu2_quiet. After NEED consecutive quiet probes -> ONE lock session of the FROZEN tools_c9be5bcd32/attrib2.sh in a
# FRESH root (AT_ROOT), window end = min(start + 1560 s, 18:30Z). Gives up at GIVEUP_UTC (default 17:55Z). A manager line
# addressed to w41-bundle041 containing HOLD or STOP after MAILBOX line FROM cancels it. Log $LOG.
# usage: attrib2_quiet_trigger.sh ROOT MAILBOX_FROM_LINE [NEED=15] [GIVEUP_UTC=2026-10-18T17:55:00Z]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$1 FROM=$2 NEED=${3:-15} GIVEUP=$(date -u -d "${4:-2026-10-18T17:55:00Z}" +%s)
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; LOG=$S/attrib2_trigger.log; MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
TOOLS=<USER_HOME>/wrf_gpu2_lanes/w41-bundle041/tools_c9be5bcd32; WT=<USER_HOME>/src/wrf_gpu2_wt/w41-bundle041
END_CAP=$(date -u -d 2026-10-18T18:30:00Z +%s)
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
[ -e "$ROOT/session.log" ] && { log "root $ROOT used: refuse"; exit 2; }
log "armed: root $ROOT need $NEED quiet probes, give up $(date -u -d @$GIVEUP +%TZ), MAILBOX > $FROM"
n=0
while :; do
  (( $(date +%s) >= GIVEUP )) && { log "give up"; exit 9; }
  if tail -n +$((FROM + 1)) "$MB" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> [^:]*w41-bundle041[^:]*:.*\b(HOLD|STOP)\b'; then
    log "manager HOLD/STOP line: cancelled"; exit 8; fi
  q=$(python3 "$HERE/e2_quiet.py" 5 20); qr=$?
  python3 "$HERE/cores_free.py" 3,13,19,29 20 "$S/attrib2_trigger_cores.jsonl" "$$" 5 > /dev/null 2>&1; cr=$?
  if (( qr == 0 && cr == 0 )) && [ ! -e /tmp/wrf_gpu2_quiet ]; then n=$((n + 1)); else
    (( n > 0 )) && log "quiet streak broken at $n ($q cores_rc=$cr)"; n=0; fi
  (( n >= NEED )) && break
  sleep 10
done
now=$(date +%s); WEND=$(( now + 1560 )); (( WEND > END_CAP )) && WEND=$END_CAP
(( WEND - now >= 1200 )) || { log "window too short ($(( WEND - now )) s): give up"; exit 9; }
log "quiet for $n probes -> launch, window end $(date -u -d @$WEND +%FT%TZ)"
cd "$WT" || exit 2
AT_ROOT=$ROOT AT_WINDOW_END_UTC=$(date -u -d @$WEND +%FT%TZ) scripts/with_gpu_lock.sh --label w41-bundle041 --timeout 900 -- \
  bash "$TOOLS/attrib2.sh" >> "$S/attrib2_launch.log" 2>&1
log "session rc=$?"
