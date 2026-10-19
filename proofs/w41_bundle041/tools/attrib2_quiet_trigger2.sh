#!/usr/bin/env bash
# w41-bundle041 ATTRIB2 last try (manager 2026-10-18T17:16:32Z): quiet-trigger, then the SAME frozen session/rules.
# Detector = the frozen tools_c9be5bcd32/core_watch.py at PERIOD 5 s (never aborts: limits 1e9), own root = this trigger:
# start only after NEED_S (120 s) of CONTINUOUS 5 s intervals whose foreign sum on CORES is < 20 % (kernel threads apart,
# as in the session watcher) and no /tmp/wrf_gpu2_quiet. Latest start LATEST_UTC (17:55Z); session window end
# = min(start + 1560 s, END_CAP 18:35Z); a manager HOLD/STOP line to w41-bundle041 after MAILBOX line FROM cancels.
# usage: attrib2_quiet_trigger2.sh ROOT FROM [NEED_S=120] [LATEST_UTC=2026-10-18T17:55:00Z] [CORES=3,13,19,29]
#        DRYLAUNCH=1 -> log 'would launch' instead of taking the lock (CPU test of the detector).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$1 FROM=$2 NEED_S=${3:-120} LATEST=$(date -u -d "${4:-2026-10-18T17:55:00Z}" +%s) CORES=${5:-3,13,19,29}
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; LOG=${TRIG_LOG:-$S/attrib2c_trigger.log}; MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
TOOLS=<USER_HOME>/wrf_gpu2_lanes/w41-bundle041/tools_c9be5bcd32; WT=<USER_HOME>/src/wrf_gpu2_wt/w41-bundle041
END_CAP=$(date -u -d 2026-10-18T18:35:00Z +%s); NEED=$(( NEED_S / 5 ))
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
[ -e "$ROOT/session.log" ] && { log "root $ROOT used: refuse"; exit 2; }
DET=$(mktemp -d "${TMPDIR:-/tmp}/w41trig2.XXXX")
taskset -c 28 nice -n 5 python3 "$TOOLS/core_watch.py" "$CORES" 20 1e9 1e9 "$$" "$DET/w.jsonl" "$DET/s.json" "$DET/ABORT" 5 \
  > "$DET/w.log" 2>&1 &
DPID=$!
cleanup() { kill -TERM "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; cp "$DET/w.jsonl" "${LOG%.log}_detector.jsonl" 2>/dev/null
  rm -f "$DET/w.jsonl" "$DET/s.json" "$DET/s.json.tmp" "$DET/w.log" "$DET/ABORT"; rmdir "$DET" 2>/dev/null; }
trap cleanup EXIT
log "armed: root $ROOT, need ${NEED_S}s continuous (${NEED} x 5 s) foreign < 20 % on $CORES, latest start $(date -u -d @$LATEST +%TZ), MAILBOX > $FROM, detector PID $DPID"
last=0
while :; do
  (( $(date +%s) >= LATEST )) && { log "NOT STARTED: no ${NEED_S}s quiet stretch before $(date -u -d @$LATEST +%TZ)"; exit 9; }
  kill -0 "$DPID" 2>/dev/null || { log "detector died: refuse"; exit 3; }
  if tail -n +$((FROM + 1)) "$MB" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> [^:]*w41-bundle041[^:]*:.*\b(HOLD|STOP)\b'; then
    log "manager HOLD/STOP line: cancelled"; exit 8; fi
  n=$(python3 - "$DET/w.jsonl" "$NEED" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]).read().splitlines() if x.strip()] if __import__("os").path.exists(sys.argv[1]) else []
run = 0
for r in reversed(rows):           # trailing run of quiet 5 s intervals (dt sanity: <= 7 s, no gap)
    if r["foreign_agg"] < 20.0 and r["dt"] <= 7.0:
        run += 1
    else:
        break
print(run)
PY
)
  if (( n != last )) && (( n == 0 )); then log "quiet run reset ($(tail -1 "$DET/w.jsonl" | cut -c1-200))"; fi
  last=$n
  if (( n >= NEED )) && [ ! -e /tmp/wrf_gpu2_quiet ]; then break; fi
  sleep 5
done
now=$(date +%s); WEND=$(( now + 1560 )); (( WEND > END_CAP )) && WEND=$END_CAP
log "quiet ${NEED_S}s reached ($n intervals) -> launch, window end $(date -u -d @$WEND +%FT%TZ)"
cleanup; trap - EXIT
if [ -n "${DRYLAUNCH:-}" ]; then log "DRYLAUNCH: would launch $TOOLS/attrib2.sh root $ROOT"; exit 0; fi
cd "$WT" || exit 2
AT_ROOT=$ROOT AT_WINDOW_END_UTC=$(date -u -d @$WEND +%FT%TZ) scripts/with_gpu_lock.sh --label w41-bundle041 --timeout 900 -- \
  bash "$TOOLS/attrib2.sh" >> "$S/attrib2c_launch.log" 2>&1
log "session rc=$?"
