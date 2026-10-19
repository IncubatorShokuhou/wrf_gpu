#!/usr/bin/env bash
# w42-bundle042: run the frozen CPU follow-up (F/followup.sh of the main bundle session) QUIET-aware, so it can overlap the
# separate HERO lock session: its WHOLE process group is SIGSTOPped while ANY /tmp/wrf_gpu2_quiet exists (the hero's
# per-arm QUIET or a foreign one) and SIGCONTed when it is gone (1 s poll; successor of the v0.4.1 quiet_pause.sh, which
# paused one PID at 10 s). Lane cores 8,9,12, nice 19, ONE RAM slot <= 12G (followup.sh header). Stopped time counts against
# the step timeouts inside followup.sh (E41 3600 s, D6 10800 s, icloud0 5400 s per domain; v0.4.1 actuals 1.5 / 10-20 min)
# -> a hero session (<= 30 min of QUIET) keeps every step inside its limit. followup.sh skips finished steps on a re-run.
# usage: setsid bash followup_quiet.sh F_DIR [GPU_END_FROM_LINE]
#   GPU_END_FROM_LINE: start only once a MAILBOX line '^<ts> w42-bundle042: GPU END' exists after that line (main session
#   done, lock released); without it start at once. Log: $L/followup_quiet.log (STOP/CONT with the QUIET label), the
#   follow-up's own output $L/followup_run.log (+ $L/followup/followup.log by followup.sh).
set -uo pipefail
F=$(cd "$1" && pwd); FROM=${2:-}
L=${FQ_LOGDIR:-<USER_HOME>/wrf_gpu2_lanes/w42-bundle042}; LANE=<DATA_ROOT>/wrf_gpu2/worktrees/w42-bundle042
MB=${FQ_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; Q=${FQ_QUIET_FILE:-/tmp/wrf_gpu2_quiet}   # FQ_*: dry tests only
log() { echo "$(date -u +%FT%TZ) $*" >> "$L/followup_quiet.log"; }
[ -s "$F/followup.sh" ] && [ -s "$F/pin.env" ] || { log "no followup.sh/pin.env in $F"; exit 2; }
log "armed: F=$F gpu_end_from=${FROM:-none}"
if [ -n "$FROM" ]; then
  until tail -n +$((FROM + 1)) "$MB" | /usr/bin/grep -E '^20[0-9T:.+Z-]+ w42-bundle042: GPU END' > /dev/null; do sleep 30; done
  log "GPU END seen"
fi
while [ -e "$Q" ]; do sleep 5; done
rm -f "$L/followup.pgid"
setsid bash -c "echo \$\$ > '$L/followup.pgid'; exec taskset -c 8,9,12 nice -n 19 '$LANE/scripts/with_ram_slot.sh' \
  --label w42-bundle042-fu --max 12G --timeout 21600 -- bash '$F/followup.sh'" >> "$L/followup_run.log" 2>&1 < /dev/null &
for _ in $(seq 1 20); do [ -s "$L/followup.pgid" ] && break; sleep 0.5; done   # E131: the pid file holds the session leader = pgid
pg=$(cat "$L/followup.pgid"); state=run
log "started followup.sh pgid $pg"
while kill -0 -- "-$pg" 2>/dev/null; do
  if [ -e "$Q" ]; then
    [ $state = run ] && { kill -STOP -- "-$pg"; state=stopped; log "STOP ($(cat "$Q" 2>/dev/null))"; }
  else
    [ $state = stopped ] && { kill -CONT -- "-$pg"; state=run; log "CONT"; }
  fi
  sleep 1
done
log "followup group $pg gone: $(tail -n 1 "$L/followup/followup.log" 2>/dev/null)"
