#!/bin/bash
# Manager DECISION 2026-10-17T18:01:57Z (a): C7 EEPS CPU twin tonight on lane cores 8,9,12 (3 ranks, one per physical
# core; 13 = GPU-lock holder core, kept free), nice 19, RAM slot 10G, start ONLY after the manager's NIGHTLY DONE,
# must end by 2026-10-18T08:30Z (run_twin watchdog). Attempt a2 (a1 started wrongly at 18:06:46Z on a loose match of
# 'NIGHTLY DONE' inside another lane's line, killed at once, registry ABORTED).
# Start condition (all): (1) a MAILBOX line appended after this script started whose MESSAGE BEGINS with
# 'NIGHTLY DONE' from the manager (regex below), or the manual file <lane>/C7_GO that I create only after reading the
# manager's DONE line; (2) /tmp/wrf_gpu2_quiet absent; (3) RAM slots file not 0. Not started if the predicted end
# (520 s/sim-h * 24 * 1.03 + 240 s) is past 08:30Z (reported in MAILBOX).
set -u
HERE=$(dirname "$(readlink -f "$0")"); LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; SCR=<DATA_ROOT>/wrf_gpu2/scratch/w41-nestqual
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; DEADLINE_ISO=2026-10-18T08:30:00Z; CORES=8,9,12
DONE_RE='^20[0-9-]+T[0-9:]+Z manager( -> [^:]+)?: +\**NIGHTLY DONE'
DEADLINE=$(date -u -d "$DEADLINE_ISO" +%s); N0=$(wc -l < "$MAILBOX"); T_START=$(date -u +%FT%TZ)
taskset -pc "$CORES" $$ > /dev/null
log(){ echo "$(date -u +%FT%TZ) c7_tonight $*" | tee -a "$LANE/twin_status.log"; }
slots(){ cat /tmp/wrf_gpu2_ram_slots/slots 2>/dev/null || echo 2; }
log "a2 waiting for a line matching /$DONE_RE/ after MAILBOX line $N0 (or $LANE/C7_GO), QUIET absent, RAM slots > 0"
until { tail -n +$((N0 + 1)) "$MAILBOX" | grep -qE "$DONE_RE" || [ -e "$LANE/C7_GO" ]; } \
      && [ ! -e /tmp/wrf_gpu2_quiet ] && [ "$(slots)" != 0 ]; do
  if (( $(date +%s) >= DEADLINE - 3600 )); then
    log "no NIGHTLY DONE before $(date -u -d @$((DEADLINE-3600)) +%TZ): C7 not started"
    echo "$(date -u +%FT%TZ) w41-nestqual: C7 EEPS twin NOT started (no NIGHTLY DONE/start condition in time); window chain C2/C3/C4 unchanged" >> "$MAILBOX"; exit 3
  fi
  sleep 60
done
log "start condition met: $(tail -n +$((N0 + 1)) "$MAILBOX" | grep -m1 -E "$DONE_RE" | cut -c1-160) GO=$([ -e "$LANE/C7_GO" ] && echo yes || echo no) slots=$(slots)"
NOW=$(date +%s); EST=$(( 520 * 24 * 103 / 100 + 240 ))
if (( NOW + EST > DEADLINE )); then
  log "C7 NOT started: predicted end $(date -u -d @$((NOW+EST)) +%FT%TZ) > $DEADLINE_ISO"
  echo "$(date -u +%FT%TZ) w41-nestqual: C7 EEPS twin NOT started: predicted end $(date -u -d @$((NOW+EST)) +%TZ) > 08:30Z deadline (rate 520 s/sim-h); C7 stays nested-experimental unless rescheduled" >> "$MAILBOX"; exit 3
fi
CASE=$SCR/cases/20260707_18z_C7
[ -s "$CASE/PREPARE.json" ] || python3 "$HERE/build_twin.py" "$LANE/guard/stage_20260707_18z/STAGE.json" "$SCR/cases" C7 \
  --hours 24 --ranks 3 --tables "$SCR/tables" > "$LANE/build_C7.log" 2>&1 || { log "build C7 FAILED"; exit 4; }
TWIN_ATTEMPT=a2 "$HERE/run_twin.sh" "$CASE" "$CORES" "$DEADLINE_ISO" live 24 10G > "$LANE/run_C7.out" 2>&1
log "C7 run_twin rc=$?"
