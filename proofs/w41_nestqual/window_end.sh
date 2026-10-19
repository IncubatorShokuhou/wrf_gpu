#!/bin/bash
# Free the CPU window cores on time (lane w41-nestqual). Manager CORE PLAN 2026-10-18T11:30:13Z: our cores are
# 0,1,2 (+16,17,18) until 18:27Z; from 18:30Z they belong to w41-swddirscreen's C 0420 twin (ALISIOS loan).
# At <t_stop> (18:26Z): if <lane>/POST_CORES holds a manager-granted core list, every lane helper (GPU-output pollers,
# stage_gpu.py, scorer waiters, score.py, pin_keeper, run_queue) is moved there (all threads) and left running.
# Otherwise running stage_gpu.py/score.py get until <t_kill> (18:29Z) to finish, then every remaining process of ours
# whose affinity lies inside 0,1,2,16,17,18 is stopped by PID (TERM, then KILL); a wrf.exe still running from a case
# of ours (twin watchdog fires at 18:27Z) is stopped via its session. Tool shells (affinity 0-31 / OS cores) are never
# selected. One MAILBOX line reports the outcome; GPU staging + D6 of unscored arms resume only under a new grant.
# usage: window_end.sh <t_stop ISO> <t_kill ISO>
set -u
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; CASES=<DATA_ROOT>/wrf_gpu2/scratch/w41-nestqual/cases
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; WIN=",0,1,2,16,17,18,"
STOP=$(date -u -d "$1" +%s) || exit 2; KILL=$(date -u -d "$2" +%s) || exit 2
taskset -pc 14,15,30,31 $$ > /dev/null    # this sleeper is agent bookkeeping, never on the window cores
log(){ echo "$(date -u +%FT%TZ) window_end $*" | tee -a "$LANE/twin_status.log"; }
inwin(){  # affinity of $1 non-empty and entirely inside the window cores
  local a; a=$(taskset -pc "$1" 2>/dev/null | awk -F': ' '{print $2}') || return 1; [ -n "$a" ] || return 1
  local c; for c in $(echo "$a" | tr ',' ' '); do
    case "$c" in *-*) for x in $(seq "${c%-*}" "${c#*-}"); do [[ $WIN == *",$x,"* ]] || return 1; done;;
                 *) [[ $WIN == *",$c,"* ]] || return 1;; esac; done; return 0; }
ours(){ pgrep -u "$(id -u)" -f 'w41-nestqual|w41_nestqual' | while read -r p; do
          [ "$p" = $$ ] || [ "$p" = $PPID ] && continue; inwin "$p" && echo "$p"; done; }
heavy(){ for p in $(ours); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | /usr/bin/grep -qE 'stage_gpu\.py|final_validation/score\.py' && echo "$p"; done; }
while (( $(date +%s) < STOP )); do sleep 20; done
if [ -s "$LANE/POST_CORES" ]; then
  TO=$(tr -d ' \n' < "$LANE/POST_CORES")
  n=0; for p in $(ours); do taskset -apc "$TO" "$p" > /dev/null 2>&1 && n=$((n+1)); done
  log "moved $n lane helper processes to granted POST_CORES $TO"
  echo "$(date -u +%FT%TZ) w41-nestqual: window end: twins done/stopped by watchdog 18:27Z; $n helper processes (GPU staging/D6 waiters) moved to granted cores $TO; 0,1,2,16,17,18 free of lane work" >> "$MAILBOX"
  exit 0
fi
while (( $(date +%s) < KILL )) && [ -n "$(heavy)" ]; do sleep 10; done
H=$(heavy | tr '\n' ' ')
for p in $(ours); do kill -TERM "$p" 2>/dev/null; done; sleep 5
for p in $(ours); do kill -KILL "$p" 2>/dev/null; done
for p in $(pgrep -x wrf.exe); do   # backstop: twin ranks of ours (cwd in our cases dir) -> their session
  case "$(readlink /proc/$p/cwd 2>/dev/null)" in "$CASES"/*)
    s=$(ps -o sid= -p "$p" | tr -d ' '); for q in $(ps -eo pid=,sid= | awk -v s="$s" '$2==s{print $1}'); do kill -KILL "$q" 2>/dev/null; done
    log "backstop: killed twin session $s (wrf.exe $p)";; esac; done
LEFT=$(ours | tr '\n' ' ')
PEND=$(for a in C2 C3 C4 C7; do [ -s "$CASES/20260707_18z_$a/NEST_GATE.json" ] || echo -n "$a "; done)
log "cores 0,1,2,16,17,18 freed; interrupted heavy: [${H}] left: [${LEFT}] arms without D6: [${PEND}]"
echo "$(date -u +%FT%TZ) w41-nestqual: window end: 0,1,2,16,17,18 free of lane work (left: [${LEFT}]); arms without D6 yet: [${PEND}] (GPU staging + scoring need ~20 min/arm, 1-3 cores, <=10 GB; resume on a core grant)" >> "$MAILBOX"
