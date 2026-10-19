#!/bin/bash
# Launch the four G2 arms' D6 nest gate on the cores granted by the manager (GO 2026-10-19T09:26:01Z, correction
# 09:26:16Z): 24 (whole day) + 3 and 29 until 12:30Z. One systemd --user scope per worker (worker.sh, RAM slot 4G
# each, nice 19, pinned): C2 on 3, C3 on 29, C4 then C7 on 24. Backstop: the scopes on 3/29 are stopped at 12:25Z
# if still running; once both have ended (and no process of ours is left on 3/29) one MAILBOX line
# 'w41-nestqual: CORES 3,29 RELEASED'. This launcher only waits (on 24).
# usage: launch.sh <tree> <tag>
set -u
TREE=$1; TAG=$2; TS=$(date -u +%H%M%S)
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
FREE=$(date -u -d 2026-10-19T12:25:00Z +%s)
taskset -pc 24 $$ > /dev/null || exit 2
log(){ echo "$(date -u +%FT%TZ) launch $*" | tee -a "$LANE/twin_status.log"; }
run(){  # unit core arm...
  local unit=$1 core=$2; shift 2
  systemd-run --user --scope -q --unit "$unit" -- taskset -c "$core" nice -n 19 bash -c '
    for a in "${@:3}"; do
      "$1/scripts/with_ram_slot.sh" --label w41-nestqual --max 4G --timeout 3600 -- bash "$1/proofs/w41_nestqual/worker.sh" "$1" "$a" "$2"
    done' _ "$TREE" "$TAG" "$@"
}
log "tree=$TREE tag=$TAG units w41nq-{C2,C3,C4C7}-$TS"
run "w41nq-C2-$TS" 3 C2 & P2=$!
run "w41nq-C3-$TS" 29 C3 & P3=$!
run "w41nq-C4C7-$TS" 24 C4 C7 & P4=$!
( while (( $(date +%s) < FREE )); do kill -0 $P2 2>/dev/null || kill -0 $P3 2>/dev/null || exit 0; sleep 20; done
  for u in "w41nq-C2-$TS" "w41nq-C3-$TS"; do systemctl --user stop "$u.scope" 2>/dev/null && log "backstop: stopped $u at 12:25Z"; done ) &
W=$!
wait $P2 $P3
left=$(for p in $(pgrep -u "$(id -u)" -f 'w41-nestqual|w41_nestqual|w41nq'); do
         a=$(taskset -pc "$p" 2>/dev/null | awk -F': ' '{print $2}'); [ "$a" = 3 ] || [ "$a" = 29 ] && echo "$p"; done | tr '\n' ' ')
log "C2/C3 scopes ended; processes left on 3/29: [${left}]"
[ -z "$left" ] && echo "$(date -u +%FT%TZ) w41-nestqual: CORES 3,29 RELEASED (C2/C3 D6 workers done; 0 lane processes on 3/29)" >> "$MAILBOX"
wait $P4
kill $W 2>/dev/null
log "all workers ended"
