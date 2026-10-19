#!/bin/bash
# Keep lane helpers on the granted cores (lane w41-nestqual). Manager CORE PLAN 2026-10-18T08:26:28Z: this lane has
# 0,1,2 (+SMT 16,17,18) only; 8,9,12 are no longer ours. The running run_queue.sh (never mutated, E163) still spawns
# its per-arm scorer waiters with 'taskset -c 8,9,12'; this loop moves every process of ours whose cmdline names the
# lane path AND whose affinity is exactly 8,9,12 to 16,17,18 (all threads). Never touches other processes.
# usage: pin_keeper.sh <run_queue pid> <end ISO-UTC>
set -u
QPID=$1; END=$(date -u -d "$2" +%s) || exit 2; TO=16,17,18
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual
taskset -pc "$TO" $$ > /dev/null
while (( $(date +%s) < END )); do
  for p in $(pgrep -u "$(id -u)" -f w41-nestqual); do
    [ "$p" = $$ ] && continue
    a=$(taskset -pc "$p" 2>/dev/null | awk -F': ' '{print $2}')
    [ "$a" = "8,9,12" ] || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | /usr/bin/grep -q 'w41-nestqual' || continue
    taskset -apc "$TO" "$p" > /dev/null 2>&1 && \
      echo "$(date -u +%FT%TZ) pin_keeper $p 8,9,12 -> $TO: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-120)" >> "$LANE/twin_status.log"
  done
  kill -0 "$QPID" 2>/dev/null || break
  sleep 20
done
