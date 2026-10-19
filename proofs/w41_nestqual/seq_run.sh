#!/bin/bash
# Sequential D6 nest gate (lane w41-nestqual), manager option A 2026-10-19T09:32:56Z: ONE worker at a time on core 24,
# each in its own systemd --user scope with MemoryMax=4G MemorySwapMax=0 (the hard enforcement a RAM slot applies),
# nice 19, outside the slot queue. A worker killed by SIGKILL (rc 137, the scope's OOM kill at MemoryMax) stops the
# sequence and is reported (never retried with a higher cap). QUIET is honoured inside worker.sh.
# usage: seq_run.sh <tree> <tag> <arm ...>
set -u
TREE=$1; TAG=$2; shift 2; TS=$(date -u +%H%M%S)
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
taskset -pc 24 $$ > /dev/null || exit 2
log(){ echo "$(date -u +%FT%TZ) seq_run $*" | tee -a "$LANE/twin_status.log"; }
log "tree=$TREE tag=$TAG arms=$* (core 24, MemoryMax=4G, one at a time)"
for ARM in "$@"; do
  systemd-run --user --scope -q --unit "w41nq-$ARM-$TS" -p MemoryMax=4G -p MemorySwapMax=0 -- \
    taskset -c 24 nice -n 19 bash "$TREE/proofs/w41_nestqual/worker.sh" "$TREE" "$ARM" "$TAG"
  R=$?
  # the cgroup OOM killer picks the largest process (usually the python compare child), so also look for SIGKILL'd
  # children in the worker's logs (wn3_score: 'exit status -9' / rc 137; shell: 'Killed')
  K=$(/usr/bin/grep -lE "exit status -9|returned non-zero exit status 137|^Killed|MemoryError" \
        "$LANE/stage_gpu_${ARM}_$TAG.log" "$LANE/score_${ARM}_$TAG.log" 2>/dev/null | tr '\n' ' ')
  log "$ARM worker rc=$R${K:+ SIGKILL/OOM signature in: $K}"
  if [ $R -eq 137 ] || [ -n "$K" ]; then
    echo "$(date -u +%FT%TZ) w41-nestqual: $ARM worker KILLED (rc $R; SIGKILL/OOM signature in: ${K:-worker itself}) in scope w41nq-$ARM-$TS at MemoryMax=4G: stopped the sequence per manager 09:32:56Z (cap NOT raised); remaining arms not run" >> "$MAILBOX"
    exit 137
  fi
done
log "done"
