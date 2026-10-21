#!/usr/bin/env bash
# w43-teacher: free remaining GPU cases right after their post_batch "<name> scored" line (D6+G-INT+extract+readback done).
# usage: bash analysis/free_after.sh DEADLINE_UTC RUN_DIR:SCORE_NAME:ISSUE ...
set -u
DL=$(date -u -d "$1" +%s); shift; MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
for spec in "$@"; do
  IFS=: read -r run name issue <<< "$spec"
  until grep -q "w43-teacher: .*post_batch $name scored" "$MBX"; do (( $("$REAL") < DL )) || exit 0; sleep 120; done
  bash analysis/free_chain_v2.sh "$spec"
done
