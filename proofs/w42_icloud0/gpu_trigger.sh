#!/usr/bin/env bash
# GPU slot 2 trigger (manager 00:38:35Z): start gpu_arm.sh on the exact line '^<ts> w42-harness: S1H GPU END' (only
# lines appended after arming, /usr/bin/grep, V042 rule), never before 13:00Z; if harness posts
# '^<ts> w42-harness: S1H NOT STARTED' start at 13:00Z. The arm tree is read from $L/GPU_TREE at FIRE time (the
# snapshot must equal the merge candidate; re-snapshot on the final main without re-arming). No start after 14:20Z
# (cap 36 min, lease to 15:00Z): post a GPU END 0 GPU-min line instead and stop.
# usage: setsid bash -c 'echo $$ > PIDFILE; exec bash gpu_trigger.sh'
set -uo pipefail
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0
LOG=$L/gpu_trigger.log
N=$(wc -l < "$MBX")
START=$(date -u -d 2026-10-19T13:00:00Z +%s); LAST=$(date -u -d 2026-10-19T14:20:00Z +%s)
TS='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}(\.[0-9]+)?(Z|\+00:00) w42-harness: '
echo "$(date -u +%FT%TZ) armed at line $N" >> "$LOG"
while :; do
  NEW=$(tail -n +"$((N + 1))" "$MBX")
  NOW=$(date -u +%s)
  if [ "$NOW" -ge "$START" ] && { echo "$NEW" | /usr/bin/grep -qE "${TS}S1H GPU END" || echo "$NEW" | /usr/bin/grep -qE "${TS}S1H NOT STARTED"; }; then break; fi
  sleep 20
done
echo "$(date -u +%FT%TZ) trigger: $(echo "$NEW" | /usr/bin/grep -m1 -E "${TS}S1H (GPU END|NOT STARTED)" | cut -c1-140)" >> "$LOG"
if [ "$(date -u +%s)" -gt "$LAST" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=4, 0 GPU-min (not started: past 14:20Z, the 36-min cap would cross the 15:00Z lease; next slot may go)" >> "$MBX"
  exit 4
fi
TREE=$(cat "$L/GPU_TREE")
echo "$(date -u +%FT%TZ) firing gpu_arm.sh TREE=$TREE HEAD $(git -C "$TREE" rev-parse HEAD) main $(git -C <USER_HOME>/src/wrf_gpu2 rev-parse main)" >> "$LOG"
TREE="$TREE" exec bash "$TREE/proofs/w42_icloud0/gpu_arm.sh" >> "$LOG" 2>&1
