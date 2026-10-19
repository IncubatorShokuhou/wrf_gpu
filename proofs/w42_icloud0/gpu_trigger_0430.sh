#!/usr/bin/env bash
# GPU slot 2 of the ALISIOS 04:30-07:30Z window (manager 04:17:26Z), immutable (E163: new script, old trigger stopped).
# Starts gpu_arm.sh of the FIXED snapshot below on the first MAILBOX line after line 14228 matching
# '^2026-10-19T<time> w42-harness: S1H GPU END' or '... w42-harness: S1H NOT STARTED' (/usr/bin/grep, anchored on the
# sender). Latest start 06:30Z (cap 36 min, hard end 07:06Z); missed -> posts a 'w42-icloud0: GPU END ... NOT STARTED'
# line (slot 3 triggers on 'w42-icloud0: GPU END') and the arm falls back to the 13:00Z window.
set -uo pipefail
TREE=<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/snap_e0041902bd
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0
LOG=$L/gpu_trigger_0430.log
LAST=$(date -u -d 2026-10-19T06:30:00Z +%s)
RE='^2026-10-19T[0-9:]{8}(\.[0-9]+)?(Z|\+00:00) w42-harness: S1H (GPU END|NOT STARTED)'
echo "$(date -u +%FT%TZ) armed (after line 14228), TREE $TREE" >> "$LOG"
until tail -n +14229 "$MBX" | /usr/bin/grep -qE "$RE"; do
  if [ "$(date -u +%s)" -gt "$LAST" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=4, 0 GPU-min (NOT STARTED: no harness S1H END/NOT STARTED by the 06:30Z latest start; icloud0 falls back to the 13:00Z window; next slot may go)" >> "$MBX"
    exit 4
  fi
  sleep 15
done
echo "$(date -u +%FT%TZ) trigger: $(tail -n +14229 "$MBX" | /usr/bin/grep -m1 -E "$RE" | cut -c1-140)" >> "$LOG"
if [ "$(date -u +%s)" -gt "$LAST" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=4, 0 GPU-min (NOT STARTED: harness line came after the 06:30Z latest start; icloud0 falls back to the 13:00Z window; next slot may go)" >> "$MBX"
  exit 4
fi
echo "$(date -u +%FT%TZ) firing gpu_arm.sh TREE=$TREE HEAD $(git -C "$TREE" rev-parse HEAD) main $(git -C <USER_HOME>/src/wrf_gpu2 rev-parse main)" >> "$LOG"
TREE="$TREE" exec bash "$TREE/proofs/w42_icloud0/gpu_arm.sh" >> "$LOG" 2>&1
