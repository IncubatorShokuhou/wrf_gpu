#!/usr/bin/env bash
# Composed requal, S1-only re-run (01:15Z prime failed on the stale bench GPUWRF_WRF_ROOT). usage: s1_trigger.sh ROOT
# Waits for the strict MAILBOX start line (default '<ts>Z manager -> w41-bdyrec: S1 GO'; S1_GO_RE overrides it, e.g. the
# previous lock holder's GPU END in the manager's queue order), posted after this start, then ONE lock
# session REQUAL_PHASES=S (prime + warm1 + warm2), then the CPU follow-up: the PROD E41 scan is moved aside first so
# the PROD cache is re-scanned with the S1 harness programs in it; D6/identities are kept (already done).
set -uo pipefail
ROOT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec
M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LANE_CPUS=${LANE_CPUS:-10,26,27}                          # manager 01:28:17Z: lane CPU after 02:00Z
LOCK_UNTIL_UTC=${LOCK_UNTIL_UTC:-2026-10-18T07:05:00Z}    # S1 needs ~12 min, end before 07:28Z
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/composed_s1_trigger.log"; }
N0=$(wc -l < "$M")
GO_RE=${S1_GO_RE:-'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w41-bdyrec: S1 GO'}
log "armed: waiting for /$GO_RE/ after MAILBOX line $N0"
host_load() {  # disclosed in the S1 receipt dir (manager 01:40:45Z): load + top CPU users with their cores
  { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -25 | cut -c1-200; } >> "$ROOT/S1/HOST_LOAD.txt"
}
until tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -qE "$GO_RE"; do sleep 30; done
log "GO: $(tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -E "$GO_RE" | head -1 | cut -c1-200)"
WAIT=$(( $(date -u -d "$LOCK_UNTIL_UTC" +%s) - $(date +%s) ))
(( WAIT > 60 )) || { echo "$(date -u +%FT%TZ) w41-bdyrec -> manager: S1 NOT STARTED: past $LOCK_UNTIL_UTC." >> "$M"; exit 1; }
host_load "before S1 session"
echo "$(date -u +%FT%TZ) w41-bdyrec: GPU START composed requal S1-only (lock queue, ~12 GPU-min, ends before 07:28Z)" >> "$M"
<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w41-bdyrec --timeout "$WAIT" -- \
  env REQUAL_PHASES=S bash "$HERE/session.sh" "$ROOT"
rc=$?
host_load "after S1 session"
MIN=$(/usr/bin/grep -oE 'END GPU part: [0-9]+ GPU-min' "$ROOT/session.log" | /usr/bin/grep -oE '[0-9]+' | tail -1)
echo "$(date -u +%FT%TZ) w41-bdyrec: GPU END composed requal S1-only rc=$rc, ${MIN:-?} GPU-min; CPU follow-up (S1 summary, PROD E41 re-scan) running." >> "$M"
log "session rc=$rc"
[ -d "$ROOT/PROD/e41" ] && mv "$ROOT/PROD/e41" "$ROOT/PROD/e41_before_s1"
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
taskset -c "$LANE_CPUS" nice -n 19 <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-bdyrec --max 12G --timeout 7200 -- \
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3 "$HERE/followup.py" "$ROOT" > "$L/composed_followup_s1.log" 2>&1
log "followup rc=$? -> $ROOT/RESULT.json"
