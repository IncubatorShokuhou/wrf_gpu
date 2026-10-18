#!/usr/bin/env bash
# Composed default requal trigger (manager 2026-10-17T23:36:25Z). usage: trigger.sh ROOT
# 1. waits for the strict MAILBOX line '<ts>Z manager -> w41-bdyrec: REQUAL GO <swddir-sha>' (posted after this start)
# 2. CPU: stage_refs.py (guarded CPU-WRF reference frames) + setup.py (snapshot = main + swddir merge, case dirs)
# 3. waits for '<ts>Z w41-shinhongtke...GPU END' (posted after this start), then ONE lock session (session.sh)
# 4. CPU follow-up (E41 before any cleanup, D6, S1 summary, identities) in a 12G RAM slot on lane cores
# Watchers use /usr/bin/grep with positive matches only (manager 23:01:33Z: 'grep' may be a ugrep function).
set -uo pipefail
ROOT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec
M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LANE_SHA=${LANE_SHA:-969baaba1c}          # critic-checked lane/w41-bdyrec src head; main must contain it
LANE_CPUS=${LANE_CPUS:-10,11,26,27,2,18}  # manager 23:00:02Z allocation until 07:45Z
LOCK_UNTIL_UTC=${LOCK_UNTIL_UTC:-2026-10-18T06:20:00Z}   # latest lock acquisition (session needs ~55 min, end 07:30Z)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/composed_trigger.log"; }
N0=$(wc -l < "$M")
quiet_wait() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }

log "armed: waiting for 'manager -> w41-bdyrec: REQUAL GO <sha>' after MAILBOX line $N0"
GO_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w41-bdyrec: REQUAL GO [0-9a-f]{7,40}'
until tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -qE "$GO_RE"; do sleep 30; done
GO_LINE=$(tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -E "$GO_RE" | head -1)
SWD=$(echo "$GO_LINE" | /usr/bin/grep -oE 'REQUAL GO [0-9a-f]{7,40}' | awk '{print $3}')
MAIN=$(git -C <USER_HOME>/src/wrf_gpu2_wt/w41-bdyrec rev-parse main)
log "GO: '$GO_LINE' -> swddir $SWD main $MAIN"

quiet_wait
STAGE=<DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/requal_refs
if [ ! -s "$L/requal_refs_receipt/STAGED_INDEX.json" ]; then
  taskset -c "$LANE_CPUS" nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    python3 "$HERE/stage_refs.py" "$STAGE" "$L/requal_refs_receipt" > "$L/requal_refs_receipt.log" 2>&1
  rc=$?; log "stage_refs rc=$rc $(tail -1 "$L/requal_refs_receipt.log" | cut -c1-300)"
  (( rc == 0 )) || { echo "$(date -u +%FT%TZ) w41-bdyrec -> manager: REQUAL STOPPED: guarded staging rc=$rc (see $L/requal_refs_receipt.log); no GPU used." >> "$M"; exit 1; }
fi
quiet_wait
taskset -c "$LANE_CPUS" nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
  python3 "$HERE/setup.py" "$ROOT" "$MAIN" "$SWD" "$LANE_SHA" "$L/requal_refs_receipt/STAGED_INDEX.json" > "$L/composed_setup.log" 2>&1
rc=$?; log "setup rc=$rc $(tail -1 "$L/composed_setup.log" | cut -c1-300)"
(( rc == 0 )) || { echo "$(date -u +%FT%TZ) w41-bdyrec -> manager: REQUAL STOPPED: setup rc=$rc ($(tail -1 "$L/composed_setup.log" | cut -c1-200)); no GPU used." >> "$M"; exit 1; }
echo "$(date -u +%FT%TZ) w41-bdyrec -> manager: REQUAL GO ACK: candidate $(python3 -c "import json;print(json.load(open('$ROOT/PLAN.json'))['candidate_head'][:12])") = main ${MAIN:0:10} + swddir ${SWD:0:10}; guarded refs staged; waiting for w41-shinhongtke GPU END, then one lock session (<= 75 min, ends before 07:30Z)." >> "$M"

# GPU END must follow the sender directly (status lines that only mention 'GPU END' must not fire)
END_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z w41-shinhongtke( -> [^:]+)?: GPU END'
until tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -qE "$END_RE"; do sleep 30; done
log "shinhongtke GPU END seen: $(tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -E "$END_RE" | head -1 | cut -c1-200)"
WAIT=$(( $(date -u -d "$LOCK_UNTIL_UTC" +%s) - $(date +%s) ))
(( WAIT > 60 )) || { echo "$(date -u +%FT%TZ) w41-bdyrec -> manager: REQUAL NOT STARTED: past $LOCK_UNTIL_UTC (no time for ~55 GPU-min before 07:30Z)." >> "$M"; exit 1; }
echo "$(date -u +%FT%TZ) w41-bdyrec: GPU START composed requal (lock queue, <= 75 GPU-min, ends before 07:30Z)" >> "$M"
<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w41-bdyrec --timeout "$WAIT" -- bash "$HERE/session.sh" "$ROOT"
rc=$?
MIN=$(/usr/bin/grep -oE 'END GPU part: [0-9]+ GPU-min' "$ROOT/session.log" 2>/dev/null | /usr/bin/grep -oE '[0-9]+' | tail -1)
echo "$(date -u +%FT%TZ) w41-bdyrec: GPU END composed requal rc=$rc, ${MIN:-?} GPU-min (session.log $ROOT/session.log); CPU follow-up (E41/D6/S1/identities) running." >> "$M"
log "session rc=$rc"

quiet_wait
taskset -c "$LANE_CPUS" nice -n 19 <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-bdyrec --max 12G --timeout 7200 -- \
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3 "$HERE/followup.py" "$ROOT" > "$L/composed_followup.log" 2>&1
log "followup rc=$? -> $ROOT/RESULT.json"
