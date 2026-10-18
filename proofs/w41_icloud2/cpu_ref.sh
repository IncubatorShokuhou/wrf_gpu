#!/usr/bin/env bash
# Original CPU-WRF 4.7.1 icloud=2 reference for the WN3 0227 3-nest 24 h arm (w41-icloud2).
# Reuses the merged w41-icloud runner (proofs/w41_icloud/cpu_reference.sh: gen2 dmpar wrf sha 1d0f5779 = the
# executable of the icloud=3 twin, 4 ranks pinned one per granted logical CPU, nice 19, sealed CASE/CPU_OUTPUTS.json +
# CASE/cpu_run.json in the score.py layout) inside a MAIN RAM slot (5.4 GB peak measured on the icloud=3 twin).
# Refuses unless stage_inputs.py wrote STAGE.json verdict OK for this CASE and the namelist says icloud = 2.
# usage: CORES=a,b,c,d bash cpu_ref.sh [CASE]   (CORES = the manager-granted cores, one per physical core)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CASE=$(realpath "${1:-<DATA_ROOT>/wrf_gpu2/scratch/w41-icloud2/wn3_icloud2}")
L=<USER_HOME>/wrf_gpu2_lanes/w41-icloud2
: "${CORES:?manager-granted cores, e.g. CORES=0,1,2,10}"
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['verdict']=='OK' else 1)" "$L/stage/STAGE.json" \
  || { echo "STAGE.json missing or not OK"; exit 2; }
[ "$(/usr/bin/grep -cE '^ *icloud *= *2,' "$CASE/input/namelist.input")" = 1 ] || { echo "namelist is not icloud=2"; exit 2; }
[ -e /tmp/wrf_gpu2_quiet ] && { echo QUIET; exit 3; }
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: CPU REF START original CPU-WRF 4.7.1 icloud=2 WN3 0227 3-nest 24 h, 4 ranks on $CORES nice 19, RAM slot 6G (~3.6 h)" >> "$MBX"
<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-icloud2-cpuref --max 6G --timeout 7200 -- \
  env CORES="$CORES" timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT:-28800}" bash "$HERE/../w41_icloud/cpu_reference.sh" "$CASE" \
  > "$L/cpu_ref.log" 2>&1
RC=$?
# the runner exits 0 after writing its receipt even when WRF failed -> report WRF's own rc from the receipt
WRC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rc'])" "$CASE/cpu_run.json" 2>/dev/null || echo missing)
[ "$WRC" = 0 ] || [ "$RC" != 0 ] || RC=1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: CPU REF END rc=$RC wrf_rc=$WRC ($(tail -1 "$L/cpu_ref.log"))" >> "$MBX"
exit $RC
