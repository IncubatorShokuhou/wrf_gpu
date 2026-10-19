#!/usr/bin/env bash
# Original CPU-WRF 4.7.1 icloud=0 reference for the WN3 0227 3-nest 24 h arm (w42-icloud0).
# Reuses the merged runner proofs/w41_icloud/cpu_reference.sh (gen2 dmpar wrf sha 1d0f5779 = the executable of the
# icloud=3/icloud=2 twins; its module_radiation_driver.F = pristine 4.7.1 minus the oracle-dump instrumentation; ranks
# pinned one per granted logical CPU, nice 19; sealed CASE/CPU_OUTPUTS.json + CASE/cpu_run.json in the score.py layout)
# inside a MAIN RAM slot (5.4 GB peak measured with 4 ranks). 4 ranks on 8,9,12,13 (manager 01:58:43Z: the WN3 namelist pins
# nproc 2x2; a 3-rank attempt stopped in the WRF decomposition check, kept in CASE/attempt1_3ranks).
# Refuses unless stage_inputs.py wrote STAGE.json verdict OK and the namelist says icloud = 0.
# usage: CORES=8,9,12,13 bash cpu_ref.sh [CASE]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CASE=$(realpath "${1:-<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/wn3_icloud0}")
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0
CORES=${CORES:-8,9,12,13}
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['verdict']=='OK' else 1)" "$CASE/STAGE.json" \
  || { echo "STAGE.json missing or not OK"; exit 2; }
[ "$(/usr/bin/grep -cE '^ *icloud *= *0,' "$CASE/input/namelist.input")" = 1 ] || { echo "namelist is not icloud=0"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: CPU REF START original CPU-WRF 4.7.1 icloud=0 WN3 0227 3-nest 24 h, $(echo "$CORES" | tr , "\n" | wc -l) ranks on $CORES nice 19 (namelist decomposition 2x2 unchanged), RAM slot 6G (~2.8 h)" >> "$MBX"
<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w42-icloud0-cpuref --max 6G --timeout 7200 -- \
  env CORES="$CORES" timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT:-28800}" bash "$HERE/../w41_icloud/cpu_reference.sh" "$CASE" \
  > "$L/cpu_ref.log" 2>&1
RC=$?
# the runner exits 0 after writing its receipt even when WRF failed -> report WRF's own rc from the receipt
WRC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rc'])" "$CASE/cpu_run.json" 2>/dev/null || echo missing)
[ "$WRC" = 0 ] || [ "$RC" != 0 ] || RC=1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: CPU REF END rc=$RC wrf_rc=$WRC ($(tail -1 "$L/cpu_ref.log"))" >> "$MBX"
exit $RC
