#!/bin/bash
# Original CPU-WRF 4.7.1 serial reference for c07sd1 (lane w41-shinhongtke), AFTER the manager's
# 'NIGHTLY DONE' line only (RAM slot, cores 13,29 nice 19). c07 serial 24 h took 509 s (v-prep).
#   proofs/w41_shinhongtke/cpu_run.sh
set -u
C=<DATA_ROOT>/wrf_gpu2/scratch/w41-shinhongtke/cases/c07sd1
REPO=<USER_HOME>/src/wrf_gpu2_wt/w41-shinhongtke
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
cd "$C/cpu" || exit 4
ls wrfout_* >/dev/null 2>&1 && { echo "reference frames exist; refusing"; exit 4; }
echo "$(date -u +%FT%TZ) c07sd1 CPU START" >> <USER_HOME>/wrf_gpu2_lanes/w41-shinhongtke/gpu_status
"$REPO/scripts/with_ram_slot.sh" --label w41-shinhongtke --max 6G --timeout 3600 -- \
  taskset -c 29 nice -n 19 ./wrf.exe > "$C/cpu.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) c07sd1 CPU END rc=$rc frames=$(ls wrfout_* 2>/dev/null | wc -l)" >> <USER_HOME>/wrf_gpu2_lanes/w41-shinhongtke/gpu_status
[ $rc = 0 ] && python3 "$REPO/proofs/w41_topowind/seal_cpu.py" "$C" 24
