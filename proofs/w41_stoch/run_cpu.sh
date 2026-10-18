#!/bin/bash
# Original serial CPU-WRF 4.7.1 24 h reference for one w41-stoch case (lane w41-stoch), then seal.
# usage: run_cpu.sh <case> <core>     (one serial wrf.exe per physical core, nice 19, timeout 2 h)
set -u
c=$1; core=$2
C=<DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/cases/$c
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-stoch
[ -L "$C/cpu" ] && { echo "$c: CPU reference is a link (reused), nothing to run"; exit 0; }
ls "$C"/cpu/wrfout_* > /dev/null 2>&1 && { echo "$c: reference frames exist, never overwrite"; exit 4; }
cd "$C/cpu" || exit 4
t0=$(date +%s)
echo "$(date -u +%FT%TZ) $c CPU START core=$core" >> "$LANE/cpu_status"
timeout 7200 taskset -c "$core" nice -n 19 ./wrf.exe > "$C/cpu.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) $c CPU END rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ls wrfout_* 2>/dev/null | wc -l) core=$core" >> "$LANE/cpu_status"
[ $rc = 0 ] && JAX_PLATFORMS=cpu taskset -c "$core" nice -n 19 python3 <USER_HOME>/src/wrf_gpu2_wt/w41-stoch/proofs/w41_stoch/seal_cpu.py "$C" 24 >> "$LANE/cpu_status" 2>&1
exit $rc
