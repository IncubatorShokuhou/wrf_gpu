#!/usr/bin/env bash
# F vs C 72 h with the final candidate tree, started after the B arm (PID $2) ends. usage: run72_F.sh CAND_TREE B_PID
set -u
CAND=${1:?}; BPID=${2:?}; L=<USER_HOME>/wrf_gpu2_lanes/w42-scorer; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-scorer/d0420_72h
while kill -0 "$BPID" 2>/dev/null; do sleep 10; done
echo "$(date -u +%FT%TZ) B arm ended: $(grep -E '^\[measure\]' $L/new72_B.log)"
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 2; done
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w42-scorer --max 12G --timeout 14400 -- \
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19 python3 $L/tools/measure.py $L/new72_F_measure.json \
  $L/tools/pause_on_quiet.sh python3 $CAND/scripts/wn3_score.py --cpu-dir $S/C --gpu-dir $S/F --out $L/new72_F --hours 72 \
  > $L/new72_F.log 2>&1
echo "$(date -u +%FT%TZ) new72 F rc=$? $(grep -E '^\[measure\]|peak RSS: scorer' $L/new72_F.log | tr '\n' ' ')"
