#!/usr/bin/env bash
# w42-scorer 72 h gate run on the #146 0420 data: hard-link the finished B/F/C wrfouts (frozen against the owner's later
# delivery/delete; no bytes copied), then score B vs C and F vs C with the candidate scorer (default = sequential
# domains at 72 h) under measure.py (tree peak RSS) + pause_on_quiet, core 25 nice 19, 12G RAM slot.
# usage: bash run72.sh CAND_TREE
set -u
CAND=${1:?cand tree}
L=<USER_HOME>/wrf_gpu2_lanes/w42-scorer; R=<DATA_ROOT>/wrf_gpu2/scratch/w41-swddirscreen/runs/20260420_18z
S=<DATA_ROOT>/wrf_gpu2/scratch/w42-scorer/d0420_72h
for a in B F C; do
  src=$R/$a/gpu; [ $a = C ] && src=$R/C/cpu
  mkdir -p $S/$a
  n=$(ls $S/$a | grep -c '^wrfout_d0' || true)
  if [ "$n" != 219 ]; then ln "$src"/wrfout_d0* $S/$a/ || exit 2; fi
  n=$(ls $S/$a | grep -c '^wrfout_d0'); [ "$n" = 219 ] || { echo "$a: $n files, expected 219"; exit 2; }
done
echo "$(date -u +%FT%TZ) linked 3x219"
for a in B F; do
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 2; done
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w42-scorer --max 12G --timeout 14400 -- \
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19 python3 $L/tools/measure.py $L/new72_${a}_measure.json \
    $L/tools/pause_on_quiet.sh python3 $CAND/scripts/wn3_score.py --cpu-dir $S/C --gpu-dir $S/$a --out $L/new72_$a --hours 72 \
    > $L/new72_$a.log 2>&1
  echo "$(date -u +%FT%TZ) new72 $a rc=$? $(grep -E '^\[measure\]|peak RSS: scorer' $L/new72_$a.log | tr '\n' ' ')"
done
