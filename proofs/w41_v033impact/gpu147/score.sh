#!/usr/bin/env bash
# GPU-vs-GPU impact scoring with the FROZEN D6 scorer (regression/impact evidence, not fidelity).
# usage: score.sh REF_ARM X_ARM [maxlead] [vars...]   (arms under <DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact/gpu147/<arm>/out)
set -u
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
G=<DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact/gpu147
O=<USER_HOME>/wrf_gpu2_lanes/w41-v033impact/gpu147/d6
W=<USER_HOME>/src/wrf_gpu2_wt/w41-v033impact
ref=$1 x=$2 maxlead=${3:-24}; shift 3 2>/dev/null || shift $#
mkdir -p $O
for d in d02 d03 d01; do
  JAX_PLATFORMS=cpu taskset -c 14,30 nice -n 19 python $W/scripts/compare_wrfout_grid.py --cpu-dir $G/$ref/out --gpu-dir $G/$x/out \
    --domain $d --init 2026-07-08T00:00:00 --min-lead 1 --max-lead $maxlead \
    --tolerance-json $W/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
    ${@:+--vars "$@"} --out-json $O/${x}_vs_${ref}_${d}_L${maxlead}.json --out-md $O/${x}_vs_${ref}_${d}_L${maxlead}.md \
    > $O/${x}_vs_${ref}_${d}_L${maxlead}.log 2>&1
  echo "$x vs $ref $d rc=$?"
done
