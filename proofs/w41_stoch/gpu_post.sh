#!/bin/bash
# CPU post-processing of one w41-stoch GPU arm, OUTSIDE the GPU lock (lane w41-stoch):
# 1) E41 cubin gate on the arm's JAX cache BEFORE any cleanup (E166/E177), 2) frozen D6 vs original CPU-WRF
# over 0-24 h, 3) stochastic pattern/seed comparison vs CPU-WRF (pattern_compare.py), then the cache is deleted.
# usage: gpu_post.sh <commit> <case>     (needs the CPU reference to be sealed for steps 2/3)
set -u
COMMIT=$1; c=$2
C=<DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/cases/$c
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-stoch
SNAP=<DATA_ROOT>/wrf_gpu2/scratch/w41-stoch/snap_$COMMIT
cd "$SNAP" || exit 4
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$PWD/src   # wave_cubin_gate imports gpuwrf
CORES=${POST_CORES:-24,25,28,29}
MANIFEST=proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
if [ "${SKIP_E41:-0}" != 1 ] && [ ! -e "$C/e41/wave_gate.json" ]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  timeout 1800 taskset -c $CORES nice -n 19 python scripts/wave_cubin_gate.py "$LANE/cache_$c/jax" "$C/e41" > "$C/e41.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $c e41 rc=$rc" >> "$LANE/gpu_status"
fi
[ "${SKIP_D6:-0}" = 1 ] && exit 0
ls "$C"/cpu/wrfout_d01_* > /dev/null 2>&1 || { echo "$c: no CPU reference yet"; exit 5; }
if [ ! -e "$C/D6_d01.json" ]; then
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  timeout 3600 taskset -c $CORES nice -n 19 python scripts/compare_wrfout_grid.py --cpu-dir "$C/cpu" --gpu-dir "$C/gpu" \
    --domain d01 --min-lead 0 --max-lead 24 --tolerance-json $MANIFEST --out-json "$C/D6_d01.json" --out-md "$C/D6_d01.md" \
    > "$C/D6_d01.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $c D6 d01 rc=$rc" >> "$LANE/gpu_status"
fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
timeout 1800 taskset -c $CORES nice -n 19 python proofs/w41_stoch/pattern_compare.py "$C/cpu" "$C/gpu" "$C/PATTERN_d01.json" > "$C/PATTERN_d01.log" 2>&1
rc=$?; echo "$(date -u +%FT%TZ) $c pattern rc=$rc" >> "$LANE/gpu_status"
rm -rf "$C/gpu_scratch"
