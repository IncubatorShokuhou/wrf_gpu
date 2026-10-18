#!/bin/bash
# CPU post-processing OUTSIDE the GPU lock (lane w41-shinhongtke), cores 2,18 nice 19 (manager 22:58:50Z):
#   post.sh <commit> e41            -> E41 cubin gate on the shared session cache (BEFORE any cleanup, E166/E177)
#   post.sh <commit> <case> [hours] -> D6 vs the ORIGINAL CPU-WRF (frozen manifest) + TKE_PBL/EL_PBL compare
set -u
COMMIT=$1; c=$2; HOURS=${3:-24}
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/w41-shinhongtke/cases
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-shinhongtke
cd "$LANE/snap_$COMMIT" || exit 4
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$PWD/src
MANIFEST=proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ "$c" = e41 ]; then
  timeout 1800 taskset -c 2,18 nice -n 19 python scripts/wave_cubin_gate.py "$LANE/cache_w41sht" "$LANE/e41" > "$LANE/e41.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) e41 rc=$rc" >> "$LANE/gpu_status"; exit 0
fi
C=$ROOT/$c
timeout 3600 taskset -c 2,18 nice -n 19 python scripts/compare_wrfout_grid.py --cpu-dir "$C/cpu" --gpu-dir "$C/gpu" \
  --domain d01 --min-lead 0 --max-lead "$HOURS" --tolerance-json $MANIFEST --out-json "$C/D6_d01.json" --out-md "$C/D6_d01.md" \
  > "$C/D6_d01.log" 2>&1
rc=$?; echo "$(date -u +%FT%TZ) $c D6 d01 rc=$rc" >> "$LANE/gpu_status"
taskset -c 2,18 nice -n 19 python proofs/w41_shinhongtke/tke_compare.py "$C" "$C/TKE_COMPARE.json" > "$C/tke_compare.log" 2>&1
rc=$?; echo "$(date -u +%FT%TZ) $c tke_compare rc=$rc" >> "$LANE/gpu_status"
rm -rf "$C/gpu_scratch"
