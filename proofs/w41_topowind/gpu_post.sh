#!/bin/bash
# CPU post-processing of one topo_wind GPU arm (lane w41-topowind), OUTSIDE the GPU lock:
# E41 cubin gate on the arm's JAX cache BEFORE any cleanup (E166/E177), then D6 vs the original CPU-WRF
# (frozen manifest) per domain over 0-24 h.  usage: gpu_post.sh <commit> <case>
set -u
COMMIT=$1; c=$2
C=<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/cases/$c
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-topowind
cd "$LANE/snap_$COMMIT" || exit 4
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$PWD/src   # wave_cubin_gate imports gpuwrf
MANIFEST=proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ "${SKIP_E41:-0}" != 1 ]; then
  timeout 1800 taskset -c 24,25,28,29 nice -n 19 python scripts/wave_cubin_gate.py "$LANE/cache_$c" "$C/e41" > "$C/e41.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $c e41 rc=$rc" >> "$LANE/gpu_status"
fi
[ "${SKIP_D6:-0}" = 1 ] && exit 0
for d in $(ls "$C"/gpu | sed -n 's/^wrfout_\(d0[0-9]\)_.*/\1/p' | sort -u); do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  timeout 3600 taskset -c 24,25,28,29 nice -n 19 python scripts/compare_wrfout_grid.py --cpu-dir "$C/cpu" --gpu-dir "$C/gpu" \
    --domain $d --min-lead 0 --max-lead 24 --tolerance-json $MANIFEST --out-json "$C/D6_$d.json" --out-md "$C/D6_$d.md" \
    > "$C/D6_$d.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $c D6 $d rc=$rc" >> "$LANE/gpu_status"
done
rm -rf "$C/gpu_scratch"
