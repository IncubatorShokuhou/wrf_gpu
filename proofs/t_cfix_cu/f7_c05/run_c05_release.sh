#!/usr/bin/env bash
# t-cfix-cu: F7 c05 pristine oracle on the RELEASE path (RTX 5090), snapshot 54cdee223, private cache; queued in the
# official lock (manager approval 2026-10-16 ~00:4xZ, lease t-cfix-cu, <=15 min arm).
set -u
SN=<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/snap_gpu; D=<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/diag; C=$D/cache_rel
mkdir -p "$C"
echo "start $(date -u +%FT%TZ) src $(git -C $SN rev-parse HEAD) dirty=$(git -C $SN status --porcelain src | wc -l)" > $D/release.status
<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label t-cfix-cu --timeout 18000 -- \
  <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label t-cfix-cu-c05rel --max 12G --timeout 1800 -- \
  env GPUWRF_JAX_CACHE_DIR=$C JAX_COMPILATION_CACHE_DIR=$C PYTHONPATH=$SN/src \
  timeout 1200 python $D/f7_c05_release.py <DATA_ROOT>/wrf_gpu2/scratch/t-cfix-cu/c05port \
  <DATA_ROOT>/wrf_gpu2/scratch/t-cfix-cu/c05w $D/f7_c05_release.json $D/f7_c05_release_fields.npz 3 > $D/release.log 2>&1
echo "arm rc=$? $(date -u +%FT%TZ)" >> $D/release.status
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: cubin gate deferred, cache kept" >> $D/release.status; exit 0; fi
env JAX_PLATFORMS=cpu taskset -c 8,9 nice -n 19 timeout 1800 python <USER_HOME>/src/wrf_gpu2/scripts/wave_cubin_gate.py "$C" $D/cubin_gate.json > $D/cubin_gate.log 2>&1
echo "cubin rc=$? $(date -u +%FT%TZ)" >> $D/release.status
rm -rf -- "${C:?}"
