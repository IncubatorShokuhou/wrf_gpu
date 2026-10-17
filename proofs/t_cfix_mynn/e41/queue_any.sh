#!/usr/bin/env bash
# Sequential deviceless arms. Usage: queue_any.sh SNAP:CASE:DOMAIN ...  (CASE dir under the F20 archive)
set -u
S=<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-mynn; L=<USER_HOME>/wrf_gpu2_lanes/t-cfix-mynn/e41
A=<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016
ulimit -c 0
for spec in "$@"; do
  IFS=: read -r snap case dom <<< "$spec"
  out="$L/DL_${snap}_${case}_${dom}"
  while [ -e /tmp/wrf_gpu2_quiet ]; do echo "QUIET wait $(date -u +%T)"; sleep 60; done
  if [ -e "$out" ]; then echo "skip $spec (exists)"; continue; fi
  echo "start $spec $(date -u +%T)"
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 \
    GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 \
    taskset -c 0,1 nice -n 19 <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label t-cfix-mynn --max 12G --timeout 5400 -- \
    timeout 5300 python "$L/dl_prod_d01.py" --source "$S/snap_$snap" --case "$A/$case" --target "$A/$case/target_config.pbtxt" \
    --domain "$dom" --out "$out" > "$out.log" 2>&1
  echo "end $spec rc=$? $(date -u +%T)"
  python3 -c "
import json;d=json.load(open('$out/receipt.json'))
print('  RESULT $spec', d.get('source_commit','')[:10], d.get('phase'), (d.get('exception') or '')[:300], 'tf', d.get('transpose_fusions'), 'over500', d.get('any_transpose_over500'), 'stack', d.get('max_stack_B'), 'kernels', d.get('kernels'), 'gate', d.get('native_gate'), 'top', d.get('top_transpose', [])[:3])
" 2>&1
done
