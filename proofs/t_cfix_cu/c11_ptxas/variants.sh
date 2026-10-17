#!/usr/bin/env bash
# t-cfix-cu c11: deviceless optimized-HLO dump per fence variant of one commit. usage: variants.sh commit snapname variant ...
set -u
S=<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-cu; L=<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/c11; T=<USER_HOME>/src/wrf_gpu2_wt/t-cfix-cu
commit=$1; snapname=$2; shift 2
snap=$S/$snapname
[ -d $snap ] || { git -C $T worktree add --detach $snap $commit >/dev/null 2>&1 && ln -s <USER_HOME>/src/wrf_pristine $snap/data/wrf_pristine; }
test "$(git -C $snap rev-parse HEAD)" = "$(git -C $T rev-parse $commit)" || { echo "snapshot HEAD mismatch"; exit 2; }
for variant in "$@"; do
  label=v_${snapname}_$variant
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  for p in tmp jax triton cuda autotune; do mkdir -p $S/c11_dl/${label}_$p; done
  <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label t-cfix-cu-c11-$variant --max 10G --timeout 1200 -- env T_CFIX_C11_FENCE=$variant JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 JAX_TRACEBACK_FILTERING=off PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/c11_dl/${label}_tmp JAX_COMPILATION_CACHE_DIR=$S/c11_dl/${label}_jax TRITON_CACHE_DIR=$S/c11_dl/${label}_triton CUDA_CACHE_PATH=$S/c11_dl/${label}_cuda GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$S/c11_dl/${label}_autotune "XLA_FLAGS=--xla_dump_to=$S/c11_dl/${label}_dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*advance_chunk.*" taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python $T/proofs/t_cfix_cu/c11_ptxas/dl_dump.py --source $snap --case <USER_HOME>/wrf_gpu2_lanes/v-prep/c11 --out $S/c11_dl/$label --target <USER_HOME>/wrf_gpu2_lanes/v-prep/c11/target_config.pbtxt --dump $S/c11_dl/${label}_dump --stop-after-opt 3 > $S/c11_dl/$label.log 2>&1
  H=$(ls $S/c11_dl/${label}_dump/*advance_chunk_fori.sm_12.0a_gpu_after_optimizations.txt 2>/dev/null | head -1)
  echo "== $label $commit $(python3 -c "import json;d=json.load(open('$S/c11_dl/$label/receipt.json'));print(d['lower'],d['compile'],round(d['wall_s']),d.get('exception'))")"
  [ -n "$H" ] && python3 $T/proofs/t_cfix_cu/c11_ptxas/scan_fusions.py $H wdm6 $L/scan_$label.json | head -6
  [ -n "$H" ] && grep -c " transpose(" $H | sed 's/^/   transposes in module: /'
  [ -n "$H" ] && cp $H $L/hlo_$label.txt && gzip -f $L/hlo_$label.txt
done
