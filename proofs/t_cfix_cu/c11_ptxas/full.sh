#!/usr/bin/env bash
# t-cfix-cu c11: FULL deviceless compile (ptxas incl.) of one fence variant; per-ptxas elapsed/RSS + kept PTX. usage: full.sh snapname variant limit_s
set -u
S=<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-cu; L=<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/c11; T=<USER_HOME>/src/wrf_gpu2_wt/t-cfix-cu
snap=$S/$1; variant=$2; limit=$3; label=full_$1_$variant
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
for p in tmp jax triton cuda autotune; do mkdir -p $S/c11_dl/${label}_$p; done
<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label t-cfix-cu-c11-full-$variant --max 16G --timeout 3000 -- env T_CFIX_C11_FENCE=$variant JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 JAX_TRACEBACK_FILTERING=off PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/c11_dl/${label}_tmp JAX_COMPILATION_CACHE_DIR=$S/c11_dl/${label}_jax TRITON_CACHE_DIR=$S/c11_dl/${label}_triton CUDA_CACHE_PATH=$S/c11_dl/${label}_cuda GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$S/c11_dl/${label}_autotune "XLA_FLAGS=--xla_dump_to=$S/c11_dl/${label}_dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*advance_chunk.*" taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python $T/proofs/t_cfix_cu/c11_ptxas/dl_dump.py --source $snap --case <USER_HOME>/wrf_gpu2_lanes/v-prep/c11 --out $S/c11_dl/$label --target <USER_HOME>/wrf_gpu2_lanes/v-prep/c11/target_config.pbtxt --dump $S/c11_dl/${label}_dump --ptxas-limit-s $limit > $S/c11_dl/$label.log 2>&1
echo "rc=$?"
python3 - $S/c11_dl/$label/receipt.json <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d.get('ptxas',{})
print(d['lower'], d['compile'], 'wall', round(d['wall_s']), 'opt_dump_s', round(d.get('after_optimizations_s') or -1), 'programs', [(r.get('compile'), round(r.get('compile_s',-1))) for r in d['programs']], d.get('exception'))
rows=sorted(p.values(), key=lambda r: -(r.get('last',0)-r['first']))
print('ptxas children', len(rows), 'sum elapsed', round(sum(r.get('last',r['first'])-r['first'] for r in rows)), 'max RSS GiB', round(max([r['peak_rss_KiB'] for r in rows] or [0])/2**20,2))
for r in rows[:8]: print('  ', round(r.get('last',r['first'])-r['first']), 's', round(r['peak_rss_KiB']/2**20,2), 'GiB', r.get('bytes'), r.get('kept'))
print('limit_hit', d.get('ptxas_limit_hit'))
PY
