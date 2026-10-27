#!/usr/bin/env bash
# One immutable domain/value preparation per arm; all four use ONE shared cache.
# No timestep execution here. Warm A/B follows only after all four + E41 pass.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
LANE=<USER_HOME>/wrf_gpu2_lanes/w44-thparams
variant=${1:?off or on}; domain=${2:?d01 or d02}
[[ $variant == off || $variant == on ]]
[[ $domain == d01 || $domain == d02 ]]
smoke=${THPARAMS_LAUNCH_SMOKE:-0}
[[ $smoke == 0 || $smoke == 1 ]]
# CUDA mini (manager 21:48Z): same entries/env/lock/supervisor, 5-column Thompson
# MODEL SWAP instead of the PROD domain program (not a grid crop). smoke=1 + mini=1
# is the mini's CPU all-phase probe.
mini=${THPARAMS_CUDA_MINI:-0}
[[ $mini == 0 || $mini == 1 ]]
[[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
headroom=${GPUWRF_LOCK_MIN_FREE_MIB:-24000}
for parameter_env in ${!GPUWRF_@}; do unset "$parameter_env"; done
export GPUWRF_LOCK_MIN_FREE_MIB="$headroom" GPUWRF_GPU_ARM_CPUS=3,19,13,29
export GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0
export GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 GPUWRF_RRTMG_MP_RE=1
export PYTHONPATH="$TREE/src" OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.25
unset XLA_FLAGS
# The failed v2 cache is diagnostic evidence; v5 never mixes with v2-v4 caches (own _v5 run dirs).
# v5 (re-pin on the main merge): every phase runs in its OWN user systemd scope (MemoryMax, no swap),
# never through the shared RAM-slot FIFO (E233); GPU children 16G, CPU smoke children 12G.
# Run dirs/caches on the <DATA_ROOT> lane scratch (root fs is at 97 %).
RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/gpu_v5
if [[ $smoke == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/launch_smoke_v5; fi
if [[ $mini == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/cuda_mini_v5; fi
if [[ $mini == 1 && $smoke == 1 ]]; then RUN=<DATA_ROOT>/wrf_gpu2/scratch/w44-thparams/cuda_mini_probe_v5; fi
CACHE=$RUN/cache
export GPUWRF_JAX_CACHE_DIR="$CACHE/jax" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$CACHE/autotune" TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/cuda"
export TMPDIR="$RUN/tmp"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu mkdir -p "$RUN/tmp" "$CACHE/jax" "$CACHE/autotune" "$CACHE/triton" "$CACHE/cuda"
stamp=$(JAX_PLATFORMS=cpu date -u +%Y%m%dT%H%M%S%N)
out=$RUN/$variant-$domain-$stamp
CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu python3 -I "$HERE/check_alignment.py" "$CASE/namelist.input" 3  # E225: hours*3600 % every dt == 0
command=(python "$HERE/cold_prepare.py" --input-dir "$CASE" --out "$out" --variant "$variant" --domain "$domain")
if [[ $mini == 1 ]]; then command+=(--mini); fi
export JAX_PLATFORMS=cuda
if [[ $smoke == 1 ]]; then
  # The sole device substitution: CPU + Pallas interpretation in the child.
  # Product flags, command construction, supervisor and receipts stay in use.
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- timeout 300 python "$HERE/outer_supervise.py" --seconds 280 -- "${command[@]}" --cpu-smoke
  JAX_PLATFORMS=cpu python - "$out" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); r=json.loads((p/'receipt.json').read_text())
assert r['status']=='PASS' and r['platform']=='cpu' and not r['gpu_executed']
assert r['cpu_smoke'] and r['replay_byte_exact'] and r['output_leaves']==20
assert r['environment']['JAX_PLATFORMS']=='cpu'
assert r['environment']['GPUWRF_FAST_DEFAULTS']=='1'
events=[json.loads(line) for line in (p/'phases.jsonl').read_text().splitlines()]
assert any(row['phase']=='mini_model_replay' and row['event']=='end' for row in events)
assert r['aot'].get('loaded') or any(row['phase']=='aot_export' and row['event']=='end' for row in events)
print('LAUNCH SMOKE PASS w44-thparams',p/'receipt.json')
PY
  exit
fi
now=$(JAX_PLATFORMS=cpu <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh)
start=$(JAX_PLATFORMS=cpu date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(JAX_PLATFORMS=cpu date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && now < end )) || { echo 'outside authorized window'; exit 5; }
# A single variant/domain gets 900s. Failed aggregate baseline: ~572s until
# first fused export (249s compile+151s export interval+load/lower); this is a
# bounded reservation, not a measured guarantee for the corrected program.
# CUDA mini: 4 arms x 50 s = 200 s GPU cap (expected ~30 s/arm: native Pallas compile ~9 s [M];
# CPU-interpret smoke arm 14-16 s [M SM08]), so the cold half incl. its CPU E41 fits 5 min.
cap=900; if [[ $mini == 1 ]]; then cap=50; fi
seconds=$((end - now - 15)); (( seconds <= cap )) || seconds=$cap
(( seconds >= 30 )) || exit 5
(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 365 )) || { echo '<DATA_ROOT> below the 365 GiB floor'; exit 5; }
# E231: refuse before the lock unless the user manager grants a memory-capped scope.
systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 -- true || { echo 'user scope refused (E231)'; exit 7; }
exec systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 -- "$TREE/scripts/with_gpu_lock.sh" --label w44-thparams --timeout 1 -- \
  timeout --signal=TERM --kill-after=10 $((seconds+5)) python "$HERE/supervise.py" --seconds "$seconds" -- "${command[@]}"
