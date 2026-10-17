#!/usr/bin/env bash
# usage: run_lower.sh <src_tree> <tag> case...   (CPU lower-only of v-prep's exact case inputs/env)
set -u
SRC=$1; TAG=$2; shift 2
L=<USER_HOME>/wrf_gpu2_lanes/t-calleraudit/c18_lower
for c in "$@"; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  W=$L/$TAG/$c; rm -rf "$W"; mkdir -p "$W/cache"
  t0=$(date +%s)
  ( cd "$SRC" && env JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 \
      GPUWRF_RRTMG_MP_RE=1 GPUWRF_JAX_CACHE_DIR=$W/cache/jax JAX_COMPILATION_CACHE_DIR=$W/cache/jax \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$W/cache/autotune PYTHONPATH=src \
      timeout 1500 scripts/with_ram_slot.sh --label t-calleraudit --max 10G --timeout 3600 -- \
      taskset -c 0,1,2,8,9,12 nice -n 19 python $L/repro_lower.py $W/marker.json run \
      --input-dir <USER_HOME>/wrf_gpu2_lanes/v-prep/$c/input --output-dir $W/out --proof-dir $W/proof \
      --scratch-dir $W/scratch --hours 3 --domains-from-namelist ) > $W/log 2>&1
  rc=$?
  echo "$c rc=$rc wall=$(( $(date +%s) - t0 ))s $(cat $W/marker.json 2>/dev/null)"
  rm -rf "$W/cache" "$W/out" "$W/scratch"
done
