#!/usr/bin/env bash
# w41-aerosol composite GPU arms for the manager's window (lease w41-aerosol): 24 h each, release defaults.
#   A1  WN3 0227 3-nest aer_opt=1 (Tegen)      CASE root <DATA_ROOT>/wrf_gpu2/scratch/w41-aerosol/wn3_aer1
#   A3  Swiss RD11 mp_physics=28 + aer_opt=3    CASE root <DATA_ROOT>/wrf_gpu2/scratch/w41-aerosol/swiss_aer3
# TREE = immutable detached snapshot of lane/w41-aerosol (data/wrf_pristine symlinked). Each arm writes CASE/gpu,
# CASE/gpu_proof, the E41 scan on the actual executables (before cache cleanup, E166/E177/E201), removes its caches.
# The GPU process runs inside the lock under its own 20G memory cap (no RAM slot around/inside the lock).
# Scoring afterwards on CPU (after the CPU references are sealed with seal_cpu.py):
#   JAX_PLATFORMS=cpu python proofs/v040/final_validation/score.py TREE CASE --domains d01 d02 d03   (A1)
#   JAX_PLATFORMS=cpu python proofs/v040/final_validation/score.py TREE CASE --domains d01            (A3)
# usage: TREE=<snapshot> bash gpu_arms.sh [A1] [A3]
set -uo pipefail
TREE=$(realpath "${TREE:?immutable snapshot}")
B=<DATA_ROOT>/wrf_gpu2/scratch/w41-aerosol
L=<USER_HOME>/wrf_gpu2_lanes/w41-aerosol/gpu_arms
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 2; }
mkdir -p "$L"; git -C "$TREE" rev-parse HEAD > "$L/source_commit.txt"
arm() {  # name case [extra CLI flags]  (single-domain runs emit t=0 themselves and reject --emit-initial-history)
  local name=$1 case=$2; shift 2
  [ -e /tmp/wrf_gpu2_quiet ] && { echo QUIET; return 3; }
  [ -e "$case/gpu" ] && { echo "refusing existing $case/gpu"; return 2; }
  local c=$L/$name/cache; mkdir -p "$c"
  local t0=$(date +%s)
  ( cd "$TREE" && env GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=0 GPUWRF_RRTMG_MP_RE=1 XLA_CLIENT_MEM_FRACTION=.30 \
      GPUWRF_JAX_CACHE_DIR="$c/jax" JAX_COMPILATION_CACHE_DIR="$c/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$c/autotune" \
      PYTHONPATH="$TREE/src" scripts/with_gpu_lock.sh --label w41-aerosol -- \
      systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- timeout 3600 python -m gpuwrf run \
      --input-dir "$case/input" --output-dir "$case/gpu" --proof-dir "$case/gpu_proof" --scratch-dir "$case/gpu_scratch" \
      --hours 24 --domains-from-namelist "$@" ) > "$L/$name.log" 2>&1
  local rc=$?
  echo "{\"arm\": \"$name\", \"rc\": $rc, \"wall_s\": $(( $(date +%s) - t0 ))}" | tee -a "$L/arms.jsonl"
  # CPU post-processing outside the GPU lock, in a normal RAM slot (manager 11:37Z rule). The gate needs the snapshot's
  # src on PYTHONPATH (sys.path[0] is scripts/); the cache goes only after an E41 PASS verdict (E166/E177).
  ( cd "$TREE" && scripts/with_ram_slot.sh --label w41-aerosol --max 6G -- env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
      PYTHONPATH="$TREE/src" taskset -c 8,9,12,13 nice -n 19 timeout 1800 python scripts/wave_cubin_gate.py "$c/jax" \
      "$L/$name.e41" --run-proof "$case/gpu_proof/nested_pipeline_run.json" ) > "$L/$name.e41.log" 2>&1
  local erc=$? v
  v=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['e41_gate'])" "$L/$name.e41/wave_gate.json" 2>/dev/null || echo MISSING)
  echo "{\"arm\": \"$name\", \"e41_rc\": $erc, \"e41\": \"$v\"}" | tee -a "$L/arms.jsonl"
  [ "$erc" = 0 ] && [ "$v" = PASS ] && rm -rf -- "$L/$name/cache" "$case/gpu_scratch"
}
[ $# -eq 0 ] && set -- A1 A3
for a in "$@"; do
  case $a in
    A1) arm A1_wn3_aer1 "$B/wn3_aer1" --emit-initial-history ;;
    A3) arm A3_swiss_aer3 "$B/swiss_aer3" ;;
  esac
done
