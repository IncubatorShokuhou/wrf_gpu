#!/usr/bin/env bash
# w41-bdyrec GPU requalification, INSIDE the official lock (manager-scheduled window).
# usage: scripts/with_gpu_lock.sh --label w41-bdyrec --timeout T -- \
#          proofs/w41_bdyrec/gpu_requal.sh ROOT [phases, default "1 2 3"]
# ROOT = staged by gpu_stage.py (PLAN.json). Release defaults: no GPUWRF_* model flag is set
# (same environment class as the sealed v0.4.0 bundle priming). Each GPU process runs in its own
# systemd scope (MemoryMax=20G, no swap; manager rule 11:37Z: never a RAM slot around/inside the lock).
#  phase 1  cold 24 h arms, sequential: Swiss / PROD / WN3-0227 on the candidate (cap 2640 s each)
#           -> 24 h original-CPU D6 + E41 (CPU, after the lock) and frames 0-3 BYTE vs the
#              sealed v0.4.0 bundle frames (WN3-0227, Swiss)
#  phase 2  warm 3 h arms on the record-TRUNCATED inputs (48 h shape), same caches
#           -> new cache entries (expected: no step program), loaded=true, frames == phase 1
#  phase 3  release control: WN3-0227 3 h cold on the release snapshot (E99 control: must
#           reproduce the bundle frames), then matched warm 3 h nsys arms release vs candidate
#           -> host<->device copies / syncs per arm, warm wall
set -uo pipefail
ROOT=$(readlink -f "$1"); PHASES=${2:-"1 2 3"}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
plan() { python3 -c "import json,sys; d=json.load(open('$ROOT/PLAN.json')); print(eval(sys.argv[1]))" "$1"; }
CAND=$(plan "d['candidate']"); REL=$(plan "d['release']")
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/requal.log"; }

inventory() {  # cache file inventory (names + sizes), excluding runtime state
  (cd "$(readlink -f "$1/cache")/jax" && find . -type f ! -name '*-atime' ! -name '.lockfile' -printf '%P %s\n' | sort) > "$2"
}

arm() {  # arm TREE CASE_DIR OUT_SUBDIR HOURS FRACTION [nsys-prefix...]
  # FRACTION is recorded only: arms run one at a time with the CLI's own C-auto pool plan,
  # the allocator environment of the sealed v0.4.0 bundle priming (the BYTE reference).
  local tree=$1 case=$2 out=$3 hours=$4 fraction=$5; shift 5
  # One cache path STRING per case: JAX/AOT keys include the cache dir (E136/E143), so the
  # short-input arm resolves its cache symlink to the full-input arm's directory.
  local cache; cache=$(readlink -f "$case/cache")
  local dom; dom=$(python3 -c "import json;print(len(json.load(open('$case/inputs.json'))['domains']))")
  local extra=(); (( dom > 1 )) && extra=(--emit-initial-history)
  mkdir -p "$case/$out/wrfout" "$case/$out/proof" "$case/$out/scratch"
  ( cd "$case/$out" &&
    env JAX_PLATFORMS=cuda PYTHONPATH="$tree/src" \
      GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/autotune" TRITON_CACHE_DIR="$cache/triton" \
      CUDA_CACHE_PATH="$cache/cuda" TMPDIR="$case/tmp" \
      systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- \
      /usr/bin/time -v timeout --signal=TERM --kill-after=15 2640 "$@" python3 -m gpuwrf.cli run \
        --input-dir "$case/input" --output-dir "$case/$out/wrfout" --proof-dir "$case/$out/proof" \
        --scratch-dir "$case/$out/scratch" --hours "$hours" --domains-from-namelist "${extra[@]}" \
      > "$case/$out/run.log" 2>&1
    rc=$?; echo "arm rc=$rc" >> "$case/$out/run.log"
    (cd "$case/$out/wrfout" && sha256sum wrfout_* > ../wrfout.sha256)
    exit $rc )
}

for phase in $PHASES; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop before phase $phase"; exit 3; fi
  case $phase in
  1) log "phase 1 cold 24 h arms, sequential (host RAM: one CLI process ~10-14 GB) (candidate $(plan "d['candidate_head']"))"
     nvidia-smi --query-gpu=memory.free,memory.used --format=csv | tee -a "$ROOT/requal.log"
     for c in Swiss PROD WN3-0227; do
       if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop before $c"; exit 3; fi
       arm "$CAND" "$ROOT/$c" run24 24 "$(plan "d['arms']['$c']['fraction']")"; log "phase 1 $c rc=$?"
     done
     for c in Swiss PROD WN3-0227; do ln -sfn run24/wrfout "$ROOT/$c/gpu"; done ;;  # frozen D6 scorer reads case/gpu
  2) log "phase 2 warm 3 h on record-truncated inputs"
     for c in Swiss PROD WN3-0227; do
       inventory "$ROOT/$c" "$ROOT/$c-short/inventory_before.txt"
       arm "$CAND" "$ROOT/$c-short" run3 3 "$(plan "d['arms']['$c']['fraction']")"; log "phase 2 $c rc=$?"
       inventory "$ROOT/$c" "$ROOT/$c-short/inventory_after.txt"
     done ;;
  3) log "phase 3 release control + matched nsys"
     C=$ROOT/WN3-0227-release
     arm "$REL" "$C" cold3 3 .18; log "phase 3 release cold rc=$?"
     NSYS=(nsys profile -t cuda,osrt --cuda-graph-trace=graph --sample=none --cpuctxsw=none --force-overwrite=true)
     arm "$REL" "$C" warm3_nsys 3 .18 "${NSYS[@]}" -o "$C/warm3_nsys/release"; log "phase 3 release nsys rc=$?"
     arm "$CAND" "$ROOT/WN3-0227" warm3_nsys 3 .18 "${NSYS[@]}" -o "$ROOT/WN3-0227/warm3_nsys/candidate"
     log "phase 3 candidate nsys rc=$?" ;;
  esac
done
log "GPU part done; CPU follow-up: gpu_followup.py $ROOT"
