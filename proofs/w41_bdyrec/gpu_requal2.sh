#!/usr/bin/env bash
# w41-bdyrec GPU requal, second session (INSIDE scripts/with_gpu_lock.sh), after the AOT capture fix.
# usage: gpu_requal2.sh ROOT CANDIDATE_TREE RELEASE_TREE [phases, default "A B C"]
# Session 1 (gpu_requal.sh phase 1) ran the 24 h cold arms on 403075dad0; its d01 AOT capture failed
# (namelist holder not picklable, fixed 8409ddfb08), so no step executable blob exists yet. The traced
# programs are unchanged by the fix (CPU key proof), so the JAX caches of session 1 serve both trees.
#  A  per case: warm 3 h FULL-input arm on the fixed candidate -> AOT blobs written ("aot-captured"),
#     frames 0-3 == session-1 24 h frames
#  B  per case: 3 h arm on the record-TRUNCATED inputs (48 h shape) -> every step program loaded from
#     those blobs (loaded=true), no new step-program cache entries, frames == A
#  C  release control: WN3-0227 3 h cold on the release snapshot (E99 control: must reproduce the bundle
#     frames), then matched warm 3 h nsys arms release vs candidate -> copies / syncs per arm, warm wall
# Release defaults (no GPUWRF_* model flag); each GPU process in its own systemd scope (MemoryMax=20G,
# no swap; manager rule 11:37Z).
set -uo pipefail
ROOT=$(readlink -f "$1"); CAND=$(readlink -f "$2"); REL=$(readlink -f "$3"); PHASES=${4:-"A B C"}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/requal2.log"; }

inventory() {  # cache file inventory (names + sizes), excluding runtime state
  (cd "$(readlink -f "$1/cache")/jax" && find . -type f ! -name '*-atime' ! -name '.lockfile' -printf '%P %s\n' | sort) > "$2"
}

arm() {  # arm TREE CASE_DIR OUT_SUBDIR HOURS [nsys-prefix...]  (same CLI call as gpu_requal.sh)
  local tree=$1 case=$2 out=$3 hours=$4; shift 4
  local cache; cache=$(readlink -f "$case/cache")
  local dom; dom=$(python3 -c "import json;print(len(json.load(open('$case/inputs.json'))['domains']))")
  local extra=(); (( dom > 1 )) && extra=(--emit-initial-history)
  mkdir -p "$case/$out/wrfout" "$case/$out/proof" "$case/$out/scratch"
  # Cap = the scope's RuntimeMaxSec: systemd stops EVERY process of the arm (a nested uutils timeout
  # runs in its own process group and survived a session group kill on 2026-10-17 15:10Z).
  local unit; unit="w41bdyrec-arm-$(basename "$case")-$out-$$-$RANDOM"
  ( cd "$case/$out" &&
    env JAX_PLATFORMS=cuda PYTHONPATH="$tree/src" \
      GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/autotune" TRITON_CACHE_DIR="$cache/triton" \
      CUDA_CACHE_PATH="$cache/cuda" TMPDIR="$case/tmp" \
      systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 -- \
      /usr/bin/time -v "$@" python3 -m gpuwrf.cli run \
        --input-dir "$case/input" --output-dir "$case/$out/wrfout" --proof-dir "$case/$out/proof" \
        --scratch-dir "$case/$out/scratch" --hours "$hours" --domains-from-namelist "${extra[@]}" \
      > "$case/$out/run.log" 2>&1
    rc=$?; echo "arm rc=$rc" >> "$case/$out/run.log"
    (cd "$case/$out/wrfout" && sha256sum wrfout_* > ../wrfout.sha256)
    exit $rc )
}

aot_summary() {  # one line: counts of nested-aot statuses in a run log
  python3 - "$1" <<'EOF'
import re, sys
lines = [l for l in open(sys.argv[1], errors="replace") if "[gpuwrf:nested-aot]" in l]
loaded = sum("loaded=true" in l for l in lines); captured = sum("aot-captured" in l for l in lines)
errors = sum(bool(re.search(r"error=(?!None)", l)) for l in lines)
print(f"nested-aot lines={len(lines)} loaded=true:{loaded} aot-captured:{captured} errors:{errors}")
EOF
}

log "session 2 phases '$PHASES' candidate $(git -C "$CAND" rev-parse HEAD) release $(git -C "$REL" rev-parse HEAD)"
for phase in $PHASES; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop before phase $phase"; exit 3; fi
  case $phase in
  A) for c in Swiss PROD WN3-0227; do
       arm "$CAND" "$ROOT/$c" run3_aot 3; log "phase A $c rc=$? $(aot_summary "$ROOT/$c/run3_aot/run.log")"
     done ;;
  B) for c in Swiss PROD WN3-0227; do
       inventory "$ROOT/$c" "$ROOT/$c-short/inventory_before.txt"
       arm "$CAND" "$ROOT/$c-short" run3 3; log "phase B $c rc=$? $(aot_summary "$ROOT/$c-short/run3/run.log")"
       inventory "$ROOT/$c" "$ROOT/$c-short/inventory_after.txt"
     done ;;
  C) C=$ROOT/WN3-0227-release
     arm "$REL" "$C" cold3 3; log "phase C release cold rc=$?"
     NSYS=(nsys profile -t cuda,osrt --cuda-graph-trace=graph --sample=none --cpuctxsw=none --force-overwrite=true)
     arm "$REL" "$C" warm3_nsys 3 "${NSYS[@]}" -o "$C/warm3_nsys/release"; log "phase C release nsys rc=$?"
     arm "$CAND" "$ROOT/WN3-0227" warm3_nsys 3 "${NSYS[@]}" -o "$ROOT/WN3-0227/warm3_nsys/candidate"
     log "phase C candidate nsys rc=$? $(aot_summary "$ROOT/WN3-0227/warm3_nsys/run.log")" ;;
  esac
done
log "GPU part done; CPU follow-up: gpu_followup.py $ROOT"
