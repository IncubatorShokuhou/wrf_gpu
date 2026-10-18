#!/usr/bin/env bash
# CPU end-to-end rehearsal of `gpuwrf optimize` (Swiss d01, 2 released configurations x 2 rungs, SYNTHETIC dates).
# usage: run_cpu_dryrun.sh ROOT [CHUNK_MINUTES]   (ROOT must not exist; private caches live under ROOT/cache)
# Every model member is a real `python -m gpuwrf.cli run` on the CPU via scripts/run_parallel_cases.sh, one at a
# time (--max-parallel 1). --chunk-minutes makes the study pause after a batch; the loop resumes it until done.
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
ROOT=$1; CHUNK=${2:-10}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO/src" GPUWRF_JAX_CACHE_DIR="$ROOT/cache/jax" JAX_COMPILATION_CACHE_DIR="$ROOT/cache/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$ROOT/cache/autotune"
py() { taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 "$@"; }
py "$REPO/proofs/w41_optimize/cpu_dryrun.py" prepare "$ROOT" || exit 2
py -m gpuwrf.cli optimize plan --study "$ROOT/study.json" --output "$ROOT/plan.json" || exit 2
resume=()
for attempt in 1 2 3 4 5 6; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  echo "== attempt $attempt $(date -u +%FT%TZ)"
  py -m gpuwrf.cli optimize run --study "$ROOT/study.json" --output "$ROOT/run" --max-parallel 1 \
     --chunk-minutes "$CHUNK" "${resume[@]}"
  rc=$?
  echo "== attempt $attempt rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 3 ] || break
  resume=(--resume)
done
[ $rc -eq 0 ] || exit $rc
py "$REPO/proofs/w41_optimize/cpu_dryrun.py" verify "$ROOT" "$ROOT/CPU_DRYRUN.json"
