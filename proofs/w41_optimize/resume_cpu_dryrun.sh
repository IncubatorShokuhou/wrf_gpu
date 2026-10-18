#!/usr/bin/env bash
# Resume the CPU optimize rehearsal in an existing ROOT (same study, same private caches) until it is DONE.
# usage: resume_cpu_dryrun.sh ROOT [CHUNK_MINUTES]
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
ROOT=$1; CHUNK=${2:-10}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO/src" GPUWRF_JAX_CACHE_DIR="$ROOT/cache/jax" JAX_COMPILATION_CACHE_DIR="$ROOT/cache/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$ROOT/cache/autotune"
py() { taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 "$@"; }
for attempt in 1 2 3 4 5 6 7 8; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  echo "== resume attempt $attempt $(git -C "$REPO" rev-parse --short HEAD) $(date -u +%FT%TZ)"
  py -m gpuwrf.cli optimize run --study "$ROOT/study.json" --output "$ROOT/run" --max-parallel 1 \
     --chunk-minutes "$CHUNK" --resume
  rc=$?
  echo "== resume attempt $attempt rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 3 ] || break
done
[ $rc -eq 0 ] || exit $rc
py "$REPO/proofs/w41_optimize/cpu_dryrun.py" verify "$ROOT" "$ROOT/CPU_DRYRUN.json"
