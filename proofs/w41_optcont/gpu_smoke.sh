#!/usr/bin/env bash
# GPU OSSE smoke of `gpuwrf optimize` continuous mode (lane w41-optcont; <= 20 GPU-min; manager window + lease only).
# usage: proofs/w41_optcont/gpu_smoke.sh ROOT [--prep-only | --gpu-only | --resume]
#   --prep-only: CPU part only (any time on lane cores); --gpu-only: lock session on a prepared ROOT (the GPU slot).
#   CPU before the lock: prepare cases/stations/study (seconds), Track P basis of the Swiss case (one RAM slot), plan.
#   GPU: optimize run, Swiss d01 3 h members, private caches, --max-parallel 4, --chunk-minutes 17 (pauses instead of
#        overrunning; continue later with the same command + --resume).
#   CPU after: verify the registered gates (REGISTERED_GATES.json), E41 cubin gate of the controls program, list caches.
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
ROOT=$1
MODE=${2:-}
RESUME=$([ "$MODE" = --resume ] && echo --resume || true)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOCK="$REPO/scripts/with_gpu_lock.sh --label w41-optcont"
CAP="systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 --"
cpu() { JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$REPO/src" taskset -c 2,18 nice -n 19 python3 "$@"; }
if [ "$MODE" != --resume ] && [ "$MODE" != --gpu-only ]; then
  test ! -e "$ROOT/study.json" || { echo "exists: $ROOT/study.json"; exit 2; }
  cpu "$REPO/proofs/w41_optcont/prepare_osse.py" "$ROOT" || exit 2
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  "$REPO/scripts/with_ram_slot.sh" --label w41-optcont --max 6G --timeout 900 -- \
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$REPO/src" taskset -c 2,18 nice -n 19 \
    python3 -m gpuwrf.cli surface-params basis --input-dir "$ROOT/cases/2023-01-15" --output "$ROOT/basis.npz" || exit 2
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$REPO/src" taskset -c 2,18 nice -n 19 \
    python3 -m gpuwrf.cli optimize plan --study "$ROOT/study.json" --output "$ROOT/plan.json" || exit 2
  echo "== CPU prep done $(date -u +%FT%TZ)"
fi
[ "$MODE" = --prep-only ] && exit 0
test -s "$ROOT/plan.json" && test -s "$ROOT/basis.npz" || { echo "not prepared: $ROOT"; exit 2; }
export GPUWRF_JAX_CACHE_DIR="$ROOT/cache/jax" JAX_COMPILATION_CACHE_DIR="$ROOT/cache/jax" \
       GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$ROOT/cache/autotune"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
# GPU wall = inside the lock (the start stamp is written after acquisition, not while queueing for it).
$LOCK -- bash -c 'date -u +%FT%TZ > "$0/gpu_start_utc.txt"; exec "$@"' "$ROOT" timeout 1260 $CAP env -u JAX_PLATFORMS PYTHONPATH="$REPO/src" \
  python3 -m gpuwrf.cli optimize run --study "$ROOT/study.json" --output "$ROOT/run" --max-parallel 4 \
  --host-gb-per-member 8 --chunk-minutes 17 $RESUME
rc=$?
date -u +%FT%TZ > "$ROOT/gpu_end_utc.txt"
echo "== optimize run rc=$rc $(date -u +%FT%TZ)"
unset GPUWRF_JAX_CACHE_DIR JAX_COMPILATION_CACHE_DIR GPUWRF_XLA_AUTOTUNE_CACHE_DIR
# E41 BEFORE any cleanup (the cache is the gate's only input), then the gates, then GPU END for the next lane.
cpu "$REPO/scripts/wave_cubin_gate.py" "$ROOT/cache/jax" "$ROOT/e41" > "$ROOT/e41.log" 2>&1; e41=$?
v=skipped; if [ "$rc" = 0 ]; then cpu "$REPO/proofs/w41_optcont/verify_osse.py" "$ROOT" "$ROOT/OSSE_GPU.json"; v=$?; fi
minutes=$(python3 -c "import datetime as d,sys;f=lambda p:d.datetime.fromisoformat(open(p).read().strip().replace('Z','+00:00'));print(round((f(sys.argv[2])-f(sys.argv[1])).total_seconds()/60,1))" "$ROOT/gpu_start_utc.txt" "$ROOT/gpu_end_utc.txt")
echo "$(date -u +%FT%TZ) w41-optcont: GPU END ${minutes} min (Swiss d01 OSSE smoke, optimize run rc=$rc, verify rc=$v, E41 gate rc=$e41; lock released) -- next lane may start" >> <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
echo "== done $(date -u +%FT%TZ) rc=$rc verify=$v e41=$e41; delete after review: $ROOT/cache $ROOT/run/inputs $ROOT/run/batch-*/runs"
exit "$rc"
