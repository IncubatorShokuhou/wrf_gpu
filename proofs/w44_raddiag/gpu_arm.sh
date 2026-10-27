#!/usr/bin/env bash
# Frozen WN3 0227 3-nest diagnostic ON/OFF/bl0. CPU setup/dry-run precedes NEED-GPU.
set -euo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/w44-raddiag
G=$L/gpu
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=${RADDIAG_TREE:?set immutable source snapshot}
PIN=${RADDIAG_PIN:?set exact source commit}
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
BASE=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227
if [ "${1:-}" == --dry-run ]; then
  bash -n "$0"
  echo "source $TREE at $PIN; base $BASE; 3h d01/d02/d03; OFF, ON, ON+icloud_bl0; official lock; E41 before cleanup"
  JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/readback.py" --selftest "$L"
  JAX_PLATFORMS=cpu PYTHONPATH="$TREE/src" taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/prepare.py" "$L" "$BASE" --dry-run
  exit
fi
if [ "${1:-}" != --locked ]; then
  [[ $(git -C "$TREE" rev-parse HEAD) == "$PIN" ]]
  [[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
  (( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 200 ))
  now=$(JAX_PLATFORMS=cpu "$REAL")
  (( now <= $(date -u -d 2026-10-23T18:05:00Z +%s) )) || { echo 'past manager start cutoff'; exit 5; }
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  # Inputs are the manager-authorized staged copy, never ALISIOS-root files.
  JAX_PLATFORMS=cpu PYTHONPATH="$TREE/src" taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/prepare.py" "$G" "$BASE"
  rc=0
  "$TREE/scripts/with_gpu_lock.sh" --label w44-raddiag --timeout 1 -- timeout --signal=TERM --kill-after=30 3600 bash "$0" --locked || rc=$?
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET; keep cache for E41"; exit 3; fi
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 "$TREE/scripts/with_ram_slot.sh" --label w44-raddiag-e41 --max 12G --timeout 1200 -- timeout 600 python "$TREE/scripts/wave_cubin_gate.py" "$G/cache/jax" "$G/e41.json" > "$G/e41.log" 2>&1 || { echo 'E41 receipt needs attention; cache retained'; exit 4; }
  echo "GPU arm rc=$rc, E41 receipt $G/e41.json; raw outputs/cache retained until readback receipts sealed"
  exit "$rc"
fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]]
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && end - now >= 1800 )) || { echo 'outside manager GPU window'; exit 5; }
mkdir -p "$G/cache/jax" "$G/cache/autotune" "$G/cache/triton" "$G/cache/cuda" "$G/tmp"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1
export GPUWRF_JAX_CACHE_DIR="$G/cache/jax" JAX_COMPILATION_CACHE_DIR="$G/cache/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$G/cache/autotune"
export TRITON_CACHE_DIR="$G/cache/triton" CUDA_CACHE_PATH="$G/cache/cuda" TMPDIR="$G/tmp" GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30
export OMP_NUM_THREADS=4 BENCH_SRC_REPO="$TREE" PYTHONPATH="$TREE/src"
unset XLA_FLAGS
journalctl -k --since '-6h' 2>/dev/null | /usr/bin/grep -i hibernat > "$G/hibernate_before.txt" || true
for arm in off on bl0; do
  now=$(JAX_PLATFORMS=cpu "$REAL")
  (( end - now >= 900 )) || { echo 'insufficient window for another cold arm'; exit 5; }
  cap=$(( end - now - 60 )); if (( cap > 1700 )); then cap=1700; fi
  /usr/bin/time -v -o "$G/$arm/time.txt" timeout --signal=TERM --kill-after=15 "$cap" python "$HERE/harness.py" --input-dir "$G/$arm/input" --out "$G/$arm/run" --hours 3 --tag "w44-raddiag-$arm" > "$G/$arm/run.log" 2>&1
  if [[ $arm == on ]]; then
    JAX_PLATFORMS=cpu timeout 120 python "$HERE/readback.py" "$G/off/run/wrfout" "$G/on/run/wrfout" "$G/BYTE_GATE.json" || { echo 'STOP: compiled prognostic byte gate failed; bl0 cancelled'; exit 42; }
  fi
done
JAX_PLATFORMS=cpu timeout 120 python "$HERE/readback.py" "$G/BL0_READBACK.json" --fields-only "$G/bl0/run/wrfout"
journalctl -k --since '-2h' 2>/dev/null | /usr/bin/grep -i hibernat > "$G/hibernate_after.txt" || true
