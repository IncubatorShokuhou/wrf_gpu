#!/usr/bin/env bash
# One short native GPU process; no nsys/GPU child process or lock waiter.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=${TOPO_TREE:?immutable model snapshot required}
PIN=${TOPO_PIN:?exact model source pin required}
MANIFEST=${TOPO_GPU_MANIFEST:?CPU-built sealed fixture manifest required}
OUT=<USER_HOME>/wrf_gpu2_lanes/w44-topo/gpu_component
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
if [[ ${1:-} == --dry-run ]]; then
  bash -n "$0"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu taskset -c 8,9,12,24,25,28 nice -n 19 timeout 30 python "$HERE/gpu_probe.py" --manifest "$MANIFEST" --out "$OUT" --dry-run
  exit
fi
[[ $(git -C "$TREE" rev-parse HEAD) == "$PIN" ]]
[[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]]
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
now=$(JAX_PLATFORMS=cpu "$REAL")
start=$(date -u -d "${TOPO_GPU_BEGIN_UTC:?explicit manager GO window required}" +%s)
end=$(date -u -d "${TOPO_GPU_END_UTC:?explicit manager GO deadline required}" +%s)
(( now >= start && end - now >= 70 )) || { echo 'outside manager GPU window'; exit 5; }
cap=$(( end - now - 10 )); if (( cap > 180 )); then cap=180; fi
mkdir -p "$OUT/cache/jax" "$OUT/cache/triton" "$OUT/cache/cuda" "$OUT/tmp"
export GPUWRF_JAX_CACHE=1 GPUWRF_JAX_CACHE_DIR="$OUT/cache/jax" JAX_COMPILATION_CACHE_DIR="$OUT/cache/jax"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export TRITON_CACHE_DIR="$OUT/cache/triton" CUDA_CACHE_PATH="$OUT/cache/cuda" TMPDIR="$OUT/tmp"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 PYTHONPATH="$TREE/src"
unset XLA_FLAGS
rc=0
"$TREE/scripts/with_gpu_lock.sh" --label w44-topo --timeout 1 -- timeout --signal=TERM --kill-after=10 "$cap" env JAX_PLATFORMS=cuda python "$HERE/gpu_probe.py" --manifest "$MANIFEST" --out "$OUT" > "$OUT/gpu.log" 2>&1 || rc=$?
echo "GPU END rc=$rc (single process exited, official lock released)"
# Tiny standalone operator payloads only. This disassembles files on CPU;
# it cannot acquire the GPU. Keep caches if QUIET or if the gate fails.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo 'QUIET; E41 cache retained'; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,24,25,28 nice -n 19 timeout 120 python "$TREE/scripts/wave_cubin_gate.py" "$OUT/cache/jax" "$OUT/e41.json" 'jit_one*' > "$OUT/e41.log" 2>&1 || { echo 'E41 incomplete/fail; cache retained'; exit 4; }
exit "$rc"
