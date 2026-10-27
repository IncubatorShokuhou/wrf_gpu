#!/usr/bin/env bash
# Optional follow-up ONLY after BYTE_GATE PASS and a separate explicit manager GO.
# Run inside the official GPU lock. Reuses the numerical arm's exact caches/env.
set -euo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/w44-raddiag
G=$L/gpu
TREE=${RADDIAG_TREE:?immutable source}
PIN=${RADDIAG_PIN:?exact commit}
HARN=${RADDIAG_TOOLS:?frozen primary tools directory}/harness.py
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
if [[ ${1:-} == --dry-run ]]; then
  bash -n "$0"
  command -v nsys
  nsys profile --help | /usr/bin/grep 'cuda-graph-trace' > /dev/null
  echo "DRY PASS: cached off/on,3h,profile-segment2,cuda-graph-trace=node; needs BYTE PASS + explicit manager GO"
  exit
fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]]
[[ $(git -C "$TREE" rev-parse HEAD) == "$PIN" ]]
JAX_PLATFORMS=cpu timeout 10 python - "$G/BYTE_GATE.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['prognostics_byte_identical'] is True
PY
now=$(JAX_PLATFORMS=cpu "$REAL")
end=$(date -u -d "${GA_END_UTC:?explicit window end}" +%s)
((end-now>=300)) || { echo 'not enough window for cached profile'; exit 5; }
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1
export GPUWRF_JAX_CACHE_DIR="$G/cache/jax" JAX_COMPILATION_CACHE_DIR="$G/cache/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$G/cache/autotune"
export TRITON_CACHE_DIR="$G/cache/triton" CUDA_CACHE_PATH="$G/cache/cuda" TMPDIR="$G/tmp" GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30
export OMP_NUM_THREADS=4 BENCH_SRC_REPO="$TREE" PYTHONPATH="$TREE/src"
unset XLA_FLAGS
for arm in off on; do
  timeout --signal=TERM --kill-after=15 150 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="$G/profile_$arm" \
    python "$HARN" --input-dir "$G/$arm/input" --out "$G/profile_$arm/run" --hours 3 \
    --tag="w44-raddiag-profile-$arm" --profile-segment=2 > "$G/profile_$arm.log" 2>&1
done
