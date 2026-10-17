#!/usr/bin/env bash
set -uo pipefail
test "${GPUWRF_GPU_LOCK_HELD:-0}" = 1 || exit 2
source_tree=$1
launcher=$2
output=<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms/c45/six-hour
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
free -g
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python "$source_tree/proofs/v040/final_validation/gpu_authorization.py" || exit 3
now=$(date -u +%s)
last=$(date -u -d '2026-10-15 19:10:00 UTC' +%s)
budget=1770
(( budget > last - now - 30 )) && budget=$((last-now-30))
(( budget >= 600 )) || exit 3
mkdir -p "$output/cache/jax" "$output/cache/autotune" "$output/cache/triton" "$output/cache/cuda" "$output/tmp"
env JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 \
  GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_RRTMG_MP_RE=1 GPUWRF_JAX_CACHE=1 \
  GPUWRF_JAX_CACHE_DIR="$output/cache/jax" JAX_COMPILATION_CACHE_DIR="$output/cache/jax" \
  GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$output/cache/autotune" \
  TRITON_CACHE_DIR="$output/cache/triton" CUDA_CACHE_PATH="$output/cache/cuda" TMPDIR="$output/tmp" \
  XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.15 \
  GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=15 "$budget" python "$launcher" "$source_tree" "$output"
rc=$?
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python - "$output" "$rc" "$budget" <<'PY'
import sys,json
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1]);s=sorted(p.glob('segment-*.json'))
if s:
 r=json.loads(s[-1].read_text());r.update(process_rc=int(sys.argv[2]),timeout_s=int(sys.argv[3]),launch_finished_utc=datetime.now(timezone.utc).isoformat());s[-1].write_text(json.dumps(r,indent=2)+'\n')
PY
exit "$rc"
