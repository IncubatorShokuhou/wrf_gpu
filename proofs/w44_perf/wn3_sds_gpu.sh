#!/usr/bin/env bash
# Run only inside the official lock, through the manager's nightly sequencer.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
SRC=${SDS_CANDIDATE_SOURCE:-$L/snap_sds_2c941de611}
EXPECTED=${SDS_CANDIDATE_COMMIT:-2c941de6119c13ff6f473bddaeeba89a76f0d741}
DRY=${1:-}
[[ -z "$DRY" || "$DRY" == --cpu-dry ]] || exit 2
[[ "$DRY" == --cpu-dry || ( "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ) ]] || exit 2
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$EXPECTED" ]] || exit 2
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no -- src)" ]] || exit 2
for v in $(compgen -e | grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | grep -v -E '^GPUWRF_GPU_(LOCK_[A-Z_]+|ARM_CPUS)$'); do unset "$v"; done
export PYTHONPATH="$SRC/src" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.166
export GPUWRF_JAX_CACHE=1 GPUWRF_JAX_CACHE_DIR="$P/jax" JAX_COMPILATION_CACHE_DIR="$P/jax"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$P/autotune" TRITON_CACHE_DIR="$P/triton" CUDA_CACHE_PATH="$P/cuda"
mkdir -p "$L/gpu/SDSWN3_tmp"
export TMPDIR="$L/gpu/SDSWN3_tmp"
if [[ "$DRY" == --cpu-dry ]]; then
 export JAX_PLATFORMS=cpu
 <USER_HOME>/miniconda3/bin/python - "$SRC" "$HERE" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
src,here=map(Path,sys.argv[1:])
assert 'jax' not in sys.modules and os.environ['JAX_PLATFORMS']=='cpu'
r={'gpu_started':False,'source_key_sha256':hashlib.sha256((src/'src/gpuwrf/runtime/aot_cheap_key.py').read_bytes()).hexdigest(),
   'script_sha256':hashlib.sha256((here/'wn3_sds_gpu.py').read_bytes()).hexdigest(),
   'pool':os.environ['XLA_CLIENT_MEM_FRACTION'],'cache':os.environ['JAX_COMPILATION_CACHE_DIR']}
Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/WN3_SDS_GPU_DRY.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
 exit 0
fi
timeout --signal=TERM --kill-after=30 240 <USER_HOME>/miniconda3/bin/python -I \
 "$HERE/../../scripts/release/bundle_consumer.py" --bundle "$P" --layer "$L/gpu/SDSWN3_layer" -- \
 <USER_HOME>/miniconda3/bin/python "$HERE/wn3_sds_gpu.py"
