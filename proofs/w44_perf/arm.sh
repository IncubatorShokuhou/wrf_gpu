#!/usr/bin/env bash
# One immutable-source PROD arm. Invoke ONLY inside the official GPU lock after
# manager GO. `--dry-run` exercises the command environment using CPU probe.py.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
PY=<USER_HOME>/miniconda3/bin/python
VERSION=${1:?version}; ARM=${2:?arm label}; KIND=${3:-warm}; DRY=${4:-}
SMOKE=${W44_LAUNCH_SMOKE:-0}
[[ "$ARM" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
[[ "$KIND" =~ ^(prime|warm|graph|nodes|radnodes|auto|host)$ ]] || exit 2
[[ "$DRY" == --dry-run || "$SMOKE" == 1 || "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo 'official GPU lock required'; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ] && ! grep -q '^w44-perf ' /tmp/wrf_gpu2_quiet; then echo QUIET; exit 3; fi
case "$VERSION" in
  v032) SRC=$L/snap_v032; EXPECT=5f682e5ae933d569ebbeb11d78f119ba291f9283; BUNDLE= ;;
  v040) BUNDLE=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0; SRC=$BUNDLE/src; EXPECT=7bb5ae669b2e4b863d6bafdd659a03fea026004f ;;
  v043) BUNDLE=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3; SRC=$BUNDLE/src; EXPECT=888804310bb6bfc0237177c0d5e5452a29d5abf6 ;;
  *) exit 2 ;;
esac
# v043 bundle's actual pin is source-identical to its release tag.
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$EXPECT" ]] || { echo 'source HEAD changed'; exit 4; }
if git -C "$SRC" symbolic-ref --quiet HEAD >/dev/null; then echo 'source must be detached'; exit 4; fi
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no -- src scripts)" ]] || { echo 'source dirty'; exit 4; }
for v in $(compgen -e | grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | grep -v -E '^GPUWRF_GPU_(LOCK_[A-Z_]+|ARM_CPUS)$'); do unset "$v"; done
export GPUWRF_JAX_CACHE=1
unset CUDA_VISIBLE_DEVICES
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.25
export W44_POOL=explicit
unset W44_CAPTURE_ROOTS W44_CAPTURE_START W44_INIT_OBSERVER W44_HOST_PROFILE
export BENCH_SRC_REPO="$SRC" PYTHONPATH="$SRC/src" PYTHONDONTWRITEBYTECODE=1
OUT=${W44_RUN_ROOT:-$L/gpu}/runs/$ARM
if [[ "$SMOKE" == 1 ]]; then
  [[ "${W44_SMOKE_ID:-}" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
  [[ "${GPUWRF_GPU_LOCK_HELD:-0}" != 1 ]] || exit 2
  OUT=${W44_RUN_ROOT:-$L/launch_smoke/$W44_SMOKE_ID}/runs/$ARM
fi
if [[ "$DRY" == --dry-run ]]; then OUT=$L/dry/runs/$ARM; export JAX_PLATFORMS=cpu; fi
[[ ! -e "$OUT" ]] || { echo 'arm already exists'; exit 4; }
mkdir -p "$OUT" "$L/gpu/tmp"
chmod 700 "$OUT"
export TMPDIR=$L/gpu/tmp
if [[ -n "$BUNDLE" ]]; then
  export GPUWRF_JAX_CACHE_DIR=$BUNDLE/jax JAX_COMPILATION_CACHE_DIR=$BUNDLE/jax
  export GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$BUNDLE/autotune TRITON_CACHE_DIR=$BUNDLE/triton CUDA_CACHE_PATH=$BUNDLE/cuda
else
  export GPUWRF_JAX_CACHE_DIR=$L/gpu/cache_v032/jax JAX_COMPILATION_CACHE_DIR=$L/gpu/cache_v032/jax
  export GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$L/gpu/cache_v032/autotune
  export TRITON_CACHE_DIR=$L/gpu/cache_v032/triton CUDA_CACHE_PATH=$L/gpu/cache_v032/cuda
fi
if [[ "$KIND" == auto ]]; then
  [[ "$VERSION" == v043 ]] || exit 2
  unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_CLIENT_MEM_FRACTION
  export W44_POOL=auto
fi
TMO=180; [[ "$KIND" == prime ]] && TMO=1500
H=("$PY" "$HERE/capture.py" --input-dir "${W44_INPUT_CASE:-<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725}" --out "$OUT/run" --hours 3 --tag "$ARM")
if [[ "$DRY" == --dry-run ]]; then H=("$PY" "$HERE/probe.py" "${H[@]:2}"); fi
NSYS=()
if [[ "$KIND" == graph || "$KIND" == nodes || "$KIND" == radnodes || "$KIND" == host ]]; then
  GRAPH=graph
  if [[ "$KIND" == nodes || "$KIND" == radnodes ]]; then GRAPH=node; export W44_CAPTURE_ROOTS=3; fi
  if [[ "$KIND" == radnodes ]]; then export W44_CAPTURE_START=100 W44_INIT_OBSERVER=1; fi
  if [[ "$KIND" == host ]]; then export W44_HOST_PROFILE=1; fi
  H+=(--profile-segment 2)
  NSYS=(nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace="$GRAPH"
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$OUT/profile")
  if [[ "$KIND" == graph ]]; then NSYS+=(--python-sampling=true --python-sampling-frequency=1000); fi
  if [[ "$DRY" == --dry-run || "$SMOKE" == 1 ]]; then NSYS=(); fi
fi
CONSUMER=()
if [[ -n "$BUNDLE" && "$DRY" != --dry-run ]]; then
  LAYER=${W44_RUN_ROOT:-$L/gpu}/layer_$VERSION
  if [[ "$SMOKE" == 1 ]]; then LAYER=${W44_RUN_ROOT:-$L/launch_smoke/$W44_SMOKE_ID}/layer_$VERSION; fi
  CONSUMER=("$PY" -I "$HERE/../../scripts/release/bundle_consumer.py" --bundle "$BUNDLE" --layer "$LAYER" --)
fi
if [[ -z "$BUNDLE" && -n "${W44_RUN_ROOT:-}" ]]; then
  CONSUMER=(bash "$HERE/old_cache_view.sh" "$W44_RUN_ROOT/layer_v032")
fi
cd "$SRC"
date +%s.%N > "$OUT/launch_unix"
if [[ "$DRY" == --dry-run ]]; then
  timeout 30 "${H[@]}" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  cat "$OUT/stdout.log"
  exit 0
fi
if [[ "$SMOKE" == 1 ]]; then
  timeout --signal=TERM --kill-after=30 "$TMO" "${CONSUMER[@]}" "${H[@]}" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  date +%s.%N > "$OUT/end_unix"
  exit 0
fi
DMON=
cleanup() { [[ -z "$DMON" ]] || { kill -TERM "$DMON" 2>/dev/null || true; wait "$DMON" 2>/dev/null || true; }; chmod -R go-rwx "$OUT"; }
trap cleanup EXIT
nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/dmon.txt" 2>&1 & DMON=$!
# The lock FD is inherited through shell exec/timeout/consumer, never a Python
# subprocess with close_fds=True. This measured process owns the only CUDA pool.
timeout --signal=TERM --kill-after=30 "$TMO" "${CONSUMER[@]}" "${NSYS[@]}" "${H[@]}" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
date +%s.%N > "$OUT/end_unix"
"$PY" "$HERE/endpoint.py" "$OUT" > "$OUT/endpoint.json"
