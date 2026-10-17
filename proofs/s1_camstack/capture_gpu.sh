#!/usr/bin/env bash
# PREPARED ONLY: manager-scheduled <=15 min GPU arm, followed by CPU inspection.
# Usage: bash proofs/s1_camstack/capture_gpu.sh /absolute/detached/snapshot
set -euo pipefail
snapshot=$(realpath "${1:?pass a detached snapshot containing this script}")
if git -C "$snapshot" symbolic-ref -q HEAD >/dev/null; then
  echo 'REFUSE: use a detached snapshot' >&2
  exit 2
fi
if [[ -n $(git -C "$snapshot" status --porcelain --untracked-files=no) ]]; then
  echo 'REFUSE: snapshot has tracked modifications' >&2
  exit 2
fi
if [[ ! -e "$snapshot/data/wrf_pristine" ]]; then
  echo 'REFUSE: snapshot needs data/wrf_pristine -> <USER_HOME>/src/wrf_pristine' >&2
  exit 2
fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
# Fixed private paths make cleanup explicit; preserve an earlier capture.
out=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-capture
if [[ -e "$out" ]]; then echo 'REFUSE: gpu-capture already exists' >&2; exit 2; fi
mkdir -p "$out/cache/jax" "$out/cache/autotune" "$out/cache/triton" "$out/cache/cuda" "$out/cache/tmp" "$out/dump"
git -C "$snapshot" rev-parse HEAD > "$out/revision.txt"
cd "$snapshot"
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
ulimit -c 0
# One fresh Python process asserts GPU in child(); no CUDA-holding parent.
# Dump native-target configuration, original lowering/options and optimized HLO.
set +e
timeout 960 scripts/with_gpu_lock.sh --timeout 60 --label s1-camstack -- \
  env JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=1 GPUWRF_FAST_DEFAULTS=1 \
  GPUWRF_JAX_CACHE_DIR="$out/cache/jax" JAX_COMPILATION_CACHE_DIR="$out/cache/jax" \
  GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$out/cache/autotune" \
  TRITON_CACHE_DIR="$out/cache/triton" CUDA_CACHE_PATH="$out/cache/cuda" TMPDIR="$out/cache/tmp" \
  XLA_FLAGS="--xla_dump_to=$out/dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=advance_chunk --xla_gpu_dump_autotune_results_to=$out/autotune_results.pb" \
  PYTHONPATH="$snapshot/src" timeout 900 python proofs/s1_camstack/capture_gpu.py \
  --out "$out" > "$out/gpu.log" 2>&1
gpu_rc=$?
set -e
echo "$gpu_rc" > "$out/gpu.exit"
# Lock is released before CPU scan. If QUIET starts, leave artifacts for resume.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set +e
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$snapshot/src" \
  taskset -c 28 nice -n 19 timeout 900 python proofs/s1_camstack/inspect_cache.py \
  "$out/cache/jax" "$out/e41" --dump "$out/dump" > "$out/inspect.log" 2>&1
scan_rc=$?
set -e
echo "$scan_rc" > "$out/inspect.exit"
# Manager 2026-10-13 11:07Z: retain target/tools AND actual executables until
# the critic is done (E177). These remain inputs to the deviceless reproduction.
if [[ "$gpu_rc" != 0 ]]; then exit "$gpu_rc"; fi
exit "$scan_rc"
