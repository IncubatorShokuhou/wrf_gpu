#!/usr/bin/env bash
# PREPARED: manager must grant a fresh <=15 min window before execution.
set -euo pipefail
candidate=$(realpath "${1:?detached candidate snapshot}")
base=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/capture-snapshot
tool=$(realpath "$(dirname "$0")/measure_gpu.py")
out=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-measure
for snapshot in "$base" "$candidate"; do
  if git -C "$snapshot" symbolic-ref -q HEAD >/dev/null ||
     [[ -n $(git -C "$snapshot" status --porcelain --untracked-files=no) ]]; then
    echo 'REFUSE: clean detached snapshots required' >&2; exit 2
  fi
  test -e "$snapshot/data/wrf_pristine"
done
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [[ -e "$out" ]]; then echo 'REFUSE: gpu-measure already exists'; exit 2; fi
mkdir -p "$out/base" "$out/candidate/cache/jax" "$out/candidate/cache/autotune"
# AOT admission can fall back and rewrite an HLO-addressed cache entry. Keep
# the captured reference immutable even when only diagnostic flags differ.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; cp -a --reflink=auto <USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-capture/cache "$out/base/cache"
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
ulimit -c 0
export CAMSTACK_MEASURE_TOOL="$tool" CAMSTACK_MEASURE_BASE="$base" CAMSTACK_MEASURE_CANDIDATE="$candidate"
# Sequential fresh processes; no CUDA-holding Python parent. Both processes
# write only owned measurement cache copies.
# Queue behind other authorized holders; only the command under the lock is
# the <=900 s GPU arm. Never interrupt a holder or alter its priority marker.
timeout 4500 "$candidate/scripts/with_gpu_lock.sh" --timeout 3600 --label s1-camstack -- \
  timeout 900 bash -c '
    set -euo pipefail
    if [[ $(date -u +%s) -ge $(date -u -d 2026-10-13T19:00:00Z +%s) ]]; then
      echo "REFUSE: GPU window closed" >&2; exit 2
    fi
    out=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-measure
    for arm in base candidate; do
      if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
      if [[ "$arm" == base ]]; then
        snapshot=$CAMSTACK_MEASURE_BASE
        cache=$out/base/cache
      else
        snapshot=$CAMSTACK_MEASURE_CANDIDATE
        cache=$out/candidate/cache
      fi
      mkdir -p "$out/$arm/tmp" "$out/$arm/triton" "$out/$arm/cuda" "$out/$arm/dump"
      git -C "$snapshot" rev-parse HEAD > "$out/$arm/revision.txt"
      env JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=1 GPUWRF_FAST_DEFAULTS=1 \
        GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
        GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/autotune" \
        TMPDIR="$out/$arm/tmp" TRITON_CACHE_DIR="$out/$arm/triton" CUDA_CACHE_PATH="$out/$arm/cuda" \
        XLA_FLAGS="--xla_dump_to=$out/$arm/dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=advance_chunk" \
        PYTHONPATH="$snapshot/src" python "$CAMSTACK_MEASURE_TOOL" --snapshot "$snapshot" \
        --out "$out/$arm" > "$out/$arm/gpu.log" 2>&1
    done
  '
