#!/usr/bin/env bash
# Manager granted 18:08Z: start before18:45Z, <=480s; reuse original arm.
set -euo pipefail
snapshot=$(realpath "${1:?detached final candidate snapshot}")
out=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-final
if git -C "$snapshot" symbolic-ref -q HEAD >/dev/null ||
   [[ -n $(git -C "$snapshot" status --porcelain --untracked-files=no) ]]; then
  echo 'REFUSE: clean detached snapshot required' >&2; exit 2
fi
test -e "$snapshot/data/wrf_pristine"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [[ -e "$out" ]]; then echo 'REFUSE: gpu-final already exists'; exit 2; fi
mkdir -p "$out/cache/jax" "$out/cache/autotune" "$out/triton" "$out/cuda" "$out/tmp" "$out/dump"
git -C "$snapshot" rev-parse HEAD > "$out/revision.txt"
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
ulimit -c 0
cd "$snapshot"
# The lease and the manager's explicit followup window are both required.
# Queue without interrupting holders; actual GPU execution is capped at480 s.
timeout 4200 scripts/with_gpu_lock.sh --timeout 3600 --label s1-camstack -- \
  timeout 480 bash -c '
    if [[ $(date -u +%s) -ge $(date -u -d 2026-10-13T18:45:00Z +%s) ]]; then
      echo "REFUSE: GPU window closed" >&2; exit 2
    fi
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
    out=<USER_HOME>/wrf_gpu2_lanes/s1-camstack/gpu-final
    exec env JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=1 GPUWRF_FAST_DEFAULTS=1 \
      GPUWRF_JAX_CACHE_DIR="$out/cache/jax" JAX_COMPILATION_CACHE_DIR="$out/cache/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$out/cache/autotune" \
      TRITON_CACHE_DIR="$out/triton" CUDA_CACHE_PATH="$out/cuda" TMPDIR="$out/tmp" \
      XLA_FLAGS="--xla_dump_to=$out/dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=advance_chunk" \
      PYTHONPATH="$1/src" python proofs/s1_camstack/measure_gpu.py --snapshot "$1" --out "$out"
  ' bash "$snapshot" > "$out/gpu.log" 2>&1
