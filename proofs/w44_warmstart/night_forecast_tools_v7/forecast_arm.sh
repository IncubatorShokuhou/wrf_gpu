#!/usr/bin/env bash
# Invoked only inside the official lock; NVML sampler is the GPU process's sibling.
set -euo pipefail
CASE=$1 OUT=$2 PHASE=$3 RUNNER=$4
if [[ ${W44_WARMSTART_CPU_PREFLIGHT:-0} == 1 ]]; then
  [[ ${JAX_PLATFORMS:-} == cpu ]] || exit 126
  exec python3 -u "$RUNNER" "$CASE" "$OUT/forecast" "$PHASE"
fi
[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]] || exit 126
mkdir -p "$OUT"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/foreign_start.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits --loop-ms=250 > "$OUT/vram.csv" &
SAMPLER=$!
finish() { kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true; }
trap finish EXIT
python3 -u "$RUNNER" "$CASE" "$OUT/forecast" "$PHASE"
