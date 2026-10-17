#!/usr/bin/env bash
set -u
# Run from an immutable snapshot, after the manager schedules a <=20 min window.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if JAX_PLATFORMS=cpu git symbolic-ref --quiet HEAD >/dev/null; then echo "Use an immutable detached snapshot"; exit 2; fi
if ! JAX_PLATFORMS=cpu git diff --quiet; then echo "Snapshot source is dirty"; exit 2; fi
export TSLIST_SOURCE_COMMIT="$(JAX_PLATFORMS=cpu git rev-parse HEAD)"
for helper in scripts/wave_cubin_gate.py scripts/e41/cubin_scan.py scripts/e41/hlo_scan.py; do
  if [ ! -f "$helper" ]; then echo "Missing required E41 helper: $helper"; exit 2; fi
done
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export GPUWRF_JAX_CACHE=1
export TSLIST_GPU_PREFIX="${TSLIST_GPU_PREFIX:-<USER_HOME>/wrf_gpu2_lanes/u2-tslist/TS_GPU}"
export GPUWRF_JAX_CACHE_DIR="${TSLIST_GPU_PREFIX}_cache"
export JAX_COMPILATION_CACHE_DIR="${TSLIST_GPU_PREFIX}_cache"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="${TSLIST_GPU_PREFIX}_autotune"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TSLIST_GPU_PREFIX}_tmp"
mkdir -p "$TMPDIR"
scripts/with_gpu_lock.sh --label u2-tslist -- timeout 1100 nsys profile \
 --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
 -o "${TSLIST_GPU_PREFIX}_trace" \
 python proofs/u2_tslist/gpu_arm.py "$TSLIST_GPU_PREFIX"
arm_rc=$?
if [ "$arm_rc" -ne 0 ]; then exit "$arm_rc"; fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 13 nice -n 19 timeout 900 python \
 scripts/wave_cubin_gate.py \
 "${TSLIST_GPU_PREFIX}_cache" "${TSLIST_GPU_PREFIX}_cubin"
gate_rc=$?
echo "E41 CPU scan exit status: $gate_rc"
if [ "$gate_rc" -eq 0 ]; then
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 13 nice -n 19 python - <<'PY'
import json,os
from pathlib import Path
gate=json.loads(Path(os.environ['TSLIST_GPU_PREFIX']+'_cubin/wave_gate.json').read_text())
assert gate['hard_stack_gate']=='PASS'
assert not gate['conservative_flags'],'E41 flags require source/native classification; not a complete PASS'
assert all(r['kinput_transpose_status']=='MEASURED_RICH_JIT_HLO' for r in gate['entries']),'E41 HLO coverage UNKNOWN'
PY
  gate_rc=$?
fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET; Nsight export/cache cleanup pending"; exit 3; fi
JAX_PLATFORMS=cpu taskset -c 13 nice -n 19 timeout 60 nsys stats --report cuda_gpu_mem_time_sum \
 "${TSLIST_GPU_PREFIX}_trace.nsys-rep" > "${TSLIST_GPU_PREFIX}_transfers.txt"
# The caller removes these exact cache/TMPDIR paths after the last dependent
# gate and trace audit. In particular, a scanner timeout MUST retain its input.
echo "Evidence retained for audit; clean ${TSLIST_GPU_PREFIX}_{cache,autotune,tmp} after final dependent check."
exit "$gate_rc"
