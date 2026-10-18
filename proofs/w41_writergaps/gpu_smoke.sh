#!/usr/bin/env bash
# w41-writergaps GPU smoke: Madrid BEP+BEM with PV+green roofs, CLI 9 h cold (00-09 UTC: night + morning PV), then CPU
# wrfout check and E41 gate.
# Usage: proofs/w41_writergaps/gpu_smoke.sh SNAPSHOT_TREE   (run from the lane worktree)
# Proves finite execution + written PV/GR history only; values are gated by the pristine oracle.
set -u
SRC=$1; G=<USER_HOME>/wrf_gpu2_lanes/w41-writergaps/gpu_smoke; HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
test -d "$SRC/src/gpuwrf" && test -e "$SRC/data/wrf_pristine" || { echo "bad snapshot $SRC"; exit 2; }
rm -rf "$G/out" && mkdir -p "$G/cache" "$G/autotune" "$G/out" "$G/scratch"
{ echo "snapshot=$SRC rev=$(git -C "$SRC" rev-parse HEAD) src_tree=$(git -C "$SRC" rev-parse HEAD:src) dirty=$(git -C "$SRC" status --porcelain | wc -l) script_sha=$(sha256sum "$0" | cut -c1-16)"; } > "$G/run_meta.txt"
cd "$SRC"
scripts/with_gpu_lock.sh --label w41-writergaps -- env PYTHONPATH="$SRC/src" GPUWRF_WRF_ROOT="$G/wrf_root" \
  GPUWRF_JAX_CACHE_DIR="$G/cache" JAX_COMPILATION_CACHE_DIR="$G/cache" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$G/autotune" \
  TMPDIR="$G/scratch" timeout 1080 python -c "
import sys,jax,time,json
t=time.time()
assert jax.devices()[0].platform=='gpu',jax.devices()
from gpuwrf.cli import main
rc=main(['run','--input-dir','$G/input','--output-dir','$G/out','--hours','9','--max-dom','1','--scratch-dir','$G/scratch'])
json.dump({'rc':rc,'device':str(jax.devices()[0]),'wall_s':time.time()-t},open('$G/gpu_run.json','w'))
sys.exit(rc)
" > "$G/gpu.log" 2>&1
echo "gpu rc=$?" >> "$G/run_meta.txt"
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$SRC/src" taskset -c 10,26,27 nice -n 19 timeout 900 python scripts/wave_cubin_gate.py "$G/cache" "$G/e41" --run-proof "$G/out/proofs/nested_pipeline_run.json" > "$G/e41.log" 2>&1
echo "e41 rc=$?" >> "$G/run_meta.txt"
JAX_PLATFORMS=cpu taskset -c 10,26,27 nice -n 19 timeout 300 python "$HERE/check_gpu_wrfout.py" "$G/out" "$G/wrfout_check.json" >> "$G/run_meta.txt" 2>&1
echo "check rc=$?" >> "$G/run_meta.txt"
cat "$G/run_meta.txt"
