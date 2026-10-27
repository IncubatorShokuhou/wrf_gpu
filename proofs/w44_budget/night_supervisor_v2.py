"""Versioned night protocol; retain the tested process-tree supervisor bytes."""
import hashlib
import importlib.util
from pathlib import Path
import sys

COMMON = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_5367b7fda6/night_supervisor.py')
assert hashlib.sha256(COMMON.read_bytes()).hexdigest() == 'e86766677bb962af73233e78a8f6ea03a61ef8ba33a59f17e2f672e6431e60f1'
spec = importlib.util.spec_from_file_location('w44_night_common', COMMON)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CAP, M.GPU_CAP, M.E41_CAP = 3600, 3000, 480
M.CPU_ENV.update(PYTHONPATH=str(M.SNAP/'src'), GPUWRF_AOT_KEEP_TRACE_METADATA='1')
PAID = '<USER_HOME>/wrf_gpu2_lanes/w44-budget/NIGHT_1792792718433431644_987496'
DRIVER = Path(__file__).with_name('night_probe_v2.py')
M.GPU_BODY = r'''
set -euo pipefail
mkdir -p "$RUN/cache" "$RUN/tmp"
ln -s <USER_HOME>/wrf_gpu2_lanes/w44-budget/NIGHT_1792792718433431644_987496/cache/jax "$RUN/cache/jax"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RUN/gpu_start.txt"
journalctl -k --no-pager 2>/dev/null | rg -i hibernat | tail -n 8 > "$RUN/hibernation.txt" || true
exec env -u JAX_PLATFORMS GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_KEEP_TRACE_METADATA=1 \
  GPUWRF_JAX_CACHE_DIR=PAID/cache/jax JAX_COMPILATION_CACHE_DIR=PAID/cache/jax \
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  GPUWRF_XLA_AUTOTUNE_CACHE_DIR=PAID/cache/jax/xla_gpu_per_fusion_autotune_cache_dir \
  TRITON_CACHE_DIR=PAID/cache/triton CUDA_CACHE_PATH=PAID/cache/cuda TMPDIR="$RUN/tmp" \
  XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.22 GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF \
  PYTHONPATH="$SNAP/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u DRIVER --out "$RUN/probe"
'''.replace('PAID',PAID).replace('DRIVER',str(DRIVER))

write = M.write_json
def write_v2(path, value):
    if path.name == 'night_receipt.json':
        value = dict(value, protocol_version=2, driver_sha256=hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
                     supervisor_v2_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     paid_cache=PAID+'/cache/jax', compile_order=['OFF','ON'], replay_order=['OFF','ON'])
    return write(path,value)
M.write_json = write_v2
run_group = M.run_group
def run_group_v2(command, log, limit, *, env=None):
    if Path(log).name == 'e41.log':
        command = [*command,'--run-proof',str(Path(log).parent/'probe/compiled_program_inventory.json')]
    return run_group(command,log,limit,env=env)
M.run_group = run_group_v2

if __name__ == '__main__':
    raise SystemExit(M.main())
