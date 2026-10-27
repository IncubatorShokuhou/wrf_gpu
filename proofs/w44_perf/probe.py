"""CPU dry-run target: validate the actual command environment without JAX/NVML."""
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
keys = ("JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR",
        "GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION",
        "GPUWRF_WRF_ROOT", "GPUWRF_CENSUS", "GPUWRF_NEST_PERF_TIMERS", "W44_CAPTURE_ROOTS", "W44_CAPTURE_START",
        "BENCH_SRC_REPO", "PYTHONPATH", "TMPDIR", "XLA_FLAGS", "W44_POOL", "W44_INIT_OBSERVER", "W44_HOST_PROFILE")
rec = {"argv": sys.argv, "affinity": sorted(os.sched_getaffinity(0)),
       "env": {k: os.environ.get(k) for k in keys}, "gpu_executed": False}
assert rec["env"]["JAX_PLATFORMS"] == "cpu"
if rec["env"]["W44_POOL"] == "auto":
    assert rec["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] is None
    assert rec["env"]["XLA_CLIENT_MEM_FRACTION"] is None
else:
    assert rec["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert rec["env"]["XLA_CLIENT_MEM_FRACTION"] == "0.25"
assert "jax" not in sys.modules
(out / "PROBE.json").write_text(json.dumps(rec, indent=2) + "\n")
print("CPU PROBE PASS")
