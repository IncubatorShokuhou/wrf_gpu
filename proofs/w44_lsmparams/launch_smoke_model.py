"""Private launch-smoke device boundary for the REAL root_gpu.py model path.

Only the private smoke lock permits this substitution. The product child env is
recorded and checked first, then ONLY the JAX device selector changes cuda->cpu
before JAX is imported. The model code that follows is the unchanged GPU path;
this never writes a native prime/measure or E41 PASS receipt.
"""
import os
from pathlib import Path
import sys
import time

NAMES = ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "GPUWRF_FAST_DEFAULTS", "GPUWRF_JAX_CACHE",
         "GPUWRF_AOT_SLIM_MODULE", "GPUWRF_CENSUS", "GPUWRF_CENSUS_SLIM", "GPUWRF_RRTMG_MP_RE",
         "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION",
         "JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR",
         "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORM_NAME")
EXPECTED = {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0", "GPUWRF_FAST_DEFAULTS": "1",
            "GPUWRF_JAX_CACHE": "1", "GPUWRF_AOT_SLIM_MODULE": "0", "GPUWRF_CENSUS": "0",
            "GPUWRF_CENSUS_SLIM": "0", "GPUWRF_RRTMG_MP_RE": "1", "OMP_NUM_THREADS": "4",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async", "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": ".25", "XLA_FLAGS": None, "JAX_PLATFORM_NAME": None}


def substitute_device():
    root = Path(os.environ["W2_LAUNCH_SMOKE_ROOT"]).resolve()
    assert Path(os.environ["GPUWRF_GPU_LOCK_FILE"]).resolve() == root / "private.lock"
    assert os.environ["GPUWRF_GPU_LOCK_HELD"] == "1"
    assert "jax" not in sys.modules, "device substitution must precede JAX import"
    if os.environ.get("W2_SMOKE_CAPPED") != "1":
        # CPU compile of the real program: take one shared 12G RAM slot (long FIFO wait allowed)
        # before any model import; the enclosing subreaper keeps holding the private flock.
        os.environ.update(W2_SMOKE_CAPPED="1", W2_SMOKE_STARTED=str(time.monotonic()))
        source = Path(__file__).resolve().parents[2]
        command = [str(source / "scripts/with_ram_slot.sh"), "--label", "w44-lsmparams-smoke",
                   "--max", "12G", "--timeout", "1800", "--", sys.executable,
                   str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        os.execv(command[0], command)
    product_env = {name: os.environ.get(name) for name in NAMES}
    assert all(product_env[name] == value for name, value in EXPECTED.items()), product_env
    from jaxlib.xla_client import generate_pjrt_gpu_plugin_options  # no device init
    allocator = generate_pjrt_gpu_plugin_options()
    assert allocator["memory_fraction"] == .25 and allocator["preallocate"] is True
    os.environ["JAX_PLATFORMS"] = "cpu"  # the only product device/environment substitution
    return {"launch_smoke": True, "product_child_env": product_env, "gpu_allocator_options": allocator,
            "device_substitution": {"JAX_PLATFORMS": ["cuda", "cpu"]},
            "native_GPU_E41_and_timing_qualified": False}
