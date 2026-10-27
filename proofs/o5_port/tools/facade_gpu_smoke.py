"""vendor_ops facade on a real non-sm_120 NVIDIA GPU (wz1080, GTX 1080 Ti sm_61; lane o5-port).

Asserts the selection on real hardware (vendor 'cuda', native facade callable, platform_triton_native,
rocm-only paths inert), then runs two release Pallas kernels THROUGH THE FACADE on the device (A2 route via
the box shim, O5_A2_ROUTE=1, because XLA refuses Triton below cc 8.0) and compares them with the same kernels
interpreted on the CPU device of the same process:
  * dyn_rk_fp32.stage_masses_fp32 (literal REAL adds via PTX .rn) -> must be bitwise;
  * dyn_rk_fp32.diagnose_pressure_fp32 is not used (needs full State); instead the ring-relax couple kernel
    boundary_apply._couple_scalar_real4_arrays (PTX mul/add .rn) -> bitwise vs the CPU f64+reduce_precision RN32.
usage: python facade_gpu_smoke.py OUT.json
"""
import json
import os
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import gpuwrf  # noqa: F401

out = {}
gpu = jax.devices()[0]
cpu = jax.devices("cpu")[0]
out["device"] = {"platform": gpu.platform, "kind": gpu.device_kind, "cc": getattr(gpu, "compute_capability", None)}
assert gpu.platform == "gpu"
from gpuwrf.backends import pallas as facade, platform as plat, vendor_ops  # noqa: E402
from jax.experimental import pallas as jpl  # noqa: E402

out["selection"] = {"runtime_backend": plat.runtime_backend(), "vendor": vendor_ops.vendor(),
                    "triton_native": facade.triton_native(), "platform_triton_native": vendor_ops.platform_triton_native(),
                    "pallas_call_is_original": facade.pallas_call is jpl.pallas_call,
                    "rocm_admitted_env": vendor_ops.rocm_native_admitted()}
sel = out["selection"]
assert sel["runtime_backend"] == "cuda" and sel["vendor"] == "cuda" and sel["pallas_call_is_original"], sel
from gpuwrf.kernels.dyn_rk_fp32 import stage_masses_fp32  # noqa: E402
from gpuwrf.coupling import boundary_apply  # noqa: E402

rng = np.random.default_rng(1080)
ny, nx, nz = 117, 267, 44  # PROD d02 plane
mub = rng.uniform(7e4, 1e5, (ny, nx)).astype(np.float32)
cur = rng.uniform(-300, 300, (ny, nx)).astype(np.float32)
ref = rng.uniform(-300, 300, (ny, nx)).astype(np.float32)
field = rng.uniform(1e-6, 2e-2, (nz, ny, nx)).astype(np.float32)
mu = rng.uniform(7e4, 1e5, (ny, nx)).astype(np.float32)
c1 = np.linspace(0.05, 1.0, nz, dtype=np.float32)
c2 = np.linspace(5e3, 0.0, nz, dtype=np.float32)


def bits(x):
    return np.asarray(x).view(np.uint32)


for label, fn, cpu_fn, args in (
        ("stage_masses_rk2", lambda a, b, c: stage_masses_fp32(a, b, c, rk_step=2),
         lambda a, b, c: stage_masses_fp32(a, b, c, rk_step=2, interpret=True), (mub, cur, ref)),
        ("scalar_couple_real4", boundary_apply._couple_scalar_real4_arrays.__wrapped__,
         boundary_apply._couple_scalar_real4_arrays.__wrapped__, (field, mu, c1, c2))):
    rec = {}
    t = time.time()
    g = jax.block_until_ready(jax.jit(fn)(*[jax.device_put(x, gpu) for x in args]))
    rec["gpu_s"] = round(time.time() - t, 2)
    jax.clear_caches()
    real_backend = jax.default_backend
    jax.default_backend = lambda: "cpu"  # the CPU reference takes the interpreted RN32 branch
    try:
        with jax.default_device(cpu):
            c = jax.block_until_ready(jax.jit(cpu_fn)(*[jax.device_put(x, cpu) for x in args]))
    finally:
        jax.default_backend = real_backend
    jax.clear_caches()
    gl, cl = jax.tree.leaves(g), jax.tree.leaves(c)
    rec["leaves"] = len(gl)
    rec["bitwise_equal_fraction"] = [float((bits(x) == bits(y)).mean()) for x, y in zip(gl, cl)]
    rec["max_abs_diff"] = [float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))) for x, y in zip(gl, cl)]
    rec["finite"] = all(bool(np.isfinite(np.asarray(x)).all()) for x in gl)
    out[label] = rec
    print(label, rec, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
os._exit(0)
