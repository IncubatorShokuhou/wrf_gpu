"""Pascal (sm_61) native-Pallas feasibility smoke on the wz1080 box (lane o5-port). REAL GPU, small.

1. device: platform/kind/compute capability;
2. the product literal-RN kernel (dyn_rk_fp32._rn chain: mul, add, div as PTX .rn.f32) on the DEFAULT route
   (XLA in-graph Triton, A1) -> expected refusal below cc 8.0;
3. the same kernel through the A2 route (stock Triton 3.6 -> PTX for the device cc -> jaxlib triton_kernel_call),
   result bitwise vs a NumPy binary32 reference of the same operation order;
4. plain (contractible) control on A2 for the FMA question.
usage: python pascal_smoke.py OUT.json   (box env: source ~/src/wrf_gpu2_env/activate.sh ~/src/o5port_tree)
"""
import json
import os
import sys
import time
import traceback

import numpy as np

out = {"argv": sys.argv}
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import gpuwrf  # noqa: E402,F401
from jax.experimental import pallas as jpl  # noqa: E402
from jax.experimental.pallas import triton as pt  # noqa: E402

dev = jax.devices()[0]
out["device"] = {"platform": dev.platform, "kind": dev.device_kind,
                 "compute_capability": getattr(dev, "compute_capability", None),
                 "client": str(getattr(dev.client, "platform_version", ""))}
assert dev.platform == "gpu", out
from gpuwrf.backends import vendor_ops  # noqa: E402
from gpuwrf.kernels._triton_kernel_call import KernelCallCompilerParams  # noqa: E402
from gpuwrf.kernels.dyn_rk_fp32 import _rn  # noqa: E402

out["vendor"] = vendor_ops.vendor()
N = 1 << 16
rng = np.random.default_rng(20261024)
a = rng.uniform(-1e3, 1e3, N).astype(np.float32)
b = rng.uniform(0.5, 7.0, N).astype(np.float32)
c = rng.uniform(-1e3, 1e3, N).astype(np.float32)
# binary32 reference, one rounding per op, same order: ((a*b)+c)/b
ref = ((np.float64(a) * b).astype(np.float32).astype(np.float64) + c).astype(np.float32)
ref = (ref.astype(np.float64) / b).astype(np.float32)


def literal(a_ref, b_ref, c_ref, o_ref):
    x, y, z = a_ref[...], b_ref[...], c_ref[...]
    o_ref[...] = _rn("div", _rn("add", _rn("mul", x, y, interpret=False), z, interpret=False), y, interpret=False)


def plain(a_ref, b_ref, c_ref, o_ref):
    x, y, z = a_ref[...], b_ref[...], c_ref[...]
    o_ref[...] = (x * y + z) / y


def run(body, params, label):
    rec = {}
    t = time.time()
    try:
        f = jax.jit(lambda x, y, z: jpl.pallas_call(
            body, out_shape=jax.ShapeDtypeStruct((N,), jnp.float32), grid=(N // 1024,),
            in_specs=[jpl.BlockSpec((1024,), lambda i: (i,))] * 3, out_specs=jpl.BlockSpec((1024,), lambda i: (i,)),
            compiler_params=params, name=label)(x, y, z))
        got = np.asarray(jax.block_until_ready(f(a, b, c)))
        rec.update(status="RAN", s=round(time.time() - t, 2), bitwise_equal=int((got.view(np.uint32) == ref.view(np.uint32)).sum()),
                   n=N, max_abs_diff=float(np.max(np.abs(got.astype(np.float64) - ref))),
                   finite=bool(np.isfinite(got).all()))
        hlo = f.lower(a, b, c).as_text()
        rec["route_in_hlo"] = ("triton_kernel_call" if "triton_kernel_call" in hlo else
                               "__gpu$xla.gpu.triton" if "__gpu$xla.gpu.triton" in hlo else "other")
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        rec.update(status="FAIL", s=round(time.time() - t, 2), error=f"{type(exc).__name__}: {str(exc)[:500]}",
                   tb=traceback.format_exc()[-800:])
    out[label] = rec
    print(label, {k: v for k, v in rec.items() if k != "tb"}, flush=True)


run(literal, pt.CompilerParams(num_warps=4), "A1_xla_literal")
run(literal, KernelCallCompilerParams(num_warps=4), "A2_kernelcall_literal")
run(plain, KernelCallCompilerParams(num_warps=4), "A2_kernelcall_plain")
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
os._exit(0)
