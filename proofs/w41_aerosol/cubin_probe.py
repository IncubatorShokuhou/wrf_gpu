"""E137/E153 CPU-only cubin probe of the aer_opt SW band kernel (rad_sw_band_sums_aer) vs the released kernel.

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 python cubin_probe.py SRC_ROOT OUT_DIR [NCOL]
Traces the release SW column solver (REAL entry + fused band sums, MP_RE radii) on NCOL real RE01 columns with and
without aer_opt=1 / aer_opt=3 aerosol, finds every pallas_call in the jaxpr (recursively), lowers each to Triton TTIR
for sm_120 and compiles it with triton.compile (timed, never launched); records registers / stack / spills / PTX f64
lines per kernel (cuobjdump --dump-resource-usage).  No GPU execution claim.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NCOL = int(sys.argv[3]) if len(sys.argv) > 3 else 64
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path[:0] = [str(SRC / "src"), str(SRC / "tests/v025/w41_aerosol")]
OUT.mkdir(parents=True, exist_ok=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import triton  # noqa: E402
from jax._src.pallas.triton import lowering  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

from gpuwrf.physics import rrtmg_sw as R  # noqa: E402
import sw_oracle_runner as runner  # noqa: E402
from tegen_fixture import RE01, SW, load  # noqa: E402

assert R._real_entry() and R._sw_band_sums_mode(), "release SW path not selected (set GPUWRF_FAST_DEFAULTS=1)"


def pallas_eqns(jaxpr, out):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            out.append(eqn)
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    pallas_eqns(inner if hasattr(inner, "eqns") else inner.jaxpr, out)
    return out


def state(fx, ae, arm, kind):
    sub = {k: (v[:NCOL] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == fx["in_t"].shape[0] else v)
           for k, v in fx.items()}
    sub_ae = {k: (v[:NCOL] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == ae["coszen"].shape[0] else v)
              for k, v in ae.items()}
    if kind == "aer3":
        sub_ae = dict(sub_ae, arm_aer_opt=np.asarray([3] * len(sub_ae["arm_aer_opt"]), np.int32),
                      in_qnwfa=np.full(sub["in_t"].shape, 3e8, np.float32), in_qnifa=np.full(sub["in_t"].shape, 2e6, np.float32))
    elif kind == "off":
        sub_ae = dict(sub_ae, arm_aer_opt=np.zeros_like(sub_ae["arm_aer_opt"]))
    captured = {}

    def fake_solve(st, **kw):
        captured["state"] = st
        raise StopIteration

    real = R.solve_rrtmg_sw_column
    R.solve_rrtmg_sw_column = fake_solve
    try:
        runner._solve(sub, sub_ae, arm, False)
    except StopIteration:
        pass
    finally:
        R.solve_rrtmg_sw_column = real
    return captured["state"]


def main():
    fx, ae = load(RE01), load(SW)
    rows = []
    for kind in ("off", "aer1", "aer3"):
        st = state(fx, ae, 1, kind)
        jx = jax.make_jaxpr(lambda s: R.solve_rrtmg_sw_column(s, with_clear_sky=True))(st).jaxpr
        for i, eqn in enumerate(pallas_eqns(jx, [])):
            name = eqn.params.get("name_and_src_info") or eqn.params.get("name")
            name = str(getattr(name, "name", name))
            if "rrtmg_sw" not in name:
                continue
            t0 = time.perf_counter()
            mod = lowering.lower_jaxpr_to_triton_module(eqn.params["jaxpr"], eqn.params["grid_mapping"], "cuda", 120)
            ttir = OUT / f"{kind}_{i}_{name}.ttir"
            ttir.write_text(mod.module.operation.get_asm(enable_debug_info=False))
            lower_s = time.perf_counter() - t0
            from gpuwrf.kernels.rad_sw_band_sums import WARPS
            warps = WARPS  # both SW band kernels launch with the released kernel's num_warps
            t1 = time.perf_counter()
            kernel = triton.compile(str(ttir), target=GPUTarget("cuda", 120, 32), options={"num_warps": int(warps)})
            compile_s = time.perf_counter() - t1
            cubin = ttir.with_suffix(".cubin")
            cubin.write_bytes(kernel.asm["cubin"])
            res = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin)], text=True)
            ptx = kernel.asm["ptx"]
            rows.append(dict(kind=kind, kernel=name, lower_s=round(lower_s, 2), compile_s=round(compile_s, 2),
                             num_warps=int(warps), resources=" ".join(res.split()),
                             ptx_f64_lines=sum(".f64" in ln for ln in ptx.splitlines()),
                             ttir_lines=len(ttir.read_text().splitlines())))
            print(rows[-1], flush=True)
            (OUT / "receipt.json").write_text(json.dumps(dict(platform="CPU compiler only", gpu_executed=False,
                                                              ncol=NCOL, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
