"""w41-swddir perf note (CPU only, E137/E153): the default SW band kernel before/after the ztdbt_nodel direct output.

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 python cubin_probe.py KERNEL_SOURCE TAG OUT_DIR [NCOL]
KERNEL_SOURCE = a rad_sw_band_sums.py (e.g. `git show main:src/gpuwrf/kernels/rad_sw_band_sums.py` for "before", the
lane file for "after"); it is installed as gpuwrf.kernels.rad_sw_band_sums before tracing, so both arms trace the SAME
rrtmg_sw caller.  Traces the release SW column solve (REAL entry + fused band sums + Thompson radii) on NCOL real RE01
columns, lowers every rrtmg_sw pallas_call to Triton TTIR for sm_120 and compiles it with triton.compile (timed, never
launched); records registers / stack / spill stores+loads (cuobjdump) and PTX/TTIR sizes.  No GPU execution claim.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOURCE, TAG, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
NCOL = int(sys.argv[4]) if len(sys.argv) > 4 else 64
assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/v025/w41_aerosol"))
OUT.mkdir(parents=True, exist_ok=True)

import jax  # noqa: E402
import triton  # noqa: E402
from jax._src.pallas.triton import lowering  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

import gpuwrf.kernels  # noqa: E402
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402

spec = importlib.util.spec_from_file_location("gpuwrf.kernels.rad_sw_band_sums", SOURCE)
K = importlib.util.module_from_spec(spec)
sys.modules["gpuwrf.kernels.rad_sw_band_sums"] = K
spec.loader.exec_module(K)
gpuwrf.kernels.rad_sw_band_sums = K

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


def column_state():
    fx, ae = load(RE01), load(SW)
    sub = {k: (v[:NCOL] if hasattr(v, "ndim") and v.ndim and v.shape[0] == fx["in_t"].shape[0] else v) for k, v in fx.items()}
    sub_ae = {k: (v[:NCOL] if hasattr(v, "ndim") and v.ndim and v.shape[0] == ae["coszen"].shape[0] else v)
              for k, v in ae.items()}
    captured = {}

    def fake_solve(st, **kw):
        captured["state"] = st
        raise StopIteration

    real = R.solve_rrtmg_sw_column
    R.solve_rrtmg_sw_column = fake_solve
    try:
        runner._solve(sub, sub_ae, 0, False)  # arm 0: aer_opt 0, the released program's operands
    except StopIteration:
        pass
    finally:
        R.solve_rrtmg_sw_column = real
    return captured["state"]


def main():
    st = column_state()
    jx = jax.make_jaxpr(lambda s: R.solve_rrtmg_sw_column(s, with_clear_sky=True))(st).jaxpr
    rows = []
    for i, eqn in enumerate(pallas_eqns(jx, [])):
        name = eqn.params.get("name_and_src_info") or eqn.params.get("name")
        name = str(getattr(name, "name", name))
        if "rrtmg_sw" not in name:
            continue
        t0 = time.perf_counter()
        mod = lowering.lower_jaxpr_to_triton_module(eqn.params["jaxpr"], eqn.params["grid_mapping"], "cuda", 120)
        ttir = OUT / f"{TAG}_{i}_{name}.ttir"
        ttir.write_text(mod.module.operation.get_asm(enable_debug_info=False))
        lower_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        kernel = triton.compile(str(ttir), target=GPUTarget("cuda", 120, 32), options={"num_warps": int(K.WARPS)})
        compile_s = time.perf_counter() - t1
        cubin = ttir.with_suffix(".cubin")
        cubin.write_bytes(kernel.asm["cubin"])
        res = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin)], text=True)
        sass = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "-sass", str(cubin)], text=True)
        ptx = kernel.asm["ptx"]
        rows.append(dict(tag=TAG, kernel=name, lower_s=round(lower_s, 2), compile_s=round(compile_s, 2),
                         resources=" ".join(res.split()), sass_lines=len(sass.splitlines()),
                         sass_stl=sass.count("STL"), sass_ldl=sass.count("LDL"), sass_mufu_ex2=sass.count("MUFU.EX2"),
                         ptx_lines=len(ptx.splitlines()), ttir_lines=len(ttir.read_text().splitlines())))
        print(rows[-1], flush=True)
    (OUT / f"{TAG}_receipt.json").write_text(json.dumps(dict(platform="CPU compiler only", gpu_executed=False,
                                                             source=str(SOURCE), ncol=NCOL, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
