"""Run the native forecast (CPU) from an input dir through execute_nested_pipeline.

usage: [DIAG_INTERPRET=1] JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python run_forecast.py <src_root> <input_dir> <out_dir> <hours>
"""
import json
import sys
import time
from pathlib import Path

SRC, INP, OUT, HOURS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
import os as _os
if _os.environ.get("DIAG_INTERPRET", "0") == "1":
    from jax.experimental import pallas as _pl
    _real_pc = _pl.pallas_call

    def _interp_pc(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pc(*a, **k)

    _pl.pallas_call = _interp_pc
    import jax.experimental.pallas.triton as _pt  # o1-smag3d CPU shim for the PTX "<op>.rn.f32" inline asm

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op = asm.split()[0].split(".")[0]
        a, b = (jax.numpy.asarray(x, jax.numpy.float64) for x in args)
        exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if jax.numpy.dtype(out_dtype) == jax.numpy.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert pipeline.__file__.startswith(SRC), pipeline.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
OUT.mkdir(parents=True, exist_ok=True)
cfg = pipeline.NestedPipelineConfig(INP, OUT / "wrfout", OUT / "proof", hours=HOURS, max_dom=1,
                                    scratch_dir=OUT / "scratch", emit_initial_history=True, aot_prefetch=False)
t0 = time.time()
payload = pipeline.execute_nested_pipeline(cfg)
summary = {"wall_s": round(time.time() - t0, 1), "verdict": payload.get("verdict"),
           "dfi": payload.get("dfi"), "keys": sorted(payload)[:40]}
(OUT / "fc_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
