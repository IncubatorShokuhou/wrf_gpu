"""E2E (release defaults, Pallas interpret via DIAG_INTERPRET=1): TDFI pre-pass on the Swiss case (CPU), then compare with the CPU-WRF oracle filtered state.

usage: [DIAG_INTERPRET=1] JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python run_tdfi_e2e.py <src_root> <tag> <oracle_dir>
(case: Swiss RD11 CPU inputs; writes <USER_HOME>/wrf_gpu2_lanes/o2-dfi/<tag>; oracle_dir supplies namelist.input)
"""
import json
import os
import sys
import time
from pathlib import Path

SRC, TAG, ORACLE = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402
import numpy as np  # noqa: E402

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
from gpuwrf.dfi.driver import run_tdfi  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert pipeline.__file__.startswith(SRC), pipeline.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o2-dfi") / TAG
LANE.mkdir(parents=True, exist_ok=True)
case = LANE / "case"
case.mkdir(exist_ok=True)
for f in CASE.iterdir():
    if f.name.startswith(("wrfout", "rsl.")) or f.name == "namelist.input":
        continue
    if not (case / f.name).exists():
        (case / f.name).symlink_to(f)
(case / "namelist.input").write_text((ORACLE / "namelist.input").read_text())
cfg = pipeline.NestedPipelineConfig(case, LANE / "out", LANE / "proof", hours=1, max_dom=1)
t0 = time.time()
fst_dir, report = run_tdfi(cfg, work_dir=LANE / "work")
report["wall_s"] = round(time.time() - t0, 1)
report["fast_defaults"] = os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset>")
(LANE / "report.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps({k: v for k, v in report.items() if not k.endswith("hcoeff")}, indent=1, default=str))
