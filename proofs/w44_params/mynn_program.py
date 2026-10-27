"""Whole native MYNN program: exact OFF HLO and compiled default-value gate.

For raw OFF comparison, launch this SAME file twice with one stable --import-src
symlink pointing sequentially to immutable old/new trees (E58/E109).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--import-src", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--action", choices=("lower", "evaluate"), required=True)
parser.add_argument("--cohort", default="sea_sc12")
parser.add_argument("--value", type=float)
parser.add_argument("--family", choices=("sqfac", "coefficients"), default="sqfac")
args = parser.parse_args()
# Do not resolve this alias: identical source locations are part of the raw gate.
sys.path.insert(0, str(args.import_src))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.backends import pallas as facade
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes

platform = jax.devices()[0].platform
assert platform == "cpu", "this tool is a CPU compile/lower proof, not a GPU run"
assert os.environ["GPUWRF_FAST_DEFAULTS"] == "1"
assert os.environ.get("GPUWRF_MYNN_FP32_COLUMNS") == "1", "native whole-MYNN route is not enabled"
assert os.environ.get("GPUWRF_MYNN_TRIDIAG_REAL") == "1", "native WRF tridiagonal route is not enabled"
raw = args.fixture.read_bytes()
row = next(r for r in json.loads(raw)["rows"] if r["label"] == args.cohort)
array = lambda v: None if v is None else jnp.asarray(v, jnp.float32)
state = P.MynnPBLColumnState(**{k: array(v) for k, v in row["state"].items()})
surface = SurfaceFluxes(**{k: array(v) for k, v in row["flux"].items()})
traces = []


def program(state, surface, factor):
    traces.append(1)
    kwargs = {} if factor is None else ({"sqfac": factor} if args.family == "sqfac" else {
        "sqfac": factor["sqfac"],
        "mynn_coefficients": {name: value for name, value in factor.items() if name != "sqfac"},
    })
    return P.step_mynn_pbl_column_with_pblh(state, row["dt"], surface=surface,
                                            edmf=True, dx=row["dx"], **kwargs)


fn = jax.jit(program)


def operands(value):
    if value is None:
        return None
    if args.family == "sqfac":
        return jnp.float32(value)
    return {name: jnp.float32(default * value) for name, default in
            {"sqfac": 3., "alp1": .23, "cphm_unst": 16., "am_st": 6.1, "entrainment": .33}.items()}


args.out.mkdir(parents=True, exist_ok=True)
receipt = {"source": str(Path(P.__file__).resolve()), "platform": platform, "gpu_executed": False,
           "fixture_sha256": hashlib.sha256(raw).hexdigest(), "cohort": args.cohort, "family": args.family,
           "work": {"ncol": state.u.shape[0], "nz": state.u.shape[1], "dt": row["dt"], "edmf": True,
                    "dx": row["dx"], "native": True, "release_defaults": True}}
if args.action == "lower":
    facade.cuda_native = lambda: True
    jax.default_backend = lambda: "gpu"  # E115: native lowering only; jax.devices remains CPU.
    factor = operands(args.value)
    with jax.enable_x64(False):
        trace = fn.trace(state, surface, factor)
        lowered = trace.lower(lowering_platforms=("cuda",))
        hlo = lowered.compiler_ir("hlo").as_hlo_text()
        jp = str(jax.make_jaxpr(program)(state, surface, factor))
    canon = re.sub(r"name_and_src_info=[^\]\n]*", "", jp)
    canon = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", canon)
    receipt.update(raw_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                   canonical_jaxpr_sha256=hashlib.sha256(canon.encode()).hexdigest(),
                   raw_hlo_bytes=len(hlo.encode()), value=args.value,
                   native_custom_calls=hlo.count("custom-call("))
    assert receipt["native_custom_calls"] > 0
    (args.out / "raw.hlo").write_text(hlo)
    (args.out / "canonical.jaxpr").write_text(canon)
else:
    results = []
    named_results = []
    with jax.enable_x64(False):
        values = (None, 3., 6.) if args.family == "sqfac" else (None, 1., 1.1)
        for value in values:
            factor = operands(value)
            result = fn(state, surface, factor)
            results.append([np.asarray(a).copy() for a in jax.tree.leaves(result)])
            names = {}
            for name in result[0].__slots__:
                leaves = jax.tree.leaves(getattr(result[0], name))
                for index, leaf in enumerate(leaves):
                    names[name if len(leaves) == 1 else f"{name}[{index}]"] = np.asarray(leaf).copy()
            names["pblh"] = np.asarray(result[1]).copy()
            named_results.append(names)
    assert traces == [1, 1], traces
    assert all(np.isfinite(a).all() for result in results for a in result)
    receipt["default_byte_exact"] = all(a.tobytes() == b.tobytes() for a, b in zip(results[0], results[1], strict=True))
    receipt["changed_leaves"] = sum(a.tobytes() != b.tobytes() for a, b in zip(results[1], results[2], strict=True))
    receipt["traces"] = len(traces)
    receipt["output_leaves"] = len(results[0])
    from root_gpu import differences
    receipt["default_differences"] = differences(named_results[0], named_results[1])
    receipt["changed_differences"] = differences(named_results[1], named_results[2])
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    assert receipt["default_byte_exact"], "default differs in compiled whole native MYNN; measure/localize before classifying"
    assert receipt["changed_leaves"] > 0
(args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
