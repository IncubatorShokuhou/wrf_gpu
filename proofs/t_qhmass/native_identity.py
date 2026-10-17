"""CPU-only CUDA frontend gate and original/candidate native kernel identity."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.triton import lowering
from gpuwrf.backends import pallas as pl
from gpuwrf.kernels import phys_hydrostatic_real as module

ROOT, OUTPUT = (Path(x).resolve() for x in sys.argv[1:3])
assert jax.devices()[0].platform == "cpu"
pl.cuda_native = lambda: True
source = subprocess.check_output(["git", "show", "ad8b0c729:src/gpuwrf/kernels/phys_hydrostatic_real.py"], cwd=ROOT, text=True)
namespace = module.__dict__.copy()
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "parent.py", "exec"), namespace)
parent = namespace["hydrostatic_profiles"]
def normalized(ir):
    ir = re.sub(r"name_and_src_info=[^\]\n]*", "", ir)
    ir = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", ir)
    return ir
rows = []
for species in (6, 7, 8, 9):
    nz, ny, nx = 44, 42, 42
    moist = tuple(jax.ShapeDtypeStruct((nz,ny,nx), jnp.float32) for _ in range(species))
    vector = jax.ShapeDtypeStruct((nz,), jnp.float32)
    args = (jax.ShapeDtypeStruct((ny,nx), jnp.float32), moist, vector, vector, vector,
            jax.ShapeDtypeStruct((), jnp.float32))
    jp = jax.make_jaxpr(module.hydrostatic_profiles)(*args).jaxpr
    params = next(e.params for e in jp.eqns if e.primitive.name == "pallas_call")
    kernel = normalized(str(params["jaxpr"]))
    lowered = lowering.lower_jaxpr_to_triton_module(params["jaxpr"], params["grid_mapping"], "cuda", 120)
    ttir = str(lowered.module)
    row = dict(species=species, native_pallas_calls=1, frontend="PASS", TTIR_bytes=len(ttir),
               TTIR_sha256=hashlib.sha256(ttir.encode()).hexdigest(),
               runtime_f64_lines=[line.strip() for line in ttir.splitlines()
                                  if "f64" in line and "arith.constant" not in line and "arith.truncf" not in line])
    assert not row["runtime_f64_lines"]
    if species <= 8:
        old = jax.make_jaxpr(parent)(*args).jaxpr
        old_params = next(e.params for e in old.eqns if e.primitive.name == "pallas_call")
        row["parent_kernel_jaxpr_BYTE"] = kernel == normalized(str(old_params["jaxpr"]))
        assert row["parent_kernel_jaxpr_BYTE"], species
    rows.append(row)
OUTPUT.write_text(json.dumps(dict(platform="cpu", GPU_execution=False, source=ROOT.name, rows=rows), indent=2)+"\n")
