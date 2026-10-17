"""Lower the actual c62 CLI program on CPU, with native CUDA dispatch (E156).

No GPU executable or forecast is run. Initialization alone uses CPU-safe
interpretation; the product Pallas and tridiagonal solver are restored at trace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--input", type=Path, default=Path("<USER_HOME>/wrf_gpu2_lanes/v-prep/c62/input"))
p.add_argument("--groundwater-control", action="store_true", help="diagnostic source clone: replace constant negation with exact subtraction")
a = p.parse_args()
a.root = a.root.resolve()
assert os.environ["JAX_PLATFORMS"] == "cpu"
sys.path[:0] = [str(a.root / "src"), str(a.root / "proofs/o1_nlbind")]
import gpuwrf
import cpu_safe
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from gpuwrf.backends import pallas as facade
from gpuwrf.contracts import state as sc
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as op

assert jax.devices()[0].platform == "cpu"
assert op.__file__.startswith(str(a.root))
sc._gpu_device = lambda: jax.devices("cpu")[0]
original_solver = jax.lax.linalg.tridiagonal_solve
original_call = pl.pallas_call
cpu_safe.install()
facade.cuda_native = lambda: False
start = time.monotonic()
record = dict(source_commit=subprocess.check_output(["git", "-C", str(a.root), "rev-parse", "HEAD"], text=True).strip(),
              input=str(a.input), input_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in a.input.iterdir() if f.is_file()}, platform="cpu", lowering_platform="cuda", gpu_executed=False)
try:
    cfg = pipeline.NestedPipelineConfig(a.input, a.output.parent / "out", a.output.parent / "proof", hours=24, max_dom=1)
    _, bundles, _, _, _, carries = pipeline._load_domains(cfg, ("d01",))
    nl, carry = bundles["d01"].namelist, carries["d01"]
    record["load_s"] = time.monotonic() - start
    print("LOADED", nl.mp_physics, record["load_s"], flush=True)
    if a.groundwater_control:
        # Diagnostic only: no installed-compiler or checkout edits. Use the
        # original source function with exactly one operation substituted.
        import ast
        from gpuwrf.physics.noahmp import groundwater_options as gw
        source = Path(gw.__file__).read_text()
        body = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "groundwater_step")
        source = ast.get_source_segment(source, body)
        old = "jnp.exp(-timean)"
        new = "jnp.exp(jnp.asarray(-0., jnp.float32) - timean)"
        assert source.count(old) == 1
        node = next(n for n in ast.parse(source.replace(old, new)).body
                    if isinstance(n, ast.FunctionDef) and n.name == "groundwater_step")
        exec(compile(ast.Module(body=[node], type_ignores=[]), gw.__file__, "exec"), gw.__dict__)
        record["diagnostic_source_substitution"] = {"old": old, "new": new}
    facade.cuda_native = lambda: True
    jax.default_backend = lambda: "gpu"
    def native_call(*args, **kwargs):
        kwargs["interpret"] = False
        return original_call(*args, **kwargs)
    pl.pallas_call = native_call
    jax.lax.linalg.tridiagonal_solve = original_solver
    clock = op.build_clock_base(nl)
    traced = op._advance_chunk_fori.trace(carry, nl, jnp.asarray(1, jnp.int32), clock,
                                        n_steps=1, cadence=int(nl.radiation_cadence_steps))
    record["trace_s"] = time.monotonic() - start - record["load_s"]
    ir = traced.lower(lowering_platforms=("cuda",)).as_text()
    record.update(status="PASS", stablehlo_sha256=hashlib.sha256(ir.encode()).hexdigest(),
                  stablehlo_bytes=len(ir.encode()), triton_custom_calls=ir.count("__gpu$xla.gpu.triton"),
                  native_facade=True, pallas_interpret=False, python_backend="gpu",
                  work={n: getattr(nl, n) for n in ("mp_physics", "dt_s", "radiation_cadence_steps", "ferrier_gid")})
    assert record["triton_custom_calls"] > 0
except Exception as exc:
    record.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    traceback.print_exc()
record.update(wall_s=time.monotonic() - start, peakRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
a.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "traceback"}), flush=True)
assert record["status"] == "PASS", record.get("error")
