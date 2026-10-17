"""Compare compiled C5 and its parent on the same CPU caller operands, including the rare branch."""
from pathlib import Path
import ast
import importlib.util
import json
import os
import subprocess
import sys

root, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
sys.path[:0] = [str(root / "src"), str(root / "tests/v025/t_cfix_mynn")]
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts import state as sc
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics.surface_layer import _mynn_surface_exner_kwargs

assert jax.devices()[0].platform == "cpu"
sc._gpu_device = lambda: jax.devices("cpu")[0]
spec = importlib.util.spec_from_file_location("c5_test_builders", root / "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
for flag in ("GPUWRF_MYNN_FP32_COLUMNS", "GPUWRF_MYNN_PHY_EXNER", "GPUWRF_MYNN_FLTV_WRF"):
    os.environ[flag] = "1"
source = subprocess.check_output(["git", "show", "db78f9bd8c:src/gpuwrf/coupling/physics_couplers.py"], cwd=root, text=True)
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_mynn_driver_limited_handles")
namespace = dict(C.__dict__)
namespace["_mynn_surface_exner_kwargs"] = _mynn_surface_exner_kwargs
exec(compile(ast.Module(body=[node], type_ignores=[]), "parent_mynn_view", "exec"), namespace)
parent = namespace["_mynn_driver_limited_handles"]
grid = T.smoke._land_grid()
old_jit = jax.jit(lambda st, *h: parent(st, grid, *h))
new_jit = jax.jit(lambda st, *h: C._mynn_driver_limited_handles(st, grid, *h))
result = {"platform": "cpu", "reference": "parent db78f9bd8c helper compiled with the same current unchanged dependencies",
          "scope": "compiled CPU regression, not WRF fidelity or GPU BYTE", "cases": {}}
with pytest.MonkeyPatch.context() as patch:
    mixed, _ = T._extreme_noahmp_step(patch)
inside, _ = T._noahmp_step(grid_backed=True)
all_beyond = mixed.replace(hfx=jnp.full_like(mixed.hfx, 1500.), qfx=jnp.full_like(mixed.qfx, 1.2e-3))
for name, state in [("mixed_two_of_six", mixed), ("inside", inside), ("all_beyond", all_beyond)]:
    handles = tuple(jnp.asarray(getattr(state, k), jnp.float32) for k in ("theta_flux", "qv_flux", "fltv"))
    before = tuple(np.asarray(getattr(state, k)).copy() for k in ("hfx", "qfx", "theta_flux", "qv_flux", "fltv"))
    old, new = old_jit(state, *handles), new_jit(state, *handles)
    row = {}
    for field, a, b in zip(("flt", "flqv", "fltv"), old, new):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
        ulp = np.abs(a.astype(float) - b.astype(float)) / np.maximum(np.spacing(np.abs(a)), np.finfo(np.float32).tiny)
        row[field] = {"byte": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                      "max_ulp": float(ulp.max()), "parent": a.tolist(), "candidate": b.tolist()}
        if name == "inside":
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
            assert np.array_equal(b.view(np.uint32), np.asarray(handles[("flt", "flqv", "fltv").index(field)]).view(np.uint32))
    for old_state, field in zip(before, ("hfx", "qfx", "theta_flux", "qv_flux", "fltv")):
        assert np.array_equal(old_state, np.asarray(getattr(state, field)))
    result["cases"][name] = row
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({n: {f: (r["byte"], r["max_ulp"]) for f, r in rows.items()} for n, rows in result["cases"].items()}), flush=True)
