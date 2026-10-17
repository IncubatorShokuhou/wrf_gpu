"""Independent C7 density API and compiled MYNN-view regression against parent 2ed8c04c26."""
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

assert jax.devices()[0].platform == "cpu"
sc._gpu_device = lambda: jax.devices("cpu")[0]
spec = importlib.util.spec_from_file_location("c7_builders", root / "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
for key in ("GPUWRF_MYNN_FP32_COLUMNS", "GPUWRF_MYNN_PHY_EXNER", "GPUWRF_MYNN_FLTV_WRF"):
    os.environ[key] = "1"
source = subprocess.check_output(["git", "show", "2ed8c04c26:src/gpuwrf/coupling/physics_couplers.py"], cwd=root, text=True)
names = ("_wrf_phy_prep_rho_from_state", "_mynn_driver_limited_handles")
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 2
namespace = dict(C.__dict__)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "parent_density_and_view", "exec"), namespace)
parent_rho, parent_view = (namespace[n] for n in names)
grid = T.smoke._land_grid()
with pytest.MonkeyPatch.context() as patch:
    mixed, _ = T._extreme_noahmp_step(patch)
inside, _ = T._noahmp_step(grid_backed=True)
all_beyond = mixed.replace(hfx=jnp.full_like(mixed.hfx, 1500.), qfx=jnp.full_like(mixed.qfx, 1.2e-3))
result = {"platform": "cpu", "parent": "2ed8c04c26", "scope": "CPU regression; unchanged WRF oracles reused, not GPU BYTE",
          "density": [], "views": []}
for input_dtype in (jnp.float32, jnp.float64):
    st = mixed.replace(**{k: jnp.asarray(getattr(mixed, k), input_dtype) for k in ("mu_total", "ph_total", "qv")})
    for output_dtype in (jnp.float32, jnp.float64):
        for alt in (False, True):
            old = jax.jit(lambda x: parent_rho(x, grid.metrics, output_dtype=output_dtype, return_alt=alt))(st)
            full = jax.jit(lambda x: C._wrf_phy_prep_rho_from_state(x, grid.metrics, output_dtype=output_dtype, return_alt=alt))(st)
            a, b = np.asarray(old), np.asarray(full)
            assert a.dtype == b.dtype and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            assert a.tobytes() == b.tobytes()
            row = dict(input_dtype=str(input_dtype), output_dtype=str(output_dtype), return_alt=alt, full_byte=True)
            if not alt:
                kts = np.asarray(jax.jit(lambda x: C._wrf_phy_prep_rho_from_state(
                    x, grid.metrics, output_dtype=output_dtype, kts_only=True))(st))
                assert kts.shape == a.shape[1:] and np.isfinite(kts).all() and kts.tobytes() == a[0].tobytes()
                row["kts_byte_full_level0"] = True
            result["density"].append(row)
for exner_flag in ("1", "0"):
    os.environ["GPUWRF_MYNN_PHY_EXNER"] = exner_flag
    jax.clear_caches()  # import-time/runtime flags can be captured by the enclosing JIT.
    old_fn = jax.jit(lambda st, *h: parent_view(st, grid, *h))
    new_fn = jax.jit(lambda st, *h: C._mynn_driver_limited_handles(st, grid, *h))
    for name, state in (("inside", inside), ("mixed_two_of_six", mixed), ("all_beyond", all_beyond)):
        handles = tuple(jnp.asarray(getattr(state, k), jnp.float32) for k in ("theta_flux", "qv_flux", "fltv"))
        before = [np.asarray(getattr(state, k)).tobytes() for k in ("hfx", "qfx", "theta_flux", "qv_flux", "fltv")]
        old, new = old_fn(state, *handles), new_fn(state, *handles)
        for key, a, b in zip(("flt", "flqv", "fltv"), old, new):
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
            ulp = np.abs(a.astype(float) - b.astype(float)) / np.maximum(np.spacing(np.abs(a)), np.finfo(np.float32).tiny)
            result["views"].append(dict(exner=exner_flag, case=name, field=key,
                                        byte=a.tobytes() == b.tobytes(), max_ulp=float(ulp.max())))
        assert before == [np.asarray(getattr(state, k)).tobytes() for k in ("hfx", "qfx", "theta_flux", "qv_flux", "fltv")]
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"full_density_byte": len(result["density"]),
                  "kts_byte": sum(r.get("kts_byte_full_level0", False) for r in result["density"]),
                  "view_arrays_byte": sum(r["byte"] for r in result["views"]),
                  "view_arrays": len(result["views"]), "max_ulp": max(r["max_ulp"] for r in result["views"])}), flush=True)
assert all(r["max_ulp"] <= 1 for r in result["views"])
