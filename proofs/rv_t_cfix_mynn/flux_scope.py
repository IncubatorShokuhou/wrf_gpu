"""CPU caller-seam witness: do MYNN-local limits reach the Tiedtke QFX operand?

No scientific forecast is run. A sentinel cumulus kernel returns its QFX argument
as RAINCV so that the actual adapter's input is observable without tracer stashes.
The actual Noah-MP producer is run twice with identical injected physical fluxes.
Only its local pbl_limits keyword is toggled for the matched control.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:0] = [str(root / "src"), str(root / "tests/v025/t_cfix_mynn")]
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts import state as state_contract
from gpuwrf.coupling import scan_adapters as A
from gpuwrf.physics import noahmp_coupler as NC

assert jax.devices()[0].platform == "cpu"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
spec = importlib.util.spec_from_file_location("f20_scope_helpers", root / "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
os.environ["GPUWRF_MYNN_FLTV_WRF"] = "1"
real_noah = NC.noah_mp_step
real_flux = NC.mynn_driver_surface_fluxes

def extreme_noah(*args, **kwargs):
    land, flux = real_noah(*args, **kwargs)
    return land, flux._replace(hfx=jnp.full_like(flux.hfx, 1500.),
                              qfx=jnp.full_like(flux.qfx, 1.2e-3),
                              lh=jnp.full_like(flux.lh, 3000.))

def sentinel(*args, **kwargs):
    z = jnp.zeros_like(args[0])
    keys = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN", "RUCUTEN", "RVCUTEN")
    return dict({key: z for key in keys}, RAINCV=args[14])

results = {}
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(NC, "noah_mp_step", extreme_noah)
    patch.setattr(A, "tiedtke_column_jax", sentinel)
    for limited in (False, True):
        def override(*args, **kwargs):
            return real_flux(*args, **dict(kwargs, pbl_limits=limited))
        patch.setattr(NC, "mynn_driver_surface_fluxes", override)
        state, _ = T._noahmp_step()
        column = A.tiedtke_adapter(state, 18., grid=T.smoke._land_grid(), stepcu=1)
        physical_qfx = np.asarray(state.qfx)
        qfx_operand = np.asarray(column.rainc_acc) - np.asarray(state.rainc_acc)
        assert np.isfinite(qfx_operand).all() and np.isfinite(physical_qfx).all()
        results[str(limited)] = dict(physical_qfx=physical_qfx.tolist(),
                                     physical_hfx=np.asarray(state.hfx).tolist(),
                                     tiedtke_qfx_operand=qfx_operand.tolist())
q0 = np.asarray(results["False"]["tiedtke_qfx_operand"])
q1 = np.asarray(results["True"]["tiedtke_qfx_operand"])
assert np.array_equal(results["False"]["physical_qfx"], results["True"]["physical_qfx"])
assert np.array_equal(results["False"]["physical_hfx"], results["True"]["physical_hfx"])
results["no_change_outside_mynn_gate"] = bool(np.array_equal(q0, q1))
results["max_abs_qfx_operand_change"] = float(np.max(np.abs(q1 - q0)))
results["limited_over_control"] = (q1 / q0).tolist()
results["scope"] = "CPU source/caller witness, sentinel cumulus; not trajectory/scientific fidelity"
out.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({key: results[key] for key in ("no_change_outside_mynn_gate", "max_abs_qfx_operand_change", "limited_over_control")}))
