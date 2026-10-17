"""Localize the rel034 NaN receipt; no change to the measured model source."""
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
spec = importlib.util.spec_from_file_location('fixture_builder', root / 'tests/dynamics/test_diffopt2_tke_smag3d.py')
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform == 'cpu'
assert Path(om.__file__).is_relative_to(root)
grid = test._build_grid(ny=6, nx=8, nz=4, dx=1000.)
state = test._build_state(grid)
report = {}
def summary(x):
    a = np.asarray(x)
    return {'nonfinite': int((~np.isfinite(a)).sum()), 'size': a.size,
            'min': float(a.min()) if np.isfinite(a).all() else None,
            'max': float(a.max()) if np.isfinite(a).all() else None}
def run(label, st):
    h = test.apply_halo(st, test.halo_spec(grid))
    nl0 = test._namelist(grid, diff_opt=0, km_opt=0)
    nl5 = dataclasses.replace(nl0, diff_opt=2, km_opt=5)
    turb = om._diffopt2_turbulence_fields(h, nl5, dz=jnp.asarray(1500., st.theta.dtype))
    report[label] = {'turbulence': dict(zip(turb._fields, map(summary, turb)))}
    for tag, nl in [('off', nl0), ('sms', nl5)]:
        out = om._augment_large_step_tendencies(h, nl.tendencies, nl, rk_step=3)
        report[label][tag] = {n: summary(getattr(out, n)) for n in ('u', 'v', 'w', 'theta')}
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2))

run('original_zero_pressure_fixture', state)
# Independent physical operand repair: WRF never has p_phy=0/rho=0. No model guard or clamp.
run('positive_pressure_fixture', state.replace(p=jnp.full_like(state.p, 90000.),
    p_total=jnp.full_like(state.p_total, 90000.)))
print(json.dumps(report, indent=2))
