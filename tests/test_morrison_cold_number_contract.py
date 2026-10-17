"""Real c08 failure: WRF writes corrected numbers before snow aggregation."""
from pathlib import Path
import ast
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['fp64', 'fp32'])
def test_actual_c08_cold_numbers_match_pristine_morrison(mode):
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.microphysics_morrison import morrison_run
    data = json.loads((ROOT/'proofs/t_c02nan/savepoints/morrison_c08_step36.json').read_text())
    source = ast.parse((ROOT/'proofs/v060/run_morrison_parity.py').read_text())
    tol = ast.literal_eval(next(n.value for n in source.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='PREDECLARED_TOL' for t in n.targets)))
    names = ('th','qv','qc','qr','qi','qs','qg','ni','ns','nr','ng','pii','p','dz','w')
    # Retain the failing operator's measured REAL8 inputs. REAL4 reference
    # inputs are their recorded WRF-kind cast, covered by the physical band.
    with jax.enable_x64():
        args = [jnp.asarray(data['inputs'][name], jnp.float64)[None,:] for name in names]
        out = jax.device_get(morrison_run(*args, jnp.asarray(data['inputs']['dt'], jnp.float64)))
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    ref = data['references'][mode]
    for name in names[:11]:
        expected = np.asarray(ref['columns'][name.upper()+'_OUT'])
        error = np.max(np.abs(out[name][0]-expected))
        if name=='th':
            bound = tol[mode+'_t_abs']
        else:
            kind = 'n' if name in ('ni','ns','nr','ng') else 'q'
            bound = max(tol[mode+'_'+kind+'_rel']*np.max(np.abs(expected)), tol[mode+'_'+kind+'_abs_floor'])
        assert error <= bound, (mode, name, error, bound)
    for name,key in [('rainncv','RAINNCV'), ('snowncv','SNOWNCV'), ('graupelncv','GRAUPELNCV'), ('sr','SR')]:
        expected = ref['scalars'][key]
        bound = tol[mode+'_sr_abs'] if name=='sr' else max(tol[mode+'_precip_rel']*abs(expected), tol[mode+'_precip_abs'])
        assert abs(float(out[name][0])-expected) <= bound, (mode, name)
