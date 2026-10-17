"""Full WRF lake trajectory oracle at the REAL32 saved-state boundary."""
import json
from pathlib import Path
import jax
from jax import numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.lake_initialization import LakeColumnState
from gpuwrf.physics.lake_step import LakeAtmosphere,lake_step

FIXTURE = Path(__file__).resolve().parents[1]/"proofs/s1_lake/fixtures"
CASES = {c['name']:c for c in json.loads((FIXTURE/'manifest.json').read_text())['cases'] if c['nsteps'] > 0}
STEP = jax.jit(lake_step)


@pytest.mark.parametrize('name',CASES)
def test_full_lake_pristine_trajectory(name):
    case = CASES[name]
    with np.load(FIXTURE/'columns.npz',allow_pickle=False) as oracle:
        s = LakeColumnState(*(jnp.asarray(oracle[f'{name}/step0/{field}'][0] if len(oracle[f'{name}/step0/{field}']) == 1 else oracle[f'{name}/step0/{field}'],jnp.float32) for field in LakeColumnState._fields))
        alb = np.float32(case['alb'])
        for i in range(1,case['nsteps']+1):
            a = LakeAtmosphere(*(jnp.asarray(x,jnp.float32) for x in (
                case['ta'],case['pbot'],case['psfc'],case['dzair'],case['qv'],case['u'],case['v'],case['lw'],case['emiss'],case['rain'],case['dt'],case['sw'],alb,case['lat'])))
            result = STEP(s,a)
            s = result.columns
            alb = result.surface.albedo
            if i in (1,case['nsteps']):
                for field,actual in s._asdict().items():
                    ref = oracle[f'{name}/step{i}/{field}']
                    actual = np.asarray(actual).reshape(-1)
                    assert np.isfinite(actual).all() and np.isfinite(ref).all()
                    np.testing.assert_allclose(actual,ref,rtol=2e-6,atol=1e-10,err_msg=f'{name} step{i} {field}')
                values = np.array(result.surface,dtype=np.float32)
                ref = oracle[f'{name}/step{i}/flux']
                assert np.isfinite(values).all() and np.isfinite(ref).all()
                np.testing.assert_allclose(values,ref,rtol=2e-6,atol=1e-6,err_msg=f'{name} step{i} surface')
