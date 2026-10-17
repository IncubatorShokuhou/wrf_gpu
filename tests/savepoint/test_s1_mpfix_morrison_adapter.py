"""Real WRF wrapper boundary: microphysics receives dry theta, not theta_m."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.coupling.scan_adapters import morrison_aero_adapter
from gpuwrf.coupling.physics_couplers import GRAVITY_M_S2, R_D_OVER_CP

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('case',(9,12))
def test_theta_m_adapter_matches_pristine_nc(case):
    jax.config.update('jax_enable_x64',True)
    data=json.loads((ROOT/f'proofs/s1_mpfix/fixtures/dry_fp32_{case}.json').read_text())
    columns=data['columns']; grid=SimpleNamespace(nz=40,ny=1,nx=1)
    state=State(**{n:jnp.zeros(shape,DEFAULT_DTYPES.dtype_for(n))
                   for n,shape in _state_field_shapes(grid,mp_physics=40).items()})
    a=lambda n:jnp.asarray(columns[n],jnp.float32)[:,None,None]
    p=a('P'); qv=a('QV_IN')
    pii=(jnp.maximum(p,1.)/1e5)**R_D_OVER_CP
    # The fixture wrapper has arbitrary PII; retain its physical T when adapting
    # it to the caller's independently diagnosed Exner function.
    th_dry=a('TH_IN')*a('PII')/pii
    rvrd=jnp.float32(461.6)/jnp.float32(287.)
    z=jnp.concatenate((jnp.zeros((1,1,1),jnp.float32),
                       jnp.cumsum(a('DZ')*jnp.float32(GRAVITY_M_S2),axis=0)),axis=0)
    updates=dict(theta=th_dry*(1.+rvrd*qv),p_total=p,ph_total=z,qv=qv)
    updates.update({n:a(key+'_IN') for n,key in
        [('qc','QC'),('qr','QR'),('qi','QI'),('qs','QS'),('qg','QG'),
         ('Ni','NI'),('Ns','NS'),('Nr','NR'),('Ng','NG'),('Nc','NC')]})
    state=state.replace(**updates)
    result=jax.jit(lambda s:morrison_aero_adapter(s,float(data['scalars']['DT']),grid))(state)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    np.testing.assert_allclose(np.asarray(result.Nc)[:,0,0],columns['NC_OUT'],rtol=2e-6,atol=0.)
