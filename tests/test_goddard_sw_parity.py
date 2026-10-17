"""Option5 SW vs pristine WRF with documented zero-init reference policy."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.physics.ra_goddard_sw import solve_sw_column

SAVE=Path(__file__).resolve().parents[1]/'proofs/s2_goddardrad/savepoints'


@pytest.mark.parametrize('case',range(1,9))
def test_native_swrad_pristine(case):
    manifest=json.loads((SAVE/'sw_native.json').read_text());path=SAVE/'sw_native.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['savepoint_sha256']
    with np.load(path) as f:data={k.split('_',1)[1]:f[k] for k in f.files if k.startswith(f'{case}_')}
    layer,interface,optics=data['layer'],data['interface'],data['optics']
    icb,cosz,albedo=data['scalar']
    args=[interface[:,0],*layer.T,cosz,*[optics[:,:,k] for k in range(6)],albedo,np.int32(icb),np.zeros(40)]
    result=jax.jit(solve_sw_column)(*args)
    expected=(*interface[:,1:].T,data['surface'])
    for a,e in zip(result,expected):
        assert a.dtype==np.float32
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a,e,**manifest['tolerance'])
