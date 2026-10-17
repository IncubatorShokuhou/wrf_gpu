"""Full public WRF driver, including LUT/cloud/ozone, units, flips and night skip."""
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
import pytest
from gpuwrf.coupling.goddard_radiation import goddard_column

SAVE=Path(__file__).resolve().parents[1]/'proofs/s2_goddardrad/savepoints'


@pytest.mark.parametrize('case',range(1,5))
def test_public_wrapper_pristine(case):
    manifest=json.loads((SAVE/'wrapper_real4.json').read_text());path=SAVE/'wrapper_real4.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['savepoint_sha256']
    with np.load(path) as f:data={k.split('_',1)[1]:f[k] for k in f.files if k.startswith(f'{case}_')}
    col=data['input'];inter=data['interface'];s=data['scalar']
    radii=np.stack((col[:,12],col[:,13],col[:,14],col[:,15],col[:,16],np.zeros(12,np.float32)),axis=-1)
    cond=np.concatenate((col[:,3:8],np.zeros((12,1),np.float32)),axis=-1)
    args=(col[:,0],col[:,1],inter[:,0],inter[:,1],col[:,2],cond,radii,col[:,8],col[:,9],col[:,10],col[:,11],s[0],np.int32(100),*s[1:])
    out=jax.jit(goddard_column)(*args)
    tol=manifest['tolerance']
    for a,e in zip(out[:2],(data['lw'],data['sw'])):
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a,e,rtol=tol['rate_rtol'],atol=tol['rate_atol'])
    # Surface down/up, then TOA down/up. Original upward ERBE uses a minus sign.
    expected=(data['swflux'][2],-data['swflux'][3],data['lwflux'][2],-data['lwflux'][3],
              data['swflux'][0],-data['swflux'][1],data['lwflux'][0],-data['lwflux'][1])
    actual=np.asarray(out[2:10])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual,expected,rtol=tol['flux_rtol'],atol=tol['flux_atol'])
    assert int(out[10])==0
    np.testing.assert_allclose(np.asarray(out[11:]),data['swextra'],rtol=tol['flux_rtol'],atol=tol['flux_atol'])
