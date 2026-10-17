"""REAL4 Goddard wrapper parity; never a stand-in for LW/SW solver qualification."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ra_goddard_common import opt_cloud, ozone_interp, sounding_interp

PROOFS = Path(__file__).resolve().parents[1]/'proofs/s2_goddardrad/savepoints'


@pytest.fixture(scope='module')
def oracle():
    path = PROOFS/'helpers_real4.npz'
    manifest = json.loads((PROOFS/'helpers_real4.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['savepoint_sha256']
    with np.load(path) as f:
        data = dict(f)
    assert all(np.isfinite(v).all() for v in data.values())
    assert all(n>0 for n in manifest['active_species_counts'])
    return data,manifest['tolerances']


@pytest.mark.parametrize('shortwave',[True,False])
def test_mie_optics_pristine_real4(oracle,shortwave):
    data,tol = oracle
    result = jax.jit(lambda dz,q,re: opt_cloud(dz,q,re,shortwave=shortwave))(data['dz'],data['q'],data['re'])
    actual = np.stack(result,axis=-1)
    expected = data['sw' if shortwave else 'lw']
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,expected,rtol=tol['optics_rtol'],atol=tol['optics_atol'])


def test_climatology_and_ozone_pristine_real4(oracle):
    data,tol = oracle
    fn = jax.jit(jax.vmap(jax.vmap(sounding_interp,in_axes=(None,0)),in_axes=(0,None)))
    actual = np.asarray(fn(data['latitude'],data['day']))
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,data['climate'],rtol=tol['climate_rtol'],atol=tol['climate_atol'])
    ozfn = jax.jit(jax.vmap(jax.vmap(ozone_interp,in_axes=(0,None)),in_axes=(0,None)))
    ozone = np.asarray(ozfn(actual,data['pressure']))
    assert ozone.dtype == np.float32 and np.isfinite(ozone).all()
    np.testing.assert_allclose(ozone,data['ozone'],rtol=tol['ozone_rtol'],atol=tol['ozone_atol'])


def test_host_table_cache_does_not_capture_tracers(oracle):
    from gpuwrf.physics.ra_goddard_common import _host_tables
    _host_tables.cache_clear()
    jax.jit(sounding_interp)(jnp.float32(46),jnp.int32(100)).block_until_ready()
    jax.jit(lambda x:sounding_interp(x,jnp.int32(212)))(jnp.float32(28)).block_until_ready()
    assert all(isinstance(x,np.ndarray) for x in _host_tables().values())


def test_runtime_asset_cannot_silently_change_under_a_cached_executable(tmp_path,monkeypatch):
    from gpuwrf.physics import ra_goddard_common as common
    (tmp_path/'goddard_tables.npz').write_bytes(b'changed asset')
    with monkeypatch.context() as m:
        m.setattr(common,'__file__',str(tmp_path/'ra_goddard_common.py'))
        common._host_tables.cache_clear()
        with pytest.raises(RuntimeError,match='SHA256'):common._host_tables()
    common._host_tables.cache_clear()
