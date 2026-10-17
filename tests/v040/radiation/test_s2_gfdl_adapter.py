"""ETARA ordering against retained native whole-driver results, no re-solve."""
from pathlib import Path
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import ra_gfdleta_adapter as port

FIXTURE=Path(__file__).resolve().parents[2]/'fixtures/s2_gfdl_etara_v1.npz'


@pytest.mark.parametrize('compiled',[False,True])
def test_native_etara_lw_then_sw(compiled):
    receipt=json.loads(FIXTURE.with_suffix('.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==receipt['fixture_sha256']
    with np.load(FIXTURE,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    def pair(exner,sw,lw):
        zero=jnp.zeros_like(exner)
        total,held_lw,held_sw=port.store_tendencies(exner,sw,lw,do_lw=True,do_sw=False,
            held_sw=zero,held_lw=zero,total=zero)
        # Second call must use the saved rates even when its unused new inputs
        # are NaN. The native driver skips RADTN on this call entirely.
        return port.store_tendencies(exner,jnp.full_like(sw,jnp.nan),jnp.full_like(lw,jnp.nan),
            do_lw=True,do_sw=True,held_sw=held_sw,held_lw=held_lw,total=total)
    run=jax.jit(pair) if compiled else pair
    actual=run(*map(jnp.asarray,(data['exner'],data['sw'],data['lw'])))
    for value,key in zip(actual,('total','held_lw','held_sw')):
        value=np.asarray(value);ref=data[key]
        assert value.dtype==ref.dtype==np.float32
        assert np.isfinite(value).all()
        assert value.tobytes()==ref.tobytes(),key
    # Genuine deletion mutant: dropping the first LW contribution fails the
    # same native total comparison on nonzero native LW profiles.
    assert data['held_sw'].tobytes()!=data['total'].tobytes()


def test_convective_interval_reset():
    top=jnp.asarray([5.,7.],jnp.float32);bottom=jnp.asarray([2.,4.],jnp.float32)
    precip=jnp.asarray([.2,.7],jnp.float32)
    new_top,new_bottom,new_precip,saved_top,saved_bottom=port.reset_convective_interval(
        top,bottom,precip,levels=44)
    np.testing.assert_array_equal(new_top,[0.,0.]);np.testing.assert_array_equal(new_bottom,[45.,45.])
    np.testing.assert_array_equal(new_precip,[0.,0.]);np.testing.assert_array_equal(saved_top,top)
    np.testing.assert_array_equal(saved_bottom,bottom)
