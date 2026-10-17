from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ocean_mixed_layer import OceanCarry, initialize, step

FIXTURES = Path(__file__).resolve().parents[1]/'proofs/s2_oml/fixtures'


def evolve(x, count, compiled=False):
    carry = OceanCarry(*(jnp.asarray(v, jnp.float32) for v in x[:7]))
    kw = dict(zip(('hfx','lh','gsw','glw','u','v','ust','f','emiss','dt','gamma','relaxation_time'),map(float,x[7:19]),strict=True))
    fn = (jax.jit(lambda c:step(c,**kw)) if compiled else lambda c:step(c,**kw))
    for _ in range(count):
        carry = fn(carry)
    return np.asarray(carry)


def test_pristine_real_columns():
    data = np.load(FIXTURES/'oml.npz')
    worst = 0
    for x, n, expected in zip(data['inputs'],data['steps'],data['expected'],strict=True):
        actual = evolve(x,int(n))
        assert np.isfinite(actual).all()
        worst = max(worst,float(np.max(np.abs(actual-expected))))
        np.testing.assert_array_equal(actual,expected)
    assert json.loads((FIXTURES/'manifest.json').read_text())['sensitivity_byte_equal']


def test_compiled_pristine_columns():
    data = np.load(FIXTURES/'oml.npz')
    for x,n,expected,fma,input_bound in zip(data['inputs'],data['steps'],data['expected'],data['fma'],data['input_ulp_bound'],strict=True):
        actual = evolve(x,int(n),True)
        assert np.isfinite(actual).all()
        # Pristine-only +/- one-ULP operand ensemble plus pristine FMA spread.
        bound = np.abs(fma-expected)+input_bound
        assert np.all(np.abs(actual-expected)<=bound), (actual,expected,bound)


@pytest.mark.parametrize('hml0',[50.,0.,-1.])
def test_init_and_land_preservation(hml0):
    tsk = jnp.asarray([300,299],jnp.float32)
    carry = initialize(tsk,hml0,h0ml=jnp.asarray([30,20]),huml=jnp.asarray([3,4]),
                       hvml=jnp.asarray([5,6]),tmoml=jnp.asarray([200.5,290]))
    np.testing.assert_array_equal(carry.tml,tsk)
    np.testing.assert_array_equal(carry.hml,[50,50] if hml0>0 else [30,20])
    np.testing.assert_array_equal(carry.huml,[3,4] if hml0<0 else [0,0])
    np.testing.assert_array_equal(carry.tmoml,[300,290] if hml0<0 else [295,294])
    new = step(carry,hfx=250,lh=700,gsw=800,glw=380,u=65,v=-20,ust=2.3,f=5e-5,emiss=.98,dt=18,water=[True,False])
    for a,b in zip(new,carry,strict=True):
        assert a.dtype == jnp.float32
        assert a[1] == b[1]
    assert new.huml[0] != carry.huml[0]


def test_initialization_requires_input_anchors():
    with pytest.raises(ValueError,match='H0ML'):
        initialize([300],0)
    with pytest.raises(ValueError,match='HUML'):
        initialize([300],-1,h0ml=[50])


def test_pristine_initialization_all_depth_modes():
    for case in json.loads((FIXTURES/'init.json').read_text())['cases']:
        out=initialize(case['tsk'],case['oml_hml0'],**{k:case[k] for k in ('h0ml','huml','hvml','tmoml')})
        np.testing.assert_array_equal(np.asarray(out).T,np.asarray(case['expected'],np.float32))
