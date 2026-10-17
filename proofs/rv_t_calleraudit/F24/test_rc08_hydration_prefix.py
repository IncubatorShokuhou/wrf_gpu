"""Exercise the exact changed hydration block without absent external savepoints."""
import ast
import inspect
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
from gpuwrf.validation import wrf_restart_hydration as hydration

@pytest.mark.parametrize('with_sr',[False,True])
def test_precipitation_projection_block_preserves_five_slots_and_optional_sr(with_sr):
    tree=ast.parse(inspect.getsource(hydration.hydrate_wrfrst))
    blocks=[n for n in tree.body[0].body if isinstance(n,ast.If)
            and ast.unparse(n.test)=='carry.noahmp_precipitation is not None']
    assert len(blocks)==1
    wrapper=ast.parse('def project(carry,restart,source,put):\n extra={}\n return extra\n')
    wrapper.body[0].body.insert(1,blocks[0])
    ns={}
    exec(compile(ast.fix_missing_locations(wrapper),hydration.__file__,'exec'),ns)
    shape=(2,3)
    fields={name:np.full(shape,value,np.float32) for name,value in
        dict(RAINCV=.36,RAINSHV=.18,RAINNCV=.72,SNOWNCV=.09,GRAUPELNCV=.036,HAILNCV=.018,SR=.4375).items()}
    held=seed_precipitation(SimpleNamespace(t_skin=jnp.zeros(shape)),with_sr=with_sr)
    observed=[]
    def put(old,raw,label,origin):
        observed.append(label)
        assert raw.shape==old.shape
        return jnp.asarray(raw,old.dtype)
    result=ns['project'](SimpleNamespace(noahmp_precipitation=held),SimpleNamespace(dt_s=18.),fields.__getitem__,put)['noahmp_precipitation']
    expected=[(fields['RAINCV']+fields['RAINSHV'])/18.,fields['RAINNCV']/18.,fields['SNOWNCV']/18.,
              fields['GRAUPELNCV']/18.,fields['HAILNCV']/18.]
    for actual,want in zip(result[:5],expected,strict=True):np.testing.assert_array_equal(actual,want)
    if with_sr:np.testing.assert_array_equal(result.sr,fields['SR'])
    else:assert result.sr is None
    assert len(observed)==5+with_sr
