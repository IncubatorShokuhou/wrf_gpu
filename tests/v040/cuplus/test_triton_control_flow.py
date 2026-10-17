"""The native literal-return enabler preserves bytes and qualification tools."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'proofs/v040/cuplus'))


def branches(xp):
    def yes(_):
        return {'scalar':xp.float64(.8),'integer':xp.int32(1),
                'profile':xp.asarray([-0.,1.,-.25],xp.float32),'flag':xp.asarray(True)}
    def no(_):
        return {'scalar':xp.float64(0.),'integer':xp.int32(-1),
                'profile':xp.asarray([0.,-1.,.25],xp.float32),'flag':xp.asarray(False)}
    return yes,no


def same(actual,expected):
    assert actual.keys()==expected.keys()
    for key in expected:
        a,b=np.asarray(actual[key]),np.asarray(expected[key])
        assert (a.shape,a.dtype,a.tobytes())==(b.shape,b.dtype,b.tobytes()),key


@pytest.mark.parametrize('module',['phys_camzm_column','phys_kfcup_column','phys_mskf_column'])
@pytest.mark.parametrize('predicate',[True,False])
def test_materialized_branch_literals_keep_bytes(module,predicate):
    import importlib
    cond=importlib.import_module('gpuwrf.kernels.'+module)._cond
    yes,no=branches(jnp)
    result=jax.jit(lambda flag:cond(flag,yes,no,None))(jnp.asarray(predicate))
    same(result,(yes if predicate else no)(None))


@pytest.mark.parametrize('scheme',['camzm','kfcup','mskf'])
@pytest.mark.parametrize('predicate',[True,False])
def test_original_literal_executor_supports_materialization(scheme,predicate):
    if scheme=='camzm':
        from literal_camzm import executor
        fn=executor(ROOT)
    else:
        from literal_kfcup import executor
        fn,_=executor(ROOT,scheme=scheme)
    yes,no=branches(np)
    same(fn.__globals__['_cond'](np.asarray(predicate),yes,no,None),
         (yes if predicate else no)(None))
