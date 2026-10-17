"""CPU-only admission, E78 deletion guards and native legacy history semantics."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.io.flg_gfdl_config import selection_errors
from gpuwrf.io.flg_gfdl_schema import marker,validate
from gpuwrf.coupling.flg_gfdl_radiation import initial_diagnostics,initial_gfdl,history_fields,previous_columns


def options(code):
    return dict(ra_lw_physics=code,ra_sw_physics=code,mp_physics=8,cu_physics=0,sf_surface_physics=4)


@pytest.mark.parametrize('code',[7,99])
def test_gpu_refusal_before_jax_import(code,tmp_path):
    root=Path(__file__).resolve().parents[3]
    case=tmp_path/'case';case.mkdir()
    (case/'namelist.input').write_text(f'&physics\n ra_lw_physics=2*{code},\n ra_sw_physics=2*{code},\n/\n')
    command=f'''import sys,os
os.environ['JAX_PLATFORMS']='cuda'
sys.argv=['gpuwrf','run','--input-dir',{str(case)!r},'--output-dir',{str(tmp_path/'out')!r}]
try:import gpuwrf
except SystemExit as error:assert error.code==2
else:raise AssertionError('GPU accepted CPU-only physics at bootstrap')
assert 'jax' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',command],cwd=root,
        env=dict(os.environ,PYTHONPATH=str(root/'src')),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'native GPU column kernel pending' in result.stderr
    assert selection_errors(options(code),backend='cpu')==[]
    assert selection_errors(dict(options(code),cu_physics=1),backend='cpu')


@pytest.fixture
def state():
    return SimpleNamespace(theta=jnp.zeros((5,2,3),jnp.float32),qv=jnp.zeros((5,2,3),jnp.float32),
                           t_skin=jnp.zeros((2,3),jnp.float32))


@pytest.mark.parametrize('code',[7,99])
def test_restart_source_schema_deletions(state,code):
    diag=initial_diagnostics(state);gfdl=initial_gfdl(state) if code==99 else None
    schema=marker(diag,gfdl);validate(diag,gfdl,options(code),schema,state_shape=state.theta.shape)
    with pytest.raises(ValueError,match='E78'):validate(diag,gfdl,options(code),None)
    with pytest.raises(ValueError,match='E78'):validate(diag,gfdl,options(99 if code==7 else 7),schema)
    with pytest.raises(ValueError,match='E78'):validate(diag._replace(glw=None),gfdl,options(code),schema)
    with pytest.raises(ValueError,match='E78'):validate(diag._replace(rswtoa=jnp.full((2,3),jnp.nan)),gfdl,options(code),schema)
    if code==99:
        with pytest.raises(ValueError,match='E78'):validate(diag,None,options(code),schema)
        with pytest.raises(ValueError,match='E78'):validate(diag,gfdl._replace(ncfrcv=gfdl.ncfrcv.astype(jnp.float32)),options(code),schema)
    # Actual pickle-restart payload validator must reject the deleted group.
    from gpuwrf.io.restart import _validate_radiation_diagnostics_schema
    payload=dict(extra_fields=dict(radiation_diagnostics=diag,gfdl_radiation=gfdl),
        state_fields=dict(qv=np.asarray(state.qv)),flg_gfdl_schema=schema,
        radiation_diagnostics_schema_version=2)
    if code==99:
        del payload['extra_fields']['gfdl_radiation']
        with pytest.raises(ValueError,match='E78'):_validate_radiation_diagnostics_schema(payload,namelist=options(code))


def test_internal_toa_is_not_history_olr(state):
    diag=initial_diagnostics(state)
    f=lambda value:jnp.full((2,3),value,jnp.float32)
    diag=diag._replace(rlwtoa=f(270),rswtoa=f(120),lw_toa_up=f(270),sw_toa_up=f(120),
                      glw=f(300),swdown=f(100),swup=f(20),swddir=f(60),gsw=f(80))
    fields=history_fields(diag)
    np.testing.assert_array_equal(fields['OLR'],0.)
    np.testing.assert_array_equal(fields['LWUPT'],0.)
    np.testing.assert_array_equal(fields['SWUPT'],0.)
    np.testing.assert_array_equal(fields['GLW'],300.)
    np.testing.assert_array_equal(fields['SWDOWN'],100.)
    np.testing.assert_array_equal(fields['GSW'],80.)
    np.testing.assert_array_equal(fields['SWDDIR'],60.)


def test_initial_convective_memory_is_native_allocation_zero(state):
    carry=initial_gfdl(state)
    for name in ('top','bottom','cuppt','htopr','hbotr'):
        np.testing.assert_array_equal(getattr(carry,name),0.)


def test_column_flattening_preserves_native_integer_counters(state):
    carry=initial_gfdl(state)
    columns=previous_columns(carry,6)
    for name in ('ncfrcv','ncfrst'):
        assert columns[name].dtype==jnp.int32
        assert columns[name].shape==(6,)
    for name in ('top','bottom','cuppt','acfrcv','acfrst'):
        assert columns[name].dtype==jnp.float32
