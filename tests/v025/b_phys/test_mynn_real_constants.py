"""BD94/BP104: original MYNN REAL PARAMETER bits and live consumers."""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import mynn_constants as MC, mynn_pbl as P
from gpuwrf.kernels import phys_mynn_columns as N
from test_mynn_qni import DATA, state_from_row as _state_from_row

WRF=json.loads((Path(__file__).parent/'fixtures/mynn_real_constants_wrf.json').read_text())


def finite_reference(reference):
    for name,entry in reference.items():
        assert np.isfinite(entry['value']),f'reference {name} value nonfinite'
        assert np.isfinite(np.asarray(entry['bits'],np.uint32).view(np.float32)),f'reference {name} bits nonfinite'


def finite_state(state):
    for name in state.__slots__:
        value=getattr(state,name)
        if value is not None:assert np.isfinite(np.asarray(value)).all(),f'input {name} nonfinite'


def state_from_row(row,dtype=jnp.float32):
    state=_state_from_row(row,dtype)
    finite_state(state)
    return state


@pytest.fixture(autouse=True)
def clear():
    finite_reference(WRF)
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.mark.parametrize('name',WRF)
def test_original_real_parameter_bits_and_legacy(monkeypatch,name):
    old=getattr(MC,name);monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    real=MC.mynn_constant(name,jnp.float32)
    assert np.isfinite(real)
    assert np.float32(real).view(np.uint32)==WRF[name]['bits']
    assert MC.mynn_constant(name,jnp.float64)==old
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',False)
    assert MC.mynn_constant(name,jnp.float32)==old


def test_live_virtual_temperature_uses_original_p608(monkeypatch):
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    s=state_from_row(DATA['rows'][1]);sqv=s.qv/(jnp.float32(1)+s.qv)
    expected=s.theta*(jnp.float32(1)+np.float32(WRF['P608']['value'])*sqv)
    with jax.enable_x64(False):actual=P._virtual_potential(s.theta,s.qv)
    assert np.isfinite(np.asarray(actual)).all() and np.isfinite(np.asarray(expected)).all()
    np.testing.assert_array_equal(actual,expected)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',False)
    assert np.any(np.asarray(P._virtual_potential(s.theta,s.qv))!=np.asarray(expected))


def test_both_dmp_suppliers_use_real_virtual_term(monkeypatch):
    from gpuwrf.kernels import phys_mynn_plume as L
    from gpuwrf.physics import mynn_edmf as E
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    s=state_from_row(DATA['rows'][1]);B=s.u.shape[0];z=jnp.zeros(B,jnp.float32)
    f=SurfaceFluxes(ustar=z+.1,theta_flux=z+.01,qv_flux=z+1e-5,tau_u=z,tau_v=z,rhosfc=z+1,fltv=z+.03,xland=z+2,t_skin=z+290)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    expected=s.theta*(jnp.float32(1)+np.float32(WRF['P608']['value'])*(s.qv/(1+s.qv)))
    class Captured(Exception):pass
    def capture(*args,**kw):
        got=np.asarray(args[8]);assert np.isfinite(got).all()
        np.testing.assert_array_equal(got,np.asarray(expected))
        raise Captured
    monkeypatch.setattr(L,'dmp_mf_columns_native',capture);monkeypatch.setattr(E,'dmp_mf_columns',capture)
    for fn in [N._native_edmf_arrays,P._edmf_arrays_from_state]:
        with jax.enable_x64(False),pytest.raises(Captured):fn(s,f,f.fltv,z+500,18.,3000.)


def test_double_level2_and_turbulence_are_unchanged(monkeypatch):
    s=state_from_row(DATA['rows'][1],jnp.float64)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',False)
    with jax.enable_x64(True):off=P._mym_level2(s)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    with jax.enable_x64(True):on=P._mym_level2(s)
    for x,y in zip(jax.tree.leaves(off),jax.tree.leaves(on)):
        assert np.isfinite(np.asarray(x)).all() and np.isfinite(np.asarray(y)).all()
        np.testing.assert_array_equal(x,y)


def test_resolved_real_constants_key(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import module_const_env_hash
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',False);a=module_const_env_hash()
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    assert module_const_env_hash()!=a


def test_import_time_registry_has_no_unresolved_entries():
    import importlib
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    for module,attr in IMPORT_TIME_ENV_CONSTANTS:assert hasattr(importlib.import_module(module),attr),(module,attr)


@pytest.mark.parametrize('name,entry',[
    ('A1','level2'),('A2','level2'),('C1','level2'),('G2','level2'),('GTR','level2'),
    ('A1','turbulence'),('A2','turbulence'),('C1','turbulence'),
    ('E1C','turbulence'),('E2C','turbulence'),('E3C','turbulence'),
    ('E4C','turbulence'),('E5C','turbulence'),
])
def test_actual_closure_reads_original_real_constant(monkeypatch,name,entry):
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    s=state_from_row(DATA['rows'][1]);observed=[];original=P._typed_mynn_constant
    if entry=='turbulence':
        with jax.enable_x64(False):level2=P._mym_level2(s)
        for x in jax.tree.leaves(level2):assert np.isfinite(np.asarray(x)).all()
        monkeypatch.setattr(P,'_mym_level2',lambda state:level2)
    def capture(n,dtype,legacy=None):
        value=original(n,dtype,legacy)
        if n==name:
            assert np.float32(value).view(np.uint32)==WRF[n]['bits']
            observed.append(n)
        return value
    monkeypatch.setattr(P,'_typed_mynn_constant',capture)
    with jax.enable_x64(False):
        if entry=='level2':out=P._mym_level2(s)
        else:out=P._mym_turbulence(s,2*s.tke,jnp.ones(s.tke.shape[:-1],jnp.float32)*.03,jnp.ones(s.tke.shape[:-1],jnp.float32)*.1,3000.,2.)
    for x in jax.tree.leaves(out):assert np.isfinite(np.asarray(x)).all()
    assert observed,f'actual {entry} does not read {name}'


def test_noah_density_consumes_real_virtual_constant(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.physics import noahmp_coupler as NO
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS','1')
    original=NO._mynn_constant;seen=[]
    def capture(name,dtype,legacy=None):
        value=original(name,dtype,legacy)
        assert name=='P608'
        assert np.float32(value).view(np.uint32)==WRF['P608']['bits']
        seen.append(name);return value
    monkeypatch.setattr(NO,'_mynn_constant',capture)
    out=NO._mynn_pbl_surface_density(SimpleNamespace(psfc=jnp.array([1e5],jnp.float32),sfctmp=jnp.array([290.],jnp.float32)),jnp.array([.01],jnp.float32))
    assert np.isfinite(np.asarray(out)).all() and seen


def test_surface_body_consumes_original_ep1(monkeypatch):
    from gpuwrf.physics import surface_layer as SL
    from gpuwrf.physics.fp32 import surface_layer_real as REAL
    from test_sfclay_snow_andreas import view
    surface=json.loads((Path(__file__).parent/'fixtures/sfclay_snow_canary.json').read_text())
    rows=next(iter(surface['arms'].values()))['columns']
    for row in rows:
        for name,value in row.items():
            if isinstance(value,(int,float,list)):assert np.isfinite(np.asarray(value)).all(),f'input surface {name} nonfinite'
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True);monkeypatch.setattr(REAL,'_NATIVE_REAL',True)
    original=SL._mynn_constant;seen=[]
    def capture(name,dtype,legacy=None):
        value=original(name,dtype,legacy)
        assert name=='P608'
        assert np.float32(value).view(np.uint32)==WRF['P608']['bits']
        seen.append(name);return value
    monkeypatch.setattr(SL,'_mynn_constant',capture)
    out=SL.surface_layer_with_diagnostics(view(rows)._replace(dx_m=9000.))
    for x in jax.tree.leaves(out):assert np.isfinite(np.asarray(x)).all()
    assert seen


@pytest.mark.parametrize('name',WRF)
@pytest.mark.parametrize('field',['value','bits'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_reference_nonfinite_rejected(name,field,bad):
    import copy
    reference=copy.deepcopy(WRF)
    reference[name][field]=float(bad) if field=='value' else int(np.float32(bad).view(np.uint32))
    with pytest.raises(AssertionError,match='reference .* nonfinite'):finite_reference(reference)


@pytest.mark.parametrize('name',['u','v','w','theta','qv','tke','p','rho','dz','qc','qi','qs','qsq','ni','exner','km','kh','el'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_input_nonfinite_rejected(name,bad):
    state=_state_from_row(DATA['rows'][1])
    state=state.replace(**{name:jnp.full_like(getattr(state,name),bad)})
    with pytest.raises(AssertionError,match='input '+name+' nonfinite'):finite_state(state)
