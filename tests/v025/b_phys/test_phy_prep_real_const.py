"""BP103: actual REAL PHY_PREP demixing uses original WRF constant order."""
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as C, noahmp_surface_hook as H

DATA=json.loads((Path(__file__).parent/'fixtures/phy_prep_real_const_wrf.json').read_text())
WRF_RATIO=np.float32(np.float32(461.6)/np.float32(287.0))


@pytest.fixture(autouse=True)
def clear_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def columns(monkeypatch,dtype=jnp.float32):
    rows=DATA['rows'];B=len(rows);shape=(1,B,1)
    f=lambda n:jnp.asarray([r[n] for r in rows],dtype).reshape(shape)
    z=jnp.zeros(shape,dtype);s=jnp.zeros((B,1),dtype)
    state=SimpleNamespace(theta=f('theta_m'),qv=f('qv'),p=jnp.full(shape,90000.,dtype),
        qc=z,qi=z,qs=z,qsq=z,qke=z,u=z,v=z,w=z,Ni=z,
        el_pbl=z,maxmf=s,maxwidth=s,ztop_plume=s,cam_kvh3d=None,t_skin=s+290,soil_moisture=s+1,
        xland=s+2,lakemask=s,mavail=s+1,roughness_m=s+.0001,ustar=s+.1)
    grid=SimpleNamespace(metrics=object())
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS','1' if dtype==jnp.float32 else '0')
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1' if dtype==jnp.float32 else '0')
    monkeypatch.setenv('GPUWRF_MYNN_PHY_EXNER','0')
    monkeypatch.setenv('GPUWRF_MYNN_QNI_MIXING','0')
    monkeypatch.setattr(C,'_wrf_hydrostatic_pressure_from_state',lambda *a,**kw:(state.p,s+1e5))
    monkeypatch.setattr(H,'_wrf_hydrostatic_pressure_from_state',lambda *a,**kw:(state.p,s+1e5))
    monkeypatch.setattr(C,'_wrf_phy_prep_rho_from_state',lambda *a,**kw:z+1)
    monkeypatch.setattr(H,'_wrf_phy_prep_rho_from_state',lambda *a,**kw:z+1)
    monkeypatch.setattr(C,'_surface_dz_from_state',lambda *a,**kw:jnp.moveaxis(z+25,0,-1))
    monkeypatch.setattr(H,'_surface_dz_from_state',lambda *a,**kw:jnp.moveaxis(z+25,0,-1))
    for mod in [C,H]:
        for n in ['_u_mass','_v_mass','_w_mass']:
            if hasattr(mod,n):monkeypatch.setattr(mod,n,lambda *a,**kw:z)
    return state,grid


@pytest.mark.parametrize('entry',['mynn','noah'])
def test_actual_native_consumer_matches_original_bits(monkeypatch,entry):
    state,grid=columns(monkeypatch);monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    fn=C._mynn_column_from_state if entry=='mynn' else H._build_column_view
    with jax.enable_x64(False):out=fn(state,grid)
    got=np.asarray(out.theta).reshape(-1);ref=np.asarray([r['WRF_theta'] for r in DATA['rows']],np.float32)
    for x in [got,ref,np.asarray(state.theta),np.asarray(state.qv)]:assert np.isfinite(x).all()
    np.testing.assert_array_equal(got.view(np.uint32),ref.view(np.uint32))
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',False)
    with jax.enable_x64(False):legacy=np.asarray(fn(state,grid).theta).reshape(-1)
    assert np.isfinite(legacy).all()
    assert np.count_nonzero(legacy!=ref)>=90,'real-sensitive operands lost'


def test_real_ratio_bits_and_double_legacy_unchanged(monkeypatch):
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    assert np.float32(C._wrf_phy_rv_over_rd(jnp.float32)).view(np.uint32)==1070456530
    assert C._wrf_phy_rv_over_rd(jnp.float64)==461.6/287.
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',False)
    assert np.float32(C._wrf_phy_rv_over_rd(jnp.float32)).view(np.uint32)==1070456529


@pytest.mark.parametrize('entry',['radiation','surface'])
def test_real_physics_temperature_consumer_uses_original_demix(monkeypatch,entry):
    state,grid=columns(monkeypatch);monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    ref=np.asarray([r['WRF_theta'] for r in DATA['rows']],np.float32)
    class Captured(Exception):pass
    def capture(theta,p):
        got=np.asarray(theta).reshape(-1)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got.view(np.uint32),ref.view(np.uint32))
        raise Captured
    monkeypatch.setattr(C,'_temperature_from_theta',capture)
    fn=C._rrtmg_column_inputs_impl if entry=='radiation' else C._surface_column_view
    with jax.enable_x64(False),pytest.raises(Captured):fn(state,grid)


def test_native_writeback_uses_real_ratio(monkeypatch):
    state,grid=columns(monkeypatch);monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    with jax.enable_x64(False):out=C._mynn_column_from_state(state,grid)
    state.replace=lambda **kw:kw
    monkeypatch.setattr(C,'_add_a2c_u_increment',lambda a,b:a)
    monkeypatch.setattr(C,'_add_a2c_v_increment',lambda a,b:a)
    monkeypatch.setattr(C,'_output_dtype',lambda *a,**kw:jnp.float32)
    expected=np.asarray([r['WRF_theta'] for r in DATA['rows']],np.float32)*(np.float32(1)+WRF_RATIO*np.asarray([r['qv'] for r in DATA['rows']],np.float32))
    with jax.enable_x64(False):got=C._state_from_mynn_output(state,out,theta_output_is_dry=True)
    assert np.isfinite(np.asarray(got['theta'])).all()
    np.testing.assert_array_equal(np.asarray(got['theta']).reshape(-1).view(np.uint32),expected.view(np.uint32))


def test_double_consumers_keep_same_values(monkeypatch):
    state,grid=columns(monkeypatch,jnp.float64)
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',False)
    with jax.enable_x64(True):a=C._mynn_column_from_state(state,grid)
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    with jax.enable_x64(True):b=C._mynn_column_from_state(state,grid)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        assert np.isfinite(np.asarray(x)).all() and np.isfinite(np.asarray(y)).all()
        np.testing.assert_array_equal(x,y)


def test_resolved_constant_flag_splits_key(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import module_const_env_hash
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',False);a=module_const_env_hash()
    monkeypatch.setattr(C,'_PHY_PREP_REAL_CONST',True)
    assert module_const_env_hash()!=a


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field',['theta_m','qv','WRF_theta'])
def test_fixture_gate_requires_finite(bad,field):
    values={n:np.asarray([r[n] for r in DATA['rows']],np.float32) for n in ['theta_m','qv','WRF_theta']}
    values[field][0]=bad
    with pytest.raises(AssertionError):
        for x in values.values():assert np.isfinite(x).all()
