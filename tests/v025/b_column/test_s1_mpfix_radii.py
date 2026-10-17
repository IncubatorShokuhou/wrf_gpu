"""Literal pristine caller flags, active producers, held radii and restarts."""
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.coupling import scan_adapters as A
from gpuwrf.physics import rrtmg_mp_re as R
from test_mp_re import _real_state
from test_mp_re_activation import _case, _exact_finite
from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry

ROOT=Path(__file__).resolve().parents[3]
FIX=ROOT/'proofs/s1_mpfix/fixtures'
NAMES=('re_cloud','re_ice','re_snow')


@pytest.mark.parametrize('mp,use',[(mp,use) for mp in (8,18,24,28,40) for use in (0,1)])
def test_activation_matches_compiled_pristine_init(monkeypatch,mp,use):
    monkeypatch.setattr(R,'_MP_RE',True)
    rows=json.loads((FIX/'radius_manifest.json').read_text())['flags']
    ref=next(row[2:] for row in rows if row[:2]==[mp,use])
    assert R.mp_re_active(mp_physics=mp,use_mp_re=use)==bool(any(ref))


@pytest.mark.parametrize('mp',(18,24,28))
@pytest.mark.parametrize('selection',({'use_mp_re':0},{'ra_lw_physics':1},{'ra_sw_physics':1}))
def test_new_schemes_keep_inactive_selection_absent(monkeypatch,mp,selection):
    _,nml,carry=_case(monkeypatch,mp_physics=mp,**selection)
    assert not R.mp_re_active(nml)
    assert all(getattr(carry.state,n) is None for n in NAMES)


@pytest.mark.parametrize('mp',(18,24,28))
def test_new_schemes_hold_previous_mp_radii_for_both_radiation_calls(monkeypatch,mp):
    monkeypatch.setattr(R,'_MP_RE',True)
    state,_=_real_state()
    state=state.ensure_conditional_leaves(mp_physics=mp)
    held={n:jnp.full_like(state.qc,v) for n,v in zip(NAMES,(13e-6,60e-6,300e-6))}
    state=state.replace(**held,qc=jnp.zeros_like(state.qc),qi=jnp.zeros_like(state.qi),qs=jnp.zeros_like(state.qs))
    sw,lw,*_=C._rrtmg_column_inputs(state,None,time_utc='2026-02-28_00:00:00',use_mp_re=int(R.mp_re_active(mp_physics=mp)))
    for column in (sw,lw):
        for n in NAMES:
            np.testing.assert_array_equal(getattr(column,n),C._to_columns(held[n]))


@pytest.mark.parametrize('kind',('lw','sw'))
@pytest.mark.parametrize('land',(1.,2.))
def test_mediation_matches_literal_pristine_rrtmg_caller(kind,land):
    d=json.loads((FIX/'radiation_mediation.json').read_text())
    result=R.prepare_radiation_radii(*[jnp.asarray(d[key],jnp.float32)[None,:]
        for key in ('T','cf')],jnp.asarray([land]),*[jnp.asarray(d[key],jnp.float32)[None,:] for key in NAMES])
    expected=np.array(d['outputs'][kind][int(land)-1])
    # Snow cutoff/mass reduction belongs downstream of this caller block.
    for i,a in enumerate(result[:3]):
        ref=expected[:,i] if i<2 else np.minimum(expected[:,i],130.)
        np.testing.assert_array_equal(np.asarray(a)[0],ref.astype(np.float32))
    np.testing.assert_array_equal(expected[:,3:],np.tile([5,5,1],(8,1)))


def test_thompson_aero_nc_radius_producer_matches_pristine(monkeypatch):
    monkeypatch.setattr(R,'_MP_RE',True)
    monkeypatch.setenv('GPUWRF_THOMPSON_WRF_CONSTANTS','1')
    d=json.loads((FIX/'thompson_aero_radii.json').read_text())
    values={n:jnp.asarray(v,jnp.float32)[None,None,:] for n,v in zip(d['names'],d['inputs'])}
    grid=SimpleNamespace(nx=1,ny=1,nz=12)
    state=State(**{k:jnp.zeros(shape,DEFAULT_DTYPES.dtype_for(k)) for k,shape in _state_field_shapes(grid).items()})
    state=state.replace(p=C._from_columns(values['p']),theta=jnp.full_like(state.theta,280.))
    state=state.ensure_conditional_leaves(mp_physics=28)
    z=jnp.zeros_like(values['T'])
    out=SimpleNamespace(**values,qr=z,qg=z,Nr=z,Ns=z,Ng=z,nwfa=z,nifa=z)
    result=C._state_from_thompson_aero_output(state,out)
    for n,ref in zip(NAMES,d['radii']):
        assert getattr(result,n).dtype==jnp.float32
        np.testing.assert_allclose(np.asarray(getattr(result,n))[:,0,0],ref,rtol=2e-6,atol=0.)
    # Pristine variable-Nc radii must not silently take the mp8 fixed-Nc route.
    wrong=R.calc_thompson_effective_radii(values['T'],values['p'],values['qv'],values['qc'],values['qi'],values['Ni'],values['qs'])
    assert np.max(np.abs(np.array(d['radii'][0])-np.asarray(wrong.cloud).ravel()))>1e-6


def test_wsm7_held_diagnostics_are_written_with_prognostics(monkeypatch):
    monkeypatch.setattr(R,'_MP_RE',True)
    fixture=json.loads((ROOT/'proofs/v013/savepoints_wsm7_fp64/wsm7_case_4.json').read_text())
    # Route independently computed pristine radii through the actual common MP
    # writeback; kernel numeric parity is carried by its existing oracle gates.
    diag={n:jnp.asarray(fixture['columns'][on],jnp.float32)[None,None,:]
          for n,on in zip(NAMES,('RE_CLOUD','RE_ICE','RE_SNOW'))}
    nz=diag['re_cloud'].shape[-1]
    grid=SimpleNamespace(nx=2,ny=1,nz=nz)
    state=State(**{k:jnp.zeros(shape,DEFAULT_DTYPES.dtype_for(k)) for k,shape in _state_field_shapes(grid,mp_physics=24).items()})
    state=state.ensure_conditional_leaves(mp_physics=24)
    diag={n:jnp.broadcast_to(v,(1,state.theta.shape[2],nz)) for n,v in diag.items()}
    tend=PhysicsTendency(diagnostics={n:v.reshape(-1,nz) for n,v in diag.items()})
    result=A._apply_mp_replacements(state,tend,ny=1,nx=state.theta.shape[2],nz=nz)
    for n in NAMES:
        np.testing.assert_array_equal(getattr(result,n),C._from_columns(diag[n]))


@pytest.mark.parametrize('mp',(18,24,28))
def test_new_radius_carries_roundtrip_and_missing_leaf_refuses(monkeypatch,tmp_path,mp):
    import pickle
    grid,nml,carry=_case(monkeypatch,mp_physics=mp)
    assert R.mp_re_active(nml)
    carry=carry.replace(state=carry.state.replace(**{n:jnp.full_like(carry.state.qc,v) for n,v in zip(NAMES,(17e-6,80e-6,200e-6))}))
    pkl=tmp_path/'carry.pkl'; write_restart(carry,nml,grid,2,pkl)
    _exact_finite(carry,read_restart(pkl)[0])
    nc=tmp_path/'carry.nc'
    write_wrfrst_carry(carry,grid,nml,nc,valid_time='2026-02-28_00:01:48',run_start='2026-02-28_00:00:00',step_index=2)
    _exact_finite(carry,read_wrfrst_carry(nc)[0])
    with pkl.open('rb') as stream: payload=pickle.load(stream)
    payload['carry']['state_field_order'].remove('re_cloud')
    del payload['carry']['state_fields']['re_cloud']
    with pkl.open('wb') as stream: pickle.dump(payload,stream)
    with pytest.raises(ValueError,match='E78'): read_restart(pkl)


def test_wsm7_adapter_preserves_disabled_radius_selection(monkeypatch):
    monkeypatch.setattr(R,'_MP_RE',True)
    state,_=_real_state()
    state=state.ensure_conditional_leaves(mp_physics=24,use_mp_re=0)
    def column(theta,qv,qc,qr,qi,qs,qg,qh,*args):
        return PhysicsTendency(state_replacements={'theta':theta,'qv':qv},
            diagnostics={n:jnp.full_like(qv,v) for n,v in zip(NAMES,(20e-6,40e-6,200e-6))})
    monkeypatch.setattr(A,'wsm7_physics_tendency',column)
    result=A.wsm7_adapter(state,18.)
    assert result.re_cloud is result.re_ice is result.re_snow is None


def test_wsm7_adapter_uses_literal_wrf_dry_theta_and_finish(monkeypatch):
    # Original phy_prep :4834, microphysics_driver :2673, and
    # moist_physics_finish_em :5735; independent NumPy caller arithmetic.
    state,_=_real_state()
    state=state.ensure_conditional_leaves(mp_physics=24,use_mp_re=0)
    th=np.asarray(state.theta); qv=np.asarray(state.qv)
    ratio=np.float32(461.6)/np.float32(287.)
    dry=th/(np.float32(1.)+ratio*qv)
    captured={}
    def column(theta,vapor,*args):
        captured['theta']=np.asarray(theta)
        return PhysicsTendency(state_replacements={'theta':theta+np.float32(2.),
                                                    'qv':vapor+np.float32(1e-4)})
    monkeypatch.setattr(A,'wsm7_physics_tendency',column)
    result=A.wsm7_adapter(state,18.)
    expected_columns=np.moveaxis(dry,0,-1).reshape(-1,dry.shape[0])
    np.testing.assert_array_equal(captured['theta'],expected_columns)
    newdry=dry+np.float32(2.); newq=qv+np.float32(1e-4)
    expected=dry*(1.+ratio*qv)+(newdry-dry)*(1.+ratio*newq)+ratio*(newq-qv)*newdry
    np.testing.assert_allclose(result.theta,expected,rtol=2e-7,atol=0.)
