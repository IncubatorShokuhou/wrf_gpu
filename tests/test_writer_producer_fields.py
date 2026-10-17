"""History retention seams, original CPU schema/constants and restart refusal.

Mocked solves below test transcription only. They are not numerical fidelity
evidence; the existing WRF-qualified physical operators are unchanged.
"""
import json
import pickle
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import jax

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.producer_history import inactive_names
from gpuwrf.io.restart_schema import validate_held_state, validate_goddard_radiation, GODDARD_RADIATION_SCHEMA_VERSION
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout, _lookup
from test_m7_netcdf_writer import synthetic_case, writer_authority
from test_goddard_operational import case

ROOT = Path(__file__).resolve().parents[1] / 'proofs/t_writer_fields'
CONSTANTS = json.loads((ROOT / 'INACTIVE_CPU_ALL_FRAMES.json').read_text())
SAMPLES = np.load(ROOT / 'producer_samples.npz')


@pytest.mark.parametrize('case_id', tuple(CONSTANTS['cases']))
@pytest.mark.parametrize('full', (False, True))
def test_authorized_constants_are_original_cpu_bits_and_schema(tmp_path, case_id, full):
    proof = CONSTANTS['cases'][case_id]
    assert len(proof['frames']) == 25
    state, grid, nl = synthetic_case()
    nl.update(mp_physics={'c12':17,'c13':18,'c14':19,'c15':21,'c16':22,
                          'c20':40,'c46':8,'c87':55}[case_id], nwp_diagnostics=0)
    if case_id == 'c46':
        nl.update(mp_physics=8, cu_physics=1, bl_pbl_physics=5, sf_surface_physics=4,
                  stochastic_config=SimpleNamespace(patterns=(None,)*4+(object(),)*3))
    names = tuple(proof['fields'])
    authority = writer_authority(grid, 'd01')
    p = prepare_wrfout_payload(state, grid, nl, tmp_path/'wrfout', domain='d01',
        domain_authority=authority, valid_time='2023-01-15T00:00:00',
        run_start='2023-01-15T00:00:00', lead_hours=0., variable_subset=names,
        full_variable_set=full, diagnostics={})
    write_prepared_wrfout(p, expected_domain='d01', expected_domain_authority=authority, variable_subset=names)
    with Dataset(p.target) as ds:
        for name, ref in proof['fields'].items():
            assert ref['constant'] and ref['bits'] == 0 and ref['dtype'] == '<f4'
            assert all(f['fields'][name]['all_equal_bits'] and f['fields'][name]['bits'] == 0 for f in proof['frames'])
            assert name in ds.variables, f'missing qualified inactive field {name}'
            v = ds[name]
            assert list(v.dimensions) == ref['dimensions']
            assert v.dtype == np.float32
            assert np.all(np.asarray(v[:]).view(np.uint32) == ref['bits'])


def test_active_branches_never_receive_inactive_constants():
    # aercu_opt>0 activates the CCN/EF*/WACT producers; MSKF_REFL_10CM stays WRF's
    # untouched +0 unless do_radar_ref=1 (t-classd, morr_two_moment_aero.F:1183-1203).
    assert inactive_names({'mp_physics': 40, 'aercu_opt': 1}, _lookup) == ('MSKF_REFL_10CM',)
    assert inactive_names({'mp_physics': 40, 'aercu_opt': 1, 'do_radar_ref': 1}, _lookup) == ()
    for mp in (17,18,19,21,22):
        assert inactive_names({'mp_physics': mp, 'nwp_diagnostics': 1}, _lookup) == ()
    cfg = SimpleNamespace(patterns=(None,)*4+(object(),)*3)
    assert inactive_names({'stochastic_config': cfg, 'cu_physics': 1,
        'bl_pbl_physics': 5, 'sf_surface_physics': 3}, _lookup) == ()
    assert inactive_names({'sf_surface_physics':2,'grid_sfdda':2},_lookup)==()
    assert inactive_names({'sf_surface_physics':4,'grid_sfdda':1},_lookup)==()


@pytest.mark.parametrize('full',(False,True))
@pytest.mark.parametrize('resolved',(False,True))
def test_noahmp_fasdas_writes_only_qualified_inactive_cpu_values(tmp_path,full,resolved):
    proof=json.loads((ROOT/'FASDAS_NOAHMP_CPU_ALL_FRAMES.json').read_text())['cases']['c78']
    assert len(proof['frames'])==25
    state,grid,nl=synthetic_case();nl.update(sf_surface_physics=4)
    if resolved:nl['data_assimilation']=SimpleNamespace(options=SimpleNamespace(surface_code=2))
    else:nl['grid_sfdda']=2
    names=('HFX_BOTH','QFX_BOTH');auth=writer_authority(grid,'d01')
    p=prepare_wrfout_payload(state,grid,nl,tmp_path/'fasdas',domain='d01',domain_authority=auth,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=24,
        full_variable_set=full,variable_subset=names)
    for name in names:
        assert proof['fields'][name]['constant'] and proof['fields'][name]['bits']==0
        assert all(f['fields'][name]['bits']==0 and f['fields'][name]['all_equal_bits'] for f in proof['frames'])
    write_prepared_wrfout(p,expected_domain='d01',expected_domain_authority=auth,variable_subset=names)
    with Dataset(p.target) as ds:
        for name in names:
            assert name in ds.variables
            assert list(ds[name].dimensions)==proof['fields'][name]['dimensions']
            assert ds[name].dtype==np.float32
            assert np.all(np.asarray(ds[name][:]).view(np.uint32)==0)
    # Distinct nonzero physical fluxes were not substituted for inactive fields.
    assert np.max(state.hfx)>0 and np.max(state.lh)>0


def test_inactive_constant_cannot_hide_nonzero_supplied_value(tmp_path):
    state,grid,nl=synthetic_case();nl['mp_physics']=40
    with pytest.raises(ValueError, match='contradicts'):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'bad',domain='d01',
            domain_authority=writer_authority(grid,'d01'),valid_time='2023-01-15T00:00:00',
            run_start='2023-01-15T00:00:00',lead_hours=0,variable_subset=('EFCG',),
            diagnostics={'EFCG':np.ones(state.qv.shape,np.float32)})


@pytest.mark.parametrize('name,mp,bl', [('RIMI',13,5), ('EXCH_TKE',8,12)])
@pytest.mark.parametrize('full', (False,True))
def test_original_cpu_profile_values_survive_writer(tmp_path,name,mp,bl,full):
    gold=SAMPLES[name]
    state,grid,nl=synthetic_case(nx=4,ny=4,nz=gold.shape[0])
    setattr(state,name.lower(),gold.copy());nl.update(mp_physics=mp,bl_pbl_physics=bl)
    auth=writer_authority(grid,'d01')
    p=prepare_wrfout_payload(state,grid,nl,tmp_path/'profile',domain='d01',domain_authority=auth,
        valid_time='2023-01-15T00:00:00',run_start='2023-01-15T00:00:00',lead_hours=24,
        variable_subset=(name,),full_variable_set=full)
    write_prepared_wrfout(p,expected_domain='d01',expected_domain_authority=auth,variable_subset=(name,))
    with Dataset(p.target) as ds:
        assert np.asarray(ds[name][0]).tobytes()==gold.tobytes()
        meta=json.loads((ROOT/'PRODUCER_SAMPLES.json').read_text())[name]['attrs']
        for attr in ('description','units','MemoryOrder','stagger'):
            assert ds[name].getncattr(attr)==meta[attr]


@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
@pytest.mark.parametrize('name,mp,bl',[('RIMI',13,5),('EXCH_TKE',8,12)])
def test_nonfinite_producer_profile_is_refused(tmp_path,name,mp,bl,bad):
    state,grid,nl=synthetic_case();nl.update(mp_physics=mp,bl_pbl_physics=bl)
    value=np.ones(state.qv.shape,np.float32);value.flat[0]=bad
    setattr(state,name.lower(),value)
    with pytest.raises(ValueError,match='nonfinite held producer history '+name):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'bad',domain='d01',
            domain_authority=writer_authority(grid,'d01'),valid_time='2023-01-15T00:00:00',
            run_start='2023-01-15T00:00:00',lead_hours=0,variable_subset=(name,))


def test_gbm_existing_solve_profile_is_retained(monkeypatch):
    from gpuwrf.coupling import scan_adapters as a
    from test_v013_operational_smoke import _grid,_base_state
    gold=SAMPLES['EXCH_TKE'];grid=_grid(nz=gold.shape[0]);state=_base_state(grid)
    columns=jnp.moveaxis(jnp.asarray(gold),0,-1).reshape(16,-1)
    def solve(*args,**kw):
        z=jnp.zeros_like(args[0]);return dict(u=z,v=z,theta=z,qv=z,qc=z,
            tke=args[8],kethl=columns,ust=kw['ust'])
    monkeypatch.setattr(a,'gbm_columns',solve)
    out=a.gbm_pbl_adapter(state,18.,grid)
    assert np.asarray(out.exch_tke).tobytes()==gold.tobytes()


def test_sbu_existing_solve_riming_output_is_retained(monkeypatch):
    from gpuwrf.coupling import scan_adapters as a
    from gpuwrf.contracts.physics_interfaces import PhysicsTendency
    from test_v013_operational_smoke import _grid,_base_state
    grid=_grid();state=_base_state(grid)
    # Distinct valid diagnostic values make a zero-fill or axis swap fail.
    gold=np.linspace(.01,.99,24*4*4,dtype=np.float32).reshape(24,4,4)
    columns=jnp.moveaxis(jnp.asarray(gold),0,-1).reshape(16,24)
    def solve(*args,**kw):
        return PhysicsTendency(state_replacements=dict(zip(('theta','qv','qc','qr','qi','qs'),args[:6])),
            diagnostics={'ri3d':columns})
    monkeypatch.setattr(a,'sbu_ylin_physics_tendency',solve)
    out=a.sbu_ylin_adapter(state,18.,grid)
    assert np.asarray(out.rimi).tobytes()==gold.tobytes()


def test_mp28_input_emission_is_held_and_consumed_without_recomputation(monkeypatch):
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.nesting.scalar_initialization import initialize_extra_scalar_fields
    from test_v016_thompson_aero_threading import _tiny_state
    state=_tiny_state();ny,nx=state.t_skin.shape
    water=np.arange(ny*nx,dtype=np.float32).reshape(ny,nx)+13
    dust=water*.125
    class Run:
        namelist={'physics':{'mp_physics':[28]}}
        def load_wrfinput(self,domain,name,**kwargs):
            if name=='QNWFA2D':return water
            if name=='QNIFA2D':return dust
            if name=='QNWFA':return np.full(state.qv.shape,1e8,np.float32)
            if name=='QNIFA':return np.full(state.qv.shape,1e6,np.float32)
            raise KeyError(name)
    seeded=initialize_extra_scalar_fields(Run(),'d01',state,None)
    seeded=pc.thompson_aero_coldstart_init(seeded)
    np.testing.assert_array_equal(seeded.nwfa2d,water)
    np.testing.assert_array_equal(seeded.nifa2d,dust)
    def forbidden(*args,**kw):raise AssertionError('emission must not be recomputed from evolving geometry')
    monkeypatch.setattr(pc,'_aerosol_surface_emission_columns',forbidden)
    monkeypatch.setattr(pc,'step_thompson_aero_column_with_precip',lambda column,*a,**kw:(column,None))
    seen=[]
    def emit(column,w,d,dt):
        seen.append((np.asarray(w),np.asarray(d)))
        return column
    monkeypatch.setattr(pc,'apply_surface_aerosol_emission',emit)
    result=pc.thompson_aero_adapter(seeded.replace(ph_total=seeded.ph_total*1.1),18.)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0][0],water)
    np.testing.assert_array_equal(seen[0][1],dust)
    np.testing.assert_array_equal(result.nwfa2d,water)
    np.testing.assert_array_equal(result.nifa2d,dust)


def test_mp28_missing_ccn_profile_runs_original_fallback_before_holding_emission():
    from gpuwrf.coupling import physics_couplers as pc
    from test_v016_thompson_aero_threading import _tiny_state
    state=_tiny_state()
    assert np.max(np.asarray(state.nwfa))==0
    water=jnp.full_like(state.t_skin,999.)
    dust=jnp.full_like(state.t_skin,.125)
    out=pc.thompson_aero_coldstart_init(state.replace(nwfa2d=water,nifa2d=dust))
    # Original thompson_init overwrites NWFA2D in the missing-CCN branch,
    # but never assigns NIFA2D. This detects incorrectly preferring all input.
    assert np.all(np.asarray(out.nwfa2d)!=999.)
    np.testing.assert_array_equal(out.nifa2d,dust)
    again=pc.thompson_aero_coldstart_init(out.replace(ph_total=out.ph_total*1.1))
    np.testing.assert_array_equal(again.nwfa2d,out.nwfa2d)


def test_goddard_keeps_actual_input_tensor_after_mp_changes(case,monkeypatch):
    from gpuwrf.coupling import goddard_radiation as g
    from gpuwrf.runtime import operational_mode as om
    grid,nl,carry=case;nz,ny,nx=carry.state.qv.shape
    captured=[]
    def kernel(fn,columns,*args):
        captured.append(np.asarray(columns['radius']))
        z=jnp.zeros((ny*nx,),jnp.float32)
        return (jnp.zeros((ny*nx,nz),jnp.float32),)*2+(z,)*14
    # The older restart fixture's static arrays are 3x4 independently of its
    # GridSpec. Use its same static scalar values on the actual caller grid.
    static=jax.tree.map(lambda value:jnp.broadcast_to(value.reshape(-1)[0],(ny,nx)),nl.radiation_static)
    _,diag=g.paired_radiation(carry.state,grid,radiation_static=static,
        time_utc=datetime(2023,1,15,12,tzinfo=timezone.utc),_kernel_call=kernel)
    assert diag.radius_inputs is not None
    np.testing.assert_array_equal(diag.radius_inputs,captured[0])
    fresh=carry.replace(radiation_diagnostics=diag)
    changed=fresh.replace(state=fresh.state.replace(re_cloud=jnp.ones_like(fresh.state.qv)))
    held=om._refresh_rrtmg_driver(changed,nl,18.,False,None)
    fields=g.history_fields(held.radiation_diagnostics)
    for i,name in enumerate(g.RADIUS_HISTORY):
        np.testing.assert_array_equal(fields[name],np.moveaxis(captured[0][...,i].reshape(ny,nx,nz),-1,0))
    assert np.all(fields['RE_RAIN_GSFC']==0)
    assert np.all(fields['RE_GRAUPEL_GSFC']==0)
    assert np.all(fields['RE_HAIL_GSFC']==0)
    with pytest.raises(ValueError,match='Goddard'):
        validate_goddard_radiation(diag,nl,1)  # old schema lacks a guaranteed held tensor
    with pytest.raises(ValueError,match='radius_inputs'):
        validate_goddard_radiation(diag._replace(radius_inputs=None),nl,GODDARD_RADIATION_SCHEMA_VERSION)


@pytest.mark.parametrize('name,selection', [('rimi',{'mp_physics':13}),
    ('exch_tke',{'bl_pbl_physics':12}),('nwfa2d',{'mp_physics':28}),('nifa2d',{'mp_physics':28})])
def test_active_restart_refuses_missing_held_history(name,selection):
    shape=(3,4,5);fields=dict(qv=np.zeros(shape,np.float32),
        rimi=np.zeros(shape,np.float32),exch_tke=np.zeros(shape,np.float32),
        nwfa2d=np.zeros(shape[1:],np.float32),nifa2d=np.zeros(shape[1:],np.float32))
    fields.pop(name)
    with pytest.raises(ValueError,match=name):validate_held_state(fields,selection)


@pytest.mark.parametrize('name,selection',[('rimi',{'mp_physics':13}),('exch_tke',{'bl_pbl_physics':12})])
@pytest.mark.parametrize('fmt',('checkpoint','netcdf'))
def test_profile_restart_bytes_and_missing_active_refusal(tmp_path,name,selection,fmt,monkeypatch):
    from gpuwrf.runtime import checkpoint as cp
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io import wrfrst_netcdf as nc
    from gpuwrf.validation.tier2 import make_ideal_grid
    from gpuwrf.physics import rrtmg_mp_re
    from test_v0110_wrfrst_netcdf import _state
    monkeypatch.setattr(rrtmg_mp_re,'_MP_RE',False)
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    grid=make_ideal_grid(4,3,5);state=_state(grid)
    gold=jnp.arange(state.qv.size,dtype=jnp.float32).reshape(state.qv.shape)/16
    state=state.replace(**{name:gold})
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,**selection)
    path=tmp_path/fmt
    if fmt=='checkpoint':
        cp.write_checkpoint(state,nl,grid,2,path)
        restored=cp.read_checkpoint(path)[0]
        assert np.asarray(getattr(restored,name)).tobytes()==np.asarray(gold).tobytes()
        with path.open('rb') as f:payload=pickle.load(f)
        payload['state_fields'].pop(name);payload['state_field_order'].remove(name)
        payload['state_field_count']-=1
        with path.open('wb') as f:pickle.dump(payload,f)
        with pytest.raises(ValueError,match=name):cp.read_checkpoint(path)
    else:
        carry=initial_operational_carry(state)
        nc.write_wrfrst_carry(carry,grid,nl,path,valid_time='2023-01-15_00:00:00',
            run_start='2023-01-15_00:00:00',step_index=2)
        restored=nc.read_wrfrst_carry(path)[0].state
        assert np.asarray(getattr(restored,name)).tobytes()==np.asarray(gold).tobytes()
        with pytest.raises(ValueError,match=name):
            nc.write_wrfrst_carry(carry.replace(state=state.replace(**{name:None})),grid,nl,tmp_path/'bad',
                valid_time='2023-01-15_00:00:00',run_start='2023-01-15_00:00:00',step_index=2)
