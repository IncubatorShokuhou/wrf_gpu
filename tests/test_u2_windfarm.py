"""Fitch vs pristine WRF, opt-in admission, turbine IO and guarded persistence."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.windfarm import WindFarm, read_windfarm, read_turbine_table, windfarm_errors
from gpuwrf.physics.wind_fitch import dragforce
from gpuwrf.io.windfarm_schema import configuration, validate_restart

ROOT=Path(__file__).resolve().parents[1]
TABLE='6\n80 80 .12 2000\n2 .9 0\n4 .85 60\n8 .8 500\n12 .7 1600\n20 .4 2000\n25 .1 2000\n'


def farm_files(path):
    (path/'wind-turbine-1.tbl').write_text(TABLE)
    (path/'windturbines-ij.txt').write_text('2 2 1\n2 2 1\n99 99 1\n')
    return read_windfarm(path,ij=1,nx=4,ny=3)


def test_pristine_asymmetric_rotor_center_operation_order():
    # Separate process: proof-only no-FMA flags must precede JAX import.
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',
             PYTHONPATH=str(ROOT/'src'),
             XLA_FLAGS='--xla_cpu_enable_fast_math=false --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp')
    run=subprocess.run([sys.executable,str(ROOT/'proofs/u2_windfarm/area_order.py')],
                       cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(run.stdout)['byte_equal']==5


def test_pristine_real32_dragforce_all_regimes():
    import importlib.util
    spec=importlib.util.spec_from_file_location('fitch_oracle',ROOT/'proofs/u2_windfarm/oracle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads((ROOT/'proofs/u2_windfarm/manifest.json').read_text())
    path=ROOT/'proofs/u2_windfarm/oracle.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['oracle_sha256']
    expected=np.load(path)
    cases=module.inputs()
    assert len(cases)==72
    outputs=[]
    for factor in (0.,.25,1.):
        selected=[case for case in cases if case['factor']==factor]
        farm=WindFarm(cases[0]['turbines'],factor)
        fn=jax.jit(lambda u,v,z,dz,qke,du,dv:dragforce(u,v,z,dz,qke,du,dv,dx=1000.,dt=18.,farm=farm))
        for case in selected:
            outputs.append(fn(*(jnp.asarray(case[k]) for k in ('u','v','z','dz','qke','du','dv'))))
    comparison={}
    for index,key in enumerate(('du','dv','qke','power')):
        actual=np.stack([np.asarray(output[index]) for output in outputs])
        if os.environ.get('FITCH_REQUIRE_BYTE')=='1':
            np.testing.assert_array_equal(actual.view(np.uint32),expected[key].view(np.uint32))
        np.testing.assert_allclose(actual,expected[key],rtol=3e-6,atol=.25 if key=='power' else 2e-7)
        diff=np.abs(actual.astype(np.float64)-expected[key].astype(np.float64))
        scale=np.maximum(np.abs(expected[key]),np.float32(1e-10))
        comparison[key]={'max_abs':float(diff.max()),'max_relative':float((diff/scale).max()),
                         'finite':bool(np.isfinite(actual).all()),'nonzero_reference':int(np.count_nonzero(expected[key]))}
    assert np.count_nonzero(expected['power'])>0
    assert np.max(expected['qke'])>.6
    # Factor=0 still has momentum drag, but no TKE source.
    assert np.array_equal(expected['qke'][:24],np.full((24,12,3,4),.6,np.float32))
    assert np.min(expected['du'])<0
    if os.environ.get('FITCH_COMPARISON_RECEIPT'):
        Path(os.environ['FITCH_COMPARISON_RECEIPT']).write_text(json.dumps(comparison,indent=2)+'\n')


@pytest.mark.parametrize('precision',[4,8])
def test_fitch_raw_coordinates_reuse_pristine_llxy_fixture(precision):
    from gpuwrf.io.obs_projection import ObservationProjection
    fixture=ROOT/'proofs/s1_obsnudge/full/savepoints/projection.npz'
    a=np.load(fixture);dtype=np.float32 if precision==4 else np.float64
    for values,truth in zip(a['inputs'],a[f'output_r{precision}'],strict=True):
        code,*parameters=values[:8]
        p=ObservationProjection(int(code),*parameters,dtype=dtype)
        raw=p.station_coordinates(*values[8:10],mass_offset=False)
        # Existing untouched oracle saves i+.5,j+.5 in REAL. Check that same
        # rounding boundary while product receives the raw llij coordinates.
        np.testing.assert_allclose(np.asarray(raw,dtype)+dtype(.5),truth[:2],rtol=4*np.finfo(dtype).eps,atol=4*np.finfo(dtype).eps)


def test_table_placement_and_identity(tmp_path):
    farm=farm_files(tmp_path)
    assert [(x,y) for x,y,_ in farm.turbines]==[(1,1),(1,1)]
    assert farm.turbines[0][2][:4]==(80.,80.,.12,2000.)
    old=farm.fingerprint
    (tmp_path/'wind-turbine-1.tbl').write_text(TABLE.replace('80 80','81 80'))
    assert read_windfarm(tmp_path,ij=1,nx=4,ny=3).fingerprint!=old


@pytest.mark.parametrize('mutant',['NaN .9 0','4 .9 0','0 .9 0','2 -.1 0','1e40 .9 0','1e-50 .9 0'])
def test_table_rejects_nonfinite_unordered_and_negative(tmp_path,mutant):
    file=tmp_path/'table.tbl';file.write_text(TABLE.replace('2 .9 0',mutant))
    with pytest.raises(ValueError):read_turbine_table(file)


def test_curve_speeds_must_be_distinct_in_wrf_real(tmp_path):
    file=tmp_path/'table.tbl';file.write_text(TABLE.replace('4 .85 60','2.00000001 .85 60'))
    with pytest.raises(ValueError,match='REAL32'):read_turbine_table(file)


def test_rotor_geometry_cannot_underflow_wrf_real(tmp_path):
    file=tmp_path/'table.tbl';file.write_text(TABLE.replace('80 80 .12','1e-50 1e-50 .12'))
    with pytest.raises(ValueError,match='REAL32'):read_turbine_table(file)


def test_latlon_map_uses_raw_wrf_ij_not_obs_offset(tmp_path):
    from gpuwrf.io.obs_projection import ObservationProjection
    farm_files(tmp_path)
    (tmp_path/'windturbines.txt').write_text('45 0 1\n')
    proj=ObservationProjection(3,45.,0.,30.,30.,0.,1000.,1000.)
    farm=read_windfarm(tmp_path,ij=0,nx=4,ny=3,projection=proj)
    # map_set knowni=knownj=1 at XLAT/XLONG(1,1): pristine llij yields 1,1.
    assert farm.turbines[0][:2]==(0,0)


def test_admission_and_binding(tmp_path):
    from gpuwrf.io.namelist_binding import bind_namelist_options,require_namelist_honoured
    from gpuwrf.io.namelist_check import validate_operational_namelist,UnsupportedSchemeError
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    from test_v013_operational_smoke import _grid,_namelist
    config={'physics':dict(windfarm_opt=1,windfarm_ij=1,windfarm_tke_factor=.5,bl_pbl_physics=5,sf_sfclay_physics=5)}
    validate_operational_namelist(config)
    require_namelist_honoured(config,('d01',))
    n=bind_namelist_options(_namelist(_grid()),config,'d01')
    assert (n.windfarm_opt,n.windfarm_ij,n.windfarm_tke_factor)==(1,1,.5)
    assert classify_scheme('windfarm_opt',1).status==SupportStatus.IMPLEMENTED
    for key,value in (('windfarm_opt',2),('bl_pbl_physics',1),('bldt',1),('windfarm_tke_factor',float('nan'))):
        bad={'physics':dict(config['physics'],**{key:value})}
        assert windfarm_errors(bad)
        with pytest.raises(UnsupportedSchemeError):validate_operational_namelist(bad)


def test_wrf_per_domain_defaults_and_global_controls():
    from gpuwrf.io.windfarm import windfarm_option
    from gpuwrf.io.namelist_binding import bound_options
    for raw,expected in ((1,[1,0,0]),([1,0,1],[1,0,1]),([1,None,1],[1,0,1]),([1,1],[1,1,0])):
        nl={'physics':{'windfarm_opt':raw}}
        assert [windfarm_option(nl,f'd{i:02d}') for i in (1,2,3)]==expected
        assert [bound_options(nl,f'd{i:02d}')['windfarm_opt'] for i in (1,2,3)]==expected
    for key in ('windfarm_ij','windfarm_tke_factor'):
        assert any('scalar' in message for message in windfarm_errors({'physics':{'windfarm_opt':1,key:[1,1]}}))


@pytest.mark.parametrize('stochastic',[False,True])
@pytest.mark.parametrize('tracers',[False,True])
@pytest.mark.parametrize('ocean',[False,True])
def test_namelist_pytree_and_restart_refusals(tmp_path,stochastic,tracers,ocean):
    from test_v013_operational_smoke import _grid,_namelist
    from gpuwrf.io.stochastic_config import bind_stochastic
    from gpuwrf.tracers.config import TracerSpecies, UserTracerConfig
    farm=farm_files(tmp_path)
    stoch=bind_stochastic({'stoch':{'sppt':1}},'d01') if stochastic else None
    tracer_config = UserTracerConfig((TracerSpecies('DYE'),), (), 'a'*64) if tracers else None
    n=replace(_namelist(_grid()),windfarm_opt=1,windfarm_ij=1,windfarm=farm,stochastic_config=stoch,
              tracer_config=tracer_config)
    if ocean:
        from gpuwrf.physics.ocean_mixed_layer import initialize
        initial = initialize(jnp.full((n.grid.ny,n.grid.nx),290.,jnp.float32),50.)
        n=replace(n,sf_ocean_physics=1,ocean_initial=initial,use_noahmp=True,sf_surface_physics=4)
    leaves,tree=jax.tree.flatten(n)
    restored=jax.tree.unflatten(tree,leaves)
    assert restored.windfarm==farm and restored.windfarm_opt==1 and restored.windfarm_ij==1
    assert restored.stochastic_config is stoch
    assert restored.tracer_config is tracer_config
    assert restored.sf_ocean_physics == int(ocean)
    if ocean:
        for actual,expected in zip(jax.tree.leaves(restored.ocean_initial),jax.tree.leaves(initial),strict=True):
            np.testing.assert_array_equal(actual,expected)
    power=np.full((3,4),100.,np.float32)
    conf=configuration(n)
    validate_restart(power,conf,power.shape,requested=n)
    for value in (None,np.full((3,4),np.nan,np.float32),np.zeros((4,3),np.float32),np.zeros((3,4),np.float64)):
        with pytest.raises(ValueError):validate_restart(value,conf,power.shape,requested=n)
    with pytest.raises(ValueError):validate_restart(power,conf,power.shape,requested=replace(n,windfarm=None))
    with pytest.raises(ValueError):validate_restart(power,conf,power.shape,requested=replace(n,windfarm_opt=0))
    with pytest.raises(ValueError):validate_restart(power,None,power.shape,requested=n)


def test_fitch_caller_reads_input_wind_and_preserves_qke_base(tmp_path):
    from gpuwrf.coupling.windfarm_hook import windfarm_sources
    from gpuwrf.coupling.physics_couplers import _u_mass,_v_mass
    from test_v013_operational_smoke import _grid,_namelist,_base_state
    grid=_grid(nz=12,ny=3,nx=4)
    entry=_base_state(grid,dz_m=20.).replace(u=jnp.full((12,3,5),8.,jnp.float32),v=jnp.full((12,4,4),2.,jnp.float32))
    after=entry.replace(u=entry.u*2,v=entry.v*2,qke=jnp.full((12,3,4),.7,jnp.float32))
    n=replace(_namelist(grid),windfarm_opt=1,windfarm=farm_files(tmp_path))
    du=jnp.full((12,3,4),.001,jnp.float32);dv=-du
    out,a,b,power=windfarm_sources(entry,after,du,dv,n)
    z=jnp.asarray(entry.ph,jnp.float32)/jnp.float32(9.81)
    expected=dragforce(_u_mass(entry,dtype=jnp.float32),_v_mass(entry,dtype=jnp.float32),z,z[1:]-z[:-1],after.qke,du,dv,
                       dx=grid.projection.dx_m,dt=n.dt_s,farm=n.windfarm)
    for got,want in zip((a,b,out.qke,power),expected):np.testing.assert_array_equal(got,want)
    np.testing.assert_array_equal(out.u,after.u)
    assert float(power[1,1])>0 and float(jnp.max(out.qke))>.7


def test_doctor_density_and_dt_scale(tmp_path):
    from gpuwrf.doctor.windfarm import density_advice
    farm=farm_files(tmp_path)
    a=density_advice(farm,dx=1000.,dt=18.,min_rotor_dz=20.,max_speed=10.)
    b=density_advice(farm,dx=500.,dt=72.,min_rotor_dz=20.,max_speed=10.)
    assert a['max_turbines_per_cell']==2 and a['max_density_m_minus2']==2e-6
    assert b['initial_drag_dt_bound']==16*a['initial_drag_dt_bound']
    assert not a['warning'] and b['warning']


def test_doctor_case_windfarm_and_manifest_are_wired(tmp_path):
    from test_doctor import case
    from gpuwrf.doctor.case import inspect_case
    from gpuwrf.doctor.common import case_inputs
    case.__wrapped__(tmp_path)
    farm_files(tmp_path)
    path=tmp_path/'namelist.input'
    path.write_text(path.read_text().replace('&physics','&physics\n windfarm_opt=1, windfarm_ij=1,',1))
    report=inspect_case(tmp_path)
    assert report['status']=='PASS',report
    advice=report['domains'][0]['advice']['windfarm']
    assert advice['turbines']==2 and advice['max_turbines_per_cell']==2
    assert {'windturbines-ij.txt','wind-turbine-1.tbl'} <= {p.name for p in case_inputs(tmp_path,path)}


def test_daily_windfarm_selects_continuous_full_carry(tmp_path):
    from types import SimpleNamespace
    from gpuwrf.integration.daily_pipeline import _run_forecast_sequence
    n=SimpleNamespace(windfarm_opt=1,sf_surface_physics=4,bl_pbl_physics=5)
    case=SimpleNamespace(state=object(),namelist=n)
    with pytest.raises(ValueError,match='windfarm.*full-carry'):
        _run_forecast_sequence(SimpleNamespace(hours=1),output_dir=tmp_path/'out',
                              case_builder=lambda config:(case,tmp_path),forecast_fn=lambda *args:None)


def test_daily_stepper_retains_power_across_segments(tmp_path,monkeypatch):
    """Structural cadence control; actual physics is qualified separately."""
    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid,_namelist,_base_state
    grid=_grid(nz=12,ny=3,nx=4);state=_base_state(grid)
    n=replace(_namelist(grid),windfarm_opt=1,windfarm=farm_files(tmp_path))
    c=initial_operational_carry(state).replace(windfarm_power=jnp.zeros((3,4),jnp.float32))
    monkeypatch.setattr(om,'_resolve_operational_suite',lambda n:None)
    monkeypatch.setattr(om,'_committed_initial_carry_for_run',lambda *args:c)
    monkeypatch.setattr(om,'build_clock_base',lambda n:None)
    monkeypatch.setattr(daily,'_commit_to_operational_device',lambda s:s)
    starts=[]
    def step(c,n,start,clock,**kwargs):
        starts.append(int(start))
        return c.replace(windfarm_power=c.windfarm_power+1)
    monkeypatch.setattr(om,'_advance_chunk',step)
    runner=daily._RucDailyStepper(state,n)
    runner(state,n,n.dt_s/3600);runner(state,n,n.dt_s/3600)
    assert starts==[1,2]
    np.testing.assert_array_equal(runner.carry.windfarm_power,np.full((3,4),2.,np.float32))


def test_pickle_netcdf_restart_and_power_history(tmp_path):
    from datetime import datetime,timezone
    import pickle
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf,bind_wrfout_domain_authority
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid,_namelist,_base_state
    from test_m7_netcdf_writer import authenticated_source_grid
    grid=_grid(nz=12,ny=3,nx=4);state=_base_state(grid)
    n=replace(_namelist(grid),windfarm_opt=1,windfarm=farm_files(tmp_path))
    power=jnp.arange(12,dtype=jnp.float32).reshape(3,4)*100.
    carry=initial_operational_carry(state).replace(windfarm_power=power)
    path=tmp_path/'restart.pkl'
    write_restart(carry,n,grid,5,path)
    out,_,_,step=read_restart(path,namelist=n)
    assert step==5
    np.testing.assert_array_equal(out.windfarm_power,power)
    np.testing.assert_array_equal(out.state.qke,state.qke)
    with path.open('rb') as f:payload=pickle.load(f)
    payload['carry']['extra_fields']['windfarm_power']=None
    with path.open('wb') as f:pickle.dump(payload,f)
    with pytest.raises(ValueError,match='Fitch'):read_restart(path)
    stamp=datetime(2026,7,25,tzinfo=timezone.utc);path=tmp_path/'restart.nc'
    write_wrfrst_carry(carry,grid,n,path,step_index=5,run_start=stamp,valid_time=stamp)
    out,_=read_wrfrst_carry(path,namelist=n)
    np.testing.assert_array_equal(out.windfarm_power,power)
    np.testing.assert_array_equal(out.state.qke,state.qke)
    with pytest.raises(ValueError,match='Fitch'):read_wrfrst_carry(path,namelist=replace(n,windfarm_opt=0))
    with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_WINDFARM_CONFIG')
    with pytest.raises(ValueError,match='Fitch'):read_wrfrst_carry(path,namelist=n)
    history=tmp_path/'history.nc'
    source=authenticated_source_grid(grid,'d01')
    write_wrfout_netcdf(state,grid,n,history,domain='d01',domain_authority=bind_wrfout_domain_authority('d01',source,grid),
                        run_start=stamp,valid_time=stamp,lead_hours=0.,diagnostics={'POWER':power})
    with Dataset(history) as ds:
        assert ds.variables['POWER'].units=='W'
        np.testing.assert_array_equal(ds.variables['POWER'][0],power)
