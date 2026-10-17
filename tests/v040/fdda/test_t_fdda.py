"""Grid-FDDA ingestion, WRF held history and daily/restart lifetime regressions."""
from dataclasses import dataclass, fields, replace
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import tarfile
from io import BytesIO
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import fdda_grid as fg
from gpuwrf.coupling import fdda_rk
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.restart_schema import validate_grid_fdda

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT/'proofs/t_fdda/savepoints/wrf_grid1_hfx.npz'
jax.config.update('jax_enable_x64', True)


def operands(**options):
    with np.load(FIXTURE) as ds:
        a = dict(ds)
    opts = dict(guv=3e-4, gt=3e-4, gq=3e-4, interval_m=60, end_h=24)
    opts.update(options)
    inputs = fg.GridFddaInputs(**{n:jnp.asarray(a[n]) for n in a if n.endswith(('_old','_new'))},
                              options=fg.GridFddaOptions(**opts), record_steps=(0,))
    return a, inputs


def call(carry, inputs, a, sec, step):
    theta = (a['s18_THM'] + np.float32(300.)) / (np.float32(1.) + fdda_rk._RVRD_REAL * a['s18_QVAPOR'])
    zw = (a['s18_PHB'] + a['s18_PH']) / np.float32(9.81)
    return fg.fdda_driver_tendencies(carry, inputs, a['s18_U'], a['s18_V'], theta,
        a['s18_QVAPOR'], a['s18_PBLH'], a['s18_HGT'], zw, sec, step, jnp.float32,
        p_phy=a['s18_P'] + a['s18_PB'])


@pytest.mark.parametrize('compiled', [False, True])
def test_held_hfx_against_original_wrf_history(compiled):
    a, inputs = operands()
    carry = fg.initial_fdda_carry(inputs, jnp.float32, with_history=True)
    # WRF first call overwrites OLD with the simulation-entry dry theta.
    initial_th = (a['s0_THM'] + np.float32(300.)) / (np.float32(1.) + fdda_rk._RVRD_REAL * a['s0_QVAPOR'])
    carry = carry.replace(t_old=jnp.asarray(initial_th - np.float32(300.)))
    operation = lambda c: call(c, inputs, a, 18., 2)
    got = (jax.jit(operation) if compiled else operation)(carry)
    expected = a['s36_HFX_FDDA']
    assert np.count_nonzero(expected) == 527
    assert np.isfinite(expected).all() and np.isfinite(np.asarray(got.hfx_fdda)).all()
    # Independent REAL input-ULP bound for (val - theta)+300 cancellation,
    # followed by WRF's literal p/(287*theta)*1004*rate*dz. No tolerance fit.
    th = (a['s18_THM']+np.float32(300.))/(np.float32(1.)+fdda_rk._RVRD_REAL*a['s18_QVAPOR'])
    z = (a['s18_PHB']+a['s18_PH'])/np.float32(9.81)
    multiplier = np.abs((a['s18_P']+a['s18_PB'])/(np.float32(287.)*th)*np.float32(1004.)*(z[1:]-z[:-1]))
    amplitude = np.maximum(np.abs(initial_th-300.),np.abs(a['t_new'][0])) + np.abs(th)
    bound = 4*np.finfo(np.float32).eps*(np.abs(expected)+multiplier*np.float32(3e-4)*amplitude)
    assert np.all(np.abs(np.asarray(got.hfx_fdda)-expected) <= bound)


def test_hfx_is_held_through_cadence_zeroing_decoupling_and_early_return():
    a, inputs = operands(stepfg=3, fgdtzero=1)
    c = fg.initial_fdda_carry(inputs,jnp.float32,with_history=True)
    c = call(c,inputs,a,36.,3)
    assert np.count_nonzero(np.asarray(c.hfx_fdda)) > 0
    held = np.asarray(c.hfx_fdda).copy()
    zero = call(c,inputs,a,54.,4)
    assert not np.any(np.asarray(zero.rthndgdten))
    np.testing.assert_array_equal(zero.hfx_fdda,held)
    finished = fg.finish_fdda_rates(c,jnp.full(a['s18_MU'].shape,9e4,jnp.float32),
                                    jnp.ones(44,jnp.float32),jnp.zeros(44,jnp.float32),jnp.float32)
    np.testing.assert_array_equal(finished.hfx_fdda,held)
    off = call(c,inputs,a,90000.,6)
    assert not np.any(np.asarray(off.rthndgdten))
    np.testing.assert_array_equal(off.hfx_fdda,held)


def test_fixture_tracked_and_loads_from_clean_archive():
    manifest = json.loads(FIXTURE.with_name('MANIFEST.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest['npz_sha256']
    if not (ROOT/'.git').exists():
        return
    result = subprocess.run(['git','archive','HEAD','proofs/t_fdda/savepoints'],cwd=ROOT,
                            check=True,capture_output=True)
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        entry = archive.extractfile('proofs/t_fdda/savepoints/wrf_grid1_hfx.npz')
        with np.load(BytesIO(entry.read())) as data:
            assert np.count_nonzero(data['s36_HFX_FDDA']) == 527


def test_wrf_metadata_preserves_all_analysis_payloads(tmp_path):
    spec=importlib.util.spec_from_file_location('fdda_writer',ROOT/'proofs/o2_fdda/make_wrffdda.py')
    writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
    source=tmp_path/'original.nc'
    with Dataset(source,'w') as ds:
        ds.TITLE='OUTPUT FROM WRF V4.7.1 MODEL'
        ds.CEN_LAT=np.float32(46.)
    target=tmp_path/'wrffdda_d01'
    with Dataset(target,'w') as ds:
        for name,size in [('Time',1),('z',2),('y',3),('x',4)]:ds.createDimension(name,size)
        for field in ('U','V','T','Q','PH'):
            for suffix in ('OLD','NEW'):
                var=ds.createVariable(f'{field}_NDG_{suffix}','f4',('Time','z','y','x'))
                var[:]=np.arange(24,dtype=np.float32).reshape(1,2,3,4)
        before={n:v[:].tobytes() for n,v in ds.variables.items()}
        writer.wrf_metadata(ds,source,'independent test input')
        assert 'V4.' in ds.TITLE and ds.CEN_LAT == np.float32(46.)
        for n,v in ds.variables.items():
            assert v[:].tobytes() == before[n]
            assert v.FieldType == 104 and v.MemoryOrder == 'XYZ'
            assert v.stagger == {'U':'X','V':'Y','PH':'Z'}.get(n.split('_NDG_')[0],'')


def test_writer_uses_wrf_fdda2d_mass_rank(tmp_path,monkeypatch):
    import sys
    spec=importlib.util.spec_from_file_location('fdda_writer_rank',ROOT/'proofs/o2_fdda/make_wrffdda.py')
    writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
    a,_=operands()
    for stamp,sec in (('00',0),('01',18)):
        with Dataset(tmp_path/f'wrfout_d01_2023-01-15_{stamp}:00:00','w') as ds:
            ds.TITLE='OUTPUT FROM WRF V4.7.1 MODEL'
            for n,s in (('Time',1),('bottom_top',44),('bottom_top_stag',45),('south_north',3),
                        ('west_east',4),('south_north_stag',4),('west_east_stag',5)):
                ds.createDimension(n,s)
            for n in ('U','V','THM','QVAPOR','PH','MU'):
                z='bottom_top_stag' if n=='PH' else 'bottom_top'
                y='south_north_stag' if n=='V' else 'south_north'
                x='west_east_stag' if n=='U' else 'west_east'
                dims=('Time',y,x) if n=='MU' else ('Time',z,y,x)
                ds.createVariable(n,'f4',dims)[0]=a[f's{sec}_{n}']
    target=tmp_path/'wrffdda_d01'
    monkeypatch.setattr(sys,'argv',['writer',str(tmp_path),'d01','2023-01-15_00:00:00','60','1',str(target)])
    writer.main()
    with Dataset(target) as ds:
        for suffix,sec in (('OLD',0),('NEW',18)):
            v=ds[f'MU_NDG_{suffix}']
            assert v.dimensions==('Time','fdda_levels','south_north','west_east')
            np.testing.assert_array_equal(v[0,0],a[f's{sec}_MU'])
        assert all(ds[n].FieldType==104 for n in ds.variables if n!='Times')


@pytest.mark.parametrize('bad', ['missing','shape','dtype','nan','inf','-inf'])
def test_grid_restart_refuses_bad_held_hfx(bad):
    _, inputs=operands()
    c=fg.initial_fdda_carry(inputs,jnp.float32,with_history=True)
    value={'missing':None,'shape':jnp.zeros((1,)),'dtype':jnp.zeros_like(c.hfx_fdda,dtype=jnp.int32),
           'nan':jnp.full_like(c.hfx_fdda,jnp.nan),'inf':jnp.full_like(c.hfx_fdda,jnp.inf),
           '-inf':jnp.full_like(c.hfx_fdda,-jnp.inf)}[bad]
    with pytest.raises(ValueError,match='E78'):
        validate_grid_fdda(c.replace(hfx_fdda=value),SimpleNamespace(data_assimilation=inputs))


def test_grid_restart_refuses_spectral_carry_with_grid_input():
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs,SpectralFddaOptions,initial_spectral_carry
    _,inputs=operands()
    shape=(1,45,3,4)
    spectral=SpectralFddaInputs(**{f.name:getattr(inputs,f.name) for f in fields(inputs)
                                  if f.name not in ('options','record_steps')},
        options=SpectralFddaOptions(interval_m=60),record_steps=(0,),
        ph_old=jnp.zeros(shape),ph_new=jnp.zeros(shape))
    group=initial_spectral_carry(spectral,jnp.float32)
    with pytest.raises(ValueError,match='E78'):
        validate_grid_fdda(group,SimpleNamespace(data_assimilation=inputs))


@pytest.mark.parametrize('mode',['pickle','netcdf','checkpoint','native'])
def test_public_restarts_preserve_and_require_hfx(tmp_path,mode):
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid,build_neutral_state
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.runtime.restart_store import RestartStore
    _,inputs=operands()
    grid=build_flat_grid(nx=4,ny=3,nz=44,dx_m=3000.)
    s=build_neutral_state(grid)
    s=s.replace(re_cloud=jnp.zeros_like(s.theta),re_ice=jnp.zeros_like(s.theta),
                re_snow=jnp.zeros_like(s.theta),sfc_wspd=jnp.zeros((3,4)))
    group=fg.initial_fdda_carry(inputs,jnp.float32,with_history=True)
    group=group.replace(hfx_fdda=jnp.arange(528,dtype=jnp.float32).reshape(44,3,4)/16.)
    carry=initial_operational_carry(s).replace(fdda=group)
    nl=OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=None,dt_s=18.,data_assimilation=inputs)
    def roundtrip(c,path):
        if mode=='pickle':
            write_restart(c,nl,grid,2,path);return read_restart(path)[0].fdda
        if mode=='netcdf':
            write_wrfrst_carry(c,grid,nl,path,valid_time='2023-01-15_00:00:36',run_start='2023-01-15_00:00:00',step_index=2)
            return read_wrfrst_carry(path,namelist=nl)[0].fdda
        if mode=='checkpoint':
            write_checkpoint(s,nl,grid,2,path,runtime_state={'fdda':c.fdda})
            return read_checkpoint_with_runtime_state(path)[-1]['fdda']
        store=RestartStore(path,10**8,2,0)
        identity={'grid_fdda_config':{'d01':True}}
        generation,_=store.save({'d01':c},{'d01':2},{'d01':18.},identity)
        return store.read(generation,expected_identity=identity)[0]['carries']['d01'].fdda
    restored=roundtrip(carry,tmp_path/'good')
    assert isinstance(restored,fg.GridFddaHistoryCarry)
    for x,y in zip(jax.tree.leaves(group),jax.tree.leaves(restored),strict=True):
        assert np.asarray(x).dtype==np.asarray(y).dtype
        np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError,match='E78'):
        roundtrip(carry.replace(fdda=group.replace(hfx_fdda=None)),tmp_path/'bad')
    with pytest.raises(ValueError,match='E78'):
        roundtrip(carry.replace(fdda=None),tmp_path/'absent')


@pytest.mark.parametrize('full',[False,True])
def test_writer_emits_real_hfx_and_subset(tmp_path,full):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout,bind_wrfout_domain_authority
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid,build_neutral_state
    from test_m7_netcdf_writer import authenticated_source_grid
    grid=build_flat_grid(nx=4,ny=3,nz=44,dx_m=3000.)
    state=build_neutral_state(grid)
    authority=bind_wrfout_domain_authority('d01',authenticated_source_grid(grid,'d01'),grid)
    hfx=np.arange(528,dtype=np.float32).reshape(44,3,4)/16.
    path=tmp_path/'wrfout'
    kwargs=dict(domain='d01',domain_authority=authority,valid_time='2023-01-15_01:00:00',
                run_start='2023-01-15_00:00:00',lead_hours=1.,variable_subset=('HFX_FDDA',),full_variable_set=full)
    prepared=prepare_wrfout_payload(state,grid,{},path,diagnostics={'HFX_FDDA':jnp.asarray(hfx)},**kwargs)
    write_prepared_wrfout(prepared,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(path) as ds:
        assert ds['HFX_FDDA'].dimensions==('Time','bottom_top','south_north','west_east')
        assert ds['HFX_FDDA'].units=='W m-2'
        np.testing.assert_array_equal(ds['HFX_FDDA'][0],hfx)
    for bad in (np.nan,np.inf,-np.inf):
        with pytest.raises(ValueError,match='nonfinite.*HFX_FDDA'):
            prepare_wrfout_payload(state,grid,{},path,diagnostics={'HFX_FDDA':np.full_like(hfx,bad)},**kwargs)


@pytest.mark.parametrize('aux_minutes',[None,15])
def test_daily_fdda_keeps_cursor_and_global_clock(monkeypatch,tmp_path,aux_minutes):
    from gpuwrf.integration import daily_pipeline as daily,nested_pipeline
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
    from test_m7_daily_pipeline import _synthetic_case
    _,inputs=operands()
    case=_synthetic_case(tmp_path)
    # Host-loop regression: arithmetic is recorded; this is not forecast fidelity.
    nl=SimpleNamespace(dt_s=18.,radiation_cadence_steps=200,sf_surface_physics=4,sf_lake_physics=0,
                       sf_ocean_physics=0,isftcflx=0,
                       data_assimilation=inputs,rk_order=3,grid=case.grid)
    case=replace(case,namelist=nl)
    starts,seeds,history=[],[],[]
    @dataclass(frozen=True)
    class RecordingCarry:
        state: object
        fdda: object

        def replace(self, **kw):
            return replace(self, **kw)

    def seed(s,n):
        seeds.append(1)
        return RecordingCarry(s, fg.initial_fdda_carry(inputs,jnp.float32,with_history=True))
    def advance(c,n,start,clock,*,n_steps,cadence):
        starts.append((int(start),float(c.fdda.hfx_fdda[0,0,0])))
        return c.replace(fdda=c.fdda.replace(hfx_fdda=jnp.full_like(c.fdda.hfx_fdda,int(start)+n_steps-1)))
    monkeypatch.setattr(om,'_resolve_operational_suite',lambda n:None)
    monkeypatch.setattr(om,'build_clock_base',lambda n:None)
    monkeypatch.setattr(om,'_committed_initial_carry_for_run',seed)
    monkeypatch.setattr(om,'_advance_chunk',advance)
    monkeypatch.setattr(daily,'_commit_to_operational_device',lambda s:s)
    monkeypatch.setattr(daily,'prepare_wrfout_payload',lambda *a,**k:object())
    monkeypatch.setattr(daily,'write_wrfout_netcdf',lambda *a,**k:None)
    monkeypatch.setattr(daily,'_emit_auxhist_frame',lambda **k:k['output_dir']/str(k['lead_minutes']))
    monkeypatch.setattr(nested_pipeline,'_noahmp_surface_diagnostics_for_output',lambda *a,**k:{})
    def state_only_fallback(s,n,hours):
        starts.append((1,0.))  # an independently recorded State-only segment resets its clock
        return s
    monkeypatch.setattr(daily,'_segmented_forecast_fn',state_only_fallback)
    def output(*a,**kw):
        carry=kw.get('operational_carry')
        if carry is None:
            history.append(None)
            return {}
        result=fdda_rk.fdda_history(carry.fdda)
        history.append(float(result['HFX_FDDA'][0,0,0]))
        return result
    monkeypatch.setattr(daily,'_surface_diagnostics_for_output',output)
    cfg=daily.DailyPipelineConfig(hours=2,dt_s=18.,async_output=False,score=False,refresh_land_state_hourly=False,
        auxhist=AuxhistStreamConfig(interval_minutes=aux_minutes) if aux_minutes else None)
    daily._run_forecast_sequence(cfg,output_dir=tmp_path/'daily',forecast_fn=daily._segmented_forecast_fn,
                                 case_builder=lambda c:(case,tmp_path))
    stride=50 if aux_minutes else 200
    assert starts==[(i,float(i-1)) for i in range(1,401,stride)]
    assert seeds==[1] and history==list(range(stride,401,stride))


@pytest.mark.parametrize('full',[False,True])
def test_actual_nested_callback_writes_held_hfx(tmp_path,monkeypatch,full):
    from test_initial_history_wrf_t0 import _writer
    writer,state=_writer(tmp_path,lambda *a,**kw:{})
    writer._full_variable_set=full
    grid=writer.bundles['d01'].grid
    values={}
    for n,shape in (('u',(grid.nz,grid.ny,grid.nx+1)),('v',(grid.nz,grid.ny+1,grid.nx)),
                    ('t',(grid.nz,grid.ny,grid.nx)),('q',(grid.nz,grid.ny,grid.nx))):
        for suffix in ('old','new'):values[f'{n}_{suffix}']=jnp.zeros((1,*shape),jnp.float32)
    inputs=fg.GridFddaInputs(**values,options=fg.GridFddaOptions(interval_m=60))
    group=fg.initial_fdda_carry(inputs,jnp.float32,with_history=True)
    expected=np.arange(grid.nz*grid.ny*grid.nx,dtype=np.float32).reshape(grid.nz,grid.ny,grid.nx)/8.
    group=group.replace(hfx_fdda=jnp.asarray(expected))
    monkeypatch.setenv('GPUWRF_DYN_RK_FP32','0')
    result=writer('d01',200,SimpleNamespace(state=state,fdda=group))
    with Dataset(result['wrfout']) as ds:
        assert 'HFX_FDDA' in ds.variables
        np.testing.assert_array_equal(ds['HFX_FDDA'][0],expected)


def test_actual_daily_history_consumes_held_hfx(monkeypatch):
    from gpuwrf.integration import daily_pipeline as daily,nested_pipeline
    _,inputs=operands()
    group=fg.initial_fdda_carry(inputs,jnp.float32,with_history=True)
    group=group.replace(hfx_fdda=jnp.arange(528,dtype=jnp.float32).reshape(44,3,4)/8.)
    carry=SimpleNamespace(noahmp_land=None,noahmp_rad=None,radiation_diagnostics=None,
                          history_diagnostics=None,fdda=group)
    monkeypatch.setattr(nested_pipeline,'_noahmp_surface_diagnostics_for_output',lambda *a,**kw:{})
    values=daily._surface_diagnostics_for_output(object(),SimpleNamespace(grid=object()),datetime(2023,1,15),
                                               lead_seconds=36.,operational_carry=carry)
    assert 'HFX_FDDA' in values
    assert values['HFX_FDDA'] is group.hfx_fdda
