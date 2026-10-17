from datetime import datetime,timedelta,timezone
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.fdda_namelist import fdda_namelist_errors
from gpuwrf.io.namelist_binding import require_namelist_honoured
from gpuwrf.io.wrffdda import load_grid_fdda
from gpuwrf.assimilation.fdda_surface import *
from gpuwrf.assimilation.fdda_grid import initial_fdda_carry


def helper():
    path=Path(__file__).parents[1]/'fdda/test_o2_fdda_namelist_io.py'
    spec=importlib.util.spec_from_file_location('surface_grid_helpers',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def namelist(code=1,**changes):
    return helper().nl(grid_sfdda=[code],sgfdda_interval_m=[60],sgfdda_end_h=[2],
        guv_sfc=[3e-4],gt_sfc=[3e-4],gq_sfc=[3e-4],rinblw=[100.],**changes)


def write_surface(path,times,ny=4,nx=5,*,dtype="f4"):
    with Dataset(path,'w') as ds:
        for k,v in (('Time',None),('DateStrLen',19),('y',ny),('x',nx),('ys',ny+1),('xs',nx+1)):
            ds.createDimension(k,v)
        tv=ds.createVariable('Times','S1',('Time','DateStrLen'))
        for i,t in enumerate(times):tv[i]=np.frombuffer(t.strftime('%Y-%m-%d_%H:%M:%S').encode(),dtype='S1')
        for var,dims,value in (('U10',('y','xs'),4.),('V10',('ys','x'),-3.),('TH2',('y','x'),290.),
                               ('Q2',('y','x'),.007),('ODIS',('y','x'),30.)):
            for s,off in (('OLD',0.),('NEW',1.)):
                a=ds.createVariable(var+'_NDG_'+s,dtype,('Time',*dims))
                for i in range(len(times)):a[i]=np.float32(value+off+i)


@pytest.mark.parametrize('code',[1,2])
def test_namelist_and_auxinput9_cursor(code,tmp_path):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    nl=namelist(code)
    validate_operational_namelist(nl)
    assert any('grid_sfdda' in w for w in require_namelist_honoured(nl,('d01',)))
    start=datetime(2023,1,15);times=[start+timedelta(hours=i) for i in range(3)]
    helper()._write_wrffdda(tmp_path/'wrffdda_d01',times)
    write_surface(tmp_path/'sgfdda_d01',times)
    inp=load_grid_fdda(tmp_path,nl,'d01',run_start=start,dt_s=54,nz=3,ny=4,nx=5,dx_m=3000,run_seconds=8000)
    assert isinstance(inp,SurfaceGridFddaInputs) and inp.surface_steps==(0,67,134)
    assert inp.options.end_h_sfc==2 and inp.options.guv_sfc==3e-4
    np.testing.assert_array_equal(inp.surface.th2_old[1],np.float32(291.))
    from gpuwrf.assimilation.fdda_surface import read_surface_record
    c=initial_fdda_carry(inp,jnp.float32)
    c=read_surface_record(c,inp,67,jnp.float32)
    assert int(c.surface_cursor)==0
    c=read_surface_record(c,inp,68,jnp.float32)
    assert int(c.surface_cursor)==1
    np.testing.assert_array_equal(c.surface.th2_old,np.float32(291.))
    validate_surface_restart(c,SimpleNamespace(data_assimilation=inp))
    with Dataset(tmp_path/'sgfdda_d01','r+') as ds:ds['Q2_NDG_NEW'][0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        load_grid_fdda(tmp_path,nl,'d01',run_start=start,dt_s=54,nz=3,ny=4,nx=5,dx_m=3000,run_seconds=8000)


@pytest.mark.parametrize('changes,word',[
    ({'grid_fdda':[0]},'requires grid_fdda=1'),({'grid_fdda':[2]},'requires grid_fdda=1'),
    ({'sgfdda_begin_h':[1]},'sgfdda_begin_h'),({'sgfdda_interval_h':[1]},'sgfdda_interval_h'),
    ({'sgfdda_end_m':[1]},'sgfdda_end_m'),({'io_form_sgfdda':11},'NetCDF'),
    ({'guv_sfc':[np.nan]},'guv_sfc'),({'gt_sfc':[-1.]},'gt_sfc'),({'gq_sfc':[-1.]},'gq_sfc'),
    ({'rinblw':[0.]},'rinblw'),({'sgfdda_interval_m':[0]},'interval'),])
def test_surface_refusals(changes,word):
    nl=namelist();nl['fdda'].update(changes)
    assert any(word in e for e in fdda_namelist_errors(nl,('d01',)))


def test_classic_noah_fasdas_and_unqualified_surface_refused():
    nl=namelist(2);nl['physics']['sf_surface_physics']=[2]
    assert any('indirect FASDAS soil' in e for e in fdda_namelist_errors(nl,('d01',)))
    nl=namelist();nl['physics']['sf_sfclay_physics']=[1]
    assert any('REGIME' in e for e in fdda_namelist_errors(nl,('d01',)))


def test_active_negative_roughness_is_not_repaired():
    # Pristine SFDDAGD's active znt0**0.2 is invalid for znt0<0.
    # The inactive IF branch may use benign operands; the active one must
    # expose the invalid arithmetic rather than flooring roughness to zero.
    from test_surface_oracle import load
    from gpuwrf.assimilation.fdda_grid import fdda_driver_tendencies
    d,inp=load('swiss_sfc1_wind_census_r4.npz')
    c=initial_fdda_carry(inp,jnp.float32)
    f=lambda n:jnp.asarray(d['frame1_'+n])
    c=fdda_driver_tendencies(c,inp,f('u'),f('v'),f('th_phy'),f('qv'),f('pblh'),d['ht'],f('z_at_w'),
        0.,1,jnp.float32,surface_context=(f('t_phy'),f('p_phy'),f('regime'),-jnp.ones_like(f('znt'))))
    assert np.isnan(np.asarray(c.rundgdten)).any() and np.isnan(np.asarray(c.rvndgdten)).any()


def storage_case():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    path=Path(__file__).parents[2]/'test_p0_5_restart_full_carry.py'
    spec=importlib.util.spec_from_file_location('surface_restart_helper',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    g=GridSpec.canary_3km_template();nz,ny,nx=g.nz,g.ny,g.nx
    state=m._state(g).ensure_conditional_leaves(mp_physics=8)
    shape=(nz,ny,nx)
    state=state.replace(re_cloud=jnp.zeros(shape),re_ice=jnp.zeros(shape),re_snow=jnp.zeros(shape),
                        sfc_wspd=jnp.zeros((ny,nx)))
    arrays={n+'_'+s:jnp.ones((2,*sz),jnp.float32) for n,sz in
            (('u',(nz,ny,nx+1)),('v',(nz,ny+1,nx)),('t',shape),('q',shape)) for s in ('old','new')}
    surf=SurfaceAnalysis(**{n+'_'+s:jnp.full((2,*sz),i+.25,jnp.float32) for i,(n,sz) in enumerate(
        (('u10',(ny,nx+1)),('v10',(ny+1,nx)),('th2',(ny,nx)),('q2',(ny,nx)),('odis',(ny,nx)))) for s in ('old','new')})
    inp=SurfaceGridFddaInputs(**arrays,options=SurfaceFddaOptions(interval_m=60,rinblw=100),record_steps=(0,200),
                              surface=surf,surface_steps=(0,200))
    fdda=initial_fdda_carry(inp,jnp.float32).replace(hfx_fdda=jnp.full(shape,12.5,jnp.float32))
    c=initial_operational_carry(state).replace(fdda=fdda)
    nl=OperationalNamelist(grid=g,tendencies=None,metrics=g.metrics,dt_s=18,acoustic_substeps=4,data_assimilation=inp)
    return g,nl,c


def exact(a,b):
    aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b)
    assert ta==tb
    for x,y in zip(aa,bb,strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()


def test_actual_restart_formats_and_missing_cursor(tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.runtime.restart_store import RestartStore,nested_identity
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    g,nl,c=storage_case()
    write_restart(c,nl,g,10,tmp_path/'r.pkl')
    exact(read_restart(tmp_path/'r.pkl')[0].fdda,c.fdda)
    write_wrfrst_carry(c,g,nl,tmp_path/'r.nc',valid_time='2023-01-15_00:03:00',run_start='2023-01-15_00:00:00',step_index=10)
    exact(read_wrfrst_carry(tmp_path/'r.nc',namelist=nl)[0].fdda,c.fdda)
    with pytest.raises(ValueError,match='E78'):write_checkpoint(c.state,nl,g,10,tmp_path/'c.pkl')
    write_checkpoint(c.state,nl,g,10,tmp_path/'c.pkl',runtime_state={'fdda':c.fdda})
    exact(read_checkpoint_with_runtime_state(tmp_path/'c.pkl')[-1]['fdda'],c.fdda)
    config=NestedPipelineConfig(tmp_path,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    ident=nested_identity(config,{'d01':SimpleNamespace(namelist=nl)},datetime(2023,1,15,tzinfo=timezone.utc))
    store=RestartStore(tmp_path/'store',10**8,2,0)
    generation,_=store.save({'d01':c},{'d01':10},{'d01':18.},ident)
    exact(store.read(generation,expected_identity=ident)[0]['carries']['d01'].fdda,c.fdda)
    bad=c.replace(fdda=c.fdda.replace(surface_cursor=None))
    with pytest.raises(ValueError,match='E78'):write_restart(bad,nl,g,10,tmp_path/'bad.pkl')
    with pytest.raises(ValueError,match='E78'):store.save({'d01':bad},{'d01':10},{'d01':18.},ident)
    with Dataset(tmp_path/'r.nc','r+') as ds:ds.delncattr('GPUWRF_SURFACE_FDDA_SCHEMA_VERSION')
    with pytest.raises(ValueError,match='E78'):read_wrfrst_carry(tmp_path/'r.nc',namelist=nl)


@pytest.mark.parametrize('damage',['surface','cursor','nonfinite','shape','stream','hfx_nonfinite','hfx_shape'])
def test_restart_surface_damage_rejected(damage):
    g,nl,c=storage_case();group=c.fdda
    if damage=='surface':group=group.replace(surface=None)
    if damage=='hfx_nonfinite':group=group.replace(hfx_fdda=jnp.full_like(group.hfx_fdda,jnp.nan))
    if damage=='hfx_shape':group=group.replace(hfx_fdda=group.hfx_fdda[:1])
    if damage=='cursor':group=group.replace(surface_cursor=jnp.asarray(10,jnp.int32))
    if damage in ('nonfinite','shape','stream'):
        value=jnp.full_like(group.surface.q2_old,jnp.nan if damage=='nonfinite' else 5)
        if damage=='shape':value=value[:1]
        group=group.replace(surface=replace(group.surface,q2_old=value))
    with pytest.raises(ValueError,match='E78'):validate_surface_restart(group,nl)


@pytest.mark.parametrize('field',[n+'_NDG_'+s for n in ('U10','V10','TH2','Q2','ODIS') for s in ('OLD','NEW')])
@pytest.mark.parametrize('value',[1e40,-1e40])
def test_actual_surface_loader_refuses_finite_double_overflow(field,value,tmp_path):
    start=datetime(2023,1,15);times=[start+timedelta(hours=i) for i in range(3)]
    helper()._write_wrffdda(tmp_path/'wrffdda_d01',times)
    write_surface(tmp_path/'sgfdda_d01',times,dtype='f8')
    with Dataset(tmp_path/'sgfdda_d01','r+') as ds:ds[field][0,0,0]=value
    assert np.isfinite(value)
    with pytest.raises(ValueError,match=field+'.*REAL32'):
        load_grid_fdda(tmp_path,namelist(),'d01',run_start=start,dt_s=54,nz=3,ny=4,nx=5,dx_m=3000,run_seconds=8000)


@pytest.mark.parametrize('modes',[(1,2),(2,1),(2,1,2)])
def test_mixed_surface_modes_share_wrf_radius_normalization(modes,tmp_path):
    from gpuwrf.io.wrfsfdda import surface_errors
    nl=namelist();n=len(modes);domain=f'd{n:02d}'
    nl['domains']={'max_dom':n}
    nl['fdda'].update({k:[v[0]]*n for k,v in nl['fdda'].items() if isinstance(v,list)})
    nl['fdda'].update(grid_sfdda=list(modes),rinblw=[100.]+[-1.]*(n-1))
    nl['physics'].update({k:[v[0]]*n for k,v in nl['physics'].items()})
    assert not surface_errors(nl,domain)
    assert not fdda_namelist_errors(nl,tuple(f'd{i+1:02d}' for i in range(n)))
    start=datetime(2023,1,15);times=[start+timedelta(hours=i) for i in range(3)]
    helper()._write_wrffdda(tmp_path/f'wrffdda_{domain}',times)
    write_surface(tmp_path/f'sgfdda_{domain}',times)
    inp=load_grid_fdda(tmp_path,nl,domain,run_start=start,dt_s=54,nz=3,ny=4,nx=5,dx_m=3000,run_seconds=8000)
    assert inp.options.rinblw==100. and inp.options.surface_code==modes[-1]
    nl['fdda']['grid_sfdda']=[2]*n
    assert any('rinblw' in e for e in surface_errors(nl,domain))
    with pytest.raises(ValueError,match='rinblw'):
        load_grid_fdda(tmp_path,nl,domain,run_start=start,dt_s=54,nz=3,ny=4,nx=5,dx_m=3000,run_seconds=8000)
