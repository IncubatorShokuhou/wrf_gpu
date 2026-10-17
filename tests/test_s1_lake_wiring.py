"""Lake operational interfaces, actual LSM exclusion and guarded IO."""
from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace
from typing import NamedTuple
import pickle
import json
from pathlib import Path

import jax
from jax import numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.coupling.lake_surface_hook import LakeBundle,LakeCarry,noahmp_without_lakes,overlay_lake_history
from gpuwrf.physics.lake_initialization import LakeInitInputs,lake_initial_state
from gpuwrf.physics.lake_step import LakeSurface,LakeStepResult
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.io.restart_schema import validate_lake,LAKE_SCHEMA_VERSION
from test_v013_operational_smoke import _grid,_base_state,_namelist
from test_m7_netcdf_writer import authenticated_source_grid


def bundle_for(shape):
    values=(17,8,100.,0.,290.,1.,0.,2.,10.)
    inp=LakeInitInputs(*(jnp.full((1,),x,jnp.int32 if i<2 else jnp.float32) for i,x in enumerate(values)))
    init=lake_initial_state(inp)
    columns=init.columns
    surface=LakeSurface(*(jnp.full((1,),x,jnp.float32) for x in (12.,34.,56.,290.,1e-5,285.,286.,.006,.08)))
    carry=LakeCarry(columns,surface,jnp.zeros((1,),jnp.float32),jnp.zeros((2,*shape),jnp.float32))
    baseline=jax.tree.map(lambda x:jnp.broadcast_to(x[0],shape+x.shape[1:]),columns)
    return LakeBundle(shape,(1,),tuple(i for i in range(np.prod(shape)) if i!=1),carry,baseline,
                       jnp.array([47.],jnp.float32),jnp.full(shape,.98,jnp.float32),jnp.zeros(shape).at[0,1].set(1.),jnp.full(shape,2.))


def test_noahmp_call_excludes_actual_lake_columns():
    class Land(NamedTuple):
        t_skin:object
        soil:object
    class Flux(NamedTuple):
        hfx:object
    shape=(2,3)
    b=bundle_for(shape)
    land=Land(jnp.arange(6.).reshape(shape),jnp.arange(24.).reshape(4,*shape))
    calls=[]
    def original_kernel(l,f,s,dt,**kwargs):
        calls.append((l.t_skin.shape,l.soil.shape,np.asarray(l.t_skin).ravel().tolist()))
        return Land(l.t_skin+10,l.soil+20),Flux(l.t_skin+30)
    output,flux=noahmp_without_lakes(original_kernel,land,land,None,1.,b,energy_params=None,rad_params=None)
    assert calls==[((1,5),(4,1,5),[0.,2.,3.,4.,5.])]
    assert float(output.t_skin[0,1])==1.
    np.testing.assert_array_equal(output.soil[:,0,1],land.soil[:,0,1])
    assert float(output.t_skin[1,2])==15.
    assert flux.hfx.shape==shape and float(flux.hfx[0,1])==0.


@pytest.mark.parametrize('ccn_conc',[1.0e8,8.0e7])
def test_optional_lake_bundle_namelist_roundtrip(ccn_conc):
    grid=_grid(nz=4,ny=2,nx=3)
    b=bundle_for((2,3))
    n=replace(_namelist(grid),sf_lake_physics=1,lake_bundle=b,lakedepth_default=15.,use_lakedepth=0,lake_min_elev=12.,ccn_conc=ccn_conc)
    leaves,tree=jax.tree.flatten(n)
    out=jax.tree.unflatten(tree,leaves)
    assert out.lake_bundle is b
    assert out.ccn_conc==ccn_conc
    assert (out.lakedepth_default,out.use_lakedepth,out.lake_min_elev)==(15.,0,12.)
    assert tree == jax.tree.structure(n)  # static aux must not retrace per step


def _mynn_flux_scene():
    """Original BP109 MYNN primitive inputs, independent of the Lake oracle."""
    from gpuwrf.coupling import noahmp_surface_hook as noah
    path=Path(__file__).parent/'v025/b_phys/fixtures/mynn_surface_exner_wrf.json'
    fixture=json.loads(path.read_text())
    records=fixture['records'][:6]
    shape=(2,3)
    values={name:jnp.asarray([r[name] for r in records],jnp.float32).reshape(shape)
            for name in ('hfx','qfx','rho','qv','tsk','p_hyd','exner')}
    grid=_grid(nz=4,ny=2,nx=3)
    state=_base_state(grid).replace(hfx=jnp.zeros(shape,jnp.float32),qfx=jnp.zeros(shape,jnp.float32),
        t_skin=values['tsk'],qv=jnp.broadcast_to(values['qv'],(4,*shape)))
    view=noah._build_column_view(state,grid)
    view=view._replace(p=jnp.broadcast_to(values['p_hyd'][...,None],view.p.shape),
                       exner=jnp.broadcast_to(values['exner'][...,None],view.p.shape))
    b=bundle_for(shape)
    columns=jax.tree.map(lambda x:jnp.broadcast_to(x[0],(6,*x.shape[1:])),b.initial.columns)
    surface=b.initial.surface._replace(hfx=values['hfx'].ravel(),qfx=values['qfx'].ravel(),tsk=values['tsk'].ravel())
    surface=jax.tree.map(lambda x:jnp.broadcast_to(x,(6,)),surface)
    carry=LakeCarry(columns,surface,jnp.zeros((6,),jnp.float32),b.initial.background)
    b=replace(b,indices=tuple(range(6)),nonlake_indices=(),initial=carry,
              latitude=jnp.full((6,),47.,jnp.float32))
    return state,b,grid,view,values,fixture


@pytest.mark.parametrize('enabled',[False,True])
def test_lake_mynn_forwards_independent_original_wrf_exner(monkeypatch,enabled):
    from gpuwrf.coupling import lake_surface_hook as lake,noahmp_surface_hook as noah,physics_couplers as physics
    from gpuwrf.physics import surface_layer
    monkeypatch.setenv('GPUWRF_MYNN_PHY_EXNER','1' if enabled else '0')
    state,b,grid,view,values,fixture=_mynn_flux_scene()
    monkeypatch.setattr(noah,'_build_column_view',lambda *args:view)
    monkeypatch.setattr(physics,'_wrf_phy_prep_rho_from_state',
                        lambda *args:jnp.broadcast_to(values['rho'],(4,*b.shape)))
    # Fixed physical flux inputs isolate the real MYNN producer, not LakeMain.
    monkeypatch.setattr(lake,'lake_step',lambda columns,atmosphere:
        LakeStepResult(columns,b.initial.surface,jnp.zeros((6,),jnp.float64)))
    original=surface_layer.mynn_driver_surface_fluxes
    seen=[]
    def capture(*args,**kwargs):
        seen.append((args,kwargs))
        return original(*args,**kwargs)
    monkeypatch.setattr(surface_layer,'mynn_driver_surface_fluxes',capture)
    radiation=SimpleNamespace(lwdn=jnp.full(b.shape,300.),soldn=jnp.full(b.shape,500.))
    precip=SimpleNamespace(prcpconv=jnp.zeros(b.shape),prcpnonc=jnp.zeros(b.shape))
    updated,_,_=lake.lake_surface_step(state,b.initial,b,radiation=radiation,
        precipitation=precip,grid=grid,dt=18.)
    assert len(seen)==1
    args,kwargs=seen[0]
    got=np.stack([np.asarray(x).ravel() for x in (updated.theta_flux,updated.qv_flux,updated.fltv)],axis=1)
    assert np.isfinite(got).all()
    if enabled:
        assert 'exner' in kwargs
        np.testing.assert_array_equal(kwargs['exner'],values['exner'])
        # Bits were generated by original REAL4 MYNN expressions and constants,
        # not the Lake kernel or the JAX flux helper under test.
        expected=np.asarray([r['reference_bits'] for r in fixture['records'][:6]],np.int32).view(np.float32)
        ulp=np.abs(got.astype(float)-expected.astype(float))/np.maximum(np.spacing(abs(expected)),np.finfo(np.float32).tiny)
        assert np.max(ulp)<=fixture['native_tolerance_ulps']
    else:
        # F20: the shared handles stay physical; MYNN's driver limits live in its own view only.
        assert kwargs=={}
        # Flag-off regression: preserve the established pressure-based API.
        expected=np.stack([np.asarray(x).ravel() for x in original(*args,**kwargs)],axis=1)
        np.testing.assert_array_equal(got,expected)


def test_zero_lake_domain_keeps_changing_land_history(monkeypatch):
    from gpuwrf.coupling import lake_surface_hook as lake,noahmp_surface_hook as noah
    b=bundle_for((2,3))
    carry=LakeCarry(jax.tree.map(lambda x:x[:0],b.initial.columns),
                    jax.tree.map(lambda x:x[:0],b.initial.surface),
                    jnp.zeros((0,),jnp.float32),b.initial.background)
    b=replace(b,indices=(),nonlake_indices=tuple(range(6)),initial=carry,latitude=b.latitude[:0])
    def unnecessary(*args,**kwargs):
        raise AssertionError('zero lakes must skip atmospheric construction and LakeMain')
    monkeypatch.setattr(noah,'_build_column_view',unnecessary)
    monkeypatch.setattr(lake,'lake_step',unnecessary)
    state=object()
    for grdflx,albedo in ((7.,.125),(-11.,.25)):
        history={'grdflx':jnp.full(b.shape,grdflx,jnp.float32),'albedo':jnp.full(b.shape,albedo,jnp.float32)}
        updated,new,produced=lake.lake_surface_step(state,carry,b,radiation=None,
            precipitation=None,grid=None,dt=18.,history=history)
        assert updated is state and produced is history
        assert new.columns is carry.columns and new.surface is carry.surface
        written=overlay_lake_history({'GRDFLX':history['grdflx'],'ALBEDO':history['albedo']},new,b)
        np.testing.assert_array_equal(written['GRDFLX'],history['grdflx'])
        np.testing.assert_array_equal(written['ALBEDO'],history['albedo'])
        assert new.background.dtype==jnp.float32
        carry=new


def test_active_lake_restart_schema_fails_closed():
    b=bundle_for((2,3))
    validate_lake(b.initial,SimpleNamespace(sf_lake_physics=1),LAKE_SCHEMA_VERSION)
    with pytest.raises(ValueError,match='E78'):
        validate_lake(None,SimpleNamespace(sf_lake_physics=1),None)
    with pytest.raises(ValueError,match='E78'):
        validate_lake(b.initial._replace(background=None),SimpleNamespace(sf_lake_physics=1),1)
    with pytest.raises(ValueError,match='E78'):
        validate_lake(b.initial,SimpleNamespace(sf_lake_physics=1),0)
    for bad in (np.nan,np.inf,-np.inf):
        damaged=b.initial._replace(columns=b.initial.columns._replace(t_lake3d=b.initial.columns.t_lake3d.at[0,0].set(bad)))
        with pytest.raises(ValueError,match='E78'):
            validate_lake(damaged,SimpleNamespace(sf_lake_physics=1),1)


def test_lake_pickle_restart_exact_and_missing_refused(tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart
    grid=_grid(nz=4,ny=2,nx=3)
    b=bundle_for((2,3))
    n=replace(_namelist(grid),sf_lake_physics=1,lake_bundle=b)
    carry=initial_operational_carry(_base_state(grid)).replace(lake=b.initial)
    path=tmp_path/'lake.pkl'
    write_restart(carry,n,grid,5,path)
    out,_,_,step=read_restart(path)
    assert step==5
    for a,c in zip(jax.tree.leaves(carry.lake),jax.tree.leaves(out.lake),strict=True):
        np.testing.assert_array_equal(a,c)
    with path.open('rb') as f: payload=pickle.load(f)
    payload['carry']['extra_fields']['lake']=None
    with path.open('wb') as f: pickle.dump(payload,f)
    with pytest.raises(ValueError,match='E78'):
        read_restart(path)


def test_lake_netcdf_restart_and_history(tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf,bind_wrfout_domain_authority
    grid=_grid(nz=4,ny=2,nx=3)
    source=authenticated_source_grid(grid,'d01')
    b=bundle_for((2,3))
    n=replace(_namelist(grid),sf_lake_physics=1,lake_bundle=b)
    state=_base_state(grid)
    carry=initial_operational_carry(state).replace(lake=b.initial)
    stamp=datetime(2026,7,25,tzinfo=timezone.utc)
    path=tmp_path/'lake.nc'
    write_wrfrst_carry(carry,grid,n,path,step_index=5,run_start=stamp,valid_time=stamp,lead_hours=0.)
    out,_=read_wrfrst_carry(path,namelist=n)
    for a,c in zip(jax.tree.leaves(carry.lake),jax.tree.leaves(out.lake),strict=True):
        np.testing.assert_array_equal(a,c)
    with Dataset(path) as ds:
        assert ds.variables['T_LAKE3D'].shape==(1,10,2,3)
        assert ds.variables['T_SOISNO3D'].shape==(1,15,2,3)
        assert ds.variables['ZI3D'].shape==(1,16,2,3)
    diagnostics=overlay_lake_history({'T2':jnp.full((2,3),280.),'PSFC':jnp.full((2,3),95000.)},b.initial,b)
    history=tmp_path/'history.nc'
    write_wrfout_netcdf(state,grid,n,history,domain='d01',domain_authority=bind_wrfout_domain_authority('d01',source,grid),run_start=stamp,valid_time=stamp,lead_hours=0.,diagnostics=diagnostics)
    with Dataset(history) as ds:
        assert int(ds.SF_LAKE_PHYSICS)==1
        assert ds.variables['T_LAKE3D'].shape==(1,10,2,3)
        assert float(ds.variables['T2'][0,0,1])==285.
        assert float(ds.variables['T2'][0,1,2])==280.
        assert np.isfinite(ds.variables['T_LAKE3D'][:]).all()
    with Dataset(path,'a') as ds: ds.delncattr('GPUWRF_LAKE_SCHEMA_VERSION')
    with pytest.raises(ValueError,match='E78'):
        read_wrfrst_carry(path,namelist=n)


def test_lake_controls_bind_and_unsupported_combinations_refuse():
    from gpuwrf.io.namelist_binding import bind_namelist_options,namelist_honour_report
    from gpuwrf.io.namelist_check import validate_operational_namelist
    grid=_grid(nz=4,ny=2,nx=3)
    config={'physics':{'sf_lake_physics':[1],'sf_surface_physics':[4],'use_lakedepth':[0],
                       'lakedepth_default':[22.],'lake_min_elev':[7.]},'dynamics':{'diff_opt':1,'km_opt':4}}
    validate_operational_namelist(config)
    n=bind_namelist_options(_namelist(grid),config,'d01')
    assert (n.sf_lake_physics,n.use_lakedepth,n.lakedepth_default,n.lake_min_elev)==(1,0,22.,7.)
    errors,_=namelist_honour_report(config,('d01',))
    assert not errors
    config['physics']['sst_update']=1
    errors,_=namelist_honour_report(config,('d01',))
    assert any('sst_update' in e for e in errors)
