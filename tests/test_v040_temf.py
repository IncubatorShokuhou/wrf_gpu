"""TEMF pristine REAL gates, held state, opt-in pair and output contracts."""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata,DycoreMetrics,GridSpec,Projection,TerrainProvenance,VerticalCoord
from gpuwrf.contracts.state import State,Tendencies,_state_field_shapes
from gpuwrf.coupling import temf
from gpuwrf.io.restart_schema import TEMF_SCHEMA_VERSION,validate_temf
from gpuwrf.runtime.operational_mode import OperationalNamelist,_initial_carry_for_run,_physics_step_forcing,_resolve_operational_suite

ROOT=Path(__file__).resolve().parents[1]
PROOF=ROOT/'proofs/s2_temf'


def _compare_module():
    spec=importlib.util.spec_from_file_location('temf_comparison',PROOF/'compare.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _grid_state():
    nz,ny,nx=10,2,3
    eta=jnp.linspace(1.,0.,nz+1,dtype=jnp.float64)
    projection=Projection('lambert',28.3,-16.4,3000.,3000.,nx,ny)
    terrain=TerrainProvenance('temf-test','temf-test',(ny,nx),'m','native-wrf-lambert',0.,True)
    metrics=DycoreMetrics.flat(ny=ny,nx=nx,nz=nz,eta_levels=eta,top_pressure_pa=5000.,provenance='temf-test')
    grid=GridSpec(projection,terrain,VerticalCoord('hybrid_eta',nz,5000.,eta),BCMetadata('ideal',(),1,'linear',True),eta,jnp.zeros((ny,nx)),metrics=metrics)
    fields={k:jnp.zeros(shape,jnp.float32) for k,shape in _state_field_shapes(grid).items()}
    z=jnp.linspace(0.,5000.,nz+1,dtype=jnp.float32)[:,None,None]
    pressure=jnp.broadcast_to(jnp.linspace(97000.,25000.,nz,dtype=jnp.float32)[:,None,None],(nz,ny,nx))
    th=jnp.broadcast_to(jnp.asarray([303.,301.,301.,304.,307.,310.,313.,316.,319.,322.],jnp.float32)[:,None,None],(nz,ny,nx))
    qv=jnp.full((nz,ny,nx),.012,jnp.float32)
    fields.update(theta=th*(1.+jnp.float32(461.6/287.)*qv),qv=qv,
                  p_total=pressure,ph_total=jnp.broadcast_to(z*9.81,(nz+1,ny,nx)),
                  mu_total=jnp.full((ny,nx),95000.,jnp.float32),
                  u=jnp.full((nz,ny,nx+1),3.,jnp.float32),v=jnp.ones((nz,ny+1,nx),jnp.float32),
                  t_skin=jnp.full((ny,nx),306.,jnp.float32),roughness_m=jnp.full((ny,nx),.1,jnp.float32),
                  xland=jnp.ones((ny,nx),jnp.float32),mavail=jnp.full((ny,nx),.5,jnp.float32),
                  lu_index=jnp.zeros((ny,nx),jnp.int32))
    return grid,State(**fields)


def _namelist(grid,**updates):
    nz,ny,nx=grid.nz,grid.ny,grid.nx
    z=lambda shape:jnp.zeros(shape,jnp.float32)
    tendencies=Tendencies(z((nz,ny,nx+1)),z((nz,ny+1,nx)),z((nz+1,ny,nx)),z((nz,ny,nx)),
                          z((nz,ny,nx)),z((nz,ny,nx)),z((nz+1,ny,nx)),z((ny,nx)))
    args=dict(grid=grid,dt_s=18.,mp_physics=0,bl_pbl_physics=10,sf_sfclay_physics=10,
              sf_surface_physics=0,use_noahmp=False,cu_physics=0,ra_lw_physics=0,ra_sw_physics=0,
              tendencies=tendencies,metrics=grid.metrics,
              time_utc='2024-06-01T12:00:00Z')
    args.update(updates)
    return OperationalNamelist(**args)


@pytest.mark.parametrize('kind',['surface','pbl'])
def test_pristine_wrf_parity(kind,tmp_path):
    # CPU backend FMA/reassociation are audited separately in TF01b/TF02b.
    # A fresh process scopes literal-oracle compiler settings to this gate.
    env={k:v for k,v in os.environ.items() if not re.search(r'(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)',k,re.I)}
    env.update(JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',PYTHONPATH=str(ROOT/'src'),
               XLA_FLAGS='--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp')
    result=subprocess.run([sys.executable,str(PROOF/'compare.py'),str(PROOF/f'{kind}_oracle.npz'),kind,str(tmp_path/f'{kind}.json')],
                          cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((tmp_path/f'{kind}.json').read_text())
    assert receipt['backend']=='cpu' and not receipt['failures']
    if kind=='pbl':
        assert receipt['scalar_comparisons']==57744
        assert max(receipt['worst_ulp'].values())==0
        assert sum(receipt['byte_mismatches'].values())==0
        assert all(receipt['nonzero'][k]>0 for k in ('wupd_temf','mf_temf','cf3d_temf','rqcblten'))
        records=_compare_module().load(PROOF/'pbl_oracle.npz')['records']
        assert any(r['outputs']['hct_temf']>r['outputs']['hd_temf'] for r in records)
        assert any(np.any(np.asarray(r['outputs']['mf_temf'])>0) and np.any(np.asarray(r['outputs']['cf3d_temf'])>0) for r in records)
        stable=[r for r in records if r['name'].startswith('stable_')]
        assert stable and all(r['outputs']['pblh']==123. and r['outputs']['kpbl']==2 for r in stable)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_numeric_gate_rejects_nonfinite_candidate_and_reference(bad):
    compare=_compare_module()
    for got,exp in (([bad],[1.]),([1.],[bad])):
        with pytest.raises(AssertionError,match='nonfinite'):
            compare.numeric_gate(got,exp)


def test_fixture_assets_load_from_clean_git_archive(tmp_path):
    archive=subprocess.run(['git','archive','HEAD','src','proofs/s2_temf','data/fixtures','data/manifests','scripts'],
                           cwd=ROOT,check=True,capture_output=True).stdout
    tree=tmp_path/'archive'; tree.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(tree,filter='data')
    (tree/'data/wrf_pristine').symlink_to('<USER_HOME>/src/wrf_pristine')
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',PYTHONPATH=str(tree/'src'))
    script="""import importlib.util,json,pathlib,sys
p=pathlib.Path('proofs/s2_temf/compare.py')
s=importlib.util.spec_from_file_location('check',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
import gpuwrf.physics.bl_temf as b
assert str(pathlib.Path(b.__file__).resolve()).startswith(str(pathlib.Path.cwd()))
for kind,n in [('pbl',72),('surface',96)]:
    d=m.load(p.parent/(kind+'_oracle.npz')); assert len(d['records'])==n
    assert d['fixture_sha256'] and d['source_json_sha256']
print('archive assets/hash/load PASS')
"""
    result=subprocess.run([sys.executable,'-c',script],cwd=tree,env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_temf_pair_is_bound_and_unpaired_choices_refused():
    from gpuwrf.io.namelist_binding import namelist_honour_report
    from gpuwrf.io.namelist_check import validate_supported_namelist
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    grid,_=_grid_state()
    for key in ('bl_pbl_physics','sf_sfclay_physics'):
        c=classify_scheme(key,10)
        assert c.status is SupportStatus.IMPLEMENTED and 'CPU-oracle-qualified' in c.reason and 'pending' in c.reason
    assert _resolve_operational_suite(_namelist(grid)).pbl.option==10
    validate_supported_namelist({'physics':{'bl_pbl_physics':10,'sf_sfclay_physics':10}})
    for pbl,sfc in ((10,5),(5,10)):
        with pytest.raises(ValueError,match='TEMF'):
            _resolve_operational_suite(_namelist(grid,bl_pbl_physics=pbl,sf_sfclay_physics=sfc))
        errors,_=namelist_honour_report({'physics':{'bl_pbl_physics':pbl,'sf_sfclay_physics':sfc,'sf_surface_physics':0}},domains=('d01',))
        assert any('TEMF' in e for e in errors)
        with pytest.raises(ValueError,match='TEMF'):
            validate_supported_namelist({'physics':{'bl_pbl_physics':pbl,'sf_sfclay_physics':sfc}})


def test_temf_slot_threads_real_registry_state_and_surface_history():
    grid,state=_grid_state(); ns=_namelist(grid)
    carry=_initial_carry_for_run(state,ns)
    assert carry.temf is not None
    np.testing.assert_array_equal(carry.temf.profiles['te_temf'],np.full((6,10),1e-3,np.float32))
    old=carry.temf
    forcing=_physics_step_forcing(carry,ns,0.,run_radiation=False)
    after=forcing.carry.temf
    validate_temf(after,ns,TEMF_SCHEMA_VERSION)
    assert np.max(after.profiles['km_temf'])>0 and np.any(np.asarray(after.surface['wm_temf'])!=old.surface['wm_temf'])
    assert np.any(np.asarray(forcing.state.hfx)!=0)
    fields=temf.history_fields(after,state.t_skin.shape)
    assert set(temf.TEMF_PROFILES+temf.TEMF_SURFACE)<=fields.keys()
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(forcing.carry))
    np.testing.assert_array_equal(forcing.carry.history_diagnostics.t2,fields['T2'])
    assert forcing.carry.temf.profiles['te_temf'].dtype==jnp.float32


def test_temf_inactive_has_no_resident_leaves():
    grid,state=_grid_state()
    carry=_initial_carry_for_run(state,_namelist(grid,bl_pbl_physics=0,sf_sfclay_physics=0))
    assert carry.temf is None


def test_daily_pipeline_preserves_held_temf_history_with_general_carry():
    from datetime import datetime,timezone
    from gpuwrf.integration.daily_pipeline import _surface_diagnostics_for_output
    grid,state=_grid_state(); ns=_namelist(grid)
    carry=_physics_step_forcing(_initial_carry_for_run(state,ns),ns,0.,run_radiation=False).carry
    fields=_surface_diagnostics_for_output(carry.state,ns,datetime(2024,6,1,12,tzinfo=timezone.utc),
                                          lead_seconds=18.,operational_carry=carry)
    expect=temf.history_fields(carry.temf,state.t_skin.shape)
    for k,v in expect.items():
        np.testing.assert_array_equal(fields[k],v)


def test_temf_restart_roundtrip_and_schema_deletion_refusal(tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart,_read_payload
    import pickle
    grid,state=_grid_state(); ns=_namelist(grid)
    carry=_physics_step_forcing(_initial_carry_for_run(state,ns),ns,0.,run_radiation=False).carry
    assert np.max(np.asarray(carry.temf.surface['wm_temf']))>0
    path=tmp_path/'temf.restart'
    write_restart(carry,ns,grid,0,path)
    restored,_,_,_=read_restart(path)
    for a,b in zip(jax.tree.leaves(carry.temf),jax.tree.leaves(restored.temf),strict=True):
        np.testing.assert_array_equal(a,b)
    uninterrupted=_physics_step_forcing(carry,ns,18.,run_radiation=False).carry
    resumed=_physics_step_forcing(restored,ns,18.,run_radiation=False).carry
    for a,b in zip(jax.tree.leaves(uninterrupted),jax.tree.leaves(resumed),strict=True):
        np.testing.assert_array_equal(a,b)
    payload=_read_payload(path)
    del payload['carry']['temf_schema_version']
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='TEMF'):
        read_restart(path)
    for group in (None,carry.temf._replace(profiles={k:v for k,v in carry.temf.profiles.items() if k!='te_temf'})):
        with pytest.raises(ValueError,match='TEMF'):
            validate_temf(group,ns,TEMF_SCHEMA_VERSION)


def test_temf_netcdf_restart_roundtrip_and_marker_deletion(tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.io.netcdf_lock import Dataset
    grid,state=_grid_state(); ns=_namelist(grid)
    carry=_physics_step_forcing(_initial_carry_for_run(state,ns),ns,0.,run_radiation=False).carry
    assert np.max(np.asarray(carry.temf.surface['wm_temf']))>0
    path=tmp_path/'wrfrst_d01'
    write_wrfrst_carry(carry,grid,ns,path,run_start='2024-06-01T12:00:00Z',valid_time='2024-06-01T12:00:00Z',step_index=0)
    restored,_=read_wrfrst_carry(path,namelist=ns)
    for a,b in zip(jax.tree.leaves(carry.temf),jax.tree.leaves(restored.temf),strict=True):
        np.testing.assert_array_equal(a,b)
    with Dataset(path,'r+') as ds:
        ds.delncattr('GPUWRF_TEMF_SCHEMA_VERSION')
    with pytest.raises(ValueError,match='TEMF'):
        read_wrfrst_carry(path,namelist=ns)


def test_temf_history_registry_names_metadata_and_missing_output_refusal():
    from gpuwrf.io.temf_schema import TEMF_HISTORY
    from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS,XYZ,XY
    assert len(TEMF_HISTORY)==18
    registry={row['name']:row for row in json.loads((PROOF/'registry.json').read_text())['rows']}
    for name,description,units,volume in TEMF_HISTORY:
        spec=WRFOUT_VARIABLE_SPECS[name.upper()]
        assert spec.dimensions==(XYZ if volume else XY)
        assert spec.description==description and spec.units==(units or '-')
        ref=registry[name]
        assert (description,units,volume)==(ref['description'],ref['units'],ref['dimensions']=='ikj')


def test_temf_history_writes_registry_arrays_and_refuses_missing_carry(tmp_path):
    from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority,write_wrfout_netcdf
    from gpuwrf.io.netcdf_lock import Dataset
    from test_m7_netcdf_writer import authenticated_source_grid
    grid,state=_grid_state(); ns=_namelist(grid); carry=_initial_carry_for_run(state,ns)
    fields=temf.history_fields(carry.temf,state.t_skin.shape)
    path=tmp_path/'wrfout_d01'
    kwargs=dict(domain='d01',domain_authority=bind_wrfout_domain_authority('d01',authenticated_source_grid(grid,'d01'),grid),
                run_start='2024-06-01T12:00:00Z',valid_time='2024-06-01T12:00:00Z',lead_hours=0.,diagnostics=fields)
    write_wrfout_netcdf(state,grid,ns,path,**kwargs)
    with Dataset(path) as ds:
        for k in temf.TEMF_PROFILES+temf.TEMF_SURFACE:
            assert ds.variables[k.upper()].dtype==np.float32
            np.testing.assert_array_equal(ds.variables[k.upper()][0],fields[k])
        np.testing.assert_array_equal(ds.variables['TH2'][0],fields['TH2'])
    del fields['te_temf']
    with pytest.raises(ValueError,match='TEMF history is missing held te_temf'):
        write_wrfout_netcdf(state,grid,ns,tmp_path/'missing',**kwargs)


def test_temf_lead_zero_history_preserves_wrfinput_surface_authority(tmp_path):
    from types import SimpleNamespace
    from test_initial_history_wrf_t0 import _writer,_wrfinput,INPUT
    from test_m7_netcdf_writer import writer_authority
    from gpuwrf.io.netcdf_lock import Dataset
    def forbidden(*args,**kwargs):
        raise AssertionError('lead-zero surface solve')
    writer,_=_writer(tmp_path,forbidden)
    grid,state=_grid_state(); ns=_namelist(grid)
    writer.bundles={'d01':SimpleNamespace(namelist=ns,grid=grid)}
    writer.domain_authorities={'d01':writer_authority(grid,'d01')}
    _wrfinput(tmp_path/'wrfinput_d01',grid.ny,grid.nx)
    with Dataset(tmp_path/'wrfinput_d01','r+') as ds:
        ds.createVariable('TH2','f4',('Time','south_north','west_east'))[0]=np.full(state.t_skin.shape,300.125,np.float32)
    carry=_initial_carry_for_run(state,ns)
    result=writer('d01',0,carry)
    with Dataset(result['wrfout']) as ds:
        for k,v in INPUT.items():
            np.testing.assert_array_equal(ds.variables[k][0],np.full(state.t_skin.shape,v,np.float32))
        np.testing.assert_array_equal(ds.variables['TH2'][0],np.full(state.t_skin.shape,300.125,np.float32))
        np.testing.assert_array_equal(ds.variables['TE_TEMF'][0],np.full(state.theta.shape,1e-3,np.float32))


def test_noahmp_uses_supplied_temf_surface_and_preserves_water_flux(monkeypatch):
    from test_noahmp_coupler import _build,HAVE_TABLES
    from gpuwrf.physics import noahmp_coupler as nc
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    if not HAVE_TABLES:
        pytest.skip('WRF parameter tables unavailable')
    state,land,static,rad,clock=_build()
    shape=state.t_skin.shape
    a=lambda v:jnp.full(shape,v,jnp.float32)
    hfx=jnp.asarray([[11.,22.,33.,44.,55.]],jnp.float32)
    rho=a(1.2)
    flux=SurfaceFluxes(ustar=a(.3),theta_flux=hfx/(rho*1004.5),qv_flux=a(1e-5),
                       tau_u=a(-.09),tau_v=a(0.),rhosfc=rho,fltv=a(.01),xland=state.xland,wspd=None)
    diag=temf.SurfaceDiagnostics(flux,hfx,a(25.),a(2.),a(1.),a(295.),a(300.),a(.008),a(.011),
                                 a(0.),state.roughness_m,a(.02),a(.015),a(.015),land.ch,land.cm)
    def forbidden(*a,**kw):
        raise AssertionError('TEMF was replaced with a MYNN surface solve')
    monkeypatch.setattr(nc,'surface_layer_with_diagnostics',forbidden)
    result=nc.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=18.,
                                    surface_diagnostics=diag,history=True)
    state_out,land_out,_,history=result
    water=np.asarray(state.xland)>=1.5
    np.testing.assert_array_equal(np.asarray(history['hfx'])[water],np.asarray(hfx)[water])
    np.testing.assert_array_equal(history['u10'],diag.u10)
    for k in ('t2','q2','hfx'):
        assert np.isfinite(np.asarray(history[k])).all()


@pytest.mark.parametrize('kind',['pbl','surface'])
def test_kernel_has_no_typed_fp64(kind):
    compare=_compare_module(); rec=compare.load(PROOF/f'{kind}_oracle.npz')['records'][0]
    # Lists in the fixture are one array per operand, rather than pytrees.
    kw={k:({ck:jnp.asarray(cv,jnp.int32 if ck=='kpbl' else jnp.float32) for ck,cv in v.items()} if k=='carry'
           else jnp.asarray(v,jnp.float32)) for k,v in rec['inputs'].items()}
    fn=temf.temf_column if kind=='pbl' else temf.temf_surface
    hlo=jax.jit(lambda a:fn(**a)).lower(kw).compiler_ir('hlo').as_hlo_text()
    assert not re.search(r'\bf64\[',hlo)
