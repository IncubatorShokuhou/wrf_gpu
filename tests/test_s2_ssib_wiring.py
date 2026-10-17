"""SSiB CLI, caller, Registry history and all three public restart interfaces.

Tiny constructed files/states test plumbing/admission, not forecast fidelity.
The original-WRF column corpus supplies the independent numerical reference.
"""
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,pickle
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from gpuwrf.io.netcdf_lock import Dataset


def _input(path,nzs=3):
    with Dataset(path,'w') as ds:
        for name,size in (('Time',1),('soil_layers_stag',nzs),('south_north',2),('west_east',3)):
            ds.createDimension(name,size)
        ds.MMINLU='USGS';ds.SF_SURFACE_PHYSICS=8;ds.CEN_LAT=45.
        ds.createDimension('DateStrLen',19)
        ds.createVariable('Times','S1',('Time','DateStrLen'))[0]=np.frombuffer(b'2023-01-15_00:00:00',dtype='S1')
        xy=('Time','south_north','west_east')
        for name,value in {'XLAND':1.,'IVGTYP':7.,'XLAT':45.,'EMISS':.96,'SNOW':2.,'CANWAT':.1,'ALBEDO':.2,'SEAICE':0.}.items():
            ds.createVariable(name,'f4',xy)[:]=value
        for name,value in [('TSLB',280.),('SMOIS',.25)]:
            ds.createVariable(name,'f4',('Time','soil_layers_stag','south_north','west_east'))[:]=value
        ds.createVariable('ZS','f4',('Time','soil_layers_stag'))[:]=[0.,.05,1.05] if nzs==3 else np.arange(nzs)


def _case(tmp_path):
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid,build_neutral_state,_zero_tendencies
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
    from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
    from gpuwrf.io.ssib_land_init import load_ssib_land
    _input(tmp_path/'wrfinput_d01');static,land=load_ssib_land(tmp_path,'d01')
    grid=build_flat_grid(nx=3,ny=2,nz=6,dx_m=3000.)
    state=build_neutral_state(grid).replace(_cast=False,sfc_wspd=jnp.zeros((2,3),jnp.float32))
    nml=OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=_zero_tendencies(grid),dt_s=18.,
        sf_surface_physics=8,sf_sfclay_physics=1,bl_pbl_physics=1,mp_physics=8,cu_physics=0,
        ra_sw_physics=1,ra_lw_physics=4,use_noahmp=False,ssib_static=static,ssib_land=land)
    z=jnp.zeros((2,3),jnp.float32)
    carry=initial_operational_carry(state).replace(ssib_land=land,ssib_rad=(z,z,z),
        noahmp_precipitation=seed_precipitation(state),history_diagnostics=seed_history_diagnostics(state))
    return grid,state,nml,carry


def _identity(nml):
    from gpuwrf.diagnostics.solar_config import configuration as solar
    from gpuwrf.physics.rrtmg_mp_re import mp_re_config
    from gpuwrf.io.pbl_epsilon_schema import epsilon_pbl_config
    from gpuwrf.io.windfarm_schema import configuration as windfarm_config
    from gpuwrf.io.urban_schema import configuration as urban_config
    return {'namelists':{'d01':'test-config'},'ssib_active':{'d01':True},
        'online_diagnostics_config':{'d01':{'prec_acc_dt':0.,'nwp_diagnostics':0,'do_radar_ref':0,'history_interval':60.}},
        'solar_interpolation_config':{'d01':solar(nml)},'mp_re_config':{'d01':mp_re_config(nml)},
        'temf_config':{'d01':{'bl_pbl_physics':1,'sf_sfclay_physics':1,'domain_shape':[6,2,3]}},
        'epsilon_pbl_config':{'d01':{**epsilon_pbl_config(nml),'sf_sfclay_physics':1,'domain_shape':[6,2,3]}},
        'windfarm_config':{'d01':windfarm_config(nml)},
        'urban_inputs_config':{'d01':urban_config(nml)},
        'urban_bep_config':{'d01':{'sf_urban_physics':0,'domain_shape':[nml.grid.nz,nml.grid.ny,nml.grid.nx]}},
        'surface_fdda_config':{'d01':None},
        'sas_config':{'d01':{'cu_physics':int(nml.cu_physics),'osas_domain_id':int(nml.osas_domain_id),
                             'domain_shape':[nml.grid.nz,nml.grid.ny,nml.grid.nx]}}}


def _store(tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    return RestartStore(tmp_path/'native',max_bytes=10_000_000,max_generations=2,reserve_bytes=0)


@pytest.mark.parametrize('namelist',[None,SimpleNamespace(sf_surface_physics=None),
    {'sf_surface_physics':None},{'physics':{'sf_surface_physics':None}}])
def test_inferred_surface_selector_keeps_ssib_inactive_and_refuses_carry(namelist):
    from gpuwrf.io.restart_schema import ssib_active,validate_ssib
    assert not ssib_active(namelist)
    validate_ssib(None,None,namelist,None,(2,3))
    with pytest.raises(ValueError,match='SSiB.*selection mismatch'):
        validate_ssib(object(),None,namelist,1,(2,3))


def test_loader_exact_prepared_soil_and_noah_refusal(tmp_path):
    from gpuwrf.io.ssib_land_init import load_ssib_land
    _input(tmp_path/'wrfinput_d01');static,land=load_ssib_land(tmp_path,'d01')
    np.testing.assert_array_equal(land.tc,np.full((2,3),280,np.float32))
    assert land.isnow.dtype==np.int32 and land.bwo4.shape==(2,3)
    assert static.mminlu==0 and not static.has_ice
    other=tmp_path/'other';other.mkdir();_input(other/'wrfinput_d01',4)
    with pytest.raises(ValueError,match='three soil'):load_ssib_land(other,'d01')


@pytest.mark.parametrize('damage',['missing','nan','ice','vegetation','water_map','depth','map'])
def test_loader_rejects_unsupported_inputs(tmp_path,damage):
    from gpuwrf.io.ssib_land_init import load_ssib_land
    _input(tmp_path/'wrfinput_d01')
    with Dataset(tmp_path/'wrfinput_d01','a') as ds:
        if damage=='missing':ds.renameVariable('TSLB','OLD_TSLB')
        elif damage=='nan':ds['SMOIS'][0,0,0,0]=np.nan
        elif damage=='ice':ds['SEAICE'][0,0,0]=.5
        elif damage=='vegetation':ds['IVGTYP'][0,0,0]=0
        elif damage=='water_map':ds['IVGTYP'][0,0,0]=16
        elif damage=='depth':ds['ZS'][0,2]=1.5
        else:ds.MMINLU='NLCD40'
    with pytest.raises(ValueError,match='SSiB'):load_ssib_land(tmp_path,'d01')


def _raw_nml():
    return {'domains':{'max_dom':1},'physics':{'sf_surface_physics':[8],'sf_sfclay_physics':[1],
        'bl_pbl_physics':[1],'mp_physics':[8],'cu_physics':[0],'ra_sw_physics':[1],'ra_lw_physics':[4],
        'num_soil_layers':3,'fractional_seaice':1}}


@pytest.mark.parametrize('pipeline',['nested','daily'])
def test_cli_admits_exact_ssib_pair(pipeline):
    from gpuwrf.io.namelist_binding import namelist_honour_report
    errors,warnings=namelist_honour_report(_raw_nml(),('d01',),replay=pipeline=='daily')
    assert not errors,(errors,warnings)


@pytest.mark.parametrize('key,value',[('sf_sfclay_physics',5),('bl_pbl_physics',5),('ra_sw_physics',4),
    ('ra_lw_physics',1),('usemonalb',1),('num_soil_layers',4),('mp_physics',6),('fractional_seaice',0)])
def test_cli_refuses_wrong_pair_before_jax(key,value):
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nml=_raw_nml();nml['physics'][key]=value
    errors,_=namelist_honour_report(nml,('d01',))
    assert any('SSiB' in error and key in error for error in errors)


def test_namelist_aux_tree_keeps_bundles(cpu_pallas_interpret,tmp_path):
    _,_,nml,_=_case(tmp_path)
    restored=jax.tree.unflatten(jax.tree.structure(nml),jax.tree.leaves(nml))
    assert restored.ssib_static is nml.ssib_static and restored.ssib_land is nml.ssib_land
    assert jax.tree.structure(restored)==jax.tree.structure(nml)
    assert jax.tree.structure(nml)==jax.tree.structure(nml)


@pytest.mark.parametrize('format',['pickle','netcdf','native'])
def test_complete_ssib_public_restart_roundtrip(cpu_pallas_interpret,tmp_path,format):
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    grid,_,nml,carry=_case(tmp_path)
    land=carry.ssib_land._replace(bwo3=jnp.full((2,3),273.25,jnp.float32),
        flo2=jnp.full((2,3),.1,jnp.float32))
    carry=carry.replace(ssib_land=land)
    if format=='pickle':
        path=tmp_path/'restart.pkl';write_restart(carry,nml,grid,2,path);restored,_,_,_=read_restart(path)
    elif format=='netcdf':
        path=tmp_path/'wrfrst';start=datetime(2023,1,15,tzinfo=timezone.utc)
        write_wrfrst_carry(carry,grid,nml,path,valid_time=start,run_start=start,step_index=2)
        restored,_=read_wrfrst_carry(path,namelist=nml)
    else:
        store=_store(tmp_path);identity=_identity(nml)
        path,_=store.save({'d01':carry},{'d01':2},{'d01':18.},identity)
        snapshot,_=store.read(path,expected_identity=identity);restored=snapshot['carries']['d01']
    for a,b in zip(jax.tree.leaves(carry.ssib_land),jax.tree.leaves(restored.ssib_land)):
        assert np.asarray(a).dtype==np.asarray(b).dtype
        np.testing.assert_array_equal(a,b)
    for a,b in zip(carry.ssib_rad,restored.ssib_rad):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('format',['pickle','netcdf','native'])
@pytest.mark.parametrize('damage',['group','snow','shape','dtype','radiation','wind','history','precipitation'])
def test_public_restart_write_refuses_bad_ssib(cpu_pallas_interpret,tmp_path,format,damage):
    from gpuwrf.io.restart import write_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry
    grid,_,nml,carry=_case(tmp_path)
    if damage=='wind':carry=carry.replace(state=carry.state.replace(_cast=False,sfc_wspd=None))
    elif damage=='history':carry=carry.replace(history_diagnostics=None)
    elif damage=='precipitation':carry=carry.replace(noahmp_precipitation=None)
    elif damage=='group':carry=carry.replace(ssib_land=None)
    elif damage=='radiation':carry=carry.replace(ssib_rad=None)
    else:
        bad=None if damage=='snow' else np.zeros((3,2),np.float32) if damage=='shape' else np.zeros((2,3),np.float64)
        carry=carry.replace(ssib_land=carry.ssib_land._replace(bwo2=bad))
    # The common state guard precedes the SSiB guard in pickle/NetCDF.
    from gpuwrf.contracts.state import mynn_sfc_wspd_enabled
    expected = (r'restart is missing held MYNN sfc_wspd \(E78\)'
                if damage == 'wind' and format != 'native' and mynn_sfc_wspd_enabled()
                else 'SSiB')
    with pytest.raises(ValueError,match=expected):
        if format=='pickle':write_restart(carry,nml,grid,2,tmp_path/'bad.pkl')
        elif format=='netcdf':
            start=datetime(2023,1,15,tzinfo=timezone.utc)
            write_wrfrst_carry(carry,grid,nml,tmp_path/'bad.nc',valid_time=start,run_start=start,step_index=2)
        else:_store(tmp_path).save({'d01':carry},{'d01':2},{'d01':18.},_identity(nml))


@pytest.mark.parametrize('damage',['group','schema','radiation','snow','shape','dtype','wind','history','precipitation'])
def test_native_public_read_refuses_hash_valid_damage(cpu_pallas_interpret,tmp_path,damage):
    _,_,nml,carry=_case(tmp_path);store=_store(tmp_path);identity=_identity(nml)
    path,_=store.save({'d01':carry},{'d01':2},{'d01':18.},identity)
    payload_path=path/'snapshot.pkl';snapshot=pickle.loads(payload_path.read_bytes())
    payload=snapshot['carries']['d01'];extra=payload['extra_fields']
    if damage=='wind':payload['state_fields']['sfc_wspd']=None
    elif damage=='history':extra['history_diagnostics']=None
    elif damage=='precipitation':extra['noahmp_precipitation']=None
    elif damage=='group':extra['ssib_land']=None
    elif damage=='schema':payload.pop('ssib_schema_version')
    elif damage=='radiation':extra['ssib_rad']=None
    else:
        bad=None if damage=='snow' else np.zeros((3,2),np.float32) if damage=='shape' else np.zeros((2,3),np.float64)
        extra['ssib_land']=extra['ssib_land']._replace(bwo2=bad)
    blob=pickle.dumps(snapshot);payload_path.write_bytes(blob)
    manifest=json.loads((path/'manifest.json').read_text());manifest.update(payload_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='SSiB'):store.read(path,expected_identity=identity,device=False)


def test_registry_history_names_and_soil_geometry(cpu_pallas_interpret,tmp_path):
    from gpuwrf.coupling.ssib_surface_hook import ssib_history_fields
    from gpuwrf.io.wrfout_writer import _dimension_sizes,WRFOUT_VARIABLE_SPECS
    from gpuwrf.io.ssib_history import SSIB_EXTRA_HISTORY
    _,_,nml,carry=_case(tmp_path);fields=ssib_history_fields(carry.ssib_land)
    assert 'SIBBR' in fields and 'SIBLUP' in fields and 'SSIB_BR' not in fields
    assert 'DZO1' not in fields and 'TKAIR' not in fields  # r-only Registry fields
    assert WRFOUT_VARIABLE_SPECS['SIBBR'].description=='SSiB Bulk Richardson Number'
    assert WRFOUT_VARIABLE_SPECS['SIBBR'].units==''
    assert len(SSIB_EXTRA_HISTORY)==15
    assert _dimension_sizes(nx=3,ny=2,nz=6,namelist=nml)['soil_layers_stag']==3
    np.testing.assert_array_equal(fields['ZS'],np.asarray([0,.05,1.05],np.float32))
    np.testing.assert_array_equal(fields['DZS'],np.asarray([.05,1,0],np.float32))


def test_nested_loader_admits_ssib_before_binding():
    from gpuwrf.integration.nested_pipeline import _domain_sf_surface_physics
    from unittest.mock import patch
    with patch('gpuwrf.integration.nested_pipeline._domain_physics_int',return_value=8):
        assert _domain_sf_surface_physics(object(),'d01')==8


def test_actual_wrfout_registry_values(cpu_pallas_interpret,tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,bind_wrfout_domain_authority,write_prepared_wrfout
    from gpuwrf.coupling.ssib_surface_hook import ssib_history_fields
    from test_m7_netcdf_writer import authenticated_source_grid
    grid,state,nml,carry=_case(tmp_path)
    land=carry.ssib_land._replace(rib=jnp.full((2,3),.125,jnp.float32),
        xlup=jnp.full((2,3),307.,jnp.float32),www2=jnp.full((2,3),.123,jnp.float32))
    diagnostics=ssib_history_fields(land)
    auth=bind_wrfout_domain_authority('d01',authenticated_source_grid(grid,'d01'),grid)
    clock=datetime(2023,1,15,tzinfo=timezone.utc)
    prepared=prepare_wrfout_payload(state,grid,nml,tmp_path/'wrfout',domain='d01',domain_authority=auth,
        valid_time=clock,run_start=clock,lead_hours=1.,diagnostics=diagnostics,full_variable_set=True)
    write_prepared_wrfout(prepared,expected_domain='d01',expected_domain_authority=auth)
    with Dataset(tmp_path/'wrfout') as ds:
        assert len(ds.dimensions['soil_layers_stag'])==3
        for name,value in diagnostics.items():
            assert ds[name].dtype==np.dtype('float32')
            np.testing.assert_array_equal(ds[name][0],np.asarray(value,np.float32))
        assert ds['SIBBR'].units==''
        assert 'DZO1' not in ds.variables and 'SFCEVP' not in ds.variables


def test_ysu_consumes_actual_ssib_fluxes_without_second_surface_solve(cpu_pallas_interpret,tmp_path,monkeypatch):
    from gpuwrf.coupling.ssib_surface_hook import ssib_ysu_step
    import gpuwrf.physics.pbl_ysu as ysu
    import gpuwrf.coupling.scan_adapters as adapters
    grid,state,nml,carry=_case(tmp_path)
    f=lambda value:jnp.full((2,3),value,jnp.float32)
    state=state.replace(hfx=f(37.),qfx=f(2e-5),ustar=f(.31),sfc_wspd=f(6.))
    land=carry.ssib_land._replace(rib=f(-.12))
    p3=lambda value:jnp.full((6,2,3),value,jnp.float32)
    physics=dict(u=p3(4.),v=p3(3.),t_air=p3(279.),qv=p3(.004),p=p3(88000.),
        p8w=jnp.full((7,2,3),89000.,jnp.float32),exner=p3(.96),dz=p3(100.),psfc=f(91000.))
    captured={}
    def solve(*args,**kwargs):
        captured.update(kwargs);captured['profile_args']=args
        zero=jnp.zeros((6,6),jnp.float32)
        return dict(u=zero,v=zero,theta=zero,qv=zero,pblh=jnp.full((6,),432.,jnp.float32))
    monkeypatch.setattr(ysu,'ysu_columns',solve)
    monkeypatch.setattr(adapters,'surface_layer_with_diagnostics',lambda *a,**k:pytest.fail('second surface solve'))
    out=ssib_ysu_step(state,land,nml.ssib_static,grid,18.,physics=physics,
        surface_diagnostics=dict(br=f(.9),fm=f(2.1),fh=f(2.2)),history=dict(u10=f(3.4),v10=f(2.1)))
    for key,value in dict(hfx=37.,qfx=2e-5,ust=.31,br=-.12,psfc=91000.,psim=2.1,psih=2.2).items():
        np.testing.assert_array_equal(captured[key],np.full((6,),value,np.float32))
    np.testing.assert_array_equal(out.pblh,np.full((2,3),432.,out.pblh.dtype))
    assert out.theta.dtype==state.theta.dtype


def test_public_initial_carry_seeds_held_wind(cpu_pallas_interpret,tmp_path):
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    _,state,nml,_=_case(tmp_path)
    state=state.replace(_cast=False,sfc_wspd=None)
    seeded=_initial_carry_for_run(state,nml)
    assert seeded.state.sfc_wspd.shape==(2,3)
    assert seeded.state.sfc_wspd.dtype==jnp.float32
    # Native REAL carry reconstruction may wrap the same land values anew.
    assert type(seeded.ssib_land) is type(nml.ssib_land)
    assert seeded.ssib_land._fields == nml.ssib_land._fields
    for name in nml.ssib_land._fields:
        actual, expected = np.asarray(getattr(seeded.ssib_land, name)), np.asarray(getattr(nml.ssib_land, name))
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes(), name
    assert seeded.noahmp_precipitation is not None
    np.testing.assert_array_equal(seeded.state.roughness_m,nml.ssib_static.initial_znt)
    np.testing.assert_array_equal(seeded.state.mavail,nml.ssib_static.initial_mavail)
    np.testing.assert_array_equal(seeded.state.ustar,np.full((2,3),.0001,np.float32))
    assert seeded.radiation_diagnostics.cloud_fraction.shape==state.qv.shape


def test_wrf_fatal_branch_is_observable_nonfinite():
    import inspect
    from gpuwrf.physics._ssib_generated import _convdim
    arrays={'dzo','wo','tssn','tssno','bwo','bto','cto','fio','flo','bio','blo','ho'}
    args=[jnp.int32(7) if name=='ioflag' else jnp.zeros((4,),jnp.float32) if name in arrays
          else jnp.float32(0) for name in inspect.signature(_convdim).parameters]
    out=jax.jit(_convdim)(*args)
    assert all(np.isnan(np.asarray(value)).all() for value in out[1:])


@pytest.mark.parametrize("pipeline", ["daily", "nested"])
def test_both_pipeline_loaders_bind_ssib_bundles(cpu_pallas_interpret,monkeypatch, tmp_path, pipeline):
    from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state, _zero_tendencies
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_m7_netcdf_writer import authenticated_source_grid
    import gpuwrf.io.lower_boundary as lower
    _input(tmp_path / "wrfinput_d01")
    (tmp_path / "namelist.input").write_text(
        "&domains\n max_dom=1, time_step=18,\n/\n&physics\n sf_surface_physics=8, "
        "num_soil_layers=3, sf_sfclay_physics=1, mp_physics=8, bl_pbl_physics=1, ra_sw_physics=1, ra_lw_physics=4, fractional_seaice=1, cu_physics=0,\n/\n")
    run = Gen2Run(tmp_path)
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    source = authenticated_source_grid(grid, "d01")
    run.grid = lambda domain: source
    state = build_neutral_state(grid)
    case = SimpleNamespace(run=run, grid=grid, metrics=grid.metrics, state=state,
                           tendencies=_zero_tendencies(grid),
                           metadata={"run_start_label": "2023-01-15_00:00:00", "standalone_native_init": True})
    mod = daily if pipeline == "daily" else nested
    monkeypatch.setattr(mod, "build_replay_case", lambda *a, **kw: case)
    monkeypatch.setattr(mod, "load_radiation_static", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(lower, "load_lower_boundary", lambda *a, **kw: None)
    if pipeline == "daily":
        monkeypatch.setattr(daily, "dealias_state_buffers", lambda s: s)
        monkeypatch.setattr(daily, "_load_static_latlon_writer_diagnostics", lambda *a, **kw: (None, {}))
        loaded, _ = daily._build_real_case(daily.DailyPipelineConfig(run_id=str(tmp_path), domain="d01"))
        nml = loaded.namelist
    else:
        monkeypatch.setattr(nested, "_root_boundary_cadence_override", lambda n, *a: n)
        monkeypatch.setattr(nested, "_commit_to_operational_device", lambda c: c)
        monkeypatch.setattr(nested, "_initial_carry_for_run", lambda s, n: initial_operational_carry(s).replace(ssib_land=n.ssib_land))
        cfg = nested.NestedPipelineConfig(tmp_path, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=1)
        _, bundles, _, _, _, carries = nested._load_domains(cfg, ("d01",))
        nml = bundles["d01"].namelist
        assert carries["d01"].ssib_land is not None
    assert nml.sf_surface_physics == 8 and not nml.use_noahmp
    assert nml.ssib_land.tc.shape == (2,3)


@pytest.mark.parametrize('start,lead,expected',[
    ('2023-01-15T18:00:00',0.,15.),('2023-01-15T18:00:00',21600.,16.),
    ('2023-01-15T23:50:00',0.,16.),('2023-12-31T00:00:00',86400.,1.)])
def test_surface_driver_calendar_day(start,lead,expected):
    from gpuwrf.coupling.ssib_surface_hook import ssib_day
    result=ssib_day(datetime.fromisoformat(start).replace(tzinfo=timezone.utc),lead)
    assert float(result)==expected and result.dtype==jnp.float32


def test_tracked_assets_survive_clean_git_archive(tmp_path):
    import io,tarfile,subprocess
    root=Path(__file__).resolve().parents[1]
    raw=subprocess.check_output(['git','archive','HEAD','proofs/s2_ssib','src/gpuwrf/physics/_ssib_generated.py','src/gpuwrf/io/ssib_landuse.py'],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(tmp_path,filter='data')
    for relative in ('native_reference.npz','native_reference.json','fma_reference_spread.json',
                     'rib_input_bound.json','UPSTREAM_WRF_BUGS.md','translate.py','landuse_reference.npz','landuse_reference.json'):
        assert (tmp_path/'proofs/s2_ssib'/relative).read_bytes()==(root/'proofs/s2_ssib'/relative).read_bytes()
    source='src/gpuwrf/physics/_ssib_generated.py'
    assert (tmp_path/source).read_bytes()==(root/source).read_bytes()
    source='src/gpuwrf/io/ssib_landuse.py'
    assert (tmp_path/source).read_bytes()==(root/source).read_bytes()
    manifest=json.loads((tmp_path/'proofs/s2_ssib/native_reference.json').read_text())
    assert hashlib.sha256((tmp_path/'proofs/s2_ssib/native_reference.npz').read_bytes()).hexdigest()==manifest['fixture_sha256']


def test_landuse_startup_byte_vs_unmodified_wrf():
    from gpuwrf.io.ssib_landuse import ssib_landuse_initial,LANDUSE_SHA256
    root=Path(__file__).resolve().parents[1]/'proofs/s2_ssib'
    manifest=json.loads((root/'landuse_reference.json').read_text())
    assert manifest['passed'] and manifest['checks']==2136 and manifest['table_sha256']==LANDUSE_SHA256
    assert hashlib.sha256((root/'landuse_reference.npz').read_bytes()).hexdigest()==manifest['fixture_sha256']
    fields=('albedo','albbck','emiss','mavail','znt','xland')
    with np.load(root/'landuse_reference.npz',allow_pickle=False) as data:
        assert len(data['rows'])==356
        for row in data['rows']:
            map,scenario,cat=(int(x) for x in row[:3])
            actual=ssib_landuse_initial(dataset=('USGS','SSIB','MODIS','MODIFIED_IGBP_MODIS_NOAH')[map-1],
                category=np.int32(cat),snowc=row[-1],day=15 if scenario%2 else 172,cen_lat=45 if scenario<3 else -45)
            for index,field in enumerate(fields):
                assert np.asarray(actual[field]).dtype==np.float32
                assert np.asarray(actual[field]).tobytes()==row[index+3].tobytes(),(map,scenario,cat,field)


def test_real_exe_input_needs_no_diagnostic_albedo_emiss(tmp_path):
    from gpuwrf.io.ssib_land_init import load_ssib_land
    _input(tmp_path/'wrfinput_d01')
    with Dataset(tmp_path/'wrfinput_d01','a') as ds:
        ds.renameVariable('EMISS','OLD_EMISS');ds.renameVariable('ALBEDO','OLD_ALBEDO')
    static,land=load_ssib_land(tmp_path,'d01')
    with np.load(Path(__file__).resolve().parents[1]/'proofs/s2_ssib/landuse_reference.npz') as data:
        row=next(row for row in data['rows'] if tuple(row[:3])==(1,1,7))
        np.testing.assert_array_equal(static.emiss,np.full((2,3),row[5],np.float32))
    assert static.initial_znt.shape==(2,3) and static.initial_mavail.shape==(2,3)
    assert np.isfinite(land.bedo).all()


@pytest.mark.parametrize('interval,cadence,midpoint',[(1801.2,100,1080.6),(0.,1,180.)])
def test_dudhia_midpoint_coszen_and_current_solcon(cpu_pallas_interpret,tmp_path,monkeypatch,interval,cadence,midpoint):
    from dataclasses import replace
    from gpuwrf.coupling.ssib_surface_hook import ssib_dudhia_columns
    import gpuwrf.coupling.physics_couplers as coupling
    import gpuwrf.physics.ra_sw_dudhia as dudhia
    _,state,nml,carry=_case(tmp_path);captured={}
    nml=replace(nml,radiation_interval_s=interval,radiation_cadence_steps=cadence,
        time_utc=datetime(2023,1,15,12,tzinfo=timezone.utc),
        radiation_static=SimpleNamespace(xlat_deg=np.full((2,3),45.,np.float32),xlong_deg=np.zeros((2,3),np.float32)))
    class Column:
        albedo=jnp.ones((6,),jnp.float32)
        def _replace(self,**values):captured.update(values);return self
    monkeypatch.setattr(coupling,'_dudhia_sw_column_inputs',lambda *a,**k:(Column(),None))
    def geometry(lat,lon,julian,minute,current,solar=None,*,solar_xtime_minutes=None):
        captured.update(current=current,midpoint=float(solar_xtime_minutes)*60,minutes=solar_xtime_minutes)
        return SimpleNamespace(coszen=jnp.full((2,3),.6,jnp.float32))
    monkeypatch.setattr(coupling,'_wrf_solar_geometry_real',geometry)
    def solcon(julian,minute,current):captured['solcon_time']=current;return jnp.float32(1290.)
    monkeypatch.setattr(coupling,'_wrf_solcon_real',solcon)
    monkeypatch.setattr(dudhia,'solve_dudhia_sw_column',lambda column:column)
    physics={name:jnp.ones((6,2,3),jnp.float32) for name in ('t_air','p','dz')}
    ssib_dudhia_columns(state,carry.ssib_land,nml.ssib_static,nml,180.,None,physics)
    assert captured['midpoint']==pytest.approx(midpoint)
    expected=np.float32(18.01 if interval else 3.)
    assert np.asarray(captured['minutes']).tobytes()==expected.tobytes()
    assert captured['current']==captured['solcon_time']==180.
    np.testing.assert_array_equal(captured['coszen'],np.full((6,),.6,np.float32))
    np.testing.assert_array_equal(captured['solcon'],np.full((6,),1290.,np.float32))
