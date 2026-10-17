"""Independent WRF fixture, caller ABI and cache/restart failure gates for mp11."""
from pathlib import Path
from types import SimpleNamespace
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'proofs/s2_cammg'))
from compiler_math import compiler_math
from gpuwrf.physics.cammg import column,tables
from gpuwrf.physics.cammg.registry import STATE_FIELDS,INTERFACE_FIELDS,SURFACE_FIELDS,HISTORY_NAMES


def test_fresh_state_import_does_not_load_physics_kernels():
    import os,subprocess
    env={**os.environ,'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0','PYTHONPATH':str(ROOT/'src')}
    result=subprocess.run([sys.executable,'-c',
        'import sys; from gpuwrf.contracts.state import State; '
        'from gpuwrf.io.namelist_binding import PHYSICS_KNOBS; '
        'assert "gpuwrf.physics.thompson_column" not in sys.modules'],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def original_call(case=2,step=1):
    prefix=f'c{case:02d}_s{step}'
    with np.load(ROOT/'proofs/s2_cammg/fixtures/convection.npz') as z:
        inputs={k.split('_input_',1)[1]:z[k] for k in z.files if k.startswith(prefix+'_input_')}
        expected={k.split('_output_',1)[1]:z[k] for k in z.files if k.startswith(prefix+'_output_')}
    return inputs,expected


@pytest.mark.parametrize('case,it,cu,shcu',[(2,1,0,0),(19,2,7,2)])
def test_original_public_byte(case,it,cu,shcu):
    inputs,expected=original_call(case,it)
    with np.errstate(all='ignore'):
        got,fatal=column.step(inputs,18.,itimestep=it,cu=cu,shcu=shcu,
                             globals=tables.get_globals(24),xp=compiler_math)
    assert fatal==0
    assert got.keys()==expected.keys()
    for name in got:
        a=np.asarray(got[name]).reshape(-1);b=expected[name]
        assert np.isfinite(a).all() and np.isfinite(b).all(),name
        assert a.dtype==b.dtype and a.shape==b.shape,name
        assert a.tobytes()==b.tobytes(),f'original public BYTE {name}'


def test_registry_and_conditional_state(cpu_pallas_interpret):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State
    from gpuwrf.contracts.physics_registry import moist_species_for_mp,number_species_for_mp,wrfout_names_for_mp
    from gpuwrf.contracts.physics_interfaces import scheme_step_spec
    grid=GridSpec.canary_3km_template()
    state=State.zeros(grid,mp_physics=11)
    assert moist_species_for_mp(11)==('qv','qc','qr','qi','qs')
    assert number_species_for_mp(11)==('Nc','Ni','Nr','Ns')
    assert all(getattr(State.zeros(grid),name) is None for name in STATE_FIELDS)
    for name in STATE_FIELDS:
        a=getattr(state,name)
        assert a.dtype==(np.int32 if name=='cam_fatal' else np.float32)
        assert a.shape==((grid.nz+1,grid.ny,grid.nx) if name in INTERFACE_FIELDS else
                         (grid.ny,grid.nx) if name in SURFACE_FIELDS else (grid.nz,grid.ny,grid.nx))
    assert 'QNDROP' not in wrfout_names_for_mp(11)
    assert set(HISTORY_NAMES.values()) <= set(wrfout_names_for_mp(11))
    assert scheme_step_spec('microphysics',11).wrf_slot=='solve_em_microphysics_driver'
    rebuilt=jax.tree.unflatten(jax.tree.structure(state),jax.tree.leaves(state))
    assert rebuilt.active_field_names()==state.active_field_names()


def test_restart_fail_closed():
    from gpuwrf.physics.cammg.restart import validate
    shape=(3,2,2);fields={'qv':np.zeros(shape,np.float32)}
    for name in (*STATE_FIELDS,'Nc','Ni','Nr','Ns'):
        fields[name]=np.zeros((4,2,2) if name in INTERFACE_FIELDS else (2,2) if name in SURFACE_FIELDS else shape,
                              np.int32 if name=='cam_fatal' else np.float32)
    validate(fields,{'mp_physics':11},1)
    for name in STATE_FIELDS:
        bad={**fields,name:None}
        with pytest.raises(ValueError,match='restart'):validate(bad,{'mp_physics':11},1)
    for value in (np.nan,np.inf,-np.inf):
        bad={**fields,'rh_old_mp':np.full(shape,value,np.float32)}
        with pytest.raises(ValueError,match='nonfinite'):validate(bad,{'mp_physics':11},1)
    with pytest.raises(ValueError,match='schema'):validate(fields,{'mp_physics':11},None)
    with pytest.raises(ValueError,match='selection'):validate(fields,{'mp_physics':8},1)


def test_original_fatal_cannot_be_disabled():
    from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
    state=SimpleNamespace(cam_fatal=np.ones((2,2),np.int32))
    with pytest.raises(RuntimeError,match='CAM-MG fatal'):
        assert_state_finite_at_boundary(state,domain='d01',step=2,enabled=False)


def test_warn_zero_is_not_a_pairing_rule():
    from gpuwrf.physics.cammg.registry import warn_configuration
    with pytest.warns(RuntimeWarning) as record:warn_configuration(1,0)
    assert len(record)==2
    assert not any('requires' in str(w.message) for w in record)
    warn_configuration(7,2)
    from gpuwrf.io.namelist_binding import bind_namelist_options
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    nl=OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.)
    config={'physics':{'mp_physics':11,'bl_pbl_physics':5,'ra_lw_physics':4,'ra_sw_physics':4,
                       'cu_physics':0,'accum_mode':7.e8,'aitken_mode':2.e8,'coarse_mode':1.e5}}
    bound=bind_namelist_options(nl,config,'d01')
    assert (bound.mp_physics,bound.bl_pbl_physics,bound.ra_sw_physics)==(11,5,4)
    assert (bound.accum_mode,bound.aitken_mode,bound.coarse_mode)==(7.e8,2.e8,1.e5)
    rebuilt=jax.tree.unflatten(jax.tree.structure(bound),jax.tree.leaves(bound))
    assert rebuilt.accum_mode==bound.accum_mode


def test_actual_baked_namespace_cannot_load_old_blob(tmp_path,monkeypatch):
    from gpuwrf.runtime import aot_cheap_key as ck,aot_executable as ax,aot_precompile as ap
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    nl=OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=11,
                          ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)
    @jax.jit
    def fn(x,nl):
        with jax.enable_x64(True):
            return x*jnp.asarray(tables._raw()['ndrop.alogsig'][0],jnp.float64)
    x=jnp.asarray([1.],jnp.float32)
    before=ck.cheap_key(fn,(x,nl),{},nl);lowered=fn.lower(x,nl);value=np.asarray(fn(x,nl))
    assert before is not None
    status=ap._serialize_domain_blob('cammg',lowered.compile(),tmp_path,lowered=lowered,
                                     cheap_key=before,key_schema=ck.KEY_SCHEMA)
    assert status['aot_written'],status
    changed={k:v.copy() for k,v in tables._raw().items()}
    changed['ndrop.alogsig'][0]*=.9
    monkeypatch.setattr(tables,'_raw',lambda:changed);fn.clear_cache()
    after=ck.cheap_key(fn,(x,nl),{},nl)
    assert after is not None and after!=before
    loaded,status=ap.load_domain_blob('cammg',tmp_path,cheap_key=after,return_status=True)
    assert loaded is None
    assert ax.hlo_sha256_from_lowered(fn.lower(x,nl))!=ax.hlo_sha256_from_lowered(lowered)
    assert np.all(np.asarray(fn(x,nl))!=value)
    fn.clear_cache()


def test_native_scope_and_actual_operand_table_kinds():
    from gpuwrf.kernels.phys_cammg import full_column,SURFACE_FIELDS as SURFACE
    raw,_=original_call()
    fields={k:jnp.asarray(v).reshape((1,) if k in SURFACE else (1,25)) for k,v in raw.items()}
    with jax.enable_x64(False):
        program=jax.make_jaxpr(lambda f,dt,it:full_column(f,dt,it))(fields,jnp.float32(18),jnp.int32(1)).jaxpr
        assert not jax.config.jax_enable_x64
    primitive=next(e for e in program.eqns if e.primitive.name=='pallas_call')
    assert any(getattr(v.aval,'dtype',None)==np.float64 for v in primitive.invars)
    assert all(v.aval.dtype==np.float32 for v in program.outvars[:-1])
    assert program.outvars[-1].aval.dtype==np.int32


def test_caller_operand_inventory_and_precipitation(cpu_pallas_interpret,monkeypatch):
    import dataclasses,importlib.util
    from gpuwrf.physics.cammg.adapter import cammg_adapter,operands
    from gpuwrf.kernels import phys_cammg
    spec=importlib.util.spec_from_file_location('cam_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=11)
    fields,th=operands(state,grid)
    expected,_=original_call()
    assert fields.keys()==expected.keys()
    assert all(v.dtype==np.float32 for v in fields.values())
    assert fields['p8w'].shape==(12,9)
    called=[]
    def kernel(fields,dt,it,**kw):
        called.append((it,kw))
        out={k:v for k,v in fields.items() if k in (*phys_cammg.MODIFIED_PROFILES,*phys_cammg.MODIFIED_SURFACES)}
        out.update(rainncv=jnp.full((12,),3.,jnp.float32),snowncv=jnp.ones((12,),jnp.float32))
        return out,jnp.zeros((12,),jnp.int32)
    monkeypatch.setattr(phys_cammg,'full_column',kernel)
    result,precip=cammg_adapter(state,18.,grid,timestep=7,return_precipitation=True)
    assert called[0][0]==7
    np.testing.assert_array_equal(result.rain_acc-state.rain_acc,2)
    np.testing.assert_array_equal(result.snow_acc-state.snow_acc,1)
    # + the scheme's own SR (CAMMGMPSCHEME SR=SR, microphysics_driver.F:3004) for Noah-classic FFROZP (t-calleraudit F24)
    assert set(precip)=={'rain','snow','graupel','ice','sr'}


def test_both_restart_formats(cpu_pallas_interpret,tmp_path):
    import dataclasses,importlib.util
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state,read_wrfrst_state
    spec=importlib.util.spec_from_file_location('cam_restart_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4);state=f._state(grid).ensure_conditional_leaves(mp_physics=11)
    nl=dataclasses.replace(f._namelist(grid),mp_physics=11,cu_physics=0)
    write_checkpoint(state,nl,grid,2,tmp_path/'checkpoint.pkl')
    a=read_checkpoint(tmp_path/'checkpoint.pkl')[0]
    write_wrfrst_state(state,grid,nl,tmp_path/'restart.nc',valid_time='2026-07-25_18:00:36',
                       run_start='2026-07-25_18:00:00',step_index=2)
    b,_=read_wrfrst_state(tmp_path/'restart.nc')
    for name in STATE_FIELDS:
        assert np.asarray(getattr(a,name)).tobytes()==np.asarray(getattr(state,name)).tobytes()
        assert np.asarray(getattr(b,name)).tobytes()==np.asarray(getattr(state,name)).tobytes()


@pytest.mark.parametrize('mode',('full','operational','subset'))
def test_emitted_history_matches_registry(mode,tmp_path):
    import importlib.util,re
    from datetime import datetime,timezone
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.config.paths import wrf_root
    spec=importlib.util.spec_from_file_location('cam_writer_fixture',ROOT/'tests/test_m7_netcdf_writer.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    state,grid,nl=f.synthetic_case();nl['mp_physics']=11
    expected={leaf:np.full(state.qv.shape,(i+1)/128,np.float32) for i,leaf in enumerate(HISTORY_NAMES)}
    for leaf,value in expected.items():setattr(state,leaf,value)
    for name in ('cam_rainnc','cam_snownc','cam_sr'):
        setattr(state,name,np.ones(state.rain_acc.shape,np.float32))
    start=datetime(2023,1,15,tzinfo=timezone.utc);authority=f.writer_authority(grid,'d01')
    omitted={'QNDROP','QGRAUP','QNGRAUPEL','QNCCN','QHAIL','QNHAIL'}
    subset=tuple((*HISTORY_NAMES.values(),*omitted,'RAINNC','SNOWNC','SR')) if mode=='subset' else None
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'history.nc',domain='d01',domain_authority=authority,
        valid_time=start,run_start=start,lead_hours=0,full_variable_set=mode!='operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    original=(wrf_root()/'Registry/registry.cam').read_text()
    with Dataset(payload.target) as ds:
        assert not omitted&set(ds.variables)
        for leaf,name in HISTORY_NAMES.items():
            var=ds[name];np.testing.assert_array_equal(var[0],expected[leaf])
            assert var.dtype==np.dtype('float32') and var.dimensions==('Time','bottom_top','south_north','west_east')
            metadata=re.search(r'"'+name+r'"\s+"([^"]*)"\s+"([^"]*)"',original)
            assert metadata is not None,name
            assert var.description==metadata[1] and var.units==metadata[2],name
        for name in ('RAINNC','SNOWNC','SR'):np.testing.assert_array_equal(ds[name][0],1)


@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_nonfinite_real_history_refused(bad,tmp_path):
    import importlib.util
    from datetime import datetime,timezone
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    spec=importlib.util.spec_from_file_location('cam_bad_writer_fixture',ROOT/'tests/test_m7_netcdf_writer.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    state,grid,nl=f.synthetic_case();nl['mp_physics']=11
    state.cldfra_mp_all=np.full(state.qv.shape,bad,np.float32)
    start=datetime(2023,1,15,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='nonfinite microphysics history field CLDFRA_MP_ALL'):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'bad.nc',domain='d01',domain_authority=f.writer_authority(grid,'d01'),
            valid_time=start,run_start=start,lead_hours=0,full_variable_set=True,variable_subset=('CLDFRA_MP_ALL',))


def test_camzm_original_icwmrdp_handoff():
    import json
    sys.path.insert(0,str(ROOT/"proofs/v040/cuplus"))
    from literal_camzm import executor
    from gpuwrf.physics.bl_camuw import ESTBL
    literal=executor(ROOT)
    manifest=json.loads((ROOT/'proofs/s2_cammg/fixtures/camzm_handoff.json').read_text())
    assert sum(row['active'] for row in manifest['rows'])>0
    names=('T_PHY','QV','P','P8W','PI_PHY','U_PHY','V_PHY','Z','Z_AT_W','TKE_PBL','TH','QC','QI','CLDFRA','DZ8W')
    with np.load(ROOT/'proofs/s2_cammg/fixtures/camzm_handoff.npz') as z:
        for row in manifest['rows']:
            prefix=f"row{row['index']}";n=row['nz'];width=1<<(n+2).bit_length()
            fields=tuple(np.pad(z[prefix+'_'+name],(1,width-len(z[prefix+'_'+name])-1)) for name in names)
            with np.errstate(all='ignore'):
                out=literal(fields,z[prefix+'_controls'],np.asarray(ESTBL+(-99999.,)*6,np.float64),kx=n,pbl=row['pbl'])
            got=np.asarray(out['ICWMRDP'])[1:n+1];truth=z[prefix+'_ICWMRDP']
            assert got.dtype==truth.dtype and np.isfinite(got).all()
            assert got.tobytes()==truth.tobytes(),row


@pytest.mark.parametrize('bl',[5,1])
def test_selected_pbl_exch_h_reaches_cammg_original_caller(bl,cpu_pallas_interpret,monkeypatch):
    """WRF module_microphysics_driver:3030 passes EXCH_H without a shift.

    Inject the independent original public caller's nonzero EXCH_H at each
    selected PBL's existing output ABI. Exercise the actual PBL reassembly,
    State carry and CAM operand seam, then the original-qualified column body.
    This tests wiring; it does not claim PBL-vs-WRF numerical qualification.
    """
    import importlib.util
    from gpuwrf.physics.cammg.adapter import operands
    from gpuwrf.coupling import physics_couplers as pc,scan_adapters as scan
    spec=importlib.util.spec_from_file_location('cam_pbl_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fields,expected=original_call(14,2)
    nz=fields['kvh3d'].size-1;grid=fixture._grid(nz=nz,ny=1,nx=1)
    state=fixture._state(grid).ensure_conditional_leaves(mp_physics=11)
    # WRF's producer writes kts:kte; unassigned top memory retains its value.
    held=np.zeros((nz+1,1,1),np.float32);held[-1,0,0]=fields['kvh3d'][-1]
    tke=np.asarray(fields['tke_pbl'],np.float32).reshape(nz+1,1,1)
    state=state.replace(cam_kvh3d=jnp.asarray(held),cam_tke_pbl=jnp.asarray(tke))
    coefficient=jnp.asarray(fields['kvh3d'][:-1]).reshape(1,1,nz)
    if bl==5:
        producer=pc._mynn_column_from_state(state,grid).replace(kh=coefficient)
        updated=pc._state_from_mynn_output(state,producer)
    else:
        def output(*args,**kwargs):
            z=jnp.zeros((1,nz),np.float64)
            return dict(u=z,v=z,theta=z,qv=z,exch_h=coefficient.reshape(1,nz))
        monkeypatch.setattr(scan,'ysu_columns',output)
        updated=scan.ysu_pbl_adapter(state,18.,grid)
    actual,_=operands(updated,grid)
    # The independent original body/enclosure rejects the old dropped-field
    # route (Nc/QNDROP/WSEDL), rather than merely asserting two port arrays agree.
    with np.errstate(all='ignore'):
        got,fatal=column.step({**fields,'kvh3d':np.asarray(actual['kvh3d']).reshape(-1)},18.,
            itimestep=2,cu=0,shcu=0,globals=tables.get_globals(nz),xp=compiler_math)
    assert fatal==0
    for name in ('nc3d','qndrop','wsedl3d'):
        assert np.asarray(got[name]).reshape(-1).tobytes()==expected[name].tobytes(),f'BL{bl} original caller {name}'
    np.testing.assert_array_equal(actual['kvh3d'].reshape(-1),fields['kvh3d'])
    # Neither WRF MYNN nor YSU is passed TKE_PBL; do not invent QKE/2 here.
    np.testing.assert_array_equal(updated.cam_tke_pbl,tke)


@pytest.mark.parametrize('bl,sf',[(5,0),(1,0),(1,1)])
def test_pbl_to_post_rk_time_seam_keeps_actual_exch_h_and_held_tke(bl,sf,cpu_pallas_interpret,monkeypatch):
    import dataclasses,importlib.util
    from gpuwrf.coupling import physics_couplers as pc,scan_adapters as scan
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    spec=importlib.util.spec_from_file_location('cam_pbl_time',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4);state=f._state(grid).ensure_conditional_leaves(mp_physics=11)
    # Distinct values at every k and column expose shifts, transposes, staleness
    # and the old generic QKE/2 substitution. Expected routing is from WRF's
    # independent pbl_driver call list, not a second port computation.
    coefficient=jnp.arange(state.qv.size,dtype=jnp.float32).reshape(state.qv.shape)+3
    seed=jnp.full_like(state.cam_tke_pbl,73.)
    top=jnp.full_like(state.cam_kvh3d[-1:],91.)
    state=state.replace(cam_tke_pbl=seed,cam_kvh3d=jnp.concatenate((jnp.zeros_like(coefficient),top)))
    def cols(v):return jnp.moveaxis(v,0,-1).reshape(-1,8)
    if bl==5:
        def producer(c,*args,**kwargs):
            return c.replace(kh=cols(coefficient)),jnp.zeros((12,),jnp.float32)
        monkeypatch.setattr(pc,'step_mynn_pbl_column_with_pblh',producer)
    else:
        def producer(*args,**kwargs):
            z=jnp.zeros((12,8),jnp.float64)
            return dict(u=z,v=z,theta=z,qv=z,exch_h=cols(coefficient))
        monkeypatch.setattr(scan,'ysu_columns',producer)
    monkeypatch.setattr(op,'surface_adapter',lambda s,*a,**kw:s)
    if sf==1:
        from gpuwrf.coupling import surface_handoff
        def paid_surface(s,*args,**kwargs):
            z=jnp.zeros_like(s.t_skin);one=jnp.ones_like(z)
            return s,dict(fm=one,fh=one,psim=z,psih=z,br=z,wspd=5*one,
                u10=2*one,v10=3*one,gz1oz0=4*one,rmol=z,tsk=s.t_skin),None
        monkeypatch.setattr(surface_handoff,'revised_mm5_surface_step',paid_surface)
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_MICROPHYSICS_WRF_ORDER','1')
    nl=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        mp_physics=11,bl_pbl_physics=bl,cu_physics=0,sf_surface_physics=0,
        sf_sfclay_physics=sf,use_noahmp=False,run_physics=True,rad_rk_tendf=1)
    forcing=op._physics_step_forcing(initial_operational_carry(state),nl,18.,
                                      run_radiation=False,first_timestep=False)
    # Exercise the actual post-RK restore seam too. Merely reaching the
    # physics output does not prove the current PBL call reaches microphysics.
    post=op._apply_physics_non_dry_updates(state,state,forcing.state)
    np.testing.assert_array_equal(post.cam_kvh3d[:-1],coefficient)
    np.testing.assert_array_equal(post.cam_kvh3d[-1:],top)
    np.testing.assert_array_equal(post.cam_tke_pbl,seed)


@pytest.mark.parametrize('tke_name',['TKE_PBL','tke_pbl'])
def test_cammg_cold_shared_pbl_fields_and_nonfinite_refusal(tke_name,cpu_pallas_interpret,tmp_path):
    import importlib.util
    from gpuwrf.physics.cammg.binding import bind_state
    from gpuwrf.io.netcdf_lock import Dataset
    spec=importlib.util.spec_from_file_location('cam_pbl_cold',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4);state=f._state(grid)
    path=tmp_path/'input.nc'
    with Dataset(path,'w') as ds:
        for key,n in [('Time',1),('bottom_top_stag',9),('south_north',3),('west_east',4)]:ds.createDimension(key,n)
        for name,value in [('EXCH_H',5.),(tke_name,7.)]:
            ds.createVariable(name,'f4',('Time','bottom_top_stag','south_north','west_east'))[:]=value
    bound=bind_state(state,path)
    np.testing.assert_array_equal(bound.cam_kvh3d,5.)
    np.testing.assert_array_equal(bound.cam_tke_pbl,7.)
    with Dataset(path,'a') as ds:ds['EXCH_H'][0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='EXCH_H.*nonfinite'):bind_state(state,path)


@pytest.mark.parametrize('bl,kernel,heat,tke',[
    (1,'ysu_columns','exch_h',False),(7,'acm2_columns','exch_h',False),
    (8,'boulac_columns','exch_h',True),(11,'shinhong_columns','exch_h',False),
    (12,'gbm_columns','kth',True)])
def test_scan_pbl_retains_shared_outputs_once(bl,kernel,heat,tke,cpu_pallas_interpret,monkeypatch):
    import importlib.util
    from gpuwrf.coupling import scan_adapters as scan
    spec=importlib.util.spec_from_file_location('cam_pbl_scan',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4);state=f._state(grid).ensure_conditional_leaves(mp_physics=11)
    state=state.replace(cam_tke_pbl=jnp.full_like(state.cam_tke_pbl,73.),
        cam_kvh3d=jnp.full_like(state.cam_kvh3d,91.))
    coefficient=jnp.arange(96,dtype=jnp.float32).reshape(12,8)+11
    energy=coefficient+100;calls=[]
    def producer(*args,**kwargs):
        calls.append(True);z=jnp.zeros((12,8),jnp.float64)
        out=dict(u=z,v=z,theta=z,qv=z,qc=z,qi=z,rublten=z,rvblten=z,rthblten=z,rqvblten=z,
            tke=energy,ust=jnp.ones((12,),jnp.float32),**{heat:coefficient})
        if bl==12:
            out['kethl']=jnp.zeros_like(coefficient)
        return out
    monkeypatch.setattr(scan,kernel,producer)
    got=scan.PBL_SCAN_ADAPTERS[bl](state,18.,grid)
    expected=jnp.moveaxis(coefficient.reshape(3,4,8),-1,0)
    np.testing.assert_array_equal(got.cam_kvh3d[:-1],expected)
    np.testing.assert_array_equal(got.cam_kvh3d[-1:],91.)
    np.testing.assert_array_equal(got.cam_tke_pbl[:-1],
        jnp.moveaxis(energy.reshape(3,4,8),-1,0) if tke else 73.)
    np.testing.assert_array_equal(got.cam_tke_pbl[-1:],73.)
    assert len(calls)==1,'PBL must not be recomputed to recover diagnostics'


def test_legacy_pbl_island_does_not_widen_shared_cam_real_carry(cpu_pallas_interpret):
    import importlib.util
    from gpuwrf.coupling.scan_adapters import _fp64_island
    spec=importlib.util.spec_from_file_location('cam_pbl_island',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(nz=8,ny=3,nx=4);state=f._state(grid).ensure_conditional_leaves(mp_physics=11)
    state=state.replace(theta=state.theta.astype(jnp.float32),_cast=False)
    def selected(wide,*args):
        assert wide.theta.dtype==np.float64
        for name in STATE_FIELDS:
            value=getattr(wide,name)
            assert value is getattr(state,name)
            assert value.dtype==(np.int32 if name=='cam_fatal' else np.float32)
        return wide
    result=_fp64_island(selected)(state,18.,grid)
    for name in STATE_FIELDS:assert getattr(result,name) is getattr(state,name)
