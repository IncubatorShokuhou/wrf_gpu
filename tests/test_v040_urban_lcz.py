"""LCZ/NUDAPT/heat gates against independently compiled pristine WRF."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import urban_slucm as us
from gpuwrf.coupling import urban_slucm_hook as hook
from gpuwrf.coupling.urban_inputs import UrbanInputs, prepare_morphology, ahe_at_lead, load_hourly, pbl_heat_tendency

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT/'proofs/v040/urban_lcz/savepoints'
sys.path.insert(0,str(ROOT/'proofs/v040/urban_slucm'))
import slucm_parity as sp

jax.config.update('jax_enable_x64',True)

def parity(precision,table):
    z = np.load(SAVE/f'slucm_{precision}_{table}.npz')
    dtype = np.float64 if precision=='fp64' else np.float32
    # Reuse the existing comparator, supplying only new inputs at the fixture boundary.
    def init(tbl,**kw):
        return us.urban_var_init(tbl,**kw,ivgtyp=z['in_ivgtyp'],use_wudapt_lcz=True)
    def step(tbl,**kw):
        fields = {k:z['in_'+v] for k,v in [('MH_URB2D','mh'),('HGT_URB2D','hgt'),
            ('STDH_URB2D','stdh'),('LP_URB2D','lp'),('LB_URB2D','lb')]}
        fields['LF_URB2D'] = np.broadcast_to(np.asarray([.1,.3,.5,.7],dtype),(25,4))
        geom = prepare_morphology(tbl,z['in_ivgtyp'],13,z['in_frc_urb'].astype(dtype),fields)
        geom = jax.tree.map(jnp.asarray,geom)
        return us.noahmp_urban(tbl,**kw,morphology=geom)
    module = SimpleNamespace(**{k:getattr(us,k) for k in dir(us) if not k.startswith('__')})
    module.urban_var_init = init; module.noahmp_urban = step
    return sp.run(precision,table,savepoint_dir=SAVE,kernel=module)

@pytest.mark.parametrize('table',['default','heat'])
@pytest.mark.parametrize('precision',['fp64','fp32'])
def test_pristine_lcz_morphology_and_diurnal_heat(table,precision):
    errors = parity(precision,table)
    z32=np.load(SAVE/f'slucm_fp32_{table}.npz'); z64=np.load(SAVE/f'slucm_fp64_{table}.npz')
    bad={}
    for k,e in errors.items():
        assert e['nonfinite']==0,k
        limit=1e-10*max(e['ref_absmax'],1e-30)
        if precision=='fp32' and k in z32:
            spread=np.max(np.abs(z32[k][:,1:]-z64[k][:,1:]))
            limit=3*spread+4*np.spacing(np.float32(e['ref_absmax']))
        if e['max_abs']>limit: bad[k]=(e['max_abs'],limit)
    assert not bad,bad

def test_pristine_regime_census_and_table_selection():
    z=np.load(SAVE/'slucm_fp64_default.npz')
    assert set(z['in_utype'])==set(range(1,12))
    assert (z['in_ivgtyp']==13).any()
    assert (z['in_mh']>0).sum()==12
    assert ((z['in_lb']>z['in_lp']) & (z['in_mh']>0)).any()
    assert ((z['in_lb']<=z['in_lp']) & (z['in_mh']>0)).any()
    hot=np.load(SAVE/'slucm_fp64_heat.npz')
    assert np.max(np.abs(hot['HFX'][:,1:]-z['HFX'][:,1:]))>10
    table=us.load_urban_table(SAVE,use_wudapt_lcz=True)
    assert table.icate==11 and np.array_equal(table.zr,np.array([37.5,17.5,6.5,37.5,17.5,6.5,3,6.5,6.5,10,10],np.float32))

def test_ahe_month_hour_reference_and_year_wrap():
    b=SimpleNamespace(start=datetime(2024,12,31,23),ahe_opt=2)
    data=np.arange(288,dtype=np.float32)[:,None,None]
    inputs=UrbanInputs(ahe=jnp.asarray(data))
    f=jax.jit(lambda s:ahe_at_lead(b,inputs,s))
    assert float(f(0)[0,0])==287
    assert float(f(3600)[0,0])==264  # WRF keeps simulation-start JULDAY/month.
    hourly=UrbanInputs(hourly=jnp.asarray([[[10.]],[[20.]]]))
    assert float(ahe_at_lead(b,hourly,3599)[0,0])==10
    assert float(ahe_at_lead(b,hourly,3600)[0,0])==20
    assert np.isnan(np.asarray(ahe_at_lead(b,hourly,7200))).all()  # no silent last-record clamp

def test_pbl_heat_tendency_independent_wrf_formula():
    raw=np.arange(12,dtype=np.float32).reshape(3,2,2)/np.float32(100)
    ahe=np.array([[0,10],[50,100]],np.float32)
    dz=np.array([[30,40],[50,60]],np.float32)
    expect=raw.copy()
    expect[0]=expect[0]+(ahe/np.float32(1200))/dz
    got=jax.jit(pbl_heat_tendency)(jnp.asarray(raw),jnp.asarray(ahe),jnp.asarray(dz))
    np.testing.assert_allclose(got,expect,rtol=1e-7,atol=1e-10)
    assert not np.array_equal(np.asarray(got[0]),raw[0])

@pytest.mark.parametrize('category',[51,61])
def test_doctor_canonical_lcz(category):
    from gpuwrf.doctor.land import category_issues
    d=SimpleNamespace(MMINLU='MODIFIED_IGBP_MODIS_NOAH',variables={
        k:np.array([[category]]) for k in ('LU_INDEX','IVGTYP')})
    assert any(k=='LCZ_WITHOUT_URBAN' and s=='ERROR' for k,m,s in category_issues(d,61,0,True))
    assert not category_issues(d,61,1,True)
    for urban in (2,3):
        assert any(k=='LCZ_MULTILAYER_UNSUPPORTED' and s=='ERROR' for k,m,s in category_issues(d,61,urban,True))
    d.variables={k:np.array([[31]]) for k in d.variables}
    assert any(k=='LCZ_LEGACY_NUMBERING' for k,m,s in category_issues(d,61,1,True))

def test_binding_admits_native_and_refuses_replay():
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nl={'physics':{'sf_urban_physics':1,'sf_surface_physics':4,'use_wudapt_lcz':1,
                   'distributed_ahe_opt':2},'dynamics':{},'noah_mp':{}}
    assert not namelist_honour_report(nl,('d01',))[0]
    assert namelist_honour_report(nl,('d01',),replay=True)[0]
    nl['physics']['sf_urban_physics']=0
    assert namelist_honour_report(nl,('d01',))[0]

def test_hourly_file_units_coverage_and_finiteness(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    p=tmp_path/'ahe.nc'; start=datetime(2026,1,1)
    with Dataset(p,'w') as ds:
        ds.createDimension('Time',2);ds.createDimension('DateStrLen',19)
        ds.createDimension('y',1);ds.createDimension('x',1)
        t=ds.createVariable('Times','S1',('Time','DateStrLen'))
        t[:]=np.asarray([list('2026-01-01_00:00:00'),list('2026-01-01_01:00:00')],dtype='S1')
        v=ds.createVariable('AHE','f4',('Time','y','x'));v.units='W m-2';v[:]=[[[10]],[[20]]]
    np.testing.assert_array_equal(load_hourly(p,start=start,hours=2,shape=(1,1)),[[[10]],[[20]]])
    assert len(load_hourly(p,start=start,hours=1,shape=(1,1)))==2  # retain later declared hours for restart extension
    from datetime import timezone
    assert len(load_hourly(p,start=start.replace(tzinfo=timezone.utc),hours=1,shape=(1,1)))==2
    with pytest.raises(ValueError,match='cover'):
        load_hourly(p,start=start,hours=3,shape=(1,1))
    with Dataset(p,'a') as ds: ds['AHE'][1]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        load_hourly(p,start=start,hours=2,shape=(1,1))


def test_loader_wires_lcz_morphology_and_ahe():
    from gpuwrf.integration.nested_pipeline import _load_urban_slucm
    # The fixture is a wrfinput seam, independent field values and canonical LCZ IDs.
    shape=(2,2)
    ivg=np.array([[53,56],[13,10]],np.int32)
    data={'FRC_URB2D':np.full(shape,.7,np.float32),
        'HGT_URB2D':np.full(shape,12,np.float32),'MH_URB2D':np.full(shape,12,np.float32),
        'STDH_URB2D':np.full(shape,3,np.float32),'LP_URB2D':np.full(shape,.3,np.float32),
        'LB_URB2D':np.ones(shape,np.float32),
        'LF_URB2D':np.broadcast_to(np.array([.1,.3,.5,.7],np.float32)[:,None,None],(4,)+shape),
        'AHE':np.broadcast_to(np.arange(288,dtype=np.float32)[:,None,None],(288,)+shape)}
    run=SimpleNamespace(namelist={'physics':{'use_wudapt_lcz':1,'distributed_ahe_opt':2}},
        wrfinput_variables=lambda domain:list(data),load_wrfinput=lambda d,k,**kw:data[k])
    class Static(SimpleNamespace):
        def replace(self,**kw): return Static(**(vars(self)|kw))
    static=Static(ivgtyp=jnp.asarray(ivg),shdmax=jnp.full(shape,.4),parameters=SimpleNamespace(isurban=13))
    land=SimpleNamespace(t_skin=jnp.full(shape,290.),tslb=jnp.full((4,)+shape,288.))
    bundle,carry,natural=_load_urban_slucm(run,'d01',land,static,dz1=np.full(shape,80.),
        run_start=datetime(2026,7,1,18),hours=2)
    np.testing.assert_array_equal(carry.utype,[[3,6],[5,0]])
    np.testing.assert_array_equal(natural.ivgtyp,[[14,14],[14,10]])
    assert bundle.requires_inputs and carry.inputs.morphology is not None
    assert carry.inputs.ahe.shape==(288,)+shape
    assert np.all(np.asarray(carry.inputs.morphology.zr)[:1]==12.)
    # Original WRF IO uses Registry labels, not the grid's member names.
    aliases={'LP_URB2D':'BUILD_AREA_FRACTION','HGT_URB2D':'BUILD_HEIGHT','LB_URB2D':'BUILD_SURF_RATIO'}
    original_data=data
    data={aliases.get(k,k):v for k,v in data.items()}
    _,aliascarry,_=_load_urban_slucm(run,'d01',land,static,dz1=np.full(shape,80.),
        run_start=datetime(2026,7,1,18),hours=2)
    for a,b in zip(jax.tree.leaves(carry),jax.tree.leaves(aliascarry)):
        np.testing.assert_array_equal(a,b)
    data=original_data
    # Raw URB_PARAM fallback has WRF's exact one-based slot map.
    packed=np.zeros((132,)+shape,np.float32)
    for k,n in [(90,'LP_URB2D'),(91,'MH_URB2D'),(92,'STDH_URB2D'),(93,'HGT_URB2D'),(94,'LB_URB2D')]:
        packed[k]=data[n]
    packed[95:99]=data['LF_URB2D']
    data={k:v for k,v in data.items() if not k.endswith('_URB2D') or k=='FRC_URB2D'}
    data['URB_PARAM']=packed
    _,rawcarry,_=_load_urban_slucm(run,'d01',land,static,dz1=np.full(shape,80.),
        run_start=datetime(2026,7,1,18),hours=2)
    for a,b in zip(jax.tree.leaves(carry.inputs.morphology),jax.tree.leaves(rawcarry.inputs.morphology)):
        np.testing.assert_array_equal(a,b)
    static.ivgtyp=jnp.full(shape,53,jnp.int32)
    with pytest.raises(ValueError,match='3353'):
        _load_urban_slucm(run,'d01',land,static,dz1=np.full(shape,80.),
            run_start=datetime(2026,7,1,18),hours=2)


def test_urban_input_restart_netcdf_and_missing_schema(tmp_path,monkeypatch):
    import json
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io import wrfrst_netcdf as rst
    tbl=us.load_urban_table(SAVE,use_wudapt_lcz=True)
    bundle=hook.UrbanBundle(tbl,13,14,use_wudapt_lcz=True,requires_inputs=True)
    inputs=UrbanInputs(ahe=jnp.arange(288*4,dtype=jnp.float32).reshape(288,2,2))
    urban=hook.initial_urban_carry(bundle,ivgtyp=jnp.full((2,2),53),tsk=jnp.full((2,2),290.),
        tslb=jnp.full((4,2,2),288.),frc_urb2d=jnp.zeros((2,2)),inputs=inputs)
    carry=SimpleNamespace(**{k:None for k in rst.OPTIONAL_CARRY_FIELDS})
    carry.urban_slucm=urban
    p=tmp_path/'urban.nc'
    with Dataset(p,'w') as ds:
        ds.createDimension('south_north',2); ds.createDimension('west_east',2)
        ds.createDimension('bottom_top',3)
        # The enclosing restart writer supplies this State plane; optional
        # group readers use its geometry even when SSiB itself is inactive.
        ds.createDimension('Time',1)
        ds.createVariable(rst.state_extension_name('t_skin'),'f4',
                          ('Time','south_north','west_east'))[0]=290.
        rst._write_optional_carry_variables(ds,carry,{})
        # These two manifests are the enclosing writer's normal responsibility.
        monkeypatch.setattr(rst,'_optional_carry_manifest',lambda ds:rst._optional_carry_field_order(carry))
        monkeypatch.setattr(rst,'_optional_carry_kind_manifest',lambda ds:rst._optional_carry_kind(carry))
        nl=SimpleNamespace(urban_bundle=bundle,bl_pbl_physics=5,sf_lake_physics=0)
        restored=rst._read_optional_carry_groups(ds,namelist=nl)['urban_slucm']
        for a,b in zip(jax.tree.leaves(urban),jax.tree.leaves(restored)):
            np.testing.assert_array_equal(a,b)
        ds.delncattr('GPUWRF_URBAN_INPUTS_SCHEMA_VERSION')
        with pytest.raises(ValueError,match='E78'):
            rst._read_optional_carry_groups(ds,namelist=nl)


def test_public_and_native_restart_require_declared_urban_inputs(tmp_path):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State,_state_field_shapes
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.runtime.restart_store import RestartStore
    from gpuwrf.io.urban_schema import configuration
    grid=GridSpec.canary_3km_template()
    shape=(grid.ny,grid.nx)
    state=State(**{k:jnp.zeros(s,DEFAULT_DTYPES.dtype_for(k)) for k,s in _state_field_shapes(grid).items()})
    state=state.ensure_conditional_leaves(mp_physics=8)
    carry=initial_operational_carry(state)
    bundle=hook.UrbanBundle(us.load_urban_table(SAVE,use_wudapt_lcz=True),13,14,
        use_wudapt_lcz=True,ahe_opt=2,requires_inputs=True,start=None)
    urban=hook.initial_urban_carry(bundle,ivgtyp=jnp.full(shape,53),tsk=jnp.full(shape,290.),
        tslb=jnp.full((4,)+shape,288.),frc_urb2d=jnp.full(shape,.7),
        inputs=UrbanInputs(ahe=jnp.ones((288,)+shape,jnp.float32),clock=jnp.array([182,18,1080,7],jnp.float32)))
    carry=carry.replace(urban_slucm=urban)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=18,acoustic_substeps=4,
        sf_urban_physics=1,urban_bundle=bundle,use_noahmp=True,sf_surface_physics=4)
    path=tmp_path/'urban.pkl'
    write_restart(carry,nl,grid,1,path)
    restored=read_restart(path,namelist=nl)[0]
    for a,b in zip(jax.tree.leaves(carry),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a,b)
    identity={'urban_inputs_config':{'d01':configuration(nl)}}
    store=RestartStore(tmp_path/'native',10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':1},{'d01':18.},identity)
    back=store.read(generation,expected_identity=identity)[0]['carries']['d01']
    np.testing.assert_array_equal(back.urban_slucm.inputs.ahe,urban.inputs.ahe)
    missing=carry.replace(urban_slucm=urban._replace(inputs=None))
    with pytest.raises(ValueError,match='E78'):
        write_restart(missing,nl,grid,1,tmp_path/'missing.pkl')
    with pytest.raises(ValueError,match='E78'):
        store.save({'d01':missing},{'d01':2},{'d01':18.},identity)
    missing_clock=carry.replace(urban_slucm=urban._replace(inputs=urban.inputs._replace(clock=None)))
    with pytest.raises(ValueError,match='clock.*E78'):
        write_restart(missing_clock,nl,grid,1,tmp_path/'missing_clock.pkl')


def test_distributed_ahe_is_excluded_from_achfx_like_wrf():
    from gpuwrf.runtime.history_accumulators import accumulate_energy,SURFACE_SOURCES,RADIATION_SOURCES
    old={k:jnp.zeros((1,),jnp.float32) for k in SURFACE_SOURCES|RADIATION_SOURCES}
    surface={'hfx':jnp.array([35.],jnp.float32),'accumulator_hfx':jnp.array([10.],jnp.float32),
             'lh':jnp.array([5.],jnp.float32),'grdflx':jnp.array([-2.],jnp.float32)}
    result=accumulate_energy(old,None,surface,18.)
    # module_surface_driver.F:3432 precedes its :4588 distributed AHE addition.
    np.testing.assert_array_equal(result['ACHFX'],[180.])
    np.testing.assert_array_equal(result['ACLHF'],[90.])


def test_heat_clock_and_ahe_against_verbatim_wrf_driver_statements():
    from gpuwrf.coupling.urban_inputs import urban_clock
    z=np.load(SAVE/'clock_reference.npz')
    bundle=SimpleNamespace(start=None,rad_interval_s=1800.,dt_s=54.,solar_every_step=False)
    inputs=UrbanInputs(longitude=jnp.full((1,1),-16.5,jnp.float32),clock=jnp.array([13,18,1080,1],jnp.float32))
    angles,months=jax.jit(jax.vmap(lambda s,clock:urban_clock(bundle,inputs._replace(clock=clock),s),
        in_axes=(0,None)))(jnp.asarray(z['lead_seconds'],jnp.float32),inputs.clock)
    actual=np.asarray(angles).reshape(-1)
    ref=z['hour_angle_fp32'];real_spread=np.abs(ref-z['hour_angle_fp64'])
    limit=3*real_spread+4*np.abs(np.spacing(ref.astype(np.float32)))
    assert np.isfinite(actual).all() and np.all(np.abs(actual-ref)<=limit)
    assert np.all(np.asarray(months)==1)
    for n,col in [(24,0),(48,1)]:
        got=us._hour_index(angles,jnp.float32(3.14159),n,lambda x:np.float32(x))
        np.testing.assert_array_equal(np.asarray(got).reshape(-1),z['hour_indices_fp32'][:,col])
    bundle=SimpleNamespace(start=None)
    inputs=UrbanInputs(ahe=jnp.arange(288,dtype=jnp.float32).reshape(288,1,1),
        clock=jnp.array([365,18,1080,12],jnp.float32))
    heat=ahe_at_lead(bundle,inputs,3600.)
    assert float(heat[0,0])==z['ahe_fp32'][0]
    assert float(jnp.float32(10)+heat[0,0])==z['ahe_fp32'][1]
    tendency=pbl_heat_tendency(jnp.full((1,1,1),.001,jnp.float32),heat,jnp.array([[20.]],jnp.float32))
    np.testing.assert_allclose(tendency[0,0,0],z['ahe_fp32'][2],rtol=1e-7,atol=1e-9)


def test_surface_ahe_routing_and_pre_ahe_history(monkeypatch):
    from test_v040_urban_slucm_wiring import _adapter_case
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    monkeypatch.setenv('GPUWRF_LAND_Q2_CAP_POSTLSM','0')
    state,land,static,rad,clock,bundle,carry,_=_adapter_case()
    base=noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=6.,
        history=True,land_history=True,urban=(bundle,carry))
    bundle=replace(bundle,ahe_opt=2,requires_inputs=True,start=datetime(2026,12,31,18))
    shape=carry.utype.shape
    ahe=jnp.broadcast_to(jnp.arange(288,dtype=jnp.float32)[:,None,None],(288,)+shape)
    carry=carry._replace(inputs=UrbanInputs(ahe=ahe))
    heated=noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=6.,
        history=True,land_history=True,urban=(bundle,carry),urban_lead_seconds=3600.)
    reference=np.load(SAVE/'clock_reference.npz')['ahe_fp32'][0]
    # Exact WRF surface-driver addition, including non-urban and water points.
    np.testing.assert_allclose(heated[3]['hfx'],np.asarray(base[3]['hfx'])+reference,rtol=1e-7,atol=1e-6)
    np.testing.assert_array_equal(heated[3]['accumulator_hfx'],base[3]['hfx'])


def test_fixture_source_and_hashes_are_sealed():
    import hashlib,json
    manifest=json.loads((SAVE/'manifest.json').read_text())
    assert manifest['wrf_commit']=='f52c197ed39d12e087d02c50f412d90d418f6186'
    assert all(k in manifest['flags'] for k in ('-finit-real=zero','-finit-integer=0','-finit-logical=false'))
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((SAVE/name).read_bytes()).hexdigest()==digest
    clock=json.loads((SAVE/'clock_manifest.json').read_text())
    assert hashlib.sha256((SAVE/'clock_reference.npz').read_bytes()).hexdigest()==clock['fixture_sha256']


def test_production_surface_hook_forwards_urban_clock(monkeypatch):
    from gpuwrf.coupling import noahmp_surface_hook as surface_hook
    class State(SimpleNamespace):
        def replace(self,**kw): return State(**(vars(self)|kw))
    z=jnp.ones((2,2),jnp.float32)
    names=('ustar','theta_flux','qv_flux','tau_u','tau_v','rhosfc','fltv','t_skin','roughness_m')
    state=State(**dict.fromkeys(names,z))
    view=SimpleNamespace(t_skin=z,roughness_m=z)
    blend=SimpleNamespace(**dict.fromkeys(names,z),wspd=None)
    monkeypatch.setattr(surface_hook,'_build_column_view',lambda *a:view)
    bundle=SimpleNamespace(requires_inputs=True)
    sentinel=object();land=object();seen=[]
    def adapter(*a,**kw):
        seen.append(kw.get('urban_lead_seconds'))
        return view,land,blend,sentinel
    monkeypatch.setattr(surface_hook,'noahmp_surface_adapter',adapter)
    result=surface_hook.noahmp_surface_step(state,land,object(),18.,urban=(bundle,sentinel),
        urban_lead_seconds=1800.)
    assert seen==[1800.] and result[-1] is sentinel


def test_manual_diurnal_bundle_cannot_silently_use_noon():
    table=us.parse_urbparm(SAVE/'URBPARM.TBL.heat')
    bundle=hook.UrbanBundle(table,13,14)
    assert bundle.requires_inputs
    carry=hook.initial_urban_carry(bundle,ivgtyp=jnp.full((1,1),13),tsk=jnp.full((1,1),290.),
        tslb=jnp.full((4,1,1),288.),frc_urb2d=jnp.full((1,1),.7))
    kwargs=dict(forcing=None,nm=None,ust=None,chs=None,chs2=None,cqs2=None,qv_mixing_ratio=None,dt=18.)
    with pytest.raises(ValueError,match='missing required urban inputs'):
        hook.apply_urban_slucm(bundle,carry,**kwargs)
    carry=carry._replace(inputs=UrbanInputs(longitude=jnp.zeros((1,1))))
    with pytest.raises(ValueError,match='dated production clock'):
        hook.apply_urban_slucm(bundle,carry,lead_seconds=0.,**kwargs)


def test_production_clock_anchors_do_not_split_static_program_key():
    from test_v040_urban_slucm_wiring import _namelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    from gpuwrf.coupling.urban_inputs import ahe_at_lead
    table=us.load_urban_table(SAVE,use_wudapt_lcz=True)
    bundle=hook.UrbanBundle(table,13,14,use_wudapt_lcz=True,ahe_opt=2,start=None)
    nl=_namelist()
    a=replace(nl,urban_bundle=bundle,sf_urban_physics=1,use_noahmp=True,sf_surface_physics=4)
    b=replace(nl,urban_bundle=replace(bundle),sf_urban_physics=1,use_noahmp=True,sf_surface_physics=4)
    assert static_config_hash(a)==static_config_hash(b)
    object.__setattr__(b.urban_bundle,'_hourly_source',('host-only-path','host-only-digest'))
    assert static_config_hash(a)==static_config_hash(b)
    changed=replace(bundle,table=replace(table,ah=table.ah+np.float32(1)))
    assert static_config_hash(a)!=static_config_hash(replace(a,urban_bundle=changed))
    inputs=UrbanInputs(ahe=jnp.arange(288,dtype=jnp.float32).reshape(288,1,1))
    f=jax.jit(lambda clock:ahe_at_lead(bundle,inputs._replace(clock=clock),0.))
    jan=jnp.array([13,18,1080,1],jnp.float32);jul=jnp.array([182,18,1080,7],jnp.float32)
    assert float(f(jan)[0,0])==18 and float(f(jul)[0,0])==162
    assert f.lower(jan).compiler_ir('hlo').as_hlo_text()==f.lower(jul).compiler_ir('hlo').as_hlo_text()


@pytest.mark.parametrize('count',[4,132,288])
def test_layered_wrf_inputs_follow_dimension_names(count):
    from contextlib import contextmanager
    from gpuwrf.coupling.urban_inputs import load_layered
    canonical=np.arange(count*6,dtype=np.float32).reshape(count,2,3)
    raw=np.transpose(canonical,(1,0,2))
    variable=SimpleNamespace(dimensions=('Time','south_north','special_category','west_east'))
    @contextmanager
    def dataset(path):
        yield SimpleNamespace(variables={'FIELD':variable})
    run=SimpleNamespace(load_wrfinput=lambda *a,**kw:raw,wrfinput_file=lambda domain:'readonly',_read_dataset=dataset)
    np.testing.assert_array_equal(load_layered(run,'d01','FIELD',(2,3),count),canonical)
    with pytest.raises(ValueError,match='expected'):
        load_layered(run,'d01','FIELD',(2,3),count+1)


def test_native_restart_identity_binds_external_hourly_source(tmp_path):
    import hashlib
    from test_v040_urban_slucm_wiring import _namelist
    from gpuwrf.runtime.restart_store import nested_identity
    path=tmp_path/'hourly-source.bin';path.write_bytes(b'first declared forcing')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    bundle=hook.UrbanBundle(us.load_urban_table(SAVE,use_wudapt_lcz=True),13,14,
        use_wudapt_lcz=True,ahe_opt=2)
    object.__setattr__(bundle,'_hourly_source',(str(path),digest))
    nl=replace(_namelist(),urban_bundle=bundle,sf_urban_physics=1,use_noahmp=True,sf_surface_physics=4)
    domains={'d01':SimpleNamespace(namelist=nl)}
    config=SimpleNamespace(input_dir=tmp_path,output_dir=tmp_path/'out',feedback=False,emit_initial_history=False)
    identity=nested_identity(config,domains,datetime(2026,1,1))
    assert identity['urban_ahe_sources']['d01']=={'path':str(path),'sha256':digest}
    path.write_bytes(b'changed forcing')
    with pytest.raises(ValueError,match='changed after initialization'):
        nested_identity(config,domains,datetime(2026,1,1))


def test_multilayer_bundles_do_not_declare_slucm_restart_inputs():
    from gpuwrf.io.urban_schema import configuration
    from gpuwrf.io.namelist_binding import namelist_honour_report
    for option in (2,3):
        bundle=SimpleNamespace(option=option)
        assert not any(configuration(SimpleNamespace(urban_bundle=bundle)).values())
        physics={'sf_urban_physics':option,'sf_surface_physics':4,'bl_pbl_physics':1,
                 'sf_sfclay_physics':1,'ra_sw_physics':4,'ra_lw_physics':4}
        errors=namelist_honour_report({'physics':physics,'dynamics':{},'noah_mp':{}},('d01',))[0]
        assert not errors,errors
        physics['use_wudapt_lcz']=1
        errors=namelist_honour_report({'physics':physics,'dynamics':{},'noah_mp':{}},('d01',))[0]
        assert any('grouped per-type geometry/carries/PBL sources' in e for e in errors),errors


@pytest.mark.parametrize('option',[2,3])
def test_multilayer_surface_adapter_does_not_read_slucm_ahe_fields(monkeypatch,option):
    from collections import namedtuple
    from test_v040_urban_slucm_wiring import _adapter_case
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    from gpuwrf.coupling import urban_multilayer_hook as multilayer
    from gpuwrf.coupling import urban_surface_layer as sf
    state,land,static,rad,clock,_,_,_= _adapter_case()
    diag=surface_layer_with_diagnostics(state,snowh=land.snowh,with_chs=True)
    # 5c0e7c11ed: the multilayer path asks the revised-MM5 layer for its PSIM/PSIH (PBL resistances) too.
    psim,psih=jnp.full_like(diag.br,1.25),jnp.full_like(diag.br,-0.75)
    calls=[]
    def revised(state,*,return_pbl_resistances=False):
        calls.append(return_pbl_resistances)
        return (diag,(psim,psih)) if return_pbl_resistances else diag
    monkeypatch.setattr(sf,'revised_surface_layer',revised)
    carry=namedtuple('Carry',['surface_layer'])(None)
    # The actual multilayer bundle has no SLUCM ahe_opt/requires_inputs fields.
    bundle=SimpleNamespace(option=option,indices=np.array([0]))
    profile=SimpleNamespace(u=jnp.ones((1,1)),v=jnp.ones((1,1)))
    monkeypatch.setattr(multilayer,'apply_urban_multilayer',
        lambda bundle,carry,**kw:(kw['nm'],kw['ust'],carry,None))
    out=noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=6.,
        history=True,land_history=True,urban=(bundle,carry,(profile,)))
    assert calls==[True]
    held=out[-1].surface_layer
    assert held is not None
    # the held handoff carries the revised-MM5 BR/PSIM/PSIH, not a recomputed or default layer
    for got,want in ((held.br,diag.br),(held.psim,psim),(held.psih,psih)):
        assert np.asarray(got).tobytes()==np.asarray(want).tobytes()
    assert 'accumulator_hfx' not in out[3]
    assert np.isfinite(np.asarray(out[3]['hfx'])).all()
