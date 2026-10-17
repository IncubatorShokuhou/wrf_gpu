"""Original one-moment Registry/caller, transport, restart and history contracts."""
import dataclasses
from datetime import datetime,timezone
import importlib.util,json
from pathlib import Path

import jax,jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


def fixture(name='test_rrtm_lw_operational_wiring.py'):
    spec=importlib.util.spec_from_file_location('one_moment_'+name.replace('.','_'),ROOT/'tests'/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('mp',[19,21])
def test_native_registry_adapter_transport(mp,monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.contracts.physics_registry import state_leaves_for_mp
    from gpuwrf.physics.nssl2mom import adapter
    from gpuwrf.runtime import operational_mode as op
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();g=f._grid(nz=8,ny=3,nx=4);st=f._state(g).ensure_conditional_leaves(mp_physics=mp)
    nml=dataclasses.replace(f._namelist(g),mp_physics=mp,moist_adv_opt=1,scalar_adv_opt=1)
    expected=('qv','qc','qr','qi','qs','qg')+(('qh','Nn') if mp==19 else ())
    assert state_leaves_for_mp(mp)==expected
    assert op._advected_scalar_species(nml)==('qv','qc','qr','qi','qs','qg')+(('qh','qvolg') if mp==19 else ())
    assert op._nested_number_scalar_species(nml)==(('Nn',) if mp==19 else ())
    assert op._root_flow_only(nml)==({'flow_only':('qh','qvolg','Nn')} if mp==19 else {})
    for name in ('Nh','qvolh'):assert getattr(st,name) is None
    assert (st.qh is not None)==(mp==19) and (st.qvolg is not None)==(mp==19)
    captured=[]
    def call(fields,dt,C,prec,**kw):
        captured.append((fields,C,kw));z=jnp.zeros_like(st.t_skin)
        return fields,dict(rain_liquid=z,snowncv=z,grplncv=z,hailncv=z),{}
    monkeypatch.setattr(adapter,'nssl2mom_column',call)
    out=adapter.nssl2mom_adapter(st,18.,g,variant=mp,first_step=True)
    fields,C,kw=captured[0]
    assert C.ipconc==0 and C.variant==mp
    assert kw['first_step'] is True and not kw['radii']
    assert kw['hail_enabled']==(mp==19)
    assert not {'qndrop','qnr','qni','qns','qng','qnh','qvolh'} & set(fields)
    assert set(fields)&{'qh','qvolg','qnn'}==({'qh','qvolg','qnn'} if mp==19 else set())
    for name in ('Nc','Ni','Nr','Ns','Ng'):
        np.testing.assert_array_equal(np.asarray(getattr(out,name)),np.asarray(getattr(st,name)))


@pytest.mark.parametrize('mp',[19,21])
def test_admission_binding_and_original_forced_flags(mp):
    from gpuwrf.io.namelist_check import validate_operational_namelist,refuse_single_domain_mp_nested,NotOperationallyWiredError
    from gpuwrf.io.namelist_binding import require_namelist_honoured,bind_namelist_options,NamelistNotHonouredError
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    from gpuwrf.physics.rrtmg_mp_re import mp_re_active
    f=fixture();g=f._grid(nz=8,ny=3,nx=4);raw={'physics':{'mp_physics':[mp],'cu_physics':[0]}}
    validate_operational_namelist(raw);require_namelist_honoured(raw,('d01',))
    assert bind_namelist_options(f._namelist(g),raw,'d01').mp_physics==mp
    assert classify_scheme('mp_physics',mp).status==SupportStatus.IMPLEMENTED
    assert not mp_re_active(mp_physics=mp,use_mp_re=1,ra_lw_physics=4,ra_sw_physics=4)
    with pytest.raises(NotOperationallyWiredError):refuse_single_domain_mp_nested(raw,2)
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({**raw,'nssl_mp_params':{'ircnw':2}},('d01',))
    require_namelist_honoured({'physics':{**raw['physics'],'nssl_2moment_on':1,'nssl_hail_on':1,
                                         'nssl_density_on':2,'nssl_ccn_on':1 if mp==19 else 0}},('d01',))
    if mp==19:
        for value in (-1,1):
            require_namelist_honoured({'physics':{**raw['physics'],'nssl_ccn_on':value}},('d01',))
        with pytest.raises(NamelistNotHonouredError,match='default CCN1 only'):
            require_namelist_honoured({'physics':{**raw['physics'],'nssl_ccn_on':0}},('d01',))
        with pytest.raises(NamelistNotHonouredError):
            require_namelist_honoured({**raw,'dynamics':{'diff_opt':2,'km_opt':2}},('d01',))
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({'physics':{**raw['physics'],'nssl_3moment':1}},('d01',))


@pytest.mark.parametrize('mp',[19,21])
@pytest.mark.parametrize('step',[1,2])
def test_actual_post_rk_variant_and_cold_start(mp,step,monkeypatch):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.physics.nssl2mom import adapter
    f=fixture();g=f._grid(nz=8,ny=3,nx=4);st=f._state(g).ensure_conditional_leaves(mp_physics=mp)
    nl=dataclasses.replace(f._namelist(g),mp_physics=mp,run_physics=True,moist_adv_opt=1,scalar_adv_opt=1)
    calls=[]
    def call(state,dt,grid,**kw):
        calls.append(kw);z=jnp.zeros_like(st.t_skin)
        return state,dict(rainncv=z,rain_liquid=z,snowncv=z,grplncv=z,hailncv=z,sr=z)
    monkeypatch.setattr(adapter,'nssl2mom_adapter',call)
    op._apply_post_rk_microphysics(st,nl,step_index=jnp.int32(step),return_precipitation=True)
    assert len(calls)==1 and calls[0]['variant']==mp
    assert bool(calls[0]['first_step'])==(step==1)


@pytest.mark.parametrize('mp',[19,21])
def test_native_restart_schema(mp,tmp_path,monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state,read_wrfrst_state
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();g=f._grid(nz=8,ny=3,nx=4);nl=dataclasses.replace(f._namelist(g),mp_physics=mp)
    st=f._state(g).ensure_conditional_leaves(mp_physics=mp)
    present=('qv','qc','qr','qi','qs','qg')+(('qh','qvolg','Nn','hail_acc') if mp==19 else ())
    st=st.replace(_cast=False,**{k:jnp.full_like(getattr(st,k),(i+1)/128) for i,k in enumerate(present)})
    write_checkpoint(st,nl,g,2,tmp_path/'restart.pkl');a,na,_,step=read_checkpoint(tmp_path/'restart.pkl')
    assert na.mp_physics==mp and step==2
    write_wrfrst_state(st,g,nl,tmp_path/'restart.nc',valid_time='2026-07-25_18:00:36',run_start='2026-07-25_18:00:00',step_index=2)
    b,_=read_wrfrst_state(tmp_path/'restart.nc')
    for got in (a,b):
        for name in present:
            x=np.asarray(getattr(st,name));y=np.asarray(getattr(got,name))
            assert x.dtype==y.dtype and x.tobytes()==y.tobytes(),(mp,name)
        for name in ('Nh','qvolh')+(('qh','qvolg','hail_acc') if mp==21 else ()):
            assert getattr(got,name) is None,(mp,name)


@pytest.mark.parametrize('mp',[19,21])
@pytest.mark.parametrize('mode',['full','operational','subset'])
def test_original_registry_history_inventory(mp,mode,tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    f=fixture('test_m7_netcdf_writer.py');st,g,nl=f.synthetic_case();nl['mp_physics']=mp
    proof=ROOT/'proofs/t_mpout'
    names={'QHAIL':'qh','QNCCN':'Nn','QVGRAUPEL':'qvolg'} if mp==19 else {}
    with np.load(proof/'cpu_history_samples.npz') as data:
        expected={name:data['mp18__'+name].copy() for name in names}
    for name,leaf in names.items():setattr(st,leaf,expected[name])
    for name in ('Nh','qvolh')+(('qh','qvolg','hail_acc') if mp==21 else ()):setattr(st,name,None)
    start=datetime(2023,1,15,tzinfo=timezone.utc);authority=f.writer_authority(g,'d01')
    omitted={'QNDROP','QNCLOUD','QNICE','QNRAIN','QNSNOW','QNGRAUPEL','QNHAIL','QVHAIL'}
    if mp==21:omitted|={'QHAIL','QNCCN','QVGRAUPEL'}
    subset=tuple({'QVAPOR',*names,*omitted}) if mode=='subset' else None
    payload=prepare_wrfout_payload(st,g,nl,tmp_path/'history.nc',domain='d01',domain_authority=authority,
        valid_time=start,run_start=start,lead_hours=0,full_variable_set=mode!='operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    schema=json.loads((proof/'cpu_history_schema.json').read_text())['mp18']['missing_schema']
    with Dataset(payload.target) as ds:
        assert set(names)<=set(ds.variables)
        assert not omitted&set(ds.variables),(mp,mode,omitted&set(ds.variables))
        for name,value in expected.items():
            var=ds[name];np.testing.assert_array_equal(var[0],value)
            assert var.dtype==np.dtype('float32') and var.dimensions==tuple(schema[name]['dimensions'])
            for key,ref in schema[name]['attrs'].items():assert var.getncattr(key)==ref,(name,key)


@pytest.mark.parametrize('name,leaf',[('QHAIL','qh'),('QNCCN','Nn'),('QVGRAUPEL','qvolg')])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_represented_one_moment_history_nonfinite_refused(name,leaf,bad,tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    f=fixture('test_m7_netcdf_writer.py');st,g,nl=f.synthetic_case();nl['mp_physics']=19
    with np.load(ROOT/'proofs/t_mpout/cpu_history_samples.npz') as data:value=data['mp18__'+name].copy()
    value.flat[0]=bad;setattr(st,leaf,value)
    start=datetime(2023,1,15,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='nonfinite microphysics history field '+name):
        prepare_wrfout_payload(st,g,nl,tmp_path/'bad.nc',domain='d01',domain_authority=f.writer_authority(g,'d01'),
            valid_time=start,run_start=start,lead_hours=0,full_variable_set=True,variable_subset=(name,))
