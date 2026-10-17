"""WRF legacy22: independent active pristine columns, no hail or hail carries.

Reuse the existing frozen NSSL bounds, including its documented REAL rounding
band. The sensitive CCNA endpoint is bounded by actual original22 one-ULP
controls, never by a port-generated alternative.
"""
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'proofs/s2_mpmore/nssl22'
NAMES={'th':'TH','qv':'QV','qc':'QC','qr':'QR','qi':'QI','qs':'QS','qg':'QH','qh':'QHL',
       'qndrop':'CCW','qnr':'CRW','qni':'CCI','qns':'CSW','qng':'CHW','qnh':'CHL','qnn':'CN',
       'qvolg':'VHW','qvolh':'VHL'}
MASS={'qndrop':'qc','qnr':'qr','qni':'qi','qns':'qs','qng':'qg','qnh':'qh','qvolg':'qg','qvolh':'qh'}
FLOOR={k:1e-14 for k in ('qv','qc','qr','qi','qs','qg','qh')}
FLOOR.update({k:1e-6 for k in ('qndrop','qnr','qni','qns','qng','qnh','qnn')})
FLOOR.update(th=0.,qvolg=1e-17,qvolh=1e-17)


def load(mode,case):
    # The legacy dump rounds R8 inputs to 16 significant digits. Case15's
    # full-precision control keeps its actual binary64 baseline available.
    path=DATA/'ulp/shift_0.json' if mode=='fp64' and case==15 else DATA/f'fixtures/{mode}_{case}.json'
    return json.loads(path.read_text())


def fields(d,dtype):
    c=d['columns'];f={k:np.asarray(c[v+'_IN'],dtype) for k,v in NAMES.items()}
    f.update({k:np.asarray(c[v],dtype) for k,v in [('pii','PII'),('p','P'),('rho','DN'),('w','W'),('dz','DZ')]})
    return f


FUN={}
def column(mode,dt):
    from gpuwrf.physics.nssl2mom.driver import nssl2mom_column
    from gpuwrf.physics.nssl2mom.constants import get_constants
    from gpuwrf.physics.nssl2mom.indices import FP32,FP64
    key=(mode,float(dt))
    if key not in FUN:
        C=get_constants(mode);prec=FP32 if mode=='fp32' else FP64
        FUN[key]=jax.jit(lambda f,first:nssl2mom_column(f,float(dt),C,prec,first_step=first,diag=True,hail_enabled=False))
    return FUN[key]


@pytest.mark.parametrize('case',range(1,18))
@pytest.mark.parametrize('mode',['fp64','fp32'])
def test_complete_original22_column(mode,case):
    d=load(mode,case);dt=float(d['scalars']['DT'])
    out,pr,diag=jax.tree.map(np.asarray,column(mode,dt)(fields(d,np.float32 if mode=='fp32' else np.float64),
                               np.bool_(d['scalars']['ITIMESTEP']==1)))
    r64=load('fp64',case);r32=load('fp32',case)
    for k,name in NAMES.items():
        p=np.asarray(out[k],np.float64);ref=np.asarray(d['columns'][name+'_OUT'],np.float64)
        assert np.isfinite(p).all() and np.isfinite(ref).all(),(mode,case,k)
        if mode=='fp64':
            scale=max(np.max(np.abs(ref)),np.max(np.abs(d['columns'][name+'_IN'])))
            bad=np.abs(p-ref)>1e-10*np.abs(ref)+1e-13*scale
            if case==15 and k=='qnn':
                controls=[json.loads((DATA/f'ulp/shift_{s}.json').read_text())['columns']['CN_OUT'][12] for s in (-1,0,1)]
                assert np.isfinite(controls).all()
                bad[12]=not any(abs(p[12]-v)<=1e-12*abs(v) for v in controls)
            assert not bad.any(),(mode,case,k,np.flatnonzero(bad),p[bad],ref[bad])
        else:
            a=np.asarray(r32['columns'][name+'_OUT']);b=np.asarray(r64['columns'][name+'_OUT'])
            assert np.isfinite(a).all() and np.isfinite(b).all(),(case,k)
            atol=1e-6*max(np.max(np.abs(a)),np.max(np.abs(b)),1e-30)
            bad=~((np.abs(p-a)<=2e-3*np.abs(a)+atol)|(np.abs(p-b)<=2e-3*np.abs(b)+atol))
            if k in MASS:
                m=MASS[k];v=NAMES[m]
                mass=np.maximum.reduce([np.abs(out[m]),np.abs(r32['columns'][v+'_OUT']),np.abs(r64['columns'][v+'_OUT'])])
                bad &= mass>=1e-10
            assert not bad.any(),(mode,case,k,np.flatnonzero(bad),p[bad],a[bad],b[bad])
    for k in ('qh','qnh','qvolh'):
        assert np.count_nonzero(out[k])==0,(mode,case,k)
    if mode=='fp64' and case==15:
        assert abs(out['qc'][12]-d['columns']['QC_OUT'][12])<=1e-15*max(np.abs(d['columns']['QC_IN']))
    for k in ('rainncv','snowncv','grplncv','hailncv'):
        p=float(pr[k]);assert np.isfinite(p)
        if mode=='fp64':
            ref=d['scalars'][k.upper()]
            assert abs(p-ref)<=1e-9*abs(ref)+1e-15,(case,k,p,ref)
        else:
            a=r32['scalars'][k.upper()];b=r64['scalars'][k.upper()]
            assert min(abs(p-a),abs(p-b))<=2e-3*max(abs(a),abs(b))+1e-12,(case,k,p,a,b)
    for k,name in [('dbz','DBZ_OUT'),('re_cloud','RE_CLOUD_OUT'),('re_ice','RE_ICE_OUT'),('re_snow','RE_SNOW_OUT')]:
        p=np.asarray(diag[k]);ref=np.asarray(d['columns'][name]);assert np.isfinite(p).all()
        if mode=='fp64':
            np.testing.assert_allclose(p,ref,rtol=1e-10,atol=1e-14,err_msg=str((case,k)))
    assert float(pr['hailncv'])==0


def fixture():
    s=importlib.util.spec_from_file_location('nohail22_state',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(s);s.loader.exec_module(f);return f


def test_no_hail_registry_transport_and_adaptor(monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.physics.nssl2mom import adapter
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();g=f._grid(nz=8,ny=3,nx=4)
    st=f._state(g).ensure_conditional_leaves(mp_physics=22)
    assert st.qvolg is not None
    for k in ('qh','Nh','qvolh','hail_acc'):assert getattr(st,k) is None,k
    nml=dataclasses.replace(f._namelist(g),mp_physics=22,moist_adv_opt=1,scalar_adv_opt=1)
    assert op._advected_scalar_species(nml)==('qv','qc','qr','qi','qs','qg','qvolg')
    assert op._nested_number_scalar_species(nml)==('Ni','Nr','Nc','Ns','Ng','Nn')
    assert set(op._root_flow_only(nml)['flow_only'])=={'qvolg','Nc','Ns','Ng','Nn'}
    seen=[]
    def capture(fields,dt,C,prec,**kw):
        seen.append((fields,kw));z=jnp.zeros((3,4),st.theta.dtype)
        return fields,dict(rain_liquid=z,snowncv=z,grplncv=z,hailncv=z),{}
    monkeypatch.setattr(adapter,'nssl2mom_column',capture)
    got=adapter.nssl2mom_adapter(st,18.,g,hail_enabled=False)
    assert seen[0][1]['hail_enabled'] is False
    assert not {'qh','qnh','qvolh'} & set(seen[0][0])
    for k in ('qh','Nh','qvolh','hail_acc'):assert getattr(got,k) is None,k


def test_public_binding_and_admission():
    from gpuwrf.io.namelist_check import validate_operational_namelist,refuse_single_domain_mp_nested,NotOperationallyWiredError
    from gpuwrf.io.namelist_binding import bind_namelist_options,require_namelist_honoured,NamelistNotHonouredError
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    f=fixture();g=f._grid(nz=8,ny=3,nx=4)
    raw={'physics':{'mp_physics':[22],'cu_physics':[0]}}
    validate_operational_namelist(raw);require_namelist_honoured(raw,('d01',))
    nml=bind_namelist_options(f._namelist(g),raw,'d01')
    assert nml.mp_physics==22
    assert classify_scheme('mp_physics',22).status==SupportStatus.IMPLEMENTED
    with pytest.raises(NotOperationallyWiredError):refuse_single_domain_mp_nested(raw,2)
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({**raw,'dynamics':{'diff_opt':2,'km_opt':2}},('d01',))
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({**raw,'nssl_mp_params':{'irenuc':2}},('d01',))
    # Original22 forcibly selects its four flags, including explicitly conflicting values.
    require_namelist_honoured({'physics':{**raw['physics'],'nssl_hail_on':1,'nssl_density_on':0,
                                         'nssl_2moment_on':0,'nssl_ccn_on':0}},('d01',))


@pytest.mark.parametrize('step',[1,2])
def test_operational_post_rk_selects_no_hail_and_first_call(monkeypatch,step):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.physics.nssl2mom import adapter
    f=fixture();g=f._grid(nz=8,ny=3,nx=4)
    st=f._state(g).ensure_conditional_leaves(mp_physics=22)
    nml=dataclasses.replace(f._namelist(g),mp_physics=22,run_physics=True,moist_adv_opt=1,scalar_adv_opt=1)
    calls=[]
    def capture(state,dt,grid,**kw):
        calls.append(kw);z=jnp.zeros_like(state.t_skin)
        return state,dict(rainncv=z,rain_liquid=z,hailncv=z,snowncv=z,grplncv=z,sr=z)
    monkeypatch.setattr(adapter,'nssl2mom_adapter',capture)
    op._apply_post_rk_microphysics(st,nml,step_index=jnp.int32(step),return_precipitation=True)
    assert len(calls)==1 and calls[0]['hail_enabled'] is False
    assert bool(calls[0]['first_step'])==(step==1)


def test_nohail_checkpoint_and_netcdf_roundtrip(tmp_path,monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state,read_wrfrst_state
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();g=f._grid(nz=8,ny=3,nx=4)
    nml=dataclasses.replace(f._namelist(g),mp_physics=22)
    st=f._state(g).ensure_conditional_leaves(mp_physics=22)
    present=('qvolg','Nc','Nr','Ni','Ns','Ng','Nn')
    st=st.replace(_cast=False,**{k:jnp.full_like(getattr(st,k),(i+1)/128) for i,k in enumerate(present)})
    write_checkpoint(st,nml,g,2,tmp_path/'nohail.pkl')
    a,na,_,step=read_checkpoint(tmp_path/'nohail.pkl')
    assert na.mp_physics==22 and step==2
    write_wrfrst_state(st,g,nml,tmp_path/'nohail.nc',valid_time='2026-07-25_18:00:36',
                       run_start='2026-07-25_18:00:00',step_index=2)
    b,_=read_wrfrst_state(tmp_path/'nohail.nc')
    for restored in (a,b):
        for k in present:
            x=np.asarray(getattr(restored,k));y=np.asarray(getattr(st,k))
            assert x.dtype==y.dtype and x.tobytes()==y.tobytes(),k
        for k in ('qh','Nh','qvolh','hail_acc'):assert getattr(restored,k) is None,k
