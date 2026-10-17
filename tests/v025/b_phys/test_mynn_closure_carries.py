"""Warm P0227 operands against literal original MYNN REAL4 statements.

The primitive oracle owns the four new terms; it does not certify a forecast.
Existing level-2/length calculations are held identical on both sides.
"""
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.kernels import phys_mynn_columns as N
from gpuwrf.coupling import physics_couplers as C

HERE = Path(__file__).parent
DATA = json.loads((HERE/'fixtures/mynn_closure_carries_wrf.json').read_text())
KEYS = ['SHSM_FLOORS', 'TKEPROD_UP', 'ELH_BUDGET', 'RHOSFC_WRF']


def finite(**values):
    for name, x in values.items():
        if x is not None:
            assert np.isfinite(np.asarray(x)).all(), name+' nonfinite'


def column(row, original=False):
    fields = row['original_state' if original else 'state']
    return P.MynnPBLColumnState(**{k: None if v is None else jnp.asarray(v,jnp.float32)
                                  for k,v in fields.items()})


def flux(row):
    return SurfaceFluxes(**{k:None if v is None else jnp.asarray(v,jnp.float32)
                            for k,v in row['flux'].items()})


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    for key in KEYS: monkeypatch.setenv('GPUWRF_MYNN_'+key,'1')
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope='session')
def original(tmp_path_factory):
    source = HERE.parents[2]/'data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F'
    text = source.read_text()
    floor = re.search(r'       cldavg = half.*?       dfq\(k\) =     dfm\(k\)',text,re.S).group(0)
    tke = re.search(r'^\s*tkeprod_up\(k\)=.*$',text,re.M).group(0)
    density = re.search(r'^\s*rhosfc     =.*$',text,re.M).group(0)
    # Bind original declarations and constants, not Python decimal approximations.
    assert 'b1  = 24.0' in text
    constants = (source.parent.parent/'share/module_model_constants.F').read_text()
    assert re.search(r'r_d\s*=\s*287\.',constants)
    assert re.search(r'r_v\s*=\s*461\.6',constants)
    assert re.search(r'p608\s*=\s*rvovrd-1\.',constants)
    wrapper = '''subroutine budget(n,sm,sh,cldfra_bl1,edmf_a1,edmf_w1,el,qkw,gm,gh,dtl,dqw,div,tkeprod_up,dzk_in,pdk,pdt,pdq,pdc,dfm,dfh) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
integer,parameter::kind_phys=c_float
real(kind_phys),parameter::zero=0.,half=.5,one=1.
real(kind_phys),intent(inout)::sm(n),sh(n)
real(kind_phys),intent(in)::cldfra_bl1(n),edmf_a1(n),edmf_w1(n),el(n),qkw(n),gm(n),gh(n),dtl(n),dqw(n),div(n),tkeprod_up(n),dzk_in(n)
real(kind_phys),intent(out)::pdk(n),pdt(n),pdq(n),pdc(n),dfm(n),dfh(n)
real(kind_phys)::edmf_a_dd1(n),edmf_w_dd1(n),TKEprod_dn(n),tcd(n),qcd(n),dfq(n)
real(kind_phys)::cldavg,mfmax,elq,elh,qdiv,gamt,gamq,gamv,dzk
integer::k
edmf_a_dd1=0.;edmf_w_dd1=0.;TKEprod_dn=0.;gamt=0.;gamq=0.;gamv=0.
pdk=0.;pdt=0.;pdq=0.;pdc=0.;dfm=0.;dfh=0.
do k=2,n
qdiv=div(k)
dzk=dzk_in(k)
'''+floor+'''
enddo
end subroutine
subroutine production(n,edmf_a1,edmf_w1,el1,tkeprod_up) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),parameter::b1=24.
real(c_float),intent(in)::edmf_a1(n),edmf_w1(n),el1(n)
real(c_float),intent(out)::tkeprod_up(n)
integer::k
do k=1,n
'''+tke+'''
enddo
end subroutine
subroutine density(n,psfc_in,tk,qv,rho) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),parameter::r_d=287.,r_v=461.6,rvovrd=r_v/r_d,p608=rvovrd-1.
real(c_float),intent(in)::psfc_in(n),tk(n),qv(n)
real(c_float),intent(out)::rho(n)
real(c_float)::rhosfc,psfc
integer::i,kts
do i=1,n
kts=i
psfc=psfc_in(i)
'''+density+'''
rho(i)=rhosfc
enddo
end subroutine
'''
    root = tmp_path_factory.mktemp('bp113_original')
    (root/'oracle.f90').write_text(wrapper)
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran',
                    '-O2','-ffp-contract=off','-ffree-line-length-none','-fPIC','-shared',
                    'oracle.f90','-o','oracle.so'],cwd=root,check=True,capture_output=True)
    (root/'provenance.json').write_text(json.dumps(dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        statements_sha256=hashlib.sha256((floor+tke+density).encode()).hexdigest(),kind=4)))
    lib=ctypes.CDLL(str(root/'oracle.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.budget.argtypes=[ctypes.c_int]+[ptr]*20
    lib.production.argtypes=[ctypes.c_int]+[ptr]*4
    lib.density.argtypes=[ctypes.c_int]+[ptr]*4
    return lib


def captured_turbulence():
    """Return-only instrumentation of the actual implementation, no math edits."""
    body=textwrap.dedent(inspect.getsource(P._mym_turbulence))
    anchor='    if _env_bool("GPUWRF_MYNN_SHSM_FLOORS", False):'
    assert body.count(anchor)==1
    body=body.replace(anchor,'    _before_sm, _before_sh = sm, sh\n'+anchor)
    anchor='        "qkw": qkw,'
    assert body.count(anchor)==1
    body=body.replace(anchor,anchor+'\n        "before_sm": _before_sm, "before_sh": _before_sh, "sm": sm, "sh": sh,\n        "dtl": dtl, "dqw": dqw, "gm": gm, "gh": gh, "dzk": dzk,\n        "qdiv_eff": jnp.where(helfand, qdiv, 1.0),')
    scope=dict(vars(P));exec(compile(body,P.__file__,'exec'),scope)
    return scope['_mym_turbulence']


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['domain'])
def test_actual_turbulence_budget_matches_original_real4(row,original):
    state=column(row);sf=flux(row);mf={k:jnp.asarray(v,jnp.float32) for k,v in row['mf'].items()}
    kw=P._mynn_wrf_turbulence_kwargs(state,mf)
    # Primitive gates retain original REAL rounding at each operation. A fused
    # JIT can contract the unchanged base production and is measured separately.
    with jax.enable_x64(False), jax.disable_jit():
        got=jax.jit(lambda s:captured_turbulence()(s,2*s.tke,sf.fltv,sf.ustar,row['dx'],sf.xland,
                   **P._mynn_length_mf_kwargs(s,mf),**P._mynn_wrf_turbulence_kwargs(s,mf)))(state)
    finite(**got,**kw)
    names=['pdk','pdt','pdq','pdc','dfm','dfh','sm','sh']
    reference={name:np.zeros_like(np.asarray(got[name])) for name in names}
    for i in range(len(state.u)):
        inputs=[np.asarray(got[n])[i].copy() for n in ['before_sm','before_sh']]
        inputs += [np.asarray(x)[i].copy() for x in [state.cldfra_bl,mf['edmf_a'],mf['edmf_w']]]
        inputs += [np.asarray(got[n])[i].copy() for n in ['el','qkw','gm','gh','dtl','dqw','qdiv_eff']]
        inputs += [np.asarray(kw['tkeprod_up'])[i].copy(),np.asarray(got['dzk'])[i].copy()]
        outputs=[np.zeros(state.u.shape[-1],np.float32) for _ in range(6)]
        original.budget(state.u.shape[-1],*inputs,*outputs)
        for n,a in zip(names[:6],outputs):reference[n][i]=a
        reference['sm'][i]=inputs[0];reference['sh'][i]=inputs[1]
    finite(**reference)
    # The source production is checked separately; all per-operation REAL4
    # primitive budgets use exact bits on these finite original operands.
    for name in names:
        if name in ['sm','sh']:
            np.testing.assert_array_equal(np.asarray(got[name])[:,1:],reference[name][:,1:])
        else:
            np.testing.assert_array_equal(np.asarray(got[name])[:,1:],reference[name][:,1:],err_msg=name)
    assert np.any(np.asarray(got['qdiv_eff'])[:,1:]<.99),'real Helfand branch absent'
    assert np.any(np.asarray(got['sh'])[:,1:]>np.asarray(got['before_sh'])[:,1:]),'cloud/MF floor not exercised'


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['domain'])
def test_plume_tke_source_and_surface_density_are_original_real4(row,original):
    state=column(row);sf=flux(row);mf={k:jnp.asarray(v,jnp.float32) for k,v in row['mf'].items()}
    got=P._mynn_wrf_turbulence_kwargs(state,mf)
    assert set(got)=={'mf_aw','tkeprod_up'}
    reference=np.empty_like(np.asarray(state.el))
    for i in range(len(state.u)):
        args=[np.asarray(x)[i].copy() for x in [mf['edmf_a'],mf['edmf_w'],state.el]]
        original.production(state.u.shape[-1],*args,reference[i])
    density=np.empty(len(state.u),np.float32)
    tk=np.asarray(state.theta[:,0]*P._column_exner(state)[:,0],np.float32)
    args=[np.asarray(x,np.float32).copy() for x in [sf.psfc,tk,state.qv[:,0]]]
    original.density(len(state.u),*args,density)
    actual=P._surface_terms(state,sf)[3]
    finite(tke=got['tkeprod_up'],reference=reference,rho=actual,original_density=density)
    np.testing.assert_allclose(got['tkeprod_up'],reference,rtol=3e-7,atol=1e-18)
    np.testing.assert_array_equal(actual,density)
    assert np.any(reference>0) and np.any(np.asarray(actual)!=np.asarray(sf.rhosfc))


@pytest.mark.parametrize('entry',['retained','native'])
def test_both_actual_closure_calls_receive_plume_velocity_and_previous_el(monkeypatch,entry):
    entry=P._step_mynn_pbl_impl_with_pblh if entry=='retained' else N._advance_native
    row=DATA['rows'][0];state=column(row,True);sf=flux(row)
    mf={k:jnp.asarray(v,jnp.float32) for k,v in row['mf'].items()}
    expected=P._mynn_wrf_turbulence_kwargs(state,mf)
    monkeypatch.setattr(P,'_edmf_arrays_from_state',lambda *a,**k:mf)
    monkeypatch.setattr(N,'_native_edmf_arrays',lambda *a,**k:mf)
    class Captured(Exception):pass
    seen=[]
    def stop(*args,**kw):
        assert 'mf_aw' in kw and 'tkeprod_up' in kw,'closure consumer missing'
        for name in expected:
            finite(candidate=kw[name],reference=expected[name]);np.testing.assert_array_equal(kw[name],expected[name])
        seen.append(1);raise Captured()
    monkeypatch.setattr(P,'_mym_turbulence',stop)
    with jax.enable_x64(False),pytest.raises(Captured):entry(state,row['dt'],False,surface=sf,edmf=True,dx=row['dx'])
    assert seen==[1]


def test_actual_producer_passes_psfc_and_previous_el(monkeypatch):
    from test_mynn_held_diagnostics import fixture
    grid,_,state=fixture();nz,ny,nx=state.theta.shape
    row=DATA['rows'][0]
    previous=jnp.asarray(row['original_state']['el'][0][:nz],jnp.float32)[:,None,None]+jnp.zeros_like(state.theta)
    psfc=jnp.asarray(row['flux']['psfc'][0],jnp.float32)+jnp.zeros_like(state.t_skin)
    state=state.replace(el_pbl=previous)
    seen=[]
    def surface_view(st,gr):
        assert st is state and gr is grid;seen.append(1);return SimpleNamespace(psfc=psfc)
    monkeypatch.setattr(C,'_surface_column_view',surface_view)
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    sf=C._surface_fluxes_from_state(state,grid)
    assert seen==[1];np.testing.assert_array_equal(sf.psfc,psfc)
    actual=C._mynn_column_from_state(state,None)
    np.testing.assert_array_equal(actual.el,C._to_columns(previous))
    monkeypatch.setenv('GPUWRF_MYNN_TKEPROD_UP','0')
    assert not np.asarray(C._mynn_column_from_state(state,None).el).any()
    monkeypatch.setenv('GPUWRF_MYNN_RHOSFC_WRF','0')
    assert C._surface_fluxes_from_state(state,grid).psfc is None


def test_new_keys_off_have_no_optional_closure_leaves(monkeypatch):
    row=DATA['rows'][0];state=column(row);sf=flux(row)
    mf={k:jnp.asarray(v,jnp.float32) for k,v in row['mf'].items()}
    for key in KEYS:monkeypatch.setenv('GPUWRF_MYNN_'+key,'0')
    assert P._mynn_wrf_turbulence_kwargs(state,mf)=={}
    np.testing.assert_array_equal(P._surface_terms(state,sf)[3],sf.rhosfc)
    assert P._mynn_wrf_turbulence_kwargs(state,None)=={}


@pytest.mark.parametrize('sums',['0','1'])
def test_live_native_plume_exports_wrf_area_weighted_velocity(monkeypatch,sums):
    row=DATA['rows'][0];state=column(row,True);sf=flux(row)
    state=state.replace(**{k:None if getattr(state,k) is None else getattr(state,k)[:1]
                           for k in state.__slots__})
    sf=SurfaceFluxes(**{k:None if v is None else v[:1] for k,v in sf._asdict().items()})
    monkeypatch.setenv('GPUWRF_MYNN_PLUME_SUMS',sums)
    with jax.enable_x64(False):
        call=lambda s,f:N._native_edmf_arrays(s,f,f.fltv,P._get_pblh(s,2*s.tke,f.xland),row['dt'],row['dx'])
        result=jax.jit(call)(state,sf)
    assert 'edmf_w' in result,'missing transient WRF plume velocity'
    finite(**result)
    # Original DMP_mf :6489-6501: area is Psig-scaled, W is its area-weighted
    # mean; MAXMF is max(A*W), not density-weighted s_aw.
    expected=jnp.max(result['edmf_a']*result['edmf_w'],axis=-1)
    np.testing.assert_allclose(result['maxmf'],expected,rtol=3e-7,atol=0)
    assert np.any(np.asarray(expected)>0)
    for key in ['SHSM_FLOORS','TKEPROD_UP']:monkeypatch.setenv('GPUWRF_MYNN_'+key,'0')
    jax.clear_caches()
    with jax.enable_x64(False):off=jax.jit(call)(state,sf)
    assert 'edmf_w' not in off


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('name',['candidate','reference','input'])
def test_finite_gate_rejects_nonfinite(bad,name):
    with pytest.raises(AssertionError,match='nonfinite'):finite(**{name:np.array([bad])})
