"""Original mym_predict on real sea-Sc and night-land operands."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.kernels import phys_mynn_columns as N

HERE=Path(__file__).parent
DATA=json.loads((HERE/'fixtures/mynn_predict_saw_wrf.json').read_text())
KEY='GPUWRF_MYNN_PREDICT_SAW_FLOOR'


def finite(**values):
    for name,value in values.items():assert np.isfinite(np.asarray(value)).all(),name+' nonfinite'


def operands(row):
    array=lambda v:None if v is None else jnp.asarray(v,jnp.float32)
    return (P.MynnPBLColumnState(**{k:array(v) for k,v in row['state'].items()}),
            SurfaceFluxes(**{k:array(v) for k,v in row['flux'].items()}),
            {k:array(v) for k,v in row['turb'].items()},jnp.asarray(row['s_aw'],jnp.float32))


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setenv(KEY,'1');monkeypatch.setenv('GPUWRF_MYNN_TRIDIAG_REAL','1')
    jax.clear_caches();yield;jax.clear_caches()


@pytest.fixture(scope='session')
def original(tmp_path_factory):
    source=HERE.parents[2]/'data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F'
    text=source.read_text()
    predict=re.search(r'SUBROUTINE\s+mym_predict\b.*?END SUBROUTINE mym_predict',text,re.S|re.I).group(0)
    assert re.search(r'DO k=kts\+1,kte-1\s+kqdz\(k\) = MAX',predict)
    assert 'IF ( closure .GE. 3.0 )' in predict  # TSQ/COV only level >=3.
    wrapper='''subroutine predict(n,dt,ust,pmz,dz,rho,el,dfq,pdk_in,pdt_in,pdq_in,pdc_in,qke_in,qsq_in,saw,out) bind(C)
use iso_c_binding
use module_bl_mynnedmf,only:mym_predict
implicit none
integer(c_int),value::n
real(c_float),value::dt,ust,pmz
real(c_float),intent(in)::dz(n),rho(n),el(n),dfq(n),pdk_in(n),pdt_in(n),pdq_in(n),pdc_in(n),qke_in(n),qsq_in(n),saw(n+1)
real(c_float),intent(out)::out(n,4)
real(c_float)::pdk(n),pdt(n),pdq(n),pdc(n),qke(n),tsq(n),qsq(n),cov(n),saw_copy(n+1),sawqke(n+1),qwt(n),qdiss(n)
pdk=pdk_in;pdt=pdt_in;pdq=pdq_in;pdc=pdc_in;qke=qke_in;qsq=qsq_in;tsq=0.;cov=0.;saw_copy=saw;sawqke=0.
call mym_predict(1,n,2.6_c_float,dt,dz,ust,0._c_float,0._c_float,pmz,1._c_float,el,dfq,rho, &
 pdk,pdt,pdq,pdc,qke,tsq,qsq,cov,saw_copy,sawqke,0,qwt,qdiss,0)
out(:,1)=qke;out(:,2)=qsq;out(:,3)=tsq;out(:,4)=cov
end subroutine
subroutine floor(n,kdz,saw,out) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),intent(in)::kdz(n+1),saw(n+1)
real(c_float),intent(out)::out(n+1)
integer::k
out=kdz
do k=2,n-1
out(k)=MAX(out(k),0.5*saw(k))
out(k)=MAX(out(k),-0.5*(saw(k)-saw(k+1)))
enddo
end subroutine
'''
    root=tmp_path_factory.mktemp('bp116_original');(root/'oracle.f90').write_text(wrapper)
    phys=source.parent
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-ffree-line-length-none','-fPIC','-shared','-I'+str(phys),
                    'oracle.f90',str(phys/'module_bl_mynnedmf.o'),str(phys/'module_bl_mynnedmf_common.o'),'-o','oracle.so'],cwd=root,check=True,capture_output=True)
    (root/'binding.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        object_sha256=hashlib.sha256((phys/'module_bl_mynnedmf.o').read_bytes()).hexdigest(),closure=2.6,edmf_tke=0,tke_budget=0)))
    lib=ctypes.CDLL(str(root/'oracle.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.predict.argtypes=[ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float]+[ptr]*12
    lib.floor.argtypes=[ctypes.c_int]+[ptr]*3
    return lib


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['label'])
def test_original_predict_qke_qsq_real4(row,original):
    st,sf,turb,saw=operands(row);qke=2*st.tke
    with jax.enable_x64(False):
        actual_qke=P._mym_predict_qke(st,qke,turb,row['dt'],sf.ustar,sf,s_aw=saw)[0]
        actual_qsq=P._mym_predict_qsq(st,st.qsq,turb,row['dt'],s_aw=saw)
        pmz=P._pmz_surface(sf.fltv,sf.ustar,st.dz[:,0])
    finite(candidate_qke=actual_qke,candidate_qsq=actual_qsq,saw=saw)
    expected=np.empty((len(st.u),4,st.u.shape[-1]),np.float32)
    for i in range(len(st.u)):
        args=[np.ascontiguousarray(np.asarray(x)[i],np.float32) for x in [st.dz,st.rho,turb['el'],turb['dfq'],turb['pdk'],turb['pdt'],turb['pdq'],turb['pdc'],qke,st.qsq,saw]]
        finite(**{f'input{k}':v for k,v in enumerate(args)})
        original.predict(st.u.shape[-1],row['dt'],float(sf.ustar[i]),float(pmz[i]),*args,expected[i])
    finite(original=expected)
    # Existing BP56 REAL tridiagonal conditioning bound, scaled per profile;
    # no threshold chosen from the candidate's error.
    for name,actual,truth in [('qke',actual_qke,expected[:,0]),('qsq',actual_qsq,expected[:,1])]:
        scale=np.max(abs(truth),axis=-1,keepdims=True)+1e-37
        error=abs(np.asarray(actual,dtype=float)-truth.astype(float))/scale
        assert np.max(error)<=64*np.finfo(np.float32).eps,(name,float(np.max(error)))
    # At closure2.6 TSQ/COV are recomputed diagnostically (WRF:3331-3346),
    # without kmdz/previous moments. PDF2 consumes only QSQ on the next call.
    qkw=np.sqrt(np.maximum(np.asarray(qke),np.float32(0)))
    el=np.asarray(turb['el']);b2l=np.float32(15)*np.float32(.25)*(el[:,1:]+el[:,:-1])/qkw[:,:-1]
    for slot,name in [(2,'pdt'),(3,'pdc')]:
        prod=np.asarray(turb[name]).copy();prod[:,0]=prod[:,1]
        reference=b2l*(prod[:,1:]+prod[:,:-1]);reference=np.concatenate([reference,reference[:,-1:]],axis=-1)
        np.testing.assert_allclose(expected[:,slot],reference,rtol=5e-7,atol=1e-20)


def test_floor_matches_original_faces_exactly_and_is_active(original):
    row=DATA['rows'][0];st,sf,turb,saw=operands(row)
    with jax.enable_x64(False):before=P._rho_interfaces(st,3*turb['dfq']);after=P._apply_predict_saw_floor(before,saw)
    before=np.asarray(before);reference=np.empty_like(before)
    for i in range(len(st.u)):original.floor(st.u.shape[-1],before[i].copy(),np.asarray(saw)[i].copy(),reference[i])
    finite(input=before,candidate=after,reference=reference)
    np.testing.assert_array_equal(after,reference)
    assert np.any(reference!=before),'pristine floor not exercised on real plume columns'
    np.testing.assert_array_equal(reference[:,-2:],before[:,-2:])


@pytest.mark.parametrize('entry',['native','retained'])
def test_actual_predictor_call_receives_existing_plume_flux(monkeypatch,entry):
    row=DATA['rows'][0];st,sf,turb,saw=operands(row);mf={k:jnp.asarray(v,jnp.float32) for k,v in row['mf'].items()}
    monkeypatch.setattr(N,'_native_edmf_arrays',lambda *a,**kw:mf)
    monkeypatch.setattr(P,'_edmf_arrays_from_state',lambda *a,**kw:mf)
    class Captured(Exception):pass
    seen=[]
    def stop(*args,**kw):
        assert 's_aw' in kw,'missing predictor MF operand'
        finite(candidate=kw['s_aw'],reference=saw);np.testing.assert_array_equal(kw['s_aw'],saw)
        seen.append(1);raise Captured()
    monkeypatch.setattr(P,'_mym_predict_qke',stop)
    fn=N._advance_native if entry=='native' else P._step_mynn_pbl_impl_with_pblh
    with jax.enable_x64(False),pytest.raises(Captured):fn(st,row['dt'],False,surface=sf,edmf=True,dx=row['dx'])
    assert seen==[1]


def test_flag_off_ignores_new_operand_and_none_stays_empty(monkeypatch):
    row=DATA['rows'][0];st,sf,turb,saw=operands(row);qke=2*st.tke
    monkeypatch.setenv(KEY,'0')
    with jax.enable_x64(False):
        baseline=P._mym_predict_qke(st,qke,turb,row['dt'],sf.ustar,sf)
        supplied=P._mym_predict_qke(st,qke,turb,row['dt'],sf.ustar,sf,s_aw=saw)
    for x,y in zip(baseline,supplied):finite(baseline=x,supplied=y);np.testing.assert_array_equal(x,y)
    assert P._mym_predict_mf_kwargs({'s_aw':saw})=={}
    monkeypatch.setenv(KEY,'1');assert P._mym_predict_mf_kwargs(None)=={}


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('side',['input','candidate','reference'])
def test_numeric_gate_rejects_nonfinite(bad,side):
    with pytest.raises(AssertionError,match='nonfinite'):finite(**{side:np.asarray([bad])})
