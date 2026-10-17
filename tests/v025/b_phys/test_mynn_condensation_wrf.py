"""Original QC-INOUT/EXIT on 1006 actual warm plume calls, REAL4 only."""
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
from gpuwrf.physics import mynn_edmf as E

HERE=Path(__file__).parent
DATA=json.loads((HERE/'fixtures/mynn_condensation_wrf.json').read_text())
FLAG='GPUWRF_MYNN_CONDENSATION_WRF'


def finite(**values):
    for name,x in values.items():assert np.isfinite(np.asarray(x)).all(),name+' nonfinite'


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv(FLAG,'1');jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope='session')
def original(tmp_path_factory):
    source=HERE.parents[2]/'data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F'
    helper=re.search(r'subroutine condensation_edmf\(QT,THL,P,zagl,THV,QC\).*?end subroutine condensation_edmf',source.read_text(),re.S|re.I).group(0)
    tokens=lambda text:re.sub(r'\s+','',re.sub(r'!.*','',text)).lower()
    assert hashlib.sha256(tokens(helper).encode()).hexdigest()==DATA['original_executable_statements_sha256']
    wrapper='''subroutine replay(n,inputs,outputs) bind(C)
use iso_c_binding
use module_bl_mynnedmf,only:condensation_edmf
implicit none
integer(c_int),value::n
real(c_float),intent(in)::inputs(5,n)
real(c_float),intent(out)::outputs(2,n)
integer::i
real(c_float)::qc,thv
do i=1,n
qc=inputs(5,i);thv=0.
call condensation_edmf(inputs(1,i),inputs(2,i),inputs(3,i),inputs(4,i),thv,qc)
outputs(1,i)=thv;outputs(2,i)=qc
enddo
end subroutine
'''
    root=tmp_path_factory.mktemp('bp113_condensation');(root/'oracle.f90').write_text(wrapper)
    phys=source.parent
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-ffree-line-length-none','-fPIC','-shared','-I'+str(phys),
                    'oracle.f90',str(phys/'module_bl_mynnedmf.o'),str(phys/'module_bl_mynnedmf_common.o'),'-o','oracle.so'],cwd=root,check=True,capture_output=True)
    lib=ctypes.CDLL(str(root/'oracle.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.replay.argtypes=[ctypes.c_int,ptr,ptr]
    def replay(inputs):
        inputs=np.ascontiguousarray(inputs,np.float32);outputs=np.empty((len(inputs),2),np.float32)
        finite(input=inputs);lib.replay(len(inputs),inputs,outputs);finite(reference=outputs)
        return outputs
    return replay


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['domain'])
def test_original_call_inputs_and_qc_exit(monkeypatch,row,original):
    from gpuwrf.physics import mynn_constants as MC
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    inputs=np.asarray(row['input'],np.float32);saved=np.asarray(row['output'],np.float32)
    finite(input=inputs,saved_reference=saved)
    reference=original(inputs)
    np.testing.assert_array_equal(reference,saved)
    def call(a):return E._condensation_edmf(a[:,0],a[:,1],a[:,2],a[:,3],**E._condensation_wrf_kwargs(a[:,4]))
    with jax.enable_x64(False):output=jax.jit(call)(jnp.asarray(inputs))
    actual=np.stack([np.asarray(x) for x in output],axis=-1)
    finite(candidate=actual,reference=reference)
    # REAL trig/pow/FMA conditioning: QC is a small subtraction of qt-qs.
    # Frozen pristine per-call bounds are absolute; no percentile tolerance.
    np.testing.assert_allclose(actual[:,0],reference[:,0],rtol=0,atol=2*np.spacing(np.float32(300)),err_msg='THV')
    np.testing.assert_allclose(actual[:,1],reference[:,1],rtol=0,atol=2e-8,err_msg='QC')
    assert np.any(inputs[:,4]>0) and np.any(reference[:,1]>0)


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['domain'])
def test_flag_off_ignores_incoming_qc_exactly(monkeypatch,row):
    monkeypatch.setenv(FLAG,'0')
    a=jnp.asarray(row['input'],jnp.float32)
    call=lambda extra:E._condensation_edmf(a[:,0],a[:,1],a[:,2],a[:,3],**extra)
    with jax.enable_x64(False):baseline=jax.jit(lambda:call({}))();supplied=jax.jit(lambda:call({'qc_initial':a[:,4]}))()
    for x,y in zip(baseline,supplied):finite(legacy=x,candidate=y);np.testing.assert_array_equal(x,y)
    assert E._condensation_wrf_kwargs(a[:,4])=={}


@pytest.mark.parametrize('niter',[1,16,50])
def test_original_cap_is_independent_of_legacy_iteration_knobs(monkeypatch,niter):
    monkeypatch.setenv('GPUWRF_MYNN_COND_NITER',str(niter))
    a=jnp.asarray(DATA['rows'][0]['input'],jnp.float32)
    with jax.enable_x64(False):
        expected=jax.jit(lambda:E._condensation_edmf(a[:,0],a[:,1],a[:,2],a[:,3],qc_initial=a[:,4],niter=50))()
        got=jax.jit(lambda:E._condensation_edmf(a[:,0],a[:,1],a[:,2],a[:,3],qc_initial=a[:,4],niter=niter))()
    for x,y in zip(got,expected):finite(candidate=x,reference=y);np.testing.assert_array_equal(x,y)


def test_every_plume_path_forwards_previous_qc():
    import inspect
    from gpuwrf.kernels import phys_mynn_plume as L,fused_edmf_plume as F
    assert '_condensation_wrf_kwargs(qc_p)' in inspect.getsource(E._dmp_scan)
    assert '_condensation_wrf_kwargs(qc_p)' in inspect.getsource(L._plume_level)
    assert '_condensation_wrf_kwargs(qc_p)' in inspect.getsource(F._plume_rise_kernel)
    for module in [L,F]:assert '**kwargs' in inspect.getsource(module._edmf_condensation)


@pytest.mark.parametrize('side',['candidate','reference','input'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_pristine_gate_rejects_nonfinite(side,bad):
    with pytest.raises(AssertionError,match='nonfinite'):finite(**{side:np.array([bad])})
