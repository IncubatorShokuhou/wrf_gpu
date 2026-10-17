"""WRF post-LSM LAND Q2 cap: original primitive and live producer operand."""
import ctypes
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import noahmp_coupler as C

HERE=Path(__file__).parent
DATA=json.loads((HERE/'fixtures/land_q2_cap_wrf.json').read_text())
KEY='GPUWRF_LAND_Q2_CAP_POSTLSM'


def finite(**values):
    for name,x in values.items():assert np.isfinite(np.asarray(x)).all(),name+' nonfinite'


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    monkeypatch.setenv(KEY,'1');jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope='session')
def original(tmp_path_factory):
    source=HERE.parents[2]/'data/wrf_pristine/WRF/phys/module_surface_driver.F'
    text=source.read_text()
    body=re.search(r'IF \(XLAND\(I,J\).LT.1.5\) THEN\s+Q2\(i,j\) = MIN\(Q2\(i,j\),1.05\*QV_CURR\(i,1,j\)\)\s+END IF',text,re.I).group(0)
    wrapper='''subroutine cap(n,input_q2,input_qv,input_xland,out) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),intent(in)::input_q2(n),input_qv(n),input_xland(n)
real(c_float),intent(out)::out(n)
real(c_float)::q2(n,1),qv_curr(n,1,1),xland(n,1)
integer::i,j
q2(:,1)=input_q2;qv_curr(:,1,1)=input_qv;xland(:,1)=input_xland;j=1
do i=1,n
'''+body+'''
enddo
out=q2(:,1)
end subroutine
'''
    root=tmp_path_factory.mktemp('bp117_original');(root/'oracle.f90').write_text(wrapper)
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-fPIC','-shared','oracle.f90','-o','oracle.so'],cwd=root,check=True,capture_output=True)
    lib=ctypes.CDLL(str(root/'oracle.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.cap.argtypes=[ctypes.c_int]+[ptr]*4
    return lib


def arrays():
    rows=DATA['records'];q2=np.asarray([r['q2'] for r in rows],np.float32);qv=np.asarray([r['qv_frame_proxy'] for r in rows],np.float32)
    land=np.asarray([r['land'] for r in rows],bool)
    finite(q2=q2,qv=qv);return q2,qv,land


def test_pristine_postlsm_literal_real4(original):
    q2,qv,land=arrays();out=np.empty_like(q2);xland=np.where(land,1.,2.).astype(np.float32)
    original.cap(len(q2),q2,qv,xland,out)
    with jax.enable_x64(False):got=jax.jit(C._land_q2_postlsm)(jnp.asarray(q2),jnp.asarray(qv),jnp.asarray(land))
    finite(candidate=got,reference=out);np.testing.assert_array_equal(got,out)
    assert np.any(out[land]!=q2[land]),'real cap operands inactive'
    np.testing.assert_array_equal(np.asarray(got)[~land],q2[~land])


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_flag_off_is_identity(monkeypatch,dtype):
    q2,qv,land=arrays();q2=jnp.asarray(q2,dtype);monkeypatch.setenv(KEY,'0')
    got=C._land_q2_postlsm(q2,jnp.asarray(qv,dtype),jnp.asarray(land))
    assert got is q2


def scene():
    spec=importlib.util.spec_from_file_location('bp117_scene',HERE.parents[1]/'test_noahmp_coupler.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);assert module.HAVE_TABLES
    return module._build()


@pytest.mark.parametrize('history_kind',['history','land_history'])
def test_live_producer_caps_after_lsm_with_entry_qv(monkeypatch,history_kind):
    state,land,static,rad,clock=scene();q2,qv,mask=arrays()
    sample=np.flatnonzero(mask & (q2>np.float32(1.05)*qv))[0]
    # Same real cap operands in a logical producer scene; physical oracle above
    # remains independent. Stop before packing/serialization or subsequent RK.
    state=state.replace(qv=jnp.full_like(state.qv,qv[sample]))
    original_step=C.noah_mp_step;seen=[]
    def step(*args,**kw):
        ls,nm=original_step(*args,**kw)
        return ls,nm._replace(q2=jnp.full_like(nm.q2,q2[sample]))
    monkeypatch.setattr(C,'noah_mp_step',step)
    class Captured(Exception):pass
    def capture(actual,entry_qv,is_land):
        finite(actual=actual,entry=entry_qv)
        assert np.any(np.asarray(actual)[np.asarray(is_land)]>1.05*np.asarray(entry_qv)[np.asarray(is_land)]),'cap ran before Noah overwrite'
        np.testing.assert_array_equal(entry_qv,C._surface(state.qv))
        seen.append(1);raise Captured()
    monkeypatch.setattr(C,'_land_q2_postlsm',capture)
    with pytest.raises(Captured):C.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=90.,**{history_kind:True})
    assert seen==[1]


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('side',['candidate','reference','input'])
def test_finite_gate_rejects_nonfinite(bad,side):
    with pytest.raises(AssertionError,match='nonfinite'):finite(**{side:np.asarray([bad])})
