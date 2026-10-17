"""All MYNN virtual-constant consumers share the existing WRF REAL switch."""
import ast
import ctypes
import inspect
import json
from pathlib import Path
import re
import subprocess
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import mynn_constants as MC, mynn_edmf as E, mynn_surface_stub as S
from test_mynn_qni import DATA, state_from_row


@pytest.fixture(autouse=True)
def caches():
    jax.clear_caches();yield;jax.clear_caches()


@pytest.fixture(scope='session')
def terminal_original(tmp_path_factory):
    source=Path(__file__).parents[3]/'data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F'
    body=source.read_text()
    statement=re.search(r'^\s*THV=\(THL\+xlvcp\*QC\).*$',body,re.M).group(0)
    common=(source.parent/'module_bl_mynnedmf_common.F').read_text()
    assert re.search(r'xlvcp\s*=\s*xlv/cp',common,re.I)
    model=(source.parent.parent/'share/module_model_constants.F').read_text()
    assert re.search(r'cp\s*=\s*7\.\*r_d/2\.',model)
    wrapper='''subroutine terminal(n,qt,thl,qc,thv) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),parameter::r_d=287.,r_v=461.6,rvovrd=r_v/r_d,cp=7.*r_d/2.,xlv=2.5e6,xlvcp=xlv/cp
real(c_float),intent(in)::qt(n),thl(n),qc(n)
real(c_float),intent(out)::thv(n)
integer::i
do i=1,n
'''+statement.replace('THV','thv(i)').replace('THL','thl(i)').replace('QT','qt(i)').replace('QC','qc(i)')+'''
enddo
end subroutine
'''
    root=tmp_path_factory.mktemp('bp113_virtual')
    (root/'original.f90').write_text(wrapper)
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-ffp-contract=off','-ffree-line-length-none','-fPIC','-shared','original.f90','-o','original.so'],cwd=root,check=True,capture_output=True)
    lib=ctypes.CDLL(str(root/'original.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.terminal.argtypes=[ctypes.c_int]+[ptr]*4
    return lib


def finite(**values):
    for name,x in values.items():assert np.isfinite(np.asarray(x)).all(),name+' nonfinite'


def test_plume_terminal_virtual_temperature_original_real4(monkeypatch,terminal_original):
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    st=state_from_row(DATA['rows'][1],jnp.float32)
    qt=st.qv/(jnp.float32(1)+st.qv)+st.qc/(jnp.float32(1)+st.qv)
    thl=st.theta-E.XLVCP*((st.qc/(1+st.qv))/((st.p/1e5)**E.RCP))
    # Isolate the original terminal statement with real operands and the QC
    # actually returned by the unchanged condensation iteration.
    with jax.enable_x64(False),jax.disable_jit():
        got,qc=E._condensation_edmf(qt,thl,st.p,jnp.cumsum(st.dz,axis=-1))
    args=[np.ascontiguousarray(x,np.float32).ravel() for x in [qt,thl,qc]]
    reference=np.empty_like(args[0]);terminal_original.terminal(len(reference),*args,reference)
    finite(candidate=got,reference=reference,qt=qt,thl=thl,qc=qc)
    np.testing.assert_array_equal(np.asarray(got).ravel(),reference)


def test_activation_uses_real_p608_at_actual_terminal(monkeypatch):
    st=state_from_row(DATA['rows'][1],jnp.float32)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    body=textwrap.dedent(inspect.getsource(E._wrf_superadiabatic_gate))
    anchor='    dthvdz0 =';assert body.count(anchor)==1
    body=body[:body.index(anchor)]+'    return tvs\n'
    scope=dict(vars(E));exec(compile(body,E.__file__,'exec'),scope)
    qv=st.qv[:,0];ts=st.theta[:,0];p608=np.float32(np.float32(461.6)/np.float32(287)-np.float32(1))
    expected=np.asarray(ts)*(np.float32(1)+p608*np.asarray(qv))
    with jax.enable_x64(False):got=scope['_wrf_superadiabatic_gate'](ts,ts,qv,st.dz[:,0],jnp.ones_like(qv),jnp.ones_like(qv))
    finite(candidate=got,reference=expected);np.testing.assert_array_equal(got,expected)


def test_neutral_hook_uses_existing_typed_constants(monkeypatch):
    st=state_from_row(DATA['rows'][1],jnp.float32);seen=[]
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    original=S.mynn_constant
    def capture(name,dtype,legacy=None):
        seen.append(name);return original(name,dtype,legacy)
    monkeypatch.setattr(S,'mynn_constant',capture)
    out=S.bulk_surface_fluxes(st.u[:,0],st.v[:,0],st.theta[:,0],st.qv[:,0],st.p[:,0])
    finite(**{k:v for k,v in out._asdict().items() if v is not None})
    assert seen==['R_D','P608']


@pytest.mark.parametrize('consumer',['condensation','activation'])
def test_actual_plume_virtual_constant_consumers(monkeypatch,consumer):
    st=state_from_row(DATA['rows'][1],jnp.float32);seen=[]
    original=MC.mynn_constant;monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
    def capture(name,dtype,legacy=None):
        result=original(name,dtype,legacy);seen.append((name,np.dtype(dtype),float(result)))
        return result
    monkeypatch.setattr(MC,'mynn_constant',capture)
    with jax.enable_x64(False):
        if consumer=='condensation':E._condensation_edmf(st.qv,st.theta,st.p,jnp.cumsum(st.dz,axis=-1),niter=1)
        else:E._wrf_superadiabatic_gate(st.theta[:,0],st.theta[:,0],st.qv[:,0],st.dz[:,0],jnp.ones_like(st.qv[:,0]),jnp.ones_like(st.qv[:,0]))
    expected='RVOVRD' if consumer=='condensation' else 'P608'
    assert len(seen)==1 and seen[0][:2]==(expected,np.dtype(np.float32))
    value=np.float32(461.6)/np.float32(287)
    if expected=='P608':value=value-np.float32(1)
    assert np.float32(seen[0][2]).view(np.uint32)==value.view(np.uint32)


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_flag_off_or_double_preserves_legacy_constants(monkeypatch,dtype):
    st=state_from_row(DATA['rows'][1],dtype)
    qt=st.qv;thl=st.theta;p=st.p;z=jnp.cumsum(st.dz,axis=-1)
    monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',False)
    off=E._condensation_edmf(qt,thl,p,z,niter=1)
    gate=E._wrf_superadiabatic_gate(thl[:,0],thl[:,0],qt[:,0],st.dz[:,0],jnp.ones_like(qt[:,0]),jnp.ones_like(qt[:,0]))
    stub=S.bulk_surface_fluxes(st.u[:,0],st.v[:,0],st.theta[:,0],qt[:,0],p[:,0])
    if dtype==jnp.float64:
        monkeypatch.setattr(MC,'_MYNN_REAL_CONSTANTS',True)
        on=E._condensation_edmf(qt,thl,p,z,niter=1)
        for a,b in zip(on,off):finite(candidate=a,legacy=b);np.testing.assert_array_equal(a,b)
        other=S.bulk_surface_fluxes(st.u[:,0],st.v[:,0],st.theta[:,0],qt[:,0],p[:,0])
        for a,b in zip(jax.tree.leaves(other),jax.tree.leaves(stub)):np.testing.assert_array_equal(a,b)
    assert gate.dtype==jnp.bool_


def test_every_mynn_p608_load_is_a_typed_helper_fallback():
    # Cover native/retained buoyancy, SGS thlv, PBLH, both DMP suppliers,
    # condensation, activation and the neutral surface hook. Native FULL/SUMS
    # and fused kernels delegate their condensation to this one shared helper.
    import gpuwrf.physics.mynn_pbl as P
    from gpuwrf.kernels import phys_mynn_columns as N,phys_mynn_plume as L,fused_edmf_plume as F
    functions=[P._virtual_potential,P._moist_virtual_potential,P._edmf_arrays_from_state,N._native_edmf_arrays,
               E._condensation_edmf,E._wrf_superadiabatic_gate,S.bulk_surface_fluxes]
    assert all('mynn_constant' in inspect.getsource(fn) for fn in functions)
    for mod in [L,F]:assert '_edmf._condensation_edmf' in inspect.getsource(mod._edmf_condensation)
    for module in [P,E,S,N]:
        tree=ast.parse(Path(module.__file__).read_text())
        for fn in [n for n in tree.body if isinstance(n,ast.FunctionDef)]:
            parents={id(child):parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
            for n in ast.walk(fn):
                if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in ['P608','RVOVRD']:
                    parent=parents[id(n)]
                    assert isinstance(parent,ast.Call) and 'mynn_constant' in ast.unparse(parent.func),(module.__name__,fn.name,n.lineno)
