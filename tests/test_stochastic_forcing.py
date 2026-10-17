"""Native RK stochastic operands versus verbatim pristine WRF tendency routines."""
import ctypes
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.stochastic_driver import PatternCarry,StochasticCarry
from gpuwrf.coupling.stochastic_forcing import scalar_sources,momentum_sources


@pytest.fixture(scope="module")
def original_rhs(tmp_path_factory):
    dest=tmp_path_factory.mktemp("original_stochastic_rhs")
    source=Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_stoch.F").read_text()
    bodies=[]
    for name in ("update_stoch_ten","perturb_physics_tend"):
        match=re.search(r"^\s*subroutine "+name+r"\(.*?^\s*end subroutine "+name+r"\s*$",source,re.I|re.M|re.S)
        assert match is not None
        bodies.append(match.group(0))
    wrapper='''
subroutine rhs(nx,ny,nz,u,v,t,q,pu,pv,pt,p,mu,mub,c1,c2) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::nx,ny,nz
real(c_float),intent(inout)::u(nx,nz,ny),v(nx,nz,ny),t(nx,nz,ny),q(nx,nz,ny),p(nx,nz,ny)
real(c_float),intent(in)::pu(nx,nz,ny),pv(nx,nz,ny),pt(nx,nz,ny),mu(nx,ny),mub(nx,ny),c1(nz),c2(nz)
call update_stoch_ten(u,v,t,pu,pv,pt,mu,mub,c1,c2, &
 1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,nz,nz)
call perturb_physics_tend(.5,2.,p,u,v,t,q, &
 1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,nz,nz)
end subroutine
'''
    (dest/"oracle.f90").write_text("module original_stochastic_rhs\ncontains\n"+"\n".join(bodies)+wrapper+"end module\n")
    fc=os.environ.get("FC","<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")
    subprocess.run([fc,"-O2","-ffp-contract=off","-fPIC","-shared","-ffree-line-length-none",
                    "-fcheck=bounds","oracle.f90","-o","oracle.so"],cwd=dest,
                   env={**os.environ,"TMPDIR":str(dest)},capture_output=True,check=True,timeout=90)
    lib=ctypes.CDLL(str(dest/"oracle.so"))
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    lib.rhs.argtypes=[ctypes.c_int]*3+[ptr]*12
    lib.rhs.restype=None
    return lib.rhs


def test_native_rk_skebs_sppt_original_wrf(original_rhs):
    path=Path(__file__).parent/"fixtures/u2_stoch_real4.npz"
    with np.load(path) as f:
        a={k:np.asarray(f[k]) for k in f.files}
    nz,ny,nx=a["u"].shape
    # The real WRF initializer fills these outer mass ghosts explicitly
    # (module_initialize_real.F:3847/3863). Reproduce that CALLER, not an A2C mean.
    mu,mub=a["mu"].copy(),a["mub"].copy()
    for value in (mu,mub):
        value[:,-1]=value[:,-2]
        value[-1]=value[-2]
    order=lambda x:np.ascontiguousarray(x.transpose(1,0,2))
    originals=[order(a[k]) for k in ("u","v","theta","qv","u_pattern","v_pattern","theta_pattern","sppt_pattern")]
    original_rhs(nx,ny,nz,*originals[:4],*originals[4:7],originals[7],mu,mub,a["c1h"],a["c2h"])
    expected=[x.transpose(1,0,2) for x in originals[:4]]
    empty=jnp.zeros((0,0),jnp.float32)
    pattern=lambda f:PatternCarry(empty,empty,empty,jnp.float32(0),jnp.float32(1),f)
    stoch=StochasticCarry((pattern((jnp.asarray(a["u_pattern"]),jnp.asarray(a["v_pattern"]))),
        pattern(jnp.asarray(a["theta_pattern"])),pattern(jnp.clip(jnp.asarray(a["sppt_pattern"]),-1.,1.)),
        None,None,None,None),(None,)*6,None,jnp.zeros(7,jnp.int32),jnp.int32(1))
    metrics=SimpleNamespace(c1h=jnp.asarray(a["c1h"][:-1]),c2h=jnp.asarray(a["c2h"][:-1]))
    live_mu,live_mub=jnp.asarray(mu[:-1,:-1]),jnp.asarray(mub[:-1,:-1])
    def native(u,v,t,q):
        t,q=scalar_sources(t,q,stoch,live_mu,live_mub,metrics)
        u,v=momentum_sources(u,v,stoch,live_mu,live_mub,metrics,nz-1)
        return u,v,t,q
    actual=jax.jit(native)(jnp.asarray(a["u"][:-1,:-1,:]),jnp.asarray(a["v"][:-1,:,:-1]),
                           jnp.asarray(a["theta"][:-1,:-1,:-1]),jnp.asarray(a["qv"][:-1,:-1,:-1]))
    truth=(expected[0][:-1,:-1,:],expected[1][:-1,:,:-1],expected[2][:-1,:-1,:-1],expected[3][:-1,:-1,:-1])
    for candidate,reference in zip(actual,truth):
        np.testing.assert_allclose(candidate,reference,rtol=3.e-6,atol=1.e-6)
    assert all(np.isfinite(x).all() for x in actual)


def test_actual_rk_caller_keeps_observation_nudging_outside_sppt():
    """WRF module_first_rk_step_part2:834/855 before fddaobs_driver:952.

    Observational sources must enter the final moist-theta conversion but must
    not receive SPPT. This asserts the source caller's independent linear
    increment, including nonzero humidity coupling and native face winds.
    """
    from gpuwrf.runtime.operational_mode import _source_leaf_dry_tendencies,_RVRD_REAL
    shape=(3,4,5)
    arr=lambda value:jnp.full(shape,value,jnp.float32)
    metrics=SimpleNamespace(c1h=jnp.ones(3,jnp.float32),c2h=jnp.zeros(3,jnp.float32),precision="fp32")
    empty=jnp.zeros((0,0),jnp.float32)
    p=PatternCarry(empty,empty,empty,jnp.float32(0),jnp.float32(1),jnp.full((3,5,6),.6,jnp.float32))
    carry=StochasticCarry((None,None,p,None,None,None,None),(None,)*6,None,jnp.zeros(7,jnp.int32),jnp.int32(1))
    qv,theta=arr(.01),arr(280.)
    obs=(jnp.full((3,4,6),.031,jnp.float32),jnp.full((3,5,5),-.017,jnp.float32),arr(.04),arr(.0001))
    def call(observation):
        return _source_leaf_dry_tendencies(jnp.full((4,5),1000.,jnp.float32),arr(.01),
            None,None,None,None,qv,theta,metrics,jnp.float32,real_glue=True,obs_tendf=observation,
            stochastic=(carry,jnp.zeros((4,5),jnp.float32),jnp.full((4,5),1000.,jnp.float32)))
    before=call(None)
    def with_observation():
        result=call(obs)
        return result.t_tendf,result.ru_tendf,result.rv_tendf
    after=jax.jit(with_observation)()
    factor=np.float32(1)+np.float32(_RVRD_REAL)*np.asarray(qv)
    increment=factor*np.asarray(obs[2])+np.float32(_RVRD_REAL)*np.asarray(theta)/factor*np.asarray(obs[3])
    np.testing.assert_allclose(after[0],np.asarray(before.t_tendf)+increment,
        rtol=3.e-6,atol=1.e-6)
    np.testing.assert_array_equal(after[1],obs[0])
    np.testing.assert_array_equal(after[2],obs[1])
