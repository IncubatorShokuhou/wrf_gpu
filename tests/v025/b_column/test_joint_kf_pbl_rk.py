"""Joint shipping d01 KF+MYNN QV producer/mask/RK vs original WRF REAL4."""
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op
from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.diagnostics import census

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CASE=Path("<USER_HOME>/wrf_gpu2_lanes/wn3/V33/p0227/run")


@pytest.fixture(scope="module")
def literal(tmp_path_factory):
    s=importlib.util.spec_from_file_location("joint_literal",HERE/"pristine_joint_kf_pbl.py")
    w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
    return w,w.build(tmp_path_factory.mktemp("pristine_joint_kf_pbl"))


def finite(*a):
    assert all(np.isfinite(np.asarray(x)).all() for x in a)


def close(got,truth,addends):
    finite(got,truth,addends)
    bound=8*np.finfo(np.float32).eps*np.maximum(np.abs(addends).astype(np.float64),np.abs(truth))+1e-36
    err=np.abs(np.asarray(got,np.float64)-truth)
    bad=err>bound
    ix=np.unravel_index(np.argmax(err/bound),err.shape)
    assert not bad.any(), dict(cells=int(bad.sum()),index=ix,got=float(np.asarray(got)[ix]),truth=float(truth[ix]),bound=float(bound[ix]),max_abs=float(err.max()))
    return bound


@pytest.fixture(scope="module",params=["2026-02-28_12:00:00","2026-03-01_00:00:00"])
def operands(request):
    from netCDF4 import Dataset
    p=CASE/("wrfrst_d01_"+request.param)
    if not p.exists():pytest.skip("requires original P0227 d01 restart")
    with Dataset(p) as d:
        d.set_auto_mask(False)
        names=["QVAPOR","THM_2","MU_1","MU_2","MUB","C1H","C2H","MAPFAC_MX","MAPFAC_MY","RQVBLTEN","RTHBLTEN","RTHCUTEN","RQVCUTEN","RQCCUTEN","RQRCUTEN","RQICUTEN","RQSCUTEN","PRATEC","NCA"]
        o={n:np.asarray(d[n][0],np.float32) for n in names}
    finite(*o.values())
    assert np.any((o["RQVCUTEN"]!=0)&(o["RQVBLTEN"]!=0))
    o["receipt"]=dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    return o


def joint_producer(o,monkeypatch):
    s=importlib.util.spec_from_file_location("joint_schema",ROOT/"tests/test_rrtm_lw_operational_wiring.py")
    f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
    nz,ny,nx=o["QVAPOR"].shape
    grid=f._grid(ny=ny,nx=nx,nz=nz)
    metrics=dataclasses.replace(grid.metrics,c1h=jnp.asarray(o["C1H"],jnp.float64),c2h=jnp.asarray(o["C2H"],jnp.float64))
    grid=dataclasses.replace(grid,metrics=metrics,bc=dataclasses.replace(grid.bc,source="GFS"))
    state=f._state(grid).replace(_cast=False,theta=jnp.asarray(o["THM_2"]),qv=jnp.asarray(o["QVAPOR"]),
        mu=jnp.asarray(o["MU_1"]),mu_total=jnp.asarray(o["MU_1"]+o["MUB"]))
    nml=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),dt_s=54.,metrics=metrics,
        run_physics=True,use_noahmp=False,mp_physics=0,cu_physics=1,bl_pbl_physics=5,
        sf_sfclay_physics=0,rad_rk_tendf=1,run_boundary=True,boundary_config=BoundaryConfig(spec_zone=1),
        cumulus_cadence_steps=6)
    np.testing.assert_array_equal(nml.metrics.c1h,o["C1H"])
    np.testing.assert_array_equal(nml.metrics.c2h,o["C2H"])
    assert state.mu_total.dtype==state.qv.dtype==jnp.float32
    rates=tuple(jnp.asarray(o[n]) for n in ["RTHCUTEN","RQVCUTEN","RQCCUTEN","RQRCUTEN","RQICUTEN","RQSCUTEN","PRATEC"])
    carry=initial_operational_carry(state).replace(cumulus_carry=(jnp.zeros_like(state.theta),jnp.asarray(o["NCA"])),cumulus_tendencies=rates,census=census.initial_census())
    def mynn(st,*a,**kw):
        return SimpleNamespace(state=st.replace(qv=st.qv+54.*jnp.asarray(o["RQVBLTEN"])),
            rthblten=jnp.asarray(o["RTHBLTEN"]),rqvblten=jnp.asarray(o["RQVBLTEN"]),
            rqcblten=None,rqiblten=None,rublten=jnp.zeros_like(st.qv),rvblten=jnp.zeros_like(st.qv))
    monkeypatch.setattr(op,"mynn_adapter_with_source_leaves",mynn)
    monkeypatch.setattr(op,"kf_adapter",lambda st,dt,w,n,**kw:(rates,w,n))
    monkeypatch.setenv("GPUWRF_KF_TEND_RK_WRF","1");monkeypatch.setenv("GPUWRF_PHYS_TEND_RK_WRF","1")
    from gpuwrf.kernels import dyn_carry_fp32
    monkeypatch.setattr(dyn_carry_fp32,"real_all_enabled",lambda:True)
    forcing=op._physics_step_forcing(carry,nml,jnp.asarray(0.),run_radiation=False)
    # Source data and mass are original. Unrelated State schema/column-driver
    # arithmetic is replaced here; all production coupling/combination is real.
    np.testing.assert_array_equal(forcing.state.qv,state.qv)
    assert forcing.dry_tendencies.moist_tendf["qv"].dtype==jnp.float32
    return forcing.dry_tendencies.moist_tendf,nml


@pytest.mark.parametrize("pd",[False,True])
def test_joint_real_d01_source_mask_and_rk(operands,literal,monkeypatch,tmp_path,pd):
    o=operands;w,lib=literal
    physics,nml=joint_producer(o,monkeypatch)
    q=o["QVAPOR"];mu=o["MU_1"];mub=o["MUB"];mut=mu+mub;dt=54.
    mus=[(mu+np.float32(f)*(o["MU_2"]-mu)).astype(np.float32) for f in [1/3,.5,1.]]
    sc=np.zeros_like(q);adv=[np.zeros_like(q) for _ in range(3)]
    truth,source=w.chain(lib,q=q,pbl=o["RQVBLTEN"],kf=o["RQVCUTEN"],sc_other=sc,adv=adv,mu1=mu,mus=mus,mub=mub,mut=mut,c1=o["C1H"],c2=o["C2H"],msftx=o["MAPFAC_MX"],msfty=o["MAPFAC_MY"],dt=dt,pd=pd)
    masked=op._physics_sc_outside_spec(physics,("qv",),1,bounded=True)
    mass=o["C1H"][:,None,None]*mut[None]+o["C2H"][:,None,None]
    addend=np.maximum(np.abs(mass*o["RQVBLTEN"]),np.abs(mass*o["RQVCUTEN"]))
    close(np.asarray(masked["qv"]),source,addend)
    assert np.any(np.asarray(masked["qv"])!=0)
    assert not np.asarray(masked["qv"])[:,0,:].any() and not np.asarray(masked["qv"])[:,-1,:].any()
    assert not np.asarray(masked["qv"])[:,:,0].any() and not np.asarray(masked["qv"])[:,:,-1].any()
    class S(SimpleNamespace):
        def replace(self,**kw):return S(**(vars(self)|kw))
    qv_physics={"qv":masked["qv"]}
    origin=S(mu_total=jnp.asarray(mut),qv=jnp.asarray(q),qv_bdy=None)
    ns=SimpleNamespace(precision="fp32",c1h=jnp.asarray(o["C1H"]),c2h=jnp.asarray(o["C2H"]))
    for index,dtr in enumerate([dt/3,dt/2,dt]):
        pd_species=("qv",) if pd and index==2 else ()
        reference=op._rk_update_scalar_pd(origin,pd_species,(qv_physics,{"qv":sc}),dtr,ns) if pd_species else origin
        tendency=op._root_scalar_stage_tendencies((jnp.asarray(adv[index]),),("qv",),origin,
            nml.boundary_config,jnp.asarray(o["MAPFAC_MY"]),bounded=True,sc_tendencies={"qv":sc})
        tendency=op._with_physics_sc(("qv",),tendency,qv_physics,pd_species)
        stage=S(mu_total=jnp.asarray(mus[index]+mub),qv=jnp.asarray(q))
        actual=op._apply_moisture_large_step(stage,reference,q_tendencies=tendency,dt_rk=dtr,metrics=ns,species=("qv",))
    finalmass=o["C1H"][:,None,None]*(mus[-1]+mub)[None]+o["C2H"][:,None,None]
    addend_q=np.maximum(np.abs(q),dt*np.abs(source)/finalmass)
    bound=close(np.asarray(actual.qv),truth,addend_q)
    drops={}
    for name in ("pbl","kf"):
        kw=dict(q=q,pbl=o["RQVBLTEN"],kf=o["RQVCUTEN"],sc_other=sc,adv=adv,mu1=mu,mus=mus,mub=mub,mut=mut,c1=o["C1H"],c2=o["C2H"],msftx=o["MAPFAC_MX"],msfty=o["MAPFAC_MY"],dt=dt,pd=pd)
        kw[name]=np.zeros_like(q)
        wrong,_=w.chain(lib,**kw)
        finite(wrong);drops[name]=int(np.count_nonzero(np.abs(wrong.astype(np.float64)-truth)>bound))
        assert drops[name]>0
    (tmp_path/"JOINT_RECEIPT.json").write_text(json.dumps(dict(primary=o["receipt"],shape=list(q.shape),spec_zone=1,pd=pd,joint_cells=int(np.count_nonzero((o["RQVCUTEN"]!=0)&(o["RQVBLTEN"]!=0))),max_source_abs=float(np.max(np.abs(np.asarray(masked["qv"])-source))),max_scalar_abs=float(np.max(np.abs(np.asarray(actual.qv)-truth))),drop_source_over_bound=drops,scope="source-isolated original-rate forcing, zero advection; not a full physics-driver/forecast replay"),indent=2)+"\n")
