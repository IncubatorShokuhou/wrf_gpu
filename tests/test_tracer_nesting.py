"""Tracer nest-start/force-down/relax/final pin against original WRF operators."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.coupling import boundary_apply as bd
from gpuwrf.nesting.boundary_construction import build_child_boundary_package
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.tracers.runtime import frozen_tendencies, final_boundary, inherit_parent
from test_s1_nestspec import operands, pristine, assert_real, mass


@pytest.mark.parametrize("selection,override,domains,refused", [
    ({"tracer_opt": [2,0]}, None, ("d01","d02"), True),
    ({"tracer_opt": [0,2]}, None, ("d01","d02"), True),
    ({"gpuwrf_tracer_file": ["","tracers.json"]}, None, ("d01","d02"), True),
    ({"tracer_opt": [0,0]}, None, ("d01","d02"), False),
    ({"tracer_opt": [2,2]}, False, ("d01","d02"), False),
    ({"tracer_opt": 2}, True, ("d01",), False),
])
def test_actual_loader_refuses_unwired_tracer_feedback_before_arrays(
        tmp_path,monkeypatch,selection,override,domains,refused):
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io import gen2_accessor,namelist_check
    raw={"domains":{"feedback":1},"dynamics":selection}
    monkeypatch.setattr(gen2_accessor,"Gen2Run",lambda _:SimpleNamespace(namelist=raw))
    monkeypatch.setattr(namelist_check,"refuse_single_domain_mp_nested",lambda *a:None)
    monkeypatch.setattr(namelist_check,"refuse_unqualified_nested_options",lambda *a:None)
    monkeypatch.delenv("GPUWRF_BATCH_ENSEMBLE",raising=False)
    class LoadingArrays(Exception):pass
    def start_arrays(*args,**kwargs):raise LoadingArrays
    monkeypatch.setattr(pipeline,"build_replay_case",start_arrays)
    config=pipeline.NestedPipelineConfig(tmp_path,tmp_path/"out",tmp_path/"proof",hours=1,
        max_dom=len(domains),feedback=override)
    if refused:
        with pytest.raises(ValueError,match="two-way tracer feedback"):
            pipeline._load_domains(config,domains,_init_kernel_call=lambda *a,**k:None)
    else:
        with pytest.raises(LoadingArrays):
            pipeline._load_domains(config,domains,_init_kernel_call=lambda *a,**k:None)


@pytest.fixture(scope="module")
def tracer_operands(operands):
    parent,child,pm,cm,weights=operands
    # Existing signed, active spatial fields plus differing hybrid masses.
    pq=jnp.stack([parent.qh*(i+1)*100 for i in range(8)]).astype(jnp.float32)
    cq=jnp.stack([child.qh*(i+1)*50 for i in range(8)]).astype(jnp.float32)
    bdy=jnp.zeros((8,2,4,5,*child.qc_bdy.shape[-2:]),jnp.float32)
    parent=parent.replace(tracers=pq)
    child=child.replace(tracers=cq,tracer_bdy=bdy)
    cfg=bd.BoundaryConfig(update_cadence_s=18.,nested_frozen_wrf_boundary_bundle=True,force_geopotential=False)
    child_grid=cm_grid(child)
    grid=replace(child_grid,bc=replace(child_grid.bc,source="wrfbdy"))
    nl=OperationalNamelist.from_grid(grid,dt_s=6.,use_flux_advection=True)
    nl=replace(nl,metrics=cm,tracer_opt=2,diff_opt=0,bl_pbl_physics=0,boundary_config=cfg,
               run_boundary=True)
    forced=build_child_boundary_package(child,parent,weights,bdy_width=5,parent_metrics=pm,child_metrics=cm,
                                        coupled_forcedown=True,parent_grid_ratio=3)
    return parent,child,pm,cm,weights,nl,forced


def cm_grid(child):
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid
    return build_flat_grid(nx=child.qv.shape[2],ny=child.qv.shape[1],nz=child.qv.shape[0],dx_m=3000.)


@pytest.mark.parametrize("i",range(8))
def test_each_tracer_forcedown_original_wrf(tracer_operands,pristine,i):
    from pristine_forcedown import forcedown
    parent,child,pm,cm,_,_,forced=tracer_operands
    coarse=np.asarray(parent.tracers[i])*mass(parent,pm)
    fine=np.asarray(child.tracers[i])*mass(child,cm)
    old,rate=forcedown(pristine[0],coarse,fine,start=(6,7),ratio=3,cadence=18.,side_len=child.qv_bdy.shape[-1])
    rec=np.asarray(forced.tracer_bdy[i])
    assert_real(rec[0],old,ulps=1)
    assert_real(((rec[1]-rec[0]).astype(np.float64)/18).astype(np.float32),rate)
    np.testing.assert_array_equal(forced.tracers,child.tracers)


@pytest.mark.parametrize("lead",[0.,6.,18.])
def test_tracer_frozen_boundary_and_final_pin_original_wrf(tracer_operands,pristine,lead):
    from pristine_boundary import Oracle
    _,_,_,cm,_,nl,state=tracer_operands
    rates=frozen_tendencies(state,nl,muts=state.mu_total,xkhh=None,pbl_rates=None,lead_seconds=lead)
    pinned=final_boundary(state,nl,lead)
    oracle=Oracle(pristine[1],state.qv.shape,nested=True,dt=6.,dtbc=lead,dts=6.)
    oracle.set("mu",state.mu_total)
    oracle.set("c1h",cm.c1h);oracle.set("c2h",cm.c2h)
    for i,q in enumerate(state.tracers):
        rec=np.asarray(state.tracer_bdy[i])
        rate=((rec[1]-rec[0]).astype(np.float64)/18).astype(np.float32)
        oracle.set("scalar",q)
        oracle.set("scalar_tend",np.zeros(state.qv.shape,np.float32))
        oracle.boundary("scalar",rec[0],rate)
        oracle.run("relax_bdy_scalar")
        oracle.set("field_tend",oracle.field("scalar_tend",state.qv.shape))
        oracle.run("spec_bdytend",tag=6)
        assert_real(rates[i],oracle.field("field_tend",state.qv.shape))
        oracle.set("field",q)
        oracle.run("spec_bdy_final",tag=4)
        assert_real(pinned.tracers[i],oracle.field("field",state.qv.shape))
        np.testing.assert_array_equal(pinned.tracers[i,:,1:-1,1:-1],q[:,1:-1,1:-1])


def test_nest_start_constant_parent_inheritance(tracer_operands,tmp_path):
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid
    parent,child,_,_,_,nl,_=tracer_operands
    pg=build_flat_grid(nx=parent.qv.shape[2],ny=parent.qv.shape[1],nz=parent.qv.shape[0],dx_m=9000.)
    parent=parent.replace(tracers=jnp.stack([jnp.full_like(parent.qv,(i+1)/16) for i in range(8)]).astype(jnp.float32))
    meta=SimpleNamespace(parent_grid_ratio=3,i_parent_start=6,j_parent_start=7)
    result=inherit_parent(child,nl,parent,nl,pg,nl.grid,meta,{},"d02",tmp_path)
    for i,q in enumerate(result.tracers):
        np.testing.assert_allclose(q,(i+1)/16,rtol=2e-7)
    with pytest.raises(ValueError,match="same ordered"):
        inherit_parent(child,replace(nl,tracer_opt=0),parent,nl,pg,nl.grid,meta,{},"d02",tmp_path)
