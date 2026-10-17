"""Independent compiled pristine-WRF feedback operator gate (not a forecast)."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.boundary_feedback import (
    StateFeedbackWeights, apply_carry_feedback, apply_feedback,
    build_feedback_weights, build_feedback_smoother, sm121_smooth, smdsm_smooth,
    _feedback_leaf,
)
from gpuwrf.io.nesting_options import resolve_nesting_options

FIXTURE = Path(__file__).parent / "fixtures/u2_feedback_oracle.npz"
MANIFEST = json.loads(FIXTURE.with_suffix(".json").read_text())


@pytest.mark.parametrize("index,case", list(enumerate(MANIFEST["cases"])))
@pytest.mark.parametrize("compiled", (False, True))
def test_verbatim_wrf_three_feedback_epochs(index, case, compiled):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == MANIFEST["fixture_sha256"]
    with np.load(FIXTURE) as f:
        p, c = f[f"p{index}"], f[f"c{index}"]
        stagger = case["stagger"]
        weights = build_feedback_weights(
            parent_grid_ratio=case["ratio"], i_parent_start=3, j_parent_start=3,
            parent_we=p.shape[-1]-(stagger=="U"),parent_sn=p.shape[-2]-(stagger=="V"),
            child_we=c.shape[-1]-(stagger=="U"),child_sn=c.shape[-2]-(stagger=="V"),
            stagger=stagger,spec_zone=case["zone"])
        smoother = build_feedback_smoother(
            parent_grid_ratio=case["ratio"], i_parent_start=3,j_parent_start=3,
            child_we=c.shape[-1]-(stagger=="U"),child_sn=c.shape[-2]-(stagger=="V"),stagger=stagger)
        got = jnp.asarray(p)
        for step in range(MANIFEST["epochs"]):
            child = c if "field" in case else c+np.float32(step)*np.float32(0.125)
            if compiled:
                got = _feedback_leaf(got,jnp.asarray(child),weights,smoother,case["smooth"])
            else:
                got = apply_feedback(got,jnp.asarray(child),weights)
                if case["smooth"]:
                    got = (sm121_smooth if case["smooth"]==1 else smdsm_smooth)(got,smoother)
            actual, expected = np.asarray(got), f[f"o{index}_{step}"]
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            assert actual.dtype == expected.dtype == np.float32
            # Eight REAL ulps relative over three repeated smooth/desmooth calls;
            # no physical tolerance and no amplitude-dependent absolute floor.
            np.testing.assert_allclose(actual,expected,rtol=8*np.finfo(np.float32).eps,atol=0)


@dataclass(frozen=True)
class Duck:
    """Only the field mapping is mocked; all feedback operators are production."""
    def replace(self, _cast=False, **updates):
        return SimpleNamespace(**{**vars(self), **updates})


@dataclass(frozen=True)
class Carry:
    state: object
    base_state: object
    h_diabatic: object

    def replace(self, **updates):
        return replace(self, **updates)


def state(ny,nx,fill,base):
    mass = jnp.full((2,ny,nx),fill,dtype=jnp.float32)
    obj = Duck()
    fields = {name:mass for name in ("w","theta","qv","qc","qr","qi","qs","qg","Ni","Nr","qke","p_perturbation","ph_perturbation")}
    fields.update(u=jnp.full((2,ny,nx+1),fill,dtype=jnp.float32),v=jnp.full((2,ny+1,nx),fill,dtype=jnp.float32),
                  mu_perturbation=mass[0],p_total=mass+base,p=mass+base,
                  ph_total=mass+base,ph=mass+base,mu_total=mass[0]+base,mu=mass[0]+base)
    for key,val in fields.items():
        object.__setattr__(obj,key,val)
    return obj


def test_registry_pressure_base_and_unsmoothed_heating():
    from gpuwrf.contracts.state import BaseState
    from gpuwrf.runtime.domain_tree import _operational_feedback
    p, c = state(20,22,2,1000),state(30,36,8,2000)
    def base(s,value):
        return BaseState(pb=jnp.full_like(s.theta,value+20),phb=jnp.full_like(s.w,value),
                         mub=jnp.full_like(s.mu_total,value),t0=jnp.zeros_like(s.theta),theta_base=jnp.zeros_like(s.theta))
    common = dict(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,parent_we=22,parent_sn=20,child_we=36,child_sn=30)
    sm = dict(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,child_we=36,child_sn=30)
    weights = StateFeedbackWeights(**{name:build_feedback_weights(stagger=st,**common) for name,st in (("mass",""),("u","U"),("v","V"))},
        **{name:build_feedback_smoother(stagger=st,**sm) for name,st in (("mass_smooth",""),("u_smooth","U"),("v_smooth","V"))},smooth_option=2)
    with np.load(FIXTURE) as fixture:
        coarse, fine, expected = (jnp.asarray(fixture[key][0]) for key in ("p0","c0","o0_0"))
    for name in ("rain_acc","rainc_acc","snow_acc","graupel_acc","hail_acc"):
        object.__setattr__(p,name,coarse)
        object.__setattr__(c,name,fine)
    pc = Carry(state=p,base_state=base(p,100),h_diabatic=coarse[None])
    cc = Carry(state=c,base_state=base(c,200),h_diabatic=fine[None])
    out = _operational_feedback(SimpleNamespace(feedback_weights=weights),pc,cc)
    assert out.state.p_perturbation is p.p_perturbation
    assert np.max(np.asarray(out.base_state.pb)-120)>50
    np.testing.assert_array_equal(out.state.p_total,out.base_state.pb+p.p_perturbation)
    assert out.state.qke is p.qke
    assert np.max(np.asarray(out.base_state.phb)-100)>50
    np.testing.assert_array_equal(out.state.ph_total,out.base_state.phb+out.state.ph_perturbation)
    np.testing.assert_array_equal(out.state.mu_total,out.base_state.mub+out.state.mu_perturbation)
    np.testing.assert_allclose(out.h_diabatic,expected[None],rtol=2*np.finfo(np.float32).eps,atol=0)
    for name in ("rain_acc","rainc_acc","snow_acc","graupel_acc","hail_acc"):
        np.testing.assert_allclose(getattr(out.state,name),expected,rtol=2*np.finfo(np.float32).eps,atol=0)
    assert np.max(np.abs(np.asarray(expected)-np.asarray(smdsm_smooth(expected,weights.mass_smooth))))>0.1


def test_nesting_binding_defaults_overrides_and_validation():
    assert resolve_nesting_options({}) == dict(feedback=False,smooth_option=2,spec_zone=1)
    raw = {"domains":{"feedback":1,"smooth_option":0},"bdy_control":{"spec_zone":2}}
    assert resolve_nesting_options(raw) == dict(feedback=True,smooth_option=0,spec_zone=2)
    assert resolve_nesting_options(raw,feedback=False)["feedback"] is False
    assert resolve_nesting_options({},feedback=True)["feedback"] is True
    for key,val in (("feedback",2),("smooth_option",3),("smooth_option",1.5),("feedback",[0,1])):
        with pytest.raises(ValueError):
            resolve_nesting_options({"domains":{key:val}})


def test_real_nested_loader_binds_feedback_and_smoothing(monkeypatch,tmp_path):
    import test_namelist_binding as binding
    from gpuwrf.integration import nested_pipeline as pipeline
    loaded = []
    from gpuwrf.nesting import initial_feedback
    initialized=[]
    def init(*args,**kwargs):
        initialized.append(args)
        return {"test_phase":"initial feedback"}
    monkeypatch.setattr(initial_feedback,"initialize_two_way_domains",init)
    original = pipeline._load_domains
    def capture(*args,**kwargs):
        result = original(*args,**kwargs)
        loaded.append(result)
        return result
    monkeypatch.setattr(pipeline,"_load_domains",capture)
    binding._load_nested(monkeypatch,tmp_path,{"feedback":"1","smooth_option":"2","sst_update":"0"},
                         names=("d01","d02"))
    hierarchy,_,metadata,*_ = loaded[0]
    assert all(edge.feedback for edge in hierarchy.nests)
    assert metadata["nesting"]==dict(feedback=True,smooth_option=2,spec_zone=1)
    assert len(initialized)==1 and metadata["initial_feedback"]=={"test_phase":"initial feedback"}


def test_real_daily_loader_records_single_domain_feedback_controls(monkeypatch,tmp_path):
    import test_namelist_binding as binding
    from gpuwrf.integration import daily_pipeline as pipeline
    from gpuwrf.io.gen2_accessor import Gen2Run
    original_init = Gen2Run.__init__
    def init(run,*args,**kwargs):
        # Modify the lane-owned test namelist before the real accessor parses it.
        file = tmp_path/"namelist.input"
        file.write_text(file.read_text().replace(" max_dom = 1,", " max_dom = 1,\n feedback = 1,\n smooth_option = 0,"))
        original_init(run,*args,**kwargs)
    monkeypatch.setattr(Gen2Run,"__init__",init)
    original = pipeline._build_real_case
    loaded = []
    def capture(*args,**kwargs):
        result = original(*args,**kwargs)
        loaded.append(result)
        return result
    monkeypatch.setattr(pipeline,"_build_real_case",capture)
    binding._build_daily_case(monkeypatch,tmp_path," mp_physics=8,\n bl_pbl_physics=0,\n sf_surface_physics=0,\n ra_lw_physics=0,\n ra_sw_physics=0,\n sf_sfclay_physics=0,\n cu_physics=0,\n")
    assert loaded[0][0].metadata["nesting"]==dict(feedback=True,smooth_option=0,spec_zone=1,
        active=False,reason="single-domain driver; no child feedback")
