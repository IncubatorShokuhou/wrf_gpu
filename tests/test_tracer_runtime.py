"""Independent conservation, analytic bulk-loss and production dispatch checks."""
from dataclasses import replace
from types import SimpleNamespace
import json
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State
from gpuwrf.runtime.operational_mode import OperationalNamelist, _stage_transport_velocities
from gpuwrf.tracers.config import TracerSpecies, UserTracerConfig, Emission
from gpuwrf.tracers.runtime import (names, stage_update, sources_and_sinks, layer_mass,
                                     frozen_tendencies, bind_inputs, validate_options)
from gpuwrf.validation.tier2 import make_ideal_grid


def case():
    grid = make_ideal_grid(8, 12, 12)
    nl = OperationalNamelist.from_grid(grid, dt_s=2., use_flux_advection=True,
        moist_adv_opt=1, scalar_adv_opt=1, diff_6th_opt=0)
    nl = replace(nl, tracer_opt=2, diff_opt=0, bl_pbl_physics=0, run_boundary=False)
    s = State.zeros(grid)
    shape = s.qv.shape
    s = s.replace(mu_total=jnp.full_like(s.mu_total, 90000.),
                  u=jnp.full_like(s.u, 4.), v=jnp.full_like(s.v, -2.),
                  ph_total=jnp.broadcast_to(jnp.arange(shape[0]+1)[:, None, None]*981., s.ph_total.shape))
    z, y, x = np.indices(shape)
    q = np.exp(-((x-5)**2+(y-7)**2)/7).astype(np.float32) * np.float32(1e-4)
    s = s.replace(tracers=jnp.stack([jnp.asarray(q)*(i+1) for i in range(8)]))
    return s, nl


@pytest.mark.parametrize("adv_opt", (1, 2))
def test_final_pd_preload_literal_original_wrf(tmp_path, monkeypatch, adv_opt):
    """The limiter receives pristine rk_update_scalar_pd's REAL numerator."""
    from gpuwrf.dynamics import flux_advection
    path = Path(__file__).parent / "v025/b_diff/pristine_scalar_rk.py"
    spec = importlib.util.spec_from_file_location("tracer_scalar_rk_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    library = oracle.build(tmp_path / "original_rk")
    state, nl = case()
    nl = replace(nl, tracer_adv_opt=adv_opt)
    rng = np.random.default_rng(291)
    q = rng.uniform(.1, 1., state.tracers.shape).astype(np.float32)
    frozen = rng.uniform(-3., 3., q.shape).astype(np.float32)
    state = state.replace(tracers=jnp.asarray(q))
    captured = []

    def observe(current, old, *args, **kwargs):
        captured.append(np.stack(old))
        return tuple(jnp.zeros_like(value) for value in current)

    monkeypatch.setattr(flux_advection, "advect_moisture_scalars", observe)
    with jax.disable_jit():
        stage_update(state, state, jnp.asarray(frozen), nl, rk_step=3, dt_rk=nl.dt_s,
                     velocities=_stage_transport_velocities(state, nl))
    expected = np.stack([oracle.pd_update(library, q=field, sc=source,
        mu1=np.asarray(state.mu_total, np.float32), mub=np.zeros(state.mu_total.shape, np.float32),
        c1=np.asarray(nl.metrics.c1h, np.float32), c2=np.asarray(nl.metrics.c2h, np.float32),
        dt=nl.dt_s, spec_zone=0, nested=False, specified=False)
        for field, source in zip(q, frozen, strict=True)])
    np.testing.assert_array_equal(captured[0], expected)


@pytest.mark.filterwarnings("error:overflow encountered:RuntimeWarning")
def test_active_tracer_admission_refuses_unwired_diffusion_and_nonfinite_clock():
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.tracers.runtime import seed_carry
    state, nl = case()
    carry = initial_operational_carry(state)
    with pytest.raises(ValueError, match="km_opt=4"):
        seed_carry(carry, replace(nl, diff_opt=1, km_opt=0, khdif=10.))
    for dt in (float("nan"), float("inf"), 1e300, 0., -1.):
        with pytest.raises(ValueError, match="positive time_step"):
            seed_carry(carry, replace(nl, dt_s=dt))


@pytest.mark.parametrize("pblmix,bl", [(0,5),(1,0),(0,0)])
def test_disabled_pbl_cannot_reuse_a_held_tracer_rate(pblmix,bl):
    state,nl=case()
    nl=replace(nl,tracer_pblmix=pblmix,bl_pbl_physics=bl)
    # WRF disables this source at dispatch even if a caller retains old memory.
    actual=frozen_tendencies(state,nl,muts=state.mu_total,xkhh=None,
        pbl_rates=jnp.ones_like(state.tracers),lead_seconds=0.)
    np.testing.assert_array_equal(actual,np.zeros(state.tracers.shape,np.float32))


@pytest.mark.parametrize("adv_opt", (0, 1, 2))
def test_resolved_transport_closes_closed_domain_mass(adv_opt):
    state, nl = case()
    nl = replace(nl, tracer_adv_opt=adv_opt)
    origin = state
    vel = _stage_transport_velocities(state, nl)
    frozen = jnp.zeros_like(state.tracers)
    for rk, fraction in ((1, 1/3), (2, 1/2), (3, 1)):
        state = stage_update(state, origin, frozen, nl, rk_step=rk, dt_rk=nl.dt_s*fraction, velocities=vel)
    before = np.asarray(origin.tracers, np.float64).sum(axis=(1,2,3))
    after = np.asarray(state.tracers, np.float64).sum(axis=(1,2,3))
    assert np.isfinite(after).all()
    np.testing.assert_allclose(after, before, rtol=3e-6, atol=0)
    assert not np.array_equal(state.tracers, origin.tracers), "tracers must actually move"


def test_bulk_losses_emission_overlap_and_mass_budget_independent():
    state, nl = case()
    cfg = UserTracerConfig((TracerSpecies("SMOKE", .02, .04, .3), TracerSpecies("DUST")),
        (Emission(0, .5, 1.5, 3,4,2,3, 2.,0.,True),
         Emission(1, 0., 9., 2,5,3,5, 0.,1e-5,False)), "a"*64)
    nl = replace(nl, tracer_opt=0, tracer_config=cfg)
    state = state.replace(tracers=state.tracers[:2], tracer_budget=jnp.zeros((2,4),jnp.float32),
                          rain_acc=jnp.full_like(state.rain_acc,.25))
    before = state.replace(rain_acc=jnp.zeros_like(state.rain_acc))
    out = sources_and_sinks(state,before,nl,jnp.float32(0))
    mass = np.asarray(layer_mass(state,nl),np.float64)
    q = np.asarray(state.tracers,np.float64)
    # Independent pressure-coordinate cell mass (uniform terrain/map in this
    # fixture) and independently assembled exponential/source law.
    mass_ref = ((np.asarray(nl.metrics.c1h)[:,None,None]*90000 + np.asarray(nl.metrics.c2h)[:,None,None])
        * -np.asarray(nl.metrics.dnw)[:,None,None] / 9.81 * 250.**2)
    np.testing.assert_allclose(mass,np.broadcast_to(mass_ref,mass.shape),rtol=2e-7)
    exponent = np.full(q.shape[1:],.02*2+.3*.25)
    exponent[0] += .04*2/100
    expect = q.copy()
    expect[0] *= np.exp(-exponent)
    expect[0,0,2,3] += 2 / mass[0,2,3]  # one second overlap, 2kg/s
    expect[1,0,3:5,2:5] += 2*1e-5*250**2 / mass[0,3:5,2:5]
    np.testing.assert_allclose(out.tracers,expect,rtol=4e-7,atol=1e-12)
    initial_mass = (q*mass[None]).sum(axis=(1,2,3))
    final_mass = (np.asarray(out.tracers)*mass[None]).sum(axis=(1,2,3))
    budget = np.asarray(out.tracer_budget,np.float64)
    np.testing.assert_allclose(final_mass, initial_mass+budget[:,0]-budget[:,1:].sum(axis=1),rtol=3e-6)
    np.testing.assert_allclose(budget[:,0],[2.,7.5],rtol=1e-7)
    assert (budget[0,1:] > 0).all()
    np.testing.assert_array_equal(budget[1,1:],0)


def test_tracer_config_survives_jax_pytree_roundtrip():
    _, nl = case()
    cfg = UserTracerConfig((TracerSpecies("SMOKE"),),(),"a"*64)
    nl = replace(nl,tracer_config=cfg,tracer_adv_opt=0,tracer_mix2_off=True,tracer_mix6_off=True,tracer_pblmix=0)
    leaves, structure = jax.tree.flatten(nl)
    restored = jax.tree.unflatten(structure,leaves)
    assert names(restored)==tuple(f"tr17_{i}" for i in range(1,9))+("SMOKE",)
    for field in ("tracer_opt","tracer_adv_opt","tracer_mix2_off","tracer_mix6_off","tracer_pblmix","tracer_config"):
        assert getattr(restored,field)==getattr(nl,field)


def test_namelist_honour_and_actual_binding(tmp_path):
    from gpuwrf.io.namelist_binding import bind_namelist_options, namelist_honour_report
    state,nl=case()
    path=tmp_path/"emit.json"
    path.write_text(json.dumps({"version":1,"species":[{"name":"SMOKE"}]}))
    raw={"dynamics":{"tracer_opt":0,"tracer_adv_opt":0,"tracer_mix2_off":True,
                     "gpuwrf_tracer_file":"emit.json"},"physics":{"tracer_pblmix":0}}
    errors,_=namelist_honour_report(raw,("d01",))
    assert errors==[]
    bound=bind_namelist_options(nl,raw,"d01")
    bound,s=bind_inputs(bound,state.replace(tracers=None),raw,"d01",tmp_path)
    assert names(bound)==("SMOKE",) and s.tracers.shape==(1,*state.qv.shape)
    assert bound.tracer_adv_opt==0 and bound.tracer_mix2_off and not bound.tracer_pblmix
    errors,_=namelist_honour_report({"dynamics":{"tracer_opt":3}},("d01",))
    assert any("WRF-Chem" in e for e in errors)


def test_custom_tracers_do_not_touch_weather_or_clip_negative_values():
    state,nl=case()
    cfg=UserTracerConfig((TracerSpecies("SMOKE"),),(),"a"*64)
    nl=replace(nl,tracer_opt=0,tracer_config=cfg)
    state=state.replace(tracers=-state.tracers[:1],tracer_budget=jnp.zeros((1,4),jnp.float32))
    result=sources_and_sinks(state,state,nl,0)
    for name in State.__slots__:
        if name in ("tracers","tracer_budget") or getattr(state,name) is None:
            continue
        np.testing.assert_array_equal(getattr(result,name),getattr(state,name))
    np.testing.assert_array_equal(result.tracers,state.tracers)


@pytest.mark.parametrize("source", ["decay", "area"])
@pytest.mark.parametrize("compiled", [False, True])
def test_accepted_json_source_overflow_is_refused_before_execution(tmp_path, source, compiled):
    from gpuwrf.tracers.config import load_config
    from gpuwrf.tracers.runtime import seed_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    state, nl = case()
    raw = {"version":1, "species":[{"name":"SMOKE"}], "emissions":[]}
    if source == "decay":
        raw["species"][0]["decay_rate_s"] = 3e38
    else:
        raw["emissions"] = [{"species":"SMOKE", "start_s":0, "end_s":2,
            "x0":2,"x1":3,"y0":2,"y1":3,"flux_kg_m2_s":1e35}]
    path = tmp_path / "overflow.json"
    path.write_text(json.dumps(raw))
    cfg = load_config(path, shape=state.qv.shape)  # Individually finite and parseable.
    nl = replace(nl, tracer_opt=0, tracer_config=cfg)
    state = state.replace(tracers=state.tracers[:1], tracer_budget=jnp.zeros((1,4),jnp.float32))
    with pytest.raises(ValueError, match="finite REAL32"):
        seed_carry(initial_operational_carry(state), nl)
    function = jax.jit(lambda value: sources_and_sinks(value,value,nl,0.)) if compiled else lambda value: sources_and_sinks(value,value,nl,0.)
    with pytest.raises(ValueError, match="finite REAL32"):
        function(state)


@pytest.mark.parametrize("overflow", ["mapped_area", "existing_budget", "combined_loss"])
def test_source_admission_resolves_geometry_and_accumulation(overflow):
    from gpuwrf.tracers.runtime import seed_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    state, nl = case()
    spec = TracerSpecies("SMOKE")
    emission = Emission(0,0.,2.,2,3,2,3,1e38,0.,True)
    if overflow == "mapped_area":
        emission = replace(emission, rate_kg_s=0., flux_kg_m2_s=1e28, is_point=False)
        nl = replace(nl, metrics=replace(nl.metrics,
            msftx=jnp.full_like(nl.metrics.msftx,.001), msfty=jnp.full_like(nl.metrics.msfty,.001)))
    elif overflow == "combined_loss":
        spec = TracerSpecies("SMOKE",1e38,1e38)
        emission = replace(emission,rate_kg_s=0.)
        state = state.replace(ph_total=jnp.broadcast_to(jnp.arange(state.qv.shape[0]+1)[:,None,None]*4.905,state.ph_total.shape))
    cfg = UserTracerConfig((spec,),(emission,),"a"*64)
    nl = replace(nl, tracer_opt=0, tracer_config=cfg)
    budget = jnp.zeros((1,4),jnp.float32)
    if overflow == "existing_budget":
        budget = budget.at[0,0].set(3e38)
    state = state.replace(tracers=state.tracers[:1],tracer_budget=budget)
    with pytest.raises(ValueError, match="finite REAL32"):
        seed_carry(initial_operational_carry(state), nl)


@pytest.mark.parametrize("rain_increment", [2., 3e38])
def test_large_finite_wet_exposures_have_finite_analytic_budget(rain_increment):
    state, nl = case()
    cfg = UserTracerConfig((TracerSpecies("SMOKE", wet_scavenging_per_mm=3e38),),(),"a"*64)
    nl = replace(nl,tracer_opt=0,tracer_config=cfg)
    before = state.replace(tracers=state.tracers[:1],tracer_budget=jnp.zeros((1,4),jnp.float32),
                           rain_acc=jnp.zeros_like(state.rain_acc),rainc_acc=jnp.zeros_like(state.rainc_acc))
    state = before.replace(rain_acc=jnp.full_like(state.rain_acc,rain_increment),
                           rainc_acc=jnp.full_like(state.rainc_acc,rain_increment))
    out = jax.jit(lambda current, previous: sources_and_sinks(current,previous,nl,0.))(state,before)
    assert np.isfinite(out.tracers).all() and np.isfinite(out.tracer_budget).all()
    np.testing.assert_array_equal(out.tracers,0.)
    expected = np.sum(np.asarray(before.tracers[0],np.float64)*np.asarray(layer_mass(before,nl),np.float64))
    np.testing.assert_allclose(out.tracer_budget[0,3],expected,rtol=3e-6)
    np.testing.assert_array_equal(out.tracer_budget[0,:3],0.)


def test_source_admission_reserves_loss_budget_for_future_emitted_inventory():
    from gpuwrf.tracers.runtime import seed_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    state, nl = case()
    maximum = float(np.finfo(np.float32).max)
    cfg = UserTracerConfig((TracerSpecies("SMOKE",decay_rate_s=.1),),
        (Emission(0,0.,2.,2,3,2,3,.3*maximum,0.,True),),"a"*64)
    nl = replace(nl,tracer_opt=0,tracer_config=cfg)
    mass = np.asarray(layer_mass(state,nl),np.float64)
    # Current inventory and all future emissions fit separately; their eventual
    # combined loss budget does not, and must be refused before any step.
    q = jnp.full((1,*state.qv.shape),.6*maximum/float(mass.sum()),jnp.float32)
    state = state.replace(tracers=q,tracer_budget=jnp.zeros((1,4),jnp.float32))
    with pytest.raises(ValueError,match="resolved loss budget.*finite REAL32"):
        seed_carry(initial_operational_carry(state),nl)
