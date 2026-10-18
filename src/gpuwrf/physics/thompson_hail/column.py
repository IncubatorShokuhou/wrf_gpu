"""Compose mp38 with the existing Thompson single-pass source equations."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .native import TC as tc, AERO as aero
from . import entry_distribution, fall_speeds, sediment_hail, final_balance
from .context import HailContext


@jax.tree_util.register_pytree_node_class
class ThompsonHailColumnState(aero.ThompsonAeroColumnState):
    __slots__ = (*aero.ThompsonAeroColumnState.__slots__, "qvolg")

    def __init__(self, *args, qvolg, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvolg = qvolg


def _real(state):
    return state.replace(**{name: jnp.asarray(getattr(state, name), jnp.float32) for name in state.__slots__})


def _render(baseline, tendencies, dt, *, rho):
    """Apply WRF's held REAL tendencies to its original prognostic baseline."""
    values = {name: (getattr(baseline, name) + np.float32(dt) * value).astype(jnp.float32)
              for name, value in tendencies.items()}
    values["qv"] = jnp.maximum(np.float32(1.e-10), values["qv"])
    return baseline.replace(**values, rho=rho)


def _accumulate(tendencies, increments):
    for name, value in increments.items():
        tendencies[name] = (tendencies[name] + value).astype(jnp.float32)


def _saturation_adjustment(state, dt, tables, baseline, tendencies):
    """Execute the original saturation branch only where WRF enters it.

    WRF :3401 does not enter for inactive cloud and subsaturation, including
    a tiny negative tended qc from ordinary REAL cancellation. Its private
    working rc floor must not turn that inactive branch into condensation.
    """
    prw, pnc, lvap, ocp, active = aero._saturation_adjustment_aero(state, dt, tables, return_rates=True)
    _accumulate(tendencies, dict(qv=-prw, qc=prw, Nc=pnc, nwfa=-pnc, T=lvap*ocp*prw))
    adjusted = _render(baseline, tendencies, dt, rho=state.rho)
    rho = tc.density_from_pressure_temperature(state.p, adjusted.T, adjusted.qv)
    adjusted = adjusted.replace(rho=jnp.where(active, rho, state.rho))
    return adjusted, active & (prw > 0.)


def _working_moments(state):
    rho = state.rho
    # WRF :3215-3389 creates these private densities/moments once before
    # saturation and rain evaporation. Only rr/nr are refreshed by evaporation.
    rr, nr, lamr, ilamr, _mvd, n0_r, _active = tc._rain_distribution(state.qr, state.Nr, rho, hail_mode="post")
    return {"rr": rr, "nr": nr,
            "lamr": lamr, "ilamr": ilamr, "n0_r": n0_r, "rain_active": state.qr > tc.R1,
            "ri": jnp.where(state.qi > tc.R1, state.qi * rho, np.float32(tc.R1)),
            "ni": jnp.where(state.qi > tc.R1, jnp.maximum(np.float32(tc.R2), state.Ni * rho), np.float32(tc.R2)),
            "rs": jnp.where(state.qs > tc.R1, state.qs * rho, np.float32(tc.R1)),
            "g": entry_distribution(state.qg, state.Ng, state.qvolg, rho, rebalance=False),
            "snow_moments": tc._snow_moments(state.qs, rho, state.T-np.float32(273.15), tc.THOMPSON_TABLES, wrf_hail=True),
            "source_rho": rho, "rain_any": jnp.any(state.qr > tc.R1, axis=-1, keepdims=True),
            "visco": tc._air_properties(state)[2]}


def _sedimentation(state, dt, vts_boost, cloud_sed_on, cloud_rho, aerosol_tables, working, baseline, tendencies):
    # Reuse the shared WRF rain/ice/snow speed and explicit flux operators;
    # mp38 supplies only its density-dependent graupel channels and rb flux.
    working["rhof"] = jnp.sqrt(np.float32(tc.RHO_NOT) / jnp.where(working["rain_any"], state.rho, working["source_rho"]))
    speeds = tc._fall_speeds(state, vts_boost, working=working)
    vr, vnr, vi, vni, vs = (v.astype(jnp.float32) for v in speeds[:5])
    rho, dz = state.rho, state.dz
    # The qualified hail upwind operator has the same original REAL mass/
    # number tendency equations as rain, ice and snow. Its unused volume leg
    # is dead code for these three calls; no prognostic density is rebuilt.
    dummy_volume = np.float32(tc.R1) / rho / np.float32(800.)
    zero = jnp.zeros_like(state.qv)
    tendencies["qr"], tendencies["Nr"], _, rain = sediment_hail(working["rr"], working["nr"], dummy_volume, rho, dz, vr, vnr, dt,
        initial_tendencies=(tendencies["qr"], tendencies["Nr"], zero))
    tendencies["qi"], tendencies["Ni"], _, ice = sediment_hail(working["ri"], working["ni"], dummy_volume, rho, dz, vi, vni, dt,
        initial_tendencies=(tendencies["qi"], tendencies["Ni"], zero))
    tendencies["qs"], _, _, snow = sediment_hail(working["rs"], jnp.full_like(state.qs, tc.R2), dummy_volume, rho, dz, vs, vs, dt,
        initial_tendencies=(tendencies["qs"], zero, zero))
    g = working["g"]
    vg, vng = fall_speeds(g, rho, working["visco"], rhof=working["rhof"])
    tendencies["qg"], tendencies["Ng"], tendencies["qvolg"], graupel = sediment_hail(g.rg, g.ng, g.rb, rho, dz, vg, vng, dt,
        initial_tendencies=(tendencies["qg"], tendencies["Ng"], tendencies["qvolg"]))
    qrate, nrate, cloudw = aero._sed_cloud_water_aero(state, dt, aerosol_tables, cloud_sed_on, cloud_rho, return_rates=True)
    _accumulate(tendencies, dict(qc=qrate, Nc=nrate))
    out = _render(baseline, tendencies, dt, rho=rho)
    return out, {"rain": rain, "snow": snow, "ice": ice, "graupel": graupel, "cloudw": cloudw}


def _body(state, dt, hail_tables=None):
    raw = _real(state)
    active, skipped = aero._no_micro_column_skip(raw)
    skipped = skipped.replace(Ng=jnp.where(raw.qg > tc.R1, raw.Ng, 0.), qvolg=jnp.where(raw.qg > tc.R1, raw.qvolg, 0.))
    state = aero._clip_species_aero(raw)
    # WRF keeps NI1D as the prognostic baseline (:1851-1875), while its
    # private ni is floored/reconstructed independently. A negative carried
    # number therefore cancels through NITEN with the original REAL order;
    # clamping the baseline first changes the threshold stress witness.
    state = state.replace(Ni=raw.Ni)
    state = state.replace(rho=tc.density_from_pressure_temperature(state.p, state.T, state.qv))
    # Bound operands (tables.ThompsonHailTables) replace every baked DOUBLE
    # collision/freezing/rain-snow table; None keeps the baked host loaders.
    cold = tc._mixed_phase_cold_tables() if hail_tables is None else None
    ctx = HailContext(state, tc.THOMPSON_TABLES, cold, raw_qv=raw.qv, operands=hail_tables)
    state = state.replace(qvolg=ctx.g.qvolg)
    state, graupel_melt, vts_boost = tc._wrf_mixed_phase_sources(state, dt, tc.THOMPSON_TABLES, ctx.cold_tables, hail_context=ctx)
    baseline, tendencies = ctx.baseline, ctx.tendencies
    working = _working_moments(state)
    qc_pre, rho_pre, branch = state.qc, state.rho, tc._condensation_branch(state)
    qr_pre = state.qr
    held_lvap, held_ocp = tc._air_properties(state)[4:6]
    state, cloud_condensed = _saturation_adjustment(state, dt, ctx.aero, baseline, tendencies)
    state = _real(state)
    cloud_sed_on = tc._wrf_l_qc_any(qc_pre, rho_pre, state.qc, branch)
    evap_active = (state.qv / tc.saturation_mixing_ratio_liquid(state.p, state.T) - np.float32(1.) < -tc.EPS) & (working["rr"] > tc.R1) & ~cloud_condensed
    working["visco"] = jnp.where(evap_active, tc._air_properties(state)[2], working["visco"])
    rho_pre_evap = state.rho
    prv, pnr, lvap, ocp, evap_active = aero._rain_evaporation_aero(state, dt,
        skip_evaporation=cloud_condensed, graupel_melt=graupel_melt, working=working, return_rates=True)
    held_lvap, held_ocp = jnp.where(evap_active, lvap, held_lvap), jnp.where(evap_active, ocp, held_ocp)
    _accumulate(tendencies, dict(qr=-prv, qv=prv, Nr=-pnr, nwfa=pnr, T=-lvap*ocp*prv))
    state = _render(baseline, tendencies, dt, rho=rho_pre_evap)
    working["rr"] = jnp.where(evap_active, jnp.maximum(np.float32(tc.R1), state.qr * rho_pre_evap), working["rr"])
    working["nr"] = jnp.where(evap_active, jnp.maximum(np.float32(tc.R2), state.Nr * rho_pre_evap), working["nr"])
    # Original :3572 refreshes working density after rain evaporation, before
    # every fall-speed and sediment flux. The older aerosol helper retains
    # its incoming rho; final writeback is too late to refresh this stage.
    state = state.replace(rho=jnp.where(evap_active, tc.density_from_pressure_temperature(state.p, state.T, state.qv), rho_pre_evap))
    cloud_rho = tc._cloud_sed_rho_stages(rho_pre, qr_pre, state)
    state, precip = _sedimentation(state, dt, vts_boost, cloud_sed_on, cloud_rho, ctx.aero, working, baseline, tendencies)
    state = _real(state)
    # Original :3945-3967 changes cumulative tendencies and deliberately
    # retains the pre-instant thermodynamics for both conditional branches.
    odt = np.float32(1.) / np.float32(dt)
    xri = jnp.maximum(np.float32(0.), state.qi)
    melt = (state.T > np.float32(tc.T_0)) & (xri > 0.)
    melt_q, melt_n = jnp.where(melt, xri * odt, 0.), jnp.where(melt, baseline.Ni * odt, 0.)
    _accumulate(tendencies, dict(qc=melt_q, Nc=melt_n, qi=-melt_q, T=-np.float32(tc.LFUS)*held_ocp*melt_q))
    tendencies["Ni"] = jnp.where(melt, -baseline.Ni * odt, tendencies["Ni"])
    xrc = jnp.maximum(np.float32(0.), baseline.qc + tendencies["qc"] * np.float32(dt))
    freeze = (state.T < np.float32(tc.HGFR)) & (xrc > 0.)
    xnc = baseline.Nc + tendencies["Nc"] * np.float32(dt)
    freeze_q, freeze_n = jnp.where(freeze, xrc * odt, 0.), jnp.where(freeze, xnc * odt, 0.)
    _accumulate(tendencies, dict(qi=freeze_q, Ni=freeze_n, qc=-freeze_q, Nc=-freeze_n,
        T=(np.float32(tc.LSUB)-held_lvap)*held_ocp*freeze_q))
    state = _render(baseline, tendencies, dt, rho=state.rho)
    state = aero._finish_aero(state, ctx.aero, working_rho=state.rho)
    state = _real(state)
    qg, ng, qb = final_balance(state.qg, state.Ng, state.qvolg, state.rho)
    state = state.replace(qg=qg, Ng=ng, qvolg=qb)
    state = state.replace(**{name: jnp.where(active[..., None], getattr(state, name), getattr(skipped, name)).astype(jnp.float32) for name in state.__slots__})
    precip = {name: jnp.where(active, value, 0.).astype(jnp.float32) for name, value in precip.items()}
    return state, precip


@partial(jax.jit, static_argnames=("dt",))
def step_thompson_hail_column_with_precip(state, dt: float, hail_tables=None):
    """True aerosol/hail option; original full-caller qualification pending."""
    return _body(state, dt, hail_tables)
