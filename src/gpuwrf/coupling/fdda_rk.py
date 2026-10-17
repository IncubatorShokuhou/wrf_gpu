"""Grid FDDA (grid_fdda=1) source lifetime between ``fddagd`` and the RK step.

Mirrors the KF RK route (coupling/kf_rk.py): the carry holds UNCOUPLED
R*NDGDTEN at step boundaries; at physics time the driver may recompute them
(first_rk_step_part1 -> fddagd_driver, after the PBL driver), they are coupled
with the step-entry masses (calculate_phy_tend), restricted like phy_fg_ten and
summed into the RK tendencies BEFORE conv_t_tendf_to_moist; after the RK the held
copy is decoupled with the final masses (phy_prep_part2).

Inputs follow WRF phy_prep: th_phy = (t+t0)/(1.+R_v/R_d*qv) (dry), z_at_w =
(phb+ph)/g, u3d/v3d = step-entry winds, pblh = this step's PBL output, ht = HGT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.fdda_grid import (
    GridFddaInputs,
    couple_fdda_rates,
    fdda_driver_tendencies,
    finish_fdda_rates,
    initial_fdda_carry,
    owned_fdda_tendf,
)

_G = 9.81  # module_model_constants.F g (phy_prep z_at_w)
_RVRD = 461.6 / 287.0
_RVRD_REAL = np.float32(np.float32(461.6) / np.float32(287.0))


def grid_fdda_active(namelist) -> bool:
    return isinstance(getattr(namelist, "data_assimilation", None), GridFddaInputs)


def _real_glue() -> bool:
    from gpuwrf.kernels.dyn_carry_fp32 import real_all_enabled

    return real_all_enabled()


def fdda_dtype():
    """Glue precision: WRF REAL under the release REAL carry, fp64 on the legacy path."""

    return jnp.float32 if _real_glue() else jnp.float64


def seed_fdda_carry(namelist):
    """fddagdinit (non-restart) carry: zero rates, record 0 pair."""

    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs, initial_spectral_carry
    if isinstance(namelist.data_assimilation, SpectralFddaInputs):
        return initial_spectral_carry(namelist.data_assimilation, fdda_dtype())
    return initial_fdda_carry(namelist.data_assimilation, fdda_dtype(), with_history=True)


def _metrics(namelist, real_glue: bool):
    metrics = namelist.metrics
    if real_glue:
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics

        metrics = real_metrics(metrics)
    return metrics


def _phy_prep_inputs(state, base_state, dtype, real_glue: bool):
    """th_phy (dry) and z_at_w as WRF phy_prep forms them (module_big_step_utilities_em.F:4834,4869)."""

    rvrd = jnp.asarray(_RVRD_REAL if real_glue else _RVRD, dtype)
    theta = jnp.asarray(state.theta, dtype)
    th_phy = theta / (jnp.asarray(1.0, dtype) + rvrd * jnp.asarray(state.qv, dtype))
    if base_state is not None:
        ph_sum = jnp.asarray(base_state.phb, dtype) + jnp.asarray(state.ph_perturbation, dtype)
    else:
        ph_sum = jnp.asarray(state.ph_total, dtype)
    # (phb+ph)/g: fence g so XLA keeps a division (E123: x/const -> x*(1/const)).
    z_at_w = ph_sum / jax.lax.optimization_barrier(jnp.asarray(_G, dtype))
    return th_phy, z_at_w


def fdda_step_sources(fdda_carry, before, pblh, base_state, namelist, lead_seconds, *,
                      specified_or_nested: bool, periodic_x: bool, surface_aux=None, surface_roughness=None):
    """Driver + coupling at physics time; returns (coupled carry, owned (ru, rv, t, qv) sources)."""

    real_glue = _real_glue()
    dtype = jnp.float32 if real_glue else jnp.float64
    inputs = namelist.data_assimilation
    metrics = _metrics(namelist, real_glue)
    th_phy, z_at_w = _phy_prep_inputs(before, base_state, dtype, real_glue)
    mu_2 = jnp.asarray(before.mu_perturbation, dtype)
    if base_state is not None:
        mub = jnp.asarray(base_state.mub, dtype)
    else:
        mub = jnp.asarray(before.mu_total, dtype) - mu_2
    mut = mub + mu_2  # calculate_full
    itimestep = 1 + jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)).astype(jnp.int32)
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs
    if isinstance(inputs, SpectralFddaInputs):
        from gpuwrf.coupling.fdda_spectral_rk import spectral_step_sources
        return spectral_step_sources(
            fdda_carry, inputs, before, th_phy, pblh, jnp.asarray(namelist.grid.terrain_height, dtype),
            z_at_w, lead_seconds, itimestep, mu_2, mub, metrics, dtype,
            specified_or_nested=specified_or_nested, periodic_x=periodic_x)
    from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaInputs
    surface_kwargs = {}
    if isinstance(inputs,SurfaceGridFddaInputs):
        if surface_aux is None or surface_roughness is None:
            raise ValueError('grid_sfdda requires this step\'s SFCLAY REGIME and ZNT')
        if base_state is None:
            raise ValueError('grid_sfdda requires explicit WRF base-state pressure')
        p_phy = jnp.asarray(base_state.pb,dtype) + jnp.asarray(before.p_perturbation,dtype)
        rcp = np.float32(np.float32(287)/np.float32(1004.5)) if real_glue else 287/1004.5
        pi_phy = (p_phy / jax.lax.optimization_barrier(jnp.asarray(1e5,dtype)))**jnp.asarray(rcp,dtype)
        t_phy = th_phy*pi_phy
        regime = jnp.asarray(surface_aux[-1],dtype).reshape(pblh.shape)
        surface_kwargs['surface_context'] = (t_phy,p_phy,regime,jnp.asarray(surface_roughness,dtype))
    carry = fdda_driver_tendencies(
        fdda_carry, inputs, before.u, before.v, th_phy, before.qv, pblh,
        jnp.asarray(namelist.grid.terrain_height, dtype), z_at_w, lead_seconds, itimestep, dtype,
        p_phy=(jnp.asarray(base_state.pb, dtype) + jnp.asarray(before.p_perturbation, dtype)
               if base_state is not None else jnp.asarray(before.p_total, dtype)), **surface_kwargs)
    carry = couple_fdda_rates(carry, mut, mu_2, mub, metrics.c1h, metrics.c2h, dtype)
    owned = owned_fdda_tendf(carry, specified_or_nested=specified_or_nested, periodic_x=periodic_x)
    return carry, owned


def fdda_history(carry, diagnostics=None):
    """Expose the held FDDAGD history leaf at output boundaries only."""

    from gpuwrf.assimilation.fdda_grid import GridFddaCarry
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaCarry

    if not isinstance(carry, GridFddaCarry) or isinstance(carry, SpectralFddaCarry):
        return diagnostics
    if getattr(carry, "hfx_fdda", None) is None:
        raise ValueError("grid FDDA history is missing held HFX_FDDA (E78)")
    return dict(diagnostics or {}, HFX_FDDA=carry.hfx_fdda)


def fdda_finish(fdda_carry, final_mu_total, namelist):
    """phy_prep_part2: decouple the held rates with the end-of-RK mass (grid%muts)."""

    real_glue = _real_glue()
    dtype = jnp.float32 if real_glue else jnp.float64
    metrics = _metrics(namelist, real_glue)
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs
    if isinstance(namelist.data_assimilation, SpectralFddaInputs):
        from gpuwrf.coupling.fdda_spectral_rk import spectral_finish
        return spectral_finish(fdda_carry, final_mu_total, metrics, dtype)
    return finish_fdda_rates(fdda_carry, final_mu_total, metrics.c1h, metrics.c2h, dtype)
