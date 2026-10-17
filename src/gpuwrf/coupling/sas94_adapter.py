"""cu_physics=94 State coupling, WRF CU_SAS ARW deep + shallow convection.

Reads time-n phy_prep inputs; cumulus_driver first-step/STEPCU cadence; held
REAL RTH/RQV/RQC/RQI/PRATEC rates in the existing seven-slot cumulus carry.
ARW phy_cu_ten does not couple SAS momentum rates. Uses the existing scale-SAS
increment application seam; coupled-forecast qualification remains pending.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    _dry_theta_view,
    _output_dtype,
    _theta_m_tendency_from_dry,
    _u_mass,
    _v_mass,
    _wrf_hydrostatic_pressure_profiles_from_state,
    _wrf_phy_prep_rho_from_state,
)
from gpuwrf.physics.cumulus_sas94 import cu_sas94_columns
from gpuwrf.coupling.scalesas_adapter import (
    _held_layout, initial_scalesas_tendencies as initial_sas94_tendencies,
)
from gpuwrf.coupling.sas_consumers import apply_sas_scalar_rates as apply_sas94_rates

_REAL = jnp.float32
_G_WRF = 9.81
_RCP_WRF = 2.0 / 7.0


def sas94_rates(state, dt: float, grid=None, *, stepcu: int = 1, pgcon: float = .55, mp_physics: int = 8):
    """Run ``CU_SAS`` on every column of ``state``; return WRF REAL driver outputs.

    Returns ``(RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, PRATEC)`` with the 3-D rates shaped
    ``(nz, ny, nx)`` and ``PRATEC`` ``(ny, nx)`` (mm/s), all float32 like WRF.
    """

    nz, ny, nx = state.theta.shape
    theta_dry = jnp.asarray(_dry_theta_view(state), _REAL)
    p_full = jnp.asarray(state.p, _REAL)
    pi_phy = (p_full / jnp.asarray(1.0e5, _REAL)) ** jnp.asarray(_RCP_WRF, _REAL)
    t_phy = theta_dry * pi_phy
    z_at_w = jnp.asarray(state.ph, _REAL) / jnp.asarray(_G_WRF, _REAL)
    dz8w = z_at_w[1:] - z_at_w[:-1]
    metrics = getattr(grid, "metrics", None)
    if metrics is not None:
        p_hyd, p_hyd_w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=_REAL)
        psfc = jnp.asarray(p_hyd_w, _REAL)[0]
        rho = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=_REAL)
    else:  # analytic callers without hybrid metrics: full pressure, EOS total density
        p_hyd = p_full
        psfc = p_full[0] + 0.5 * (p_full[0] - p_full[1])
        qv = jnp.asarray(state.qv, _REAL)
        rho = p_full / (jnp.asarray(287.0, _REAL) * t_phy * (1.0 + jnp.asarray(0.608, _REAL) * qv)) * (1.0 + qv)
    w = jnp.asarray(state.w, _REAL)
    if w.shape[0] == nz:  # mass-level w fallback: pad a top face
        w = jnp.concatenate([w, w[-1:]], axis=0)

    def cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(jnp.asarray(field3d, _REAL), 0, -1).reshape(ny * nx, field3d.shape[0])

    # Registry.EM_COMMON: passive qv has no QC/QI, Kessler and WSM3 no QI.
    # CU_SAS only assigns RQCCUTEN/RQICUTEN for allocated species (P_Q* >=
    # PARAM_FIRST_SCALAR). Dummy absent-species inputs are zero in WRF.
    has_qc = int(mp_physics) != 0
    has_qi = int(mp_physics) not in (0,1,3)
    zero = None if has_qc and has_qi else jnp.zeros(state.qv.shape,dtype=_REAL)
    out = cu_sas94_columns(
        cols(t_phy), cols(state.qv), cols(state.qc if has_qc else zero),
        cols(state.qi if has_qi else zero), cols(p_hyd), cols(pi_phy),
        cols(rho), cols(dz8w), cols(_u_mass(state)), cols(_v_mass(state)), cols(w),
        jnp.asarray(psfc, _REAL).reshape(ny * nx), jnp.asarray(state.xland, _REAL).reshape(ny * nx),
        float(dt), int(stepcu), pgcon=pgcon, literals="r4",
    )

    def back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    return (back(out["RTHCUTEN"]), back(out["RQVCUTEN"]),
            back(out["RQCCUTEN"]) if has_qc else zero,
            back(out["RQICUTEN"]) if has_qi else zero,
            out["PRATEC"].reshape(ny,nx))


def sas94_cadence_step(entry_state, state, held, dt: float, grid, *, stepcu: int, itimestep,
                       specified_or_nested=False, periodic_x=False,
                       pgcon: float = .55, mp_physics: int = 8):
    """WRF cumulus_driver gate on the one-based ``itimestep``; returns ``(state, held_rates)``.

    ``entry_state`` is the time-n physics entry (what every WRF physics driver reads);
    ``state`` is the slot input the held rates are applied to. ``held=None`` (carry without a
    seeded cumulus leaf, e.g. single-step harnesses) runs the scheme unconditionally.
    """

    if held is None:
        rates = _held_layout(sas94_rates(entry_state,dt,grid,stepcu=stepcu,pgcon=pgcon,mp_physics=mp_physics))
        return apply_sas94_rates(state, rates, dt, entry_state=entry_state, specified_or_nested=specified_or_nested, periodic_x=periodic_x), None

    stepcu = max(1, int(stepcu))
    step = jnp.asarray(itimestep, dtype=jnp.int32)
    run_cu = (step == 1) | (jnp.mod(step, stepcu) == 0)
    rates = jax.lax.cond(
        run_cu,
        lambda _u: _held_layout(sas94_rates(entry_state,dt,grid,stepcu=stepcu,pgcon=pgcon,mp_physics=mp_physics)),
        lambda _u: tuple(jnp.asarray(h, _REAL) for h in held),
        None,
    )
    return apply_sas94_rates(state, rates, dt, entry_state=entry_state, specified_or_nested=specified_or_nested, periodic_x=periodic_x), rates
