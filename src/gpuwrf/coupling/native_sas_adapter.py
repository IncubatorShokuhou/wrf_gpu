"""WRF ARW CU_KSAS/CU_NSAS caller and held REAL scalar/momentum rates.

The first seven leaves keep the existing cumulus precipitation/restart layout;
RU/RV follow them. Prognostic inputs are time-n phy_prep; surface flux/PBL
diagnostics are the just-completed drivers. HPBL_HOLD is the pre-PBL value,
not a new prognostic memory. These rates use the existing step-entry coupling
seam; coupled-forecast D6 qualification remains pending.
"""
from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    _dry_theta_view, _u_mass, _v_mass,
    _wrf_hydrostatic_pressure_profiles_from_state, _wrf_phy_prep_rho_from_state,
)
from gpuwrf.coupling.scalesas_adapter import initial_scalesas_tendencies
from gpuwrf.coupling.sas_consumers import apply_sas_scalar_rates, add_sas_momentum
from gpuwrf.physics.cumulus_ksas14 import cu_ksas14_columns
from gpuwrf.physics.cumulus_nsas96 import cu_nsas96_columns

REAL = jnp.float32


def initial_native_sas_tendencies(state):
    return initial_scalesas_tendencies(state) + (jnp.zeros(state.theta.shape, REAL),) * 2


def native_sas_rates(state, dt, grid, *, scheme, diagnostics=None, stepcu=1,
                     mp_physics=8, pgcon=.55, dx_factor_nsas=0):
    """Literal first_rk -> phy_prep -> CU_KSAS/CU_NSAS interface, REAL32."""
    if scheme not in (14, 96):
        raise ValueError("native SAS scheme must be 14 or 96")
    diag = state if diagnostics is None else diagnostics
    nz, ny, nx = state.theta.shape
    def cols(a):
        a = jnp.asarray(a, REAL)
        return jnp.moveaxis(a, 0, -1).reshape(ny * nx, a.shape[0])
    def surface(a):
        return jnp.zeros((ny * nx,), REAL) if a is None else jnp.asarray(a, REAL).reshape(ny * nx)
    def back(a):
        return jnp.moveaxis(a.reshape(ny, nx, nz), -1, 0)
    p = jnp.asarray(state.p, REAL)
    pi = (p / jnp.asarray(1.e5, REAL)) ** jnp.asarray(2./7., REAL)
    t = jnp.asarray(_dry_theta_view(state), REAL) * pi
    z = jnp.asarray(state.ph, REAL) / jnp.asarray(9.81, REAL)
    dz = z[1:] - z[:-1]
    metrics = grid.metrics
    p_hyd, p8w, _ = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=REAL)
    rho = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=REAL)
    w = jnp.asarray(state.w, REAL)
    if w.shape[0] == nz:
        w = jnp.concatenate((w, w[-1:]), axis=0)
    has_qc, has_qi = int(mp_physics) != 0, int(mp_physics) not in (0, 1, 3)
    zero = jnp.zeros(state.qv.shape, REAL)
    args = [cols(a) for a in (t, state.qv, state.qc if has_qc else zero,
            state.qi if has_qi else zero, p_hyd, pi, rho, dz,
            _u_mass(state), _v_mass(state), w)]
    common = dict(dt=float(dt), stepcu=int(stepcu), dx=float(grid.projection.dx_m),
                  mp_physics=int(mp_physics), literals="r4")
    xland, hpbl = surface(state.xland), surface(diag.pblh)
    if scheme == 14:
        eta = jnp.asarray(grid.eta_levels, REAL)
        znu = jnp.asarray(.5, REAL) * (eta[:-1] + eta[1:])
        out = cu_ksas14_columns(*args, xland, hpbl, surface(state.pblh), znu, **common)
    else:
        out = cu_nsas96_columns(*args, cols(p8w), xland, hpbl, surface(diag.hfx),
            surface(diag.qfx), row_width=nx, pgcon=float(pgcon),
            dx_factor_nsas=int(dx_factor_nsas), **common)
    # Registry absence is also enforced on outputs, like the optional WRF arguments.
    return (back(out["RTHCUTEN"]), back(out["RQVCUTEN"]),
            back(out["RQCCUTEN"]) if has_qc else zero, zero,
            back(out["RQICUTEN"]) if has_qi else zero, zero,
            out["PRATEC"].reshape(ny, nx), back(out["RUCUTEN"]), back(out["RVCUTEN"]))


ksas14_rates = partial(native_sas_rates, scheme=14)
nsas96_rates = partial(native_sas_rates, scheme=96)


def apply_native_sas_rates(state, rates, dt, *, entry_state, specified_or_nested=False,
                           periodic_x=False, periodic_y=False):
    out = apply_sas_scalar_rates(state, rates, dt, entry_state=entry_state,
        specified_or_nested=specified_or_nested, periodic_x=periodic_x)
    return add_sas_momentum(out, float(dt)*rates[7], float(dt)*rates[8],
        specified_or_nested=specified_or_nested, periodic_x=periodic_x, periodic_y=periodic_y)


def native_sas_cadence_step(entry_state, state, held, dt, grid, *, scheme, stepcu,
                            itimestep, mp_physics=8, pgcon=.55, dx_factor_nsas=0,
                            specified_or_nested=False, periodic_x=False, periodic_y=False):
    stepcu = max(1, int(stepcu))
    run = lambda _: native_sas_rates(entry_state, dt, grid, scheme=scheme,
        diagnostics=state, stepcu=stepcu, mp_physics=mp_physics, pgcon=pgcon,
        dx_factor_nsas=dx_factor_nsas)
    if held is None:
        rates = run(None)
    else:
        step = jnp.asarray(itimestep, jnp.int32)
        rates = jax.lax.cond((step == 1) | (jnp.mod(step, stepcu) == 0), run,
                             lambda _: tuple(jnp.asarray(x, REAL) for x in held), None)
    return (apply_native_sas_rates(state, rates, dt, entry_state=entry_state,
            specified_or_nested=specified_or_nested, periodic_x=periodic_x, periodic_y=periodic_y), None if held is None else rates)
