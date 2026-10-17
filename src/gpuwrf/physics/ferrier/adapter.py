"""WRF FER_HIRES/FER_HIRES_ADVECT caller mapping for the retained State API."""
import jax.numpy as jnp

from gpuwrf.backends import pallas as pl
from .native import native_column


def ferrier_adapter(state, dt, grid=None, *, option, gid=1, return_precipitation=False):
    from gpuwrf.coupling.physics_couplers import (
        P0_PA, R_D_OVER_CP, _dry_theta_view, _theta_m_from_microphysics,
    )
    from gpuwrf.coupling.scan_adapters import _wrf_moist_prep_den_delz

    state = state.ensure_conditional_leaves(mp_physics=option)
    real = jnp.float32
    th = jnp.asarray(_dry_theta_view(state), real)
    p = jnp.asarray(state.p, real)
    qv = jnp.asarray(state.qv, real)
    pii = (p / jnp.asarray(P0_PA, real)) ** jnp.asarray(R_D_OVER_CP, real)
    # RHO_PHY/DZ8W are the post-RK moist_physics_prep_em arrays (solve_em.F:3750): DRY
    # rho = 1./(al+alb) (module_big_step_utilities_em.F:5542), dz8w from (ph+phb)/g (:5480/:5488).
    rho, dz = _wrf_moist_prep_den_delz(state, grid, real)
    rho, dz = jnp.asarray(rho, real), jnp.asarray(dz, real)
    extra = {"rimef": state.f_rimef_phy} if option == 5 else {"qrimef": state.qrimef}
    output = native_column(th, qv, state.qc, state.qr, state.qi, p, pii, rho, dz,
                           dt, option=option, gid=gid, interpret=not pl.cuda_native(), **extra)
    updates = {
        "theta": _theta_m_from_microphysics(th, state.qv, output["TH_OUT"], output["QV_OUT"], state.theta.dtype),
        "qv": output["QV_OUT"], "qc": output["QC_OUT"], "qr": output["QR_OUT"], "qi": output["QI_OUT"],
        "rain_acc": state.rain_acc + output["RAINNCV"].astype(state.rain_acc.dtype),
        "ferrier_sr": output["SR"],
    }
    if option == 5:
        updates.update(qt=output["QT_OUT"], f_ice_phy=output["F_ICE_PHY_OUT"],
                       f_rain_phy=output["F_RAIN_PHY_OUT"], f_rimef_phy=output["F_RIMEF_PHY_OUT"])
    else:
        updates["qrimef"] = output["QRIMEF_OUT"]
    result = state.replace(**updates)
    if return_precipitation:
        # Original Ferrier wrappers only update RAINNCV and SR, not SNOWNCV,
        # GRAUPELNCV or HAILNCV. Preserve the surface driver's actual explicit
        # MP_* inputs rather than inventing a snow accumulator from SR.
        zero = jnp.zeros_like(output["RAINNCV"])
        return result, {"rain": output["RAINNCV"], "snow": zero, "graupel": zero, "ice": zero, "sr": output["SR"]}
    return result
