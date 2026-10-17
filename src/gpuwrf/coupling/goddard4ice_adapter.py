"""Original WRF mp7 caller mapping and Registry-held diagnostics."""
import jax.numpy as jnp

from gpuwrf.backends import pallas as pl
from gpuwrf.contracts.precision import GODDARD4ICE_CARRY_LEAVES


def goddard4ice_adapter(state, dt, grid=None, *, step_index=1, use_mp_re=1,
                       ra_lw_physics=4, ra_sw_physics=4, return_precipitation=False):
    from gpuwrf.coupling.physics_couplers import (
        P0_PA, R_D_OVER_CP, _dry_theta_view, _theta_m_from_microphysics,
    )
    from gpuwrf.coupling.scan_adapters import _wrf_moist_prep_den_delz
    from gpuwrf.physics.goddard4ice.native import native_column

    state = state.ensure_conditional_leaves(mp_physics=7, use_mp_re=use_mp_re,
        ra_lw_physics=ra_lw_physics, ra_sw_physics=ra_sw_physics)
    real = jnp.float32
    th = jnp.asarray(_dry_theta_view(state), real)
    p, qv = jnp.asarray(state.p, real), jnp.asarray(state.qv, real)
    exner = (p / jnp.asarray(P0_PA, real)) ** jnp.asarray(R_D_OVER_CP, real)
    # RHO/DZ8W/Z are the post-RK moist_physics_prep_em arrays (solve_em.F:3750): DRY
    # rho = 1./(al+alb) (module_big_step_utilities_em.F:5542), dz8w and z from (ph+phb)/g.
    rho, dz = _wrf_moist_prep_den_delz(state, grid, real)
    rho, dz = jnp.asarray(rho, real), jnp.asarray(dz, real)
    gravity = jnp.asarray(9.81, real)  # module_model_constants.F: g
    zw = jnp.asarray(state.ph, real) / gravity
    z = jnp.asarray(.5, real) * (zw[:-1] + zw[1:])
    w = jnp.asarray(state.w[:-1], real)  # solve_em.F:3904 passes W_2 bottom faces directly
    dx = 3000. if grid is None else float(grid.projection.dx_m)
    topo = jnp.zeros_like(state.xland) if grid is None else jnp.asarray(grid.terrain_height, real)
    diagnostics = {name: getattr(state, name) for name in GODDARD4ICE_CARRY_LEAVES if name.startswith("acphys")}
    output = native_column(th, qv, state.qc, state.qr, state.qi, state.qs, state.qg, state.qh,
        rho, exner, p, dz, z, w, dt, dx=dx, xland=state.xland, topo=topo,
        itimestep=step_index, diagnostics=diagnostics, interpret=not pl.cuda_native())
    updates = {name: output[name] for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
    updates.update({name: output[name] for name in GODDARD4ICE_CARRY_LEAVES if name != "goddard_sr"})
    updates["phys_tot"] = output["phys_tot"]
    if state.re_cloud is not None:
        updates.update({name: output[name + "_gsfc"] * jnp.float32(1.e-6)
                        for name in ("re_cloud", "re_ice", "re_snow")})
    updates["theta"] = _theta_m_from_microphysics(th, state.qv, output["theta"], output["qv"], state.theta.dtype)
    # Original surface driver supplies total RAINNCV plus separate snow,
    # graupel and hail channels. Cloud-ice precipitation belongs only to total.
    rain = output["rain_other"]  # original rain + cloud-ice sedimentation, no subtractive cancellation
    updates.update(rain_acc=state.rain_acc + rain,
                   snow_acc=state.snow_acc + output["snownc"],
                   graupel_acc=state.graupel_acc + output["graupelnc"],
                   hail_acc=state.hail_acc + output["hailnc"], goddard_sr=output["sr"])
    result = state.replace(**updates)
    if return_precipitation:
        return result, {"rain": rain, "snow": output["snownc"],
                        "graupel": output["graupelnc"], "ice": jnp.zeros_like(rain), "hail": output["hailnc"],
                        "total": output["rainnc"], "sr": output["sr"]}
    return result
