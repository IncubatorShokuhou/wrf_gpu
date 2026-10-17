"""Port side of the F20 public-call gates: the product MYNN column kernel on the fixture columns.

The kernel receives exactly the operands the WRF caller receives (``f20_oracle.run``); ``w`` is either the
bottom faces kts..kte (what ``physics_couplers._mynn_column_from_state`` now passes) or, for the mutant, the
old mass average. Surface kinematic fluxes come from the product ``mynn_driver_surface_fluxes`` with the
MYNN-driver limits, exactly as the default Noah-MP producer forms them.
"""
import numpy as np
import jax.numpy as jnp

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.surface_layer import mynn_driver_surface_fluxes


def column_inputs(cols, *, w_mode="face", pbl_limits=True):
    f = lambda a: jnp.asarray(np.asarray(a, np.float32))
    nz = cols["th"].shape[1]
    w = cols["w"][:, :nz] if w_mode == "face" else np.float32(0.5) * (cols["w"][:, :nz] + cols["w"][:, 1:])
    zeros = jnp.zeros((cols["th"].shape[0], nz), jnp.float32)
    state = P.MynnPBLColumnState(
        f(cols["u"]), f(cols["v"]), f(w), f(cols["th"]), f(cols["qv"]), f(np.float32(0.5) * cols["qke"]),
        f(cols["p"]), f(cols["rho"]), f(cols["dz"]), zeros, zeros, f(cols["el_pbl"]),
        qc=f(cols["qc"]), qi=f(cols["qi"]), qs=f(cols["qs"]), qsq=zeros, exner=f(cols["exner"]),
        ni=f(cols["qni"]))
    driver = mynn_driver_surface_fluxes(f(cols["hfx"]), f(cols["qfx"]), f(cols["rho"][:, 0]), f(cols["qv"][:, 0]),
                                        f(cols["tsk"]), f(cols["p"][:, 0]), exner=f(cols["exner"][:, 0]),
                                        pbl_limits=pbl_limits)
    ust, wspd = f(cols["ust"]), f(cols["wspd"])
    surface = SurfaceFluxes(
        ustar=ust, theta_flux=driver.flt, qv_flux=driver.flqv,
        tau_u=-(ust * ust) * f(cols["u"][:, 0]) / wspd, tau_v=-(ust * ust) * f(cols["v"][:, 0]) / wspd,
        rhosfc=f(cols["rho"][:, 0]), fltv=driver.fltv, xland=f(cols["xland"]), t_skin=f(cols["tsk"]),
        wspd=wspd, psfc=f(cols["psfc"]))
    return state, surface, driver


def run_kernel(cols, dt, dx, **kwargs):
    state, surface, _ = column_inputs(cols, **kwargs)
    out, pblh = P.step_mynn_pbl_column_with_pblh(state, float(dt), surface=surface, edmf=True, dx=float(dx))
    rth = (np.asarray(out.theta) - np.asarray(state.theta)) / np.float32(dt)
    return {"maxmf": np.asarray(out.maxmf), "pblh": np.asarray(pblh), "rthblten": rth}
