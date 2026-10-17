"""WRF ``phy_prep`` staging shared by the legacy surface/PBL/cumulus callers.

Pristine WRF builds these once per step in ``phy_prep`` (dyn_em/module_big_step_utilities_em.F:4834-4970)
and dyn_em/module_first_rk_step_part1.F hands the SAME arrays to every column driver:
surface (:633-662) ``P_PHY=p_hyd P8W=p_hyd_w TH_PHY=th_phy T_PHY=t_phy RHO=rho``, PBL (:1130-1139) and cumulus
(:1379-1400) ``TH=th_phy T=t_phy P=p_hyd PI=pi_phy RHO=rho DZ8W=dz8w P8W=p_hyd_w``.

* ``th_phy = (t+t0)/(1+R_v/R_d*qv)`` (use_theta_m=1, :4834): the State carries theta_m, physics gets DRY theta.
* ``pi_phy = (p_phy/p1000mb)**rcp`` with the NON-hydrostatic ``p_phy = p+pb`` (:4851-4853); ``t_phy = th_phy*pi_phy``.
* ``rho = (1+qv)/alt`` TOTAL density (:4854), ``z_at_w = (ph+phb)/g`` with WRF ``g = 9.81`` (:4866),
  ``dz8w`` (:4874), ``z`` mass heights (:4891).
* ``p_hyd_w``/``p_hyd`` hydrostatic face/mass pressure (:4935-4968); the surface pressure every driver sees is
  ``PSFC = p8w(kts) = p_hyd_w(1)`` (module_surface_driver.F:1988, module_bl_myjpbl PINT(LOWLYR), BMJ PSFC).

Arithmetic follows the carry dtype like ``_dry_theta_view`` (REAL fp32 carry -> WRF REAL); the hydrostatic and
density reconstructions are the existing qualified REAL producers (KF/Tiedtke/ACM2/YSU already use them).
Without dycore metrics (isolated adapter tests) the pressure/density members are ``None``.
"""
from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    P0_PA,
    R_D_OVER_CP,
    WRF_PHYSICS_G_M_S2,
    _dry_theta_view,
    _wrf_hydrostatic_pressure_profiles_from_state,
    _wrf_phy_prep_rho_from_state,
)


def wrf_phy_prep(state, metrics=None):
    """Return the WRF phy_prep fields as bottom-up ``(nz, ny, nx)`` / ``(nz+1, ny, nx)`` arrays."""

    th_phy = _dry_theta_view(state)
    dtype = th_phy.dtype
    p_phy = jnp.asarray(state.p, dtype)
    pi_phy = (p_phy / jnp.asarray(P0_PA, dtype)) ** jnp.asarray(R_D_OVER_CP, dtype)
    z_at_w = jnp.asarray(state.ph, dtype) / jnp.asarray(WRF_PHYSICS_G_M_S2, dtype)
    out = dict(th_phy=th_phy, p_phy=p_phy, pi_phy=pi_phy, t_phy=th_phy * pi_phy, z_at_w=z_at_w,
               dz8w=z_at_w[1:] - z_at_w[:-1], z=0.5 * (z_at_w[:-1] + z_at_w[1:]),
               rho=None, p_hyd=None, p_hyd_w=None, psfc=None)
    if metrics is not None:
        p_hyd, p_hyd_w, psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=dtype)
        out.update(p_hyd=p_hyd, p_hyd_w=jnp.asarray(p_hyd_w, dtype), psfc=psfc,
                   rho=_wrf_phy_prep_rho_from_state(state, metrics, output_dtype=dtype))
    return out


__all__ = ["wrf_phy_prep"]
