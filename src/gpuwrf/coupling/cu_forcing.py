"""WRF RTHFTEN/RQVFTEN: the held large-scale advective forcing of G3/GD/Tiedtke/NTiedtke.

Lifetime in WRF (one step = solve_em):

* ``rk_tendency`` (module_em.F:654-665) stores ``RTHFTEN = t_tend*msfty`` -- the
  coupled theta *advection* tendency -- at every RK stage when cu_physics is
  GD/GF/G3/NTiedtke; ``rk_scalar_tend`` (module_em.F:1365-1376) stores
  ``RQVFTEN = advect_tend(qv)*msfty`` for GD/G3/GF/KF/MSKF/Tiedtke/NTiedtke.
  The last (rk_order) stage wins.
* ``phy_prep_part2`` (module_big_step_utilities_em.F:5277-5331, called at
  solve_em.F:3669 after advance_ppt and BEFORE microphysics with
  ``mut = grid%muts``) decouples both by ``c1h*muts + c2h`` and, with
  use_theta_m=1, converts the moist-theta advection to a dry-theta tendency:
  ``th_phy = (t_2+t0)/(1+Rv/Rd*qv)``,
  ``RTHFTEN = th_phy/(t_2+t0)*(RTHFTEN - Rv/Rd*th_phy*RQVFTEN)``.
* The NEXT step's cumulus_driver reads the held, uncoupled pair
  (module_cumulus_driver.F:866-883 for G3: ``(RTHFTEN+RTHRATEN+RTHBLTEN)*pi``,
  ``RQVFTEN+RQVBLTEN``; GD folds the same sum inside GRELLDRV).  Both are zero
  at itimestep 1 (Registry state, zero-initialised).

The port carries the pair in ``OperationalCarry.cu_forcing``: coupled during the
step (captured at the final RK stage), uncoupled/dry between steps.
"""
from __future__ import annotations

import jax.numpy as jnp

# Cumulus schemes whose caller now reads the held WRF forcing pair.
CU_FORCING_SCHEMES = (5, 93, 3, 6, 16)  # G3, GD, GF, Tiedtke (RQVFTEN), New-Tiedtke
# RTHFTEN exists only for GD/GF/G3/NTiedtke: set_tend at module_em.F:654-665 and the phy_prep_part2
# decouple + dry conversion (module_big_step_utilities_em.F:5306-5331); Tiedtke (6) reads RQVFTEN only.
CU_RTHFTEN_SCHEMES = (5, 93, 3, 16)
# ... and those whose closure also reads this step's RTHRATEN (radiation_driver runs first).
CU_RTHRATEN_SCHEMES = (5, 93, 3, 16)

# WRF REAL constants of phy_prep_part2 (share/module_model_constants.F).
_T0 = 300.0
_R_V = 461.6
_R_D = 287.0


def cu_forcing_required(cu_physics) -> bool:
    return int(cu_physics) in CU_FORCING_SCHEMES


def initial_cu_forcing(theta):
    """itimestep 1: WRF RTHFTEN/RQVFTEN are zero (zero-initialised Registry state)."""
    zero = jnp.zeros(jnp.shape(theta), jnp.asarray(theta).dtype)
    return (zero, zero)


def capture_coupled(theta_advection, qv_advection, msfty, dtype, held_rthften=None):
    """set_tend: ``field = advective_tendency * msfty`` (module_big_step_utilities_em.F:5824-5830).

    ``theta_advection=None`` (Tiedtke, not in CU_RTHFTEN_SCHEMES): WRF never sets RTHFTEN, so the
    held leaf (``held_rthften``, zero) is returned unchanged and no theta advection is computed.
    """
    m = jnp.asarray(msfty, dtype)[None, :, :]
    rthften = held_rthften if theta_advection is None else jnp.asarray(theta_advection, dtype) * m
    return (rthften, jnp.asarray(qv_advection, dtype) * m)


def finish_cu_forcing(coupled, theta_m, qv, muts, metrics, *, rthften=True):
    """phy_prep_part2: decouple by the final mass, then dry-convert RTHFTEN (use_theta_m=1).

    ``rthften=False`` (Tiedtke): only RQVFTEN is decoupled (:5291-5301); RTHFTEN is neither
    decoupled nor dry-converted (:5306-5331) and keeps its held value.

    ``theta_m`` is the total moist potential temperature ``t_2 + t0`` and ``qv``
    the post-RK vapour, both at the phy_prep_part2 point (after advance_ppt,
    before microphysics).  Arithmetic in the carry dtype (WRF REAL under the
    release carry), in WRF's statement order.
    """
    rthften_c, rqvften_c = coupled
    dtype = jnp.asarray(rthften_c).dtype
    c1 = jnp.asarray(metrics.c1h, dtype)[:, None, None]
    c2 = jnp.asarray(metrics.c2h, dtype)[:, None, None]
    mass = c1 * jnp.asarray(muts, dtype)[None] + c2
    rqvften = jnp.asarray(rqvften_c, dtype) / mass
    if not rthften:
        return (rthften_c, rqvften)
    rthften = jnp.asarray(rthften_c, dtype) / mass
    rvrd = jnp.asarray(_R_V, dtype) / jnp.asarray(_R_D, dtype)
    t_total = jnp.asarray(theta_m, dtype)
    th_phy = t_total / (jnp.asarray(1.0, dtype) + rvrd * jnp.asarray(qv, dtype))
    rthften = th_phy / t_total * (rthften - rvrd * th_phy * rqvften)
    return (rthften, rqvften)
