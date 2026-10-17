"""SPNUDGING caller lifetime: fddagd_driver -> phy_fg_ten -> phy_prep_part2.

Preserve two unusual details of pristine WRF: calculate_phy_tend does NOT
mass-couple RPHNDGDTEN, and phy_prep_part2's spectral branch does NOT decouple
RQVNDGDTEN. These are source behavior, not corrections to the WRF algorithm.
"""
from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.assimilation.fdda_grid import couple_fdda_rates, finish_fdda_rates, owned_fdda_tendf
from gpuwrf.assimilation.fdda_spectral import spectral_driver_tendencies


def spectral_step_sources(carry, inputs, before, th_phy, pblh, ht, z_at_w, lead_seconds,
                          itimestep, mu_2, mub, metrics, dtype, *, specified_or_nested, periodic_x):
    carry = spectral_driver_tendencies(
        carry, inputs, before.u, before.v, th_phy, before.ph_perturbation, before.qv,
        pblh, ht, z_at_w, lead_seconds, itimestep, dtype)
    carry = couple_fdda_rates(carry, mub + mu_2, mu_2, mub, metrics.c1h, metrics.c2h, dtype)
    owned = owned_fdda_tendf(carry, specified_or_nested=specified_or_nested, periodic_x=periodic_x)
    # add_a2a_ph: same horizontal ownership as add_a2a, all physical w levels.
    rp = carry.rphndgdten
    if specified_or_nested:
        xs = slice(None) if periodic_x else slice(1, -1)
        rp = jnp.zeros_like(rp).at[:, 1:-1, xs].set(rp[:, 1:-1, xs])
    return carry, (*owned, rp)


def spectral_finish(carry, final_mu_total, metrics, dtype):
    out = finish_fdda_rates(carry, final_mu_total, metrics.c1h, metrics.c2h, dtype)
    mass_f = (jnp.asarray(metrics.c1f, dtype)[:, None, None] * jnp.asarray(final_mu_total, dtype)[None]
              + jnp.asarray(metrics.c2f, dtype)[:, None, None])
    return out.replace(rphndgdten=carry.rphndgdten / mass_f, rqvndgdten=carry.rqvndgdten)
