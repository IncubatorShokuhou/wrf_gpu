"""ETARA tendency/cumulus-state boundary; no radiation option admission here."""
from __future__ import annotations

import jax.numpy as jnp


def store_tendencies(exner,shortwave,longwave,*,do_lw,do_sw,held_sw,held_lw,total):
    """Map top-down RADTN rates to bottom-up theta rates, exactly as ETARA.

    Both flags identify the second (SW) driver call and skip the solve. A
    single LW call stores both rates, then the second call adds held SW.
    Flags are static scheme choices, never array-valued timestep predicates.
    Inputs contain physical mass levels, with no unused WRF padding.
    """
    if not (do_lw and do_sw):
        if do_lw:
            held_lw=longwave[...,::-1]/exner
            held_sw=shortwave[...,::-1]/exner
            total=total+held_lw
        if do_sw:
            held_sw=shortwave[...,::-1]/exner
    if do_sw:
        total=total+held_sw
    return total,held_lw,held_sw


def reset_convective_interval(top,bottom,precipitation,*,levels):
    """Save source cloud bounds then reset accumulated convection after solve."""
    saved_top,saved_bottom=top,bottom
    return (jnp.zeros_like(top),jnp.full_like(bottom,levels+1),
            jnp.zeros_like(precipitation),saved_top,saved_bottom)
