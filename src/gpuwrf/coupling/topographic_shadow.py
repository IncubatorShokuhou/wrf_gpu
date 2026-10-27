"""Lifecycle of the existing radiation carry's optional WRF HGT_SHAD plane."""
from __future__ import annotations

import jax.numpy as jnp


def active(namelist):
    return int(getattr(namelist, "topo_shading", 0)) == 1 and int(getattr(namelist, "slope_rad", 0)) == 1


def nested(namelist):
    return not bool(getattr(getattr(namelist, "boundary_config", None), "force_geopotential", True))


def cold_kwargs(namelist, state):
    if not active(namelist):
        return {}
    return {"previous_shadow_height_m": jnp.zeros_like(state.t_skin, dtype=jnp.float32),
            "nested_shadow": nested(namelist)}


def held_kwargs(namelist, diagnostics):
    if not active(namelist):
        return {}
    value = getattr(diagnostics, "shadow_height_m", None)
    if value is None:
        raise ValueError("topo_shading=1 radiation carry lacks WRF HGT_SHAD (E78)")
    return {"previous_shadow_height_m": value, "nested_shadow": nested(namelist)}


def seed(carry, namelist):
    diagnostics = carry.radiation_diagnostics
    if not active(namelist) or not hasattr(diagnostics, "shadow_height_m"):
        return carry
    # module_integrate opens cold nests before the parent's first solve.
    # HGT_SHAD begins at allocator zero, then evolves only on radiation calls.
    return carry.replace(radiation_diagnostics=diagnostics._replace(
        shadow_height_m=jnp.zeros_like(carry.state.t_skin, dtype=jnp.float32)))


def preserve_initial_height(seeded, computed):
    """Keep WRF's cold HGT_SHAD seed across the separate LSM init solve.

    The init solve supplies held surface fluxes, not an extra toposhad history
    call. Reuse the existing plane rather than allocating another zero array.
    """
    height = getattr(seeded, "shadow_height_m", None)
    return computed if height is None else computed._replace(shadow_height_m=height)


__all__ = ["active", "nested", "cold_kwargs", "held_kwargs", "seed", "preserve_initial_height"]
