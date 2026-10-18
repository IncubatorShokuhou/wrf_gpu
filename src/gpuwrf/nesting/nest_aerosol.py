"""WRF nest aerosol (aer_opt=1): a child's ``aerod`` is the parent's held ``aerod``, SINT-interpolated level-wise.

WRF interpolates the Tegen climatology only on domain 1 (module_radiation_driver.F:1827,
``IF ( aer_opt .EQ. 1 .AND. id .EQ. 1 )``; module_physics_init.F:2214 for ``aerosol_in``).  The six species of
the 4-D ``aerod`` array carry Registry ``df=(p2c)`` (Registry.EM_COMMON:1274-1279): nest start and every
force-down interpolate the parent's field (interp_fcn SINT, mass point, per level and species) onto the child,
with no re-interpolation to the child pressure; d03 receives d02's copy.  Same mechanism as ``o3rad``
(nesting/nest_o3.py); the leaf exists only on nested aer_opt=1 runs.
"""

from __future__ import annotations

import jax

from gpuwrf.nesting.interp import interp_sint_full

_COMPILED: dict = {}


def force_child_aerod(parent_aerod, mass_weights, *, parent_grid_ratio: int):
    """SINT of the parent ``aerod (6, z, py, px)`` onto the full child mass grid, per species and level."""

    nspec, nz = parent_aerod.shape[:2]
    flat = parent_aerod.reshape((nspec * nz,) + parent_aerod.shape[2:])
    out = interp_sint_full(flat, mass_weights, parent_grid_ratio=int(parent_grid_ratio))
    return out.reshape((nspec, nz) + out.shape[1:]).astype(parent_aerod.dtype)


def _compiled_force(mass_weights, parent_grid_ratio: int, batched: bool):
    key = (id(mass_weights), int(parent_grid_ratio), bool(batched))
    hit = _COMPILED.get(key)
    if hit is None or hit[0] is not mass_weights:
        def one(field):
            return force_child_aerod(field, mass_weights, parent_grid_ratio=parent_grid_ratio)

        hit = (mass_weights, jax.jit(jax.vmap(one) if batched else one))
        _COMPILED[key] = hit
    return hit[1]


def force_child_carry_aerod(child_carry, parent_carry, weights, *, parent_grid_ratio: int, batched: bool = False):
    """Return ``child_carry`` with its held ``aerod`` forced from ``parent_carry``; unchanged when either has none."""

    child_aerod = getattr(child_carry, "aerod", None)
    parent_aerod = getattr(parent_carry, "aerod", None)
    if child_aerod is None or parent_aerod is None:
        return child_carry
    forced = _compiled_force(weights.mass, int(parent_grid_ratio), bool(batched))(parent_aerod)
    return child_carry.replace(aerod=forced.astype(child_aerod.dtype))


__all__ = ["force_child_aerod", "force_child_carry_aerod"]
