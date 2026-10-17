"""REAL4 P3 lookup coordinates and literal WRF interpolation order.

Coordinates and lower-cell indices are ONE-BASED, matching module_mp_p3.F.
Ice-only callers must supply positive mass and number after P3's own cleanup.
All operations broadcast over columns/levels; no domain-wide level loops.
"""

from typing import NamedTuple

import jax.numpy as jnp


class IceCoordinates(NamedTuple):
    mass_index: object
    rime_index: object
    density_index: object
    mass: object
    rime: object
    density: object


class Coordinate(NamedTuple):
    index: object
    value: object


def _real(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _coordinate(value, size):
    # Fortran INT truncates BEFORE the real coordinate is clamped.
    index = jnp.clip(value.astype(jnp.int32), 1, size - 1)
    return Coordinate(index, jnp.clip(value, _real(1), _real(size)))


def _ice(qi, ni, qir, rho, size, scale, offset):
    mass = _coordinate((jnp.log10(qi / ni) + _real(18)) * _real(scale) - _real(offset), size)
    rime = _coordinate(qir / qi * _real(3) + _real(1), 4)
    density = _coordinate(jnp.where(rho <= _real(650),
                                    (rho - _real(50)) * _real(.005) + _real(1),
                                    (rho - _real(650)) * _real(.004) + _real(4)), 5)
    return IceCoordinates(mass.index, rime.index, density.index, mass.value, rime.value, density.value)


def ice_coordinates(qi, ni, qir, rho):
    """WRF find_lookupTable_indices_1a (50 mass bins)."""
    return _ice(*map(_real, (qi, ni, qir, rho)), 50, 3.444606, 10)


def rain_coordinates(qr, nr):
    """WRF find_lookupTable_indices_1b; inactive rain maps to (1, 1)."""
    qr, nr = map(_real, (qr, nr))
    active = (qr >= _real(1.e-14)) & (nr > _real(0))
    ratio = jnp.where(active, qr, _real(1)) / (
        _real(3.14159265) * _real(1000) * jnp.where(active, nr, _real(1)))
    radius = ratio ** (_real(1) / _real(3))
    value = (jnp.log10(radius) + _real(5)) * _real(10.70415)
    return _coordinate(jnp.where(active, value, _real(1)), 30)


def mu_coordinates(qi, mu):
    """WRF find_lookupTable_indices_1c for triple-moment ice."""
    qi, mu = map(_real, (qi, mu))
    return _coordinate(jnp.where(qi >= _real(1.e-14), mu * _real(.5) + _real(1), _real(1)), 11)


def ice_pair_coordinates(qi1, ni1, qir1, bir1, qi2, ni2, qir2, bir2):
    """WRF indices_2, including its DIFFERENT collector/collectee thresholds."""
    qi1, ni1, qir1, bir1, qi2, ni2, qir2, bir2 = map(
        _real, (qi1, ni1, qir1, bir1, qi2, ni2, qir2, bir2))
    bsmall = _real(1.e-14) * (_real(1) / _real(900))
    rho1 = jnp.where(bir1 >= bsmall, qir1 / jnp.where(bir1 >= bsmall, bir1, _real(1)), _real(0))
    rho2 = jnp.where(bir2 >= _real(1.e-15),
                     qir2 / jnp.where(bir2 >= _real(1.e-15), bir2, _real(1)), _real(0))
    return (_ice(qi1, ni1, qir1, rho1, 25, 1.722303, 5),
            _ice(qi2, ni2, qir2, rho2, 25, 1.722303, 5))


def _interpolate(table, indices, values, order, channel):
    """Reduce corners in WRF's stated order, never as a weighted corner sum.

    Python recursion is static (at most 6 axes / 64 gathers), not a runtime
    recurrence. Each gather returns the broadcast column shape only.
    """
    if not getattr(table, '_p3_table_ref', False):
        table = _real(table)
    if table.ndim != len(indices) + (channel is not None):
        raise ValueError("P3 table rank does not match the selected moment/category coordinates")
    if channel is not None:
        if not isinstance(channel, int) or not 1 <= channel <= table.shape[-1]:
            raise ValueError("P3 process channel must be a valid static one-based integer")
    lower = tuple(index - 1 for index in indices)

    def cell(depth, corner):
        if depth < 0:
            return table[corner + (() if channel is None else (channel - 1,))]
        axis = order[depth]
        lo = cell(depth - 1, corner)
        upper = list(corner)
        upper[axis] = upper[axis] + 1
        hi = cell(depth - 1, tuple(upper))
        return lo + (values[axis] - indices[axis].astype(jnp.float32)) * (hi - lo)

    return cell(len(order) - 1, lower)


def ice_lookup(table, ice: IceCoordinates, channel: int, mu: Coordinate | None = None):
    """WRF access_lookup_table[_3mom], one process channel at a time."""
    indices = (ice.density_index, ice.rime_index, ice.mass_index)
    values = (ice.density, ice.rime, ice.mass)
    if mu is None:
        return _interpolate(table, indices, values, (2, 1, 0), channel)
    return _interpolate(table, (mu.index,) + indices, (mu.value,) + values, (3, 2, 1, 0), channel)


def rain_ice_lookup(table, ice: IceCoordinates, rain: Coordinate, channel: int,
                    mu: Coordinate | None = None):
    """WRF access_lookup_table_coll[_3mom]. Entries are already log10 values."""
    indices = (ice.density_index, ice.rime_index, ice.mass_index, rain.index)
    values = (ice.density, ice.rime, ice.mass, rain.value)
    if mu is None:
        return _interpolate(table, indices, values, (2, 3, 1, 0), channel)
    return _interpolate(table, (mu.index,) + indices, (mu.value,) + values, (3, 4, 2, 1, 0), channel)


def ice_ice_lookup(table, collector: IceCoordinates, collectee: IceCoordinates):
    """WRF access_lookup_table_colli; table axis order starts with COLLECTEE."""
    indices = (collectee.mass_index, collectee.rime_index, collectee.density_index,
               collector.mass_index, collector.rime_index, collector.density_index)
    values = (collectee.mass, collectee.rime, collectee.density,
              collector.mass, collector.rime, collector.density)
    return _interpolate(table, indices, values, (0, 3, 4, 5, 1, 2), None)
