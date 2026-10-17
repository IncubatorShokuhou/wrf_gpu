"""REAL4 wrapper operations from WRF module_ra_goddard.F (option 5).

The solvers have an explicit DOUBLE island; climatology and Mie interpolation
here retain the original REAL4 arithmetic. This module alone is not a radiation
scheme and is not admitted by the operational dispatcher.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from .goddard_table_digest import TABLES_SHA256


def _verify_tables():
    path=Path(__file__).with_name('goddard_tables.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=TABLES_SHA256:
        raise RuntimeError('Goddard runtime asset SHA256 differs from its source/AOT identity')


# Pure host integrity check, no backend initialization. A cached executable or
# checkpoint import cannot silently accept a changed LUT with the old AOT key.
_verify_tables()


@lru_cache(maxsize=1)
def _host_tables():
    # Cache host arrays only (E206). No tracer/device array escapes a trace.
    _verify_tables()
    with np.load(Path(__file__).with_name('goddard_tables.npz')) as data:
        return {key: data[key] for key in data.files}


def sounding_interp(latitude, julday):
    """WRF sounding_interp: one latitude and day -> (33,6) REAL4 sounding."""
    latitude = jnp.asarray(latitude, jnp.float32)
    day = jnp.asarray(julday, jnp.int32)
    summer = jnp.where(day < 30, (30-day).astype(jnp.float32)/jnp.float32(183),
                       (day-30).astype(jnp.float32)/jnp.float32(183))
    winter = jnp.float32(1)-summer
    winter = jnp.where(day > 212, (day-212).astype(jnp.float32)/jnp.float32(183), winter)
    summer = jnp.where(day > 212, jnp.float32(1)-winter, summer)
    summer = jnp.where(latitude < 0, winter, summer)
    winter = jnp.where(latitude < 0, jnp.float32(1)-summer, winter)
    lat = jnp.abs(latitude)
    north = 2*jnp.trunc((jnp.float32(90)-lat)/jnp.float32(22.5)).astype(jnp.int32)
    south = north+2
    climate = jnp.asarray(_host_tables()['climate'])
    # A corrupt/unsupported equatorial caller must not silently use JAX's
    # clamped out-of-bounds gather in place of WRF's undefined profile11 read.
    get=lambda index:jnp.take(climate,index,axis=1,mode='fill')
    cn = summer*get(north) + winter*get(north+1)
    cs = summer*get(south) + winter*get(south+1)
    lat_north = jnp.float32(90)-(north//2).astype(jnp.float32)*jnp.float32(22.5)
    lat_south = lat_north-jnp.float32(22.5)
    wn = (lat-lat_south)/jnp.float32(22.5)
    ws = jnp.float32(1)-wn
    return wn*cn + ws*cs


def ozone_interp(sounding, pressure_mb):
    """WRF ozone_interp: pressure at mass layers -> ozone mass fraction.

    The original searches downward from the top of the sounding, selecting the
    higher-altitude interval at an exact shared endpoint. searchsorted(side=right)
    preserves that choice, including the original surface/TOA saturation.
    """
    p = jnp.asarray(pressure_mb, jnp.float32)*jnp.float32(100)
    s = jnp.asarray(sounding, jnp.float32)
    index = jnp.clip(jnp.searchsorted(-s[:, 1], -p, side='right')-1, 0, 31)
    weight = (p-s[index+1, 1])/(s[index, 1]-s[index+1, 1])
    interp = s[index, :]*weight[..., None]+s[index+1, :]*(jnp.float32(1)-weight[..., None])
    interp = jnp.where((p <= s[-1, 1])[..., None], s[-1], interp)
    interp = jnp.where((p >= s[0, 1])[..., None], s[0], interp)
    return interp[..., 4]/interp[..., 5]


def opt_cloud(dz, mass_concentration, effective_radius, *, shortwave):
    """WRF opt_cloud with (...,6) species and (...,) layer thickness.

    Species order ice/cloud/rain/snow/graupel/hail; mass concentration g/m3,
    radius micrometres, dz metres. Returns tau/ssa/asymmetry (...,nband).
    Radius saturation and the denominator floors are literal upstream physics.
    """
    q = jnp.asarray(mass_concentration, jnp.float32)
    radius = jnp.asarray(effective_radius, jnp.float32)
    dz = jnp.asarray(dz, jnp.float32)
    tables = _host_tables()
    bins = jnp.asarray(tables['radius'])
    radius = jnp.clip(radius, bins[0], bins[-1])
    index = jnp.clip(jnp.searchsorted(bins, radius, side='left')-1, 0, 107)
    weight2 = (radius-bins[index])/(bins[index+1]-bins[index])
    weight1 = jnp.float32(1)-weight2
    nband, branch = (11, 0) if shortwave else (10, 1)
    species = jnp.arange(6)

    def interp(name):
        tab = jnp.asarray(tables[name][:, :, :nband, branch])
        return weight1[..., None]*tab[index, species, :] + weight2[..., None]*tab[index+1, species, :]

    tau_species = interp('extinction')*q[..., None]*dz[..., None, None]*jnp.float32(1.e-3)
    scattering = tau_species*interp('ssa')
    asymmetry = scattering*interp('asymmetry')
    # Six-species accumulation is left-to-right, like gfortran's scalar SUM.
    zeros = jnp.zeros(q.shape[:-1]+(nband,), jnp.float32)
    def add_species(t, accum):
        tau, scat, asym = accum
        return (tau+tau_species[..., t, :], scat+scattering[..., t, :], asym+asymmetry[..., t, :])
    tau, scat, asym = lax.fori_loop(0, 6, add_species, (zeros, zeros, zeros))
    ssa = scat/jnp.maximum(tau, jnp.float32(1.e-8))
    asym = asym/jnp.maximum(tau*ssa, jnp.float32(1.e-8))
    return tau, ssa, asym
