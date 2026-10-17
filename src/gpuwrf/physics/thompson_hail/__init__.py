"""Variable-density operators for the shared Thompson microphysics port.

These are the mp38-specific operators from module_mp_thompson.F. The
candidate composition in ``.column`` remains unadmitted until it passes
the original production-caller qualification.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import thompson_column as tc
from . import _coefficients as constants


R1 = np.float32(1.0e-12)
R2 = np.float32(1.0e-6)
OBMG = np.float32(1.0) / np.float32(3.0)
OGG2 = np.float32(1.0) / np.float32(constants.CGG[0][1])
OGG3 = np.float32(1.0) / np.float32(constants.CGG[0][2])
MVD_FACTOR = np.float32(np.float32(3.0) + np.float32(0.672))
D0R = np.float32(50.0e-6)
MVD_MAX = np.float32(25.4e-3)
A_COEFF = np.float32(0.47244157)
B_COEFF = np.float32(0.54698726)
BFALL = np.float32(np.float32(3.0) * B_COEFF - np.float32(1.0))
RHO_NOT = np.float32(101325.0) / np.float32(np.float32(287.05) * np.float32(298.0))


class HailDistribution(NamedTuple):
    qg: jax.Array
    Ng: jax.Array
    qvolg: jax.Array
    rg: jax.Array
    ng: jax.Array
    rb: jax.Array
    index: jax.Array
    lamg: jax.Array
    ilamg: jax.Array
    n0g: jax.Array


def density_index(qg, qvolg):
    """WRF NINT(qg/qb*.01)+1, converted to a zero-based coefficient index."""
    return jnp.clip(jnp.floor(qg / qvolg * np.float32(0.01) + np.float32(0.5)), 0, 8).astype(jnp.int32)


def entry_distribution(qg, Ng, qvolg, rho, *, rebalance=True):
    """WRF hail entry and slope operators (:1914-1950, :2135-2139).

    Mass, number and volume mixing ratios are per kg dry air. WRF stores
    the working volume rb in the SAME per-kg units, unlike rg and ng.
    """
    active = qg > R1
    rg = jnp.where(active, qg * rho, R1)
    ng = jnp.maximum(R2, Ng * rho)
    # Entry :1916 uses qg1d, while the post-source :3266 floor and density
    # bin use the held REAL volume moment rg/rho (different rounding).
    density_mass = qg if rebalance else rg/rho
    rb = jnp.minimum(density_mass / np.float32(50.), jnp.maximum(density_mass / np.float32(800.), qvolg))
    rb = jnp.where(active, rb, R1 / rho / np.float32(800.))
    index = jnp.where(active, density_index(density_mass, rb), 4)
    amg = jnp.take(jnp.asarray(constants.AM_G, jnp.float32), index)
    cgg2, cgg3 = np.float32(constants.CGG[0][1]), np.float32(constants.CGG[0][2])
    if rebalance:
        default_lamg = (MVD_FACTOR / np.float32(1.5e-3)).astype(jnp.float64)
        ng_default = (cgg2 * OGG3 * rg * default_lamg ** np.float32(3.) / amg).astype(jnp.float32)
        ng = jnp.where(ng <= R2, ng_default, ng)
        lamg = ((amg * cgg3 * OGG2 * ng / rg) ** OBMG).astype(jnp.float64)
        mvd = (MVD_FACTOR / lamg).astype(jnp.float32)
        limited = (mvd > MVD_MAX) | (mvd < D0R)
        clipped = jnp.minimum(MVD_MAX, jnp.maximum(D0R, mvd))
        limited_lamg = (MVD_FACTOR / clipped).astype(jnp.float64)
        corrected_ng = (cgg2 * OGG3 * rg * limited_lamg ** np.float32(3.) / amg).astype(jnp.float32)
        ng = jnp.where(active, jnp.where(limited, corrected_ng, ng), R2)
    else:
        # Original hail sedimentation prep (:3266-3282) recomputes the
        # density bin and slope without another mass/number size balance.
        ng = jnp.where(active, ng, R2)
    # The later slope pass recomputes the REAL power before widening to DOUBLE.
    lamg = ((amg * cgg3 * OGG2 * ng / rg) ** OBMG).astype(jnp.float64)
    ilamg = np.float32(1.) / lamg
    n0g = (ng * OGG2).astype(jnp.float64) * lamg ** np.float32(constants.CGE[0][1])
    return HailDistribution(jnp.where(active, qg, 0.), jnp.where(active, Ng, 0.),
                            jnp.where(active, rb, 0.), rg, ng, rb, index, lamg, ilamg, n0g)


def fall_speeds(distribution, rho, visco, *, fill_down=True, rhof=None):
    """Original Brown--Lawler hail fall speeds (:3742-3783), per column."""
    d = distribution
    xrho = jnp.maximum(np.float32(50.), jnp.minimum(d.rg / rho / d.rb, np.float32(800.)))
    afall = A_COEFF * ((np.float32(4.) * xrho * np.float32(9.8)) / (np.float32(3.) * rho)) ** B_COEFF
    afall = afall * visco ** np.float32(np.float32(1.) - np.float32(2.) * B_COEFF)
    rhof = jnp.sqrt(RHO_NOT / rho) if rhof is None else rhof
    cgg = jnp.asarray(constants.CGG, jnp.float32)
    vt = (rhof * afall * cgg[d.index, 5] * OGG3 * d.ilamg ** BFALL).astype(jnp.float32)
    vtn = (rhof * afall * cgg[d.index, 6] / cgg[d.index, 11] * d.ilamg ** BFALL).astype(jnp.float32)
    active = d.rg > R1
    if fill_down:
        return tc._fill_down(jnp.where(active, vt, 0.), active), tc._fill_down(jnp.where(active, vtn, 0.), active)
    return jnp.where(active, vt, 0.), jnp.where(active, vtn, 0.)


def final_balance(qg, Ng, qvolg, rho):
    """Original final hail mass/number/volume writeback (:4058-4078).

    Inputs have the accumulated physical tendencies already applied. Unlike
    entry, final writeback always reconstructs Ng from the bounded median
    volume diameter, even when neither diameter limit was crossed.
    """
    active = qg > R1
    ng = jnp.maximum(R2 / rho, Ng)
    qb = jnp.minimum(qg / np.float32(50.), jnp.maximum(qg / np.float32(800.), qvolg))
    # Inactive values are only used as safe internal operands before WRF's
    # zero writeback; this guard does not alter any active computation.
    index = density_index(jnp.where(active, qg, R1), jnp.where(active, qb, R1 / np.float32(800.)))
    amg = jnp.take(jnp.asarray(constants.AM_G, jnp.float32), index)
    cgg2, cgg3 = np.float32(constants.CGG[0][1]), np.float32(constants.CGG[0][2])
    lamg = ((amg * cgg3 * OGG2 * ng / jnp.where(active, qg, R1)) ** OBMG).astype(jnp.float64)
    mvd = jnp.minimum(MVD_MAX, jnp.maximum(D0R, (MVD_FACTOR / lamg).astype(jnp.float32)))
    lamg = (MVD_FACTOR / mvd).astype(jnp.float64)
    ng = (cgg2 * OGG3 * qg * lamg ** np.float32(3.) / amg).astype(jnp.float32)
    return jnp.where(active, qg, 0.), jnp.where(active, ng, 0.), jnp.where(active, qb, 0.)


def volume_tendency(*, pbg_scw, pbg_rfz, pbg_gcw, pbg_rci, pbg_rcs,
                    pbg_rcg, pbg_sml, pbg_gml, prg_gde, prg_ihm, index, rho):
    """The original DOUBLE rate sum and REAL qb tendency (:3112-3116)."""
    density = jnp.take(jnp.asarray(constants.RHO_G, jnp.float32), index)
    return ((pbg_scw + pbg_rfz + pbg_gcw + pbg_rci + pbg_rcs + pbg_rcg
             + pbg_sml - pbg_gml + (prg_gde - prg_ihm) / density)
            * (np.float32(1.) / rho)).astype(jnp.float32)


def sediment_hail(rg, ng, rb, rho, dz, vt_mass, vt_number, dt, *, initial_tendencies=None):
    """Original hail flux tendencies (:3898-3938), including volume units.

    This returns additive REAL mass/number/volume tendencies and surface
    precipitation. It keeps WRF's private working floors separate from the
    prognostic tendencies, as the original final writeback does. The shared
    native Thompson kernel can clone this helper into its column namespace;
    this function by itself makes no GPU scheduling/performance claim.
    """
    dt = np.float32(dt)
    nz = rg.shape[-1]
    levels = jnp.arange(nz)
    falling = vt_mass > np.float32(1.0e-3)
    highest = jnp.max(jnp.where(falling, levels + 1, 0), axis=-1)
    ksed1 = jnp.where(highest == nz, nz - 1, highest)
    nstep = jnp.max(jnp.where(falling, jnp.trunc(dt / (dz / jnp.maximum(vt_mass, R1)) + np.float32(1.)).astype(jnp.int32), 0), axis=-1)
    onstep = jnp.where(nstep > 0, np.float32(1.) / jnp.maximum(nstep, 1).astype(jnp.float32), 1.)
    zero = jnp.zeros_like(rg)
    qten, nten, bten = (zero, zero, zero) if initial_tendencies is None else initial_tendencies
    initial = (jnp.int32(0), rg, ng, rb, qten, nten, bten, jnp.zeros(rg.shape[:-1], jnp.float32))
    lower = levels < ksed1[..., None]
    top = levels == nz - 1
    update = lower | top
    odzq, orho = np.float32(1.) / dz, np.float32(1.) / rho

    def condition(carry):
        return carry[0] < jnp.max(nstep)

    def body(carry):
        n, rg, ng, rb, qten, nten, bten, precip = carry
        live = n < nstep
        live_level = live[..., None] & update
        sed_g, sed_n, sed_b = vt_mass * rg, vt_number * ng, vt_mass * rb
        above_g = jnp.concatenate((sed_g[..., 1:], jnp.zeros_like(sed_g[..., :1])), axis=-1)
        above_n = jnp.concatenate((sed_n[..., 1:], jnp.zeros_like(sed_n[..., :1])), axis=-1)
        above_b = jnp.concatenate((sed_b[..., 1:], jnp.zeros_like(sed_b[..., :1])), axis=-1)
        # Preserve the top-layer subtraction separately from the lower-layer
        # difference, exactly as the original Fortran's two branches.
        flux_g = jnp.where(top, -sed_g, above_g - sed_g)
        flux_n = jnp.where(top, -sed_n, above_n - sed_n)
        flux_b = jnp.where(top, -sed_b, above_b - sed_b)
        qinc = flux_g * odzq * onstep[..., None] * orho
        ninc = flux_n * odzq * onstep[..., None] * orho
        binc = flux_b * odzq * onstep[..., None]
        qten = qten + jnp.where(live_level, qinc, 0.)
        nten = nten + jnp.where(live_level, ninc, 0.)
        bten = bten + jnp.where(live_level, binc, 0.)
        rg = jnp.where(live_level, jnp.maximum(R1, rg + flux_g * odzq * dt * onstep[..., None]), rg)
        ng = jnp.where(live_level, jnp.maximum(R2, ng + flux_n * odzq * dt * onstep[..., None]), ng)
        bfloor = jnp.where(top, R1 / rho / np.float32(800.), rg / rho / np.float32(800.))
        rb = jnp.where(live_level, jnp.maximum(bfloor, rb + flux_b * odzq * dt * onstep[..., None]), rb)
        precip = precip + jnp.where(live & (rg[..., 0] > R1 * np.float32(1000.)), sed_g[..., 0] * dt * onstep, 0.)
        return n + 1, rg, ng, rb, qten, nten, bten, precip

    result = jax.lax.while_loop(condition, body, initial)
    return result[4], result[5], result[6], result[7]
