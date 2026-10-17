"""Explicit WRF DOUBLE helper islands for the standalone lake port.

Literal algorithms from module_sf_lake.F: Combo, MoninObukIni, Tridiagonal.
These components are not connected to operational forecasts. The QSat oracle
is staged, but its port is deferred: CPU LLVM contracts its Horner operations
even across optimization barriers, missing the frozen cold-limit tolerance.
"""
from __future__ import annotations

import jax
from jax import numpy as jnp


def lake_combine(dz, wliq, wice, temperature, dz2, wliq2, wice2, temperature2):
    """WRF Combo: merged snow thickness [m], liquid/ice [kg/m²], T [K]."""
    with jax.enable_x64():
        dz, wliq, wice, t, dz2, wliq2, wice2, t2 = (
            jnp.asarray(x, jnp.float64) for x in (dz,wliq,wice,temperature,dz2,wliq2,wice2,temperature2))
        cpice, cpliq, hfus, tfrz = 2117.27, 4188.0, 333700.0, 273.16
        dzc, wicec, wliqc = dz + dz2, wice + wice2, wliq + wliq2
        h = (cpice*wice+cpliq*wliq) * (t-tfrz) + hfus*wliq
        h2 = (cpice*wice2+cpliq*wliq2) * (t2-tfrz) + hfus*wliq2
        hc = h+h2
        heat_capacity = cpice*wicec+cpliq*wliqc
        tc = jnp.where(hc < 0.0, tfrz+hc/heat_capacity,
                       jnp.where(hc <= hfus*wliqc, tfrz,
                                 tfrz+(hc-hfus*wliqc)/heat_capacity))
        return dzc, wliqc, wicec, tc


def lake_monin_obukhov_initial(ur, thv, dthv, zldis, z0m):
    """WRF MoninObukIni: stability wind speed [m/s], Obukhov length [m]."""
    with jax.enable_x64():
        ur, thv, dthv, zldis, z0m = (jnp.asarray(x, jnp.float64) for x in (ur, thv, dthv, zldis, z0m))
        um = jnp.where(dthv >= 0.0, jnp.maximum(ur, 0.1), jnp.sqrt(ur*ur+0.5*0.5))
        rib = 9.80616*zldis*dthv / (thv*um*um)
        stable = rib*jnp.log(zldis/z0m)/(1.0-5.0*jnp.minimum(rib, 0.19))
        stable = jnp.minimum(2.0, jnp.maximum(stable, 0.01))
        unstable = jnp.maximum(-100.0, jnp.minimum(rib*jnp.log(zldis/z0m), -0.01))
        return um, zldis/jnp.where(rib >= 0.0, stable, unstable)


def lake_tridiagonal(a, b, c, rhs, *, jtop, previous, enabled=True):
    """WRF Tridiagonal with per-column zero-based top and filter membership.

    Layers are the last axis. Slots above jtop and disabled columns retain
    ``previous`` exactly; no dense matrix or cross-column reduction is built.
    Static short vertical loops expose column-local arithmetic to compilation.
    A final lake kernel must fuse these helpers and be GPU-qualified.
    """
    with jax.enable_x64():
        a, b, c, rhs, u = (jnp.asarray(x, jnp.float64) for x in (a,b,c,rhs,previous))
        jtop = jnp.asarray(jtop, jnp.int32)
        enabled = jnp.asarray(enabled, bool)
        bet = jnp.take_along_axis(b, jtop[..., None], axis=-1)[..., 0]
        gam = jnp.zeros_like(b)
        for j in range(b.shape[-1]):
            active, first = enabled & (j >= jtop), j == jtop
            if j:
                gj = c[..., j-1]/bet
                next_bet = b[..., j]-a[..., j]*gj
                next_u = (rhs[..., j]-a[..., j]*u[..., j-1])/next_bet
                gam = gam.at[..., j].set(jnp.where(active & ~first, gj, 0.0))
                candidate = jnp.where(first, rhs[..., j]/bet, next_u)
                bet = jnp.where(active & ~first, next_bet, bet)
            else:
                candidate = rhs[..., j]/bet
            u = u.at[..., j].set(jnp.where(active, candidate, u[..., j]))
        for j in range(b.shape[-1]-2, -1, -1):
            candidate = u[..., j]-gam[..., j+1]*u[..., j+1]
            u = u.at[..., j].set(jnp.where(enabled & (j >= jtop), candidate, u[..., j]))
        return u
