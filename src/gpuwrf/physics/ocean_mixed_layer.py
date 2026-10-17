"""WRF 4.7.1 ``module_sf_oml.F`` OML1D, in WRF REAL arithmetic.

HUML/HVML are depth-integrated currents (m2/s), not velocities. The seven
Registry fields are resident, including the initial-depth/temperature anchors.
The caller selects water points, exactly as ocean_driver does (XLAND > 1.5).
"""
from typing import NamedTuple

import jax.numpy as jnp


class OceanCarry(NamedTuple):
    tml: object
    t0ml: object
    hml: object
    h0ml: object
    huml: object
    hvml: object
    tmoml: object


def initialize(tsk, hml0=50.0, *, h0ml=None, huml=None, hvml=None, tmoml=None):
    """Pristine omlinit start_of_simulation branch (restart never calls this)."""
    tsk = jnp.asarray(tsk, jnp.float32)
    if hml0 > 0:
        h0ml = jnp.full_like(tsk, hml0)
    elif h0ml is None:
        raise ValueError('oml_hml0<=0 requires wrfinput H0ML')
    h0ml = jnp.asarray(h0ml, jnp.float32)
    if hml0 >= 0:
        huml, hvml, tmoml = jnp.zeros_like(tsk), jnp.zeros_like(tsk), tsk - jnp.float32(5)
    else:
        if any(x is None for x in (huml, hvml, tmoml)):
            raise ValueError('oml_hml0<0 requires HUML/HVML/TMOML in wrfinput')
        tmoml = jnp.asarray(tmoml, jnp.float32)
        tmoml = jnp.where((tmoml > 200) & (tmoml <= 201), tsk, tmoml)
    return OceanCarry(tsk, tsk, h0ml, h0ml, jnp.asarray(huml, jnp.float32),
                      jnp.asarray(hvml, jnp.float32), tmoml)


def step(carry, *, hfx, lh, gsw, glw, u, v, ust, f, emiss, dt,
         gamma=0.14, relaxation_time=0.0, water=True):
    """One OML1D call, with surface_driver's DELTSM=DTBL (seconds).

    No fitted safety floors: max/min and the zero-depth/zero-wind branches below
    are the original WRF operations. Inactive land operands use a finite dummy
    column; land carry is restored without entering singular ocean arithmetic.
    """
    R = jnp.float32
    old = OceanCarry(*(jnp.asarray(x, R) for x in carry))
    water = jnp.asarray(water, bool)
    tml, t0, h, h0, hu, hv, tmoml = (
        jnp.where(water, x, R(dummy))
        for x, dummy in zip(old, (300, 300, 50, 50, 0, 0, 295), strict=True))
    hfx, lh, gsw, glw, u, v, ust, f, emiss = (
        jnp.asarray(x, R) for x in (hfx, lh, gsw, glw, u, v, ust, f, emiss))
    dt, gam = R(dt), R(gamma)
    alp = jnp.maximum((tml - R(273.15)) * R(1.e-5), R(1.e-6))
    bv2 = alp * R(9.81) * gam
    thp = t0 - gam * (h - h0)
    a1 = (tml - thp) * h - R(0.5) * gam * h * h
    q = (-hfx - lh + gsw + glw * emiss - R(5.67051e-8) * emiss * tml * tml * tml * tml) / (R(1000) * R(4200))
    wspd = jnp.maximum(jnp.sqrt(u * u + v * v), R(1.e-10))
    ust2 = R(0.5) * ust
    taux = R(1) / R(1000) * (ust2 * ust2 * u / wspd)
    tauy = R(1) / R(1000) * (ust2 * ust2 * v / wspd)
    hu2 = hu + dt * (f * hv + taux)
    hv2 = hv + dt * (-f * hu2 + tauy)
    a2 = a1 + q * dt
    b2 = hu2 * hu2 + hv2 * hv2
    hsqrd = -a2 / gam + jnp.sqrt(a2 * a2 / (gam * gam) + R(2) * b2 / bv2)
    depth = jnp.maximum(jnp.sqrt(jnp.maximum(hsqrd, R(0))), h)
    nonzero = depth != 0
    tml = jnp.where((tml >= tmoml) & nonzero,
                    jnp.maximum(t0 - gam * (depth - h0) + R(0.5) * gam * depth
                                + a2 / jnp.where(nonzero, depth, R(1)), tmoml), t0)
    if relaxation_time > 0:
        tau = R(relaxation_time)
        tml = tml - (tml - t0) * dt / tau
        depth = depth - (depth - h0) * dt / tau
        hu2 = hu2 - hu2 * dt / tau
        hv2 = hv2 - hv2 * dt / tau
    new = OceanCarry(tml, t0, depth, h0, hu2, hv2, tmoml)
    return OceanCarry(*(jnp.where(water, x, y) for x, y in zip(new, old, strict=True)))
