"""WRF Zeng--Beljaars warm layer / cool skin, surface_driver's pre-sfclay phase.

Bulk SST is distinct from SSTSK and remains held between auxinput4 reads.
SSTSK and DTW are REAL prognostic restart fields, initialized to the WRF
allocator zeros (Registry: neither has an initial-input stream).
"""
from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SSTSkinState(NamedTuple):
    bulk_sst: object  # WRF SST, including the most recent aux4 record [K]
    sstsk: object     # WRF SSTSK, REAL; zero on inactive land columns [K]
    dtw: object       # WRF DTW, REAL warm-layer temperature difference [C]


def sst_skin_update(xland, glw, gsw, hfx, qfx, tsk, ust, emiss, skin, dt,
                    stbolt=5.67051e-8):
    """Literal module_sf_sstskin.F:67-130, all work in WRF REAL (float32)."""
    xland, glw, gsw, hfx, qfx, tsk, ust, emiss = (
        jnp.asarray(value, jnp.float32) for value in (xland, glw, gsw, hfx, qfx, tsk, ust, emiss))
    old, dtwo = jnp.asarray(skin.sstsk, jnp.float32), jnp.asarray(skin.dtw, jnp.float32)
    z1, an, zk, rho, rhow, cw, g, znuw, zkw = map(jnp.float32, (3., .3, .4, 1.2, 1025., 4190., 9.8, 1.e-6, 1.4e-7))
    qo = glw - emiss * jnp.float32(stbolt) * old**4 - 2.5e6 * qfx - hfx
    us = jnp.maximum(ust, .01)
    tb = tsk - 273.15
    q, sw = qo / (rhow * cw), gsw / (rhow * cw)
    f1 = 1.0 - .27 * jnp.exp(-2.8 * z1) - .45 * jnp.exp(-.07 * z1)
    alw = 1.e-5 * jnp.maximum(tb, 1.)
    con4 = 16. * g * alw * znuw**3 / zkw**2
    usw = jnp.sqrt(rho / rhow) * us
    con5 = con4 / usw**4
    q2 = jnp.maximum(1. / (rhow * cw), -q)
    zlan = 6. / (1. + (con5 * q2)**.75)**.333
    dep = zlan * znuw / usw
    fs = .065 + 11. * dep - (6.6e-5 / dep) * (1. - jnp.exp(-dep / 8.e-4))
    fs = jnp.maximum(fs, .01)
    dtc = jnp.minimum(dep * (q + sw * fs) / zkw, 0.)
    con1 = jnp.sqrt(5. * z1 * g * alw / an)
    con2 = zk * g * alw
    qn = q + sw * f1
    # WRF really uses MAX with a POSITIVE qn1 when cooling a warm layer.
    qn1 = jnp.sqrt(jnp.maximum(dtwo, 0.)) * usw**2 / con1
    qn = jnp.where((dtwo > 0.) & (qn < 0.), jnp.maximum(qn, qn1), qn)
    zeta = z1 * con2 * qn / usw**3
    phi = jnp.where(zeta > 0., 1. + 5. * zeta,
                    1. / jnp.sqrt(jnp.where(zeta > 0., 1., 1. - 16. * zeta)))
    con3 = zk * usw / (z1 * phi)
    dt = jnp.float32(dt)
    dtw = (dtwo + (an + 1.) / an * (q + sw * f1) * dt / z1) / (1. + (an + 1.) * con3 * dt)
    dtw = jnp.maximum(0., dtw)
    ts = tb + dtw + dtc
    water = xland >= 1.5
    return SSTSkinState(skin.bulk_sst, jnp.where(water, ts + 273.15, old), jnp.where(water, dtw, dtwo))


def bind_sst_skin(namelist, run_dir, domain, *, state):
    """Read bulk SST once; initialize the distinct prognostic REAL work fields."""
    if not namelist.sst_skin:
        return namelist
    if not namelist.use_noahmp:
        raise ValueError("sst_skin=1 requires native prognostic Noah-MP; CPU-history replay is unsupported")
    from gpuwrf.io.gen2_accessor import Gen2Run
    run = Gen2Run(run_dir)
    with run.input_read_scope(domain):
        bulk = np.asarray(run.load_wrfinput(domain, "SST", lazy=False), np.float32).squeeze()
        ice = np.asarray(run.load_wrfinput(domain, "SEAICE", lazy=False), np.float32).squeeze()
    if bulk.shape != state.t_skin.shape or not np.isfinite(bulk).all():
        raise ValueError("sst_skin requires a finite SST field with the model surface shape")
    if np.any(ice != 0) or not np.isfinite(ice).all():
        raise ValueError("sst_skin sea ice requires an unported sea-ice tile")
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    return replace(namelist, sst_skin_initial=SSTSkinState(jnp.asarray(bulk), zero, zero))


def apply_sst_skin(carry, namelist, lower_fields=None):
    """After radiation/SST update, before surface-layer/land fluxes (WRF:1795)."""
    if bool(namelist.sst_skin) != (carry.sst_skin is not None):
        raise ValueError("sst_skin selection and restart carry disagree (E78)")
    if not namelist.sst_skin:
        return carry
    state, skin, rad = carry.state, carry.sst_skin, carry.radiation_diagnostics
    if rad is None:
        raise ValueError("sst_skin=1 requires held radiation diagnostics")
    bulk = skin.bulk_sst if lower_fields is None else jnp.asarray(lower_fields["SST"], jnp.float32)
    water = state.xland > 1.5
    # surface_driver:1804 resets from SST only with SST_UPDATE /= 1. With
    # aux4 active, the preceding SST phase already applied the valid-range
    # predicate (250<SST<350); preserve its TSK/TSLB result, including holds.
    tsk = (jnp.where(water, bulk.astype(state.t_skin.dtype), state.t_skin)
           if lower_fields is None else state.t_skin)
    skin = skin._replace(bulk_sst=bulk)
    # RRTMG-SW writes GSW = SWDFLX(1,1)-SWUFLX(1,1), not an albedo proxy.
    glw, gsw = rad.glw, rad.swnorm - rad.swup_topographic
    if carry.solar_interpolation is not None:
        glw, gsw = carry.solar_interpolation.fluxes.glw, carry.solar_interpolation.fluxes.gsw
    zero = jnp.zeros_like(tsk)
    def old_flux(name):
        value = getattr(state, name, None)
        return zero if value is None else value
    skin = sst_skin_update(state.xland, glw, gsw, old_flux("hfx"), old_flux("qfx"),
                           tsk, state.ustar, rad.surface_emissivity, skin, namelist.dt_s)
    state = state.replace(t_skin=jnp.where(water, skin.sstsk.astype(tsk.dtype), tsk))
    land = carry.noahmp_land
    if land is not None and lower_fields is None:
        top = jnp.where(water, bulk.astype(land.tslb.dtype), land.tslb[0])
        land = land.replace(tslb=land.tslb.at[0].set(top))
    return carry.replace(state=state, noahmp_land=land, sst_skin=skin)


def validate_restart(skin, enabled, version, shape):
    """E78: all three finite REAL leaves and the exact active selection."""
    if enabled not in (0, 1):
        raise ValueError("sst_skin restart selection must be 0/1 (E78)")
    enabled = bool(enabled)
    if not enabled and skin is None:
        return
    if not enabled or version != 1 or not isinstance(skin, SSTSkinState):
        raise ValueError("sst_skin restart schema/selection is missing or inconsistent (E78)")
    for value in skin:
        arr = np.asarray(value)
        if arr.shape != tuple(shape) or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError("sst_skin restart needs finite REAL bulk SST/SSTSK/DTW with the surface shape (E78)")
