"""Noah-MP Schaake96 soil hydrology + runoff (Sprint S4).

Ports WATER (module_sf_noahmplsm.F:5954-6261) restricted to the Schaake branch
(opt_run=3) and SOILWATER (:7234-7556), with opt_inf=1 / opt_frz=1 frozen-soil
treatment. NO GROUNDWATER (SIMGM/SIMTOP CUT).

Steps: canopy interception update (CANLIQ/CANICE/FWET), infiltration, Schaake96
surface + subsurface runoff, the Richards-like soil-moisture tridiagonal solve
(SOILWATER), and supercooled-liquid (SH2O <= SMC). Consumes the transpiration /
evaporation sinks from the energy step (S1) as soil-moisture withdrawals.

FULLY PARALLEL to author (savepoints supply the ET inputs as fixtures); integrates
serially with S1 (consumes its ET). Oracle = WRF SMC/SH2O/SMCWTD/SFCRUNOFF/UDRUNOFF
savepoint parity + a water-mass conservation check (the LH 18x over-flux must
collapse to ~1x once evaporation is soil-hydraulic + canopy-resistance limited).
"""

from __future__ import annotations
from gpuwrf.physics.noahmp.land_columns import layer_row
from functools import partial

from gpuwrf._x64_config import configure_jax_x64

import jax.numpy as jnp

from gpuwrf.physics.noahmp.precision import real_scalar, native_real_enabled
from gpuwrf.physics.noahmp.land_columns import stack_layers,set_layer,add_layer
from gpuwrf.kernels.phys_noahmp_columns import column_call, columns_enabled
from gpuwrf.kernels.phys_noahmp_layers import (  # GPUWRF_NOAHMP_LAYER_SELECT / _LISTS
    lists_enabled, put, to_list)
from jax import config

from gpuwrf.contracts.noahmp_state import NSOIL
from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing

configure_jax_x64()

_TFRZ = 273.16
_HFUS = 0.3336e6
_CWAT = 4.188e6
_CICE = 2.094e6
_DENH2O = 1000.0
_DENICE = 917.0
_REFDK = 2.0e-6
_REFKDT = 3.0
_FRZK = 0.15
_SLOPE_TYPE_1 = 0.1
_SLOPETYP = 1   # WRF default subsurface-runoff slope category (opt_run=3)


def _soil_field(value, template: jnp.ndarray) -> jnp.ndarray:
    """Broadcast a soil-vector/field value to ``(NSOIL, ny, nx)``."""

    arr = jnp.asarray(value, dtype=template.dtype)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape[0] != NSOIL:
        raise ValueError(f"expected first dimension {NSOIL}, got {arr.shape}")
    return jnp.broadcast_to(
        arr.reshape((NSOIL,) + (1,) * (template.ndim - 1)),
        template.shape,
    )


def _surface_field(value, template: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=template.dtype)
    return jnp.broadcast_to(arr, template.shape)


def _category_index(category: jnp.ndarray, size: int) -> jnp.ndarray:
    """Row index into a 1-BASED parameter table (axis-0 length ncat+1, dummy row 0).

    The frozen ``NoahMPParameters`` soil/veg tables keep WRF's 1-based category
    layout (``tables._parse_soilparm`` fills rows 1..ncat; row 0 is an all-zero
    placeholder), so the category id IS the row index — identical to
    ``noahmp_driver._gather_vec`` and the phenology gathers. The previous
    ``category - 1`` shifted every category one row down: ISLTYP=1 (sand) read
    the all-zero row 0 (SMCMAX=0 -> smc/smcmax=inf -> NaN soil moisture, the
    v0.14 d01 LU16 preflight blocker) and every other soil type silently ran
    WATER with the previous category's hydraulic parameters.
    """
    return jnp.clip(category.astype(jnp.int32), 0, max(size - 1, 0))


def _gather_by_category(
    arr: jnp.ndarray,
    category: jnp.ndarray,
    template: jnp.ndarray,
) -> jnp.ndarray:
    """Gather 1-based category table values using WRF's 1-based category ids."""

    idx = _category_index(category, arr.shape[0])
    return jnp.asarray(jnp.take(arr, idx, axis=0), dtype=template.dtype)


def _soil_param(parameters, name: str, static: NoahMPStatic, template: jnp.ndarray, default) -> jnp.ndarray:
    """Return a soil parameter as ``(NSOIL, ny, nx)``.

    The frozen table bundle stores soil values by category, but oracle fixtures
    often inject already-expanded layer fields. This accepts both without
    widening the public interface.
    """

    arr = jnp.asarray(getattr(parameters, name, default), dtype=template.dtype)
    surface_shape = template.shape[1:]
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 1:
        if arr.shape[0] == NSOIL:
            return _soil_field(arr, template)
        gathered = _gather_by_category(arr, static.isltyp, template[0])
        return jnp.broadcast_to(gathered[jnp.newaxis, ...], template.shape)
    if arr.ndim == 2:
        if arr.shape[0] == NSOIL and arr.shape[1:] == surface_shape:
            return jnp.asarray(arr, dtype=template.dtype)
        if arr.shape[0] == NSOIL:
            idx = _category_index(static.isltyp, arr.shape[1])
            values = [jnp.take(arr[k], idx, axis=0) for k in range(NSOIL)]
            return jnp.stack(values, axis=0).astype(template.dtype)
        if arr.shape[1] == NSOIL:
            idx = _category_index(static.isltyp, arr.shape[0])
            values = [jnp.take(arr[:, k], idx, axis=0) for k in range(NSOIL)]
            return jnp.stack(values, axis=0).astype(template.dtype)
    if arr.ndim == template.ndim and arr.shape[0] == NSOIL:
        return jnp.broadcast_to(arr, template.shape)
    raise ValueError(f"unsupported {name} parameter shape {arr.shape}")


def _veg_param(parameters, name: str, static: NoahMPStatic, template: jnp.ndarray, default) -> jnp.ndarray:
    arr = jnp.asarray(getattr(parameters, name, default), dtype=template.dtype)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 1:
        if arr.shape[0] == template.shape[0] and arr.shape == template.shape:
            return arr
        return _gather_by_category(arr, static.ivgtyp, template)
    return jnp.broadcast_to(arr, template.shape)


def _geometry(static: NoahMPStatic, template: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    zsoil = _soil_field(static.zsoil, template)
    dzs = _soil_field(static.dzs, template)
    return zsoil, dzs


def _canwater(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,
    fveg: jnp.ndarray,
    ch2op: jnp.ndarray,
    dt: jnp.ndarray,
    frozen_canopy=None, phen=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """WRF CANWATER, with ET already provided as mass fluxes.

    FROZEN_CANOPY is ENERGY's flag from the TV entering ENERGY (:2210-2216, the same TV that
    picked LATHEAV for ECAN); without it the post-ENERGY TV is used (legacy).
    """

    del static
    canliq = jnp.asarray(land_state.canliq)
    canice = jnp.asarray(land_state.canice)
    tv = jnp.asarray(land_state.tv)
    frozen_canopy = tv <= _TFRZ if frozen_canopy is None else jnp.asarray(frozen_canopy, dtype=bool)

    ecan = jnp.asarray(et_fluxes.ecan, dtype=canliq.dtype)
    etran = jnp.maximum(jnp.asarray(et_fluxes.etran, dtype=canliq.dtype), 0.0)
    qevac = jnp.where(frozen_canopy, 0.0, jnp.maximum(ecan, 0.0))
    qdewc = jnp.where(frozen_canopy, 0.0, jnp.maximum(-ecan, 0.0))
    qsubc = jnp.where(frozen_canopy, jnp.maximum(ecan, 0.0), 0.0)
    qfroc = jnp.where(frozen_canopy, jnp.maximum(-ecan, 0.0), 0.0)

    maxliq = fveg * ch2op * (land_state.lai + land_state.sai) if not _cw_elai() else _cw_fveg(fveg, phen) * ch2op * _cw_area(land_state, phen)
    canliq = jnp.maximum(0.0, canliq + (qdewc - jnp.minimum(canliq / dt, qevac)) * dt)
    canliq = jnp.where(canliq <= 1.0e-6, 0.0, canliq)

    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((forcing.sfctmp - _TFRZ) / 2.59))
    maxsno = fveg * 6.6 * (0.27 + 46.0 / jnp.maximum(bdfall, 1.0e-12)) * (land_state.lai + land_state.sai) if not _cw_elai() else _cw_fveg(fveg, phen) * 6.6 * (0.27 + 46.0 / jnp.maximum(bdfall, 1.0e-12)) * _cw_area(land_state, phen)
    canice = jnp.maximum(0.0, canice + (qfroc - jnp.minimum(canice / dt, qsubc)) * dt)
    canice = jnp.where(canice <= 1.0e-6, 0.0, canice)

    fwet_ice = jnp.maximum(0.0, canice) / jnp.maximum(maxsno, 1.0e-6)
    fwet_liq = jnp.maximum(0.0, canliq) / jnp.maximum(maxliq, 1.0e-6)
    fwet = jnp.where((canice > 0.0) & (canice >= canliq), fwet_ice, fwet_liq)
    fwet = jnp.minimum(fwet, 1.0) ** 0.667

    cmc = canliq + canice
    qmeltc = jnp.where(
        (canice > 1.0e-6) & (tv > _TFRZ),
        jnp.minimum(canice / dt, (tv - _TFRZ) * _CICE * canice / _DENICE / (dt * _HFUS)),
        0.0,
    )
    canice_melt = jnp.maximum(0.0, canice - qmeltc * dt)
    canliq_melt = jnp.maximum(0.0, cmc - canice_melt)
    tv_melt = fwet * _TFRZ + (1.0 - fwet) * tv
    canice = jnp.where(qmeltc > 0.0, canice_melt, canice)
    canliq = jnp.where(qmeltc > 0.0, canliq_melt, canliq)
    tv = jnp.where(qmeltc > 0.0, tv_melt, tv)

    cmc = canliq + canice
    qfrzc = jnp.where(
        (canliq > 1.0e-6) & (tv < _TFRZ),
        jnp.minimum(canliq / dt, (_TFRZ - tv) * _CWAT * canliq / _DENH2O / (dt * _HFUS)),
        0.0,
    )
    canliq_frz = jnp.maximum(0.0, canliq - qfrzc * dt)
    canice_frz = jnp.maximum(0.0, cmc - canliq_frz)
    tv_frz = fwet * _TFRZ + (1.0 - fwet) * tv
    canliq = jnp.where(qfrzc > 0.0, canliq_frz, canliq)
    canice = jnp.where(qfrzc > 0.0, canice_frz, canice)
    tv = jnp.where(qfrzc > 0.0, tv_frz, tv)

    return canliq, canice, fwet, tv, etran


def _wdfcnd1(
    smc: jnp.ndarray,
    fcr: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    factr = jnp.maximum(0.01, smc / smcmax)
    wdf = dwsat * factr ** (bexp + 2.0) * (1.0 - fcr)
    wcnd = dksat * factr ** (2.0 * bexp + 3.0) * (1.0 - fcr)
    return wdf, wcnd


def _wdfcnd2(
    smc: jnp.ndarray,
    sice: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    factr1 = jnp.minimum(0.05 / smcmax, jnp.maximum(0.01, smc / smcmax))
    factr2 = jnp.maximum(0.01, smc / smcmax)
    expon = bexp + 2.0
    wdf = dwsat * factr2**expon
    vkwgt = 1.0 / (1.0 + (500.0 * sice) ** 3.0)
    wdf_ice = vkwgt * wdf + (1.0 - vkwgt) * dwsat * factr1**expon
    wdf = jnp.where(sice > 0.0, wdf_ice, wdf)
    wcnd = dksat * factr2 ** (2.0 * bexp + 3.0)
    return wdf, wcnd


def _infil(
    qinsur: jnp.ndarray,
    dt: jnp.ndarray,
    zsoil: jnp.ndarray,
    sh2o: jnp.ndarray,
    sice: jnp.ndarray,
    sicemax: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    smcwlt: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
    kdt: jnp.ndarray,
    frzx: jnp.ndarray,
    *, literal=False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """WRF INFIL for opt_run=3 / opt_inf=1. Returns ``(pddum, runsrf)`` in m/s."""

    smcav = layer_row(smcmax,0) - layer_row(smcwlt,0)
    dmax0 = -layer_row(zsoil,0) * smcav
    dice = -layer_row(zsoil,0) * layer_row(sice,0)
    dmax0 = dmax0 * (1.0 - (layer_row(sh2o,0) + layer_row(sice,0) - layer_row(smcwlt,0)) / smcav)
    dd = dmax0
    for k in range(1, NSOIL):
        thickness = layer_row(zsoil,k - 1) - layer_row(zsoil,k)
        dice = dice + thickness * layer_row(sice,k)
        dmax = thickness * smcav
        dmax = dmax * (1.0 - (layer_row(sh2o,k) + layer_row(sice,k) - layer_row(smcwlt,k)) / smcav)
        dd = dd + dmax

    dt1 = dt / 86400.0
    val = 1.0 - jnp.exp(-kdt * dt1)
    ddt = dd * val
    px = jnp.maximum(0.0, qinsur * dt)
    infmax = (px * (ddt / (px + ddt))) / dt

    acrt = 3.0 * frzx / dice
    frozen_sum = 1.0 + acrt + 0.5 * acrt**2
    if literal:frozen_sum=(1.+acrt**2/2.)+acrt  # INFIL J=1,2 accumulation order.
    frozen_factor = 1.0 - jnp.exp(-acrt) * frozen_sum
    fcr = jnp.where(dice > 1.0e-2, frozen_factor, 1.0)
    infmax = infmax * fcr

    _, wcnd = _wdfcnd2(layer_row(sh2o,0), sicemax, layer_row(bexp,0), layer_row(smcmax,0), layer_row(dksat,0), layer_row(dwsat,0))
    infmax = jnp.maximum(infmax, wcnd)
    infmax = jnp.minimum(infmax, px / dt)
    runsrf = jnp.maximum(0.0, qinsur - infmax)
    pddum = qinsur - runsrf
    active = qinsur > 0.0
    return jnp.where(active, pddum, 0.0), jnp.where(active, runsrf, 0.0)


def _srt(
    zsoil: jnp.ndarray,
    dt: jnp.ndarray,
    pddum: jnp.ndarray,
    etrani: jnp.ndarray,
    qseva: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    fcr: jnp.ndarray,
    fcrmax: jnp.ndarray,
    slope: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
    *, opt_inf=1, sicemax=None, opt_run=3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del dt
    smx = smc
    if opt_inf==1:
        wdf, wcnd = _wdfcnd1(smx, fcr, bexp, smcmax, dksat, dwsat)
    else:
        smx=sh2o
        wdf,wcnd=_wdfcnd2(smx,sicemax,bexp,smcmax,dksat,dwsat)

    denom0 = -layer_row(zsoil,0)
    temp0 = -layer_row(zsoil,1)
    ddz0 = 2.0 / temp0
    dsmdz0 = 2.0 * (layer_row(smx,0) - layer_row(smx,1)) / temp0
    wflux0 = layer_row(wdf,0) * dsmdz0 + layer_row(wcnd,0) - pddum + layer_row(etrani,0) + qseva

    denom1 = layer_row(zsoil,0) - layer_row(zsoil,1)
    temp1 = layer_row(zsoil,0) - layer_row(zsoil,2)
    ddz1 = 2.0 / temp1
    dsmdz1 = 2.0 * (layer_row(smx,1) - layer_row(smx,2)) / temp1
    wflux1 = layer_row(wdf,1) * dsmdz1 + layer_row(wcnd,1) - layer_row(wdf,0) * dsmdz0 - layer_row(wcnd,0) + layer_row(etrani,1)

    denom2 = layer_row(zsoil,1) - layer_row(zsoil,2)
    temp2 = layer_row(zsoil,1) - layer_row(zsoil,3)
    ddz2 = 2.0 / temp2
    dsmdz2 = 2.0 * (layer_row(smx,2) - layer_row(smx,3)) / temp2
    wflux2 = layer_row(wdf,2) * dsmdz2 + layer_row(wcnd,2) - layer_row(wdf,1) * dsmdz1 - layer_row(wcnd,1) + layer_row(etrani,2)

    denom3 = layer_row(zsoil,2) - layer_row(zsoil,3)
    qdrain = slope * layer_row(wcnd,3)
    if opt_run in (1,2):qdrain=jnp.zeros_like(qdrain)
    if opt_run==4:qdrain=(1.-fcrmax)*layer_row(wcnd,3)
    wflux3 = -(layer_row(wdf,2) * dsmdz2) - layer_row(wcnd,2) + layer_row(etrani,3) + qdrain

    denom = stack_layers([denom0, denom1, denom2, denom3])
    ddz = stack_layers([ddz0, ddz1, ddz2, jnp.zeros_like(ddz2)])
    wflux = stack_layers([wflux0, wflux1, wflux2, wflux3])

    ai0 = jnp.zeros_like(wflux0)
    bi0 = layer_row(wdf,0) * layer_row(ddz,0) / layer_row(denom,0)
    ci0 = -bi0
    ai1 = -layer_row(wdf,0) * layer_row(ddz,0) / layer_row(denom,1)
    ci1 = -layer_row(wdf,1) * layer_row(ddz,1) / layer_row(denom,1)
    bi1 = -(ai1 + ci1)
    ai2 = -layer_row(wdf,1) * layer_row(ddz,1) / layer_row(denom,2)
    ci2 = -layer_row(wdf,2) * layer_row(ddz,2) / layer_row(denom,2)
    bi2 = -(ai2 + ci2)
    ai3 = -layer_row(wdf,2) * layer_row(ddz,2) / layer_row(denom,3)
    ci3 = jnp.zeros_like(ai3)
    bi3 = -(ai3 + ci3)

    ai = stack_layers([ai0, ai1, ai2, ai3])
    bi = stack_layers([bi0, bi1, bi2, bi3])
    ci = stack_layers([ci0, ci1, ci2, ci3])
    rhstt = wflux / (-denom)
    return rhstt, ai, bi, ci, qdrain, wcnd


def _rosr12(ai: jnp.ndarray, bi: jnp.ndarray, ci: jnp.ndarray, rhs: jnp.ndarray, *, literal=False) -> jnp.ndarray:
    c = set_layer(ci, NSOIL - 1, 0.0)
    p0 = -layer_row(c,0) / layer_row(bi,0)
    d0 = layer_row(rhs,0) / layer_row(bi,0)
    p1 = -layer_row(c,1) / (layer_row(bi,1) + layer_row(ai,1) * p0)
    d1 = (layer_row(rhs,1) - layer_row(ai,1) * d0) / (layer_row(bi,1) + layer_row(ai,1) * p0)
    p2 = -layer_row(c,2) / (layer_row(bi,2) + layer_row(ai,2) * p1)
    d2 = (layer_row(rhs,2) - layer_row(ai,2) * d1) / (layer_row(bi,2) + layer_row(ai,2) * p1)
    p3 = -layer_row(c,3) / (layer_row(bi,3) + layer_row(ai,3) * p2)
    d3 = (layer_row(rhs,3) - layer_row(ai,3) * d2) / (layer_row(bi,3) + layer_row(ai,3) * p2)
    if literal:
        # Original ROSR12 multiplies the rounded reciprocal (not division).
        inv1=1./(layer_row(bi,1)+layer_row(ai,1)*p0)
        p1=-layer_row(c,1)*inv1;d1=(layer_row(rhs,1)-layer_row(ai,1)*d0)*inv1
        inv2=1./(layer_row(bi,2)+layer_row(ai,2)*p1)
        p2=-layer_row(c,2)*inv2;d2=(layer_row(rhs,2)-layer_row(ai,2)*d1)*inv2
        inv3=1./(layer_row(bi,3)+layer_row(ai,3)*p2)
        d3=(layer_row(rhs,3)-layer_row(ai,3)*d2)*inv3
    x3 = d3
    x2 = p2 * x3 + d2
    x1 = p1 * x2 + d1
    x0 = p0 * x1 + d0
    del p3
    return stack_layers([x0, x1, x2, x3])


def _sstep(
    dt: jnp.ndarray,
    dzs: jnp.ndarray,
    sice: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    ai: jnp.ndarray,
    bi: jnp.ndarray,
    ci: jnp.ndarray,
    rhstt: jnp.ndarray,
    smcmax: jnp.ndarray,
    *, literal=False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del smc
    ai_dt = ai * dt
    bi_dt = 1.0 + bi * dt
    ci_dt = ci * dt
    rhstt_dt = rhstt * dt

    delta = _rosr12(ai_dt, bi_dt, ci_dt, rhstt_dt,**({"literal":True} if literal else {}))
    sh2o = sh2o + delta

    epore3 = jnp.maximum(1.0e-4, layer_row(smcmax,3) - layer_row(sice,3))
    wplus3 = jnp.maximum(layer_row(sh2o,3) - epore3, 0.0) * layer_row(dzs,3)
    sh2o = set_layer(sh2o, 3, jnp.minimum(epore3, layer_row(sh2o,3)))
    sh2o = add_layer(sh2o, 2, wplus3 / layer_row(dzs,2))

    epore2 = jnp.maximum(1.0e-4, layer_row(smcmax,2) - layer_row(sice,2))
    wplus2 = jnp.maximum(layer_row(sh2o,2) - epore2, 0.0) * layer_row(dzs,2)
    sh2o = set_layer(sh2o, 2, jnp.minimum(epore2, layer_row(sh2o,2)))
    sh2o = add_layer(sh2o, 1, wplus2 / layer_row(dzs,1))

    epore1 = jnp.maximum(1.0e-4, layer_row(smcmax,1) - layer_row(sice,1))
    wplus1 = jnp.maximum(layer_row(sh2o,1) - epore1, 0.0) * layer_row(dzs,1)
    sh2o = set_layer(sh2o, 1, jnp.minimum(epore1, layer_row(sh2o,1)))
    sh2o = add_layer(sh2o, 0, wplus1 / layer_row(dzs,0))

    epore0 = jnp.maximum(1.0e-4, layer_row(smcmax,0) - layer_row(sice,0))
    wplus0 = jnp.maximum(layer_row(sh2o,0) - epore0, 0.0) * layer_row(dzs,0)
    sh2o_base = set_layer(sh2o, 0, jnp.minimum(epore0, layer_row(sh2o,0)))

    sh2o_true = add_layer(sh2o_base, 1, wplus0 / layer_row(dzs,1))
    epore1b = jnp.maximum(1.0e-4, layer_row(smcmax,1) - layer_row(sice,1))
    wplus1b = jnp.maximum(layer_row(sh2o_true,1) - epore1b, 0.0) * layer_row(dzs,1)
    sh2o_true = set_layer(sh2o_true, 1, jnp.minimum(epore1b, layer_row(sh2o_true,1)))
    sh2o_true = add_layer(sh2o_true, 2, wplus1b / layer_row(dzs,2))

    epore2b = jnp.maximum(1.0e-4, layer_row(smcmax,2) - layer_row(sice,2))
    wplus2b = jnp.maximum(layer_row(sh2o_true,2) - epore2b, 0.0) * layer_row(dzs,2)
    sh2o_true = set_layer(sh2o_true, 2, jnp.minimum(epore2b, layer_row(sh2o_true,2)))
    sh2o_true = add_layer(sh2o_true, 3, wplus2b / layer_row(dzs,3))

    epore3b = jnp.maximum(1.0e-4, layer_row(smcmax,3) - layer_row(sice,3))
    wplus3b = jnp.maximum(layer_row(sh2o_true,3) - epore3b, 0.0) * layer_row(dzs,3)
    sh2o_true = set_layer(sh2o_true, 3, jnp.minimum(epore3b, layer_row(sh2o_true,3)))

    mask = wplus0 > 0.0
    sh2o = jnp.where(mask[jnp.newaxis, ...], sh2o_true, sh2o_base)
    wplus = jnp.where(mask, wplus3b, wplus0)
    smc = sh2o + sice
    return sh2o, smc, wplus


def _srt_lists(zsoil, pddum, etrani, qseva, smx, fcr, slope, bexp, smcmax, dksat, dwsat,
               *, opt_inf=1, sh2o=None, sicemax=None, opt_run=3, fcrmax=None):
    """:func:`_srt` on per-layer lists (GPUWRF_NOAHMP_LAYER_LISTS): same expressions, list outputs."""
    if opt_inf==1:
        pairs = [_wdfcnd1(layer_row(smx,k), layer_row(fcr,k), layer_row(bexp,k), layer_row(smcmax,k), layer_row(dksat,k), layer_row(dwsat,k)) for k in range(NSOIL)]
    else:
        smx=sh2o
        pairs=[_wdfcnd2(layer_row(smx,k),sicemax,layer_row(bexp,k),layer_row(smcmax,k),layer_row(dksat,k),layer_row(dwsat,k)) for k in range(NSOIL)]
    wdf, wcnd = [w for w, _ in pairs], [c for _, c in pairs]

    denom0 = -layer_row(zsoil,0)
    temp0 = -layer_row(zsoil,1)
    ddz0 = 2.0 / temp0
    dsmdz0 = 2.0 * (layer_row(smx,0) - layer_row(smx,1)) / temp0
    wflux0 = layer_row(wdf,0) * dsmdz0 + layer_row(wcnd,0) - pddum + layer_row(etrani,0) + qseva

    denom1 = layer_row(zsoil,0) - layer_row(zsoil,1)
    temp1 = layer_row(zsoil,0) - layer_row(zsoil,2)
    ddz1 = 2.0 / temp1
    dsmdz1 = 2.0 * (layer_row(smx,1) - layer_row(smx,2)) / temp1
    wflux1 = layer_row(wdf,1) * dsmdz1 + layer_row(wcnd,1) - layer_row(wdf,0) * dsmdz0 - layer_row(wcnd,0) + layer_row(etrani,1)

    denom2 = layer_row(zsoil,1) - layer_row(zsoil,2)
    temp2 = layer_row(zsoil,1) - layer_row(zsoil,3)
    ddz2 = 2.0 / temp2
    dsmdz2 = 2.0 * (layer_row(smx,2) - layer_row(smx,3)) / temp2
    wflux2 = layer_row(wdf,2) * dsmdz2 + layer_row(wcnd,2) - layer_row(wdf,1) * dsmdz1 - layer_row(wcnd,1) + layer_row(etrani,2)

    denom3 = layer_row(zsoil,2) - layer_row(zsoil,3)
    qdrain = slope * layer_row(wcnd,3)
    if opt_run==4:qdrain=(1.-fcrmax)*layer_row(wcnd,3)
    wflux3 = -(layer_row(wdf,2) * dsmdz2) - layer_row(wcnd,2) + layer_row(etrani,3) + qdrain

    denom = [denom0, denom1, denom2, denom3]
    ddz = [ddz0, ddz1, ddz2]
    wflux = [wflux0, wflux1, wflux2, wflux3]

    bi0 = layer_row(wdf,0) * layer_row(ddz,0) / layer_row(denom,0)
    ci0 = -bi0
    ai1 = -layer_row(wdf,0) * layer_row(ddz,0) / layer_row(denom,1)
    ci1 = -layer_row(wdf,1) * layer_row(ddz,1) / layer_row(denom,1)
    bi1 = -(ai1 + ci1)
    ai2 = -layer_row(wdf,1) * layer_row(ddz,1) / layer_row(denom,2)
    ci2 = -layer_row(wdf,2) * layer_row(ddz,2) / layer_row(denom,2)
    bi2 = -(ai2 + ci2)
    ai3 = -layer_row(wdf,2) * layer_row(ddz,2) / layer_row(denom,3)
    ci3 = jnp.zeros_like(ai3)
    bi3 = -(ai3 + ci3)

    ai = [jnp.zeros_like(wflux0), ai1, ai2, ai3]
    bi = [bi0, bi1, bi2, bi3]
    ci = [ci0, ci1, ci2, ci3]
    rhstt = [wflux[k] / (-layer_row(denom,k)) for k in range(NSOIL)]
    return rhstt, ai, bi, ci, qdrain


def _rosr12_lists(ai, bi, c, rhs, *,literal=False):
    """:func:`_rosr12` on lists (``layer_row(c,NSOIL-1)`` only enters the unused ``p3``)."""
    p0 = -layer_row(c,0) / layer_row(bi,0)
    d0 = layer_row(rhs,0) / layer_row(bi,0)
    p1 = -layer_row(c,1) / (layer_row(bi,1) + layer_row(ai,1) * p0)
    d1 = (layer_row(rhs,1) - layer_row(ai,1) * d0) / (layer_row(bi,1) + layer_row(ai,1) * p0)
    p2 = -layer_row(c,2) / (layer_row(bi,2) + layer_row(ai,2) * p1)
    d2 = (layer_row(rhs,2) - layer_row(ai,2) * d1) / (layer_row(bi,2) + layer_row(ai,2) * p1)
    d3 = (layer_row(rhs,3) - layer_row(ai,3) * d2) / (layer_row(bi,3) + layer_row(ai,3) * p2)
    if literal:
        inv1=1./(layer_row(bi,1)+layer_row(ai,1)*p0)
        p1=-layer_row(c,1)*inv1;d1=(layer_row(rhs,1)-layer_row(ai,1)*d0)*inv1
        inv2=1./(layer_row(bi,2)+layer_row(ai,2)*p1)
        p2=-layer_row(c,2)*inv2;d2=(layer_row(rhs,2)-layer_row(ai,2)*d1)*inv2
        inv3=1./(layer_row(bi,3)+layer_row(ai,3)*p2)
        d3=(layer_row(rhs,3)-layer_row(ai,3)*d2)*inv3
    x3 = d3
    x2 = p2 * x3 + d2
    x1 = p1 * x2 + d1
    x0 = p0 * x1 + d0
    return [x0, x1, x2, x3]


def _sstep_lists(dt, dzs, sice, sh2o, ai, bi, ci, rhstt, smcmax, *,literal=False):
    """:func:`_sstep` on lists: the same slot updates (``set_layer``/``add_layer``) in order."""
    tp = layer_row(sh2o,0).dtype
    delta = _rosr12_lists([a * dt for a in ai], [1.0 + b * dt for b in bi], [c * dt for c in ci],
                          [r * dt for r in rhstt],**({"literal":True} if literal else {}))
    sh2o = [layer_row(sh2o,k) + delta[k] for k in range(NSOIL)]

    def overflow(layers, k, nxt):
        epore = jnp.maximum(1.0e-4, layer_row(smcmax,k) - layer_row(sice,k))
        wplus = jnp.maximum(layers[k] - epore, 0.0) * layer_row(dzs,k)
        put(layers, k, jnp.minimum(epore, layers[k]), tp)
        if nxt is not None:
            put(layers, nxt, layers[nxt] + wplus / dzs[nxt], tp)
        return wplus

    overflow(sh2o, 3, 2)
    overflow(sh2o, 2, 1)
    overflow(sh2o, 1, 0)
    wplus0 = overflow(sh2o, 0, None)
    sh2o_base = list(sh2o)
    sh2o_true = list(sh2o_base)
    put(sh2o_true, 1, layer_row(sh2o_true,1) + wplus0 / layer_row(dzs,1), tp)
    overflow(sh2o_true, 1, 2)
    overflow(sh2o_true, 2, 3)
    wplus3b = overflow(sh2o_true, 3, None)

    mask = wplus0 > 0.0
    sh2o = [jnp.where(mask, layer_row(sh2o_true,k), sh2o_base[k]) for k in range(NSOIL)]
    wplus = jnp.where(mask, wplus3b, wplus0)
    smc = [layer_row(sh2o,k) + layer_row(sice,k) for k in range(NSOIL)]
    return sh2o, smc, wplus


def _soilwater(
    qinsur: jnp.ndarray,
    qseva: jnp.ndarray,
    etrani: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    sice: jnp.ndarray,
    zsoil: jnp.ndarray,
    dzs: jnp.ndarray,
    smcwtd: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    smcref: jnp.ndarray,
    smcwlt: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
    kdt: jnp.ndarray,
    frzx: jnp.ndarray,
    slope: jnp.ndarray,
    dt: jnp.ndarray,
    urban: jnp.ndarray | None = None,
    *, opt_inf=1, opt_run=3, psisat=None,fsatmx=None,timean=None,groundwater=None,hydrology=None,opt_infdv=0,opt_tdrn=0,drain_layer_opt=0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if opt_run in (6,7,8) or opt_tdrn:
        from gpuwrf.physics.noahmp.runoff_options import advanced_soilwater
        if hydrology is None:raise ValueError('Noah-MP runoff/tile choices require their actual tables/maps (E78)')
        return advanced_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcref,smcwlt,dksat,dwsat,kdt,frzx,slope,dt,hydrology,urban,
            opt_run=opt_run,opt_inf=opt_inf,opt_infdv=opt_infdv,opt_tdrn=opt_tdrn,drain_layer_opt=drain_layer_opt)
    if opt_run==1:
        from gpuwrf.physics.noahmp.groundwater_options import prognostic_soilwater
        if groundwater is None:raise ValueError("Noah-MP opt_run=1 needs its groundwater state (E78)")
        return prognostic_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcwlt,dksat,dwsat,psisat,dt,fsatmx,timean,
                                   groundwater,urban,opt_inf=opt_inf)
    if opt_run==2:
        from gpuwrf.physics.noahmp.groundwater_options import equilibrium_soilwater
        return equilibrium_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcwlt,dksat,dwsat,psisat,dt,fsatmx,timean,
                                     urban,opt_inf=opt_inf)
    del smcref, smcwtd
    epore = jnp.maximum(1.0e-4, smcmax - sice)
    rsat = jnp.sum(jnp.maximum(0.0, sh2o - epore) * dzs, axis=0)
    sh2o = jnp.minimum(epore, sh2o)

    ice_frac = jnp.minimum(1.0, sice / smcmax)
    # WRF REAL EXP(-4.0): native REAL gets a REAL scalar, not an f64 0-d array promoting FCR to f64.
    fcr = (jnp.maximum(0.0, jnp.exp(-4.0 * (1.0 - ice_frac)) - jnp.exp(real_scalar(-4.0)))
           / (1.0 - jnp.exp(real_scalar(-4.0))))
    sicemax = jnp.max(sice, axis=0)
    fcrmax = jnp.max(fcr, axis=0)

    # Pristine SOILWATER sets the urban top-layer impermeable fraction after
    # computing FCRMAX (:7377); keep that caller order and the legacy path.
    if urban is not None:
        fcr = set_layer(fcr, 0, jnp.where(urban, 0.95, layer_row(fcr,0)))

    if opt_run==4:
        from gpuwrf.physics.noahmp.surface_options import bats_infiltration
        pddum0,runsrf0=bats_infiltration(qinsur,smc,smcmax,dzs,layer_row(fcr,0))
    else:
        pddum0, _ = _infil(qinsur, dt, zsoil, sh2o, sice, sicemax, bexp, smcmax, smcwlt, dksat, dwsat, kdt, frzx,
                            **({"literal":True} if opt_inf==2 else {}))
    niter = jnp.where(pddum0 * dt > layer_row(dzs,0) * layer_row(smcmax,0), real_scalar(6.0), real_scalar(3.0))
    dtfine = dt / niter

    qdrain_save = jnp.zeros_like(qinsur)
    runsrf_save = jnp.zeros_like(qinsur)

    if lists_enabled():
        sh2o, smc, rsat, qdrain_save, runsrf_save = _soilwater_loop_lists(
            qinsur, qseva, etrani, sh2o, smc, sice, sicemax, zsoil, dzs, fcr, bexp, smcmax, smcwlt, dksat, dwsat,
            kdt, frzx, slope, niter, dtfine, rsat, qdrain_save, runsrf_save,
            **({"opt_inf":opt_inf,"opt_run":opt_run,"fcrmax":fcrmax,
                **({"pddum0":pddum0,"runsrf0":runsrf0} if opt_run==4 else {})} if (opt_inf,opt_run)!=(1,3) else {}))
    for iter_idx in range(0 if lists_enabled() else 6):
        active = iter_idx < niter
        if opt_run==4:pddum,runsrf=pddum0,runsrf0  # WRF holds BATS inputs through the fine substeps.
        else:pddum, runsrf = _infil(qinsur, dtfine, zsoil, sh2o, sice, sicemax, bexp, smcmax, smcwlt, dksat, dwsat, kdt, frzx,
                                    **({"literal":True} if opt_inf==2 else {}))
        pddum = jnp.where((qinsur > 0.0) & active, pddum, 0.0)
        runsrf = jnp.where((qinsur > 0.0) & active, runsrf, 0.0)
        rhstt, ai, bi, ci, qdrain, _ = _srt(
            zsoil, dtfine, pddum, etrani, qseva, sh2o, smc, fcr, fcrmax,
            slope, bexp, smcmax, dksat, dwsat,
            **({"opt_inf":opt_inf,"sicemax":sicemax,"opt_run":opt_run} if (opt_inf,opt_run)!=(1,3) else {}),
        )
        sh2o_next, smc_next, wplus = _sstep(dtfine, dzs, sice, sh2o, smc, ai, bi, ci, rhstt, smcmax,
                                          **({"literal":True} if (opt_inf,opt_run)!=(1,3) else {}))
        sh2o = jnp.where(active[jnp.newaxis, ...], sh2o_next, sh2o)
        smc = jnp.where(active[jnp.newaxis, ...], smc_next, smc)
        rsat = rsat + jnp.where(active, wplus, 0.0)
        qdrain_save = qdrain_save + jnp.where(active, qdrain, 0.0)
        runsrf_save = runsrf_save + runsrf

    qdrain = qdrain_save / niter
    runsrf = runsrf_save / niter
    runsrf = runsrf * 1000.0 + rsat * 1000.0 / dt
    qdrain = qdrain * 1000.0

    mliq = sh2o * dzs * 1000.0
    runsub = jnp.zeros_like(qinsur)
    for iz in range(NSOIL - 1):
        xs = jnp.where(layer_row(mliq,iz) < 0.0, 0.01 - layer_row(mliq,iz), 0.0)
        mliq = add_layer(mliq, iz, xs)
        mliq = add_layer(mliq, iz + 1, -xs)

    xs = jnp.where(layer_row(mliq,NSOIL - 1) < 0.01, 0.01 - layer_row(mliq,NSOIL - 1), 0.0)
    mliq = add_layer(mliq, NSOIL - 1, xs)
    runsub = runsub - xs / dt
    sh2o = mliq / (dzs * 1000.0)
    if (opt_inf,opt_run)==(1,3):smc = sh2o + sice
    return sh2o, smc, runsrf, qdrain, runsub


def _soilwater_loop_lists(qinsur, qseva, etrani, sh2o, smc, sice, sicemax, zsoil, dzs, fcr, bexp, smcmax, smcwlt,
                          dksat, dwsat, kdt, frzx, slope, niter, dtfine, rsat, qdrain_save, runsrf_save,
                          *,opt_inf=1,opt_run=3,fcrmax=None,pddum0=None,runsrf0=None):
    """The 6-trip SOILWATER sub-step loop of :func:`_soilwater` on per-layer lists (same order);
    under GPUWRF_NOAHMP_COLUMN_KERNELS the whole loop is one column kernel."""
    args = (qinsur, qseva, to_list(etrani), to_list(sh2o), to_list(smc), to_list(sice), sicemax, to_list(zsoil),
            to_list(dzs), to_list(fcr), to_list(bexp), to_list(smcmax), to_list(smcwlt), to_list(dksat),
            to_list(dwsat), kdt, frzx, slope, niter, dtfine, rsat, qdrain_save, runsrf_save)
    if opt_run==4:args=(*args,fcrmax,pddum0,runsrf0)
    fn=partial(_soilwater_loop_core,opt_inf=opt_inf,opt_run=opt_run) if (opt_inf,opt_run)!=(1,3) else _soilwater_loop_core
    if columns_enabled():
        out = column_call(fn, args, grid_shape=jnp.shape(qinsur), name="b_core_noah_soilwater")
    else:
        out = fn(*args)
    sh2o_l, smc_l, rsat, qdrain_save, runsrf_save = out
    return jnp.stack(sh2o_l), jnp.stack(smc_l), rsat, qdrain_save, runsrf_save


def _soilwater_loop_core(qinsur, qseva, etrani, sh2o_l, smc_l, sice_l, sicemax, zsoil, dzs, fcr, bexp, smcmax,
                         smcwlt, dksat, dwsat, kdt, frzx, slope, niter, dtfine, rsat, qdrain_save, runsrf_save,
                         fcrmax=None,pddum0=None,runsrf0=None,*,opt_inf=1,opt_run=3):
    for iter_idx in range(6):
        active = iter_idx < niter
        if opt_run==4:
            pddum,runsrf=pddum0,runsrf0
        else:
            pddum, runsrf = _infil(qinsur, dtfine, zsoil, sh2o_l, sice_l, sicemax, bexp, smcmax, smcwlt, dksat, dwsat,
                                   kdt, frzx,**({"literal":True} if opt_inf==2 else {}))
        pddum = jnp.where((qinsur > 0.0) & active, pddum, 0.0)
        runsrf = jnp.where((qinsur > 0.0) & active, runsrf, 0.0)
        rhstt, ai, bi, ci, qdrain = _srt_lists(zsoil, pddum, etrani, qseva, smc_l, fcr, slope, bexp, smcmax,
                                               dksat, dwsat,
            **({"opt_inf":opt_inf,"sh2o":sh2o_l,"sicemax":sicemax,"opt_run":opt_run,"fcrmax":fcrmax}
               if (opt_inf,opt_run)!=(1,3) else {}))
        sh2o_next, smc_next, wplus = _sstep_lists(dtfine, dzs, sice_l, sh2o_l, ai, bi, ci, rhstt, smcmax,
                                                **({"literal":True} if (opt_inf,opt_run)!=(1,3) else {}))
        sh2o_l = [jnp.where(active, sh2o_next[k], sh2o_l[k]) for k in range(NSOIL)]
        smc_l = [jnp.where(active, smc_next[k], smc_l[k]) for k in range(NSOIL)]
        rsat = rsat + jnp.where(active, wplus, 0.0)
        qdrain_save = qdrain_save + jnp.where(active, qdrain, 0.0)
        runsrf_save = runsrf_save + runsrf
    return sh2o_l, smc_l, rsat, qdrain_save, runsrf_save


def noahmp_water_hydro(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,
    dt: float,
    *, history: bool = False,
    sneqv_before_snow=None,
    qrain_ground=None,
    frozen_canopy=None, phen=None, irrigation=None, fveg_override=None, options=None,groundwater=None,tile_drain=None,
    mmf=None,
) -> NoahMPLandState:
    """Advance soil/canopy water one ``dt`` (Schaake96).

    Returns the land carry with SMC/SH2O/SMCWTD/CANLIQ/CANICE/FWET/SFCRUNOFF/
    UDRUNOFF updated; thermal and snow fields untouched (those are S2/S3).

    WRF order (WATER :6095-6160, used by noah_mp_step): ``land_state`` already holds the
    post-SNOWWATER pack, ``sneqv_before_snow`` is the SNEQV that split QSNSUB/QSNFRO off the
    ground vapour/dew flux (:6116-6126) and ``et_fluxes.qmelt`` is the snow-to-soil water
    flux (PONDING+PONDING1+PONDING2)/DT + QSNBOT in mm/s (:6149-6152). ``qrain_ground`` is
    PRECIP_HEAT's ground rain QRAIN; it enters QINSUR unchanged on ISNOW == 0 (:6150) — the
    frozen channels are not subtracted (OPT_SNF=1 never reads PRCPSNOW/GRPL/HAIL, :1163-1229).
    ``frozen_canopy`` is ENERGY's FROZEN_CANOPY (TV entering ENERGY <= TFRZ) for CANWATER.
    """

    from gpuwrf.physics.noahmp.precision import real_tree
    land_state, forcing, static, et_fluxes = real_tree((land_state, forcing, static, et_fluxes))
    smc = jnp.asarray(land_state.smois)
    sh2o = jnp.asarray(land_state.sh2o)
    surface = jnp.asarray(land_state.smcwtd)
    dt_arr = jnp.asarray(dt, dtype=smc.dtype)
    parameters = static.parameters

    zsoil, dzs = _geometry(static, smc)
    bexp = _soil_param(parameters, "bexp", static, smc, 5.0)
    smcmax = _soil_param(parameters, "smcmax", static, smc, 0.45) if not _urban_soil_on() else _urban_soil(_soil_param(parameters, "smcmax", static, smc, 0.45), static, parameters, "smcmax")
    smcref = _soil_param(parameters, "smcref", static, smc, 0.35) if not _urban_soil_on() else _urban_soil(_soil_param(parameters, "smcref", static, smc, 0.35), static, parameters, "smcref")
    smcwlt = _soil_param(parameters, "smcwlt", static, smc, 0.10) if not _urban_soil_on() else _urban_soil(_soil_param(parameters, "smcwlt", static, smc, 0.10), static, parameters, "smcwlt")
    dksat = _soil_param(parameters, "dksat", static, smc, 1.0e-6)
    dwsat = _soil_param(parameters, "dwsat", static, smc, 1.0e-5)

    sice = jnp.maximum(0.0, smc - sh2o)

    ch2op = _veg_param(parameters, "ch2op", static, surface, 0.1)
    # FVEG (dveg=4) = SHDMAX, a per-column wrfinput field on NoahMPStatic (= VEGMAX/100),
    # NOT an MPTABLE parameter (arbiter: module_sf_noahmplsm.F:864). Must match the
    # phenology/energy FVEG. Fall back: instantaneous SHDFAC, then 1.0 (defensive).
    fveg_src = static.shdmax if static.shdmax is not None else static.shdfac
    if fveg_src is None:
        fveg = jnp.broadcast_to(jnp.asarray(1.0, dtype=surface.dtype), surface.shape)
    else:
        fveg = jnp.broadcast_to(
            jnp.asarray(fveg_src, dtype=surface.dtype), surface.shape
        )
        fveg = jnp.where(fveg <= 0.05, 0.05, fveg)
    ivgtyp = jnp.asarray(static.ivgtyp)
    fveg = jnp.where((ivgtyp == 25) | (ivgtyp == 26) | (ivgtyp == 27), 0.0, fveg)
    if fveg_override is not None:
        fveg=jnp.asarray(fveg_override,dtype=surface.dtype)
    canliq, canice, fwet, tv, etran = _canwater(land_state, forcing, static, et_fluxes, fveg, ch2op, dt_arr,
                                                frozen_canopy=frozen_canopy, phen=phen)

    ground_et = jnp.where(jnp.abs(et_fluxes.edir) > 0.0, et_fluxes.edir, et_fluxes.qseva)
    qvap = jnp.maximum(jnp.asarray(ground_et, dtype=surface.dtype), 0.0)
    qdew = jnp.maximum(-jnp.asarray(ground_et, dtype=surface.dtype), 0.0)
    sneqv_w = land_state.sneqv if sneqv_before_snow is None else jnp.asarray(
        sneqv_before_snow, dtype=jnp.asarray(land_state.sneqv).dtype)
    has_snow = sneqv_w > 0.0
    qsnsubl = jnp.where(has_snow, jnp.minimum(qvap, sneqv_w / dt_arr), 0.0)
    qseva_mm_s = qvap - qsnsubl
    qsdew_mm_s = jnp.where(has_snow, 0.0, qdew)

    frozen_ground = land_state.tg <= _TFRZ
    sice0_frozen = layer_row(sice,0) + (qsdew_mm_s - qseva_mm_s) * dt_arr / (layer_row(dzs,0) * 1000.0)
    sh2o0_frozen = layer_row(sh2o,0) + jnp.minimum(sice0_frozen, 0.0)
    sice0_frozen = jnp.maximum(sice0_frozen, 0.0)
    sh2o = set_layer(sh2o, 0, jnp.where(frozen_ground, sh2o0_frozen, layer_row(sh2o,0)))
    sice = set_layer(sice, 0, jnp.where(frozen_ground, sice0_frozen, layer_row(sice,0)))
    qseva_mm_s = jnp.where(frozen_ground, 0.0, qseva_mm_s)
    qsdew_mm_s = jnp.where(frozen_ground, 0.0, qsdew_mm_s)
    smc = sh2o + sice

    if qrain_ground is None:
        qrain_total = jnp.maximum(forcing.prcpconv, 0.0) + jnp.maximum(forcing.prcpnonc, 0.0)
        solid_prcp = jnp.maximum(et_fluxes.qsnow, forcing.prcpsnow + forcing.prcpgrpl + forcing.prcphail)
        qrain = jnp.maximum(qrain_total - jnp.maximum(solid_prcp, 0.0), 0.0)
    else:
        qrain = jnp.asarray(qrain_ground, dtype=surface.dtype)
    qinsur_mm_s = jnp.maximum(et_fluxes.qmelt, 0.0) + qsdew_mm_s + jnp.where(land_state.isnow == 0, qrain, 0.0)
    qinsur = qinsur_mm_s * 0.001
    qseva = qseva_mm_s * 0.001
    if irrigation is not None:
        from gpuwrf.physics.noahmp.irrigation import soil_irrigation
        bundle, irrigation_out, opt_irrm = irrigation
        sh2o, qinsur, irrigation_out = soil_irrigation(bundle,irrigation_out,smc,sh2o,sice,zsoil,qinsur,dt_arr,
                                                     opt_irrm=opt_irrm)

    btrani = _soil_field(et_fluxes.btrani, smc)
    etrani = jnp.maximum(etran, 0.0)[jnp.newaxis, ...] * btrani * 0.001

    kdt_default = _REFKDT * layer_row(dksat,0) / _REFDK
    smcref_safe = jnp.maximum(layer_row(smcref,0), 1.0e-12)
    frzx_default = _FRZK * (layer_row(smcmax,0) / smcref_safe) * (0.412 / 0.468)
    kdt = _surface_field(getattr(parameters, "kdt", kdt_default), surface)
    frzx = _surface_field(getattr(parameters, "frzx", frzx_default), surface)
    # SLOPE: the frozen S0b NoahMPParameters carries the per-slope-type table
    # (nslope+1,); WRF gathers it by SLOPETYP (TRANSFER_MP_PARAMETERS, opt_run=3
    # default SLOPETYP=1). A pre-gathered scalar (oracle fixtures) is used as-is.
    slope_raw = jnp.asarray(getattr(parameters, "slope", _SLOPE_TYPE_1), dtype=surface.dtype)
    if slope_raw.ndim == 1 and slope_raw.shape[0] > 1:
        slope_val = slope_raw[jnp.clip(jnp.int32(_SLOPETYP), 0, slope_raw.shape[0] - 1)]
    else:
        slope_val = slope_raw
    slope = _surface_field(slope_val, surface)

    if mmf is None:
        soil_result = _soilwater(
            qinsur, qseva, etrani, sh2o, smc, sice, zsoil, dzs, land_state.smcwtd,
            bexp, smcmax, smcref, smcwlt, dksat, dwsat, kdt, frzx, slope, dt_arr,
            urban=((ivgtyp == 13) | ((ivgtyp >= 51) & (ivgtyp <= 61)))
            if native_real_enabled() else None,
            **({"opt_inf":options.opt_inf,"opt_run":options.opt_run} if options is not None else {}),
            **({"psisat":_soil_param(parameters,"psisat",static,smc,.35),"fsatmx":options.fsatmx,"timean":options.timean}
               if options is not None and options.opt_run in (1,2) else {}),
            **({"groundwater":groundwater} if groundwater is not None else {}),
            **({'hydrology':getattr(parameters,'hydrology',None),'opt_infdv':options.opt_infdv,'opt_tdrn':options.opt_tdrn,
                'drain_layer_opt':options.drain_layer_opt} if options is not None and (options.opt_run in (6,7,8) or options.opt_tdrn) else {}),
        )
        if options is not None and options.opt_run==1:*soil_result,groundwater_out,qin=soil_result
        elif options is not None and options.opt_run==2:*soil_result,zwt=soil_result
        if options is not None and options.opt_tdrn:*soil_result,qtl,zwt=soil_result
        sh2o,smc,runsrf_mm_s,qdrain_mm_s,runsub_mm_s=soil_result
        if options is not None and ((options.opt_inf,options.opt_run)!=(1,3) or options.opt_tdrn):
            # WATER:6239-6241 recomputes SMC AFTER SOILWATER's SH2O-only final floor.
            smc=sh2o+sice
    else:
        if not native_real_enabled():
            raise ValueError('MMF opt_run=5 requires the qualified WRF REAL Noah-MP path')
        from gpuwrf.physics.noahmp.groundwater import soilwater5
        psisat = _soil_param(parameters, 'psisat', static, smc, .1)
        sh2o, smc, runsrf_mm_s, qdrain_mm_s, runsub_mm_s, mmf_aux, mmf_carry = soilwater5(
            qinsur, qseva, etrani, sh2o, smc, sice, zsoil, dzs, land_state.smcwtd,
            mmf.carry, mmf.static.smoiseq, bexp, smcmax, dksat, dwsat, psisat,
            mmf.fsatmx, dt_arr, (ivgtyp == 13) | ((ivgtyp >= 51) & (ivgtyp <= 61)),
            opt_inf=mmf.opt_inf)
        # WRF's driver never runs the natural-soil column routine on inactive
        # tiles. Keep their caller-owned groundwater memory unchanged.
        mmf_carry = mmf_carry._replace(**{name: jnp.where(mmf.eligibility,
            getattr(mmf_carry, name), getattr(mmf.carry, name)) for name in ('zwt', 'deeprech', 'rech')})
        mmf = mmf._replace(carry=mmf_carry)

    runsrf_m = runsrf_mm_s * dt_arr * 0.001
    runsub_m = (runsub_mm_s + qdrain_mm_s) * dt_arr * 0.001

    updated = land_state.replace(
        smois=smc,
        sh2o=sh2o,
        canliq=canliq,
        canice=canice,
        fwet=fwet,
        tv=tv,
        sfcrunoff=land_state.sfcrunoff + runsrf_m,
        udrunoff=land_state.udrunoff + runsub_m,
        **({'smcwtd': mmf_aux} if mmf is not None else {}),
    )
    if mmf is not None:
        return (updated, (runsrf_mm_s * dt_arr, (runsub_mm_s + qdrain_mm_s) * dt_arr), mmf) if history else (updated, mmf)
    result = (updated, (runsrf_mm_s * dt_arr, (runsub_mm_s + qdrain_mm_s) * dt_arr)) if history else updated
    if history and options is not None and options.opt_run==2:result=(*result,{"ZWT":zwt})
    if history and options is not None and options.opt_run==1:result=(*result,{"QIN":qin})
    if history and options is not None and options.opt_tdrn:result=(*result,{'ZWT':zwt})
    if irrigation is not None:result=(result,irrigation_out)
    if groundwater is not None:result=(result,groundwater_out)
    return (result,tile_drain+qtl*dt_arr) if tile_drain is not None else result


def _urban_soil_on() -> bool:
    """GPUWRF_NOAH_URBAN_SOIL_PARAMS: see noahmp_driver._urban_soil_on (WRF urban soil, TRANSFER_MP_PARAMETERS :1694-1700).

    WATER sees the urban SMCMAX/SMCREF/SMCWLT; FRZX below is formed from them, as WRF forms FRZX after the override.
    """
    from gpuwrf.physics.noahmp.noahmp_driver import _urban_soil_on as on
    return on()


def _urban_soil(value, static, parameters, name):
    from gpuwrf.physics.noahmp.noahmp_driver import urban_soil
    return urban_soil(value, static.ivgtyp, parameters, name)


def _cw_elai() -> bool:
    """GPUWRF_NOAH_CANWATER_ELAI (V033-W2, default 0; read at trace time -> cheap-key trace-env hash).

    WRF NOAHMP_SFLX hands CANWATER (module_sf_noahmplsm.F:6116 -> :6265) PHENOLOGY's snow-burial-adjusted ELAI/ESAI
    (:1255-1358; ESAI < .05 -> 0, ELAI < .05 or ESAI == 0 -> 0) and the caller FVEG (:864-875: SHDMAX floored at .05,
    urban/barren -> 0, ELAI+ESAI == 0 -> 0): MAXLIQ = FVEG*CH2OP*(ELAI+ESAI) (:6323), MAXSNO = FVEG*6.6*(0.27+46/BDFALL)
    *(ELAI+ESAI) (:6351) -> FWET and the canopy melt/freeze TV pin. The legacy port used the unburied LAI+SAI and a local
    SHDMAX-only FVEG (identical except on snow-buried canopies). ON: both from ``phen`` (the driver's PHENOLOGY);
    OFF: the legacy expressions stay inline on their original lines (OFF HLO + locations identical).
    """
    from os import environ
    return environ.get("GPUWRF_NOAH_CANWATER_ELAI", "0") == "1"


def _cw_phen(phen):
    if phen is None:
        raise ValueError("GPUWRF_NOAH_CANWATER_ELAI=1 needs the step's PHENOLOGY (phen=) at the CANWATER call")
    return phen


def _cw_fveg(fveg, phen):
    """CANWATER FVEG = the NOAHMP_SFLX/PHENOLOGY FVEG (key on)."""
    return jnp.asarray(_cw_phen(phen).fveg, dtype=jnp.asarray(fveg).dtype)


def _cw_area(land_state, phen):
    """CANWATER canopy area ELAI+ESAI (REAL sum as in WRF, then the carry dtype; key on)."""
    p = _cw_phen(phen)
    return jnp.asarray(p.elai + p.esai, dtype=jnp.asarray(land_state.lai).dtype)


__all__ = ["noahmp_water_hydro"]
