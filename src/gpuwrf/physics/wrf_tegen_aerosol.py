"""WRF ``aer_opt=1`` Tegen aerosol climatology for RRTMG shortwave (WRF v4.7.1).

Direct REAL (float32) transcription of the WRF chain

* ``module_physics_init.F::aerosol_in`` (:5221-5360, domain 1 only, once at init): bilinear
  latitude/longitude interpolation (``interp_vec`` :5407-5467, periodic in BOTH axes) of the six
  ECMWF/Tegen aerosol types (organic carbon, sea salt, dust, black carbon, sulfate, stratospheric)
  from ``run/aerosol*.formatted`` -> ``aerodm(i, 12 levels, j, 12 months, 6)`` in AOD per Pa;
* ``module_radiation_driver.F::aer_time_int`` (:5232-5339, every radiation call on domain 1):
  the ``ozn_time_int`` month factors on ``intJULIAN = JULIAN + 1``;
* ``module_radiation_driver.F::aer_p_int`` (:5341-5503): pressure interpolation onto the model
  mass levels (``P = p_hyd``, ``P8W = p_hyd_w``, module_first_rk_step_part1.F:280) times the layer
  pressure thickness -> per-layer AOD ``aerod(i, k, j, 6)``;
* ``module_ra_rrtmg_sw.F::rrtmg_sw`` ``iaer = 6`` (:9438-9462) with the ``swaerpr`` band tables
  (:5041-5131) -> per band ``ztaua/zomga/zasya``; RRTMG_SWRAD zeroes the extra top layer (:11455).

Nests get the parent's ``aerod`` through Registry ``df=(p2c)`` (nesting/nest_aerosol.py).  The
longwave scheme has no aerosol outside WRF-Chem.  Nothing here runs unless a domain selects
``aer_opt=1`` (default 0), so the released programs are unchanged.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.assets import asset_path, ensure_asset

TEGEN_ASSET = asset_path("data/fixtures/wrf-tegen-aerosol-v1.npz")
TEGEN_ASSET_SHA256 = "47cf804c1f4d5b6273d1897221b5a40a67637f055fd38a5f78d5be471cdf6df1"
NAEREC = 6
NMONTHS = 12
NLEV = 12
_DATE_OZ = np.asarray([16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350], dtype=np.float32)

# module_ra_rrtmg_sw.F::swaerpr (:5045-5130), rows = RRTMG SW bands 16..29, columns = the six types.
_TAUA_A = (0.10849, 0.66699, 0.65255, 0.11600, 0.06529, 0.04468)
_TAUA_B = (0.20543, 0.84642, 0.84958, 0.21673, 0.28270, 0.10915)
_TAUA_C = (0.52838, 0.93285, 0.93449, 0.53078, 0.67148, 0.46608)
_TAUA_D = (1.69446, 1.11855, 1.09212, 1.72145, 1.03858, 1.12044)
_PIZA_A = (.5230504, .7868518, .8531531, .4048149, .8748231, .2355667)
_PIZA_B = (.8287144, .9949396, .9279543, .6765051, .9467578, .9955938)
_PIZA_C = (.8970131, .9984940, .9245594, .7768385, .9532763, .9999999)
_PIZA_D = (.9148907, .9956173, .7504584, .8131335, .9401905, .9999999)
_ASYA_A = (0.700610, 0.818871, 0.702399, 0.689886, .4629866, .1907639)
_ASYA_B = (0.636342, 0.802467, 0.691305, 0.627497, .6105750, .4760794)
_ASYA_C = (0.668431, 0.788530, 0.698682, 0.657422, .6735182, .6519706)
_ASYA_D = (0.729019, 0.803129, 0.784592, 0.712208, .7008249, .7270548)
_BAND_GROUP = "AABBBBBCCDDDDA"  # band 1-2 A, 3-7 B, 8-9 C, 10-13 D, 14 A
RSRTAUA = np.asarray([dict(A=_TAUA_A, B=_TAUA_B, C=_TAUA_C, D=_TAUA_D)[g] for g in _BAND_GROUP], dtype=np.float32)
RSRPIZA = np.asarray([dict(A=_PIZA_A, B=_PIZA_B, C=_PIZA_C, D=_PIZA_D)[g] for g in _BAND_GROUP], dtype=np.float32)
RSRASYA = np.asarray([dict(A=_ASYA_A, B=_ASYA_B, C=_ASYA_C, D=_ASYA_D)[g] for g in _BAND_GROUP], dtype=np.float32)


class TegenAerosolStatic(NamedTuple):
    """Domain-1 ``aerodm`` (aerosol_in output) and the table pressure levels ``pina`` (hPa).

    ``aerodm`` is ``(ny, nx, 12 months, 12 levels, 6 types)`` float32 AOD per Pa; a traced pytree
    child of the radiation namelist (never a compile constant).
    """

    aerodm: object
    pina_hpa: object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_wrf_tegen_aerosol_asset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(pina_hpa (12,), lat_od (46,), lon_od (72,), aerodin [type, month, lat, level, lon])`` float32."""

    asset = ensure_asset(TEGEN_ASSET)
    if _sha256(asset) != TEGEN_ASSET_SHA256:
        raise RuntimeError(f"WRF Tegen aerosol asset hash drift: {asset}")
    with np.load(asset, allow_pickle=False) as archive:
        if set(archive.files) != {"pressure_hpa", "latitude_deg", "longitude_deg", "aerodin"}:
            raise RuntimeError(f"WRF Tegen aerosol asset schema drift: {archive.files}")
        values = tuple(np.asarray(archive[name], dtype=np.float32)
                       for name in ("pressure_hpa", "latitude_deg", "longitude_deg", "aerodin"))
    if [v.shape for v in values] != [(12,), (46,), (72,), (6, 12, 46, 12, 72)]:
        raise RuntimeError(f"WRF Tegen aerosol asset shape drift: {[v.shape for v in values]}")
    return values


def _interp_vec(locvec: np.ndarray, locwant: np.ndarray):
    """``interp_vec(locvec, locwant, periodic=.true., ...)`` (module_physics_init.F:5407), REAL, 0-based."""

    locvec = np.asarray(locvec, np.float32)
    want = np.asarray(locwant, np.float32)
    n = locvec.size
    inside = (locvec[None, :-1] <= want.reshape(-1, 1)) & (locvec[None, 1:] > want.reshape(-1, 1))
    found = inside.any(axis=1)
    first = np.argmax(inside, axis=1)
    loc1 = np.where(found, first, n - 1)
    loc2 = np.where(found, first + 1, 0)
    below = want.reshape(-1) < locvec[0]
    v1 = np.where(found, locvec[loc1], np.where(below, locvec[n - 1] - np.float32(360.0), locvec[n - 1]))
    v2 = np.where(found, locvec[loc2], np.where(below, locvec[0], locvec[0] + np.float32(360.0)))
    v1, v2 = v1.astype(np.float32), v2.astype(np.float32)
    w2 = ((want.reshape(-1) - v1) / (v2 - v1)).astype(np.float32)
    w1 = (np.float32(1.0) - w2).astype(np.float32)
    shape = want.shape
    return loc1.reshape(shape), loc2.reshape(shape), w1.reshape(shape), w2.reshape(shape)


def wrf_aerosol_in(xlat_deg, xlong_deg) -> np.ndarray:
    """Host ``aerosol_in``: ``aerodm`` ``(ny, nx, 12, 12, 6)`` float32 for mass-point XLAT/XLONG ``(ny, nx)``."""

    _pina, lat_od, lon_od, aerodin = _load_wrf_tegen_aerosol_asset()
    lat = np.asarray(xlat_deg, np.float32)
    lon = np.asarray(xlong_deg, np.float32)
    if lat.shape != lon.shape or lat.ndim != 2 or not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError(f"aerosol_in needs finite 2-D XLAT/XLONG of one shape, got {lat.shape} {lon.shape}")
    jlat1, jlat2, wlat1, wlat2 = _interp_vec(lat_od, lat)
    ilon1, ilon2, wlon1, wlon2 = _interp_vec(lon_od, lon)
    table = np.moveaxis(aerodin, (2, 4), (0, 1))  # [lat, lon, type, month, level]

    def corner(j, i):
        return table[j, i]  # (ny, nx, type, month, level)

    w = lambda x: x[..., None, None, None]  # noqa: E731
    out = (w(wlon1) * (w(wlat1) * corner(jlat1, ilon1) + w(wlat2) * corner(jlat2, ilon1))
           + w(wlon2) * (w(wlat1) * corner(jlat1, ilon2) + w(wlat2) * corner(jlat2, ilon2)))
    return np.ascontiguousarray(np.moveaxis(out.astype(np.float32), 2, 4))  # (ny, nx, month, level, type)


def build_tegen_aerosol_static(xlat_deg, xlong_deg) -> TegenAerosolStatic:
    """Domain-1 static: host ``aerosol_in`` on the supplied mass-point XLAT/XLONG plus ``pina``."""

    pina = _load_wrf_tegen_aerosol_asset()[0]
    return TegenAerosolStatic(jnp.asarray(wrf_aerosol_in(xlat_deg, xlong_deg)), jnp.asarray(pina))


def bind_tegen_aerosol(namelist):
    """Domain-1 ``aer_opt=1`` binding: ``tegen_aerosol`` from the namelist's supplied WRF XLAT/XLONG."""

    from dataclasses import replace

    static = getattr(namelist, "radiation_static", None)
    if static is None or getattr(static, "xlat_deg", None) is None:
        raise ValueError("aer_opt=1 needs the real WRF XLAT/XLONG (radiation_static) of domain 1")
    return replace(namelist, tegen_aerosol=build_tegen_aerosol_static(
        np.asarray(static.xlat_deg, np.float32), np.asarray(static.xlong_deg, np.float32)))


def wrf_aer_time_factors(julian0):
    """``aer_time_int`` month indices (0-based nm, np) and REAL factors from the 0-based REAL ``JULIAN``."""

    f = jnp.float32
    intjulian = jnp.asarray(julian0, f) + f(1.0)
    ijul = intjulian.astype(jnp.int32)
    intjulian = intjulian - ijul.astype(f)
    ijul = jnp.mod(ijul, jnp.int32(365))
    ijul = jnp.where(ijul == 0, jnp.int32(365), ijul)
    intjulian = intjulian + ijul.astype(f)
    dates = jnp.asarray(_DATE_OZ, f)
    after = dates > intjulian
    np1 = jnp.where(jnp.any(after), jnp.argmax(after), 0).astype(jnp.int32)
    nm = jnp.where(np1 > 0, np1 - 1, 11).astype(jnp.int32)
    cdayozp, cdayozm = jnp.take(dates, np1), jnp.take(dates, nm)
    days = f(365.0)
    wrap = np1 == 0
    deltat = jnp.where(wrap, cdayozp + days - cdayozm, cdayozp - cdayozm)
    december = intjulian > cdayozp
    fact1 = jnp.where(wrap, jnp.where(december, (cdayozp + days - intjulian) / deltat, (cdayozp - intjulian) / deltat),
                      (cdayozp - intjulian) / deltat)
    fact2 = jnp.where(wrap, jnp.where(december, (intjulian - cdayozm) / deltat, (intjulian + days - cdayozm) / deltat),
                      (intjulian - cdayozm) / deltat)
    return nm, np1, fact1.astype(f), fact2.astype(f)


def wrf_tegen_aerod(static: TegenAerosolStatic, p_hyd_pa, p_hyd_w_pa, julian0):
    """``aer_time_int`` + ``aer_p_int``: per-layer AOD ``(ny, nx, nz, 6)`` float32.

    ``p_hyd_pa`` ``(ny, nx, nz)`` / ``p_hyd_w_pa`` ``(ny, nx, nz+1)`` are the radiation P / P8W columns
    (bottom-up, Pa); ``julian0`` is WRF's 0-based REAL ``grid%julian`` of the radiation call.
    """

    f = jnp.float32
    aerodm = jnp.asarray(static.aerodm, f)
    nm, np1, fact1, fact2 = wrf_aer_time_factors(julian0)
    aerodt = jnp.take(aerodm, nm, axis=-3) * fact1 + jnp.take(aerodm, np1, axis=-3) * fact2  # (ny,nx,12lev,6)
    return wrf_aer_p_int(aerodt, p_hyd_pa, p_hyd_w_pa, static.pina_hpa)


def wrf_aer_p_int(aerodt, p_hyd_pa, p_hyd_w_pa, pina_hpa):
    """``aer_p_int``: ``aerodt (..., 12, 6)`` AOD per Pa on the table levels -> per-layer AOD ``(..., nz, 6)``."""

    f = jnp.float32
    aerodt = jnp.asarray(aerodt, f)
    pin = jnp.asarray(pina_hpa, f)
    pmid = jnp.asarray(p_hyd_pa, f) * f(0.01)
    pf = jnp.asarray(p_hyd_w_pa, f)
    dpm = (pf[..., :-1] - pf[..., 1:])[..., None]
    # Pointwise equivalent of the top-down kupper search: pin(ku) < pmid <= pin(ku+1); pmid == pin(1)
    # keeps WRF's initial/stale kupper = 1 (the searchsorted clip below).
    ku = jnp.clip(jnp.searchsorted(pin, pmid, side="left") - 1, 0, NLEV - 2)
    lower = jnp.take_along_axis(aerodt[..., None, :, :], ku[..., None, None], axis=-2)[..., 0, :]
    upper = jnp.take_along_axis(aerodt[..., None, :, :], (ku + 1)[..., None, None], axis=-2)[..., 0, :]
    p0 = jnp.take(pin, ku)[..., None]
    p1 = jnp.take(pin, ku + 1)[..., None]
    pm = pmid[..., None]
    dpu = pm - p0
    dpl = p1 - pm
    inside = (lower * dpl + upper * dpu) / (dpl + dpu)
    top = aerodt[..., None, 0, :] * pm / pin[0]
    bottom = jnp.broadcast_to(aerodt[..., None, NLEV - 1, :], inside.shape)
    value = jnp.where(pm < pin[0], top, jnp.where(pm > pin[NLEV - 1], bottom, inside))
    return (value * dpm).astype(f)


def ecaer_band_optics(ecaer):
    """``rrtmg_sw`` ``iaer = 6``: per-band ``(ztaua, zomga, zasya)`` ``(..., nlay, 14)`` from ``ecaer (..., nlay, 6)``."""

    f = jnp.float32
    ec = jnp.asarray(ecaer, f)
    taus, omgs, asys = [], [], []
    for ib in range(RSRTAUA.shape[0]):
        ztaua = zomga = zasya = None
        for ia in range(NAEREC):
            x = f(RSRTAUA[ib, ia]) * ec[..., ia]
            y = x * f(RSRPIZA[ib, ia])
            z = y * f(RSRASYA[ib, ia])
            ztaua = x if ztaua is None else ztaua + x
            zomga = y if zomga is None else zomga + y
            zasya = z if zasya is None else zasya + z
        zasya = jnp.where(zomga != f(0.0), zasya / jnp.where(zomga != f(0.0), zomga, f(1.0)), zasya)
        zomga = jnp.where(ztaua != f(0.0), zomga / jnp.where(ztaua != f(0.0), ztaua, f(1.0)), zomga)
        taus.append(ztaua)
        omgs.append(zomga)
        asys.append(zasya)
    return jnp.stack(taus, axis=-1), jnp.stack(omgs, axis=-1), jnp.stack(asys, axis=-1)


__all__ = [
    "NAEREC", "RSRASYA", "RSRPIZA", "RSRTAUA", "TEGEN_ASSET", "TEGEN_ASSET_SHA256", "TegenAerosolStatic",
    "bind_tegen_aerosol", "build_tegen_aerosol_static", "ecaer_band_optics", "wrf_aer_p_int", "wrf_aer_time_factors", "wrf_aerosol_in",
    "wrf_tegen_aerod",
]
