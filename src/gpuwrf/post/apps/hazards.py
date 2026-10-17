"""Explicit screening diagnostics; inputs are instantaneous resolved fields."""
from __future__ import annotations
import numpy as np
from ._numeric import finite, thermo


def supercooled_liquid_water(temperature_k, pressure_pa, qv, qc, qr):
    """Supercooled liquid g/m3 (cloud+rain) in -40<T<0 C; flag above zero.

    This is a thermodynamic icing-potential screen, not icing severity,
    accretion, droplet-size prediction, or the UPP IFI algorithm.
    """
    t = finite("temperature_k", temperature_k, minimum=150, maximum=350)
    cloud = finite("qc", qc, minimum=0)
    rain = finite("qr", qr, minimum=0)
    rho_d, _, _ = thermo(t, pressure_pa, qv)
    liquid = 1000 * rho_d * (cloud + rain)
    return np.where((t > 233.15) & (t < 273.15), liquid, 0)


def frost_flags(t2_k, tsk_k, *, air_threshold_c=0, surface_threshold_c=0):
    """Threshold flags at one output time, not crop injury probabilities."""
    air = finite("t2_k", t2_k, minimum=150, maximum=350)
    skin = finite("tsk_k", tsk_k, minimum=150, maximum=350)
    at = float(finite("air_threshold_c", air_threshold_c, minimum=-100, maximum=50))
    st = float(finite("surface_threshold_c", surface_threshold_c, minimum=-100, maximum=50))
    return (air <= 273.15 + at).astype(np.int8), (skin <= 273.15 + st).astype(np.int8)


def visibility(temperature_k, pressure_pa, qv, qc, qr, qi, qs):
    """UPP CALVIS 'R' four-species extinction; result m, ceiling 24135 m.

    Mixing ratios kg/kg dry air. Follows CALVIS density/volume convention,
    RD=287.04, D608=.608, liquid=1000 and ice=917 kg/m3. No aerosol or
    humidity-only extinction; RH is returned separately by the dataset runner.
    """
    t = finite("temperature_k", temperature_k, minimum=150, maximum=350)
    p = finite("pressure_pa", pressure_pa, minimum=1)
    vapor = finite("qv", qv, minimum=0)
    cloud, rain, ice, snow = [finite(n, a) for n, a in
                              (("qc", qc), ("qr", qr), ("qi", qi), ("qs", qs))]
    rho = p / (287.04 * t * (1 + .608 * vapor))
    volume = (1 + vapor) / rho + (cloud + rain) / 1000 + (ice + snow) / 917
    finite("hydrometeor mixture volume", volume, minimum=1e-12)
    # Literal CALVIS R: retain raw q in mixture volume, MAX(0,C) for extinction.
    lc, lp, ic, ip = [np.maximum(0, 1000 * a / volume) for a in (cloud, rain, ice, snow)]
    beta = 144.7 * lc**.88 + 2.24 * lp**.75 + 327.8 * ic + 10.36 * ip**.7776 + 1e-10
    return 1000 * np.minimum(24.135, -np.log(.02) / beta)


def hot_dry_windy(temperature_k, pressure_pa, qv, u, v, height_agl, *, top_m=500):
    """Instantaneous HDW screen: max wind * max surface-adjusted VPD below 500 m.

    Srock et al. (2018): adiabatic dry adjustment to surface pressure, retained
    mixing ratio; separate vertical maxima need not occur at the same level.
    Lowest model level is the pressure reference. Output hPa m/s. Not the daily
    afternoon maximum or percentile-based operational HDWI.
    Vertical axis is last; no levels above top_m are used.
    """
    t = finite("temperature_k", temperature_k, minimum=150, maximum=350)
    p = finite("pressure_pa", pressure_pa, minimum=1)
    q = finite("qv", qv, minimum=0)
    z = finite("height_agl", height_agl, minimum=0)
    uu, vv = finite("u", u), finite("v", v)
    if any(t.shape != a.shape for a in (p, q, z, uu, vv)):
        raise ValueError("HDW profile shapes must match (..., level)")
    if t.ndim < 1 or np.any(np.diff(z, axis=-1) <= 0):
        raise ValueError("HDW heights must increase strictly")
    top = float(finite("top_m", top_m, minimum=1))
    if np.any(z[..., 0] > top) or np.any(z[..., -1] < top):
        raise ValueError("HDW requires a profile spanning the chosen layer top")
    # Drop the upper atmosphere before thermodynamics: adiabatic transport of
    # stratospheric air to the surface is neither needed nor physically this
    # product. A column-dependent mask still excludes extra levels in the crop.
    end = int(np.max(np.sum(z <= top, axis=-1)))
    t, p, q, z, uu, vv = [a[..., :end] for a in (t, p, q, z, uu, vv)]
    mask = z <= top
    adjusted_t = t * (p[..., :1] / p)**(287.05 / 1004.0)
    adjusted_t = np.where(mask, adjusted_t, t[..., :1])
    _, _, deficit = thermo(adjusted_t, p[..., :1], q)
    wind = np.hypot(uu, vv)
    return np.max(np.where(mask, wind, 0), axis=-1) * np.max(np.where(mask, deficit, 0), axis=-1)
