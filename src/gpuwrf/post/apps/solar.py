"""Solar irradiance and simple PVWatts DC, independently testable vs pvlib.

SPA geometry uses the optional pvlib package. Erbs and isotropic transposition
are explicit NumPy formulas; no aerosol, horizon, tracking or spectral model.
"""
from __future__ import annotations

import numpy as np
from ._numeric import finite


def erbs(ghi, zenith_deg, day_of_year):
    """Erbs et al. (1982) diffuse fraction, pvlib's default 87-degree cutoff.

    GHI/DNI/DHI W/m2. Uses Spencer extraterrestrial irradiance (1366.1 W/m2).
    At zenith >87 degrees DNI=0, DHI=GHI to avoid horizon division artifacts.
    """
    g = finite("ghi", ghi, minimum=0)
    z = finite("zenith_deg", zenith_deg, minimum=0, maximum=180)
    day = finite("day_of_year", day_of_year, minimum=1, maximum=366)
    angle = 2 * np.pi * (day - 1) / 365
    extra = 1366.1 * (1.00011 + .034221 * np.cos(angle) + .00128 * np.sin(angle)
                      + .000719 * np.cos(2 * angle) + .000077 * np.sin(2 * angle))
    cosz = np.cos(np.deg2rad(z))
    kt = np.clip(g / (extra * np.maximum(cosz, .065)), 0, 1)
    df = np.where(kt <= .22, 1 - .09 * kt,
                  np.where(kt <= .8, .9511 - .1604 * kt + 4.388 * kt**2
                           - 16.638 * kt**3 + 12.336 * kt**4, .165))
    diffuse = df * g
    direct = (g - diffuse) / np.maximum(cosz, .001)
    bad = (z > 87) | (direct < 0)
    return np.where(bad, 0, direct), np.where(bad, g, diffuse)


def plane_of_array(ghi, dni, dhi, zenith_deg, solar_azimuth_deg, *, tilt=30, azimuth=180, albedo=.2):
    """Isotropic sky + Lambertian ground, north=0/east=90 azimuth, W/m2."""
    g = finite("ghi", ghi, minimum=0)
    n = finite("dni", dni, minimum=0)
    d = finite("dhi", dhi, minimum=0)
    z = np.deg2rad(finite("zenith_deg", zenith_deg, minimum=0, maximum=180))
    a = np.deg2rad(finite("solar_azimuth_deg", solar_azimuth_deg, minimum=0, maximum=360))
    beta = np.deg2rad(float(finite("tilt", tilt, minimum=0, maximum=180)))
    gamma = np.deg2rad(float(finite("azimuth", azimuth, minimum=0, maximum=360)))
    alb = float(finite("albedo", albedo, minimum=0, maximum=1))
    projection = np.cos(z) * np.cos(beta) + np.sin(z) * np.sin(beta) * np.cos(a - gamma)
    beam = n * np.maximum(0, projection) * (z < np.pi / 2)
    return beam + d * (1 + np.cos(beta)) / 2 + g * alb * (1 - np.cos(beta)) / 2


def pvwatts_dc(poa_wm2, cell_temperature_c, *, dc_rating_w=1000, gamma_pdc=-.004):
    """NREL PVWatts: Pdc=Pdc0*(POA/1000)*(1+gamma*(Tcell-25)), W.

    POA is treated as effective irradiance (no IAM or spectral losses).
    """
    poa = finite("poa_wm2", poa_wm2, minimum=0)
    temp = finite("cell_temperature_c", cell_temperature_c, minimum=-100, maximum=150)
    rating = float(finite("dc_rating_w", dc_rating_w, minimum=1e-12))
    gamma = float(finite("gamma_pdc", gamma_pdc, minimum=-.02, maximum=0))
    return np.maximum(0, rating * poa / 1000 * (1 + gamma * (temp - 25)))


def solar_position(times, latitude, longitude, *, altitude=0):
    """NREL SPA via pvlib (UTC, true zenith for Erbs; apparent for POA).

    Returns zenith/apparent zenith/azimuth arrays shaped (time, *lat.shape).
    Geometry is evaluated one timestamp at a time to bound temporary memory.
    """
    try:
        import pandas as pd
        from pvlib.solarposition import get_solarposition
    except ImportError as exc:
        raise ImportError('solar geometry requires pip install "gpuwrf[apps]"') from exc
    lat = finite("latitude", latitude, minimum=-90, maximum=90)
    lon = finite("longitude", longitude, minimum=-180, maximum=180)
    lat, lon = np.broadcast_arrays(lat, lon)
    alt = np.broadcast_to(finite("altitude", altitude), lat.shape)
    stamps = pd.DatetimeIndex(times)
    if stamps.hasnans or stamps.empty:
        raise ValueError("solar geometry requires valid UTC timestamps")
    stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")
    rows = []
    for stamp in stamps:
        pos = get_solarposition(pd.DatetimeIndex([stamp] * lat.size), lat.ravel(), lon.ravel(),
                                altitude=alt.ravel(), method="nrel_numpy")
        rows.append(np.stack([pos[key].to_numpy().reshape(lat.shape)
                              for key in ("zenith", "apparent_zenith", "azimuth")]))
    result = finite("SPA result", np.stack(rows))
    return result[:, 0], result[:, 1], result[:, 2]
