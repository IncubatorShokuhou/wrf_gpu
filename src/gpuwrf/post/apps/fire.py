"""Classic daily Canadian FWI (Van Wagner & Pickett 1985 equations).

NumPy implementation of the published equations, independent of CFFDRS code.
Requires local standard noon weather, preceding 24h rainfall and previous codes.
Not the distinct FWI2025 system, overwintering, or a fire-spread model.
Uses CFFDRS's continuous FFMC moisture coefficient 250*59.5/101,
instead of its historical rounded 147.2, consistently in FFMC and ISI.
"""
from __future__ import annotations
import numpy as np
from ._numeric import finite

# Published latitude/month factors (January through December).
_DMC_N = np.array([6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6])
_DMC_LOW_N = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])
_DMC_LOW_S = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2])
_DMC_S = np.array([11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8])
_DC_N = np.array([-1.6, -1.6, -1.6, .9, 3.8, 5.8, 6.4, 5, 2.4, .4, -1.6, -1.6])
_DC_S = np.array([6.4, 5, 2.4, .4, -1.6, -1.6, -1.6, -1.6, -1.6, .9, 3.8, 5.8])
_FFMC_MOISTURE = 250.0 * 59.5 / 101.0


def daily_fwi(temperature_c, rh_percent, wind_kmh, rain_mm, *, latitude, month,
              ffmc_previous, dmc_previous, dc_previous):
    """One daily update; all inputs broadcast, output seven unitless components."""
    t = finite("temperature_c", temperature_c, minimum=-80, maximum=60)
    rh = finite("rh_percent", rh_percent, minimum=0, maximum=100)
    w = finite("wind_kmh", wind_kmh, minimum=0, maximum=500)
    rain = finite("rain_mm", rain_mm, minimum=0)
    lat = finite("latitude", latitude, minimum=-90, maximum=90)
    mon = finite("month", month, minimum=1, maximum=12)
    if np.any(mon != np.floor(mon)):
        raise ValueError("month must be an integer")
    mi = mon.astype(int) - 1
    f0 = finite("ffmc_previous", ffmc_previous, minimum=0, maximum=101)
    d0 = finite("dmc_previous", dmc_previous, minimum=0)
    c0 = finite("dc_previous", dc_previous, minimum=0)
    # FFMC: moisture, intercepted rain, wetting/drying equilibrium and rate.
    moisture = _FFMC_MOISTURE * (101 - f0) / (59.5 + f0)
    effective = np.maximum(rain - .5, 1e-12)
    wetted = moisture + 42.5 * effective * np.exp(-100 / (251 - moisture)) * (-np.expm1(-6.93 / effective))
    wetted += .0015 * np.maximum(moisture - 150, 0)**2 * np.sqrt(effective)
    moisture = np.where(rain > .5, np.minimum(250, wetted), moisture)
    equilibrium_term = .18 * (21.1 - t) * (-np.expm1(-.115 * rh))
    dry_equilibrium = .942 * rh**.679 + 11 * np.exp((rh - 100) / 10) + equilibrium_term
    wet_equilibrium = .618 * rh**.753 + 10 * np.exp((rh - 100) / 10) + equilibrium_term
    def rate(humidity):
        return (.424 * (1 - humidity**1.7) + .0694 * np.sqrt(w) * (1 - humidity**8)) * .581 * np.exp(.0365 * t)
    drying = dry_equilibrium + (moisture - dry_equilibrium) * 10**(-rate(rh / 100))
    wetting = wet_equilibrium - (wet_equilibrium - moisture) * 10**(-rate((100 - rh) / 100))
    moisture = np.where(moisture > dry_equilibrium, drying,
                        np.where(moisture < wet_equilibrium, wetting, moisture))
    ffmc = np.clip(59.5 * (250 - moisture) / (_FFMC_MOISTURE + moisture), 0, 101)
    # DMC: wetting response followed by monthly/latitude-adjusted drying.
    length = np.select([lat > 30, lat > 10, lat > -10, lat > -30],
                       [_DMC_N[mi], _DMC_LOW_N[mi], 9, _DMC_LOW_S[mi]], _DMC_S[mi])
    drying_dmc = 1.894e-4 * (np.maximum(t, -1.1) + 1.1) * (100 - rh) * length
    b = np.where(d0 <= 33, 100 / (.5 + .3 * d0),
                 np.where(d0 <= 65, 14 - 1.3 * np.log(np.maximum(d0, 1)),
                          6.2 * np.log(np.maximum(d0, 1)) - 17.2))
    net_rain = np.maximum(.92 * rain - 1.27, 0)
    wet_m = 20 + 280 * np.exp(-.023 * d0) + 1000 * net_rain / (48.77 + b * net_rain)
    # Algebraically identical rain equation; tiny protects log in inactive dry cells.
    after_rain = np.maximum(0, 43.43 * (5.6348 - np.log(np.maximum(wet_m - 20, 1e-300))))
    dmc = np.where(rain > 1.5, after_rain, d0) + drying_dmc
    # DC: deep-fuel wetting and potential evapotranspiration.
    seasonal = np.where(lat > 20, _DC_N[mi], np.where(lat <= -20, _DC_S[mi], 1.4))
    evap = np.maximum(0, .5 * (.36 * (np.maximum(t, -2.8) + 2.8) + seasonal))
    recharge = np.maximum(.83 * rain - 1.27, 0)
    # logaddexp avoids an avoidable overflow for extreme previous drought codes.
    logratio = np.log(np.maximum(3.937 * recharge / 800, 1e-300)) + c0 / 400
    soaked_dc = np.maximum(0, c0 - 400 * np.logaddexp(0, logratio))
    dc = np.where(rain > 2.8, soaked_dc, c0) + evap
    # ISI, BUI and FWI (classic equations 24-30).
    m = _FFMC_MOISTURE * (101 - ffmc) / (59.5 + ffmc)
    isi = .208 * np.exp(.05039 * w) * 91.9 * np.exp(-.1386 * m) * (1 + m**5.31 / 4.93e7)
    denom = dmc + .4 * dc
    bui_low = np.divide(.8 * dmc * dc, denom, out=np.zeros_like(denom), where=denom > 0)
    bui_high = dmc - (1 - np.divide(bui_low, dmc, out=np.zeros_like(denom), where=dmc > 0)) * (.92 + (.0114 * dmc)**1.7)
    bui = np.maximum(0, np.where(dmc <= .4 * dc, bui_low, bui_high))
    fuel = np.where(bui <= 80, .626 * bui**.809 + 2, 1000 / (25 + 108.64 * np.exp(-.023 * bui)))
    intermediate = .1 * isi * fuel
    fwi = np.where(intermediate <= 1, intermediate,
                   np.exp(2.72 * (.434 * np.log(np.maximum(1, intermediate)))**.647))
    result = dict(ffmc=ffmc, dmc=dmc, dc=dc, isi=isi, bui=bui, fwi=fwi, dsr=.0272 * fwi**1.77)
    return {name: finite(name, value) for name, value in result.items()}
