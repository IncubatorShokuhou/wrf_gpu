"""MetPy-based sounding diagnostics; explicit undefined-level status flags."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .common import (PROFILE, field, height_agl, metpy, positive, pressure,
                     profile_wind, temperature, variable)
from .products import FILL


def _surface(ds):
    return (positive(field(ds, "PSFC"), "PSFC"),
            positive(field(ds, "T2"), "T2"), positive(field(ds, "Q2"), "Q2"))


def _dewpoint(calc, units, p, q):
    # WRF QVAPOR/Q2 are mixing ratio (dry mass denominator), not specific humidity.
    return calc.dewpoint(calc.vapor_pressure(p * units.Pa, q * units.dimensionless))


def heat_index_nws(t_kelvin, rh):
    """NWS/WPC Rothfusz + Steadman fallback and low/high RH corrections."""
    t = (t_kelvin - 273.15)*1.8 + 32.
    r = rh*100.
    simple = .5*(t + 61. + (t-68.)*1.2 + r*.094)
    initial = .5*(simple+t)
    regression = (-42.379 + 2.04901523*t + 10.14333127*r - .22475541*t*r
                  - .00683783*t*t - .05481717*r*r + .00122874*t*t*r
                  + .00085282*t*r*r - .00000199*t*t*r*r)
    low = (r < 13.) & (t >= 80.) & (t <= 112.)
    adjustment = (13.-r)/4. * np.sqrt(np.maximum(0., (17.-np.abs(t-95.))/17.))
    regression -= np.where(low, adjustment, 0.)
    high = (r > 85.) & (t >= 80.) & (t <= 87.)
    regression += np.where(high, (r-85.)/10.*(87.-t)/5., 0.)
    return (np.where(initial >= 80., regression, initial)-32.)/1.8+273.15


def thermal(ds, options):
    calc, units = metpy()
    p, t, q = _surface(ds)
    td = _dewpoint(calc, units, p, q)
    rh = calc.relative_humidity_from_mixing_ratio(p*units.Pa, t*units.K, q*units.dimensionless).magnitude
    if np.any((rh < 0) | (rh > 1)):
        raise ValueError("thermal indices require 0 <= surface relative humidity <= 1; no saturation clipping")
    speed = np.hypot(field(ds, "U10"), field(ds, "V10"))
    wet = calc.wet_bulb_temperature(p*units.Pa, t*units.K, td).to("K").magnitude
    apparent = calc.apparent_temperature(t*units.K, rh*units.dimensionless, speed*units("m/s"),
                                         face_level_winds=False, mask_undefined=False).to("K").magnitude
    return xr.Dataset({
        "wet_bulb_temperature_2m": variable(wet, "K", "2 m wet bulb temperature (MetPy Normand method)"),
        "heat_index_2m": variable(heat_index_nws(t, rh), "K", "NWS/WPC heat index using 2 m temperature and humidity",
                                  comment="shade/light-wind assumption; NWS fallback below 80 F"),
        "apparent_temperature": variable(apparent, "K", "MetPy apparent temperature (wind chill/heat index piecewise)",
                                         comment="2 m temperature/humidity and 10 m wind; face_level_winds=False"),
    })


def sounding(ds):
    calc, units = metpy()
    ps, ts, qs = _surface(ds)
    p, t, q, z = pressure(ds), temperature(ds), positive(field(ds, "QVAPOR", PROFILE), "QVAPOR"), height_agl(ds)
    p = np.concatenate([ps[:, None], p], axis=1)
    t = np.concatenate([ts[:, None], t], axis=1)
    q = np.concatenate([qs[:, None], q], axis=1)
    # Treat the 2 m thermodynamic sample as the surface parcel at PSFC and z=0.
    z = np.concatenate([np.zeros_like(z[:, :1]), z], axis=1)
    if np.any(np.diff(p, axis=1) >= 0):
        raise ValueError("sounding pressure must strictly decrease from PSFC")
    td = _dewpoint(calc, units, p, q).to("K").magnitude
    if not np.isfinite(td).all():
        raise ValueError("undefined dewpoint in sounding")
    return p, t, td, z


def instability(ds, options):
    calc, units = metpy()
    p, t, td, z = sounding(ds)
    shape = p.shape[:1]+p.shape[2:]
    arrays = {key: np.full(shape, FILL) for key in ("surface_based_cape", "surface_based_cin", "lcl_pressure", "lfc_pressure", "lifted_index")}
    for it, y, x in np.ndindex(shape):
        pc, tc, dc = p[it, :, y, x]*units.Pa, t[it, :, y, x]*units.K, td[it, :, y, x]*units.K
        cape, cin = calc.surface_based_cape_cin(pc, tc, dc)
        parcel = calc.parcel_profile(pc, tc[0], dc[0])
        lcl, _ = calc.lcl(pc[0], tc[0], dc[0])
        lfc, _ = calc.lfc(pc, tc, dc, parcel_temperature_profile=parcel, which="bottom")
        vals = {"surface_based_cape": float(cape.to("J/kg").magnitude), "surface_based_cin": float(cin.to("J/kg").magnitude),
                "lcl_pressure": float(lcl.to("Pa").magnitude)}
        if np.isfinite(lfc.magnitude):
            vals["lfc_pressure"] = float(lfc.to("Pa").magnitude)
        if pc[-1].magnitude <= 50000. <= pc[0].magnitude:
            vals["lifted_index"] = float(calc.lifted_index(pc, tc, parcel).to("delta_degC").magnitude.item())
        for key, value in vals.items():
            if not np.isfinite(value):
                raise ValueError(f"{key}: non-finite MetPy diagnostic at {(it, y, x)}")
            arrays[key][it, y, x] = value
    result = xr.Dataset()
    for key, a in arrays.items():
        unit = "Pa" if "pressure" in key else "K" if key == "lifted_index" else "J kg-1"
        result[key] = variable(a, unit, key.replace("_", " "),
            parcel="surface PSFC/T2/Q2", method="MetPy surface_based_cape_cin; bottom LFC; LI at 500 hPa")
        result[key].encoding["_FillValue"] = FILL
        if key in ("lfc_pressure", "lifted_index"):
            result[f"{key}_valid"] = variable((a != FILL).astype("i1"), "1", f"{key} is defined in sampled profile")
    # Pressure levels also represented as AGL heights using log-pressure interpolation.
    for level in ("lcl", "lfc"):
        a = arrays[f"{level}_pressure"]
        values = np.full(shape, FILL)
        for it, y, x in np.ndindex(shape):
            target = a[it, y, x]
            if p[it, -1, y, x] <= target <= p[it, 0, y, x]:
                values[it, y, x] = np.interp(np.log(target), np.log(p[it, ::-1, y, x]), z[it, ::-1, y, x])
        result[f"{level}_height_agl"] = variable(values, "m", f"{level.upper()} height above ground (log pressure interpolation)")
        result[f"{level}_height_agl"].encoding["_FillValue"] = FILL
        result[f"{level}_height_valid"] = variable((values != FILL).astype("i1"), "1", f"{level.upper()} is inside model vertical coverage")
    return result


def srh(ds, options):
    calc, units = metpy()
    if "storm_u" not in options or "storm_v" not in options:
        raise ValueError("srh requires explicit storm_u and storm_v (earth-relative m/s); use 0,0 for a stationary storm")
    storm_u, storm_v = float(options["storm_u"]), float(options["storm_v"])
    if not np.isfinite([storm_u, storm_v]).all():
        raise ValueError("storm motion must be finite")
    z, u, v = profile_wind(ds, options)
    # Constant wind in 0–10 m; upper heights retain model mass samples.
    z = np.concatenate([np.zeros_like(z[:, :1]), z], axis=1)
    u, v = np.concatenate([u[:, :1], u], axis=1), np.concatenate([v[:, :1], v], axis=1)
    shape = z.shape[:1]+z.shape[2:]
    result = xr.Dataset()
    for depth in (1000, 3000):
        if np.any(z[:, -1] < depth):
            raise ValueError(f"model profile does not reach {depth} m AGL for SRH")
        values = np.empty(shape)
        for it, y, x in np.ndindex(shape):
            _, _, total = calc.storm_relative_helicity(z[it, :, y, x]*units.m, u[it, :, y, x]*units("m/s"), v[it, :, y, x]*units("m/s"),
                depth*units.m, bottom=0*units.m, storm_u=storm_u*units("m/s"), storm_v=storm_v*units("m/s"))
            values[it, y, x] = float(total.to("m^2/s^2").magnitude)
        result[f"srh_0_{depth//1000}km"] = variable(values, "m2 s-2", f"signed storm relative helicity 0–{depth} m AGL",
            storm_eastward_velocity=storm_u, storm_northward_velocity=storm_v,
            comment="10 m winds held constant to ground; MetPy line integral with interpolated layer top")
    return result


def first_zero(z, t):
    """Lowest 273.15 K crossing; all-cold surface is zero, no crossing is fill."""
    shape = t.shape[:1]+t.shape[2:]
    values, status = np.full(shape, FILL), np.zeros(shape, dtype="i1")
    for it, y, x in np.ndindex(shape):
        tc, zc = t[it, :, y, x]-273.15, z[it, :, y, x]
        if tc[0] <= 0:
            values[it, y, x], status[it, y, x] = 0., 2
            continue
        crossings = np.flatnonzero((tc[:-1] > 0) & (tc[1:] <= 0))
        if len(crossings):
            k = crossings[0]
            values[it, y, x] = zc[k] + (zc[k+1]-zc[k])*tc[k]/(tc[k]-tc[k+1])
            status[it, y, x] = 1
    return values, status


def freezing(ds, options):
    calc, units = metpy()
    p, t, td, z = sounding(ds)
    wet = calc.wet_bulb_temperature(p*units.Pa, t*units.K, td*units.K).to("K").magnitude
    if not np.isfinite(wet).all():
        raise ValueError("non-finite wet bulb profile")
    result = xr.Dataset()
    for key, profile in (("freezing_level", t), ("wet_bulb_zero", wet)):
        values, status = first_zero(z, profile)
        result[f"{key}_height_agl"] = variable(values, "m", f"lowest {key.replace('_', ' ')} height above ground",
            comment="linear height interpolation; surface <=273.15 K gives zero; unresolved crossing is missing")
        result[f"{key}_height_agl"].encoding["_FillValue"] = FILL
        result[f"{key}_status"] = variable(status, "1", f"{key} coverage status", flag_values=np.array([0, 1, 2], dtype="i1"),
            flag_meanings="above_model_top_or_absent resolved_crossing surface_at_or_below_freezing")
    return result
