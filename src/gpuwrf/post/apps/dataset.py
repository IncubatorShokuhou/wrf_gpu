"""WRF/xarray adapters for application products; all model work stays untouched."""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..common import PROFILE, SURFACE, field, height_agl, pressure, temperature, variable
from ._numeric import finite, thermo
from .wind import interpolate_profile, rotor_wind, read_power_curve, wind_power, sampled_energy_wh
from .solar import erbs, solar_position, plane_of_array, pvwatts_dc
from .hazards import supercooled_liquid_water, frost_flags, visibility, hot_dry_windy
from .fire import daily_fwi


def _options(options, allowed):
    # Core options are shared among all requested products; only check values used.
    return {k: v for k, v in options.items() if k in allowed}


def _times(ds):
    if "Time" not in ds.coords or not np.issubdtype(ds.Time.dtype, np.datetime64):
        raise ValueError("application products need a decoded UTC Time coordinate (read_wrfout)")
    times = ds.Time.values.astype("datetime64[ns]")
    if times.size == 0 or np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Time must contain valid strictly increasing UTC timestamps")
    return times


def _result(ds, values, **attrs):
    # Retain geographic coordinates and a consistent horizontal grid.
    result = xr.Dataset(values, coords={"Time": ds.Time})
    for name in ("XLAT", "XLONG"):
        if name in ds:
            result = result.assign_coords({name: ds[name]})
    result.attrs.update(application_product_version="1", **attrs)
    return result


def _winds(ds):
    u = field(ds, "U", ("Time", "bottom_top", "south_north", "west_east_stag"))
    v = field(ds, "V", ("Time", "bottom_top", "south_north_stag", "west_east"))
    return (u[..., :-1] + u[..., 1:]) / 2, (v[:, :, :-1] + v[:, :, 1:]) / 2


def _last(a):
    return np.moveaxis(a, 1, -1)


def wind_product(ds, options):
    opts = _options(options, {"hub_height", "rotor_diameter", "samples", "veer"})
    if "veer" in opts and not isinstance(opts["veer"], bool):
        raise ValueError("veer must be a JSON boolean")
    u, v = _winds(ds)
    z = height_agl(ds)
    # A 10m anchor extends only wind, never thermodynamic profiles.
    if np.any(z[:, 0] <= 10):
        raise ValueError("lowest mass level must exceed the 10m anchor")
    zw = np.concatenate([np.full_like(z[:, :1], 10), z], axis=1)
    uw = np.concatenate([field(ds, "U10")[:, None], u], axis=1)
    vw = np.concatenate([field(ds, "V10")[:, None], v], axis=1)
    hub, rews = rotor_wind(_last(zw), _last(uw), _last(vw), **opts)
    values = {"hub_wind_speed": variable(hub, "m s-1", "Hub-height interpolated wind speed"),
              "rotor_equivalent_wind_speed": variable(rews, "m s-1", "Area-cubic rotor-equivalent wind speed")}
    if "power_curve" in options:
        t, p, q = temperature(ds), pressure(ds), field(ds, "QVAPOR", PROFILE)
        density_d, _, _ = thermo(t, p, q)
        rho = interpolate_profile(_last(z), _last(density_d * (1 + q)), [opts.get("hub_height", 100)])[..., 0]
        speed, power = read_power_curve(options["power_curve"])
        output = wind_power(rews, rho, speed, power, reference_density=options.get("reference_density", 1.225))
        values["wind_power"] = variable(output, "W", "Power-curve estimate with equivalent-speed density correction")
        stamps = _times(ds)
        seconds = (stamps - stamps[0]) / np.timedelta64(1, "s")
        values["wind_energy"] = variable(sampled_energy_wh(output, seconds), "W h", "Cumulative trapezoidal sampled wind energy",
                                          cell_methods="time: sum (trapezoidal integration of sampled power)")
        values["hub_air_density"] = variable(rho, "kg m-3", "Moist-air density at hub height")
    return _result(ds, values, method="IEC-style sampled REWS; scalar speed and optional hub-direction projection",
                   limitations="Instantaneous samples; no turbulence, wakes or IEC measurement certification",
                   hub_height_m=float(opts.get("hub_height", 100)), rotor_diameter_m=float(opts.get("rotor_diameter", 100)),
                   rotor_samples=int(opts.get("samples", 9)), veer_projection=int(opts.get("veer", True)))


def solar_product(ds, options):
    stamps = _times(ds)
    g = finite("SWDOWN", field(ds, "SWDOWN"), minimum=0)
    lat, lon, altitude = field(ds, "XLAT"), field(ds, "XLONG"), field(ds, "HGT")
    if any(not np.array_equal(a, np.broadcast_to(a[:1], a.shape)) for a in (lat, lon, altitude)):
        raise ValueError("solar geometry requires a stationary grid")
    z, apparent, az = solar_position(stamps, lat[0], lon[0], altitude=altitude[0])
    present = [name in ds for name in ("SWDDNI", "SWDDIF")]
    if any(present) and not all(present):
        raise ValueError("supply both SWDDNI and SWDDIF, or neither for Erbs decomposition")
    if all(present):
        n = finite("SWDDNI", field(ds, "SWDDNI"), minimum=0)
        d = finite("SWDDIF", field(ds, "SWDDIF"), minimum=0)
        method = "WRF SWDDNI/SWDDIF"
    else:
        import pandas as pd
        day = pd.DatetimeIndex(stamps).dayofyear.to_numpy()[:, None, None]
        n, d = erbs(g, z, day)
        method = "Erbs-1982 decomposition of SWDOWN (estimated DNI/DHI)"
    poa = plane_of_array(g, n, d, apparent, az,
                         **_options(options, {"tilt", "azimuth", "albedo"}))
    air_c = finite("T2", field(ds, "T2"), minimum=150, maximum=350) - 273.15
    if "cell_temperature_c" in options:
        cell = np.broadcast_to(finite("cell_temperature_c", options["cell_temperature_c"], minimum=-100, maximum=150), g.shape)
        thermal = "user-supplied cell temperature"
    else:
        noct = float(finite("noct_c", options.get("noct_c", 45), minimum=20, maximum=100))
        cell = air_c + (noct - 20) * poa / 800
        thermal = "NOCT approximation: Tcell=T2+(NOCT-20)*POA/800, no wind cooling"
    dc = pvwatts_dc(poa, cell, **_options(options, {"dc_rating_w", "gamma_pdc"}))
    values = {name: variable(a, unit, desc) for name, a, unit, desc in (
        ("ghi", g, "W m-2", "Global horizontal irradiance (WRF SWDOWN)"),
        ("dni", n, "W m-2", "Direct normal irradiance"),
        ("dhi", d, "W m-2", "Diffuse horizontal irradiance"),
        ("poa_irradiance", poa, "W m-2", "Isotropic plane-of-array irradiance"),
        ("pv_cell_temperature", cell, "degree_Celsius", "Estimated or supplied PV cell temperature"),
        ("pv_dc_power", dc, "W", "PVWatts DC module power"))}
    return _result(ds, values, irradiance_method=method, solar_geometry="NREL SPA via pvlib nrel_numpy",
                   thermal_method=thermal, pv_model="PVWatts DC; isotropic sky, no IAM/spectral/inverter/shading losses",
                   panel_tilt_deg=float(options.get("tilt", 30)), panel_azimuth_deg=float(options.get("azimuth", 180)),
                   dc_rating_w=float(options.get("dc_rating_w", 1000)), gamma_pdc=float(options.get("gamma_pdc", -.004)))


def icing_product(ds, options):
    slw = supercooled_liquid_water(temperature(ds), pressure(ds), field(ds, "QVAPOR", PROFILE),
                                   field(ds, "QCLOUD", PROFILE), field(ds, "QRAIN", PROFILE))
    return _result(ds, {"supercooled_liquid_water": variable(slw, "g m-3", "Cloud+rain liquid at -40<T<0 Celsius", dims=PROFILE),
                        "icing_potential": variable((slw > 0).astype(np.int8), "1", "Supercooled-liquid presence screen", dims=PROFILE,
                                                    flag_values=np.array([0, 1], dtype=np.int8), flag_meanings="absent present")},
                   limitations="Thermodynamic screen only; no droplet spectrum/accretion or aviation severity forecast")


def frost_product(ds, options):
    opts = _options(options, {"air_threshold_c", "surface_threshold_c"})
    air, skin = frost_flags(field(ds, "T2"), field(ds, "TSK"), **opts)
    flags = {"units": "1", "flag_values": np.array([0, 1], dtype=np.int8), "flag_meanings": "above_threshold at_or_below_threshold"}
    return _result(ds, {"air_frost": variable(air, long_name="2m temperature frost threshold", **flags),
                        "surface_frost": variable(skin, long_name="Skin temperature frost threshold", **flags)},
                   air_threshold_c=float(opts.get("air_threshold_c", 0)), surface_threshold_c=float(opts.get("surface_threshold_c", 0)),
                   limitations="Instantaneous threshold screen; no crop injury or within-output-period minima")


def visibility_product(ds, options):
    t, p = temperature(ds)[:, 0], pressure(ds)[:, 0]
    qv, qc, qr, qi, qs = [field(ds, name, PROFILE)[:, 0] for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW")]
    vis = visibility(t, p, qv, qc, qr, qi, qs)
    _, rh, _ = thermo(t, p, qv)
    rhmin = float(finite("fog_rh_percent", options.get("fog_rh_percent", 95), minimum=0, maximum=100))
    fog = ((vis < 1000) & (rh >= rhmin) & (qc > 0)).astype(np.int8)
    return _result(ds, {"visibility": variable(vis, "m", "Lowest-level hydrometeor visibility (UPP CALVIS R)"),
                        "lowest_level_rh": variable(rh, "%", "Lowest-level relative humidity over liquid"),
                        "fog_potential": variable(fog, "1", "Visibility<1km with cloud liquid and RH above threshold",
                                                   flag_values=np.array([0, 1], dtype=np.int8), flag_meanings="absent present")},
                   method="UPP CALVIS R hydrometeor extinction; 2-percent contrast, 24.135km ceiling",
                   negative_condensate_input_count=int(sum(np.count_nonzero(a < 0) for a in (qc, qr, qi, qs))),
                   fog_rh_percent=rhmin, limitations="Lowest resolved level, no aerosol/RH-only extinction; fog is a screening flag")


def fire_product(ds, options):
    u, v = _winds(ds)
    hdw = hot_dry_windy(*[_last(a) for a in (temperature(ds), pressure(ds), field(ds, "QVAPOR", PROFILE),
                                            u, v, height_agl(ds))], top_m=options.get("hdw_top_m", 500))
    return _result(ds, {"hdw_instantaneous": variable(hdw, "hPa m s-1", "Instantaneous sampled Hot-Dry-Windy screen")},
                   method="max wind times max dry-adiabatically surface-adjusted VPD in selected AGL layer",
                   layer_top_m=float(options.get("hdw_top_m", 500)),
                   limitations="Lowest model level substitutes surface pressure; no daily afternoon maximum or climatological percentile")


def _accumulated_rain(ds, options):
    bucket = float(finite("bucket_mm", options.get("bucket_mm", ds.attrs.get("BUCKET_MM", -1))))
    total = np.zeros_like(field(ds, "RAINNC"))
    for name in ("RAINNC", "RAINC", "RAINSH"):
        if name == "RAINSH" and name not in ds:
            continue
        a = finite(name, field(ds, name), minimum=0)
        # Pristine WRF module_diag_misc buckets RAINNC/RAINC only.
        # RAINSH has no I_RAINSH Registry field and remains unbucketed.
        if bucket > 0 and name != "RAINSH":
            counter = field(ds, "I_" + name)
            if np.any(counter < 0) or np.any(counter != np.floor(counter)):
                raise ValueError("rain bucket counters must be nonnegative integers")
            a = a + bucket * counter
        elif name != "RAINSH" and "I_" + name in ds:
            if np.any(field(ds, "I_" + name) != 0):
                raise ValueError("nonzero rain bucket counters require positive bucket_mm")
        total += a
    if np.any(np.diff(total, axis=0) < 0):
        raise ValueError("rain accumulation decreased; supply a continuous history without restart resets")
    return total


def fwi_product(ds, options):
    required = {"ffmc_previous", "dmc_previous", "dc_previous", "utc_offset_hours"}
    if not required.issubset(options):
        raise ValueError("app_fwi requires previous codes and explicit local-standard utc_offset_hours")
    stamps = _times(ds)
    offset = float(finite("utc_offset_hours", options["utc_offset_hours"], minimum=-12, maximum=14))
    local = stamps + np.timedelta64(int(round(offset * 3600)), "s")
    noon = (local - local.astype("datetime64[D]")) == np.timedelta64(12, "h")
    # Discard spin-up noons lacking their preceding noon; never guess rainfall.
    lookup = {s: i for i, s in enumerate(stamps)}
    indices = list(np.flatnonzero(noon & (stamps >= stamps[0] + np.timedelta64(24, "h"))))
    if not indices:
        raise ValueError("app_fwi needs exact local-standard noon weather and full preceding 24h rain coverage")
    if any(stamps[i] - np.timedelta64(24, "h") not in lookup for i in indices):
        raise ValueError("FWI noon is missing its exact preceding-noon rainfall sample")
    selected = stamps[indices]
    if np.any(np.diff(selected) != np.timedelta64(24, "h")):
        raise ValueError("FWI recurrence requires consecutive daily noons; missing noon values refused")
    # A missing early noon would misdate the supplied previous codes.
    first_eligible_day = (local[0] + np.timedelta64(24, "h")).astype("datetime64[D]")
    expected_first = first_eligible_day + np.timedelta64(12, "h")
    if expected_first < local[0] + np.timedelta64(24, "h"):
        expected_first += np.timedelta64(24, "h")
    if local[indices[0]] != expected_first:
        raise ValueError("first eligible local-standard noon is missing; cannot apply previous codes")
    last_noon = local[-1].astype("datetime64[D]") + np.timedelta64(12, "h")
    if last_noon > local[-1]:
        last_noon -= np.timedelta64(24, "h")
    if local[indices[-1]] != last_noon:
        raise ValueError("last eligible local-standard noon is missing")
    rain = _accumulated_rain(ds, options)
    t, p, q = field(ds, "T2"), field(ds, "PSFC"), field(ds, "Q2")
    _, rh, _ = thermo(t, p, q)
    w = np.hypot(field(ds, "U10"), field(ds, "V10")) * 3.6
    lat = field(ds, "XLAT")
    state = {k: options[k] for k in ("ffmc_previous", "dmc_previous", "dc_previous")}
    rows = []
    for i in indices:
        j = lookup[stamps[i] - np.timedelta64(24, "h")]
        month = int(local[i].astype("datetime64[M]").astype(int) % 12 + 1)
        result = daily_fwi(t[i] - 273.15, rh[i], w[i], rain[i] - rain[j], latitude=lat[i], month=month, **state)
        rows.append(result)
        state = {"ffmc_previous": result["ffmc"], "dmc_previous": result["dmc"], "dc_previous": result["dc"]}
    subset = ds.isel(Time=indices)
    values = {"fwi_" + k: variable(np.stack([r[k] for r in rows]), "1", "Classic daily Canadian " + k.upper(),
                                    cell_methods="time: point (local standard noon)") for k in rows[0]}
    # Rainfall is an interval input; retain explicit bounds on the daily outputs.
    bounds = np.stack([selected - np.timedelta64(24, "h"), selected], axis=-1)
    output = _result(subset, values, method="Classic FWI-1985 daily recurrence (not FWI2025)",
                     utc_offset_hours=offset, previous_code_date_utc=str(selected[0] - np.timedelta64(24, "h")),
                     limitations="T2/RH2/U10 are model approximations to station noon weather; no overwintering; previous codes supplied")
    output["fwi_rain_24h"] = variable(np.stack([rain[i] - rain[lookup[stamps[i] - np.timedelta64(24, "h")]] for i in indices]),
                                      "mm", "Preceding noon-to-noon accumulated rain", cell_methods="time: sum")
    output["time_bounds"] = xr.DataArray(bounds, dims=("Time", "bounds"))
    output["fwi_rain_24h"].attrs["comment"] = "Interval specified by time_bounds"
    return output
