"""Wind, precipitation, cloud and snow products with explicit sampling."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .common import (PROFILE, SURFACE, earth_wind, field, interp_height,
                     positive, pressure, profile_wind, speed_direction, variable)

FILL = -9999.


def _wind_speed_10m(u, v):
    """Canonical rotation-invariant speed and metadata shared by both products."""
    return variable(np.hypot(u, v), "m s-1", "10 m wind speed",
                    standard_name="wind_speed", height_agl_m=10.)


def speed10(ds, options):
    return xr.Dataset({"wind_speed_10m": _wind_speed_10m(field(ds, "U10"), field(ds, "V10"))})


def wind10(ds, options):
    raw_u, raw_v = field(ds, "U10"), field(ds, "V10")
    speed = _wind_speed_10m(raw_u, raw_v)
    u, v = earth_wind(ds, raw_u, raw_v, options)
    direction = np.mod(np.degrees(np.arctan2(-u, -v)), 360.)
    direction = np.where(speed.values == 0, 0., direction)
    return xr.Dataset({
        "wind_speed_10m": speed,
        "wind_from_direction_10m": variable(direction, "degree", "10 m meteorological wind from direction",
                                            standard_name="wind_from_direction", height_agl_m=10., calm_convention="0 degrees"),
    })


def wind_height(ds, options):
    heights = np.asarray(options.get("heights", [100., 200.]), dtype=float)
    if heights.ndim != 1 or len(heights) == 0 or np.any(np.diff(heights) <= 0):
        raise ValueError("wind-height needs one or more strictly increasing heights")
    z, u, v = profile_wind(ds, options)
    uh, vh = interp_height(z, u, heights), interp_height(z, v, heights)
    speed, direction = speed_direction(uh, vh)
    dims = ("Time", "height", "south_north", "west_east")
    result = xr.Dataset({
        "eastward_wind": variable(uh, "m s-1", "earth-relative eastward wind", dims, standard_name="eastward_wind"),
        "northward_wind": variable(vh, "m s-1", "earth-relative northward wind", dims, standard_name="northward_wind"),
        "wind_speed": variable(speed, "m s-1", "wind speed at requested AGL height", dims, standard_name="wind_speed"),
        "wind_from_direction": variable(direction, "degree", "meteorological wind from direction", dims,
                                         standard_name="wind_from_direction", calm_convention="0 degrees"),
    }, coords={"height": ("height", heights, {"units": "m", "standard_name": "height", "positive": "up", "long_name": "height above ground"})})
    if len(heights) == 1:
        return result
    result["bulk_wind_shear"] = variable(np.hypot(uh[:, -1]-uh[:, 0], vh[:, -1]-vh[:, 0]), "m s-1",
        "vector wind difference between highest and lowest requested heights",
        lower_height_agl_m=float(heights[0]), upper_height_agl_m=float(heights[-1]))
    veer = (direction[:, -1]-direction[:, 0]+180.) % 360. - 180.
    valid = (speed[:, -1] > 0) & (speed[:, 0] > 0)
    result["wind_veer"] = variable(np.where(valid, veer, FILL), "degree", "signed clockwise wind direction change, upper minus lower",
                                   lower_height_agl_m=float(heights[0]), upper_height_agl_m=float(heights[-1]))
    result["wind_veer"].encoding["_FillValue"] = FILL
    result["wind_veer_valid"] = variable(valid.astype("i1"), "1", "wind veer is defined (both endpoint winds non-calm)")
    return result


def max_random_overlap(cf):
    """Maximum in contiguous cloudy blocks, random between clear-separated blocks."""
    clear = np.ones(cf.shape[:1] + cf.shape[2:])
    block = np.zeros_like(clear)
    for k in range(cf.shape[1]):
        cloudy = cf[:, k] > 0.
        clear *= np.where(cloudy, 1., 1.-block)
        block = np.where(cloudy, np.maximum(block, cf[:, k]), 0.)
    return 1.-clear*(1.-block)


def cloud(ds, options):
    cf, p = field(ds, "CLDFRA", PROFILE), pressure(ds)
    if np.any((cf < 0) | (cf > 1)):
        raise ValueError("CLDFRA must lie in [0,1]")
    result = xr.Dataset()
    for name, low, high in (("low", 68000., np.inf), ("mid", 44000., 68000.), ("high", 0., 44000.)):
        mask = (p >= low) & (p < high)
        # Empty layers contain no modelled cloud; coverage is supplied separately.
        result[f"cloud_cover_{name}"] = variable(max_random_overlap(np.where(mask, cf, 0.)), "1", f"{name} cloud area fraction",
            pressure_lower_bound_pa=low, pressure_upper_bound_pa=high if np.isfinite(high) else "surface",
            overlap="maximum within contiguous cloudy levels, random across clear gaps",
            comment="Computed from model CLDFRA; empty pressure layer = 0 with sampled flag 0")
        result[f"cloud_layer_{name}_sampled"] = variable(mask.any(axis=1).astype("i1"), "1", f"{name} pressure layer has model mass levels")
    result["cloud_cover_total"] = variable(max_random_overlap(cf), "1", "total column cloud area fraction",
        standard_name="cloud_area_fraction", overlap="maximum within contiguous cloudy levels, random across clear gaps")
    return result


def accumulator(ds, name, options):
    a = positive(field(ds, name), name, zero=True)
    if name in ("RAINNC", "RAINC"):
        bucket = options.get("bucket_mm", ds.attrs.get("BUCKET_MM", -1.))
        bucket = float(bucket)
        if not np.isfinite(bucket) or bucket == 0:
            raise ValueError("bucket_mm must be finite and positive, or negative for disabled")
        count_name = f"I_{name}"
        if bucket > 0:
            count = positive(field(ds, count_name), count_name, zero=True)
            if np.any(count != np.floor(count)):
                raise ValueError(f"{count_name}: bucket counters must be integers")
            a = a + bucket*count
        elif count_name in ds and np.any(field(ds, count_name) != 0):
            raise ValueError("nonzero rain bucket counters require positive bucket_mm")
    if np.any(np.diff(a, axis=0) < 0):
        raise ValueError(f"{name}: accumulation decreased; reset/restart or missing bucket metadata")
    return a


def intervals(ds, total, name):
    if not np.issubdtype(ds.Time.dtype, np.datetime64) or np.isnat(ds.Time.values).any():
        raise ValueError("interval products require datetime64 Time")
    dt = np.diff(ds.Time.values) / np.timedelta64(1, "s")
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("interval products require at least two strictly increasing timestamps")
    delta = np.diff(total, axis=0)
    if np.any(delta < 0):
        raise ValueError(f"{name}: accumulation decreased; reset/restart or missing bucket metadata")
    amount = np.concatenate([np.full_like(total[:1], FILL), delta], axis=0)
    return amount, np.r_[0., dt]


def interval_metadata(ds):
    times = ds.Time.values
    starts = np.r_[times[:1], times[:-1]]
    return xr.Dataset({"time_bounds": (("Time", "bounds"), np.stack([starts, times], axis=1)),
                       "interval_valid": variable(np.r_[0, np.ones(len(times)-1)].astype("i1"), "1", "previous history interval exists", ("Time",))})


def precip(ds, options):
    total = accumulator(ds, "RAINNC", options) + accumulator(ds, "RAINC", options)
    if "RAINSH" in ds:
        total += accumulator(ds, "RAINSH", options)
    amount, dt = intervals(ds, total, "total precipitation")
    result = interval_metadata(ds)
    result["precipitation_accumulated"] = variable(total, "mm", "cumulative liquid-equivalent precipitation since model accumulation origin",
        comment="RAINNC + RAINC + optional RAINSH; rain bucket counters restored; origin is model-defined")
    result["precipitation_interval"] = variable(amount, "mm", "previous history interval liquid-equivalent precipitation", cell_methods="time: sum")
    result["precipitation_rate"] = variable(np.concatenate([np.full_like(total[:1], FILL), amount[1:]/dt[1:, None, None]*3600.]),
        "mm h-1", "mean precipitation rate over previous history interval", cell_methods="time: mean")
    for key in ("precipitation_interval", "precipitation_rate"):
        result[key].encoding["_FillValue"] = FILL
    for hours in options.get("rolling_hours", [1., 3., 6.]):
        hours = float(hours)
        if not np.isfinite(hours) or hours <= 0:
            raise ValueError("rolling hours must be positive and finite")
        duration = np.timedelta64(round(hours*3600.*1e9), "ns")
        if duration <= np.timedelta64(0, "ns"):
            raise ValueError("rolling window is below timestamp precision")
        times = ds.Time.values
        values = np.full_like(total, FILL)
        valid = np.zeros(len(times), dtype="i1")
        for i, end in enumerate(times):
            start = end-duration
            j = np.searchsorted(times, start)
            if j < len(times) and times[j] == start:
                values[i] = total[i]-total[j]
                valid[i] = 1
        suffix = format(hours, "g").replace(".", "p") + "h"
        key = f"precipitation_rolling_{suffix}"
        result[key] = variable(values, "mm", f"trailing {hours:g} hour liquid-equivalent precipitation",
            cell_methods=f"time: sum (interval: {hours:g} hours)", window_seconds=hours*3600.,
            comment="requires an exact window-start history sample; no temporal interpolation")
        result[key].encoding["_FillValue"] = FILL
        result[f"{key}_valid"] = xr.DataArray(valid, dims="Time", attrs={"units": "1", "long_name": "exact trailing window is available"})
    return result


def snow(ds, options):
    result = xr.Dataset({
        "snow_depth": variable(positive(field(ds, "SNOWH"), "SNOWH", zero=True), "m", "surface snow depth", standard_name="surface_snow_thickness"),
        "snow_water_equivalent": variable(positive(field(ds, "SNOW"), "SNOW", zero=True), "kg m-2", "surface snow water equivalent", standard_name="surface_snow_amount"),
    })
    amount, _ = intervals(ds, accumulator(ds, "SNOWNC", options), "SNOWNC")
    result["snowfall_interval_water_equivalent"] = variable(amount, "mm", "previous history interval grid-scale snow and ice water equivalent",
        cell_methods="time: sum", comment="SNOWNC difference; excludes graupel, hail, and convective solid precipitation; not snow depth")
    result["snowfall_interval_water_equivalent"].encoding["_FillValue"] = FILL
    return xr.merge([result, interval_metadata(ds)])


def precip_type(ds, options):
    # An interval composition diagnostic, NOT a thermodynamic rain/freezing-rain classifier.
    if int(ds.attrs.get("MP_PHYSICS", -1)) != 8:
        raise ValueError("precip-type partition is qualified only for Thompson MP_PHYSICS=8")
    total = accumulator(ds, "RAINNC", options)
    amount, _ = intervals(ds, total, "RAINNC")
    solid = np.zeros_like(amount[1:])
    for name in ("SNOWNC", "GRAUPELNC", "HAILNC"):
        if name == "HAILNC" and name not in ds:
            continue  # Thompson mp=8 has no separate hail category.
        increment, _ = intervals(ds, accumulator(ds, name, options), name)
        solid += increment[1:]
    if np.any(solid < 0) or np.any(solid > amount[1:] + 1e-5):
        raise ValueError("solid precipitation is inconsistent with RAINNC; scheme partition unsupported")
    fraction = np.divide(solid, amount[1:], out=np.zeros_like(solid), where=amount[1:] > 0)
    kind = np.where(amount[1:] == 0, 0, np.where(fraction == 0, 1, np.where(fraction >= 1.-1e-5, 2, 3)))
    result = interval_metadata(ds)
    result["grid_scale_precipitation_phase"] = variable(np.concatenate([np.full_like(kind[:1], -1), kind]), "1",
        "previous history interval grid-scale liquid/solid composition", cell_methods="time: point",
        flag_values=np.array([0, 1, 2, 3], dtype="i4"), flag_meanings="none liquid solid mixed",
        comment="Thompson mp=8 only; excludes convective precipitation; no freezing-rain diagnosis")
    result["grid_scale_precipitation_phase"].encoding["_FillValue"] = -1
    result["grid_scale_precipitation_phase"].attrs.pop("cell_methods")
    result["grid_scale_precipitation_phase"].attrs["sampling"] = "previous history interval composition"
    return result
