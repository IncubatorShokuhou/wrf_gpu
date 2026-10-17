"""Stream surface fields from one-domain wrfout files; retain only station rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .observations import finite, validate_rows
from .sampling import StationSampler, earth_wind


def _times(ds):
    if "Times" not in ds.variables:
        raise ValueError("wrfout requires Times")
    times = np.asarray(ds.variables["Times"][:])
    return [pd.Timestamp(b"".join(row).decode().strip().replace("_", "T"), tz="UTC") for row in times]


def _field(ds, name, index, units=None):
    variable = ds.variables[name]
    value = variable[index] if variable.dimensions[0] == "Time" else variable[:]
    if np.ma.isMaskedArray(value) and np.any(value.mask):
        raise ValueError(f"{name}: masked/fill values")
    value = finite(value, name)
    if value.ndim != 2:
        raise ValueError(f"{name}: expected 2D surface mass-grid field")
    if units is not None and str(getattr(variable, "units", "")).strip() not in units:
        raise ValueError(f"{name}: unsupported/missing units {getattr(variable, 'units', None)!r}")
    return value


def extract_wrfout(paths, stations, *, method="bilinear", land_aware=True,
                   lapse_rate=0.0, wind_height="as-is", wind_frame="grid", gust_variable=None, gust_hours=None):
    from gpuwrf.io.netcdf_lock import Dataset

    if not np.isfinite(lapse_rate) or lapse_rate < 0:
        raise ValueError("lapse_rate must be finite and >=0 K/m")
    if wind_height not in ("as-is", "neutral-log"):
        raise ValueError("wind_height must be as-is or neutral-log")
    if gust_variable and (gust_hours is None or not np.isfinite(gust_hours) or gust_hours < 0):
        raise ValueError("gust variable requires explicit gust_hours >=0 (0 means instantaneous)")
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("no wrfout inputs")
    inventory, domains, issues = [], set(), set()
    for path in paths:
        with Dataset(path) as ds:
            inventory.extend((time, path, i) for i, time in enumerate(_times(ds)))
            domain = getattr(ds, "GRID_ID", None)
            issue = getattr(ds, "SIMULATION_START_DATE", getattr(ds, "START_DATE", None))
            if domain is None or issue is None:
                raise ValueError("wrfout needs GRID_ID and SIMULATION_START_DATE (or START_DATE)")
            domains.add(domain)
            issues.add(issue)
    if len(domains) != 1 or len(issues) != 1:
        raise ValueError("select a single domain and forecast issue per source")
    inventory.sort()
    if len({time for time, _, _ in inventory}) != len(inventory):
        raise ValueError("duplicate wrfout valid times")
    sampler = None
    previous = None
    rows = []
    mismatch = stations.wind_height_m.ne(10).sum()
    audit = {"files": [str(p) for p in paths], "frames": len(inventory), "method": method,
             "land_aware": land_aware, "lapse_rate_K_per_m": lapse_rate,
             "wind_height": wind_height, "non_10m_stations": int(mismatch), "wind_frame": wind_frame,
             "gust_variable": gust_variable, "gust_hours": gust_hours, "rain_resets": "refused", "sampling": [],
             "undefined_vector_direction_excluded": 0}
    factor = (np.log(stations.wind_height_m / stations.roughness_m) / np.log(10 / stations.roughness_m)).to_numpy() if wind_height == "neutral-log" else np.ones(len(stations))
    for time, path, index in inventory:
        with Dataset(path) as ds:
            lat, lon = _field(ds, "XLAT", index), _field(ds, "XLONG", index)
            land = _field(ds, "LANDMASK", index) if land_aware else None
            if sampler is None:
                sampler = StationSampler(lat, lon, stations, landmask=land, method=method, land_aware=land_aware)
                static_land = land
                audit["sampling"] = [{"station_uid": uid, "flat_nodes": nodes.tolist(), "weights": weights.tolist()}
                                     for uid, nodes, weights in zip(stations.station_uid, sampler.nodes, sampler.weights)]
            elif not np.array_equal(lat, sampler.lat) or not np.array_equal(lon, sampler.lon) or (land_aware and not np.array_equal(land, static_land)):
                raise ValueError("moving grid/landmask is unsupported; station stencil cannot be reused")
            fields = {}
            if "T2" in ds.variables:
                fields["t2"] = sampler.sample(_field(ds, "T2", index, {"K"})) - 273.15
                if lapse_rate:
                    terrain = sampler.sample(_field(ds, "HGT", index, {"m", "meters"}))
                    fields["t2"] -= lapse_rate * (stations.elevation_m.to_numpy() - terrain)
            components = None
            if ("U10" in ds.variables) != ("V10" in ds.variables):
                raise ValueError("U10 and V10 must be supplied together")
            if "U10" in ds.variables:
                u, v = (_field(ds, name, index, {"m s-1", "m s^-1", "m/s"}) for name in ("U10", "V10"))
                # Preserve the scalar speed arm before applying rounded WRF
                # rotation fields, whose norm need not be exactly one.
                speed = np.hypot(u, v)
                attrs = {key: ds.getncattr(key) for key in ds.ncattrs()}
                c = _field(ds, "COSALPHA", index) if "COSALPHA" in ds.variables else None
                s = _field(ds, "SINALPHA", index) if "SINALPHA" in ds.variables else None
                u, v = earth_wind(u, v, lat, lon, attrs, cosalpha=c, sinalpha=s, frame=wind_frame)
                # ALISIOS canonical speed arm: interpolate speed, not magnitude of interpolated vector.
                fields["wind"] = sampler.sample(speed)*factor
                us, vs = sampler.sample(u)*factor, sampler.sample(v)*factor
                components = us, vs
                fields["wind_direction"] = np.rad2deg(np.arctan2(-us, -vs)) % 360
                audit["undefined_vector_direction_excluded"] += int(np.count_nonzero(np.hypot(us, vs) < 1e-12))
            if gust_variable:
                if gust_variable not in ds.variables:
                    raise ValueError(f"requested gust variable {gust_variable} missing")
                fields["gust"] = sampler.sample(_field(ds, gust_variable, index, {"m s-1", "m s^-1", "m/s"}))*factor
            # Require complete total accumulation. Buckets cannot be reconstructed without bucket_mm.
            rain = None
            if ("RAINC" in ds.variables) != ("RAINNC" in ds.variables):
                raise ValueError("rain requires both RAINC and RAINNC")
            if "RAINC" in ds.variables:
                for counter in ("I_RAINC", "I_RAINNC"):
                    if counter in ds.variables and np.any(_field(ds, counter, index) != 0):
                        raise ValueError("bucketed precipitation unsupported; reconstruct totals before verification")
                rain = sum((_field(ds, name, index, {"mm"}) for name in ("RAINC", "RAINNC")))
                if "RAINSH" in ds.variables:
                    rain += _field(ds, "RAINSH", index, {"mm"})
        for variable, values in fields.items():
            finite(values, variable)
            for n, uid in enumerate(stations.station_uid):
                if variable == "wind_direction" and np.hypot(components[0][n], components[1][n]) < 1e-12:
                    continue
                record = {"station_uid": uid, "valid_time": time, "variable": variable, "value": values[n], "interval_start": time}
                if variable == "gust":
                    record["interval_start"] = time-pd.Timedelta(hours=gust_hours)
                if variable == "wind":
                    record.update(u_ms=components[0][n], v_ms=components[1][n])
                rows.append(record)
        if rain is not None:
            if previous is not None:
                prevtime, prev_rain = previous
                delta = rain - prev_rain
                if (delta < 0).any():
                    raise ValueError("precipitation accumulation reset/decrease; no clipping permitted")
                hours = (time-prevtime).total_seconds()/3600
                sampled_delta = sampler.sample(delta)
                rows.extend({"station_uid": uid, "valid_time": time, "variable": "precip_rate", "value": sampled_delta[n]/hours,
                             "interval_start": prevtime} for n, uid in enumerate(stations.station_uid))
            previous = time, rain
        elif previous is not None:
            raise ValueError("precipitation fields disappear between frames")
    return validate_rows(pd.DataFrame(rows)), audit
