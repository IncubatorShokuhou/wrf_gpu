"""Optional Meteostat 2 adapters; only explicit observation providers allowed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .observations import finite, utc, validate_rows


def meteostat_rows(data, station_uid):
    """Metric Meteostat output: Celsius, km/h, degrees, hourly mm.

    Missing provider measurements are unavailable and counted, never filled.
    This is source-ingest handling, not permission to drop non-finite forecasts.
    """
    times = utc(data.index, naive_utc=True)
    rows, missing = [], 0
    mapping = {"temp": ("t2", 1), "wspd": ("wind", 1/3.6), "wdir": ("wind_direction", 1),
               "wpgt": ("gust", 1/3.6), "prcp": ("precip_rate", 1)}
    for column, (variable, factor) in mapping.items():
        if column not in data:
            continue
        for time, raw in zip(times, data[column]):
            if pd.isna(raw):
                missing += 1
                continue
            value = float(finite([raw], f"Meteostat {column}")[0])*factor
            rows.append({"station_uid": station_uid, "valid_time": time, "variable": variable,
                         "value": value, "interval_start": time-pd.Timedelta(hours=1) if variable in ("gust", "precip_rate") else time})
    if not rows:
        return pd.DataFrame(), missing
    return validate_rows(pd.DataFrame(rows)), missing


def fetch_meteostat(stations, start, end, *, provider="isd-lite"):
    """Fetch station (not interpolated Point) observations from METAR or NOAA ISD.

    The Meteostat blended HOURLY dump can contain model substitutes, so it is
    deliberately not supported as observation truth. No forecast provider used.
    """
    if provider not in ("metar", "isd-lite"):
        raise ValueError("provider must be metar or isd-lite")
    if "meteostat_id" not in stations or stations.meteostat_id.isna().any():
        raise ValueError("provider fetch requires meteostat_id for every station")
    start, end = utc([start, end])
    if end <= start:
        raise ValueError("provider end must follow start")
    try:
        import meteostat as ms
    except ImportError as exc:
        raise ValueError('install the optional extra: pip install "gpuwrf[verify]" (Meteostat 2)') from exc
    if not hasattr(ms, "hourly"):
        raise ValueError("Meteostat 2 required; legacy Hourly API is unsupported")
    chosen = ms.Provider.METAR if provider == "metar" else ms.Provider.ISD_LITE
    parts, missing, unavailable_stations = [], 0, 0
    for station in stations.itertuples():
        # A bare Station(id=...) bypasses provider identifier/metadata lookup.
        # Use the string ID so METAR/ISD can resolve their native identifiers.
        ts = ms.hourly(str(station.meteostat_id), start.to_pydatetime(), end.to_pydatetime(),
                       timezone="UTC", providers=[chosen])
        native = ts.fetch(fill=False, clean=True, units=ms.UnitSystem.METRIC)
        if native is None or native.empty:
            unavailable_stations += 1
            continue
        rows, absent = meteostat_rows(native, station.station_uid)
        missing += absent
        if not rows.empty:
            parts.append(rows)
    if not parts:
        raise ValueError("provider returned no observation measurements")
    return validate_rows(pd.concat(parts, ignore_index=True)), {
        "provider": f"meteostat:{provider}", "start": start.isoformat(), "end": end.isoformat(),
        "missing_measurements": missing, "stations_without_data": unavailable_stations,
        "fill": False, "clean": True, "model_substitutes": False}
