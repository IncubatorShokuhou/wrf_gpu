"""Strict station/observation schemas and read-only ALISIOS export adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ("t2", "wind", "wind_direction", "gust", "precip_rate")
NATIVE = {"t2_c": "t2", "wind_speed_ms": "wind", "wind_dir_deg": "wind_direction",
          "wind_gust_ms": "gust", "precip_rate_mmh": "precip_rate", "precip_mm": "precip_rate"}
KEYS = ["station_uid", "valid_time", "variable", "interval_start"]


def finite(values, name):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: NaN/Inf/missing numeric values are not scoreable")
    return array


def utc(values, name="valid_time", *, naive_utc=False):
    """Require explicit timezone except for schema-declared UTC native exports."""
    for value in values:
        time = pd.Timestamp(value)
        if pd.isna(time) or (time.tzinfo is None and not naive_utc):
            raise ValueError(f"{name}: timestamps must declare UTC/offset")
    return pd.to_datetime(values, utc=True)


def read_stations(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"station_uid": str, "station_id": str, "meteostat_id": str})
    if "station_uid" not in data and "station_id" in data:
        data = data.rename(columns={"station_id": "station_uid"})
    if "elevation_m" not in data and "altitude_m" in data:
        data["elevation_m"] = data.get("altitude_verified_m", data.altitude_m).fillna(data.altitude_m)
    required = ["station_uid", "lat", "lon", "elevation_m"]
    if not set(required) <= set(data):
        raise ValueError(f"stations require {required}")
    if data.empty or data.station_uid.isna().any() or data.station_uid.str.strip().eq("").any():
        raise ValueError("station_uid must be nonempty")
    if data.station_uid.duplicated().any():
        raise ValueError("duplicate station_uid")
    for column in required[1:]:
        data[column] = finite(data[column], column)
    if not data.lat.between(-90, 90).all() or not data.lon.between(-180, 180).all():
        raise ValueError("station latitude/longitude out of range")
    # Defaults are assumptions and appear in the report; supply measured metadata when known.
    for column, default in (("landmask", 1), ("wind_height_m", 10), ("roughness_m", 0.03)):
        if column not in data:
            data[column] = default
        data[column] = finite(data[column], column)
    if not data.landmask.isin([0, 1]).all():
        raise ValueError("station landmask must be 0 (water) or 1 (land)")
    if (data.roughness_m <= 0).any() or (data.wind_height_m <= data.roughness_m).any() or (data.roughness_m >= 10).any():
        raise ValueError("wind heights require 0 < roughness_m < min(10, wind_height_m)")
    return data


def validate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Canonical long format: °C, m/s, meteorological degrees, mm/h."""
    data = data.copy()
    required = {"station_uid", "valid_time", "variable", "value"}
    if not required <= set(data):
        raise ValueError(f"rows require {sorted(required)}")
    if data.empty:
        raise ValueError("no observation/forecast rows")
    if data.station_uid.isna().any() or data.station_uid.astype(str).str.strip().eq("").any():
        raise ValueError("station_uid must be nonempty")
    data["station_uid"] = data.station_uid.astype(str)
    data["valid_time"] = utc(data.valid_time)
    data["value"] = finite(data.value, "value")
    if not data.variable.isin(VARIABLES).all():
        raise ValueError(f"supported variables: {VARIABLES}")
    rain = data.variable.eq("precip_rate")
    if "interval_start" not in data:
        data["interval_start"] = data.valid_time
    if data.loc[~rain, "interval_start"].isna().any():
        data.loc[~rain & data.interval_start.isna(), "interval_start"] = data.loc[~rain & data.interval_start.isna(), "valid_time"]
    data["interval_start"] = utc(data.interval_start, "interval_start")
    instant = ~data.variable.isin(["gust", "precip_rate"])
    if (data.loc[instant, "interval_start"] != data.loc[instant, "valid_time"]).any():
        raise ValueError("instantaneous variables must have interval_start=valid_time")
    gust = data.variable.eq("gust")
    if (data.loc[gust, "interval_start"] > data.loc[gust, "valid_time"]).any():
        raise ValueError("gust interval_start cannot follow valid_time")
    if (data.loc[rain, "interval_start"] >= data.loc[rain, "valid_time"]).any():
        raise ValueError("precip_rate needs a positive, explicit interval_start")
    if data.duplicated(KEYS).any():
        raise ValueError("duplicate station/time/variable/interval rows (select one issue/source)")
    # Distinct overlapping precipitation intervals would double-count water.
    for _, part in data[rain].groupby("station_uid"):
        part = part.sort_values("interval_start")
        if (part.interval_start.iloc[1:].reset_index(drop=True) < part.valid_time.iloc[:-1].reset_index(drop=True)).any():
            raise ValueError("overlapping precipitation intervals")
    positive = data.variable.isin(["wind", "gust", "precip_rate"])
    if (data.loc[positive, "value"] < 0).any():
        raise ValueError("negative speed/gust/precipitation")
    directions = data.loc[data.variable.eq("wind_direction"), "value"]
    if not directions.between(0, 360).all():
        raise ValueError("wind direction outside [0,360]")
    data.loc[data.variable.eq("wind_direction"), "value"] %= 360
    if ("u_ms" in data) != ("v_ms" in data):
        raise ValueError("wind components require both u_ms and v_ms, or neither")
    for component in ("u_ms", "v_ms"):
        if component in data:
            finite(data.loc[data.variable.eq("wind"), component], component)
    return data.sort_values(KEYS).reset_index(drop=True)


def read_observations(path: str | Path, format="csv", *, precip_hours=None) -> tuple[pd.DataFrame, dict]:
    """Read canonical CSV or an ObsDB joined export; never open the live DB."""
    raw = pd.read_csv(path, dtype={"station_uid": str, "station_id": str})
    audit = {"format": format, "input_rows": len(raw), "qc_excluded": 0}
    if "qc_flag" in raw:
        flags = finite(raw.qc_flag, "qc_flag")
        audit["qc_excluded"] = int(np.count_nonzero(flags != 0))
        raw = raw.loc[flags == 0].copy()
    if format == "csv":
        return validate_rows(raw), audit
    if format != "alisios":
        raise ValueError("observation format must be csv or alisios")
    if "qc_flag" not in raw:
        raise ValueError("ALISIOS ObsDB export requires qc_flag")
    raw = raw.rename(columns={"station_id": "station_uid", "ts_utc": "valid_time"})
    if raw.columns.duplicated().any():
        raise ValueError("ALISIOS export must contain station_id OR station_uid")
    required = {"station_uid", "valid_time", "variable", "value"}
    if not required <= set(raw):
        raise ValueError(f"ALISIOS ObsDB export requires {sorted(required)} (ts_utc accepted)")
    supported = raw.variable.isin(NATIVE)
    audit["unsupported_excluded"] = int((~supported).sum())
    raw = raw[supported].copy()
    raw["value"] = finite(raw.value, "ObsDB value")
    # Validate each admitted measurement before means/maxima/circular means
    # can hide a malformed native truth value. Explicit QC exclusions stay first.
    positive = raw.variable.isin(["wind_speed_ms", "wind_gust_ms", "precip_rate_mmh", "precip_mm"])
    if (raw.loc[positive, "value"] < 0).any():
        raise ValueError("negative native speed/gust/precipitation")
    directions = raw.loc[raw.variable.eq("wind_dir_deg"), "value"]
    if not directions.between(0, 360).all():
        raise ValueError("native wind direction outside [0,360]")
    raw["valid_time"] = utc(raw.valid_time, naive_utc=True)
    # Exact native duplicates are ambiguous upstream ingest batches, not extra evidence.
    if raw.duplicated(["station_uid", "valid_time", "variable"]).any():
        raise ValueError("duplicate native ObsDB observations")
    rows = []
    for (station, native), part in raw.groupby(["station_uid", "variable"]):
        variable = NATIVE[native]
        part = part.copy()
        if variable == "precip_rate":
            if precip_hours is None or not np.isfinite(precip_hours) or precip_hours <= 0:
                raise ValueError("ALISIOS precipitation requires declared precip_hours (no assumed accumulation)")
            # Native rate/accumulation represents the interval ending at ts_utc.
            for record in part.itertuples():
                rows.append({"station_uid": station, "valid_time": record.valid_time,
                             "variable": variable,
                             "value": record.value / precip_hours if native == "precip_mm" else record.value,
                             "interval_start": record.valid_time - pd.Timedelta(hours=precip_hours)})
            continue
        part["hour"] = part.valid_time.dt.floor("h")
        for hour, group in part.groupby("hour"):
            values = group.value.to_numpy()
            if variable == "wind_direction":
                sine, cosine = np.sin(np.deg2rad(values)).mean(), np.cos(np.deg2rad(values)).mean()
                if np.hypot(sine, cosine) < 1e-12:
                    # Opposing directions have no defined circular mean.
                    audit["undefined_direction_excluded"] = audit.get("undefined_direction_excluded", 0) + 1
                    continue
                value = np.rad2deg(np.arctan2(sine, cosine)) % 360
            else:
                value = values.max() if variable == "gust" else values.mean()
            # ALISIOS native scalar aggregation labels the start of each hour.
            # Canonical interval gusts use the window END to match history maxima.
            valid = hour+pd.Timedelta(hours=1) if variable == "gust" else hour
            rows.append({"station_uid": station, "valid_time": valid,
                         "variable": variable, "value": value, "interval_start": hour})
    return validate_rows(pd.DataFrame(rows)), audit


def read_forecast_csv(path):
    """Station predictions from a driving model or an ALISIOS issued export."""
    data = pd.read_csv(path, dtype={"station_uid": str})
    data = data.rename(columns={"prediction_value": "value", "prediction_u_ms": "u_ms", "prediction_v_ms": "v_ms"})
    for source in ("source", "source_id", "rung_id"):
        if source in data and (data[source].isna().any() or data[source].nunique() != 1):
            raise ValueError("select one forecast source before exporting")
    if "issue_time" in data:
        if data.issue_time.isna().any() or data.issue_time.astype(str).str.strip().eq("").any():
            raise ValueError("declared issue_time must be non-missing and nonempty on every row")
        if data.issue_time.nunique() != 1:
            raise ValueError("select one issue before exporting")
    if "prediction_dir_deg" in data:
        wind = data[data.variable.eq("wind")].copy()
        finite(wind.prediction_dir_deg, "prediction_dir_deg")
        if not data.variable.eq("wind_direction").any():
            wind["variable"], wind["value"] = "wind_direction", wind.prediction_dir_deg
            data = pd.concat([data, wind], ignore_index=True)
    return validate_rows(data)
