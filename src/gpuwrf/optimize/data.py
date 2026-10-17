"""Reuse verify schemas/sampling, with an isolated instantaneous solar extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpuwrf.verify.observations import KEYS, finite, validate_rows, utc

UNITS = {"t2": "degC", "wind": "m/s", "precip_rate": "mm/h", "solar": "W/m2"}
OBJECTIVES = {"T2": ("t2",), "WS10": ("wind",), "precip": ("precip_rate",),
              "solar": ("solar",), "all": tuple(UNITS)}


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    """Exclusive creation: a completed study/model is never silently replaced."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def validate(data):
    if not {"station_uid", "valid_time", "variable", "value"} <= set(data):
        raise ValueError("station rows require station_uid,valid_time,variable,value")
    solar = data.loc[data.variable.eq("solar")].copy()
    other = data.loc[~data.variable.eq("solar")].copy()
    parts = [validate_rows(other)] if len(other) else []
    if len(solar):
        if solar.station_uid.isna().any() or solar.station_uid.astype(str).str.strip().eq("").any():
            raise ValueError("station_uid must be nonempty")
        solar["station_uid"] = solar.station_uid.astype(str)
        solar["valid_time"] = utc(solar.valid_time)
        solar["value"] = finite(solar.value, "solar")
        if (solar.value < 0).any():
            raise ValueError("solar must be nonnegative W/m2")
        if "interval_start" not in solar:
            solar["interval_start"] = solar.valid_time
        solar["interval_start"] = utc(solar.interval_start.fillna(solar.valid_time))
        if (solar.interval_start != solar.valid_time).any():
            raise ValueError("solar uses instantaneous SWDOWN, interval averages are unsupported")
        parts.append(solar)
    if not parts:
        raise ValueError("no station rows")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(KEYS).any():
        raise ValueError("duplicate station/time/variable/interval rows")
    return result.sort_values(KEYS).reset_index(drop=True)


def read_csv(path, *, observations=False):
    raw = pd.read_csv(path, dtype={"station_uid": str})
    if observations and "qc_flag" in raw:
        raw = raw.loc[finite(raw.qc_flag, "qc_flag") == 0]
    # Canonical verify CSV only: no live observations, implicit units or timezones.
    return validate(raw)


def forecast(paths, stations, **sampling):
    from gpuwrf.verify.observations import read_forecast_csv
    from gpuwrf.verify.wrfout import extract_wrfout, _times, _field
    from gpuwrf.verify.sampling import StationSampler
    from gpuwrf.io.netcdf_lock import Dataset

    paths = [Path(p) for p in paths]
    if len(paths) == 1 and paths[0].suffix.lower() == ".csv":
        raw = pd.read_csv(paths[0], dtype={"station_uid": str})
        if "variable" not in raw:
            raise ValueError("station forecast CSV requires variable")
        if raw.variable.eq("solar").any():
            # Solar extension still enforces the verifier's one-source/one-issue rule.
            for key in ("issue_time", "source", "source_id", "rung_id"):
                if key in raw and (raw[key].isna().any() or raw[key].nunique() != 1):
                    raise ValueError("select one forecast issue/source")
            return validate(raw), {"format": "canonical station CSV including solar"}
        return read_forecast_csv(paths[0]), {"format": "verify station CSV"}
    result, audit = extract_wrfout(paths, stations, **sampling)
    solar = []
    reference = None
    for path in paths:
        with Dataset(path) as ds:
            if "SWDOWN" not in ds.variables:
                continue
            for index, time in enumerate(_times(ds)):
                lat, lon = _field(ds, "XLAT", index), _field(ds, "XLONG", index)
                land = _field(ds, "LANDMASK", index) if sampling.get("land_aware", True) else None
                if reference is None:
                    reference = (lat, lon, land)
                elif any(not np.array_equal(a, b) for a, b in zip(reference, (lat, lon, land))):
                    raise ValueError("moving solar grid/landmask is unsupported")
                sampler = StationSampler(lat, lon, stations, landmask=land,
                    method=sampling.get("method", "bilinear"), land_aware=sampling.get("land_aware", True))
                values = sampler.sample(_field(ds, "SWDOWN", index, {"W m-2", "W m^-2", "W/m2", "W/m^2"}))
                solar.extend({"station_uid": uid, "valid_time": time, "interval_start": time,
                              "variable": "solar", "value": value} for uid, value in zip(stations.station_uid, values))
    return validate(pd.concat([result, pd.DataFrame(solar)], ignore_index=True)), audit


def paired(observations, predictions, stations, variables):
    """Exact verify support; rain observations integrate onto forecast intervals."""
    from gpuwrf.verify.scores import aggregate_intervals

    observations, predictions = validate(observations), validate(predictions)
    known = set(stations.station_uid)
    if (set(observations.station_uid) | set(predictions.station_uid)) - known:
        raise ValueError("unregistered station")
    rain = predictions.loc[predictions.variable.eq("precip_rate")]
    obs_other = observations.loc[~observations.variable.eq("solar")]
    if len(obs_other):
        obs_other = aggregate_intervals(obs_other, rain)
    observations = pd.concat([obs_other, observations.loc[observations.variable.eq("solar")]], ignore_index=True)
    return observations.loc[observations.variable.isin(variables), KEYS+["value"]].rename(columns={"value": "obs"}).merge(
        predictions.loc[predictions.variable.isin(variables), KEYS+["value"]].rename(columns={"value": "pred"}),
        on=KEYS, validate="one_to_one")


def metrics(pairs, variables, scales):
    """Equal station weight within variables; dimensionless mean across objectives."""
    finite(pairs.pred, "prediction")
    finite(pairs.obs, "observation")
    details = []
    for variable in variables:
        selected = pairs.loc[pairs.variable.eq(variable)]
        if selected.station_uid.nunique() < 2:
            raise ValueError(f"{variable}: need common support at two or more stations")
        errors = selected.assign(squared=(selected.pred-selected.obs)**2)
        mse = float(errors.groupby("station_uid").squared.mean().mean())
        details.append({"variable": variable, "units": UNITS[variable], "n": len(selected),
                        "stations": int(selected.station_uid.nunique()), "rmse": float(np.sqrt(mse)),
                        "scale": scales[variable]})
    loss = float(np.mean([r["rmse"]/r["scale"] for r in details]))
    if not np.isfinite(loss):
        raise ValueError("nonfinite objective (overflow)")
    return {"loss": loss, "metrics": details}


def uncertainty(pairs, variables, scales, *, repetitions=400, seed=0):
    """Date-block bootstrap: sample all stations of a UTC date together."""
    dates = pairs.valid_time.dt.strftime("%Y-%m-%d")
    groups = [pairs.loc[dates.eq(day)] for day in sorted(set(dates))]
    if len(groups) < 2:
        return {"method": "UTC date-block bootstrap", "interval95": None, "dates": len(groups)}
    random = np.random.default_rng(seed)
    losses = [metrics(pd.concat([groups[i] for i in random.integers(len(groups), size=len(groups))]), variables, scales)["loss"]
              for _ in range(repetitions)]
    return {"method": "UTC date-block bootstrap; correlated dates can understate uncertainty",
            "interval95": np.quantile(losses, [0.025, 0.975]).tolist(), "dates": len(groups),
            "repetitions": repetitions, "seed": seed}
