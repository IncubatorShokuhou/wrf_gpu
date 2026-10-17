"""Station bias or linear MOS, evaluated with expanding chronological folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNITS, identity, paired, read_csv, validate, write_json
from gpuwrf.verify.observations import KEYS


def coefficients(pairs, method):
    if method not in ("bias", "mos"):
        raise ValueError("method must be bias or mos")
    fitted = []
    for (station, variable), rows in pairs.groupby(["station_uid", "variable"]):
        x, y = rows.pred.to_numpy(), rows.obs.to_numpy()
        if len(rows) < 3:
            raise ValueError("each station/variable requires at least three training pairs")
        if method == "bias":
            intercept, slope = float(np.mean(y-x)), 1.0
        else:
            xc, yc = x-x.mean(), y-y.mean()
            variance = float(np.dot(xc, xc))
            if variance <= 0:
                raise ValueError("MOS requires varying forecasts per station/variable")
            slope = float(np.dot(xc, yc)/variance)
            intercept = float(y.mean()-slope*x.mean())
        if not np.isfinite([intercept, slope]).all():
            raise ValueError("nonfinite calibration coefficients")
        fitted.append({"station_uid": station, "variable": variable, "units": UNITS[variable],
                       "intercept": intercept, "slope": slope, "n": len(rows)})
    return fitted


def correct(pairs, fitted):
    table = pd.DataFrame(fitted)
    if table.empty or table.duplicated(["station_uid", "variable"]).any():
        raise ValueError("empty/duplicate calibration groups")
    for key in ("intercept", "slope"):
        if not np.isfinite(table[key].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite calibration coefficients")
    joined = pairs.merge(table[["station_uid", "variable", "intercept", "slope"]],
                         on=["station_uid", "variable"], how="left", validate="many_to_one")
    if joined.intercept.isna().any():
        raise ValueError("unseen station/variable; no pooled fallback")
    value = joined.intercept + joined.slope*joined.pred
    if not np.isfinite(value).all():
        raise ValueError("nonfinite calibrated predictions")
    nonnegative = joined.variable.ne("t2")
    clipped = int((nonnegative & value.lt(0)).sum())
    # Explicit statistical support constraint, never applied to model state.
    value.loc[nonnegative] = value.loc[nonnegative].clip(lower=0)
    return value.to_numpy(), clipped


def require_held_out_support(groups, scores):
    """A global model label cannot qualify a station/variable with no CV pairs."""
    qualified = {(row["station_uid"], row["variable"]) for row in scores
                 if isinstance(row.get("n"), int) and not isinstance(row["n"], bool) and row["n"] > 0}
    missing = set(groups)-qualified
    if missing:
        names = ", ".join(f"{station}/{variable}" for station, variable in sorted(missing))
        raise ValueError(f"calibration groups lack chronological held-out support: {names}")


def fit(station_path, observation_path, forecast_path, *, method="bias", min_train_dates=3):
    from gpuwrf.verify.observations import read_stations

    if isinstance(min_train_dates, bool) or not isinstance(min_train_dates, int) or min_train_dates < 3:
        raise ValueError("min_train_dates must be an integer >=3")
    stations = read_stations(station_path)
    obs, pred = read_csv(observation_path, observations=True), read_csv(forecast_path)
    if set(pred.variable)-set(UNITS):
        raise ValueError("calibration supports T2, scalar wind speed, precip_rate and solar")
    pairs = paired(obs, pred, stations, tuple(UNITS))
    if pairs.empty:
        raise ValueError("no common calibration support")
    dates = pairs.valid_time.dt.strftime("%Y-%m-%d")
    unique = sorted(set(dates))
    if len(unique) <= min_train_dates:
        raise ValueError("need training dates plus at least one later validation date")
    folds, evaluations = [], []
    for day in unique[min_train_dates:]:
        cutoff = pd.Timestamp(day, tz="UTC")
        train = pairs.loc[pairs.valid_time.lt(cutoff)]
        test = pairs.loc[dates.eq(day) & pairs.interval_start.ge(cutoff)]
        if test.empty:
            raise ValueError("validation date has only intervals crossing the date split")
        fitted = coefficients(train, method)
        # Every scored group has at least the promised number of prior UTC dates.
        for station, variable in set(zip(test.station_uid, test.variable)):
            group = train.loc[train.station_uid.eq(station) & train.variable.eq(variable)]
            if group.valid_time.dt.date.nunique() < min_train_dates:
                raise ValueError("insufficient past training dates for validation group")
        corrected, clipped = correct(test, fitted)
        eval_rows = test.copy()
        eval_rows["corrected"] = corrected
        evaluations.append(eval_rows)
        folds.append({"validation_date": day, "train_end": train.valid_time.max().isoformat(),
                      "train_n": len(train), "test_n": len(test), "support_clipped": clipped,
                      "coefficients": fitted})
    scored = pd.concat(evaluations, ignore_index=True)
    scores = []
    for (station, variable), group in scored.groupby(["station_uid", "variable"]):
        raw = group.pred-group.obs
        error = group.corrected-group.obs
        scores.append({"station_uid": station, "variable": variable, "units": UNITS[variable], "n": len(group),
                       "raw_rmse": float(np.sqrt(np.mean(raw*raw))),
                       "corrected_rmse": float(np.sqrt(np.mean(error*error))),
                       "corrected_bias": float(error.mean())})
    require_held_out_support(zip(pairs.station_uid, pairs.variable), scores)
    result = {"schema": "gpuwrf.calibration.v1", "method": method,
              "fitted_through": pairs.valid_time.max().isoformat(), "coefficients": coefficients(pairs, method),
              "cross_validation": {"method": "expanding prior UTC dates; next date held out", "folds": folds, "scores": scores},
              "constraint": "max(0, intercept+slope*x) for speed, rainfall rate and solar; no clipping of temperature",
              "sampling": "canonical verify station CSV; exact times/intervals; fixed issue/source selected by exporter",
              "inputs": [identity(p) for p in (station_path, observation_path, forecast_path)],
              "status": "OFFLINE_CROSS_VALIDATED; no guarantee of future improvement"}
    json.dumps(result, allow_nan=False)
    return result


def apply(model_path, forecast_path, output):
    model = json.loads(Path(model_path).read_text())
    if model.get("schema") != "gpuwrf.calibration.v1" or model.get("method") not in ("bias", "mos"):
        raise ValueError("unsupported calibration model")
    rows = read_csv(forecast_path)
    require_held_out_support(zip(rows.station_uid, rows.variable), model.get("cross_validation", {}).get("scores", []))
    cutoff = pd.Timestamp(model["fitted_through"])
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("model fitted_through must declare UTC/offset")
    if rows.valid_time.le(cutoff).any() or rows.interval_start.le(cutoff).any():
        raise ValueError("application must follow fitted_through, including interval starts")
    values, clipped = correct(rows.rename(columns={"value": "pred"}), model["coefficients"])
    rows["value"] = values
    rows = validate(rows)
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix+".json").exists():
        raise ValueError("calibration output exists")
    # Preserve canonical scalar fields only: raw vector components would disagree
    # with calibrated wind speed. The fit never claims vector/direction calibration.
    with output.open("x") as stream:
        rows[[*KEYS, "value"]].to_csv(stream, index=False)
    write_json(output.with_suffix(output.suffix+".json"), {
        "schema": "gpuwrf.calibration.application.v1", "model": identity(model_path),
        "forecast": identity(forecast_path), "output": identity(output), "support_clipped": clipped})
    return {"output": str(output), "n": len(rows), "support_clipped": clipped}
