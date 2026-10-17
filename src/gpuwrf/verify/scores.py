"""Descriptive, exact-support scores. No statistical promotion/release verdict."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .observations import KEYS, VARIABLES, finite, validate_rows

UNITS = {"t2": "degC", "wind": "m/s", "wind_direction": "degree", "gust": "m/s", "precip_rate": "mm/h"}


def scalar_scores(pred, obs):
    pred, obs = finite(pred, "prediction"), finite(obs, "observation")
    if pred.shape != obs.shape or pred.ndim != 1:
        raise ValueError("scores need equally shaped 1D pairs")
    if not pred.size:
        return {"n": 0, "bias": None, "mae": None, "rmse": None}
    error = pred-obs
    return {"n": int(pred.size), "bias": float(error.mean()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error*error)))}


def precip_scores(pred, obs, threshold):
    """Inclusive >= events, ETS=(H-Hrandom)/(H+M+F-Hrandom), FBI=(H+F)/(H+M)."""
    pred, obs = finite(pred, "precip prediction"), finite(obs, "precip observation")
    if pred.shape != obs.shape or pred.ndim != 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("precip contingency needs equal 1D pairs and positive finite threshold")
    p, o = pred >= threshold, obs >= threshold
    h, m, f, c = (int(x.sum()) for x in (p & o, ~p & o, p & ~o, ~p & ~o))
    random = (h+m)*(h+f)/pred.size if pred.size else 0
    denominator = h+m+f-random
    return {"threshold_mmh": float(threshold), "hits": h, "misses": m, "false_alarms": f,
            "correct_negatives": c, "ets": (h-random)/denominator if denominator else None,
            "fbi": (h+f)/(h+m) if h+m else None}


def aggregate_intervals(data, targets):
    """Integrate disjoint, piecewise-constant rates onto complete target windows.

    No time interpolation for instantaneous variables. Missing rain coverage is
    unavailable, never zero. The target cadence is explicit and reported.
    """
    data = validate_rows(data)
    rain = data[data.variable.eq("precip_rate")]
    groups = {uid: part.sort_values("interval_start") for uid, part in rain.groupby("station_uid")}
    rows = []
    for target in targets.drop_duplicates(["station_uid", "interval_start", "valid_time"]).itertuples():
        part = groups.get(target.station_uid)
        if part is None:
            continue
        start, end = target.interval_start, target.valid_time
        cursor, amount = start, 0.0
        # Source intervals are sorted/disjoint: examine only actual overlaps,
        # rather than rescanning the whole station history for every window.
        left = int(part.valid_time.searchsorted(start, side="right"))
        right = int(part.interval_start.searchsorted(end, side="left"))
        for record in part.iloc[left:right].itertuples():
            a, b = max(start, record.interval_start), min(end, record.valid_time)
            if b <= a:
                continue
            if a != cursor:
                break
            amount += record.value*(b-a).total_seconds()/3600
            cursor = b
        if cursor == end:
            rows.append({"station_uid": target.station_uid, "valid_time": end,
                         "variable": "precip_rate", "value": amount/((end-start).total_seconds()/3600),
                         "interval_start": start})
    return pd.concat([data[~data.variable.eq("precip_rate")], pd.DataFrame(rows)], ignore_index=True)


def _parts(data, variable, label):
    return data.loc[data.variable.eq(variable), KEYS + ["value"]].rename(columns={"value": label})


def _components(data):
    wind = data[data.variable.eq("wind")][["station_uid", "valid_time", "value"]].copy()
    directions = data[data.variable.eq("wind_direction")][["station_uid", "valid_time", "value"]].rename(columns={"value": "direction"})
    if {"u_ms", "v_ms"} <= set(data):
        return data[data.variable.eq("wind")][["station_uid", "valid_time", "u_ms", "v_ms"]]
    merged = wind.merge(directions, on=["station_uid", "valid_time"], validate="one_to_one")
    angle = np.deg2rad(merged.direction.to_numpy())
    merged["u_ms"], merged["v_ms"] = -merged.value*np.sin(angle), -merged.value*np.cos(angle)
    return merged[["station_uid", "valid_time", "u_ms", "v_ms"]]


def score_sources(observations, sources, *, thresholds=(0.1, 1, 5)):
    """Intersect truth and EVERY forecast arm before computing each metric."""
    obs = validate_rows(observations)
    if not sources or not all(isinstance(name, str) and name.strip() for name in sources):
        raise ValueError("need named forecast sources")
    sources = {name: validate_rows(data) for name, data in sources.items()}
    thresholds = tuple(thresholds)
    if not thresholds or any(not np.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("precip thresholds must be positive finite mm/h values")
    names = list(sources)
    # Direction eligibility must be common across ALL arms, not independently selected.
    speed = _parts(obs, "wind", "obs").drop(columns=["variable", "interval_start"])
    for i, data in enumerate(sources.values()):
        speed = speed.merge(_parts(data, "wind", f"p{i}").drop(columns=["variable", "interval_start"]),
                            on=["station_uid", "valid_time"], validate="one_to_one")
    strong = speed.loc[(speed[["obs", *[f"p{i}" for i in range(len(names))]]] > 4).all(axis=1), ["station_uid", "valid_time"]]
    vector = _components(obs).rename(columns={"u_ms": "ou", "v_ms": "ov"})
    for i, data in enumerate(sources.values()):
        vector = vector.merge(_components(data).rename(columns={"u_ms": f"u{i}", "v_ms": f"v{i}"}),
                              on=["station_uid", "valid_time"], validate="one_to_one")
    metrics = []
    for variable in VARIABLES:
        paired = _parts(obs, variable, "obs")
        for i, data in enumerate(sources.values()):
            paired = paired.merge(_parts(data, variable, f"p{i}"), on=KEYS, validate="one_to_one")
        before_direction_filter = len(paired)
        if variable == "wind_direction":
            paired = paired.merge(strong, on=["station_uid", "valid_time"], validate="one_to_one")
        for i, name in enumerate(names):
            pred, truth = paired[f"p{i}"].to_numpy(), paired.obs.to_numpy()
            if variable == "wind_direction":
                error = (pred-truth+180) % 360-180
                values = scalar_scores(error, np.zeros_like(error))
                values["median_abs_error"] = float(np.median(np.abs(error))) if len(error) else None
            else:
                values = scalar_scores(pred, truth)
            values.update(source=name, variable=variable, units=UNITS[variable],
                          n_stations=int(paired.station_uid.nunique()),
                          observation_rows=int(obs.variable.eq(variable).sum()),
                          forecast_rows=int(sources[name].variable.eq(variable).sum()),
                          status="SCORED_DIAGNOSTIC" if len(paired) else "UNAVAILABLE_NO_COMMON_SUPPORT")
            if variable == "wind_direction":
                values["weak_wind_excluded"] = before_direction_filter-len(paired)
            if variable == "wind":
                squared = (vector[f"u{i}"]-vector.ou)**2 + (vector[f"v{i}"]-vector.ov)**2
                values["vector_n"] = len(vector)
                values["vector_rmse"] = float(np.sqrt(squared.mean())) if len(vector) else None
            if variable == "precip_rate":
                values["contingency"] = [precip_scores(pred, truth, threshold) for threshold in thresholds]
            metrics.append(values)
    comparisons = []
    if len(names) == 2:
        for variable in VARIABLES:
            a, b = [next(row for row in metrics if row["source"] == name and row["variable"] == variable) for name in names]
            comparisons.append({"variable": variable, "n": a["n"], "candidate": names[0], "baseline": names[1],
                                "rmse_delta": a["rmse"]-b["rmse"] if a["n"] else None,
                                "rmse_ratio": a["rmse"]/b["rmse"] if a["n"] and b["rmse"] else None})
    return {"schema": "gpuwrf.station_verification.v1", "status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
            "metrics": metrics, "comparisons": comparisons,
            "matched_keys": KEYS, "direction_min_speed_ms_exclusive": 4,
            "precip_threshold_units": "mm/h"}
