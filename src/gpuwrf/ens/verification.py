"""Ensemble verification using the station extractor and schemas of gpuwrf verify."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpuwrf.verify.observations import KEYS, read_forecast_csv, read_observations, read_stations
from gpuwrf.verify.scores import UNITS, aggregate_intervals, score_sources
from gpuwrf.verify.wrfout import extract_wrfout

from .members import identity, read_members
from .statistics import crps, levels, products, rank_histogram, reliability, spread_skill


def verify_members(member_path, station_path, observation_path, *, variable="t2", thresholds=(0,),
                   quantiles=(0.1, 0.5, 0.9), observation_format="csv", precip_hours=None, **extraction):
    if variable not in ("t2", "wind", "gust", "precip_rate"):
        raise ValueError("ensemble scores require t2, wind, gust or precip_rate (no linear wind-direction scores)")
    thresholds = levels(thresholds, "thresholds").tolist()
    quantiles = levels(quantiles, "quantiles", probability=True).tolist()
    members = read_members(member_path)
    stations = read_stations(station_path)
    obs, obs_audit = read_observations(observation_path, observation_format, precip_hours=precip_hours)
    if set(obs.station_uid)-set(stations.station_uid):
        raise ValueError("observations contain unregistered stations")
    sources, audits = {}, {}
    for member in members:
        files = member["files"]
        if len(files) == 1 and files[0].suffix.lower() == ".csv":
            data = read_forecast_csv(files[0])
            audit = {"format": "station CSV", "corrections": "already applied by producer"}
        else:
            data, audit = extract_wrfout(files, stations, **extraction)
        if set(data.station_uid)-set(stations.station_uid):
            raise ValueError("forecast contains unregistered stations")
        sources[member["id"]], audits[member["id"]] = data, audit
    target = next(iter(sources.values()))
    target = target[target.variable.eq("precip_rate")]
    obs = aggregate_intervals(obs, target)
    sources = {name: aggregate_intervals(data, target) for name, data in sources.items()}
    # The existing verifier supplies the authoritative common-support count and
    # deterministic scores. Use its exact keys/interval adapters for scalar rows.
    deterministic = score_sources(obs, sources)
    selected = [data.loc[data.variable.eq(variable)].set_index(KEYS).value.rename(name)
                for name, data in [("observation", obs), *[(f"member{i}", d) for i, d in enumerate(sources.values())]]]
    pairs = pd.concat(selected, axis=1, join="inner").sort_index()
    if pairs.empty:
        raise ValueError("no common station/time/variable/interval support for every member and observations")
    if any(row["n"] != len(pairs) for row in deterministic["metrics"] if row["variable"] == variable):
        raise ValueError("ensemble support disagrees with gpuwrf verify")
    truth = pairs.observation.to_numpy()
    values = pairs.drop(columns="observation").to_numpy().T
    result = products(values, thresholds=thresholds, quantiles=quantiles)
    empirical, fair = crps(values, truth), crps(values, truth, fair=True)
    table = pairs.reset_index()[KEYS].copy()
    table["observation"], table["mean"], table["spread"] = truth, result["mean"], result["spread"]
    table["crps"], table["fair_crps"] = empirical, fair
    for index, quantile in enumerate(quantiles):
        table[f"quantile_{quantile!r}"] = result["quantile"][index]
    events = []
    for index, threshold in enumerate(thresholds):
        probability = result["probability"][index]
        event = truth > threshold
        table[f"probability_gt_{threshold!r}"] = probability
        events.append({"threshold": float(threshold), "brier_score": float(np.mean((probability-event)**2)),
                       "reliability": reliability(probability, event)})
    report = {"schema": "gpuwrf.ensemble.verification.v1", "status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
              "variable": variable, "units": UNITS[variable], "members": list(sources), "member_count": len(members),
              "n": len(pairs), "n_stations": int(table.station_uid.nunique()), "matched_keys": KEYS,
              "crps": float(empirical.mean()), "fair_crps": float(fair.mean()),
              "fair_assumption": "iid draws required for unbiased interpretation; not established for physics/lagged members",
              "spread_skill": spread_skill(values, truth), "rank_histogram": rank_histogram(values, truth).tolist(),
              "rank_ties": "fractional uniform tie allocation (expected random ranks)", "event": "X > threshold",
              "events": events, "spread_ddof": 1, "quantile_method": "linear", "quantile_levels": list(quantiles),
              "support": {"observation_rows": int(obs.variable.eq(variable).sum()),
                          "member_rows": {name: int(data.variable.eq(variable).sum()) for name, data in sources.items()},
                          "precipitation_target": members[0]["id"]},
              "extraction": audits, "observations": obs_audit, "deterministic_member_scores": deterministic,
              "inputs": [identity(p) for p in dict.fromkeys([member_path, station_path, observation_path,
                          *[p for m in members for p in m["files"]]])]}
    # Catch non-finite derived values before writing any products.
    json.dumps(report, allow_nan=False)
    return report, table


def write_verification(report, table, output):
    """JSON/CSV and standard standalone SVG figures; matplotlib is an optional extra."""
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ValueError('ensemble figures require pip install "gpuwrf[ens]"') from exc
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"ensemble.json").write_text(payload)
    table.to_csv(output/"paired.csv", index=False)
    fig = Figure(figsize=(6, 4))
    axis = fig.subplots()
    axis.bar(np.arange(len(report["rank_histogram"])), report["rank_histogram"])
    axis.set(xlabel="Observation rank (0..M)", ylabel="Count (fractional ties)", title="Ensemble rank histogram")
    fig.savefig(output/"rank_histogram.svg", bbox_inches="tight")
    fig.clear()
    axis = fig.subplots()
    axis.plot([0, 1], [0, 1], "--", color="gray")
    for event in report["events"]:
        occupied = [row for row in event["reliability"] if row["n"]]
        axis.plot([row["forecast_probability"] for row in occupied], [row["observed_frequency"] for row in occupied],
                  "o-", label=f"> {event['threshold']:g} {report['units']}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast probability", ylabel="Observed frequency",
             title="Reliability (common station/time support)")
    axis.legend()
    fig.savefig(output/"reliability.svg", bbox_inches="tight")
    fig.clear()
    axis = fig.subplots()
    # Exact same paired samples for both axes. Counts and pooled statistics are
    # in JSON; this descriptive scatter does not imply independent samples.
    axis.scatter(table.spread, np.abs(table["mean"]-table.observation), s=12, alpha=0.5)
    axis.set(xlabel=f"Sample spread ({report['units']})", ylabel=f"Absolute ensemble-mean error ({report['units']})",
             title="Spread and skill samples")
    fig.savefig(output/"spread_skill.svg", bbox_inches="tight")
