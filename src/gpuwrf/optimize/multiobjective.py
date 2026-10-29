"""Offline, paired regional/point skill guard for optimizer waves.

Inputs are instantaneous, already co-located values, never raw WRF/MSG grids.
Cloud adapters supply fractional/binary cloud cover on their declared footprint.
All candidates are tested together: selection cannot shrink the testing family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gpuwrf.verify.observations import finite, utc

KEYS = ["case_id", "station_uid", "valid_time", "variable"]
UNITS = {"cloud": "fraction", "wind": "m/s", "t2": "degC", "rh": "%", "gust": "m/s"}
REFERENCE_ARMS = ("wn3_0p05", "wn3_0p1")
SCHEMA = "gpuwrf.optimize.multiobjective.v1"


def _rows(frame, name, variables):
    """Validate before any join/aggregation can hide NaN or missing support."""
    frame = frame.copy()
    if "case_id" not in frame:
        frame["case_id"] = "case"
    if not set(KEYS + ["value"]) <= set(frame) or frame.empty:
        raise ValueError(f"{name}: need nonempty station_uid,valid_time,variable,value rows")
    if frame.station_uid.isna().any() or frame.station_uid.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: missing station_uid")
    frame["station_uid"] = frame.station_uid.astype(str)
    if frame.case_id.isna().any() or frame.case_id.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: missing case_id")
    frame["case_id"] = frame.case_id.astype(str)
    frame["valid_time"] = utc(frame.valid_time).astype("datetime64[ns, UTC]")
    frame["value"] = finite(frame.value, name)
    if set(frame.variable) != set(variables):
        raise ValueError(f"{name}: variables must match the declared objectives exactly")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name}: duplicate point/time/variable; select one forecast issue")
    if "interval_start" in frame:
        if not utc(frame.interval_start, "interval_start").equals(frame.valid_time):
            raise ValueError(f"{name}: only instantaneous aligned values are supported")
    for variable, lo, hi in (("cloud", 0, 1), ("rh", 0, 100), ("wind", 0, float("inf")), ("gust", 0, float("inf"))):
        if not frame.loc[frame.variable.eq(variable), "value"].between(lo, hi).all():
            raise ValueError(f"{name}: {variable} values outside [{lo}, {hi}]")
    return frame[KEYS + ["value"]].sort_values(KEYS).reset_index(drop=True)


def _sign_p(delta):
    """One-sided binomial sign test of median(delta) <= 0; ties nonpositive.

    Dropping ties is invalid for this composite null with an atom at zero:
    conditional positive frequency can exceed 1/2 despite a zero median.
    """
    n = len(delta)
    positive = int(np.count_nonzero(delta > 0))
    if not n or positive <= n / 2:
        return 1.0  # conservative shortcut, cannot be a degradation discovery
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            - n * math.log(2) for k in range(positive, n + 1)]
    peak = max(logs)
    return min(1.0, math.exp(peak) * sum(math.exp(x - peak) for x in logs))


def _holm(rows, alpha):
    """Strong FWER control under arbitrary spatial/baseline dependence."""
    order = sorted(range(len(rows)), key=lambda n: (rows[n]["p_value"], n))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(rows) - rank) * rows[index]["p_value"]))
        rows[index]["p_adjusted"] = previous
        rows[index]["significantly_worse"] = previous <= alpha


def _interval(delta, repetitions, seed):
    """Paired date-block bootstrap, pointwise 95% CI of the median daily effect."""
    rng = np.random.default_rng(seed)
    medians = []
    # Bounded workspace even when a study spans many dates.
    for start in range(0, repetitions, 128):
        index = rng.integers(len(delta), size=(min(128, repetitions-start), len(delta)))
        medians.extend(np.median(delta[index], axis=1).tolist())
    return np.quantile(medians, [0.025, 0.975]).tolist()


def _simultaneous(rows, alpha):
    """Distribution-free sign/order-statistic CIs, Bonferroni over full family."""
    tail = alpha / len(rows)
    for row in rows:
        delta = np.sort(row.pop("_daily"))
        n, chosen, probability = len(delta), 0, 0.0
        for k in range(1, n//2 + 1):
            probability += math.comb(n, k-1)/(1 << n)
            if 2*probability <= tail:
                chosen = k
            else:
                break
        row["simultaneous_interval"] = [float(delta[chosen-1]), float(delta[n-chosen])] if chosen else [None, None]
        row["interval_status"] = "SIMULTANEOUS_FINITE" if chosen else "UNBOUNDED_INSUFFICIENT_DATES"


def assess(observations, candidates, reference, control, regions, *, scales, weights=None,
           evaluation_contract, margins=None, parameters=None, points=None, alpha=0.05, repetitions=400, seed=0):
    """Joint loss + no significant degradation versus BOTH supplied forecasts.

    ``regions`` maps each point ID to one region. Every variable must cover every
    registered point (or its explicit ``points[variable]`` registry); every
    forecast must exactly cover the observation keys.
    Loss is equal-region/equal-point MAE, scaled and weighted across variables.
    Each guard tests the median of paired daily MAE differences (physical units)
    against a predeclared practical margin, at each point AND each region.
    Holm's family includes every candidate, variable, location and baseline.
    A non-rejection is a screening result, never proof of non-inferiority.
    """
    from .evaluation import bind, weighted_mae

    variables = sorted(scales)
    if not variables or not set(variables) <= set(UNITS):
        raise ValueError(f"scales must declare a subset of {tuple(UNITS)}")
    weights = {v: 1.0 for v in variables} if weights is None else dict(weights)
    margins = {v: 0.0 for v in variables} if margins is None else dict(margins)
    for name, values, zero in (("scales", scales, False), ("weights", weights, False), ("margins", margins, True)):
        if set(values) != set(variables) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) or (x < 0 if zero else x <= 0) for x in values.values()):
            raise ValueError(f"{name}: require a {'nonnegative' if zero else 'positive'} finite value per variable")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("bootstrap repetitions must be an integer >= 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not candidates or any(not isinstance(k, str) or not k.strip() for k in candidates):
        raise ValueError("need named candidates")
    if not regions or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                          for k, v in regions.items()):
        raise ValueError("regions must map nonempty point IDs to nonempty region names")
    points = {v: sorted(regions) for v in variables} if points is None else points
    if set(points) != set(variables) or any(not isinstance(p, list) or not p or
            any(not isinstance(s, str) or s not in regions for s in p) or len(set(p)) != len(p)
            for p in points.values()) or set().union(*(set(p) for p in points.values())) != set(regions):
        raise ValueError("points must declare unique registered points per variable, covering the region map")
    parameters = parameters or {}
    if set(parameters) - set(candidates):
        raise ValueError("parameters name an unknown candidate")
    for values in parameters.values():
        if not isinstance(values, dict) or any(not isinstance(k, str) or not k.strip() or isinstance(v, bool)
                or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in values.items()):
            raise ValueError("parameters must map names to finite numeric settings")
    truth = _rows(observations, "observations", variables)
    cohort = bind(evaluation_contract, truth, regions)
    for field, value in (("scales", scales), ("weights", weights), ("margins", margins), ("alpha", alpha)):
        if evaluation_contract[field] != value:
            raise ValueError(f"{field} differs from ALISIOS evaluation contract; optimizer cannot set gates")
    for variable in variables:
        if set(truth.loc[truth.variable.eq(variable), "station_uid"]) != set(points[variable]):
            raise ValueError(f"{variable}: missing/unregistered observation stations")
    support = truth[KEYS]

    def errors(frame, name):
        frame = _rows(frame, name, variables)
        if not frame[KEYS].equals(support):
            raise ValueError(f"{name}: missing/extra station/time/variable support")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.abs(frame.value.to_numpy() - truth.value.to_numpy())
        finite(delta, f"{name} absolute errors")
        result = support.assign(error=delta)
        result["region"] = result.station_uid.map(regions)
        result["date"] = result.valid_time.dt.strftime("%Y-%m-%d")
        for field in ("episode", "regime", "lead"):
            result[field] = cohort[field]
        return result

    if not isinstance(reference, dict) or set(reference) != set(REFERENCE_ARMS):
        raise ValueError("both WN3 reference arms wn3_0p05 and wn3_0p1 are required; missing arm is UNAVAILABLE")
    baseline_errors = {name: errors(reference[name], name) for name in REFERENCE_ARMS}
    baseline_errors["control"] = errors(control, "control")
    objective = hashlib.sha256(json.dumps(evaluation_contract, sort_keys=True, allow_nan=False).encode())
    objective.update(truth.to_json(orient="split", date_format="iso", date_unit="ns").encode())
    for name, frame in sorted(baseline_errors.items()):
        objective.update(name.encode())
        objective.update(frame.to_json(orient="split", date_format="iso", date_unit="ns").encode())
    tests, scores = [], []
    for cid in sorted(candidates):
        error = errors(candidates[cid], cid)
        variable_loss = weighted_mae(error, evaluation_contract)
        loss = sum(float(variable_loss[v]) / scales[v] * weights[v] for v in variables) / sum(weights.values())
        finite([loss], "joint loss")
        scores.append({"id": cid, "loss": loss, "mae": {v: float(variable_loss[v]) for v in variables}})
        for baseline, base in baseline_errors.items():
            # Region daily errors average each point equally, regardless of sample count.
            daily = error.assign(delta=error.error.to_numpy()-base.error.to_numpy()).groupby(
                ["variable", "region", "lead", "regime", "station_uid", "date"], sort=True).delta.mean().reset_index()
            region_daily = daily.groupby(["variable", "region", "lead", "regime", "date"], sort=True).delta.mean().reset_index()
            for level, frame, columns in (("point", daily, ["variable", "region", "lead", "regime", "station_uid"]),
                                          ("region", region_daily, ["variable", "region", "lead", "regime"])):
                for key, selected in frame.groupby(columns, sort=True):
                    variable, region = key[:2]
                    delta = selected.sort_values("date").delta.to_numpy()
                    enough = len(delta) >= 2
                    tests.append({"candidate": cid, "baseline": baseline, "variable": variable, "region": region,
                                  "level": level, "point": key[4] if level == "point" else None,
                                  "lead": key[2], "regime": key[3],
                                  "units": UNITS[variable], "dates": len(delta), "sufficient_support": enough,
                                  "effect": float(np.median(delta)), "mean_effect": float(np.mean(delta)),
                                  "interval95": _interval(delta, repetitions, seed) if enough else None,
                                  "_daily": delta,
                                  "margin": margins[variable],
                                  "p_value": _sign_p(delta-margins[variable]) if enough else 1.0})
    _holm(tests, alpha)
    _simultaneous(tests, alpha)
    for score in scores:
        relevant = [r for r in tests if r["candidate"] == score["id"]]
        score["admissible"] = all(r["sufficient_support"] and not r["significantly_worse"] for r in relevant)
        score["status"] = ("HARM_VETO" if any(r["significantly_worse"] for r in relevant) else
                           "INSUFFICIENT_DATES" if not score["admissible"] else "ADMISSIBLE")
    scores.sort(key=lambda r: (not r["admissible"], r["loss"], r["id"]))
    flags = {r["id"]: r["admissible"] for r in scores}
    impacts = []
    for test in tests:
        if test["level"] == "region":
            for parameter, setting in sorted((parameters.get(test["candidate"]) or {"candidate": None}).items()):
                impacts.append({**test, "parameter": parameter, "setting": setting,
                                "joint_parameter_change": len(parameters.get(test["candidate"], {})) > 1,
                                "admissible": flags[test["candidate"]]})
    report = {"schema": SCHEMA, "scores": scores, "tests": tests, "impact_matrix": impacts,
              "objective_identity": objective.hexdigest(),
              "recommended": next((r["id"] for r in scores if r["admissible"]), None),
              "support": {"rows": len(truth), "points": len(regions), "regions": sorted(set(regions.values()))},
              "scales": dict(sorted(scales.items())), "weights": dict(sorted(weights.items())),
              "points": {v: sorted(points[v]) for v in variables},
              "evaluation_contract": json.loads(json.dumps(evaluation_contract, sort_keys=True)),
              "statistics": {"test": "one-sided exact paired sign test; UTC-date mean MAE deltas; median > margin",
                             "assumptions": "independent UTC date blocks; no symmetry assumption; ties counted as nonpositive",
                             "multiple_testing": "Holm strong FWER across all candidates/variables/points/regions/leads/regimes/three baselines",
                             "family_size": len(tests), "alpha": alpha,
                             "simultaneous_confidence": 1-alpha,
                             "interval": "simultaneous (1-alpha) exact sign/order-statistic median intervals, Bonferroni over full family; null bounds mean unbounded",
                             "pointwise_interval": "descriptive 95% paired date-block percentile bootstrap of median daily effect",
                             "repetitions": repetitions, "seed": seed},
              "status": "EVALUATED",
              "qualification": "SCREENING_ONLY; non-rejection does not establish non-inferiority; no release promotion"}
    json.dumps(report, allow_nan=False)
    return report


def markdown(report):
    """Human view of the exact JSON impact matrix; positive effects are worse."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Multi-objective forecast screening", "", report["status"], "", report["qualification"], "",
             f"Recommended admissible candidate: {cell(report['recommended'])}", "",
             "Positive effect = larger daily absolute error than the baseline (physical units).",
             "CI = simultaneous sign/order-statistic interval; null bounds mean unbounded due to few dates.",
             report["statistics"]["multiple_testing"] + "; alpha=" + str(report["statistics"]["alpha"]), "",
             "Joint parameter changes are associations, not isolated causal parameter effects.", "",
             "| Candidate | Parameter | Setting | Variable | Region | Lead | Regime | Baseline | Effect | Simultaneous CI | Adjusted p | Admissible |",
             "|---|---|---|---|---|---|---|---|---:|---|---:|---|"]
    for row in report["impact_matrix"]:
        values = [row[k] for k in ("candidate", "parameter", "setting", "variable", "region", "lead", "regime", "baseline")]
        values += [f"{row['effect']:.6g}", row["simultaneous_interval"], f"{row['p_adjusted']:.6g}", row["admissible"]]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    return "\n".join(lines) + "\n"


def assess_file(path):
    """Read only explicit facade/export files; bind every input by SHA-256."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from .data import identity

    refuse_protected_inputs("optimize assess", [path])
    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"assessment requires schema {SCHEMA}")
    required = {"schema", "observations", "references", "control", "candidates", "regions", "evaluation_contract"}
    allowed = required | {"points", "repetitions", "seed"}
    if set(raw) - allowed or required - set(raw):
        raise ValueError("assessment has missing or unknown fields")
    resolve = lambda p: path.parent / str(p)
    refspecs = raw["references"]
    if not isinstance(refspecs, dict) or set(refspecs) != set(REFERENCE_ARMS) or any(not isinstance(s, dict) or "path" not in s for s in refspecs.values()):
        raise ValueError("UNAVAILABLE: two WN3 Parquet references wn3_0p05/wn3_0p1 with explicit contracts required")
    contract_spec = raw["evaluation_contract"]
    if not isinstance(contract_spec, dict) or set(contract_spec) != {"path", "sha256"}:
        raise ValueError("evaluation_contract must bind an externally frozen ALISIOS file path and SHA256")
    files = [resolve(raw["observations"]), resolve(raw["control"]), resolve(contract_spec["path"])]
    files += [resolve(refspecs[k]["path"]) for k in REFERENCE_ARMS]
    files += [resolve(v["forecast"]) for v in raw["candidates"].values()]
    refuse_protected_inputs("optimize assess forecasts", files)
    provenance = [identity(p) for p in [path, *files]]
    if identity(files[2])["sha256"] != contract_spec["sha256"]:
        raise ValueError("ALISIOS evaluation contract SHA256 mismatch")
    contract = json.loads(files[2].read_text())
    read = lambda p: pd.read_csv(p, dtype={"station_uid": str})
    from .reference import read as read_reference
    reference = {k: read_reference(resolve(refspecs[k]["path"]), refspecs[k]) for k in REFERENCE_ARMS}
    if any(set(r.variable) != set(contract["scales"]) for r in reference.values()):
        raise ValueError("reference variables must exactly match ALISIOS scoring objectives")
    result = assess(read(files[0]), {k: read(resolve(v["forecast"])) for k, v in raw["candidates"].items()},
                    reference, read(files[1]), raw["regions"], evaluation_contract=contract,
                    **{k: contract[k] for k in ("scales", "weights", "margins", "alpha")},
                    parameters={k: v.get("parameters", {}) for k, v in raw["candidates"].items()},
                    **{k: raw[k] for k in ("points", "repetitions", "seed") if k in raw})
    if provenance != [identity(p) for p in [path, *files]]:
        raise ValueError("assessment inputs changed while scoring")
    result["inputs"] = provenance
    result["reference_contracts"] = refspecs
    return result


def command(args):
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from .data import write_json
    import sys

    try:
        md = args.output.with_suffix(".md")
        refuse_protected_inputs("optimize assess output", [], outputs=[args.output, md])
        if args.output == md or args.output.exists() or md.exists():
            raise ValueError("use a new JSON output path and adjacent Markdown path")
        result = assess_file(args.assessment)
        write_json(args.output, result)
        with md.open("x", encoding="utf-8") as stream:
            stream.write(markdown(result))
        print(json.dumps({"output": str(args.output), "recommended": result["recommended"], "status": result["status"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({"status": "UNAVAILABLE", "recommended": None, "admissible": False, "reason": str(exc)}))
        print(f"gpuwrf optimize assess: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(command(parser.parse_args()))
