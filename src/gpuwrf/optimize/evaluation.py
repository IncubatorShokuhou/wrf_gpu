"""Consume a frozen external evaluation contract; never choose optimizer gates."""
from __future__ import annotations

import hashlib
import json
import math
import re

import pandas as pd


def support_digest(rows):
    """Immutable pair denominator, independent of input row order/storage units."""
    from .multiobjective import KEYS
    frame = rows.copy()
    if "case_id" not in frame:
        frame["case_id"] = "case"
    frame["valid_time"] = pd.to_datetime(frame.valid_time, utc=True).astype("datetime64[ns, UTC]")
    frame["valid_time"] = frame.valid_time.map(lambda t: t.isoformat())
    payload = frame[KEYS].sort_values(KEYS).to_json(orient="values")
    return hashlib.sha256(payload.encode()).hexdigest()


def bind(contract, truth, regions):
    required = {"authority", "contract_id", "frozen", "support_sha256", "cases", "episode_weights",
                "regime_weights", "lead_bins", "scales", "weights", "margins", "alpha", "regions"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("evaluation contract must explicitly freeze authority, support, cohorts, weights and gates")
    if contract["authority"] != "ALISIOS" or contract["frozen"] is not True or not isinstance(contract["contract_id"], str) or not contract["contract_id"].strip():
        raise ValueError("evaluation gate authority must be an externally frozen ALISIOS contract")
    if contract["regions"] != regions:
        raise ValueError("region assignments differ from frozen evaluation contract")
    if not isinstance(contract["support_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", contract["support_sha256"]) or support_digest(truth) != contract["support_sha256"]:
        raise ValueError("pair denominator differs from frozen evaluation contract")
    cases = contract["cases"]
    if not isinstance(cases, dict) or set(cases) != set(truth.case_id):
        raise ValueError("missing/extra cases in fixed episode/regime cohort")
    for case in cases.values():
        if not isinstance(case, dict) or set(case) != {"issue", "episode", "regime"} or any(not isinstance(v, str) or not v.strip() for v in case.values()):
            raise ValueError("each frozen case requires issue, episode and regime")
        timestamp = pd.Timestamp(case["issue"])
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("case issue must declare UTC/offset")
    for name, field in (("episode_weights", "episode"), ("regime_weights", "regime")):
        values = contract[name]
        if not isinstance(values, dict) or set(values) != {c[field] for c in cases.values()} or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError(f"{name} must fix positive weights for every declared cohort")
    if any(len({c["regime"] for c in cases.values() if c["episode"] == e}) != 1 for e in contract["episode_weights"]):
        raise ValueError("each episode must belong to one fixed regime")
    bins = contract["lead_bins"]
    if not isinstance(bins, list) or not bins:
        raise ValueError("fixed lead_bins are required")
    previous, names = None, set()
    for item in bins:
        if not isinstance(item, dict) or set(item) != {"name", "start_hour", "end_hour"} or not isinstance(item["name"], str) or not item["name"].strip() or item["name"] in names:
            raise ValueError("lead bins need unique names and explicit start/end hours")
        a, b = item["start_hour"], item["end_hour"]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (a,b)) or not 0 <= a < b or (previous is not None and a < previous):
            raise ValueError("lead bins must be sorted, nonnegative and nonoverlapping")
        previous = b
        names.add(item["name"])
    result = truth.copy()
    result["episode"] = result.case_id.map(lambda c: cases[c]["episode"])
    result["regime"] = result.case_id.map(lambda c: cases[c]["regime"])
    issues = pd.to_datetime(result.case_id.map(lambda c: cases[c]["issue"]), utc=True)
    lead = (result.valid_time - issues).dt.total_seconds() / 3600
    result["lead"] = None
    for item in bins:
        result.loc[lead.ge(item["start_hour"]) & lead.lt(item["end_hour"]), "lead"] = item["name"]
    if result.lead.isna().any():
        raise ValueError("observation support outside frozen lead bins")
    return result[["episode", "regime", "lead"]]


def weighted_mae(error, contract):
    """Equal point/region weight; frozen episode weights within each regime."""
    point = error.groupby(["variable", "regime", "episode", "region", "station_uid"], sort=True).error.mean()
    episode = point.groupby(level=["variable", "regime", "episode", "region"]).mean().groupby(
        level=["variable", "regime", "episode"]).mean()
    out = {}
    for variable in sorted(set(error.variable)):
        regimes = {}
        for regime in sorted(contract["regime_weights"]):
            part = episode.loc[(variable, regime)]
            expected = {c["episode"] for c in contract["cases"].values() if c["regime"] == regime}
            if set(part.index) != expected:
                raise ValueError("missing variable/episode support; fixed weight denominator cannot shrink")
            weights = contract["episode_weights"]
            regimes[regime] = sum(float(v)*weights[e] for e,v in part.items()) / sum(weights[e] for e in expected)
        weights = contract["regime_weights"]
        out[variable] = sum(v*weights[r] for r,v in regimes.items()) / sum(weights.values())
    return out
