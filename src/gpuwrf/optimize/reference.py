"""ALISIOS WN3 point/footprint Parquet contract; no native-grid interpretation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gpuwrf.verify.observations import finite, utc

VARIABLES = {"cloud_occurrence": ("cloud", "1"), "wind_speed_ms": ("wind", "m s-1"),
             "t2_c": ("t2", "degC"), "rh_pct": ("rh", "%"), "wind_gust_ms": ("gust", "m s-1")}
KEYS = ["case_id", "valid_time", "location_id", "variable", "reference_role", "native_statistic", "native_interval"]
REQUIRED = set(KEYS + ["value", "units", "native_lead_hour", "operator_sha256", "calibrator_sha256",
                       "available", "unavailable_reason"])
INSTANT_INTERVAL = "instant_valid_time; ensemble mean not time average"


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def from_table(table, spec):
    """Select one explicit WN3 role/statistic/interval; unavailable never becomes zero.

    ``spec`` declares role, per-native-variable contracts, and original issue
    timestamps by case_id. Aligned truth/model CSVs must carry the same case_id.
    Unselected roles/statistics are not aliases or substitutes for this selection.
    """
    if not isinstance(spec, dict) or set(spec) - {"path", "role", "contracts", "issues"}:
        raise ValueError("WN3 reference: unknown contract fields")
    if not REQUIRED <= set(table) or table.empty:
        raise ValueError("WN3 reference: incomplete Parquet schema")
    role, contracts, issues = spec.get("role"), spec.get("contracts"), spec.get("issues")
    if role not in ("wn3_raw", "wn3_calibrated") or not isinstance(contracts, dict) or not contracts:
        raise ValueError("WN3 reference: declare role and per-variable native contracts")
    if not isinstance(issues, dict) or not issues:
        raise ValueError("WN3 reference: declare original issue timestamps by case_id")
    if not set(contracts) <= set(VARIABLES):
        raise ValueError("WN3 reference: unsupported native variable")
    for variable, contract in contracts.items():
        if not isinstance(contract, dict) or set(contract) != {"native_statistic", "native_interval", "operator_sha256", "calibrator_sha256"}:
            raise ValueError(f"WN3 {variable}: declare statistic, interval, operator and calibrator hashes")
        if any(not isinstance(contract[k], str) or not contract[k].strip() for k in ("native_statistic", "native_interval")):
            raise ValueError("WN3 reference: native statistic/interval must be explicit")
        if not _sha(contract["operator_sha256"]):
            raise ValueError("WN3 reference: invalid operator_sha256")
        if (role == "wn3_calibrated" and not _sha(contract["calibrator_sha256"])) or (
                role == "wn3_raw" and contract["calibrator_sha256"] is not None):
            raise ValueError("WN3 reference: calibrated role needs its bound calibrator; raw cannot alias calibrated")
    table = table.copy()
    table["valid_time"] = utc(table.valid_time)
    for key in KEYS:
        if table[key].isna().any() or (table[key].astype(str).str.strip() == "").any():
            raise ValueError(f"WN3 reference: missing {key}")
        if key != "valid_time" and not table[key].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"WN3 reference: {key} must contain strings")
    if table.duplicated(KEYS).any():
        raise ValueError("WN3 reference: duplicate canonical key")
    if not table.reference_role.isin(["wn3_raw", "wn3_calibrated"]).all():
        raise ValueError("WN3 reference: unknown reference role")
    parts = []
    for variable, contract in sorted(contracts.items()):
        part = table.loc[table.reference_role.eq(role) & table.variable.eq(variable) &
                         table.native_statistic.eq(contract["native_statistic"]) &
                         table.native_interval.eq(contract["native_interval"])].copy()
        if part.empty:
            raise ValueError(f"WN3 {variable}: UNAVAILABLE selected role/statistic/interval")
        if not part.available.map(lambda x: isinstance(x, (bool, np.bool_))).all() or not part.available.all():
            raise ValueError(f"WN3 {variable}: UNAVAILABLE reference prevents a win")
        if not part.unavailable_reason.fillna("").eq("").all():
            raise ValueError("WN3 reference: available row has an unavailable_reason")
        finite(part.value, f"WN3 {variable} available value")
        if not part.units.eq(VARIABLES[variable][1]).all():
            raise ValueError(f"WN3 {variable}: wrong units; conversion belongs to the bound upstream operator")
        if not part.operator_sha256.eq(contract["operator_sha256"]).all():
            raise ValueError("WN3 reference: operator SHA mismatch")
        if role == "wn3_raw":
            if part.calibrator_sha256.notna().any():
                raise ValueError("WN3 raw reference carries an unexpected calibrator")
        elif not part.calibrator_sha256.eq(contract["calibrator_sha256"]).all():
            raise ValueError("WN3 reference: calibrator SHA mismatch")
        if set(part.case_id) - set(issues):
            raise ValueError("WN3 reference: missing original issue timestamp")
        issue_times = pd.Series(utc([issues[c] for c in part.case_id]), index=part.index)
        lead = finite(part.native_lead_hour, "WN3 native lead")
        if not part.native_lead_hour.map(lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))).all() or (
                lead < 0).any() or not np.equal(lead, np.floor(lead)).all() or not np.array_equal(
                (part.valid_time - issue_times).dt.total_seconds().to_numpy(), lead * 3600):
            raise ValueError("WN3 reference: native lead does not match original issue and exact valid_time")
        # The canonical scorer only accepts instantaneous co-located values.
        if contract["native_interval"] != INSTANT_INTERVAL:
            raise ValueError("WN3 reference: interval/mean products need an interval-aware scorer; no instantaneous alias")
        part["variable"] = VARIABLES[variable][0]
        parts.append(part.rename(columns={"location_id": "station_uid"})[["case_id", "station_uid", "valid_time", "variable", "value"]])
    return pd.concat(parts, ignore_index=True)


def read(path, spec):
    try:
        table = pd.read_parquet(path)
    except ImportError as exc:
        raise ValueError("WN3 Parquet needs pyarrow; install gpuwrf[optimize]") from exc
    return from_table(table, spec)
