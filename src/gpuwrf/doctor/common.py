"""Host-only metadata shared by doctor and reproducibility commands."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import math

from gpuwrf.io.namelist_parser import parse_namelist


def value(nl, key, domain=0, default=None):
    for group in nl.values():
        if key in group:
            raw = group[key]
            if isinstance(raw, (list, tuple)):
                raw = raw[domain] if domain < len(raw) else None
            return default if raw is None else raw
    return default


def finite_number(raw, label, *, positive=False):
    number = float(raw)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def time_label(raw):
    return datetime.strptime(raw.strip(), "%Y-%m-%d_%H:%M:%S")


def char_times(ds, name="Times"):
    if name not in ds.variables:
        raise ValueError(f"missing {name}")
    import numpy as np
    var = ds.variables[name]
    if len(var.shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional character array")
    labels = [b"".join(np.asarray(row).tolist()).decode("ascii").strip() for row in var[:]]
    result = [time_label(s) for s in labels]
    if not result or any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} must have nonempty, strictly increasing times")
    return result


def run_window(nl):
    def stamp(prefix):
        parts = [value(nl, f"{prefix}_{key}", default=0 if key in {"hour", "minute", "second"} else None)
                 for key in ("year", "month", "day", "hour", "minute", "second")]
        return datetime(*map(int, parts)) if all(p is not None for p in parts) else None
    start = stamp("start")
    if start is None:
        raise ValueError("namelist requires start_year/month/day")
    duration = sum(finite_number(value(nl, f"run_{key}", default=0), f"run_{key}") * scale
                   for key, scale in (("days", 86400), ("hours", 3600), ("minutes", 60), ("seconds", 1)))
    end = start + timedelta(seconds=duration) if duration > 0 else stamp("end")
    if end is None or end <= start:
        raise ValueError("namelist requires a positive run duration or end time")
    return start, end


def support_report(nl):
    """Catalog classifications plus the actual CLI validator/honour checks."""
    from gpuwrf.io.scheme_catalog import classify_scheme, classify_control, classify_feature_switch
    from gpuwrf.io.wrf_scheme_catalog import WRF_SCHEME_CATALOG
    from gpuwrf.io.namelist_check import validate_operational_namelist, collect_namelist_warnings
    from gpuwrf.io.namelist_binding import namelist_honour_report
    options = []
    for group, settings in nl.items():
        for key, raw in settings.items():
            for i, item in enumerate(raw if isinstance(raw, list) else [raw]):
                if item is None:
                    continue
                entry = (classify_scheme(key, int(item)) if key in WRF_SCHEME_CATALOG
                         else classify_control(key, item) or classify_feature_switch(key, item))
                if entry is not None:
                    options.append(dict(group=group, index=i, requested=item, **asdict(entry)))
    errors = []
    try:
        validate_operational_namelist(nl)
    except ValueError as exc:
        errors.append(str(exc))
    domains = tuple(f"d{i+1:02d}" for i in range(int(value(nl, "max_dom", default=1))))
    binding_errors, warnings = namelist_honour_report(nl, domains, replay=False)
    return {"options": options, "errors": errors + binding_errors,
            "warnings": list(collect_namelist_warnings(nl)) + warnings}


def case_inputs(case, namelist):
    """Inputs the commands bind; history and unrelated files are excluded."""
    case = Path(case)
    paths = {Path(namelist).resolve()}
    for pattern in ("wrfinput_d[0-9][0-9]", "wrfbdy_d[0-9][0-9]", "wrflowinp_d[0-9][0-9]",
                    "*PARM.TBL", "MPTABLE.TBL", "LANDUSE.TBL", "URBPARM*.TBL",
                    "windturbines*.txt", "wind-turbine-*.tbl"):
        paths.update(p.absolute() for p in case.glob(pattern) if p.is_file())
    return sorted(paths, key=str)
