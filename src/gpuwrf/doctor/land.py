"""Shared land-category authority for doctor and the pre-JAX run plan."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def category_issues(ds, count, urban, confirmed_lcz=False):
    """WRF integer category bounds, plus unsupported LCZ without urban refusal.

    Custom category tables may use 21/24/40/61 entries; the count alone is not
    a physics qualification. WRF 4.7.1 LCZ indices 51..61 require urban physics;
    legacy 31..41/custom tables remain ambiguous unless explicitly LCZ-labelled.
    """
    issues = []
    confirmed_lcz = confirmed_lcz or "LCZ" in str(getattr(ds, "MMINLU", "")).upper()
    if not np.isfinite(count) or count != int(count) or count < 1:
        return [("LAND_CATEGORIES", "NUM_LAND_CAT must be a positive integer", "ERROR")]
    for key in ("LU_INDEX", "IVGTYP"):
        if key not in ds.variables:
            issues.append(("CATEGORY_RANGE", f"missing {key}", "ERROR"))
            continue
        raw = np.ma.asarray(ds.variables[key][:], dtype=float)
        data = np.asarray(raw.filled(np.nan))
        if not np.isfinite(data).all() or ((data < 1) | (data > count) | (data != np.floor(data))).any():
            issues.append(("CATEGORY_RANGE", f"{key} must be finite integer in 1..{int(count)}", "ERROR"))
        canonical = ((data >= 51) & (data <= 61)).any()
        legacy = ((data >= 31) & (data <= 41)).any()
        if (canonical or legacy) and int(urban) == 0:
            classes = "51..61" if canonical else "31..41 (legacy numbering)"
            issues.append(("LCZ_WITHOUT_URBAN", f"{key} contains categories {classes} without urban physics (sf_urban_physics=0); supply a supported urban scheme and consistent land-use tables; indices alone can also be non-LCZ custom categories", "ERROR" if confirmed_lcz else "WARNING"))
        if confirmed_lcz and legacy and not canonical and int(urban) != 0:
            issues.append(("LCZ_LEGACY_NUMBERING",f"{key}: WRF 4.7.1 uses LCZ 51..61; regenerate legacy 31..41 inputs with real.exe", "ERROR"))
        if canonical and int(urban) in (2,3):
            issues.append(("LCZ_MULTILAYER_UNSUPPORTED",f"{key}: categories 51..61 with sf_urban_physics={urban} require per-type grouped BEP/BEM geometry, carries and PBL sources; this LCZ port supports sf_urban_physics=1 only", "ERROR"))
    return issues


def preflight(case, namelist, domains):
    """Check available NetCDF ICs; retain the CLI's structural dry-run semantics.

    Dry-run historically permits missing/non-NetCDF placeholders. They remain
    outside this category-only check and are explicitly reported as unchecked;
    ``doctor`` performs full file/field validation.
    """
    from .common import parse_namelist, value
    from gpuwrf.io.netcdf_lock import Dataset
    nl = parse_namelist(namelist)
    checked, unchecked, warnings = [], [], []
    for domain in domains:
        path = Path(case) / f"wrfinput_{domain}"
        if not path.is_file():
            unchecked.append(str(path))
            continue
        with path.open("rb") as stream:
            magic = stream.read(8)
        if not (magic[:3] == b"CDF" or magic == b"\x89HDF\r\n\x1a\n"):
            unchecked.append(str(path))
            continue
        with Dataset(path) as ds:
            count = float(getattr(ds, "NUM_LAND_CAT", value(nl, "num_land_cat", default=24)))
            declared = value(nl, "num_land_cat")
            if declared is not None and float(declared) != count:
                raise ValueError(f"{path.name}: NUM_LAND_CAT={count} differs from num_land_cat={declared}")
            issues = category_issues(ds, count, value(nl, "sf_urban_physics", int(domain[1:])-1, 0), bool(value(nl, "use_wudapt_lcz", default=False)))
            warnings.extend(message for code,message,severity in issues if severity == "WARNING")
            errors = [(code,message) for code,message,severity in issues if severity == "ERROR"]
            if errors:
                raise ValueError(f"{path.name}: " + "; ".join(f"{code}: {message}" for code,message in errors))
        checked.append(str(path))
    return {"checked": checked, "unchecked": unchecked, "warnings": warnings, "scope": "land categories only; use gpuwrf doctor for complete input validation"}
