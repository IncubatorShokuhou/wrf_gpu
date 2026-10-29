"""Host-only admission and expansion of cloud-study runtime operands.

The catalog follows the public binders, including their refusals. Bounds here
are input admission, not clamps on model fields or evidence of regional skill.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np


def catalog():
    from gpuwrf.runtime_parameters import MYNN
    from gpuwrf.physics.thompson_parameters import PARAMETERS, COEFFICIENT_BOUNDS, EXPERIMENTAL_PARAMETERS

    out = {}
    for family, entries in (("mynn_parameters", MYNN), ("thompson_parameters", PARAMETERS)):
        for name, (default, unit, source) in entries.items():
            bounds = [0., float(np.finfo(np.float32).max)]
            admitted = True
            if family == "thompson_parameters":
                admitted = name not in EXPERIMENTAL_PARAMETERS
                bounds = list(COEFFICIENT_BOUNDS.get(name, (0., float(np.finfo(np.float32).max))))
                if name == "nt_c":
                    bounds = [2., 1999.e6]
            out[f"{family}.{name}"] = {
                "family": family, "name": name, "default": default, "units": unit,
                "provenance": source, "value_bounds": bounds, "admitted": admitted,
                "qualification": "EXPERIMENTAL" if family == "mynn_parameters" else
                                 ("runtime admitted" if admitted else "refused pending GPU/E41/cost qualification"),
                "program": "same names/shapes/dtypes: values reuse program; enabling/changing names needs a key",
            }
    return out


def is_runtime(control):
    return isinstance(control, str) and "." in control


def parameter(item, run_domains):
    """Normalize a runtime scalar parameter into the existing log-multiplier EKI schema."""
    control = item["control"]
    row = catalog().get(control)
    if row is None or not row["admitted"]:
        raise ValueError(f"runtime control {control!r} is unavailable; see cloud control catalog")
    if item.get("classes", "all") != "all":
        raise ValueError("runtime cloud coefficients are WRF module constants; classes must be 'all'")
    domains = item.get("domains", list(run_domains))
    if not isinstance(domains, list) or sorted(domains) != sorted(run_domains):
        raise ValueError("runtime cloud coefficients must set the same value on every run domain")
    low, high = row["value_bounds"]
    feasible = [low / row["default"], high / row["default"]]
    bounds = item.get("bounds", [max(.5, feasible[0]), min(2., feasible[1])])
    if (not isinstance(bounds, list) or len(bounds) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds) or
            not 0 < bounds[0] <= 1 <= bounds[1] or not bounds[0] < bounds[1] or
            bounds[0] < feasible[0] or bounds[1] > feasible[1]):
        raise ValueError(f"{control}: bounds must be positive multipliers containing 1 within {feasible}")
    return {"name": item["name"], "control": control, "classes": "all", "domains": sorted(domains),
            "bounds": list(map(float, bounds)), "bounds_source": "study" if "bounds" in item else "default",
            "feasible_factors": feasible, "runtime": row}


def validate_activity(parameters, config, run_domains):
    from gpuwrf.io.namelist_binding import namelist_value

    for p in parameters:
        if "runtime" not in p:
            continue
        for domain in run_domains:
            row = p["runtime"]
            key, expected = ("mp_physics", 8) if row["family"] == "thompson_parameters" else ("bl_pbl_physics", 5)
            if int(namelist_value(config, key, domain)) != expected:
                raise ValueError(f"{p['control']} requires active {key}={expected} on {domain}")
            if row["name"] == "entrainment":
                value = config.get("physics", {}).get("bl_mynn_edmf", 1)
                values = value if isinstance(value, list) else [value] * len(run_domains)
                if values[int(domain[1:])-1] != 1:
                    raise ValueError("entrainment needs active MYNN EDMF")


def expand(parameters, theta):
    """Small JSON file, one scalar per selected name/domain; no arrays allocated on the device."""
    from gpuwrf.runtime_parameters import SCHEMA, check_wrf_domain_scope

    domains = {}
    for p, log_factor in zip(parameters, theta, strict=True):
        if "runtime" not in p:
            continue
        factor = math.exp(float(log_factor))
        if not p["bounds"][0] <= factor <= p["bounds"][1]:
            # exp(log(endpoint)) can differ by one float64 ULP; permit only that rounding.
            if factor < np.nextafter(p["bounds"][0], -math.inf) or factor > np.nextafter(p["bounds"][1], math.inf):
                raise ValueError(f"{p['name']}: multiplier outside admitted bounds")
        row = p["runtime"]
        value = np.float32(row["default"] * factor)
        if not np.isfinite(value) or value < np.float32(row["value_bounds"][0]) or value > np.float32(row["value_bounds"][1]):
            raise ValueError(f"{p['name']}: expanded scalar outside REAL admission")
        for domain in p["domains"]:
            domains.setdefault(domain, {}).setdefault(row["family"], {})[row["name"]] = float(value)
    check_wrf_domain_scope(domains)
    return {"schema": SCHEMA, "domains": domains}


def cloud_arm(raw, config, max_dom):
    """One explicitly named static arm per study; repeat the study for other arms.

    Existing MYNN on/off branches are reused; missing PDF variants remain
    refused. cal_cldfra3 is the WRF icloud=3 selection, not a second flag.
    """
    from gpuwrf.io.namelist_binding import icloud_refusals, namelist_value

    if not isinstance(raw, dict) or set(raw) - {"icloud", "icloud_bl", "bl_mynn_cloudpdf", "bl_mynn_edmf"}:
        raise ValueError("cloud_arm takes icloud, icloud_bl, bl_mynn_cloudpdf, bl_mynn_edmf")
    allowed = {"icloud": (0, 1, 2, 3), "icloud_bl": (0, 1), "bl_mynn_cloudpdf": (2,), "bl_mynn_edmf": (0, 1)}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed[key]:
            raise ValueError(f"{key}={value!r} unavailable in cloud studies; admitted {allowed[key]}; v45-mynnopts owns missing branches")
    cfg = copy.deepcopy(config)
    # Main now binds cloudpdf explicitly; an admitted CASE(2) arm overrides a
    # case's inherited selection rather than merely recording metadata.
    cfg.setdefault("physics", {}).update({k: v if k == "bl_mynn_cloudpdf" else [v]*max_dom
                                         for k, v in raw.items()})
    from gpuwrf.io.mynn_budget_options import validate_options
    for i in range(1, max_dom+1):
        validate_options(cfg, domain=i-1)
        domain = f"d{i:02d}"
        if set(raw) & {"icloud_bl", "bl_mynn_cloudpdf", "bl_mynn_edmf"} and int(namelist_value(cfg, "bl_pbl_physics", domain)) != 5:
            raise ValueError("MYNN cloud arms require active bl_pbl_physics=5")
        cloud = namelist_value(cfg, "icloud", domain)
        if "icloud" in raw and raw["icloud"] != 1 and int(namelist_value(cfg, "cu_physics", domain)) == 11:
            raise ValueError("cu_physics=11 forces icloud=1 in WRF; the requested cloud arm would be ignored")
        errors = icloud_refusals(1 if cloud is None else cloud, **{
            key: namelist_value(cfg, key, domain) for key in
            ("mp_physics", "cu_physics", "ra_lw_physics", "ra_sw_physics")})
        if errors:
            raise ValueError("; ".join(errors))
    token = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]
    return cfg, {"selections": raw, "key": f"cloud-{token}", "compile_cost": "separate static arm; first use may compile"}
