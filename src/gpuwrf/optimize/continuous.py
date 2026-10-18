"""Continuous parameters for ``gpuwrf optimize`` (schema ``gpuwrf.optimize.continuous.v1``, EXPERIMENTAL, opt-in).

Ensemble Kalman inversion (``surface_params.eki``: Tikhonov EKI, host fp64, tempered dt = 1/iterations) over the
Track P traced Noah-MP surface controls and the host ``ctopo`` control, inside ONE released/admitted physics
configuration. Every member is an ordinary ensemble-launcher case whose input directory holds its own control file
(``surface_controls.npz``, passed as ``--surface-controls``): the controls ride traced inputs (``ctopo`` replaces the
value of an already traced namelist child), so every member, iteration, the twin truth and the held-out prior and
posterior runs share one compiled program per record signature -- the first member compiles, the rest load it. The
launcher batches, admission math, chunk pause/``--resume``, salvage and SHA-bound inputs are ``study.run``'s.

Truth enters only through a scorer:
  stations  local station metadata + canonical observation CSV (``surface_params.stations.collect``).
  twin      OSSE: truth members run with known multipliers; synthetic observations = the truth histories sampled at
            the study stations + seeded Gaussian noise (sigma per variable). EKI sees only these observations; the
            registered ``tolerance_log`` decides RECOVERED / NOT_RECOVERED.
ALISIOS paths are refused (stage a copy once with ``alisios_guard.facade_stage``); real-station studies are a
pilot-phase item, not run here. Results are diagnostics: no release promotion, no regional-skill claim.

Iteration k (0 .. iterations-1) runs ``members`` forecasts per training case; theta (log multipliers) is updated
from the stacked station predictions; afterwards the posterior-mean and the prior (multiplier 1) parameter sets run
on every training and held-out case. Design notes: lane w41-optcont (.agent/kernel/lanes/w41-optcont.notes.md).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import identity

SCHEMA = "gpuwrf.optimize.continuous.v1"
RESULT_SCHEMA = "gpuwrf.optimize.continuous.result.v1"
VARIABLES = ("t2", "wind")
# Accumulated variables need interval-aware collection (interval_start in the twin CSV, the collector keys and the
# train/test split); the reused Track P collector keys omit it, so continuous studies refuse them (critic 00:10Z).
INTERVAL_VARIABLES = ("precip_rate",)
# Static wrfinput fields the control basis depends on (vegetation/soil classes, SHDMAX, start_em CTOPO inputs).
STATIC_FIELDS = ("IVGTYP", "ISLTYP", "SHDMAX", "HGT", "XLAND", "LANDMASK", "VAR_SSO", "VAR")
TABLE_CONTROLS = ("z0mvt", "eg", "vcmx25", "mp", "albsat", "albdry", "rhol", "rhos", "laim", "saim")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def read_admitted(path):
    """(resolved path, text) of a study file, read only after the ALISIOS preflight admitted its real path (a link
    into a protected root is refused before one byte is read; critic cr-w41-optcont 00:24Z)."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    refuse_protected_inputs("gpuwrf optimize", [path])
    path = Path(path).resolve(strict=True)
    return path, path.read_text()


def is_continuous(text) -> bool:
    """Schema dispatch on the ADMITTED study text (``read_admitted``), never on a fresh read of the path."""
    try:
        return json.loads(text).get("schema") == SCHEMA
    except (ValueError, AttributeError):
        return False


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def static_signature(input_dir, domains):
    """sha256 per domain over the static wrfinput fields the control basis depends on (absent fields named)."""
    import hashlib

    from gpuwrf.io.netcdf_lock import Dataset

    out = {}
    for domain in domains:
        digest = hashlib.sha256()
        with Dataset(Path(input_dir)/f"wrfinput_{domain}") as ds:
            names = list(STATIC_FIELDS) + ([] if "SHDMAX" in ds.variables else ["VEGFRA"])
            for name in names:
                if name in ds.variables:
                    data = np.ascontiguousarray(np.asarray(ds.variables[name][:]))
                    digest.update(f"{name}:{data.dtype.str}:{data.shape}".encode()+data.tobytes())
                else:
                    digest.update(f"{name}:absent".encode())
        out[domain] = digest.hexdigest()
    return out


def basis_current(basis, manifest, used):
    """Refuse a basis whose table-valued controls no longer equal the runtime Noah-MP tables (stale MPTABLE)."""
    import types

    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    from gpuwrf.surface_params import controls as sc

    parameters = None
    for domain, control in sorted(used):
        if control not in TABLE_CONTROLS:
            continue
        if parameters is None:
            parameters = load_noahmp_parameters(wrf_run_dir())
        vt = np.asarray(basis[f"{domain}/_ivgtyp"])
        nl = types.SimpleNamespace(noahmp_static=types.SimpleNamespace(ivgtyp=vt, parameters=parameters),
                                   noahmp_energy_params=types.SimpleNamespace(z0mvt=np.zeros(vt.shape)))
        # Compare in the consumer dtype the basis records: the release path holds the tables in REAL, so the basis
        # values are the REAL-rounded table entries (exactly what the members read), the fresh load is float64.
        dtype = np.dtype(manifest["domains"][domain]["controls"][control]["dtype"])
        if not np.array_equal(sc.table_base(control, nl).astype(dtype), np.asarray(basis[f"{domain}/{control}"]).astype(dtype)):
            raise ValueError(f"basis {domain}/{control} differs from the runtime Noah-MP tables; rebuild the basis "
                             "(gpuwrf surface-params basis) with the tables the members run")


def _parameters(raw, manifest):
    from gpuwrf.surface_params.controls import ALL_CONTROLS, HOST_CONTROLS

    if not isinstance(raw, list) or not raw:
        raise ValueError("parameters must be a nonempty list")
    out, used = [], set()
    for item in raw:
        if not isinstance(item, dict) or not {"name", "control"} <= set(item):
            raise ValueError("every parameter needs name and control")
        extra = set(item) - {"name", "control", "classes", "domains", "prior_sd_log", "bounds"}
        if extra:
            raise ValueError(f"unknown parameter keys {sorted(extra)}")
        name, control = item["name"], item["control"]
        if not isinstance(name, str) or not _ID.fullmatch(name):
            raise ValueError("parameter names must be safe identifiers")
        if control not in ALL_CONTROLS:
            raise ValueError(f"unknown control {control!r}; known: {sorted(ALL_CONTROLS)}")
        classes = item.get("classes", "all")
        if classes != "all":
            if (control in HOST_CONTROLS or not isinstance(classes, list) or not classes
                    or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in classes)):
                raise ValueError(f"{name}: classes must be 'all' or vegetation categories (never for {sorted(HOST_CONTROLS)})")
            classes = sorted(set(classes))
        domains = item.get("domains", ["d01"])
        if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
            raise ValueError(f"{name}: domains must be a nonempty list without repeats")
        for domain in domains:
            info = manifest["domains"].get(domain, {}).get("controls", {}).get(control)
            if info is None or not info.get("active"):
                raise ValueError(f"{name}: {domain}/{control} is not an active control of the basis "
                                 f"({(info or {}).get('reason', 'domain/control absent')})")
            used.add((domain, control))
        sd = _finite_positive(item.get("prior_sd_log", 0.5), f"{name} prior_sd_log")
        given = "bounds" in item
        bounds = item.get("bounds", list(DEFAULT_BOUNDS))
        # The table/start_em value (factor 1) must stay inside; an endpoint may BE 1 (e.g. ctopo on valley cells).
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) for b in bounds)
                or not 0 < bounds[0] <= 1 <= bounds[1] or not bounds[0] < bounds[1]):
            raise ValueError(f"{name}: bounds must be positive multipliers [low <= 1 <= high] with low < high")
        out.append({"name": name, "control": control, "classes": classes, "domains": domains,
                    "prior_sd_log": sd, "bounds": [float(b) for b in bounds],
                    "bounds_source": "study" if given else "default"})
    if len({p["name"] for p in out}) != len(out):
        raise ValueError("duplicate parameter name")
    for i, a in enumerate(out):  # one class may be scaled by two parameters only by explicit, disjoint classes
        for b in out[i+1:]:
            if a["control"] == b["control"] and set(a["domains"]) & set(b["domains"]) and (
                    a["classes"] == "all" or b["classes"] == "all" or set(a["classes"]) & set(b["classes"])):
                raise ValueError(f"parameters {a['name']} and {b['name']} scale the same cells of {a['control']}")
    return out, used


DEFAULT_BOUNDS = (0.2, 5.0)
_MARGIN = 1e-6  # relative shrink of a DEFAULT endpoint only if its REAL-rounded expansion leaves the physical bounds


def feasible_factors(basis, manifest, parameter):
    """(lowest, highest) factor whose expansion is physically valid on EVERY cell the parameter scales.

    Traced controls: spec.lower <= base x f <= spec.upper (and rho x f + tau <= 1 for leaf/stem reflectance) on cells
    with base > 0; host ctopo: 0 <= 1 + f (CTOPO - 1) <= upper (CTOPO < 1 valley cells cap f at 1/(1 - CTOPO), so a
    CTOPO = 0 ramp cell caps it at 1). The EKI clips theta to the bounds, so bounds inside this range mean no member
    can be refused by the control validation."""
    from gpuwrf.surface_params.controls import ALL_CONTROLS, HOST_CONTROLS, _TRANSMITTANCE

    spec = ALL_CONTROLS[parameter["control"]]
    lowest, highest = 0.0, math.inf
    for domain in parameter["domains"]:
        base = np.asarray(basis[f"{domain}/{parameter['control']}"], dtype=np.float64)
        if parameter["control"] in HOST_CONTROLS:
            below, above = base[base < 1.0], base[base > 1.0]
            if below.size:
                highest = min(highest, float(np.min(1.0/(1.0-below))))
            if above.size:
                highest = min(highest, float(np.min((spec.upper-1.0)/(above-1.0))))
            continue
        mask = np.ones(base.shape[-2:], bool) if parameter["classes"] == "all" else \
            np.isin(np.asarray(basis[f"{domain}/_ivgtyp"]), parameter["classes"])
        cells = base[..., mask]
        positive = cells[cells > 0]
        if positive.size:
            highest = min(highest, float(np.min(spec.upper/positive)))
            if spec.lower > 0:
                lowest = max(lowest, float(np.max(spec.lower/positive)))
        if parameter["control"] in _TRANSMITTANCE:
            tau = np.asarray(basis[f"{domain}/_{_TRANSMITTANCE[parameter['control']]}"], dtype=np.float64)
            tau = np.broadcast_to(tau, base.shape)[..., mask]  # the layout expand_from_basis checks against
            ok = cells > 0
            if np.any(ok):
                highest = min(highest, float(np.min((1.0-tau[ok])/cells[ok])))
    return lowest, highest


def _feasible_bounds(basis, manifest, parameters):
    """Fail closed on prior support that reaches physically invalid control values (the whole range is checked at
    load, never discovered when a member is expanded mid-study); default bounds are intersected with it."""
    from gpuwrf.surface_params.cli import expand_from_basis, member_multipliers

    def valid(parameter, factor):  # the exact member expansion (REAL rounding, transmittance) at one factor
        try:
            expand_from_basis(basis, manifest, member_multipliers({"parameters": [parameter]}, [math.log(factor)]))
            return True
        except ValueError:
            return False

    for parameter in parameters:
        lowest, highest = feasible_factors(basis, manifest, parameter)
        # REAL rounding can push the analytic edge itself out of bounds (1.1 m x 5/1.1 -> 5.0000001 m): step inside
        if math.isfinite(highest) and not valid(parameter, highest):
            highest *= 1.0-_MARGIN
        if lowest > 0 and not valid(parameter, lowest):
            lowest *= 1.0+_MARGIN
        parameter["feasible_factors"] = [lowest, None if math.isinf(highest) else highest]
        low, high = parameter["bounds"]
        refused = (f"{parameter['name']}: bounds [{low:g}, {high:g}] reach physically invalid {parameter['control']} "
                   f"values; feasible factors are [{lowest:.6g}, {highest:.6g}] (e.g. ctopo on valley cells with "
                   "CTOPO < 1 needs high <= 1/(1 - CTOPO))")
        if parameter["bounds_source"] == "default":
            low, high = max(low, lowest), min(high, highest)
            if not low <= 1.0 <= high or not low < high:
                raise ValueError(f"{parameter['name']}: no default prior range around the table value is physically "
                                 f"feasible (feasible factors [{lowest:.6g}, {highest:.6g}])")
            parameter["bounds"] = [low, high]
        elif low < lowest or high > highest:
            raise ValueError(refused)
        # values are linear in the factor, so valid endpoints mean every clipped member is valid
        if not all(valid(parameter, factor) for factor in parameter["bounds"]):
            raise ValueError(refused)


def _scorer(raw, parameters, resolve):
    if not isinstance(raw, dict) or raw.get("kind") not in ("stations", "twin"):
        raise ValueError("scorer must be {kind: stations|twin, ...}")
    variables = raw.get("variables")
    if isinstance(variables, list) and any(v in INTERVAL_VARIABLES for v in variables):
        raise ValueError(f"scorer variables {INTERVAL_VARIABLES} are not supported by continuous studies yet: accumulated "
                         "observations need interval-aware collection (interval_start); use t2/wind")
    if not isinstance(variables, list) or not variables or len(set(variables)) != len(variables) or \
            any(v not in VARIABLES for v in variables):
        raise ValueError(f"scorer variables must be a nonempty subset of {VARIABLES}")
    sigma = raw.get("sigma", {})
    if not isinstance(sigma, dict) or set(sigma) != set(variables):
        raise ValueError("scorer sigma must give one observation error per variable (representation + instrument)")
    out = {"kind": raw["kind"], "variables": variables,
           "sigma": {v: _finite_positive(sigma[v], f"sigma {v}") for v in variables}}
    if raw["kind"] == "stations":
        if set(raw) - {"kind", "variables", "sigma", "observations"}:
            raise ValueError("stations scorer takes kind, variables, sigma, observations")
        out["observations"] = str(resolve(raw["observations"]))
    else:
        if set(raw) - {"kind", "variables", "sigma", "truth", "noise_seed", "tolerance_log"}:
            raise ValueError("twin scorer takes kind, variables, sigma, truth, noise_seed, tolerance_log")
        truth = raw.get("truth")
        names = {p["name"] for p in parameters}
        if not isinstance(truth, dict) or set(truth) != names:
            raise ValueError("twin truth must give one multiplier per parameter")
        for p in parameters:
            value = _finite_positive(truth[p["name"]], f"truth {p['name']}")
            if not p["bounds"][0] <= value <= p["bounds"][1]:
                raise ValueError(f"twin truth {p['name']} outside its bounds")
        seed = raw.get("noise_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("twin noise_seed must be a nonnegative integer")
        out.update(truth={k: float(v) for k, v in truth.items()}, noise_seed=seed,
                   tolerance_log=_finite_positive(raw.get("tolerance_log"), "twin tolerance_log"))
    return out


def load_continuous(path, *, text=None):
    """Validate a continuous study and bind every input by content (fail closed); ``text`` = the admitted bytes the
    CLI dispatched on (``read_admitted``), else the file is read after the refusal."""
    from gpuwrf import __version__
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from gpuwrf.ens.workflows import _case
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.io.namelist_check import (refuse_single_domain_mp_nested, refuse_unqualified_nested_options,
                                          validate_operational_namelist)
    from gpuwrf.surface_params.cli import load_basis

    from . import released
    from .study import _candidates, forecast_inputs, model_digest, record_signature, _geometry

    path = Path(path).resolve(strict=True)
    refuse_protected_inputs("gpuwrf optimize continuous", [path])
    raw = json.loads(path.read_text() if text is None else text)
    if raw.get("schema") != SCHEMA or not str(raw.get("region", "")).strip():
        raise ValueError(f"study requires schema {SCHEMA} and a region")
    allowed = {"schema", "region", "hours", "budget_case_hours", "test_start", "cases", "configuration", "admission",
               "e41", "basis", "parameters", "eki", "scorer", "stations", "sampling"}
    if set(raw) - allowed:
        raise ValueError(f"unknown study keys {sorted(set(raw) - allowed)}")
    resolve = lambda p: (path.parent/str(p)).resolve(strict=True)  # noqa: E731
    hours = _positive_int(raw.get("hours"), "hours")
    budget = _positive_int(raw.get("budget_case_hours"), "budget_case_hours")
    e41 = raw.get("e41", "any")
    if e41 not in released.E41_LEVELS:
        raise ValueError(f"e41 must be one of {released.E41_LEVELS}")
    split = pd.Timestamp(raw["test_start"], tz="UTC")
    if split != split.normalize():
        raise ValueError("test_start must be a UTC date")
    case_dirs = [resolve(item["input_dir"]) for item in raw.get("cases", [])]
    stations, basis_path = resolve(raw["stations"]), resolve(raw["basis"])
    scorer_raw = raw.get("scorer") or {}
    observations = [resolve(scorer_raw["observations"])] if scorer_raw.get("kind") == "stations" else []
    # Fail closed BEFORE any payload read: ALISIOS data roots (dir aliases, namelist-named files, runtime roots).
    refuse_protected_inputs("gpuwrf optimize continuous", [*case_dirs, stations, basis_path, *observations],
                            runtime=True)
    cases = []
    for item, directory in zip(raw["cases"], case_dirs):
        if not isinstance(item.get("id"), str) or not _ID.fullmatch(item["id"]):
            raise ValueError("case IDs must be safe unique names")
        date = pd.Timestamp(item["date"], tz="UTC")
        if date != date.normalize():
            raise ValueError("case date must be YYYY-MM-DD")
        native = _case(directory, directory/"namelist.input", hours)
        if native["issue"].strftime("%Y-%m-%d") != date.strftime("%Y-%m-%d"):
            raise ValueError("declared case date disagrees with namelist/wrfinput initialization")
        config = native["config"]
        count = config.get("domains", {}).get("max_dom", 1)
        count = _positive_int(count[0] if isinstance(count, list) else count, "max_dom")
        domain = item.get("domain", f"d{count:02d}")
        if domain not in [f"d{n:02d}" for n in range(1, count+1)]:
            raise ValueError("case scoring domain does not exist")
        cases.append({"id": item["id"], "date": date.strftime("%Y-%m-%d"), "input_dir": str(directory),
                      "domain": domain, "config": config, "max_dom": count, "issue": native["issue"].isoformat(),
                      "source_namelist": identity(directory/"namelist.input"),
                      "forcing": [identity(p) for p in forecast_inputs(directory)],
                      "records": record_signature(directory, count), "geometry": _geometry(directory),
                      "split": "train" if date < split else "test"})
    if not cases or len({c["id"] for c in cases}) != len(cases) or len({c["date"] for c in cases}) != len(cases):
        raise ValueError("need cases with unique IDs and one initialization per UTC date")
    if len({c["input_dir"] for c in cases}) != len(cases):
        raise ValueError("duplicate input case directory")
    if len({c["max_dom"] for c in cases}) != 1 or len({json.dumps(c["geometry"], sort_keys=True) for c in cases}) != 1:
        raise ValueError("study cases must share one domain count and d01 geometry (one program)")
    if any(not any(c["split"] == s for c in cases) for s in ("train", "test")):
        raise ValueError("need at least one training date and one held-out test date")
    max_dom = cases[0]["max_dom"]
    source = model_digest()
    data = released.load()
    if ("configuration" in raw) == ("admission" in raw):
        raise ValueError("give exactly one of configuration (released ID/run name) or admission (receipt)")
    selection = {"configurations": [raw["configuration"]]} if "configuration" in raw else {}
    admissions, evidence = _candidates(selection, data, max_dom, e41, [raw["admission"]] if "admission" in raw else [],
                                       resolve, source)
    (base,) = admissions.values()
    entry = base.pop("entry")
    for case in cases:
        cfg = copy.deepcopy(case["config"])
        cfg.setdefault("physics", {}).update(base["physics"])
        validate_operational_namelist(cfg)
        refuse_single_domain_mp_nested(cfg, max_dom)
        refuse_unqualified_nested_options(cfg, max_dom)
        require_namelist_honoured(cfg, tuple(f"d{i:02d}" for i in range(1, max_dom+1)))
    basis, manifest = load_basis(basis_path)
    parameters, used = _parameters(raw.get("parameters"), manifest)
    basis_current(basis, manifest, used)
    _feasible_bounds(basis, manifest, parameters)
    geometry = cases[0]["geometry"]
    for parameter in parameters:  # one geometry per study: the basis must be of this grid (d01 extents)
        if "d01" in parameter["domains"] and manifest["domains"]["d01"]["grid_shape"] != [geometry["south_north"],
                                                                                         geometry["west_east"]]:
            raise ValueError("the basis d01 grid differs from the study cases' d01 grid")
    if manifest.get("input_dir") is None:
        raise ValueError("the basis does not record its input case (rebuild it with gpuwrf surface-params basis)")
    domains = sorted({d for p in parameters for d in p["domains"]})
    refuse_protected_inputs("gpuwrf optimize continuous basis case", [manifest["input_dir"]])
    reference = static_signature(manifest["input_dir"], domains)
    for case in cases:
        if static_signature(case["input_dir"], domains) != reference:
            raise ValueError(f"case {case['id']} static land/terrain fields differ from the basis case "
                             f"{manifest['input_dir']}; one basis serves only cases of one static geography")
    # The basis records which controls are ACTIVE under the basis case's own namelist (Noah-MP dveg/opt_crs, topo_wind,
    # PBL): every member must run exactly those physics, else a control could be inert or refused in the member.
    from gpuwrf.io.namelist_parser import parse_namelist
    basis_view = _physics_view(parse_namelist(Path(manifest["input_dir"])/"namelist.input"), max_dom)
    for case in cases:
        cfg = copy.deepcopy(case["config"])
        cfg.setdefault("physics", {}).update(base["physics"])
        view = _physics_view(cfg, max_dom)
        if view != basis_view:
            keys = sorted(k for k in set(view) | set(basis_view) if view.get(k) != basis_view.get(k))
            raise ValueError(f"case {case['id']} with the study configuration runs other physics than the basis case "
                             f"({keys[:6]}); build the basis from a case that runs the study configuration")
    eki = raw.get("eki", {})
    if not isinstance(eki, dict) or set(eki) - {"members", "iterations", "seed", "tikhonov"}:
        raise ValueError("eki takes members, iterations, seed, tikhonov")
    members, iterations = _positive_int(eki.get("members"), "eki members"), _positive_int(eki.get("iterations"), "eki iterations")
    if members < 2:
        raise ValueError("eki needs at least 2 members")
    seed = eki.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("eki seed must be a nonnegative integer")
    tikhonov = eki.get("tikhonov", True)
    if not isinstance(tikhonov, bool):
        raise ValueError("eki tikhonov must be true or false")
    scorer = _scorer(scorer_raw, parameters, resolve)
    sampling = raw.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object of station extraction options")
    train = sum(c["split"] == "train" for c in cases)
    test = len(cases)-train
    planned = (members*iterations*train + 2*len(cases) + (len(cases) if scorer["kind"] == "twin" else 0))*hours
    if planned > budget:
        raise ValueError(f"budget needs at least {planned} case-hours (members x iterations x training dates, prior "
                         f"and posterior on every date{', twin truth' if scorer['kind'] == 'twin' else ''}; "
                         f"given {budget})")
    configuration = {"id": base["id"], "physics": base["physics"], "admission": base["admission"],
                     "status": base["status"], "release": {"release": data["release"], "best_E41": entry["best_E41"]}}
    study = {"schema": SCHEMA, "region": raw["region"], "hours": hours, "budget_case_hours": budget,
             "test_start": split.isoformat(), "e41": e41, "cases": sorted(cases, key=lambda c: c["date"]),
             "configuration": configuration, "parameters": parameters,
             "eki": {"members": members, "iterations": iterations, "seed": seed, "tikhonov": tikhonov},
             "scorer": scorer, "stations": str(stations), "sampling": sampling,
             "basis": {**identity(basis_path), "input_dir": manifest["input_dir"], "static_signature": reference},
             "model_sha256": source, "gpuwrf_version": __version__,
             "inputs": [identity(p) for p in (path, stations, basis_path, *observations)] + evidence +
                       [c["source_namelist"] for c in cases] + [identity(released.RELEASED_PATH)],
             "planned_case_hours": planned, "candidates": [],
             "program": {"configuration": base["id"], "record_signatures": sorted(
                 {json.dumps(c["records"], sort_keys=True) for c in cases}),
                 "note": "every member shares the configuration's controls program (values are traced inputs)"},
             "qualification": ("EXPERIMENTAL continuous calibration of traced Track P controls inside one released/"
                               "admitted configuration; diagnostics only, no release promotion or regional skill claim")}
    study["resources"] = _resources(study, members*train)
    return study


def _physics_view(config, max_dom):
    """&physics and &noah_mp per-domain values (first max_dom entries) -- what decides control activity."""
    view = {}
    for group in ("physics", "noah_mp"):
        for key, value in (config.get(group) or {}).items():
            values = value if isinstance(value, list) else [value]
            view[f"{group}.{key}"] = json.dumps(values[:max_dom])
    return view


def _resources(study, widest):
    from .resources import admission, member_pool

    if study["cases"][0]["max_dom"] > 1:
        return {"sizing": "launcher C-auto memory plans (nested)", "widest_batch": widest}
    pool = member_pool(study["cases"][0]["geometry"])
    return {"sizing": "fixed per-member pool (--pool-gib; single-domain has no C-auto plan)", "member": pool,
            "widest_batch": widest, "admission_dev_rtx5090": admission(pool["need_gib"], members=widest)}


# ---------------------------------------------------------------------------------------------------- the driver
def _problem(study, obs=None, sigma=None):
    from gpuwrf.surface_params import eki

    params = study["parameters"]
    p = len(params)
    return eki.EKIProblem(names=tuple(q["name"] for q in params), prior_mean=np.zeros(p),
                          prior_cov=np.diag([q["prior_sd_log"]**2 for q in params]),
                          obs=np.zeros(1) if obs is None else np.asarray(obs, np.float64),
                          obs_cov=np.ones(1) if sigma is None else np.asarray(sigma, np.float64)**2,
                          lower=np.log([q["bounds"][0] for q in params]), upper=np.log([q["bounds"][1] for q in params]),
                          tikhonov=study["eki"]["tikhonov"])


def _write_controls(path, fields, provenance):
    """Write once; an existing file (interrupted run) must hold exactly these fields (content digest per domain)."""
    from gpuwrf.surface_params.controls import controls_digest, load_controls, save_controls

    path = Path(path)
    if path.exists():
        existing, _ = load_controls(path)
        if set(existing) != set(fields) or any(controls_digest(existing[d]) != controls_digest(fields[d]) for d in fields):
            raise ValueError(f"existing member control file {path} holds other values (study changed?)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_controls(path, fields, provenance=provenance)
    return identity(path)


def _keys_after(keys, issue, *, before=None):
    """Indices of forecast keys (station, time, variable): valid after the issue time (and before the split)."""
    issue = pd.Timestamp(issue)
    keep = []
    for n, key in enumerate(keys):
        when = pd.Timestamp(key[1])
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        if when > issue and (before is None or when < before):
            keep.append(n)
    return keep


def program_report(ledger):
    """One executable for all members? Cheap keys from every member's own run proof + launcher warm/cold receipts."""
    keys, unknown, cold, warm = {}, [], [], 0
    for name, entry in sorted(ledger["members"].items()):
        proof = Path(entry["files"][0]["path"]).parent/"proofs"/"nested_pipeline_run.json"
        try:
            domains = (json.loads(proof.read_text()).get("metadata") or {}).get("nested_aot", {}).get("domains") or {}
        except (OSError, ValueError, AttributeError):
            domains = {}
        found = {d: r.get("cheap_key") for d, r in domains.items() if isinstance(r, dict) and r.get("cheap_key")}
        if not found:
            unknown.append(name)
        for domain, key in found.items():
            keys.setdefault(domain, {}).setdefault(key, []).append(name)
        if entry.get("aot_cache_warm") is True and not entry.get("aot_cold_executables"):
            warm += 1
        else:
            cold.append(name)
    one = bool(keys) and all(len(v) == 1 for v in keys.values()) and not unknown
    return {"members": len(ledger["members"]), "cheap_keys": {d: sorted(v) for d, v in keys.items()},
            "members_without_proof": unknown, "cold_or_unknown_members": cold, "warm_members": warm,
            "verdict": "ONE_PROGRAM" if one else "NOT_PROVEN",
            "rule": ("ONE_PROGRAM = every member's run proof names the same AOT cheap key per domain; cold members "
                     "should be only the first wave (one per record signature)")}


def driver(study):
    """``study.run`` driver: tempered TEKI over the study's parameters, members through the launcher."""
    from gpuwrf.surface_params import eki
    from gpuwrf.surface_params.cli import expand_from_basis, load_basis, member_multipliers
    from gpuwrf.surface_params.stations import collect
    from gpuwrf.verify.observations import read_stations
    from gpuwrf.verify.wrfout import extract_wrfout

    params, scorer, cfg = study["parameters"], study["scorer"], study["configuration"]
    train = [c for c in study["cases"] if c["split"] == "train"]
    test = [c for c in study["cases"] if c["split"] == "test"]
    cutoff = pd.Timestamp(study["test_start"])
    spec = {"parameters": params}
    variables, sigma_of = tuple(scorer["variables"]), scorer["sigma"]

    def drive(*, prefetch, files, output, ledger, **_):
        basis_path = Path(study["basis"]["path"])
        if identity(basis_path) != {k: study["basis"][k] for k in ("path", "sha256", "bytes")}:
            raise ValueError("control basis changed since the study was loaded")
        basis, manifest = load_basis(basis_path)

        def candidate(cid, theta):
            fields = expand_from_basis(basis, manifest, member_multipliers(spec, theta))
            ident = _write_controls(output/"controls"/f"{cid}.npz", fields,
                                    {"member": cid, "theta": [float(t) for t in theta], "study_basis": study["basis"]["sha256"]})
            return {"id": cid, "physics": cfg["physics"], "program": cfg["id"], "controls": ident,
                    "theta": [float(t) for t in theta]}

        # 1) observations (twin: from truth members; stations: the study file)
        if scorer["kind"] == "twin":
            truth = candidate("truth", np.log([scorer["truth"][q["name"]] for q in params]))
            prefetch([(truth, c) for c in study["cases"]])
            obs_path = output/"twin"/"observations.csv"
            text = _twin_observations(study, truth, files, read_stations, extract_wrfout)
            if obs_path.exists():
                if obs_path.read_text() != text:
                    raise ValueError("existing twin observations differ from the truth members (changed run)")
            else:
                obs_path.parent.mkdir(parents=True, exist_ok=True)
                obs_path.write_text(text)
        else:
            obs_path = Path(scorer["observations"])
            if identity(obs_path) not in study["inputs"]:
                raise ValueError("observations changed since the study was loaded")

        def matrices(members, cases, *, split):
            parts, ys, sigmas, keys = [], [], [], []
            for case in cases:
                pred, y, sig, k = collect([files(m, case)[0] for m in members], study["stations"], obs_path,
                                          variables=variables, sigma=sigma_of, **study["sampling"])
                keep = _keys_after(k, case["issue"], before=cutoff if split == "train" else None)
                if not keep:
                    raise ValueError(f"no observation support after the issue time for case {case['id']}")
                parts.append(pred[:, keep])
                ys.append(y[keep])
                sigmas.append(sig[keep])
                keys += [(case["id"], *k[n]) for n in keep]
            return np.concatenate(parts, axis=1), np.concatenate(ys), np.concatenate(sigmas), keys

        # 2) tempered TEKI iterations
        state = eki.initial_ensemble(_problem(study), study["eki"]["members"], seed=study["eki"]["seed"])
        iterations, history, support = study["eki"]["iterations"], [], None
        for k in range(iterations):
            members = [candidate(f"i{k:02d}m{j:03d}", row) for j, row in enumerate(state.theta)]
            prefetch([(m, c) for m in members for c in train])
            g, y, sig, keys = matrices(members, train, split="train")
            if support is None:
                support = keys
            elif keys != support:
                raise ValueError("observation support changed between EKI iterations")
            problem = _problem(study, y, sig)
            state = eki.update(problem, state, g, dt=1.0/iterations, seed=study["eki"]["seed"]+1+k)
            history.append({"iteration": k, "members": [m["id"] for m in members], **state.history[-1],
                            "observations": int(y.size)})
        estimate = state.theta.mean(axis=0)

        # 3) prior and posterior mean on every date (train misfit + held-out diagnostic)
        prior, posterior = candidate("prior", np.zeros(len(params))), candidate("posterior", estimate)
        prefetch([(m, c) for m in (prior, posterior) for c in study["cases"]])
        evaluation = {}
        for split, cases in (("train", train), ("test", test)):
            g, y, sig, keys = matrices([prior, posterior], cases, split=split)
            evaluation[split] = {name: _scores(g[n], y, sig, keys) for n, name in enumerate(("prior", "posterior"))}
        report = {"schema": RESULT_SCHEMA, "region": study["region"], "configuration": cfg["id"],
                  "parameters": [{"name": q["name"], "control": q["control"], "classes": q["classes"],
                                  "domains": q["domains"]} for q in params],
                  "estimate_multipliers": dict(zip((q["name"] for q in params), np.exp(estimate).tolist())),
                  "posterior_sd_log": dict(zip((q["name"] for q in params), state.theta.std(axis=0, ddof=1).tolist())),
                  "ensemble_final_multipliers": np.exp(state.theta).tolist(), "iterations": history,
                  "evaluation": evaluation, "controls": {"prior": prior["controls"], "posterior": posterior["controls"]},
                  "program": program_report(ledger), "scorer": {k: v for k, v in scorer.items() if k != "truth"},
                  "budget_case_hours": study["budget_case_hours"], "planned_case_hours": study["planned_case_hours"],
                  "evaluated_case_hours": len(ledger["members"])*study["hours"], "model_sha256": study["model_sha256"],
                  "train_cases": [c["id"] for c in train], "test_cases": [c["id"] for c in test],
                  "status": "HELD_OUT_DIAGNOSTIC; EXPERIMENTAL continuous calibration, no release promotion"}
        if scorer["kind"] == "twin":
            truth_log = {q["name"]: math.log(scorer["truth"][q["name"]]) for q in params}
            errors = {q["name"]: abs(float(estimate[n])-truth_log[q["name"]]) for n, q in enumerate(params)}
            report["twin"] = {"truth_multipliers": scorer["truth"], "abs_log_error": errors,
                              "tolerance_log": scorer["tolerance_log"],
                              "verdict": ("RECOVERED" if all(e <= scorer["tolerance_log"] for e in errors.values())
                                          else "NOT_RECOVERED"),
                              "observations": identity(obs_path)}
        json.dumps(report, allow_nan=False)
        return report

    return drive


def _scores(pred, y, sigma, keys):
    variables = sorted({k[3] for k in keys})
    out = {"normalised_misfit": float(np.mean(((pred-y)/sigma)**2)), "observations": int(y.size)}
    for variable in variables:
        mask = np.array([k[3] == variable for k in keys])
        out[f"rmse_{variable}"] = float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))
        out[f"bias_{variable}"] = float(np.mean(pred[mask]-y[mask]))
    return out


def _twin_observations(study, truth, files, read_stations, extract_wrfout):
    """Synthetic canonical observation CSV: truth histories at the stations + seeded N(0, sigma) noise, fixed order."""
    scorer = study["scorer"]
    stations = read_stations(study["stations"])
    rows = []
    for case in study["cases"]:
        frame, _ = extract_wrfout(files(truth, case)[0], stations, **study["sampling"])
        frame = frame[frame.variable.isin(scorer["variables"])]
        frame = frame.loc[pd.to_datetime(frame.valid_time, utc=True) > pd.Timestamp(case["issue"])]
        rows.append(frame[["station_uid", "valid_time", "variable", "value"]])
    table = pd.concat(rows, ignore_index=True)
    table["valid_time"] = pd.to_datetime(table.valid_time, utc=True)
    table = table.sort_values(["valid_time", "station_uid", "variable"]).reset_index(drop=True)
    rng = np.random.default_rng(scorer["noise_seed"])
    noise = rng.standard_normal(len(table))*np.array([scorer["sigma"][v] for v in table.variable])
    values = table.value.to_numpy(np.float64)+noise
    positive = table.variable.eq("wind").to_numpy()  # wind speed stays nonnegative
    values = np.where(positive, np.abs(values), values)
    lines = ["station_uid,valid_time,variable,value"]
    lines += [f"{u},{t.strftime('%Y-%m-%dT%H:%M:%SZ')},{v},{x!r}"
              for u, t, v, x in zip(table.station_uid, table.valid_time, table.variable, values.tolist())]
    return "\n".join(lines)+"\n"


def run(study, output, **kwargs):
    """``study.run`` with the continuous driver (same launcher, admission, chunk/resume, ledger and receipts)."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    from .study import run as study_run

    refuse_protected_inputs("gpuwrf optimize continuous", [], outputs=[output])
    return study_run(study, output, driver=driver(study), **kwargs)
