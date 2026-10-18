"""Strict, versioned configuration for ``gpuwrf da`` (schema ``gpuwrf.da.v1``).

The configuration is one JSON object. Every key is checked before any data file is opened: unknown
keys, wrong types, out-of-range numbers, naive (offset-less) times and unsupported domains are refused
with a ``ConfigError`` that names the offending key. Relative paths are resolved against the directory
of the configuration file. ``validate`` returns a :class:`DaConfig` whose ``data`` is the normalised
JSON-able dict (defaults filled in) and whose ``sha256`` identifies it in every receipt.

Sections and owners (lanes w41-da-core / -ens / -cycle):

``schema, description, t_a, domain, template_wrfinput, control_file, obs, params, moisture,
governance, localisation, experiments``  core (this module);
``ensemble``  validated here (probe-known keys), owned scientifically by the ensemble lane;
``inflation`` and ``cycle``  delegated to ``inflation.validate_config_section`` and
``cycle_loop.validate_config_section``; a build without the delegate refuses the section.

See ``docs/da.md`` for the meaning of each key.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import re

SCHEMA = "gpuwrf.da.v1"
KINDS = ("sonde", "surface", "cloud")
SUPPORTED_DOMAINS = ("d01",)
ENSEMBLE_KINDS = ("members", "same_phase")
SCALES = ("innovation", "innovation_bands", "innovation_debiased")
FRAME_NAMES = ("U", "V", "T", "QVAPOR", "P", "PB", "PH", "PHB", "T2", "Q2", "U10", "V10", "PSFC", "MU", "MUB")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_DELEGATED = {"inflation": "gpuwrf.assimilation.letkf.inflation",
              "cycle": "gpuwrf.assimilation.letkf.cycle_loop"}


class ConfigError(ValueError):
    """The configuration is invalid; nothing has been read or written."""


# ------------------------------------------------------------------------------------------- primitives

def _fail(where, message):
    raise ConfigError(f"{where}: {message}")


def _keys(where, value, allowed, required=()):
    if not isinstance(value, dict):
        _fail(where, f"must be an object, got {type(value).__name__}")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        _fail(where, f"unknown key(s) {unknown}; allowed {sorted(allowed)}")
    missing = [k for k in required if k not in value]
    if missing:
        _fail(where, f"missing required key(s) {missing}")
    return value


def _str(where, value, *, choices=None, pattern=None):
    if not isinstance(value, str) or not value:
        _fail(where, f"must be a non-empty string, got {value!r}")
    if choices is not None and value not in choices:
        _fail(where, f"must be one of {list(choices)}, got {value!r}")
    if pattern is not None and not pattern.match(value):
        _fail(where, f"{value!r} does not match {pattern.pattern}")
    return value


def _bool(where, value):
    if not isinstance(value, bool):
        _fail(where, f"must be true or false, got {value!r}")
    return value


def _num(where, value, *, lo=None, hi=None, lo_open=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f"must be a number, got {value!r}")
    if integer and (not isinstance(value, int) and not float(value).is_integer()):
        _fail(where, f"must be an integer, got {value!r}")
    if not math.isfinite(float(value)):
        _fail(where, f"must be finite, got {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        _fail(where, f"must be {'>' if lo_open else '>='} {lo}, got {value!r}")
    if hi is not None and value > hi:
        _fail(where, f"must be <= {hi}, got {value!r}")
    return int(value) if integer else float(value)


def _path(where, value, base):
    _str(where, value)
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (Path(base) / path))


def utc(where, value):
    """ISO 8601 time with an explicit offset -> aware UTC datetime (whole seconds)."""
    if isinstance(value, datetime):
        when = value
    else:
        _str(where, value)
        try:
            when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            _fail(where, f"not an ISO 8601 time: {value!r}")
    if when.tzinfo is None:
        _fail(where, "time must carry an explicit UTC offset (e.g. 2026-05-04T12:00:00Z)")
    when = when.astimezone(timezone.utc)
    if when.microsecond:
        _fail(where, "time must be whole seconds")
    return when


def iso(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------- sections

def _letkf_param_specs():
    from .analysis import LetkfParams
    out = {}
    for f in dc_fields(LetkfParams):
        if f.name in ("negative_qv", "supersaturation"):
            continue                                   # moisture section
        out[f.name] = f.default
    return out


_PARAM_RANGES = {
    "loc_h_km": dict(lo=0, lo_open=True), "loc_v_lnp": dict(lo=0, lo_open=True),
    "loc_v_lnp_low": dict(lo=0, lo_open=True), "p_low_pa": dict(lo=0, lo_open=True),
    "loc_v_sfc_lnp": dict(lo=0, lo_open=True), "stride_h": dict(lo=1, integer=True),
    "stride_v": dict(lo=1, integer=True), "pmax": dict(lo=1, integer=True), "inflation": dict(lo=1.0),
    "rtps_alpha": dict(lo=0.0, hi=1.0), "gross_k": dict(lo=0, lo_open=True),
    "spec_bdy_width": dict(lo=0, integer=True), "taper_ramp": dict(lo=0, integer=True),
    "workers": dict(lo=1, integer=True),
}
_ANALYSABLE = ("U", "V", "T", "QVAPOR")
_DIAGNOSTIC = ("T2", "Q2", "U10", "V10", "PSFC")


def _params(where, value):
    specs = _letkf_param_specs()
    _keys(where, value, specs)
    out = {}
    for key, raw in value.items():
        w = f"{where}.{key}"
        if key == "rebalance":
            out[key] = _bool(w, raw)
        elif key == "variables":
            if not isinstance(raw, list) or not raw:
                _fail(w, "must be a non-empty list")
            for name in raw:
                _str(w, name, choices=_ANALYSABLE)
            if len(set(raw)) != len(raw):
                _fail(w, "duplicate names")
            out[key] = list(raw)
        elif key == "diagnostic_variables":
            if not isinstance(raw, list):
                _fail(w, "must be a list")
            for name in raw:
                _str(w, name, choices=_DIAGNOSTIC)
            out[key] = list(raw)
        else:
            out[key] = _num(w, raw, **_PARAM_RANGES[key])
    return out


def _moisture(where, value):
    from .moisture import MoisturePolicy
    allowed = {f.name for f in dc_fields(MoisturePolicy)}
    _keys(where, value, allowed)
    out = dict(value)
    for key in ("log_floor_kgkg", "max_added_column_kg_m2", "max_added_fraction"):
        if key in out:
            out[key] = _num(f"{where}.{key}", out[key], lo=0.0)
    try:
        policy = MoisturePolicy(**out)
    except ValueError as exc:
        _fail(where, str(exc))
    return policy.as_dict()


_SOURCE_KEYS = {
    "igra2": ({"format", "path", "station", "valid_time", "drift", "include_surface", "ascent_mps"},
              ("path", "station")),
    "isd_lite": ({"format", "path", "station", "isd_history", "station_id", "window_minutes"}, ("path",)),
    "csv_surface": ({"format", "path", "window_minutes", "include_pressure"}, ("path",)),
    "csv_profile": ({"format", "path", "window_minutes"}, ("path",)),
    "obsdb_parquet": ({"format", "path", "window_minutes", "window_hours", "options"}, ("path",)),
    "obsbatch_json": ({"format", "path"}, ("path",)),
    "msg_clm_npz": ({"format", "path", "stride", "min_pixels", "role"}, ("path",)),
}


def _source(where, value, base):
    if not isinstance(value, dict) or "format" not in value:
        _fail(where, "each source needs a 'format'")
    fmt = _str(f"{where}.format", value["format"], choices=sorted(_SOURCE_KEYS))
    allowed, required = _SOURCE_KEYS[fmt]
    _keys(where, value, allowed, required)
    out = {"format": fmt, "path": _path(f"{where}.path", value["path"], base)}
    if fmt == "igra2":
        out["station"] = _str(f"{where}.station", value["station"], pattern=re.compile(r"^[A-Z0-9]{11}$"))
        if "valid_time" in value:
            out["valid_time"] = iso(utc(f"{where}.valid_time", value["valid_time"]))
        for key in ("drift", "include_surface"):
            if key in value:
                out[key] = _bool(f"{where}.{key}", value[key])
        if "ascent_mps" in value:
            out["ascent_mps"] = _num(f"{where}.ascent_mps", value["ascent_mps"], lo=0, lo_open=True)
    elif fmt == "isd_lite":
        if ("station" in value) == ("isd_history" in value):
            _fail(where, "give exactly one of 'station' {id, lat, lon, elevation_m} or 'isd_history' + 'station_id'")
        if "station" in value:
            st = _keys(f"{where}.station", value["station"], {"id", "lat", "lon", "elevation_m"},
                       ("id", "lat", "lon", "elevation_m"))
            out["station"] = {"id": _str(f"{where}.station.id", st["id"]),
                              "lat": _num(f"{where}.station.lat", st["lat"], lo=-90, hi=90),
                              "lon": _num(f"{where}.station.lon", st["lon"], lo=-360, hi=360),
                              "elevation_m": _num(f"{where}.station.elevation_m", st["elevation_m"], lo=-500)}
            if "station_id" in value:
                _fail(where, "'station_id' only goes with 'isd_history'")
        else:
            out["isd_history"] = _path(f"{where}.isd_history", value["isd_history"], base)
            out["station_id"] = _str(f"{where}.station_id", value.get("station_id"),
                                     pattern=re.compile(r"^\d{6}-\d{5}$"))
    if fmt in ("isd_lite", "csv_surface", "csv_profile", "obsdb_parquet") and "window_minutes" in value:
        out["window_minutes"] = _num(f"{where}.window_minutes", value["window_minutes"], lo=0, hi=180)
    if fmt == "csv_surface" and "include_pressure" in value:
        out["include_pressure"] = _bool(f"{where}.include_pressure", value["include_pressure"])
    if fmt == "obsdb_parquet":
        if "window_hours" in value:
            out["window_hours"] = _num(f"{where}.window_hours", value["window_hours"], lo=0, lo_open=True, hi=12)
        if "options" in value:
            opts = _keys(f"{where}.options", value["options"],
                         {"networks", "hourly_tolerance_s", "include_pressure", "dedup"})
            out["options"] = dict(opts)
    if fmt == "msg_clm_npz":
        if "stride" in value:
            out["stride"] = _num(f"{where}.stride", value["stride"], lo=1, integer=True)
        if "min_pixels" in value:
            out["min_pixels"] = _num(f"{where}.min_pixels", value["min_pixels"], lo=1, integer=True)
        out["role"] = _str(f"{where}.role", value.get("role", "withhold"), choices=("withhold",))
    return out


def _obs(where, value, base):
    _keys(where, value, {"sources", "sonde_superob", "error_overrides"}, ("sources",))
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        _fail(f"{where}.sources", "must be a non-empty list")
    out = {"sources": [_source(f"{where}.sources[{i}]", s, base) for i, s in enumerate(sources)],
           "sonde_superob": _bool(f"{where}.sonde_superob", value.get("sonde_superob", True))}
    if "error_overrides" in value:
        overrides = value["error_overrides"]
        from .obs import DEFAULT_OBS_ERRORS
        _keys(f"{where}.error_overrides", overrides, {"SONDE", "SURFACE", "CLOUD"})
        for kind, entries in overrides.items():
            _keys(f"{where}.error_overrides.{kind}", entries, DEFAULT_OBS_ERRORS[kind])
            for var, spec in entries.items():
                w = f"{where}.error_overrides.{kind}.{var}"
                _keys(w, spec, {"sigma", "sigma_rh"})
                for key, v in spec.items():
                    _num(f"{w}.{key}", v, lo=0, lo_open=True)
        out["error_overrides"] = overrides
    return out


def _ensemble(where, value, base):
    kind = _str(f"{where}.kind", value.get("kind") if isinstance(value, dict) else None, choices=ENSEMBLE_KINDS)
    common = {"kind", "names", "scale", "inflation_rho", "band_edges_pa", "per_band", "rho_min", "rho_max", "n_min"}
    if kind == "members":
        _keys(where, value, common | {"member_glob", "member_files", "fourd"})
        if ("member_glob" in value) == ("member_files" in value):
            _fail(where, "give exactly one of 'member_glob' or 'member_files'")
    else:
        _keys(where, value, common | {"runs_glob", "max_age_days", "max_members"}, ("runs_glob",))
    out = {"kind": kind}
    if "member_glob" in value:
        out["member_glob"] = _path(f"{where}.member_glob", value["member_glob"], base)
    if "member_files" in value:
        files = value["member_files"]
        if not isinstance(files, list) or len(files) < 2:
            _fail(f"{where}.member_files", "needs at least 2 member files")
        out["member_files"] = [_path(f"{where}.member_files[{i}]", f, base) for i, f in enumerate(files)]
        if len(set(out["member_files"])) != len(out["member_files"]):
            _fail(f"{where}.member_files", "duplicate member files")
    if kind == "same_phase":
        out["runs_glob"] = _path(f"{where}.runs_glob", value["runs_glob"], base)
        out["max_age_days"] = _num(f"{where}.max_age_days", value.get("max_age_days", 45), lo=0, lo_open=True)
        if value.get("max_members") is not None:
            out["max_members"] = _num(f"{where}.max_members", value["max_members"], lo=2, integer=True)
    names = value.get("names", list(FRAME_NAMES))
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        _fail(f"{where}.names", "must be a list of wrfout variable names")
    missing = [n for n in ("U", "V", "T", "QVAPOR", "P", "PB") if n not in names]
    if missing:
        _fail(f"{where}.names", f"must include {missing}")
    out["names"] = list(names)
    scale = value.get("scale", 1.0)
    if isinstance(scale, str):
        out["scale"] = _str(f"{where}.scale", scale, choices=SCALES)
    else:
        out["scale"] = _num(f"{where}.scale", scale, lo=0, lo_open=True)
    if out["scale"] == "innovation_bands" and kind == "members":
        _fail(f"{where}.scale", "'innovation_bands' rescales archive frames; use it with kind 'same_phase'")
    if out["scale"] == "innovation_debiased" and kind != "members":
        _fail(f"{where}.scale", "'innovation_debiased' needs kind 'members'")
    if "inflation_rho" in value:
        out["inflation_rho"] = _num(f"{where}.inflation_rho", value["inflation_rho"], lo=1.0)
    if "band_edges_pa" in value:
        edges = value["band_edges_pa"]
        if not isinstance(edges, list) or not edges:
            _fail(f"{where}.band_edges_pa", "must be a non-empty list")
        edges = [_num(f"{where}.band_edges_pa[{i}]", e, lo=0, lo_open=True) for i, e in enumerate(edges)]
        if any(b >= a for a, b in zip(edges, edges[1:])):
            _fail(f"{where}.band_edges_pa", "must be strictly decreasing pressures")
        out["band_edges_pa"] = edges
    if "per_band" in value:
        out["per_band"] = _bool(f"{where}.per_band", value["per_band"])
    for key, kw in (("rho_min", dict(lo=1.0)), ("rho_max", dict(lo=1.0)), ("n_min", dict(lo=2, integer=True))):
        if key in value:
            out[key] = _num(f"{where}.{key}", value[key], **kw)
    if "fourd" in value:
        fd = _keys(f"{where}.fourd", value["fourd"], {"member_run_glob", "window_hours"},
                   ("member_run_glob", "window_hours"))
        out["fourd"] = {"member_run_glob": _path(f"{where}.fourd.member_run_glob", fd["member_run_glob"], base),
                        "window_hours": _num(f"{where}.fourd.window_hours", fd["window_hours"], lo=0, lo_open=True,
                                             hi=12)}
    return out


def _governance(where, value):
    _keys(where, value, {"plugin", "options", "hash_inputs"})
    plugin = value.get("plugin", "none")
    _str(f"{where}.plugin", plugin)
    if plugin not in ("none", "alisios") and not re.match(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$", plugin):
        _fail(f"{where}.plugin", "must be 'none', 'alisios' or 'package.module:factory'")
    options = value.get("options", {})
    if not isinstance(options, dict):
        _fail(f"{where}.options", "must be an object")
    return {"plugin": plugin, "options": options,
            "hash_inputs": _bool(f"{where}.hash_inputs", value.get("hash_inputs", True))}


def _localisation(where, value):
    _keys(where, value, {"variables"})
    table = value.get("variables", {})
    _keys(f"{where}.variables", table, _ANALYSABLE + _DIAGNOSTIC)
    for state, factors in table.items():
        w = f"{where}.variables.{state}"
        if not isinstance(factors, dict):
            _fail(w, "must map obs keys ('*', 'sonde', 'surface:T2', ...) to factors")
        for key, factor in factors.items():
            if key != "*" and not re.match(r"^(sonde|surface|cloud)(:[A-Z0-9]+)?$", key):
                _fail(w, f"bad obs key {key!r}")
            _num(f"{w}.{key}", factor, lo=0.0, hi=1.0)
    return {"variables": table}


def _experiment(where, value, base):
    allowed = {"name", "assimilate", "withhold", "params", "write_mean", "write_members", "profile_site", "land"}
    _keys(where, value, allowed, ("name", "assimilate"))
    out = {"name": _str(f"{where}.name", value["name"], pattern=NAME_RE)}
    for key in ("assimilate", "withhold"):
        kinds = value.get(key, [])
        if not isinstance(kinds, list):
            _fail(f"{where}.{key}", "must be a list of observation kinds")
        for k in kinds:
            _str(f"{where}.{key}", k, choices=KINDS)
        if len(set(kinds)) != len(kinds):
            _fail(f"{where}.{key}", "duplicate kinds")
        out[key] = list(kinds)
    if not out["assimilate"]:
        _fail(f"{where}.assimilate", "must name at least one observation kind")
    if "cloud" in out["assimilate"]:
        _fail(f"{where}.assimilate", "cloud observations are not assimilated in this build (no cloud analysis); "
                                     "list them under 'withhold' for diagnostics")
    overlap = set(out["assimilate"]) & set(out["withhold"])
    if overlap:
        _fail(where, f"kinds {sorted(overlap)} are both assimilated and withheld")
    out["params"] = _params(f"{where}.params", value.get("params", {}))
    out["write_mean"] = _bool(f"{where}.write_mean", value.get("write_mean", False))
    # land state of the written mean wrfinput: 'cycled' = the control frame's evolved Noah-MP state with the gpuwrf
    # land-cycle marker (warm start from a forecast frame); 'cold' = no marker, the model cold-initialises
    # snow/canopy as for any real.exe input (frames that are themselves wrfinputs)
    out["land"] = _str(f"{where}.land", value.get("land", "cycled"), choices=("cycled", "cold"))
    if "write_members" in value:
        wm = _keys(f"{where}.write_members", value["write_members"], {"source_case_dir", "hours"},
                   ("source_case_dir", "hours"))
        out["write_members"] = {"source_case_dir": _path(f"{where}.write_members.source_case_dir",
                                                         wm["source_case_dir"], base),
                                "hours": _num(f"{where}.write_members.hours", wm["hours"], lo=1, integer=True)}
    if "profile_site" in value:
        site = _keys(f"{where}.profile_site", value["profile_site"], {"lat", "lon"}, ("lat", "lon"))
        out["profile_site"] = {"lat": _num(f"{where}.profile_site.lat", site["lat"], lo=-90, hi=90),
                               "lon": _num(f"{where}.profile_site.lon", site["lon"], lo=-360, hi=360)}
    return out


def _delegated(name, where, value, base=None):
    module_name = _DELEGATED[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = None
    validator = getattr(module, "validate_config_section", None)
    if validator is None:
        _fail(where, f"section {name!r} is not supported by this build ({module_name}.validate_config_section "
                     "missing)")
    try:
        # validators that take base_dir resolve their relative paths against the config directory, as member_files
        if base is not None and "base_dir" in inspect.signature(validator).parameters:
            return validator(value, base_dir=base)
        return validator(value)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        _fail(where, str(exc))


# ------------------------------------------------------------------------------------------- top level

TOP_KEYS = {"schema", "description", "t_a", "domain", "template_wrfinput", "control_file", "ensemble", "obs",
            "params", "moisture", "governance", "localisation", "experiments", "inflation", "cycle"}


@dataclass(frozen=True)
class DaConfig:
    data: dict
    source: str | None
    sha256: str

    @property
    def t_a(self) -> datetime:
        return utc("t_a", self.data["t_a"])

    @property
    def domain(self) -> str:
        return self.data["domain"]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

    def with_overrides(self, **changes) -> "DaConfig":
        """New validated config with top-level keys replaced (e.g. ``t_a``, ``ensemble``) — for cycle drivers."""
        data = json.loads(json.dumps(self.data))
        data.update(changes)
        return validate(data, base_dir=Path(self.source).parent if self.source else Path.cwd(), source=self.source)


def validate(raw: dict, *, base_dir=None, source=None) -> DaConfig:
    """Validate and normalise a configuration object; raises ConfigError before anything is read."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    _keys("config", raw, TOP_KEYS, ("schema", "t_a", "ensemble", "obs", "experiments"))
    if raw["schema"] != SCHEMA:
        _fail("config.schema", f"must be {SCHEMA!r}, got {raw['schema']!r}")
    out = {"schema": SCHEMA}
    if "description" in raw:
        out["description"] = _str("config.description", raw["description"])
    out["t_a"] = iso(utc("config.t_a", raw["t_a"]))
    domain = raw.get("domain", "d01")
    if domain not in SUPPORTED_DOMAINS:
        _fail("config.domain", f"only {list(SUPPORTED_DOMAINS)} is supported (nested analysis: parent/child base "
                               f"states differ, A1 R6), got {domain!r}")
    out["domain"] = domain
    for key in ("template_wrfinput", "control_file"):
        if key in raw:
            out[key] = _path(f"config.{key}", raw[key], base)
    out["ensemble"] = _ensemble("config.ensemble", raw["ensemble"], base)
    out["obs"] = _obs("config.obs", raw["obs"], base)
    out["params"] = _params("config.params", raw.get("params", {}))
    out["moisture"] = _moisture("config.moisture", raw.get("moisture", {}))
    out["governance"] = _governance("config.governance", raw.get("governance", {}))
    if "localisation" in raw:
        out["localisation"] = _localisation("config.localisation", raw["localisation"])
    experiments = raw["experiments"]
    if not isinstance(experiments, list) or not experiments:
        _fail("config.experiments", "must be a non-empty list")
    out["experiments"] = [_experiment(f"config.experiments[{i}]", e, base) for i, e in enumerate(experiments)]
    names = [e["name"] for e in out["experiments"]]
    if len(set(names)) != len(names):
        _fail("config.experiments", "experiment names must be unique")
    if sum(e["write_mean"] for e in out["experiments"]) > 1:
        _fail("config.experiments", "at most one experiment may set write_mean")
    if any(e["write_mean"] or "write_members" in e for e in out["experiments"]):
        if "template_wrfinput" not in out:
            _fail("config.template_wrfinput", "required when an experiment writes wrfinput files")
    if any(e["write_mean"] for e in out["experiments"]) and out["ensemble"]["kind"] == "members" \
            and "control_file" not in out:
        _fail("config.control_file", "write_mean with a 'members' ensemble needs the control (deterministic "
                                     "background) wrfout frame")
    if out["ensemble"]["kind"] == "same_phase" and "control_file" not in out:
        _fail("config.control_file", "a 'same_phase' ensemble is centred on the control wrfout frame; give it")
    if any("write_members" in e for e in out["experiments"]) and out["ensemble"]["kind"] != "members":
        _fail("config.experiments", "write_members needs ensemble kind 'members' (each member's own background)")
    for name in ("inflation", "cycle"):
        if name in raw:
            out[name] = _delegated(name, f"config.{name}", raw[name], base)
    text = json.dumps(out, sort_keys=True, separators=(",", ":"))
    return DaConfig(data=out, source=None if source is None else str(source),
                    sha256=hashlib.sha256(text.encode()).hexdigest())


def load(path) -> DaConfig:
    """Read and validate a JSON configuration file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read ({exc})") from exc
    return validate(raw, base_dir=path.resolve().parent, source=path.resolve())
