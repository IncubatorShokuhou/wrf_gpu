"""Host-only tracer restart admission and content-bound source/sink selection."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import math
import re

import numpy as np

SCHEMA_VERSION = 1
STATE_FIELDS = ("tracers", "tracer_bdy", "tracer_budget")
_DEFAULTS = {"tracer_opt": 0, "tracer_adv_opt": 1, "tracer_mix2_off": False,
             "tracer_mix6_off": False, "tracer_pblmix": 1}


def _get(source, key, default=None):
    if isinstance(source, Mapping):
        if key in source:
            value = source[key]
        else:
            value = default
            for section in ("dynamics", "physics", "domains", "bdy_control"):
                group = source.get(section, {})
                if isinstance(group, Mapping) and key in group:
                    value = group[key]
                    break
    else:
        value = getattr(source, key, default)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"tracer restart requires a resolved per-domain {key}")
        value = value[0]
    return value


def configuration(namelist):
    """Canonical JSON-safe identity; paths are replaced by exact file content SHA."""
    result = {key: _get(namelist, key, default) for key, default in _DEFAULTS.items()}
    for key in ("tracer_opt", "tracer_adv_opt", "tracer_pblmix"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"tracer restart invalid {key}")
        result[key] = int(value)
    for key in ("tracer_mix2_off", "tracer_mix6_off"):
        if result[key] not in (True, False):
            raise ValueError(f"tracer restart invalid {key}")
        result[key] = bool(result[key])
    if result["tracer_opt"] not in (0, 2):
        raise ValueError("tracer restart supports tracer_opt=0/2 only")
    if result["tracer_adv_opt"] not in (0, 1, 2) or result["tracer_pblmix"] not in (0, 1):
        raise ValueError("tracer restart unsupported transport/PBL selector")
    custom = _get(namelist, "tracer_config")
    if custom is not None:
        custom = asdict(custom) if is_dataclass(custom) else custom
        if not isinstance(custom, Mapping) or set(custom) != {"species", "emissions", "source_sha256"}:
            raise ValueError("tracer restart custom configuration is incomplete")
        if not custom["species"] or not isinstance(custom["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", custom["source_sha256"]):
            raise ValueError("tracer restart custom species/content SHA is invalid")
        try:
            custom = json.loads(json.dumps(custom, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracer restart custom configuration is invalid") from exc
        rates = {"decay_rate_s", "deposition_velocity_m_s", "wet_scavenging_per_mm"}
        names = set()
        if not isinstance(custom["species"], list) or not isinstance(custom["emissions"], list):
            raise ValueError("tracer restart species/emissions must be lists")
        for species in custom["species"]:
            if not isinstance(species, dict) or set(species) != rates | {"name"}:
                raise ValueError("tracer restart species configuration is incomplete")
            name = species["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in names:
                raise ValueError("tracer restart species name is invalid or duplicated")
            names.add(name)
            for key in rates:
                value = species[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 or value > float(np.finfo(np.float32).max)):
                    raise ValueError("tracer restart source/sink coefficient is invalid")
        emission_keys = {"species_index", "start_s", "end_s", "x0", "x1", "y0", "y1", "rate_kg_s", "flux_kg_m2_s", "is_point"}
        for source in custom["emissions"]:
            if not isinstance(source, dict) or set(source) != emission_keys:
                raise ValueError("tracer restart emission configuration is incomplete")
            for key in ("species_index", "x0", "x1", "y0", "y1"):
                if type(source[key]) is not int or source[key] < 0:
                    raise ValueError("tracer restart emission index is invalid")
            for key in ("start_s", "end_s", "rate_kg_s", "flux_kg_m2_s"):
                value = source[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 or value > float(np.finfo(np.float32).max)):
                    raise ValueError("tracer restart emission rate/time is invalid")
            if (source["species_index"] >= len(names) or source["end_s"] <= source["start_s"]
                    or source["x1"] <= source["x0"] or source["y1"] <= source["y0"]
                    or type(source["is_point"]) is not bool):
                raise ValueError("tracer restart emission species/time/geometry is invalid")
    result["tracer_config"] = custom
    # Canonical configuration dictionaries carry the resolved flag directly.
    boundary = _get(namelist, "boundary_config")
    result["nested"] = bool(_get(namelist, "nested", False) or (
        _get(namelist, "run_boundary", True)
        and _get(boundary, "nested_frozen_wrf_boundary_bundle", False)
        and not _get(boundary, "force_geopotential", True)))
    if _get(namelist, "gpuwrf_tracer_file", "") and custom is None:
        raise ValueError("tracer restart requires the bound tracer file configuration")
    if result["tracer_opt"] == 0 and custom is None:
        result["nested"] = False
    else:
        # Source intervals use the saved step index times this resolved dt.
        # A different dt would silently replay emissions at a different time.
        dt = _get(namelist, "dt_s", _get(namelist, "time_step"))
        if (isinstance(dt, bool) or not isinstance(dt, (int, float, np.integer, np.floating))
                or not math.isfinite(dt) or dt <= 0 or dt > float(np.finfo(np.float32).max)):
            raise ValueError("tracer restart requires a positive finite dt_s")
        result["dt_s"] = float(dt)
    return result


def count(config):
    return (8 if config["tracer_opt"] == 2 else 0) + (
        len(config["tracer_config"]["species"]) if config["tracer_config"] is not None else 0)


def validate_restart(state_fields, tracer_tendencies, config, version, mass_shape, *, requested=None):
    """Refuse missing, mis-shaped, non-REAL or nonfinite active tracer memory."""
    populated = any(state_fields.get(key) is not None for key in STATE_FIELDS) or tracer_tendencies is not None
    if config is None:
        if populated or version is not None:
            raise ValueError("tracer restart lacks explicit configuration")
        config = configuration(None)
    else:
        keys = {*_DEFAULTS, "tracer_config", "nested"}
        if isinstance(config, Mapping) and (config.get("tracer_opt") == 2
                or config.get("tracer_config") is not None):
            keys.add("dt_s")
        if not isinstance(config, Mapping) or set(config) != keys:
            raise ValueError("tracer restart configuration is incomplete")
    config = configuration(config)
    n = count(config)
    if requested is not None and configuration(requested) != config:
        raise ValueError("tracer restart requested species/source/sink configuration differs")
    if n == 0:
        if populated:
            raise ValueError("tracer restart contains tracer state while tracers are OFF")
        return
    if version != SCHEMA_VERSION:
        raise ValueError("tracer restart schema version is missing or unsupported")

    def array(name, value, shape):
        if value is None:
            raise ValueError(f"tracer restart missing {name}")
        value = np.asarray(value)
        if value.shape != tuple(shape) or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
            raise ValueError(f"tracer restart invalid {name}: expected finite REAL32 {tuple(shape)}")

    array("tracers", state_fields.get("tracers"), (n, *mass_shape))
    array("tracer_tendencies", tracer_tendencies, (n, *mass_shape))
    if config["tracer_config"] is not None:
        array("tracer_budget", state_fields.get("tracer_budget"), (n, 4))
    elif state_fields.get("tracer_budget") is not None:
        raise ValueError("tracer restart budget exists without custom tracers")
    boundary = state_fields.get("tracer_bdy")
    if config["nested"]:
        shape = np.shape(boundary)
        if len(shape) != 6 or shape[:3] != (n, 2, 4) or shape[3] < 1 or shape[4] != mass_shape[0] or shape[5] < max(mass_shape[1:]):
            raise ValueError("tracer restart invalid nested tracer_bdy shape")
        array("tracer_bdy", boundary, shape)
    elif boundary is not None:
        raise ValueError("tracer restart root domain carries nested tracer_bdy")
