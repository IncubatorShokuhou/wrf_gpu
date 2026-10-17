"""Resolved WRF-Solar configuration and restart selection/clock checks.

No geometry or device admission belongs here: the operational runtime owns
that gate. Configurations are bound namelists, normalized saved dictionaries,
or single-domain raw dictionaries containing resolved scalar values.
"""
from collections.abc import Mapping
import math


SOLAR_INTERPOLATION_SCHEMA_VERSION = 1
CONFIG_KEYS = (
    "swint_opt", "ra_sw_physics", "ra_lw_physics", "sf_surface_physics",
    "solar_opt", "topo_shading", "slope_rad", "sf_urban_physics", "sf_lake_physics",
    "dt_s", "radiation_interval_s", "radiation_cadence_steps",
)


def _scalar(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Solar {name} needs a resolved domain scalar")
        return value[0]
    return value


def _lookup(namelist, name, default):
    if isinstance(namelist, Mapping):
        if name in namelist:
            return _scalar(namelist[name], name)
        for section in ("physics", "time_control", "domains"):
            values = namelist.get(section, {})
            if isinstance(values, Mapping) and name in values:
                return _scalar(values[name], name)
        return default
    return getattr(namelist, name, default)


def _integer(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Solar {name} must be an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Solar {name} must be a finite integer")
    return int(number)


def configuration(namelist):
    """Canonical selectors and BOTH nominal and step-count radiation clocks."""
    surface = _lookup(namelist, "sf_surface_physics", None)
    if surface is None:
        surface = 4 if _lookup(namelist, "use_noahmp", False) else 2
    dt = float(_lookup(namelist, "dt_s", _lookup(namelist, "time_step", 0.)))
    interval = float(_lookup(namelist, "radiation_interval_s", 0.))
    if interval == 0.:
        radt = _lookup(namelist, "radt", None)
        if radt is not None:
            interval = float(radt) * 60.
    cadence = _lookup(namelist, "radiation_cadence_steps", None)
    if cadence is None:
        cadence = max(1, math.floor(interval / dt + .5)) if dt > 0 and math.isfinite(dt) and interval > 0 and math.isfinite(interval) else 0
    cadence = _integer(cadence, "radiation_cadence_steps")
    if interval == 0.:
        interval = dt * cadence
    result = {
        "swint_opt": _integer(_lookup(namelist, "swint_opt", 0), "swint_opt"),
        "ra_sw_physics": _integer(_lookup(namelist, "ra_sw_physics", 4), "ra_sw_physics"),
        "ra_lw_physics": _integer(_lookup(namelist, "ra_lw_physics", 4), "ra_lw_physics"),
        "sf_surface_physics": _integer(surface, "sf_surface_physics"),
        "solar_opt": _integer(_lookup(namelist, "solar_opt", 0), "solar_opt"),
        "topo_shading": _integer(_lookup(namelist, "topo_shading", 0), "topo_shading"),
        "slope_rad": _integer(_lookup(namelist, "slope_rad", 0), "slope_rad"),
        "sf_urban_physics": _integer(_lookup(namelist, "sf_urban_physics", 0), "sf_urban_physics"),
        "sf_lake_physics": _integer(_lookup(namelist, "sf_lake_physics", 0), "sf_lake_physics"),
        "dt_s": dt, "radiation_interval_s": interval, "radiation_cadence_steps": cadence,
    }
    return result


def validate_options(namelist):
    """Validate the qualified scheme and fixed-clock scope, returning config."""
    cfg = configuration(namelist)
    if cfg["swint_opt"] == 0:
        return cfg
    if cfg["swint_opt"] != 1:
        raise ValueError("Solar swint_opt supports only 0/1")
    if (cfg["ra_sw_physics"], cfg["ra_lw_physics"], cfg["sf_surface_physics"]) != (4, 4, 4):
        raise ValueError("Solar swint_opt=1 requires RRTMG ra_sw/lw_physics=4 and Noah-MP sf_surface_physics=4")
    if any(cfg[name] != 0 for name in ("solar_opt", "topo_shading", "slope_rad", "sf_urban_physics", "sf_lake_physics")):
        raise ValueError("Solar swint_opt=1 requires solar_opt/topo_shading/slope_rad/urban/lake options off")
    if not math.isfinite(cfg["dt_s"]) or cfg["dt_s"] <= 0:
        raise ValueError("Solar interpolation requires a positive finite fixed dt_s")
    if not math.isfinite(cfg["radiation_interval_s"]) or cfg["radiation_interval_s"] <= 0 or cfg["radiation_cadence_steps"] <= 0:
        raise ValueError("Solar interpolation requires positive finite radiation interval and cadence")
    return cfg


def validate_restart(group, saved_config, version, shape, *, requested=None):
    """E78: restore twelve REAL leaves and the exact active configuration.

    Missing markers/groups in genuinely inactive old restarts remain valid.
    An active file must have the complete version-1 normalized config, even
    when the caller supplies a requested namelist. Requested active selection
    cannot turn an inactive old checkpoint into a seeded Solar continuation.
    """
    from gpuwrf.diagnostics.solar import validate_restart_solar

    if saved_config is not None and not isinstance(saved_config, Mapping):
        raise ValueError("restart Solar interpolation config is malformed (E78)")
    try:
        saved = configuration(saved_config)
        wanted = configuration(requested) if requested is not None else saved
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("restart Solar interpolation config is malformed (E78)") from error
    active = bool(saved["swint_opt"] or wanted["swint_opt"] or group is not None)
    if not active:
        validate_restart_solar(group, False, version, shape)
        return
    if version != SOLAR_INTERPOLATION_SCHEMA_VERSION or saved_config is None or set(saved_config) != set(CONFIG_KEYS):
        raise ValueError("restart Solar interpolation schema/config is missing (E78)")
    try:
        saved = validate_options(saved)
        wanted = validate_options(wanted)
    except ValueError as error:
        raise ValueError(f"restart Solar interpolation selection/clock is invalid (E78): {error}") from error
    validate_restart_solar(group, bool(saved["swint_opt"]), version, shape)
    if requested is not None:
        validate_restart_solar(group, bool(wanted["swint_opt"]), version, shape)
        if saved != wanted:
            raise ValueError("restart Solar interpolation cadence/selection differs (E78)")
