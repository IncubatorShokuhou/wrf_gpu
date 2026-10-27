"""Opt-in, radiation-call history; no fields or device work when rad_diag=0."""
from collections.abc import Mapping
from typing import NamedTuple


SCHEMA_VERSION = 1
HISTORY_NAMES = ("RTHRATLW", "RTHRATSW", "RTHRATLWC", "RTHRATSWC",
                 "QC_RAD", "QI_RAD", "QS_RAD", "QG_RAD")


class RadiationHistory(NamedTuple):
    rthratlw: object
    rthratsw: object
    rthratlwc: object
    rthratswc: object
    qc_rad: object
    qi_rad: object
    qs_rad: object
    qg_rad: object


def selected(namelist):
    if isinstance(namelist, Mapping):
        value = namelist.get("rad_diag", namelist.get("gpuwrf", {}).get("rad_diag", 0))
    else:
        value = getattr(namelist, "rad_diag", 0)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("rad_diag restart needs a resolved domain scalar")
        value = value[0]
    if value not in (0, 1):
        raise ValueError("gpuwrf rad_diag supports only 0/1")
    return bool(value)


def initial(state):
    import jax.numpy as jnp
    zero = jnp.zeros(state.theta.shape, jnp.float32)
    return RadiationHistory(*(zero for _ in HISTORY_NAMES))


def history_fields(diagnostics):
    held = getattr(diagnostics, "rad_diag", None)
    return {} if held is None else dict(zip(HISTORY_NAMES, held, strict=True))


def validate_restart(diagnostics, namelist, version, shape=None):
    """Refuse old/incomplete checkpoints when ON; accept inactive old files."""
    import numpy as np
    held = getattr(diagnostics, "rad_diag", None)
    active = selected(namelist)
    if not active and held is None:
        return
    if not active or version != SCHEMA_VERSION or not isinstance(held, RadiationHistory):
        raise ValueError("restart rad_diag selection/schema is missing or differs (E78)")
    for name, value in zip(HISTORY_NAMES, held, strict=True):
        array = np.asarray(value)
        if (shape is not None and array.shape != tuple(shape)) or array.dtype != np.float32:
            raise ValueError(f"restart rad_diag {name} shape/REAL dtype differs (E78)")
        if not np.isfinite(array).all():
            raise ValueError(f"restart rad_diag {name} is nonfinite")
