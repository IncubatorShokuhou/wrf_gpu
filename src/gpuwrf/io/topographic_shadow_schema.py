"""Host restart admission for the optional WRF HGT_SHAD radiation leaf."""
from __future__ import annotations

SCHEMA_VERSION = 1


def configuration(namelist):
    from gpuwrf.io.restart_schema import physics_option
    return (physics_option(namelist, "topo_shading", 0) == 1
            and physics_option(namelist, "slope_rad", 0) == 1
            and physics_option(namelist, "ra_sw_physics", 4) == 4
            and physics_option(namelist, "ra_lw_physics", 4) == 4)


def marker(diagnostics):
    return SCHEMA_VERSION if getattr(diagnostics, "shadow_height_m", None) is not None else None


def validate(diagnostics, version, shape, *, namelist=None, required=None):
    import numpy as np
    value = getattr(diagnostics, "shadow_height_m", None)
    needed = configuration(namelist) if required is None else bool(required)
    if not needed and value is None and version is None:
        return
    if version != SCHEMA_VERSION or value is None:
        raise ValueError("restart topo_shading=1 lacks held HGT_SHAD/schema (E78)")
    value = np.asarray(value)
    if shape is None or value.shape != tuple(shape) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("restart HGT_SHAD shape/dtype/nonfinite differs (E78)")


__all__ = ["SCHEMA_VERSION", "configuration", "marker", "validate"]
