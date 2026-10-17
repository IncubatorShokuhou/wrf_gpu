"""Pre-device admission for QNSE (bl_pbl_physics=4) on sea-ice input.

WRF QNSEPBL reads SICE=xice (module_pbl_driver.F:1547) at land points:
SNOW>0 .OR. SICE>0.5 scales QFC1 by RLIVWV (module_bl_qnsepbl.F:488-499).
The port carries no XICE and no sea-ice surface path (WRF Noah-MP skips
XICE>=threshold points, module_sf_noahmpdrv.F:722-728, and the surface driver
runs seaice_noah, module_surface_driver.F:3364), so the QNSE adapter's SICE=0
is exact only on sea-ice-free input; nonzero aux4 SEAICE is refused separately
(io/lower_boundary.py).  Inland glaciers (IVGTYP=ISICE, SEAICE=0) stay admitted.

Called at every public source-loading boundary (build_replay_case,
build_noahmp_land_state) and per domain by both pipelines before any model work.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: Every name WRF input/history carries the sea-ice fraction under (Registry XICE, dataname SEAICE).
SEA_ICE_ALIASES = ("SEAICE", "XICE")


def domain_bl_pbl_physics(namelist: Mapping[str, Any] | None, domain: str, default: int = 5) -> int:
    """Per-domain ``&physics bl_pbl_physics`` (max_dom list or scalar)."""
    value = ((namelist or {}).get("physics") or {}).get("bl_pbl_physics", default)
    if isinstance(value, (list, tuple)):
        index = max(int(domain[1:]) - 1, 0)
        value = value[min(index, len(value) - 1)] if value else default
    return int(default if value is None else value)


def check_qnse_sea_ice_fields(domain: str, bl_pbl_physics, fields: Mapping[str, Any]) -> None:
    """Refuse QNSE when ANY present sea-ice alias is > 0 or non-finite, or the aliases disagree."""
    if int(bl_pbl_physics) != 4:
        return
    arrays = {key: np.asarray(fields[key], np.float64) for key in SEA_ICE_ALIASES if key in fields}
    for key, value in arrays.items():
        if not np.isfinite(value).all() or np.any(value > 0.0):
            raise ValueError(
                f"{domain}: bl_pbl_physics=4 (QNSE) with sea ice ({key}>0 or non-finite) is not supported: "
                "WRF passes SICE=XICE to QNSEPBL and runs seaice_noah; the port has neither")
    if len(arrays) > 1:
        first, *others = arrays.values()
        if any(o.shape != first.shape or not np.array_equal(o, first) for o in others):
            raise ValueError(
                f"{domain}: bl_pbl_physics=4 (QNSE) with conflicting sea-ice fields "
                f"{'/'.join(arrays)}: refusing an ambiguous XICE")


def _netcdf_sea_ice(path) -> dict[str, Any]:
    from gpuwrf.io.netcdf_lock import Dataset
    with Dataset(str(path)) as ds:
        return {key: np.asarray(ds[key][:1]) for key in SEA_ICE_ALIASES if key in ds.variables}


def refuse_qnse_sea_ice(run, domain: str, bl_pbl_physics=None, *, history_source=None) -> None:
    """Raise before any model/device work if a QNSE domain's input carries sea ice.

    Reads every present alias from ``wrfinput_<domain>`` (an absent file holds no
    sea ice; the loaders themselves require it) and, for a replay initial state,
    from ``history_source`` (first time level).
    """
    if bl_pbl_physics is None:
        try:
            namelist = run.namelist
        except FileNotFoundError:
            return   # no namelist: no QNSE selection to admit
        bl_pbl_physics = domain_bl_pbl_physics(namelist, domain)
    if int(bl_pbl_physics) != 4:
        return
    try:
        names = set(run.wrfinput_variables(domain))
    except FileNotFoundError:
        names = set()
    check_qnse_sea_ice_fields(domain, bl_pbl_physics,
                              {key: run.load_wrfinput(domain, key, lazy=False) for key in SEA_ICE_ALIASES if key in names})
    if history_source is not None:
        check_qnse_sea_ice_fields(domain, bl_pbl_physics, _netcdf_sea_ice(history_source))
