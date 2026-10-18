"""WRF ``aer_opt`` admission (pure Python, no JAX): which radiation-aerosol options this port runs.

aer_opt is a single-entry &physics key (Registry default 0, one value for every domain).  Implemented:
``aer_opt=1`` = the WRF Tegen/ECMWF six-type monthly climatology fed to RRTMG SW
(physics/wrf_tegen_aerosol.py; module_radiation_driver.F aer_time_int/aer_p_int, RRTMG_SWRAD iaer=6).
``aer_opt=3`` = the Thompson mp28 water/ice-friendly aerosol optics (physics/wrf_aer_opt3.py; gt_aod +
calc_aerosol_rrtmg_sw, RRTMG_SWRAD iaer=10).  WRF reads them only in RRTMG SW (and FARMS, which the port
refuses), and aer_opt=3 only when QNWFA exists (mp28), so every other combination would silently run
without aerosol -> refused.  The longwave scheme has no aerosol outside WRF-Chem.
"""

from __future__ import annotations

from typing import Any, Mapping

IMPLEMENTED_AER_OPT = (0, 1, 3)


def selected_aer_opt(wrf_namelist: Mapping[str, Any]) -> int:
    from gpuwrf.io.namelist_binding import namelist_value

    value = namelist_value(wrf_namelist, "aer_opt", "d01")
    return 0 if value is None else int(value)


def aerosol_refusals(wrf_namelist: Mapping[str, Any], domains: tuple[str, ...]) -> list[str]:
    from gpuwrf.io.namelist_binding import namelist_value

    aer_opt = selected_aer_opt(wrf_namelist)
    if aer_opt == 0:
        return []
    if aer_opt not in IMPLEMENTED_AER_OPT:
        return [f"&physics aer_opt={aer_opt}: only aer_opt=0, 1 (Tegen climatology) and 3 (Thompson mp28 "
                "aerosols) in RRTMG SW are implemented; aer_opt=2 (aer_type/AOD550 inputs) is not"]
    errors = []
    for domain in domains:
        sw = namelist_value(wrf_namelist, "ra_sw_physics", domain)
        if sw is None or int(sw) != 4:
            errors.append(f"{domain}: aer_opt={aer_opt} feeds only RRTMG SW (ra_sw_physics=4); ra_sw_physics={sw} "
                          "would run without aerosol")
        if aer_opt == 3:
            mp = namelist_value(wrf_namelist, "mp_physics", domain)
            if mp is None or int(mp) != 28:
                errors.append(f"{domain}: aer_opt=3 needs mp_physics=28 (QNWFA/QNIFA); mp_physics={mp} "
                              "would run without aerosol")
    if aer_opt == 3 and len(domains) > 1:
        errors.append("aer_opt=3 is qualified for single domains only (as the port's mp_physics=28)")
    return errors


__all__ = ["IMPLEMENTED_AER_OPT", "aerosol_refusals", "selected_aer_opt"]
