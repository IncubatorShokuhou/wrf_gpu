"""Static Ferrier root/child ID from the actual WRF domains namelist."""
from dataclasses import replace


def bind_ferrier_domain(namelist, wrf_namelist, domain):
    if int(namelist.mp_physics) not in (5, 15):
        return namelist
    value = wrf_namelist.get("domains", {}).get("grid_id", 1)
    index = int(domain[1:]) - 1
    values = value if isinstance(value, (list, tuple)) else (value,)
    gid = int(values[index]) if index < len(values) else 1
    if gid < 1:
        raise ValueError("Ferrier requires a positive WRF grid_id")
    return replace(namelist, ferrier_gid=gid)
