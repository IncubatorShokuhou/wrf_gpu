"""WRF history fields whose surface-driver producer is disabled by sfclay=0."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

# module_surface_driver.F:1490 returns before PSFC=p8w (:1988) and before
# the sfclay/LSM writes of these diagnostics. They keep their initialized values,
# even though the dynamical pressure, temperature and wind continue to evolve.
HELD_SURFACE_FIELDS = ("PSFC", "T2", "Q2", "U10", "V10", "TH2")


def load_disabled_surface_history(path: Path, shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Read the initialized host fields once; absent TH2 is allocator zero."""
    fields = {}
    with Dataset(path, "r") as initial:
        if shape is None:
            shape = (len(initial.dimensions["south_north"]), len(initial.dimensions["west_east"]))
        for name in HELD_SURFACE_FIELDS:
            if name == "TH2" and name not in initial.variables:
                fields[name] = np.zeros(shape, dtype=np.float32)
                continue
            variable = initial.variables[name]
            value = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
            value = np.asarray(np.ma.filled(value, np.nan))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{path}:{name} is not a finite initialized surface field of shape {shape}")
            fields[name] = value
    return fields
