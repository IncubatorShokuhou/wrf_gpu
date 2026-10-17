"""Record an unmodified 2x2 surface crop from retained original CPU-WRF Swiss.

Reproduction only: no station observations are read. This is reader integration
evidence, not a forecast-fidelity or optimization-skill claim.
"""
from pathlib import Path
import json
import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.optimize.data import identity

BASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
OUT = Path(__file__).resolve().parent
sources = [BASE/f"wrfout_d01_2023-01-15_{hour:02d}:00:00" for hour in (11, 12)]
fields = ("XLAT", "XLONG", "LANDMASK", "T2", "SWDOWN", "RAINC", "RAINNC", "RAINSH")
arrays = {name: [] for name in fields}
times, attrs, units = [], {}, {}
for source in sources:
    with Dataset(source) as ds:
        times.append(np.asarray(ds.variables["Times"][0]))
        attrs = {name: ds.getncattr(name) for name in ("GRID_ID", "MAP_PROJ", "SIMULATION_START_DATE")}
        for name in fields:
            arrays[name].append(np.asarray(ds.variables[name][0, 20:22, 20:22]))
            units[name] = str(getattr(ds.variables[name], "units", ""))
np.savez_compressed(OUT/"cpu_wrf_surface.npz", Times=np.stack(times), **{k: np.stack(v) for k, v in arrays.items()})
receipt = {"kind": "original CPU-WRF Swiss 2x2 crop; reader integration only; no station truth",
           "sources": [identity(p) for p in sources], "slice": {"j": [20, 22], "i": [20, 22]},
           "attributes": {k: int(v) if isinstance(v, np.integer) else v for k, v in attrs.items()},
           "units": units, "fixture": identity(OUT/"cpu_wrf_surface.npz")}
(OUT/"cpu_wrf_surface.json").write_text(json.dumps(receipt, indent=2)+"\n")
