"""Frame-by-frame RMSE of port wrfout vs CPU-WRF wrfout (same valid times), selected fields.

usage: python compare_forecast.py <port_dir> <wrf_dir> <out.json> [label]
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

FIELDS = ("T2", "U10", "V10", "PSFC", "Q2", "THM", "U", "V", "W", "QVAPOR", "PH", "MU", "TSK")


def main() -> None:
    port, wrf, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = {}
    for f in sorted(port.glob("wrfout_d01_*")):
        ref = wrf / f.name
        if not ref.exists():
            continue
        with netCDF4.Dataset(f) as a, netCDF4.Dataset(ref) as b:
            rows[f.name[11:]] = {
                v: float(np.sqrt(np.mean((np.asarray(a.variables[v][0], np.float64)
                                          - np.asarray(b.variables[v][0], np.float64)) ** 2)))
                for v in FIELDS if v in a.variables and v in b.variables
            }
    out.write_text(json.dumps(rows, indent=1))
    for t, r in rows.items():
        print(t, " ".join(f"{k}:{v:.3g}" for k, v in r.items()))


if __name__ == "__main__":
    main()
