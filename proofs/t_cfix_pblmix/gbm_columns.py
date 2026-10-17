#!/usr/bin/env python3
"""t-cfix-pblmix F17: GBM operands on real cloudy CPU-WRF c08 (mp=10 + GBM bl=12) columns.

real_columns.py profiles/surface + real TKE_PBL (k=1..kx of the Z-staggered history
field).  RTHRATEN is not a history field: a CONSTRUCTED radiative heating profile
is used (disclosed): -2.5e-4 K/s (-0.9 K/h) at every cloud-top level (qc>1e-6 with
qc<=1e-6 above) and -2e-5 K/s elsewhere inside cloud, +5e-6 K/s in clear air.
Usage: gbm_columns.py <wrfout[,...]> <ncol/file> <out.txt> <out.json>
"""
import json
import sys

import netCDF4
import numpy as np

from real_columns import FIELDS2, extract

PROF = ("U", "V", "T", "QV", "QC", "P", "PI", "DZ", "TKE", "RAD")
SCAL = ("PSFC", "ZNT", "UST", "HFX", "QFX", "TSK", "GZ1OZ0", "WSPD", "BR", "PSIM", "PSIH", "XLAND")


def rthraten(qc):
    cloud = qc > 1e-6
    above = np.concatenate([cloud[1:], [False]])
    top = cloud & ~above
    return np.where(top, -2.5e-4, np.where(cloud, -2.0e-5, 5.0e-6)).astype(np.float32).astype(np.float64)


def build(path, ncol):
    d = extract(path, ncol)
    with netCDF4.Dataset(path) as nc:
        tke = np.asarray(nc.variables["TKE_PBL"][0], np.float64)
    d["columns"]["TKE"] = [np.float32(np.maximum(tke[: d["kx"], j, i], 1e-3)).astype(np.float64).tolist()
                           for j, i in zip(d["j"], d["i"])]
    d["columns"]["RAD"] = [rthraten(np.asarray(q)).tolist() for q in d["columns"]["QC"]]
    return d


if __name__ == "__main__":
    files, ncol, out_txt, out_json = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    parts = [build(f, ncol) for f in files]
    keys = set(parts[0]["columns"])
    data = dict(source=files, j=sum((p["j"] for p in parts), []), i=sum((p["i"] for p in parts), []),
                kx=parts[0]["kx"], columns={k: sum((p["columns"][k] for p in parts), []) for k in keys},
                scalars={k: sum((p["scalars"][k] for p in parts), []) for k in FIELDS2})
    n, kx = len(data["j"]), data["kx"]
    with open(out_txt, "w") as f:
        f.write(f"{n} {kx}\n")
        for c in range(n):
            for name in PROF:
                f.write(" ".join(f"{v:.17e}" for v in data["columns"][name][c]) + "\n")
            f.write(" ".join(f"{data['scalars'][k][c]:.17e}" for k in SCAL) + "\n")
    json.dump(data, open(out_json, "w"))
    print(n, "columns")
