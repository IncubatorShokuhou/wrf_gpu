"""Bitwise wrfout comparison of two run directories (every variable, every global/var attribute).

usage: python compare_wrfout.py DIR_A DIR_B OUT_JSON
Files are paired by name. A variable/attribute differing in any byte is reported.
Whole-file SHA equality is reported separately (it also covers header layout).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

a_dir, b_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = sorted(p.name for p in a_dir.glob("wrfout_*"))
b_names = sorted(p.name for p in b_dir.glob("wrfout_*"))
report = {"a": str(a_dir), "b": str(b_dir), "files_a": len(names), "files_b": len(b_names),
          "same_file_set": names == b_names, "files": {}}
total_diff = 0
for name in names:
    pa, pb = a_dir / name, b_dir / name
    entry = {"sha_equal": hashlib.sha256(pa.read_bytes()).digest() == hashlib.sha256(pb.read_bytes()).digest()}
    diffs = []
    with Dataset(pa) as da, Dataset(pb) as db:
        if set(da.variables) != set(db.variables):
            diffs.append({"variables": sorted(set(da.variables) ^ set(db.variables))})
        ga = {k: da.getncattr(k) for k in da.ncattrs()}
        gb = {k: db.getncattr(k) for k in db.ncattrs()}
        for key in sorted(set(ga) | set(gb)):
            if key not in ga or key not in gb or not np.array_equal(np.asarray(ga[key]), np.asarray(gb[key])):
                diffs.append({"global_attr": key})
        for var in sorted(set(da.variables) & set(db.variables)):
            va, vb = da.variables[var], db.variables[var]
            xa, xb = np.ma.getdata(va[:]), np.ma.getdata(vb[:])
            if xa.shape != xb.shape or xa.dtype != xb.dtype or xa.tobytes() != xb.tobytes():
                diffs.append({"var": var})
        entry["variables"] = len(da.variables)
    entry["diffs"] = diffs
    total_diff += len(diffs)
    report["files"][name] = entry
report["total_diffs"] = total_diff
report["all_sha_equal"] = all(e["sha_equal"] for e in report["files"].values())
out.write_text(json.dumps(report, indent=1, default=str))
print(json.dumps({k: report[k] for k in ("files_a", "files_b", "same_file_set", "total_diffs", "all_sha_equal")}))
