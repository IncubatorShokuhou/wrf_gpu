"""w42-bundle042: per-VARIABLE byte comparison of two wrfout sets (CPU, netCDF4 only; JAX is never imported).

usage: python3 var_diff.py DIR_A DIR_B OUT_JSON [GLOB=wrfout_d0*]
For every file name present in both directories: every variable's dtype, shape and raw values (np.array_equal, NaN-equal),
every variable attribute and every global attribute are compared. Output: per variable the number of files where it differs
(+ first file, max |a-b| for numeric data), differing global/variable attributes, files only in one set, and
'only_vars_differ' = the set of differing variables. Use: v0.4.2 (N) vs v0.4.1 (O) RW3T arms of the same inputs -> the
tuple's "byte delta vs v0.4.1 = history COSZEN on non-initial frames only" claim (GPU-vs-GPU: release diff, not fidelity).
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

A, B, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
GLOB = sys.argv[4] if len(sys.argv) > 4 else "wrfout_d0*"
fa = {p.name: p for p in sorted(A.glob(GLOB))}
fb = {p.name: p for p in sorted(B.glob(GLOB))}
common = sorted(set(fa) & set(fb))
var_diff, gattr_diff, vattr_diff, struct = {}, {}, {}, []


def attrs(obj):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ((k, obj.getncattr(k)) for k in obj.ncattrs())}


for name in common:
    with netCDF4.Dataset(fa[name]) as da, netCDF4.Dataset(fb[name]) as db:
        for d in (da, db):
            d.set_auto_maskandscale(False)
        ga, gb = attrs(da), attrs(db)
        for k in sorted(set(ga) | set(gb)):
            if str(ga.get(k)) != str(gb.get(k)):
                gattr_diff.setdefault(k, {"files": 0, "first": name, "a": str(ga.get(k))[:80], "b": str(gb.get(k))[:80]})["files"] += 1
        va, vb = set(da.variables), set(db.variables)
        if va != vb:
            struct.append({"file": name, "only_a": sorted(va - vb), "only_b": sorted(vb - va)})
        for v in sorted(va & vb):
            xa, xb = da.variables[v], db.variables[v]
            if attrs(xa) != attrs(xb):
                vattr_diff.setdefault(v, {"files": 0, "first": name})["files"] += 1
            a, b = xa[...], xb[...]
            same = a.dtype == b.dtype and a.shape == b.shape and (
                np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b))
            if not same:
                rec = var_diff.setdefault(v, {"files": 0, "first": name, "max_abs": 0.0})
                rec["files"] += 1
                if a.shape == b.shape and a.dtype.kind in "fiu":
                    with np.errstate(invalid="ignore"):
                        m = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))) if a.size else 0.0
                    rec["max_abs"] = max(rec["max_abs"], float(m))
res = {"dir_a": str(A), "dir_b": str(B), "common_files": len(common), "only_a": sorted(set(fa) - set(fb)),
       "only_b": sorted(set(fb) - set(fa)), "var_diff": var_diff, "only_vars_differ": sorted(var_diff),
       "global_attr_diff": gattr_diff, "var_attr_diff": vattr_diff, "structure_diff": struct}
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({"common_files": len(common), "only_vars_differ": sorted(var_diff),
                  "files_per_var": {k: v["files"] for k, v in var_diff.items()}, "global_attrs_differ": sorted(gattr_diff),
                  "var_attrs_differ": sorted(vattr_diff), "structure_diff_files": len(struct)}))
