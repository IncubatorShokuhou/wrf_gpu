"""Value identity of two wrfout sets (CPU only): for every common frame of DOMAIN, every variable's raw bytes equal?
Global attributes compared separately (a root-only run may differ only in header attributes).
usage: JAX_PLATFORMS=cpu python3 frames_equal.py OUT_JSON DIR_A DIR_B DOMAIN [MAX_LEAD_FILES]
"""
import glob, json, os, sys

import numpy as np
from netCDF4 import Dataset

out, a, b, dom = sys.argv[1:5]
lim = int(sys.argv[5]) if len(sys.argv) > 5 else None
names = sorted({os.path.basename(p) for p in glob.glob(f"{a}/wrfout_{dom}_*")} & {os.path.basename(p) for p in glob.glob(f"{b}/wrfout_{dom}_*")})
names = names[:lim] if lim else names
rows, first_diff = [], None
for n in names:
    with Dataset(f"{a}/{n}") as x, Dataset(f"{b}/{n}") as y:
        vx, vy = set(x.variables), set(y.variables)
        diff = []
        for v in sorted(vx & vy):
            p, q = np.asarray(x[v][:]), np.asarray(y[v][:])
            if p.shape != q.shape or p.tobytes() != q.tobytes():
                d = float(np.nanmax(np.abs(p.astype(float) - q.astype(float)))) if p.shape == q.shape and np.issubdtype(p.dtype, np.number) else None
                diff.append({"var": v, "max_abs_diff": d})
        ax, ay = {k: str(x.getncattr(k)) for k in x.ncattrs()}, {k: str(y.getncattr(k)) for k in y.ncattrs()}
        attr_diff = sorted(k for k in set(ax) | set(ay) if ax.get(k) != ay.get(k))
    rows.append({"file": n, "vars_only_a": sorted(vx - vy), "vars_only_b": sorted(vy - vx), "vars_differing": len(diff),
                 "differing": diff[:20], "global_attr_diff": attr_diff})
    if diff and first_diff is None:
        first_diff = n
res = {"a": a, "b": b, "domain": dom, "frames": len(rows), "frames_values_identical": sum(r["vars_differing"] == 0 and not r["vars_only_a"] and not r["vars_only_b"] for r in rows),
       "first_value_diff_frame": first_diff, "rows": rows}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("domain", "frames", "frames_values_identical", "first_value_diff_frame")}),
      "attr diffs (first frame):", rows[0]["global_attr_diff"] if rows else None)
