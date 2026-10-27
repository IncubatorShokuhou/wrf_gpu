"""R0 control: the root-clock lag fit (lag148.py method, PSFC) on the CPU-vs-CPU pair 0420 C2 (4x1) vs C (2x2); a pair without the
GPU root-clock defect should give ~0 s on d01 rings 1-9. Guarded (A-O5-9NEST-148, group cpu_twin_149_0420). usage: python lag_0420.py <out.json>"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gread148 as g
T0 = datetime(2026, 4, 21, 0)  # WRF start = issue 20260420_18z + 6 h (first wrfout 2026-04-21_00:00:00)
GROUP = "cpu_twin_149_0420_C_2x2_vs_C2_4x1"


def lead_of(name):
    t = datetime.strptime(name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
    s = (t - T0).total_seconds(); L = int(round(s / 3600))
    return L if abs(s - 3600 * L) < 60 else None


listed, _ = g.admitted()
pairs = {}
for p, f in listed.items():
    if f.get("group") == GROUP and "/wrfout_" in p:
        name = Path(p).name; dom = name.split("_")[1]; L = lead_of(name)
        if dom in ("d01", "d02") and L is not None and L <= 13:
            pairs.setdefault((dom, L), {})[Path(p).parent.name] = p
pairs = {k: v for k, v in pairs.items() if set(v) == {"C", "C2"}}
import netCDF4
sess, rdir, _ = g.session(sorted({p for v in pairs.values() for p in v.values()}), "L0420c_lag_control")
psfc = {}
with sess:
    for (dom, L), v in pairs.items():
        for arm, p in v.items():
            with netCDF4.Dataset(p) as nc:
                psfc[(dom, L, arm)] = nc.variables["PSFC"][0].astype(np.float64)
g.close_receipt(sess, rdir, {"n_files": 2 * len(pairs)})
out = {}
for dom in ("d01", "d02"):
    ny, nx = psfc[(dom, 1, "C")].shape
    y, x = np.mgrid[0:ny, 0:nx]; rg = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))
    res = {}
    for name, m in [(f"ring{r}", rg == r) for r in range(10)] + [("interior", rg >= 10)]:
        d, t = [], []
        for L in range(1, 13):
            d.append((psfc[(dom, L, "C2")] - psfc[(dom, L, "C")])[m].mean())
            t.append(((psfc[(dom, L + 1, "C")] - psfc[(dom, L - 1, "C")]) / 7200.0)[m].mean())
        d, t = np.array(d), np.array(t)
        res[name] = {"lag_s": round(float((d * t).sum() / (t * t).sum()), 2), "mean_abs_diff_Pa": round(float(np.abs(d).mean()), 5)}
    out[dom] = res
    print(dom, " ".join(f"{k}:{v['lag_s']}s/{v['mean_abs_diff_Pa']}Pa" for k, v in res.items()))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
