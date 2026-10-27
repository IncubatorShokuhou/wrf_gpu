"""CPU-vs-CPU decomposition floor on a convective day (#149 0420, WN3 3-nest, original CPU-WRF 4.7.1 C 2x2 vs C2 4x1, 72 h).

Guarded (A-O5-9NEST-148, group cpu_twin_149_0420_C_2x2_vs_C2_4x1). Per domain and lead: C = RMSE(C2 - C) like the D6 scorer
(2-D fields all cells, W all levels), bias, plus RAINNC domain means and cells >= 10 mm. Frames paired by nearest hour
(d01 native dt54 alarm times). Values stay in memory; the JSON holds derived statistics only.

usage: JAX_PLATFORMS=cpu python cpu_twin_0420.py <out.json>
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gread148 as g  # noqa: E402

GROUP = "cpu_twin_149_0420_C_2x2_vs_C2_4x1"
T0 = datetime(2026, 4, 21, 0)  # WRF start = issue 20260420_18z + 6 h (first wrfout 2026-04-21_00:00:00)
FIELDS = ["W", "U10", "V10", "RAINNC", "T2", "PSFC"]


def lead_of(name):
    t = datetime.strptime(name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
    s = (t - T0).total_seconds()
    L = int(round(s / 3600))
    return L if abs(s - 3600 * L) < 60 else None


def main(out_path):
    import netCDF4
    listed, _ = g.admitted()
    files = [p for p, f in listed.items() if f.get("group") == GROUP and "/wrfout_" in p]
    pairs = {}
    for p in files:
        arm = Path(p).parent.name  # C or C2
        name = Path(p).name
        dom, L = name.split("_")[1], lead_of(name)
        if L is not None:
            pairs.setdefault((dom, L), {})[arm] = p
    pairs = {k: v for k, v in pairs.items() if set(v) == {"C", "C2"}}
    sess, rdir, _ = g.session(sorted({p for v in pairs.values() for p in v.values()}), sys.argv[2] if len(sys.argv) > 2 else "T0420_cpu_twin")
    out = {}
    with sess:
        for (dom, L) in sorted(pairs):
            a = {}
            for arm, p in pairs[(dom, L)].items():
                with netCDF4.Dataset(p) as nc:
                    nc.set_auto_mask(False)
                    a[arm] = {f: nc.variables[f][0].astype(np.float64) for f in FIELDS}
            row = {}
            for f in FIELDS:
                d = a["C2"][f] - a["C"][f]
                if not (np.isfinite(a["C"][f]).all() and np.isfinite(a["C2"][f]).all()):
                    raise ValueError(f"non-finite {dom} {L} {f}")
                x, y = a["C"][f].ravel(), a["C2"][f].ravel()
                r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
                row[f] = {"rmse": float(np.sqrt(np.mean(d * d))), "bias": float(d.mean()), "pearson_r": r,
                          "std_C": float(x.std()), "std_C2": float(y.std())}
            r = a["C"]["RAINNC"]
            r2 = a["C2"]["RAINNC"]
            row["rain"] = {"mean_C": float(r.mean()), "mean_C2": float(r2.mean()), "ge10_C": int((r >= 10).sum()),
                           "ge10_C2": int((r2 >= 10).sum()), "max_C": float(r.max()), "max_C2": float(r2.max())}
            out.setdefault(dom, {})[L] = row
    g.close_receipt(sess, rdir, {"n_pairs": len(pairs)})
    Path(out_path).write_text(json.dumps(out, indent=1))
    for dom in sorted(out):
        for L in (1, 3, 6, 12, 17, 24, 36, 48, 60, 72):
            if L in out[dom]:
                r = out[dom][L]
                print(dom, L, " ".join(f"{f}={r[f]['rmse']:.3g}" for f in FIELDS), "mean", round(r["rain"]["mean_C"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
