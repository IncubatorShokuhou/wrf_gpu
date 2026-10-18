"""WRF's own topo_wind effect (CPU-WRF tw!=0 vs the admitted tw=0 composite reference), lane w41-topowind.
usage: python wrf_effect.py <tw_cpu_dir> <ref_cpu_dir> <out_json>"""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4
tw, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
for f in sorted(tw.glob("wrfout_d01_*")):
    g = ref / f.name
    if not g.exists():
        continue
    with netCDF4.Dataset(f) as a, netCDF4.Dataset(g) as b:
        land = np.asarray(a["XLAND"][0]) < 1.5
        ws = lambda d: np.hypot(np.asarray(d["U10"][0], np.float64), np.asarray(d["V10"][0], np.float64))
        wa, wb = ws(a), ws(b)
        rows.append(dict(time=f.name[11:], land_ws10_tw=float(wa[land].mean()), land_ws10_ref=float(wb[land].mean()),
                         ws10_rmse=float(np.sqrt(np.mean((wa - wb) ** 2))),
                         t2_rmse=float(np.sqrt(np.mean((np.asarray(a["T2"][0], np.float64) - np.asarray(b["T2"][0])) ** 2))),
                         ust_land_tw=float(np.asarray(a["UST"][0])[land].mean()), ust_land_ref=float(np.asarray(b["UST"][0])[land].mean())))
summary = dict(frames=len(rows), mean_land_ws10_change=float(np.mean([r["land_ws10_tw"] - r["land_ws10_ref"] for r in rows[1:]])),
               max_ws10_rmse=max(r["ws10_rmse"] for r in rows), rows=rows)
out.write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
