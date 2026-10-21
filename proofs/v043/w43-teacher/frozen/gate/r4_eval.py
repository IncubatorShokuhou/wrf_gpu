"""R4 cell-watch data per case, captured while the GPU frames still exist (manager 2026-10-06T05:29:12Z + 05:31:13Z, same semantics as
integrate RC33P/r4_watch.py): for every frame the frozen wn3_integrity checks (frames_checked of the integrity JSON, else GPU frames
0/36/last) and every R4 field F in ECAN EVC CANLIQ CANWAT FWET, over the cells where CPU F != 0: CPU CANLIQ max, CPU CANWAT max, CPU and
GPU hourly rain max (RAINNC+RAINC increment from the PREVIOUS GPU frame), GPU CANLIQ / CANWAT max. integ_classes.classify combines the
fields that are degenerate in that frame (max over the union of cells = max of the per-field maxima). Non-finite values raise.
Usage: r4_eval.py --cpu-dir C --gpu-dir G --out OUT.json [--integrity INTEGRITY.json]"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

R4 = ("ECAN", "EVC", "CANLIQ", "CANWAT", "FWET")
NAMES = R4 + ("RAINNC", "RAINC")


def read(path):
    with Dataset(path) as ds:
        out = {}
        for n in NAMES:
            v = np.ma.asarray(ds[n][0], dtype=np.float64).filled(np.nan)
            if not np.isfinite(v).all():
                raise ValueError(f"non-finite {n} in {path}")
            out[n] = v
    return out


ap = argparse.ArgumentParser(); ap.add_argument("--cpu-dir", type=Path, required=True); ap.add_argument("--gpu-dir", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--integrity", type=Path); a = ap.parse_args()
integ = json.loads(a.integrity.read_text()) if a.integrity else None
res = {"rule": "manager 2026-10-06T05:29:12Z + 05:31:13Z R4 cell watch (bounds 1e-2 mm; GPU canopy dried)", "cpu_dir": str(a.cpu_dir),
       "gpu_dir": str(a.gpu_dir), "frames": {}}
for dom in ("d01", "d02", "d03"):
    gfiles = sorted(p.name for p in a.gpu_dir.glob(f"wrfout_{dom}_*") if not p.name.endswith(".tmp"))
    if integ and dom in integ.get("domains", {}):
        checked = [os.path.basename(f) for f in integ["domains"][dom]["frames_checked"]]
    else:
        checked = [gfiles[0], gfiles[36], gfiles[-1]]
    for name in checked:
        i = gfiles.index(name); key = f"{dom} {name[11:24]}"
        if i == 0:
            res["frames"][key] = {"prev": None}
            continue
        prev = gfiles[i - 1]
        c, cp, g, gp = read(a.cpu_dir / name), read(a.cpu_dir / prev), read(a.gpu_dir / name), read(a.gpu_dir / prev)
        rc = c["RAINNC"] + c["RAINC"] - cp["RAINNC"] - cp["RAINC"]; rg = g["RAINNC"] + g["RAINC"] - gp["RAINNC"] - gp["RAINC"]
        per = {}
        for f in R4:
            cells = c[f] != 0
            n = int(cells.sum())
            per[f] = {"cells": n} if n == 0 else {
                "cells": n, "cpu_canliq_max": float(c["CANLIQ"][cells].max()), "cpu_canwat_max": float(c["CANWAT"][cells].max()),
                "cpu_rain_1h_max": float(rc[cells].max()), "gpu_rain_1h_max": float(rg[cells].max()),
                "gpu_canliq_max": float(g["CANLIQ"][cells].max()), "gpu_canwat_max": float(g["CANWAT"][cells].max())}
        res["frames"][key] = {"prev": prev[11:24], "fields": per}
tmp = a.out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, a.out)
print(a.out, len(res["frames"]), "frames")
