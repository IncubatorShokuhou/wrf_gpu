"""Guarded extraction of compact derived arrays for one domain (CPU reference vs GPU v0.4.2 warm72).

usage: python analysis/extract.py <dom> <label>
Reads every paired frame of <dom> inside ONE GuardedRead session (admission A-148) and writes
analysis/data/<dom>.npz (untracked): per-frame 2-D fields, W (3-D, d02..d09 only), hydrometeor column maxima,
and statics. Pairing: same valid time; CPU d01 names carry +18/+36/+54 s step offsets -> pair by nearest
minute (as the scorer's --init pairing does for d01; here d01 is only used for timelines).
"""
import os, re, sys, glob, json
from datetime import datetime
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread

S = "<DATA_ROOT>/alisios/scratch"
CPU_REF = S + "/n148-cpuref-20261018_12z/cpu_reference_a2_a3"
GPU = S + "/n148-gpu-v042/runs/warm72/wrfout"
T0 = datetime(2026, 10, 18, 18)
F2D = ["RAINNC", "RAINC", "U10", "V10", "T2", "Q2", "PSFC", "GRAUPELNC", "SNOWNC", "QRAINXY", "HFX", "LH", "SWDOWN"]
FCOL = ["QRAIN", "QNRAIN", "QGRAUP", "QCLOUD", "QSNOW", "QICE"]   # column max per cell
STATIC = ["HGT", "LANDMASK", "XLAT", "XLONG"]


def lead_of(name):
    t = datetime.strptime(name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
    return int(round((t - T0).total_seconds() / 3600.0))


def files(dom):
    cpu = {lead_of(os.path.basename(p)): os.path.realpath(p) for p in glob.glob(f"{CPU_REF}/wrfout_{dom}_*")}
    gpu = {lead_of(os.path.basename(p)): os.path.realpath(p) for p in glob.glob(f"{GPU}/wrfout_{dom}_*")}
    leads = sorted(set(cpu) & set(gpu))
    return leads, cpu, gpu


def main(dom, label):
    import netCDF4
    leads, cpu, gpu = files(dom)
    targets = [cpu[L] for L in leads] + [gpu[L] for L in leads]
    gread.use_add()
    sess, rdir = gread.session(targets, label)
    want_w = dom != "d01"
    out = {}
    with sess:
        for side, src in (("cpu", cpu), ("gpu", gpu)):
            acc = {f: [] for f in F2D + [c + "_colmax" for c in FCOL] + (["W"] if want_w else [])}
            for L in leads:
                with netCDF4.Dataset(src[L]) as nc:
                    nc.set_auto_mask(False)
                    for f in F2D:
                        acc[f].append(nc.variables[f][0].astype(np.float32) if f in nc.variables else None)
                    for c in FCOL:
                        acc[c + "_colmax"].append(nc.variables[c][0].max(axis=0).astype(np.float32) if c in nc.variables else None)
                    if want_w:
                        acc["W"].append(nc.variables["W"][0].astype(np.float32))
                    if side == "gpu" and L == leads[0]:
                        for s in STATIC:
                            out[s] = nc.variables[s][0].astype(np.float32)
            for f, v in acc.items():
                if all(x is not None for x in v):
                    out[f"{side}_{f}"] = np.stack(v)
                else:
                    out[f"{side}_{f}_missing"] = np.array([x is None for x in v])
    gread.close_receipt(sess, rdir)
    out["leads"] = np.array(leads)
    (HERE / "data").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "data" / f"{dom}.npz", **out)
    json.dump({"dom": dom, "label": label, "n_leads": len(leads), "n_files": len(targets),
               "cpu_files": [cpu[L] for L in leads], "gpu_files": [gpu[L] for L in leads]},
              open(rdir / "files.json", "w"), indent=0)
    print(dom, "leads", len(leads), "files", len(targets))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
