"""Forecast-level Track P oracle: does a traced control move the GPU forecast like the same MPTABLE edit moves
original CPU-WRF?  (NumPy/netCDF only.)

usage: python response_compare.py <gpu_off_dir> <gpu_ctl_dir> <cpu_ref_dir> <cpu_edit_dir> <out.json> [domain]
Per paired hourly frame and field: dG = GPU(ctl) - GPU(off), dC = CPU(edit) - CPU(ref) over LAND cells;
reports pattern correlation corr(dG, dC), RMS ratio rms(dG)/rms(dC), and rms(dG - dC) next to the native
GPU-vs-CPU difference rms(GPU(off) - CPU(ref)) (the noise floor the response must stand above).
Static-response fields (ZNT/ALBEDO/EMISS/LAI) should match tightly; for chaotic fields the RMS ratio ~1 and the
noise-scale ratio rms(dG-dC)/(sqrt2*rms(GPU(off)-CPU(ref))) ~1 says the response difference is of native-noise size
(descriptive; medians over hours, single hours vary).
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

FIELDS = ("ZNT", "LAI", "EMISS", "ALBEDO", "TSK", "T2", "HFX", "LH", "U10", "V10", "UST")


def frames(d, dom):
    return {p.name.split("_", 2)[2]: p for p in sorted(Path(d).glob(f"wrfout_{dom}_*"))}


def read(path, name):
    with Dataset(path) as ds:
        if name not in ds.variables:
            return None
        return np.asarray(ds.variables[name][0], np.float64)


def main(go, gc, cr, ce, out, dom="d01"):
    sets = [frames(x, dom) for x in (go, gc, cr, ce)]
    times = sorted(set.intersection(*(set(s) for s in sets)))
    if not times:
        raise SystemExit("no common frames")
    land = read(sets[2][times[0]], "LANDMASK") > 0.5
    result = {"frames": len(times), "land_cells": int(land.sum()), "fields": {}}
    for name in FIELDS:
        rows = []
        for t in times:
            a, b, c, e = (read(s[t], name) for s in sets)
            if any(x is None for x in (a, b, c, e)):
                break
            dg, dc = (b - a)[land], (e - c)[land]
            floor = float(np.sqrt(np.mean(((a - c)[land]) ** 2)))
            rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
            corr = float(np.corrcoef(dg, dc)[0, 1]) if dg.std() > 0 and dc.std() > 0 else None
            rows.append({"time": t, "corr": corr, "rms_gpu_response": rms(dg), "rms_cpu_response": rms(dc),
                         "rms_response_difference": rms(dg - dc), "rms_native_gpu_minus_cpu": floor})
        if rows:
            corr = [r["corr"] for r in rows if r["corr"] is not None]
            result["fields"][name] = {
                "median_corr": float(np.median(corr)) if corr else None,
                "min_corr": float(np.min(corr)) if corr else None,
                "median_rms_ratio": float(np.median([r["rms_gpu_response"] / r["rms_cpu_response"]
                                                     for r in rows if r["rms_cpu_response"] > 0] or [np.nan])),
                # rms(dG - dC) / (sqrt 2 * native GPU-CPU rms): ~1 = the response difference has the scale of two
                # independent native-noise realisations (compatible with noise, not a proof of origin); >> 1 = suspect
                "median_noise_explained_ratio": float(np.median([
                    r["rms_response_difference"] / (np.sqrt(2.0) * r["rms_native_gpu_minus_cpu"])
                    for r in rows if r["rms_native_gpu_minus_cpu"] > 0] or [np.nan])),
                "per_frame": rows}
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("median_corr", "min_corr", "median_rms_ratio", "median_noise_explained_ratio")}
                      for k, v in result["fields"].items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6], *(sys.argv[6:7] or ["d01"]))
