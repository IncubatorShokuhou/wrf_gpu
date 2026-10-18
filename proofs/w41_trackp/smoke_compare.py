"""Track P GPU smoke verdicts from the arm outputs (NumPy/netCDF only, no model code).

usage: python smoke_compare.py <work_dir> <out.json>
G1 no recompile: within each controls family the first successful arm compiles, every later one adds no cache file
G2 PRIOR == OFF: S1 (prior controls) vs S0 (released program) per variable/frame: bitwise count, max |diff|
G3 response: S2 (spec_T) vs S1 differs, static geography unchanged; LAI(T)/LAI(prior) == 1.3 from the first forecast
   hour on (t=0 = wrfinput LAI) on cells with table LAI (dveg 4 phenology = LAIM(month) interpolation, multiplier 1.3) to REAL round-off
G4 PROD: P1/P2 vs P0 finite, differ, all arms rc 0 on an asserted GPU
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
arms, order = {}, []
for line in (W / "arms.jsonl").read_text().splitlines():
    if line.startswith('{"arm"'):
        rec = json.loads(line)
        arms[rec["arm"]] = rec  # last record per arm wins
        order.append(rec)


def files(name, dom):
    return {p.name: p for p in sorted((W / "runs" / name).glob(f"wrfout_{dom}_*"))}


def variables(path):
    with Dataset(path) as ds:
        return {k: np.asarray(v[:]) for k, v in ds.variables.items() if k not in ("Times",) and v.dtype.kind == "f"}


def pair(a_name, b_name, dom):
    fa, fb = files(a_name, dom), files(b_name, dom)
    common = sorted(set(fa) & set(fb))
    per_var, finite = {}, True
    for t in common:
        va, vb = variables(fa[t]), variables(fb[t])
        for k in sorted(set(va) & set(vb)):
            a, b = va[k].astype(np.float64), vb[k].astype(np.float64)
            finite &= bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
            e = per_var.setdefault(k, {"frames": 0, "bitwise_frames": 0, "max_abs": 0.0})
            e["frames"] += 1
            e["bitwise_frames"] += int(np.array_equal(va[k], vb[k]))
            e["max_abs"] = max(e["max_abs"], float(np.nanmax(np.abs(a - b))) if a.size else 0.0)
    n_bit = sum(e["bitwise_frames"] == e["frames"] for e in per_var.values())
    return {"frames": len(common), "variables": len(per_var), "variables_bitwise_all_frames": n_bit,
            "finite": finite, "per_var": per_var}


res = {"arms": arms}
# per program family (Swiss controls S1/S2, PROD controls P*): the first successful arm compiles, every later
# successful arm must add no cache file (S0 = the released program, a different executable, is not in a family)
g1 = {}
for family in (("S1_swiss_prior", "S2_swiss_T"), ("P0_prod_prior", "P1_prod_A", "P2_prod_B")):
    ok = [r for r in order if r["arm"] in family and r.get("rc") == 0]
    for r in ok[1:]:
        g1[r["arm"]] = r.get("cache_files_after") == r.get("cache_files_before")
    if ok:
        g1["cold:" + ok[0]["arm"]] = "compiled here"
res["G1_no_recompile"] = g1
res["devices"] = sorted({v.get("device", "") for v in arms.values()})
if "S0_swiss_off" in arms and "S1_swiss_prior" in arms:
    res["G2_prior_vs_off"] = pair("S0_swiss_off", "S1_swiss_prior", "d01")
if "S1_swiss_prior" in arms and "S2_swiss_T" in arms:
    resp = pair("S1_swiss_prior", "S2_swiss_T", "d01")
    fp, ft = files("S1_swiss_prior", "d01"), files("S2_swiss_T", "d01")
    ratios = []
    # the t=0 history frame carries the wrfinput LAI (written before the first Noah step), so it is unedited by
    # construction (checked against original CPU-WRF on the edited MPTABLE); the gate starts at the first forecast hour
    for t in sorted(set(fp) & set(ft))[1:]:
        with Dataset(fp[t]) as a, Dataset(ft[t]) as b:
            lp, lt = np.asarray(a["LAI"][0], np.float64), np.asarray(b["LAI"][0], np.float64)
            m = lp > 0.05
            if m.any():
                r = lt[m] / lp[m]
                ratios.append({"time": t, "cells": int(m.sum()), "min": float(r.min()), "max": float(r.max())})
    resp["lai_ratio"] = ratios
    resp["lai_ratio_ok"] = bool(ratios) and all(abs(r["min"] - 1.3) < 1e-5 and abs(r["max"] - 1.3) < 1e-5
                                                for r in ratios)
    res["G3_T_vs_prior"] = {k: v for k, v in resp.items() if k != "per_var"} | {
        "changed": {k: v["max_abs"] for k, v in resp["per_var"].items() if v["max_abs"] > 0}}
for b in ("P1_prod_A", "P2_prod_B"):
    if "P0_prod_prior" in arms and b in arms:
        for dom in ("d01", "d02"):
            r = pair("P0_prod_prior", b, dom)
            res[f"G4_{b}_{dom}"] = {k: v for k, v in r.items() if k != "per_var"} | {
                "n_changed": sum(v["max_abs"] > 0 for v in r["per_var"].values())}
OUT.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k.startswith("G") and k != "G2_prior_vs_off"} |
                 {"G2": {k: v for k, v in res.get("G2_prior_vs_off", {}).items() if k != "per_var"}}, indent=1,
                 default=str)[:4000])
