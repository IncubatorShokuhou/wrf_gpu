"""Compare root_cause.py arms: constants intervention and canopy-sweep census (manager 04:05Z (b)).

A  const vs off must be byte-identical on every carry leaf (else the mechanism is NOT operand constness).
B  census runs (instrumented off/on): columns whose canopy Newton loop ran a different number of sweeps
   ("flip" columns), their canopy-temperature difference versus the WRF exit tolerance (|DTV| <= 0.01 K),
   and whether every column with a large ON-vs-OFF difference is a flip column.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def load(tag):
    meta = json.loads((a.dir / f"{tag}.json").read_text())
    with np.load(a.dir / f"{tag}.npz") as data:
        arrays = [data[f"L{i:04d}"] for i in range(len(meta["leaf_names"]))]
    return meta, dict(zip(meta["leaf_names"], arrays))


def ulps(x, y):
    if x.dtype == np.float32:
        kind, sign = np.int32, 0x7fffffff
    elif x.dtype == np.float64:
        kind, sign = np.int64, 0x7fffffffffffffff
    else:
        return None
    def ordered(v):
        bits = v.view(kind).astype(np.int64) if kind is np.int32 else v.view(kind)
        return np.where(bits < 0, -(bits & sign), bits)
    return np.abs(ordered(x).astype(np.float64) - ordered(y).astype(np.float64))


def table(reference, candidate):
    rows = []
    for name in reference:
        if name.startswith("CENSUS.") or name.startswith("NAMED."):
            continue
        x, y = reference[name], candidate[name]
        assert x.shape == y.shape and x.dtype == y.dtype, name
        if x.tobytes() == y.tobytes():
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        u = ulps(x, y)
        nz = x != 0
        rows.append({"leaf": name, "dtype": str(x.dtype), "cells": int(np.count_nonzero(x != y)),
                     "max_abs": float(d.max()), "max_ulps": None if u is None else float(u.max()),
                     "max_rel": float((d[nz] / np.abs(x[nz].astype(np.float64))).max()) if nz.any() else 0.0})
    return rows


report = {"dir": str(a.dir)}
off_m, off = load("off")
on_m, on = load("on")
const_m, const = load("const")
assert off_m["git_head"] == on_m["git_head"] == const_m["git_head"]
report["head"] = off_m["git_head"]
report["on_vs_off"] = table(off, on)
report["const_vs_off"] = table(off, const)
report["A_const_restores_off_bytes"] = not report["const_vs_off"]

if (a.dir / "off_census.npz").exists():
    _, offc = load("off_census")
    _, onc = load("on_census")
    s0, s1 = offc["CENSUS.w2_active_sweeps"], onc["CENSUS.w2_active_sweeps"]
    d0, d1 = offc["CENSUS.w2_last_dtv"], onc["CENSUS.w2_last_dtv"]
    tv0, tv1 = offc["NAMED.noahmp_land.tv"], onc["NAMED.noahmp_land.tv"]
    flip = s0 != s1
    tv_ulps = ulps(tv0, tv1)
    dtv = np.abs(tv0.astype(np.float64) - tv1.astype(np.float64))
    large = tv_ulps > 6
    report["B_census"] = {
        "columns": int(s0.size), "flip_columns": int(flip.sum()),
        "sweeps_off_histogram": {int(k): int(v) for k, v in zip(*np.unique(s0, return_counts=True))},
        "flip_sweep_pairs": sorted({(int(x), int(y)) for x, y in zip(s0[flip], s1[flip])}),
        "tv_large_columns": int(large.sum()), "tv_large_outside_flip": int((large & ~flip).sum()),
        "tv_max_ulps_non_flip": float(tv_ulps[~flip].max()) if (~flip).any() else 0.0,
        "tv_max_abs_K_flip": float(dtv[flip].max()) if flip.any() else 0.0,
        "flip_tv_within_last_sweep_dtv": bool(np.all(dtv[flip] <= np.maximum(d0, d1)[flip] + 1e-6)) if flip.any() else True,
        "flip_last_dtv_max_K": float(np.maximum(d0, d1)[flip].max()) if flip.any() else 0.0,
        "wrf_exit_tolerance_K": 0.01,
        "instrumented_on_vs_off": table(offc, onc)}
    report["B_mechanism_confirmed"] = bool(flip.any() and report["B_census"]["tv_large_outside_flip"] == 0
                                         and report["B_census"]["tv_max_ulps_non_flip"] <= 6
                                         and report["B_census"]["flip_tv_within_last_sweep_dtv"])
a.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("on_vs_off", "const_vs_off")}, indent=2, default=str)[:4000])
