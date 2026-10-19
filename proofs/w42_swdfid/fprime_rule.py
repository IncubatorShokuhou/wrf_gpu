"""w42-swdfid PRE-REGISTERED F' decision rule (#146 0420; frozen before the F' GPU run, sha posted to MAILBOX).

Question (ALISIOS Q2): is F's colder / cloudier land in the 04-22 and 04-23 afternoons a SYSTEMATIC effect of the SWDDIR
fix or a chance realization?  F' = a second realization of F (same source/pins as F, fresh cold cache).

Data: hourly wrfout of the arms B, Bp (v0.4.0 x2) and F, Fp (fix x2) under RUNS/<arm>/gpu, domains d02 and d03, frames
12:00..17:00Z (6 frames) of 2026-04-22 and 2026-04-23.  Land = LANDMASK > .5 of B's first frame (same cells in every arm).
Per cell c = (domain, afternoon) [4 cells] and arm X:
    T(X, c)  = mean over the 6 frames of the land-mean T2 [K]
    K(X, c)  = mean over the 6 frames of the land-mean clearness SWDOWN/SWDNT over cells with SWDNT > 50 W/m2
    D_X(c)   = T(X, c) - (T(B, c) + T(Bp, c)) / 2          (fix-arm deviation from the v0.4.0 pair)
    N(c)     = |T(B, c) - T(Bp, c)| / 2                       (noise half-spread of the v0.4.0 pair)
    DK_X(c), NK(c) likewise for K.
Verdict on Fp (F's own values are reported for reference; F is NOT re-judged):
    REPEATS        if  #{c : D_Fp(c) < 0} >= 3   and   #{c : D_Fp(c) < -N(c)} >= 3   and   sum_c D_Fp(c) <= 0.5 * sum_c D_F(c)
    CHANCE         if  #{c : D_Fp(c) < 0} <= 1   or    sum_c D_Fp(c) >= 0
    INCONCLUSIVE   otherwise
Secondary, descriptive only (no verdict): the same counts for DK (cloudier = DK < 0), and every number per cell.
REPEATS supports a systematic tendency (then: mechanism study / more cases; under pure chance REPEATS still has a few-10 %
probability because cells are correlated).  CHANCE means F's afternoons are not reproduced by a second F realization.

usage: python fprime_rule.py RUNS_DIR OUT_JSON [--pre]   (--pre: Fp absent, record F/B/Bp reference values only)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

DOMAINS = ("d02", "d03")
DAYS = ("2026-04-22", "2026-04-23")
HOURS = (12, 13, 14, 15, 16, 17)


def land_means(arm_dir: Path, dom: str, day: str, land: np.ndarray) -> tuple[float, float]:
    temps, clear = [], []
    for h in HOURS:
        path = arm_dir / f"wrfout_{dom}_{day}_{h:02d}:00:00"
        with Dataset(path) as ds:
            t2 = np.asarray(ds["T2"][0], np.float64)
            sw, top = np.asarray(ds["SWDOWN"][0], np.float64), np.asarray(ds["SWDNT"][0], np.float64)
        if not (np.all(np.isfinite(t2[land])) and np.all(np.isfinite(sw[land])) and np.all(np.isfinite(top[land]))):
            raise ValueError(f"non-finite field in {path}")  # E200: no NaN false pass
        temps.append(float(t2[land].mean()))
        sel = land & (top > 50.0)
        clear.append(float(np.mean(sw[sel] / top[sel])))
    return float(np.mean(temps)), float(np.mean(clear))


def verdict(d_fp, n, d_f):
    neg = sum(1 for x in d_fp if x < 0)
    beyond = sum(1 for x, s in zip(d_fp, n) if x < -s)
    if neg >= 3 and beyond >= 3 and sum(d_fp) <= 0.5 * sum(d_f):
        return "REPEATS"
    if neg <= 1 or sum(d_fp) >= 0:
        return "CHANCE"
    return "INCONCLUSIVE"


def main(runs: Path, out: Path, pre: bool) -> None:
    arms = ("B", "Bp", "F") if pre else ("B", "Bp", "F", "Fp")
    cells, result = [], {"schema": "w42-swdfid.fprime_rule.v1", "rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "runs": str(runs), "arms": list(arms), "pre": pre, "cells": {}}
    for dom in DOMAINS:
        with Dataset(runs / "B" / "gpu" / sorted(p.name for p in (runs / "B" / "gpu").glob(f"wrfout_{dom}_*"))[0]) as ds:
            land = np.asarray(ds["LANDMASK"][0]) > 0.5
        for day in DAYS:
            vals = {a: land_means(runs / a / "gpu", dom, day, land) for a in arms}
            ref_t = 0.5 * (vals["B"][0] + vals["Bp"][0])
            ref_k = 0.5 * (vals["B"][1] + vals["Bp"][1])
            row = {"land_cells": int(land.sum()), "T": {a: v[0] for a, v in vals.items()}, "K": {a: v[1] for a, v in vals.items()},
                   "N": abs(vals["B"][0] - vals["Bp"][0]) / 2, "NK": abs(vals["B"][1] - vals["Bp"][1]) / 2}
            for a in arms:
                if a in ("F", "Fp"):
                    row[f"D_{a}"] = vals[a][0] - ref_t
                    row[f"DK_{a}"] = vals[a][1] - ref_k
            result["cells"][f"{dom}_{day}"] = row
            cells.append(row)
    d_f = [c["D_F"] for c in cells]
    n = [c["N"] for c in cells]
    result["F_reference"] = {"D_F": d_f, "sum_D_F": sum(d_f), "N": n, "neg": sum(x < 0 for x in d_f),
                             "beyond": sum(x < -s for x, s in zip(d_f, n))}
    if not pre:
        d_fp = [c["D_Fp"] for c in cells]
        result["Fp"] = {"D_Fp": d_fp, "sum_D_Fp": sum(d_fp), "neg": sum(x < 0 for x in d_fp),
                        "beyond": sum(x < -s for x, s in zip(d_fp, n)), "verdict": verdict(d_fp, n, d_f),
                        "DK_Fp_neg": sum(c["DK_Fp"] < 0 for c in cells),
                        "DK_Fp_beyond": sum(c["DK_Fp"] < -c["NK"] for c in cells)}
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=1))
    for key, c in result["cells"].items():
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("T", "K")})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), "--pre" in sys.argv[3:])
