"""TMN land-invariance across issue dates (follow-up to #147; manager 2026-10-18T09:55/09:57Z, alisios follow-up).

Reads ONLY TMN, LANDMASK, XLAND of wrfinput_d02..d0N (+ the W9N namelist text) inside alisios_guard.guarded_session
(scan -> any non-ALLOW day refuses everything -> admit -> netCDF/open spy), one session per part, receipts in
tmn/receipts_<part>/. No values are logged; only derived pair statistics are written.
usage: JAX_PLATFORMS=cpu python tmn_invariance.py OUT_JSON
"""
import datetime as dtm
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<USER_HOME>/src/wrf_gpu2_wt/w41-v033impact/src")
from gpuwrf.assimilation.letkf.alisios_guard import guarded_session  # noqa: E402

OUT = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
PURPOSE = ("w41-v033impact: v0.3.3 fused-bake impact follow-up (manager 09:55Z/09:57Z, alisios "
           "TO_WRF_GPU2_V033_IMPACT_FOLLOWUP_20261018): TMN/LANDMASK/XLAND only (+W9N namelist geometry), "
           "land-invariance of TMN across issue dates; no values logged")
CUTOFF = "20260915"
STORE = Path("<DATA_ROOT>/alisios/state/manager/runs/gpu_input_store_20261004")
PART1 = ["20260116_18z", "20260125_18z", "20260216_18z", "20260327_18z", "20260415_18z", "20260420_18z",
         "20260516_18z", "20260616_18z", "20260707_18z", "20260805_18z"]
W9N = Path("<DATA_ROOT>/alisios/scratch/w1011-ds-coast-wrf-value/stage2/cases")
PART2 = ["20260107_18z", "20260316_12z", "20260619_12z", "20260830_12z"]


def days(issue):
    d = dtm.date(int(issue[:4]), int(issue[4:6]), int(issue[6:8]))
    return [d.strftime("%Y%m%d"), (d + dtm.timedelta(days=1)).strftime("%Y%m%d")]


def read_part(name, files, issues, namelists=()):
    for issue in issues:
        assert issue[:8] < CUTOFF, issue
    allowed = {str(p): days(issue) for p, issue in files + list(namelists)}
    data, nls = {}, {}
    rdir = HERE / f"receipts_{name}"
    with guarded_session(allowed, sorted({i[:8] for i in issues}), rdir, purpose=PURPOSE) as s:
        import netCDF4
        for p, issue in files:
            with netCDF4.Dataset(p) as ds:
                data[(issue, Path(p).name[-3:])] = {v: np.asarray(ds[v][0], dtype=np.float64) for v in ("TMN", "LANDMASK", "XLAND")}
        for p, issue in namelists:
            text = Path(p).read_text()
            nls[issue] = {k: re.search(rf"^\s*{k}\s*=\s*([^\n]*)", text, re.M | re.I).group(1).strip().rstrip(",")
                          if re.search(rf"^\s*{k}\s*=", text, re.M | re.I) else None
                          for k in ("max_dom", "parent_id", "parent_grid_ratio", "feedback", "sst_update", "tmn_update",
                                    "sf_surface_physics", "e_we", "e_sn", "dx", "start_year", "start_month", "start_day",
                                    "start_hour")}
    return data, nls, {"receipt_dir": str(rdir), "guard_result": s.receipt.get("guard_result"),
                       "observed_opens": len(s.receipt["spy"]["observed"]), "refused": s.receipt["spy"]["refused_paths"]}


def pair_stats(data, issues, dom):
    rows = []
    for a, b in itertools.combinations(issues, 2):
        A, B = data[(a, dom)], data[(b, dom)]
        land_a, land_b = A["LANDMASK"] > 0.5, B["LANDMASK"] > 0.5
        same_mask = bool(np.array_equal(land_a, land_b)) and bool(np.array_equal(A["XLAND"], B["XLAND"]))
        land = land_a & land_b
        water = (~land_a) & (~land_b)
        d = A["TMN"] - B["TMN"]
        rows.append({"a": a, "b": b, "landmask_xland_identical": same_mask,
                     "land_cells": int(land.sum()), "water_cells": int(water.sum()),
                     "land_max_abs_dTMN": float(np.abs(d[land]).max()) if land.any() else None,
                     "land_n_diff": int(np.count_nonzero(d[land])),
                     "water_max_abs_dTMN": float(np.abs(d[water]).max()) if water.any() else None,
                     "water_rmse_dTMN": float(np.sqrt(np.mean(d[water] ** 2))) if water.any() else None})
    return rows


result = {"schema": "w41-v033impact.tmn_land_invariance.v1", "purpose": PURPOSE,
          "label": "[M] guarded reads of TMN/LANDMASK/XLAND; land = LANDMASK==1 in both dates", "parts": {}}
f1 = [(STORE / i / f"wrfinput_{d}", i) for i in PART1 for d in ("d02", "d03")]
data1, _, rec1 = read_part("part1_store", f1, PART1)
result["parts"]["part1_wn3_store"] = {"issues": PART1, "domains": ["d02", "d03"], "session": rec1,
                                      "pairs": {d: pair_stats(data1, PART1, d) for d in ("d02", "d03")}}
f2 = [(W9N / i / "input42" / f"wrfinput_d0{k}", i) for i in PART2 for k in range(2, 10)]
n2 = [(W9N / i / "input42" / "namelist.input", i) for i in PART2]
data2, nls2, rec2 = read_part("part2_w9n", f2, PART2, n2)
result["parts"]["part2_w9n_9nest"] = {"issues": PART2, "domains": [f"d0{k}" for k in range(2, 10)], "session": rec2,
                                      "namelist_geometry": nls2,
                                      "pairs": {f"d0{k}": pair_stats(data2, PART2, f"d0{k}") for k in range(2, 10)}}
summary = {}
for part, rec in result["parts"].items():
    lm = [r["land_max_abs_dTMN"] or 0.0 for rows in rec["pairs"].values() for r in rows]
    wm = [r["water_max_abs_dTMN"] or 0.0 for rows in rec["pairs"].values() for r in rows]
    summary[part] = {"n_pairs": len(lm), "max_land_abs_dTMN_all_pairs_domains": max(lm), "max_water_abs_dTMN": max(wm),
                     "all_landmask_identical": all(r["landmask_xland_identical"] for rows in rec["pairs"].values() for r in rows),
                     "land_TMN_date_invariant": max(lm) == 0.0}
result["summary"] = summary
OUT.write_text(json.dumps(result, indent=1))
print(json.dumps(summary, indent=1))
print(json.dumps(result["parts"]["part2_w9n_9nest"]["namelist_geometry"], indent=1))
print(json.dumps({k: v["session"] for k, v in result["parts"].items()}, indent=1))
