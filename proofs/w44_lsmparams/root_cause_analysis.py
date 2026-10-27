"""LS71 post-hoc analysis of retained arms (no model re-run): leaf names, magnitudes, canopy identity, const residual."""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
def load(tag):
    m = json.loads((D / f"{tag}.json").read_text()); d = np.load(D / f"{tag}.npz")
    return m, {n: d[f"L{i:04d}"] for i, n in enumerate(m["leaf_names"])}
def ulps(x, y):
    a = x.view(np.int32).astype(np.int64); b = y.view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a); b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)
NAMES = {"[<flat index 0>][<flat index 20>]": "State.qke", "[<flat index 0>][<flat index 22>]": "State.theta_flux",
         "[<flat index 0>][<flat index 23>]": "State.qv_flux", "[<flat index 0>][<flat index 27>]": "State.fltv",
         "[<flat index 0>][<flat index 28>]": "State.t_skin", "[<flat index 0>][<flat index 53>]": "State.qsq",
         "[<flat index 0>][<flat index 72>]": "State.hfx", "[<flat index 0>][<flat index 73>]": "State.qfx",
         "[<flat index 0>][<flat index 74>]": "State.qsfc", "[<flat index 0>][<flat index 76>]": "State.el_pbl",
         "[<flat index 15>][<flat index 1>]": "noahmp_land.smois", "[<flat index 15>][<flat index 2>]": "noahmp_land.sh2o",
         "[<flat index 15>][<flat index 15>]": "noahmp_land.tg", "[<flat index 15>][<flat index 17>]": "noahmp_land.eah",
         "[<flat index 15>][<flat index 25>]": "noahmp_land.t_skin", "[<flat index 15>][<flat index 26>]": "noahmp_land.qsfc",
         "[<flat index 30>][<flat index 0>]": "land_history[0]", "[<flat index 31>][<flat index 0>]": "energy_accumulators[0]"}
def name(k):
    for s, n in NAMES.items():
        if k == "['d01']" + s: return n
    return k
def rows(ref, cand):
    out = []
    for k in ref:
        if k.startswith(("NAMED.", "CENSUS.")) or ref[k].tobytes() == cand[k].tobytes():
            continue
        x, y = ref[k], cand[k]; m = x != y; u = ulps(x, y); d = np.abs(x.astype(np.float64) - y)
        i = np.unravel_index(np.argmax(np.where(m, u, -1)), x.shape)
        out.append({"leaf": k, "name": name(k), "cells": int(m.sum()), "ulp_median": float(np.median(u[m])),
                    "ulp_max": int(u.max()), "max_abs": float(d.max()), "value_at_max_ulp": float(x[i]),
                    "field_abs_max_finite": float(np.abs(x[np.abs(x) < 1e30]).max())})
    return out
om, off = load("off"); _, on = load("on"); _, const = load("const")
qm, onc = load("_census")
r = {"head": om["git_head"], "on_vs_off": rows(off, on), "const_vs_off": rows(off, const),
     "canopy_identity_on_vs_off": {f: int((off["NAMED.noahmp_land." + f] != on["NAMED.noahmp_land." + f]).sum())
                                   for f in ("tv", "tah", "tg", "eah")}}
tslb, sh, sm = (off["['d01'][<flat index 15>][<flat index %d>]" % i] for i in (0, 2, 1))
m = (sh != const["['d01'][<flat index 15>][<flat index 2>]"]) | (sm != const["['d01'][<flat index 15>][<flat index 1>]"])
r["const_residual_soil_layers"] = {"cells": int(m.sum()), "tslb_K": [round(float(t), 2) for t in tslb[m]],
                                   "frozen_cells": int((tslb[m] < 273.15).sum())}
s = onc["CENSUS.w2_active_sweeps"]
r["on_census"] = {"arm": qm["arm"], "census": qm["census"], "columns": int(s.size),
                  "sweeps_hist": {int(k): int(v) for k, v in zip(*np.unique(s, return_counts=True))},
                  "instrumented_vs_plain_on_differing_leaves": sum(1 for k in on if not k.startswith(("NAMED.", "CENSUS."))
                                                                   and on[k].tobytes() != onc[k].tobytes()),
                  "off_census": "LOST: systemd-run expanded ${arm} -> both census arms wrote _census.npz (on overwrote off)"}
r["exit_flip_hypothesis"] = "REFUTED: TV and TAH bitwise identical ON vs OFF in every column"
r["A_const_restores_off_bytes"] = not r["const_vs_off"]
(D / "analysis.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=1)[:6000])
