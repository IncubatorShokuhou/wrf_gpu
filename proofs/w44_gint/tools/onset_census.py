"""One-sided and symmetric 'whole-field zero' frame census on retained teacher extracts (alisios.teacher_extract.v2), CPU only.

For each pair (reference extract dir, candidate extract dir) and each field F in FIELDS, domain file in DOMS, lead 1..72:
  ref_on = any(F_ref > 0), cand_on = any(F_cand > 0)  (strictly positive; all operands must be finite, E200)
  REF_ONLY  = ref_on and not cand_on   (what the G-INT degeneracy rule can flag: candidate constant 0, reference varies)
  CAND_ONLY = cand_on and not ref_on   (the mirror the rule cannot see)
For every one-sided frame: the active side's max and cell count, and the distance (h) to the nearest lead where the other side
is active (onset/decay offset). Pairs: GPU vs CPU-WRF (8 teacher cases) and CPU twin0227 vs its CPU-WRF reference
(CPU-vs-CPU control). usage: JAX_PLATFORMS=cpu python3 onset_census.py OUT_JSON
"""
import json, sys
from pathlib import Path

import numpy as np

T = Path("<USER_HOME>/wrf_gpu2_lanes/w43-teacher")
PAIRS = {f"gpu_{i}": (T / ("te_cpu" if i.startswith("2024") else "te_repro") / f"{i}_cpu", T / "te_gpu" / f"{i}_gpu")
         for i in ("20260227", "20260502", "20260614", "20260220", "20260608", "20260120", "20240302", "20240218")}
PAIRS["twin_20260227"] = (T / "te_repro/20260227_cpu", Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/contingency/twin0227_cpu"))
FIELDS = ("LWP", "IWP", "RAINNC", "rain_increment")
DOMS = ("d01_crop_sfc", "d02_crop_sfc", "d03_sfc")


def main():
    out = {"rule": __doc__.split("\n\n")[0], "pairs": {}}
    for name, (rd, cd) in PAIRS.items():
        rec = {"ref": str(rd), "cand": str(cd), "domains": {}}
        for dom in DOMS:
            r, c = np.load(rd / f"{dom}.npz"), np.load(cd / f"{dom}.npz")
            drec = {}
            for f in FIELDS:
                a, b = r[f][1:73].astype(np.float64), c[f][1:73].astype(np.float64)
                assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), (name, dom, f)
                ron = (a > 0).reshape(72, -1).any(1); con = (b > 0).reshape(72, -1).any(1)
                rows = []
                for k in range(72):
                    if ron[k] != con[k]:
                        side, other = (a, con) if ron[k] else (b, ron)
                        on = np.nonzero(other)[0]
                        dist = int(np.abs(on - k).min()) if len(on) else None
                        rows.append({"lead": k + 1, "side": "REF_ONLY" if ron[k] else "CAND_ONLY",
                                     "active_max": float(side[k].max()), "active_cells": int((side[k] > 0).sum()),
                                     "hours_to_other_side_active": dist})
                drec[f] = {"both": int((ron & con).sum()), "both0": int((~ron & ~con).sum()),
                           "REF_ONLY": int((ron & ~con).sum()), "CAND_ONLY": int((~ron & con).sum()), "frames": rows}
            rec["domains"][dom] = drec
        out["pairs"][name] = rec
    tot = {}
    for name, rec in out["pairs"].items():
        t = {"REF_ONLY": 0, "CAND_ONLY": 0, "both": 0, "both0": 0}
        for drec in rec["domains"].values():
            for v in drec.values():
                for k in t:
                    t[k] += v[k]
        tot[name] = t
    out["totals"] = tot
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for name, t in tot.items():
        print(name, t)


if __name__ == "__main__":
    main()
