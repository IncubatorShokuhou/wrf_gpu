"""t-wsmdz W02: split the W01 caller-input sensitivity into dz-only and rho-only arms
(pristine WRF wsm5/wsm6 both arms), and check EOS dry density vs hypsometric 1/alt.
Usage: python w02_split.py <case_dir> <scheme> <out.json> <hour>..."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w01_columns as w

case, scheme, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
rep = {"schema": "t-wsmdz-w02-v1", "case": str(case), "scheme": scheme, "hours": {}}
for h in [int(x) for x in sys.argv[4:]]:
    f = sorted((case / "cpu").glob("wrfout_d01_*"))[h]
    x, gw, gp = w.wrf_inputs(f)
    P = w.run_pristine(scheme, x, gw, 18.0, f"s{h}P")
    Qdz = w.run_pristine(scheme, x, dict(den=gw["den"], delz=gp["delz"]), 18.0, f"s{h}dz")
    Qrho = w.run_pristine(scheme, x, dict(den=gp["den"], delz=gw["delz"]), 18.0, f"s{h}rho")
    # EOS dry density with WRF constants from the same history operands
    thm = x["th"] * (np.float32(1) + np.float32(np.float32(461.6) / np.float32(287.0)) * x["qv"])
    rho_eos = x["p"] / (np.float32(287.0) * thm * x["pii"])
    rep["hours"][str(h)] = {"dz_only": w.compare(P, Qdz, x), "rho_only": w.compare(P, Qrho, x),
        "rho_eos_over_hyp": {"median": float(np.median(rho_eos / gw["den"])),
                             "p01": float(np.percentile(rho_eos / gw["den"], 1)),
                             "p99": float(np.percentile(rho_eos / gw["den"], 99))}}
    for arm in ("dz_only", "rho_only"):
        r = rep["hours"][str(h)][arm]
        print(h, arm, {n: round(r[n].get("rel_sum", r[n].get("rel_to_increment", 0)), 6) for n in r if "max_abs" in r[n]})
    print(h, "rho_eos/hyp", rep["hours"][str(h)]["rho_eos_over_hyp"])
out.write_text(json.dumps(rep, indent=1))
