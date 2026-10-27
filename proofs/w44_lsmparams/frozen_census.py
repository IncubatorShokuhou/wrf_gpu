"""Activity census for the frozen-soil changed-table oracle (critic F5/F5', E95/E154).

Reads only the oracle JSON. Per arm and column (top soil layer L1, dz = -zsoil[0]):
- ice gain dI = (SMC_OUT - SH2O_OUT) - (SMC_IN - SH2O_IN), and the frost-deposition bound
  D = max(0, -QFX*dt) / (1000 * dz) (all deposited ice in L1);
- PHASECHANGE froze (not deposition) iff SH2O_OUT < SH2O_IN and dI > 10*D;
- changed-table consumer activity: SH2O_OUT/SMC_OUT moved relative to the unchanged-table base arm.
Pass: every arm freezes on at least one column; BEXP/PSISAT/SMCMAX move L1 SH2O_OUT on a freezing
column by more than the frozen 5e-6 field tolerance (supercooled limit binding); DKSAT moves SH2O/SMC.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--oracle", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
oracle = json.loads(a.oracle.read_text())
base = oracle["arms"]["base"]["oracle"]
TOL = 5e-6  # score.py frozen sh2o/smois atol
report = {"oracle_sha256": hashlib.sha256(a.oracle.read_bytes()).hexdigest(),
          "columns": [c["name"] for c in oracle["columns"]], "rule": __doc__.strip(), "arms": {}}
for name, arm in oracle["arms"].items():
    rows = []
    for column, value, reference in zip(oracle["columns"], arm["oracle"], base, strict=True):
        s = column["state_in"]
        smc_in, sh2o_in = np.asarray(s["smc"], np.float64), np.asarray(s["sh2o"], np.float64)
        smc, sh2o = np.asarray(value["smc_out"], np.float64), np.asarray(value["sh2o_out"], np.float64)
        stc = np.asarray(s["stc"][3:], np.float64)  # soil layers follow 3 snow slots
        dz1 = -float(column["zsoil"][0])
        gain = (smc[0] - sh2o[0]) - (smc_in[0] - sh2o_in[0])
        deposition = max(0., -float(value["driver"]["qfx"]) * float(column["dt"])) / (1000. * dz1)
        froze = bool(sh2o[0] < sh2o_in[0] and gain > 10. * deposition)
        rows.append({"column": column["name"], "initial_soil_T_K": stc.tolist(),
                     "frozen_layers_in": int(np.sum(stc < 273.15)), "ice_layers_out": int(np.sum(smc > sh2o)),
                     "L1_sh2o_in": sh2o_in[0], "L1_sh2o_out": sh2o[0], "L1_ice_gain": gain,
                     "L1_deposition_bound": deposition, "L1_phase_change_froze": froze,
                     "sh2o_moved_vs_base": float(np.max(np.abs(sh2o - np.asarray(reference["sh2o_out"])))),
                     "smc_moved_vs_base": float(np.max(np.abs(smc - np.asarray(reference["smc_out"])))),
                     "L1_sh2o_out_minus_base": float(sh2o[0] - reference["sh2o_out"][0])})
    freezing = [r for r in rows if r["L1_phase_change_froze"]]
    if name in ("bexp", "psisat", "smcmax"):
        consumer = any(abs(r["L1_sh2o_out_minus_base"]) > TOL for r in freezing)
        why = "supercooled limit binds: L1 SH2O_OUT of a freezing column moves vs base by > 5e-6"
    elif name == "dksat":
        consumer = any(max(r["sh2o_moved_vs_base"], r["smc_moved_vs_base"]) > 0 for r in rows)
        why = "infiltration/drainage moves SH2O or SMC vs base"
    else:
        consumer, why = True, "reference arm"
    report["arms"][name] = {"factor": arm["factor"], "columns": rows, "freezing_columns": [r["column"] for r in freezing],
                            "phase_change_active": bool(freezing), "consumer_active": consumer, "consumer_rule": why}
report["pass"] = all(arm["phase_change_active"] and arm["consumer_active"] for arm in report["arms"].values())
a.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({n: {k: v for k, v in arm.items() if k in ("freezing_columns", "consumer_active")} for n, arm in report["arms"].items()}))
raise SystemExit(0 if report["pass"] else 1)
