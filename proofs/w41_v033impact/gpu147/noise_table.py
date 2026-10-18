"""P-F (wrong-load signal + compile noise) vs V-F (fresh-vs-fresh compile noise floor) per field/domain/lead,
from census.json (reference F). GPU-vs-GPU impact evidence, not fidelity.
usage: python noise_table.py census.json OUT_JSON OUT_MD
"""
import json
import sys
from pathlib import Path

c = json.loads(Path(sys.argv[1]).read_text())
FIELDS = ("T2", "U10", "V10", "TSK", "SST", "PSFC", "Q2", "HFX", "LH", "RAINNC", "TSLB", "T", "U", "V", "W", "QVAPOR")
LEADS = (1, 6, 24)
arms = c["arms"]
rows, md = [], ["| dom | lead | field | P-F rmse | P-F max | V-F rmse | V-F max | P-F water rmse | P-F land rmse |",
                "|---|---|---|---|---|---|---|---|---|"]
summary = {}
for dom in sorted(arms.get("P", {})):
    per_p = {r.get("lead_index"): r for r in arms["P"][dom]}
    per_v = {r.get("lead_index"): r for r in arms.get("V", {}).get(dom, [])}
    summary[dom] = {"P_differing_vars_per_lead": [len(r.get("differing", {})) for r in arms["P"][dom]],
                    "V_differing_vars_per_lead": [len(r.get("differing", {})) for r in arms.get("V", {}).get(dom, [])],
                    "P_differing_var_names_all_leads": sorted({v for r in arms["P"][dom] for v in r.get("differing", {})}),
                    "V_differing_var_names_all_leads": sorted({v for r in arms.get("V", {}).get(dom, []) for v in r.get("differing", {})})}
    for lead in LEADS:
        rp, rv = per_p.get(lead, {}), per_v.get(lead, {})
        for f in FIELDS:
            sp = rp.get("differing", {}).get(f)
            sv = rv.get("differing", {}).get(f)
            row = {"dom": dom, "lead": lead, "field": f,
                   "PF_rmse": sp["rmse"] if sp else 0.0, "PF_max": sp["max_abs"] if sp else 0.0,
                   "VF_rmse": sv["rmse"] if sv else 0.0, "VF_max": sv["max_abs"] if sv else 0.0,
                   "PF_water_rmse": (sp.get("water") or {}).get("rmse") if sp else 0.0,
                   "PF_land_rmse": (sp.get("land") or {}).get("rmse") if sp else 0.0}
            rows.append(row)
            md.append(f"| {dom} | {lead} | {f} | {row['PF_rmse']:.3g} | {row['PF_max']:.3g} | {row['VF_rmse']:.3g} | "
                      f"{row['VF_max']:.3g} | {row['PF_water_rmse'] if row['PF_water_rmse'] is not None else '-'} | "
                      f"{row['PF_land_rmse'] if row['PF_land_rmse'] is not None else '-'} |")
Path(sys.argv[2]).write_text(json.dumps({"label": "GPU-vs-GPU impact (not fidelity); reference F (fresh 0707 compile)",
                                         "summary": summary, "rows": rows}, indent=1))
Path(sys.argv[3]).write_text("\n".join(md) + "\n")
print(json.dumps(summary, indent=1))
