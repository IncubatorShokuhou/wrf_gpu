"""Count non-finite reference (CPU) and candidate (GPU) values per field and lead from existing comparator JSONs
(by_lead n / finite_cpu / finite_gpu are exact element counts over the whole field; masked netCDF values count as
non-finite because read_var fills them with NaN). usage: python count_nonfinite.py OUT_JSON LABEL=DIR ..."""
import json
import sys
from pathlib import Path

out, cases = Path(sys.argv[1]), [a.split("=", 1) for a in sys.argv[2:]]
result = {}
for label, d in cases:
    rows = {}
    for dom in ("d01", "d02", "d03"):
        rep = json.loads((Path(d) / f"{dom}.json").read_text())
        fields = rep["field_summaries"]
        n_fields = n_leads = 0
        for name, f in sorted(fields.items()):
            if name == "Times":
                continue
            n_fields += 1
            for lead in f.get("by_lead", []):
                n_leads += 1
                bad_cpu, bad_gpu = lead["n"] - lead["finite_cpu"], lead["n"] - lead["finite_gpu"]
                if bad_cpu or bad_gpu:
                    rows.setdefault(f"{dom}/{name}", []).append({"lead_h": lead["lead_h"], "n": lead["n"],
                                                                "nonfinite_cpu": bad_cpu, "nonfinite_gpu": bad_gpu})
        rows[f"{dom}/_checked"] = {"numeric_fields": n_fields, "field_leads": n_leads,
                                   "non_numeric": [x["field"] for x in rep["inventory"]["non_numeric_common"]],
                                   "incompatible": [x["field"] for x in rep["inventory"]["incompatible_common"]]}
    result[label] = rows
out.write_text(json.dumps(result, indent=1) + "\n")
for label, rows in result.items():
    bad = {k: v for k, v in rows.items() if not k.endswith("_checked")}
    print(label, {k: v for k, v in rows.items() if k.endswith("_checked")}, "NONFINITE:", json.dumps(bad)[:1500])
