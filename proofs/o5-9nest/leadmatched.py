"""Lead-matched same-resolution comparison per #148 D6 row (critic F2/F3; rev 2.1 alignment): breach leads with r_X(τ) >= r_C(τ) of the
0420 CPU-CPU twin at the same lead (d02 <-> 0420 d02, 1 km leaves <-> 0420 d03), breach leads with r_X >= min r_C over τ12-72, and leads
with rmse/std X > C. Inputs: results/{scalefree,rows148,cpu_twin_0420}.json. usage: python leadmatched.py"""
import json
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
s = json.load(open(R / "scalefree.json")); rows = json.load(open(R / "rows148.json"))["rows"]; c = json.load(open(R / "cpu_twin_0420.json"))
out, lines = [], []
for i, r in enumerate(rows, 1):
    f, dom = r["field"], r["dom"]; ref = "d02" if dom == "d02" else "d03"
    br = r["breach_leads"]; rx = s["x148"][f"{dom}/{f}"]
    rc = lambda L: (c[ref].get(str(L)) or {}).get(f, {}).get("pearson_r")
    soc = lambda L: (lambda v: v[f]["rmse"] / v[f]["std_C"] if v and v[f]["std_C"] else None)(c[ref].get(str(L)))
    k0 = sum(1 for L in br if rc(L) is not None and rx[str(L)]["r"] >= rc(L)); n0 = sum(1 for L in br if rc(L) is not None)
    cmin = min(v[f]["pearson_r"] for L, v in c[ref].items() if 12 <= int(L) <= 72 and v[f]["pearson_r"] is not None)
    kmin = sum(1 for L in br if rx[str(L)]["r"] >= cmin)
    so = [(rx[str(L)]["rmse_over_std"], soc(L)) for L in br if soc(L)]
    worse = sum(1 for a, b in so if a > b)
    lines.append(f"row{i:2d} {r['stage']} {dom} {f:6s} n={len(br):2d} | lead-matched r_X>=r_C(0420 {ref}) {k0}/{n0} | "
                 f"r_X>=min r_C(τ12-72)={cmin:.2f}: {kmin}/{len(br)} | rmse/std X>C {worse}/{len(so)}")
    out.append(dict(row=i, stage=r["stage"], dom=dom, field=f, n=len(br), lead_matched=[k0, n0], cmin_tau12_72=cmin, ge_min=kmin, rmse_std_worse=[worse, len(so)]))
(R / "leadmatched.json").write_text(json.dumps(out, indent=1)); (R / "leadmatched.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
