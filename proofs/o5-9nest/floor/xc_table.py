"""X/C statistics from own-twin cases (w43-teacher scorer JSONs): X = GPU v0.4.2 vs CPU ref, C = CPU decomposition twin vs CPU ref.
[M] numbers from the scorer by_lead RMSE; ratios only where C > 0."""
import json, statistics as st, sys
d = json.load(open("curves_0227_era5.json"))
rows = {}
for key in sorted(d):
    C = {int(k): v for k, v in d[key]["C"].items()}
    X = {int(k): v for k, v in d[key]["X"].items()}
    r = {l: X[l] / C[l] for l in range(1, 73) if C[l] > 0}
    late = [r[l] for l in range(12, 73) if l in r]
    rows[key] = {
        "seed_xc_tau1": round(r.get(1, float("nan")), 3), "seed_xc_tau3": round(r.get(3, float("nan")), 3),
        "xc_median_tau12_72": round(st.median(late), 3), "xc_max_tau12_72": round(max(late), 3),
        "frac_leads_x_gt_4c3_tau12_72": round(sum(v > 4 / 3 for v in late) / len(late), 3),
        "xc_tau72": round(r.get(72, float("nan")), 3),
        "X_max": round(max(X.values()), 5), "C_max": round(max(C.values()), 5)}
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(f"{'case/dom/field':24s} seed1 seed3 med12-72 max12-72 frac>4/3 xc72   Xmax    Cmax")
for k, v in rows.items():
    if "/d01/" in k: continue
    print(f"{k:24s} {v['seed_xc_tau1']:5.2f} {v['seed_xc_tau3']:5.2f} {v['xc_median_tau12_72']:8.2f} {v['xc_max_tau12_72']:8.2f} {v['frac_leads_x_gt_4c3_tau12_72']:8.2f} {v['xc_tau72']:5.2f} {v['X_max']:7.4g} {v['C_max']:7.4g}")
