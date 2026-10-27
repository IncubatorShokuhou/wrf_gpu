"""Aggregate replay.py output over captured steps (markdown + JSON summary)."""
import json
import sys
from collections import defaultdict


def main():
    d = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else None
    steps = d["steps"]
    checks = defaultdict(lambda: [0, 0])
    agg = defaultdict(lambda: defaultdict(float))
    worst = {}
    for st in steps:
        for k, v in st["checks"].items():
            if v is None:
                continue
            checks[k][0] += int(bool(v))
            checks[k][1] += 1
        for sp, v in st["species"].items():
            a = agg[sp]
            for mode, key in (("ieee", "classify_fin_chain"), ("ftz", "classify_fin_chain_ftz"), ("iso", "classify_fin_isolated")):
                c = v[key]
                for cls in ("port_negative", "FAITHFUL", "LARGER_IN_PORT", "SMALLER_IN_PORT", "WRF_ONLY"):
                    a[f"{mode}_{cls}"] += c[cls]
                if c["port_min"] is not None and c["port_min"] < a.get(f"{mode}_port_min", 0.0):
                    a[f"{mode}_port_min"] = c["port_min"]
                    a[f"{mode}_wrf_at_port_min"] = c["port_min_cell"]["wrf_off"]
                    a[f"{mode}_port_min_step"] = st["step"]
                if c["wrf_min"] is not None:
                    a[f"{mode}_wrf_min"] = min(a.get(f"{mode}_wrf_min", 0.0), c["wrf_min"])
                r = c.get("diff_over_eps_scale_at_port_neg", {}).get("max", 0.0)
                a[f"{mode}_max_diff_eps_S"] = max(a[f"{mode}_max_diff_eps_S"], r)
            cb = v["classify_fin_chain"].get("bands")
            if cb:
                for band, rec in cb.items():
                    a[f"{band}_n"] += rec["n"]
                    a[f"{band}_same"] += rec["wrf_same_sign"]
                    a[f"{band}_wrfnonneg"] += rec["wrf_nonnegative"]
                    a[f"{band}_maxrel"] = max(a[f"{band}_maxrel"], rec.get("max_rel_diff_same_sign", 0.0))
                a["wrf_only_above_1epsS"] += v["classify_fin_chain"]["WRF_ONLY_above_1epsS"]
                a["wrf_neg_total"] += v["classify_fin_chain"]["wrf_negative"]
                a["wrf_neg_R0"] += v["classify_fin_chain"]["wrf_negative_R0"]
            for mode in ("ieee", "ftz"):
                ac = v.get(f"allcell_fin_{mode}")
                if ac:
                    for k2, val in ac.items():
                        a[f"allcell_{mode}_{k2}"] = max(a[f"allcell_{mode}_{k2}"], val)
            b = v["birth"]
            a["negative"] += b["negative"]
            a["born_preadd"] += b["born_in_preadd"]
            for part, n in (b.get("preadd_dominant_negative_sc_part") or {}).items():
                a[f"dominant_{part}"] += n
            for k in ("hdiff", "diff6", "sc", "pre", "adv", "upd", "fin"):
                r = v["chain"][k].get("rel_max_interior", v["chain"][k]["rel_max"])
                a[f"chainrel_{k}"] = max(a[f"chainrel_{k}"], r)
                r = v["isolated"][k].get("rel_max_interior", v["isolated"][k]["rel_max"])
                a[f"isorel_{k}"] = max(a[f"isorel_{k}"], r)
            for k in ("diff6", "sc", "pre", "fin"):
                r = v["chain_ftz"][k].get("rel_max_interior", v["chain_ftz"][k]["rel_max"])
                a[f"ftzrel_{k}"] = max(a[f"ftzrel_{k}"], r)
    print(f"steps replayed: {len(steps)} ({steps[0]['step']}..{steps[-1]['step']})")
    print("wiring checks (true/total):", {k: f"{v[0]}/{v[1]}" for k, v in checks.items()})
    print("| species | mode | port neg | FAITHFUL | LARGER | SMALLER | WRF-only | port min (step) | WRF at that cell | WRF min | max diff/(eps S) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sp, a in agg.items():
        for mode in ("ieee", "ftz"):
            if not a[f"{mode}_port_negative"] and not a[f"{mode}_WRF_ONLY"]:
                continue
            print(f"| {sp} | {mode} | {int(a[mode + '_port_negative'])} | {int(a[mode + '_FAITHFUL'])} | {int(a[mode + '_LARGER_IN_PORT'])} | "
                  f"{int(a[mode + '_SMALLER_IN_PORT'])} | {int(a[mode + '_WRF_ONLY'])} | {a.get(mode + '_port_min', 0):.6g} "
                  f"({int(a.get(mode + '_port_min_step', 0))}) | {a.get(mode + '_wrf_at_port_min', 0):.6g} | {a.get(mode + '_wrf_min', 0):.6g} | "
                  f"{a[mode + '_max_diff_eps_S']:.3g} |")
    print()
    print("| species | negatives | born in pre-add | dominant part (pre-add-born) | max interior rel diff chain: hdiff/diff6/sc/pre/adv/fin |")
    print("|---|---:|---:|---|---|")
    for sp, a in agg.items():
        dom = {k[9:]: int(v) for k, v in a.items() if k.startswith("dominant_")}
        rel = "/".join(f"{a['chainrel_' + k]:.1e}" for k in ("hdiff", "diff6", "sc", "pre", "adv", "fin"))
        print(f"| {sp} | {int(a['negative'])} | {int(a['born_preadd'])} | {dom} | {rel} |")
    if any("R0_n" in a for a in agg.values()):
        print()
        print("| species | R0 n (WRF>=0) | R1 n, same sign, max rel diff | R2 n, same sign, max rel diff | WRF neg total (R0) | WRF-only >1 eps*S | all-cell max diff/(eps S) 7x7x3 / 7x7x5 (IEEE; FTZ) |")
        print("|---|---|---|---|---|---:|---|")
        for sp, a in agg.items():
            if not (a["R0_n"] or a["R1_n"] or a["R2_n"] or a["wrf_neg_total"]):
                continue
            print(f"| {sp} | {int(a['R0_n'])} ({int(a['R0_wrfnonneg'])}) | {int(a['R1_n'])}, {int(a['R1_same'])}/{int(a['R1_n'])}, {a['R1_maxrel']:.1e} | "
                  f"{int(a['R2_n'])}, {int(a['R2_same'])}/{int(a['R2_n'])}, {a['R2_maxrel']:.1e} | {int(a['wrf_neg_total'])} ({int(a['wrf_neg_R0'])}) | "
                  f"{int(a['wrf_only_above_1epsS'])} | {a['allcell_ieee_S7x7x3']:.2g} / {a['allcell_ieee_S7x7x5']:.2g}; "
                  f"{a['allcell_ftz_S7x7x3']:.2g} / {a['allcell_ftz_S7x7x5']:.2g} |")
    if out:
        json.dump({"steps": [s["step"] for s in steps], "checks": {k: v for k, v in checks.items()},
                   "species": {sp: dict(a) for sp, a in agg.items()}}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
