"""Tables + pre-registered gate for the Swiss twin OSSE (reads osse/<arm>/report.json). Usage: osse_tables.py out.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-da-ens/osse")
BANDS = ("0-1km", "1-2km", "2-5km")
VARS = ("T", "U", "V")


def rms(d, var):
    vals = [d[var][b] for b in BANDS if d[var].get(b) is not None]
    return float(np.sqrt(np.mean(np.square(vals))))


def arm(name):
    r = json.loads((ROOT / name / "report.json").read_text())
    rows = []
    for c in r["cycles"]:
        prior = c["forecast"]["+3h"]
        row = {"cycle": c["cycle"], "t_a": c["t_a"],
               "prior_spread": {v: rms(prior["spread"], v) for v in VARS},
               "prior_rmse": {v: rms(prior["rmse"], v) for v in VARS},
               "first_hour_retention": ({v: rms(c["forecast"]["+1h"]["spread"], v) / rms(c["forecast"]["+0h"]["spread"], v)
                                         for v in VARS} if "+0h" in c["forecast"] else None),
               "forecast_wall_s": c.get("forecast_wall_s")}
        if "posterior" in c:
            row.update({"post_spread": {v: rms(c["posterior"]["spread"], v) for v in VARS},
                        "post_rmse": {v: rms(c["posterior"]["rmse"], v) for v in VARS},
                        "written_spread": {v: rms(c["written"]["spread"], v) for v in ("T", "U", "V")},
                        "analysis_wall_s": c["analysis_wall_s"], "peak_rss_mb": c.get("peak_rss_mb_process"),
                        "inflation": c["inflation"].get("rho") if "rho" in c["inflation"] else
                        {k: v["rho"] for k, v in c["inflation"]["table"].items()}})
        rows.append(row)
    return {"rows": rows, "lbc": r.get("lbc"), "ic_balance": r.get("ic_balance")}


EXPECTED_CYCLES = (0, 1, 2, 3)        # priors at 03/06/09/12Z; analyses at cycles 0-2
# registered prior/analysis valid time of every cycle (gate pre-registration 10:13Z / F1 12:36:39Z)
EXPECTED_T_A = {0: "2023-01-15_03:00:00", 1: "2023-01-15_06:00:00", 2: "2023-01-15_09:00:00",
                3: "2023-01-15_12:00:00"}


def completeness(a) -> list:
    """Reasons an arm cannot be gated (empty = complete): every expected cycle exactly once at its registered valid
    time, posterior + written for the analysis cycles, all gate inputs finite. An incomplete report never passes
    (critic delta 12:45Z / 12:59Z)."""
    rows = a["rows"]
    cyc = [r.get("cycle") for r in rows]
    why = []
    if sorted(cyc, key=str) != list(EXPECTED_CYCLES):
        why.append(f"cycles {sorted(cyc, key=str)} != {list(EXPECTED_CYCLES)}")
    for r in rows:
        if r.get("t_a") != EXPECTED_T_A.get(r.get("cycle")):
            why.append(f"cycle {r.get('cycle')}: t_a {r.get('t_a')!r} != registered {EXPECTED_T_A.get(r.get('cycle'))!r}")
        need = ["prior_spread", "prior_rmse"] + (["post_spread", "post_rmse", "written_spread"]
                                                 if r["cycle"] < EXPECTED_CYCLES[-1] else [])
        for k in need:
            vals = list((r.get(k) or {}).values())
            if len(vals) != len(VARS) or not np.all(np.isfinite(np.asarray(vals, float))):
                why.append(f"cycle {r['cycle']}: {k} missing or non-finite")
    return why


def gate(a1, a0):
    for a in (a1, a0):
        why = completeness(a)
        if why:
            return {"PASS": False, "INCOMPLETE": why}
    rows = sorted(a1["rows"], key=lambda r: r["cycle"])          # complete + unique: cycle i == registered time i
    later = [r for r in rows if r["cycle"] >= 1]
    ratio = [float(np.sqrt(np.mean([(r["prior_spread"][v] / r["prior_rmse"][v]) ** 2 for v in VARS]))) for r in later]
    g1 = all(0.5 <= x <= 2.0 for x in ratio)
    base = next(r for r in rows if r["cycle"] == 0)["prior_spread"]          # the 03Z prior, chosen explicitly
    g2 = all(r["prior_spread"][v] >= 0.5 * base[v] for r in later for v in VARS)
    surv = [r["written_spread"][v] / r["post_spread"][v] for r in rows if "post_spread" in r for v in VARS]
    g3 = all(abs(x - 1.0) <= 0.01 for x in surv)
    def mean_post(a):
        return float(np.mean([r["post_rmse"][v] for r in a["rows"] if "post_rmse" in r for v in VARS]))
    g4 = mean_post(a1) < mean_post(a0)
    return {"G1_spread_over_rmse_later_cycles": ratio, "G1": g1, "G2_no_compounding_decay": g2,
            "G3_written_equals_posterior": [min(surv), max(surv)], "G3": g3,
            "G4_mean_post_rmse_A1_A0": [mean_post(a1), mean_post(a0)], "G4": g4, "PASS": g1 and g2 and g3 and g4}


def gate_prospective(a):
    """F1 gate, registered in MAILBOX before F1 ran: G1-G3 as for A1; G4' = mean over the analyses of the
    posterior RMSE < mean of the prior RMSE (T/U/V, 0-5 km), i.e. the analyses improve on their own prior (no A0
    on the new truth)."""
    g = gate(a, a)
    if "INCOMPLETE" in g:
        return g
    post = float(np.mean([r["post_rmse"][v] for r in a["rows"] if "post_rmse" in r for v in VARS]))
    prior = float(np.mean([r["prior_rmse"][v] for r in a["rows"] if "post_rmse" in r for v in VARS]))
    g.pop("G4_mean_post_rmse_A1_A0"), g.pop("G4"), g.pop("PASS")
    g.update({"G4p_mean_post_vs_prior_rmse": [post, prior], "G4p": post < prior})
    g["PASS"] = g["G1"] and g["G2_no_compounding_decay"] and g["G3"] and g["G4p"]
    return g


def markdown(out) -> str:
    """Per-arm cycle table (rms over the 0-1/1-2/2-5 km bands, interior) for ENS_DESIGN.md."""
    f = lambda d: " / ".join(f"{d[v]:.2f}" for v in VARS) if d else "–"
    lines = []
    for name in ("A0", "A1", "A1b", "F1"):
        if name not in out:
            continue
        lines += [f"**{name}**" + {"A1b": " (POST-HOC)", "F1": " (PROSPECTIVE, new truth)"}.get(name, ""), "",
                  "| t (prior) | prior spread T/U/V | prior RMSE T/U/V | posterior spread | posterior RMSE | written spread "
                  "| 1st-h spread retention | analysis wall s / peak RSS MB |", "|---|---|---|---|---|---|---|---|"]
        for r in out[name]["rows"]:
            t = r["t_a"][11:13] + "Z"
            wall = (f"{r['analysis_wall_s']:.0f} / {r['peak_rss_mb']:.0f}" if r.get("analysis_wall_s") is not None
                    else "–")
            lines.append(f"| {t} | {f(r['prior_spread'])} | {f(r['prior_rmse'])} | {f(r.get('post_spread'))} | "
                         f"{f(r.get('post_rmse'))} | {f(r.get('written_spread'))} | {f(r.get('first_hour_retention'))} | {wall} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = {name: arm(name) for name in ("A0", "A1", "A1b", "F1") if (ROOT / name / "report.json").exists()}
    if "A0" in out and "A1" in out:
        out["gate_A1"] = gate(out["A1"], out["A0"])
        out["gate_A0_contrast"] = gate(out["A0"], out["A0"])
    if "A0" in out and "A1b" in out:   # POST-HOC arm (estimator v2), same criteria, not the pre-registered verdict
        out["gate_A1b_post_hoc"] = gate(out["A1b"], out["A0"])
    if "F1" in out:
        out["gate_F1_prospective"] = gate_prospective(out["F1"])
    text = json.dumps(out, indent=1)
    print(markdown(out))
    print(json.dumps({k: v for k, v in out.items() if k.startswith("gate")}, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + "\n")
