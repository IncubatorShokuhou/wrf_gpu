"""w41-rw3add: descriptive summary over both pre-registered sessions (NOT a pre-registered verdict; the verdicts are
receipts/S1_VERDICT.json and S2_VERDICT.json). Counted arms only (accept.py), pairs = same-index N/O both counted.
usage: python3 summary.py RECEIPTS_DIR OUT_JSON"""
import json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1]); out = {}
M = ("whole_s_per_fch", "forecast_only_s", "d03_step_mean")
pairs, arms = [], {}
for s in ("S1", "S2"):
    acc = json.loads((R / f"{s}_ACCEPT.json").read_text()); det = json.loads((R / f"{s}_DETAIL24.json").read_text())
    ver = json.loads((R / f"{s}_VERDICT.json").read_text())
    counted = [n for n in acc["counted"] if n in det["arms"] and det["arms"][n]]
    for n in counted:
        arms[f"{s}.{n}"] = {k: det["arms"][n][k] for k in M + ("rss_gib", "vram_pid_peak_mib", "cli_main_wait_share",
                                                                 "cgroup_cpu_pressure_s", "foreign_agg_mean_pct", "user_s")}
    for i in "12345":
        if f"N{i}" in counted and f"O{i}" in counted:
            pairs.append({"pair": f"{s}.N{i}/O{i}", **{m: round(det["arms"][f"N{i}"][m] / det["arms"][f"O{i}"][m], 4) for m in M}})
    out[s] = {"verdict": ver["verdict"], **{k: ver["all_counted_24h"]["metrics"]["whole_s_per_fch"].get(k)
                                            for k in ("pooled_N_over_O", "AA_O", "BB_N", "S")}, "counted": counted}
out["pairs"] = pairs
out["pair_stats"] = {m: dict(n=len(pairs), mean=round(st.mean(p[m] for p in pairs), 4), median=round(st.median(p[m] for p in pairs), 4),
                             min=min(p[m] for p in pairs), max=max(p[m] for p in pairs)) for m in M}
for k in ("rss_gib", "vram_pid_peak_mib"):
    out[f"{k}_mean"] = {kind: round(st.mean(a[k] for n, a in arms.items() if n.split(".")[1][0] == kind and a[k]), 3) for kind in "NO"}
out["arms"] = arms
out["note"] = "descriptive only; pre-registered verdicts: S1 PASS (narrow), S2 INCONCLUSIVE (range spread from host-load drift)"
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("S1", "S2", "pair_stats", "rss_gib_mean", "vram_pid_peak_mib_mean")}, indent=1))
for p in pairs: print(p)
