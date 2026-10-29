"""v45-rc v0.4.5: final real-world gate table RW1-RW4 (+ Swiss branch, E41, icloud=0, RW3 fidelity/speed/memory/compile).
CPU, reads receipts only. Rules = PREREG.json (rules.*). Missing evidence = MISSING/INCOMPLETE, never PASS.

usage: python3 -I rw_verdict_v044.py OUT_JSON SHORT_VERDICT.json RW3T_VERDICT.json RW3T_DETAIL24.json RW3T_DETAIL72.json
                                     FLOOR72_N72.json D6_N72_DIR BYTES_RW3.json PREREG.json
"""
import hashlib
import json
import statistics as st
import sys
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


out = Path(sys.argv[1])
short, rw3t, d24, d72, floor, d6dir, byt, prereg = (sys.argv[i] for i in range(2, 10))
short, rw3t, d24, d72, floor, byt = (load(x) for x in (short, rw3t, d24, d72, floor, byt))
d6 = load(Path(d6dir) / "summary.json")
res = dict(prereg_sha256=hashlib.sha256(Path(prereg).read_bytes()).hexdigest() if Path(prereg).is_file() else None)
for k in ("RW1", "RW2", "RW4", "swiss_identity", "E41", "icloud0", "compile_s_cold_minus_warm"):
    res[k] = (short or {}).get(k, "MISSING")
pref = ((byt or {}).get("pairs") or {}).get("PREFIX N72 vs A3") or {}
if isinstance(res["RW1"], dict):
    res["RW1"]["new_prefix_checks"] = dict(res["RW1"].get("new_prefix_checks") or {}, **{"PREFIX N72 vs A3": pref.get("verdict", "MISSING")})
# RW3 fidelity: frozen floor rule + integrity; the QGRAUP-trace class is a manager decision (v0.4.2 precedent), never auto-PASS
integ = None
qgraup_only = False
if d6:
    integ = d6.get("output_integrity_pass")
    if integ is False:
        doms = d6.get("domains") or {}
        findings = {d: (v.get("output_integrity") or {}) for d, v in doms.items()}
        bad = {d: f for d, f in findings.items() if f.get("pass") is False}
        f3 = bad.get("d03") or {}
        qgraup_only = bool(bad) and set(bad) == {"d03"} and set(f3.get("degenerate_fields") or {}) == {"QGRAUP"} and \
            all(max(x.get("cpu_range", [0, 1])) < 1e-9 for x in f3["degenerate_fields"]["QGRAUP"]) and \
            not f3.get("missing_variables_vs_cpu") and not ((f3.get("global_attrs") or {}).get("by_frame"))
fid = dict(binding72_A3=(floor or {}).get("binding72_A3", "MISSING"), raw72=(floor or {}).get("raw72", "MISSING"),
           floor_counts=(floor or {}).get("counts"), coverage_complete=(floor or {}).get("coverage_complete"),
           output_integrity=integ if d6 else "MISSING", qgraup_trace_class_only=qgraup_only)
fid["verdict"] = ("MISSING" if not floor or not d6 else
                  "PASS" if fid["binding72_A3"] == "PASS" and integ is True else
                  "MANAGER_DECISION (QGRAUP trace class, v0.4.2 precedent)" if fid["binding72_A3"] == "PASS" and qgraup_only else "STOP")
res["RW3_fidelity"] = fid
res["RW3_speed"] = {k: (rw3t or {}).get(k) for k in ("verdict", "primary_metric", "counted", "pairs_24h", "pair_72h")} if rw3t else "MISSING"
mem = {}
if d24:
    arms = d24.get("arms") or {}
    for m in ("rss_gib", "vram_pid_peak_mib"):
        N = [a[m] for n, a in arms.items() if a and n.startswith("N") and a.get(m)]
        O = [a[m] for n, a in arms.items() if a and n.startswith("O") and a.get(m)]
        if N and O:
            r = st.mean(N) / st.mean(O)
            mem[m] = dict(N=N, O=O, ratio=round(r, 4), over_10pct=r > 1.10)
if d72:
    a = d72.get("arms") or {}
    if a.get("N72") and a.get("O72"):
        mem["pair_72h"] = {m: dict(N72=a["N72"].get(m), O72=a["O72"].get(m)) for m in ("rss_gib", "vram_pid_peak_mib", "whole_s_per_fch", "d03_step_median")}
res["RW3_memory"] = mem or "MISSING"
stop = []
if isinstance(res["RW3_speed"], dict) and res["RW3_speed"].get("verdict") == "REAL_REGRESSION":
    stop.append("RW3 speed REAL_REGRESSION")
stop += [f"RW3 memory {m} > 1.10" for m, v in mem.items() if isinstance(v, dict) and v.get("over_10pct")]
c = res["compile_s_cold_minus_warm"]
if isinstance(c, dict) and c.get("over_10pct"):
    stop.append("compile s > 1.10 x v0.4.4 (cross-session basis: manager may rule)")
if fid["verdict"] == "STOP":
    stop.append("RW3 fidelity")
if isinstance(res["swiss_identity"], dict) and res["swiss_identity"].get("verdict") in ("STOP", "STOP_DIAGNOSE"):
    stop.append("Swiss identity/D6")
res["stop_items"] = stop
gates = dict(RW1=(res["RW1"] or {}).get("verdict") if isinstance(res["RW1"], dict) else "MISSING",
             RW2=(res["RW2"] or {}).get("verdict") if isinstance(res["RW2"], dict) else "MISSING",
             RW3_fidelity=fid["verdict"], RW3_speed=(res["RW3_speed"] or {}).get("verdict") if isinstance(res["RW3_speed"], dict) else "MISSING",
             RW4=(res["RW4"] or {}).get("verdict") if isinstance(res["RW4"], dict) else "MISSING",
             swiss=(res["swiss_identity"] or {}).get("verdict") if isinstance(res["swiss_identity"], dict) else "MISSING",
             E41=(res["E41"] or {}).get("verdict") if isinstance(res["E41"], dict) else "MISSING")
res["table"] = gates
res["overall"] = ("STOP" if stop else "PASS" if all(v in ("PASS", "IDENTICAL", "INCOMPLETE_AS_V043", "HARD_PASS_WITH_DIAGNOSTICS", "HARD_PASS_HLO_UNKNOWN") for v in gates.values())
                  else "MANAGER_DECISION_OR_INCOMPLETE")
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps(dict(table=gates, stop=stop, overall=res["overall"]), indent=1))
