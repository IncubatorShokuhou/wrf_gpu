"""w43-release RW3T verdict (copy of the w41-rw3add verdict.py; changes per manager 2026-10-19T05:59:13Z: PRIMARY metric =
per-arm stepping median d03_step_median (median of the d03 hourly publication intervals, leads 2..H, s per forecast hour);
whole-run s/fc-h is a DISCLOSED secondary: an arm whose arm-window <DATA_ROOT> (nvme3n1) await > 50 ms (disk_watch summary)
is marked whole_run DISTURBED and left out of the whole-run pooled ratio (not a regression); PREREG sha written to the result).

usage: python3 rw3t_verdict.py ACCEPT_JSON DETAIL24_JSON DETAIL72_JSON BYTES_JSON OUT_JSON PREREG_JSON DISK_JSON
PREREG_JSON = the session's own pre-registration (receipts/PREREG.json for session 1, PREREG2.json for session 2); its
sha256 is computed from the file and written into the result (critic cr-w41-rw3add FIX item 1).
Counted 24 h arms only (accept.py). Per metric m (primary whole_s_per_fch; secondary forecast_only_s, d03_step_mean):
pooled N/O = mean(N) / mean(O); A/A = max/min - 1 over counted O arms, B/B the same over N; S = max(A/A, B/B).
PASS iff pooled + S <= 1.10; REAL_REGRESSION iff pooled - S > 1.10; else INCONCLUSIVE (also with < 2 counted arms of a kind).
Strict subset (shadow watcher would also pass the contract 20/50 % rules) reported separately; 72 h pair as one pair.
"""
import hashlib
import json
import statistics as st
import sys

acc, d24, d72, byt = (json.load(open(p)) for p in sys.argv[1:5])
OUT = sys.argv[5]
PREREG_SHA = hashlib.sha256(open(sys.argv[6], "rb").read()).hexdigest()
counted = set(acc["counted"])
METRICS = ("d03_step_median", "whole_s_per_fch", "forecast_only_s", "d03_step_mean", "d01_step_median")
PRIMARY = "d03_step_median"
disk = json.load(open(sys.argv[7]))["arms"] if len(sys.argv) > 7 else {}
DISTURBED = {n for n, v in disk.items() if isinstance(v, dict) and (str(v.get("disposition", "")).startswith("DISCLOSED")
                                                                    or ((v.get("nvme3n1") or {}).get("await_ms") or 0) > 50)}


def block(names):
    arms = {n: d24["arms"][n] for n in names if d24["arms"].get(n)}
    res = {}
    for m in METRICS:
        skip = DISTURBED if m == "whole_s_per_fch" else set()   # whole-run of a disk-DISTURBED arm is disclosed only
        N = [a[m] for n, a in arms.items() if n[0] == "N" and n not in skip]
        O = [a[m] for n, a in arms.items() if n[0] == "O" and n not in skip]
        r = dict(N=N, O=O)
        if skip & set(arms):
            r["excluded_disturbed"] = sorted(skip & set(arms))
        if len(N) >= 1 and len(O) >= 1:
            r["pooled_N_over_O"] = round(st.mean(N) / st.mean(O), 4)
        if len(N) >= 2 and len(O) >= 2:
            aa, bb = max(O) / min(O) - 1, max(N) / min(N) - 1
            S = max(aa, bb)
            p = r["pooled_N_over_O"]
            r.update(AA_O=round(aa, 4), BB_N=round(bb, 4), S=round(S, 4),
                     verdict="PASS" if p + S <= 1.10 else "REAL_REGRESSION" if p - S > 1.10 else "INCONCLUSIVE")
        else:
            r["verdict"] = "INCONCLUSIVE (fewer than 2 counted arms of a kind)"
        res[m] = r
    return dict(arms=sorted(arms), metrics=res)


names24 = [n for n in d24["arms"] if n in counted]
strict = [n for n in names24 if (d24["arms"][n] or {}).get("strict_rules_would_pass")]
pairs = {}
for i in "12345":
    n, o = f"N{i}", f"O{i}"
    if n in counted and o in counted:
        pairs[f"{n}/{o}"] = {m: round(d24["arms"][n][m] / d24["arms"][o][m], 4) for m in METRICS}
p72 = None
if {"N72", "O72"} <= counted and d72["arms"].get("N72") and d72["arms"].get("O72"):
    p72 = {m: round(d72["arms"]["N72"][m] / d72["arms"]["O72"][m], 4) for m in METRICS}
    p72.update(N72={m: d72["arms"]["N72"][m] for m in METRICS}, O72={m: d72["arms"]["O72"][m] for m in METRICS})
res = dict(prereg_sha256=PREREG_SHA, counted=sorted(counted),
           all_counted_24h=block(names24), strict_subset_24h=block(strict), pairs_24h=pairs, pair_72h=p72,
           bytes=byt, primary_metric=PRIMARY, whole_run_disturbed_arms=sorted(DISTURBED),
           disk_rule="arm-window <DATA_ROOT> (nvme3n1) await > 50 ms -> whole-run DISTURBED (manager 05:59:13Z)",
           verdict=block(names24)["metrics"][PRIMARY]["verdict"])
open(OUT, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps({"verdict": res["verdict"], "primary": res["all_counted_24h"]["metrics"][PRIMARY],
                  "whole_run_secondary": res["all_counted_24h"]["metrics"]["whole_s_per_fch"],
                  "disturbed": sorted(DISTURBED), "strict_primary": res["strict_subset_24h"]["metrics"][PRIMARY],
                  "pairs": pairs, "pair_72h": p72}, indent=1))
