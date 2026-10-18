"""RW3 attribution report from attrib.sh receipts (CPU, reads files): N = v0.4.1 bundle, O = v0.4.0 bundle, C = 06ab6c9156
(v0.4.1 minus the fused-namelist-arguments change). WN3 0227 3-nest 24 h warm, interleaved N1 O1 C1 N2 O2 C2.

usage: python3 attrib_report.py ATTRIB_DIR OUT_JSON
Per arm: whole-run s/fc-h = (last original wrfout mtime - arm t0)/24 (E162), d01 publication cadence median/mean over leads
2-24 (stepping incl. output), process wall, per-PID VRAM peak, host max RSS, core-gate disposition. Ratios: N/O per pair,
C/O and N/C per pair, A/A spreads |X1/X2 - 1|. Verdict rule (manager 12:39:43Z): > 10 % after a clean gate = real.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = A / "runs"
gate = {}
for line in (A / "core_gate.tsv").read_text().splitlines() if (A / "core_gate.tsv").exists() else []:
    f = line.split("\t")
    gate[f[0]] = f[2] + (f" ({f[3]})" if len(f) > 3 else "")


def arm(name, hours=24):
    d = R / name
    try:
        t0 = float((d / "t0_unix").read_text()); t1 = float((d / "t_end_unix").read_text())
        mt = json.loads((d / "mtimes.json").read_text())
    except (OSError, ValueError):
        return None
    d01 = sorted(v for k, v in mt.items() if k.startswith("wrfout_d01_"))
    steps = [b - a for a, b in zip(d01[1:], d01[2:])]
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (d / "time.txt").read_text())
    peak = 0
    try:
        for row in csv.reader(open(d / "vram_apps.csv")):
            if len(row) >= 3 and row[2].strip().isdigit():
                peak = max(peak, int(row[2]))
    except OSError:
        pass
    aot = [l for l in (d / "run.log").read_text().splitlines() if "[gpuwrf:nested-aot]" in l]
    return dict(rc=(d / "rc.txt").read_text().strip(), files=len(mt), whole_s_per_fch=round((max(mt.values()) - t0) / hours, 4),
                step_median=round(statistics.median(steps), 4), step_mean=round(statistics.mean(steps), 4),
                lead1_s=round(d01[1] - t0, 2) if len(d01) > 1 else None, process_wall_s=round(t1 - t0, 2),
                vram_pid_peak_mib=peak or None, rss_gib=round(int(rss.group(1)) / 2**20, 3) if rss else None,
                all_loaded=bool(aot) and all("loaded=true" in l for l in aot), core_gate=gate.get(name))


arms = {n: arm(n) for n in ("N1", "O1", "C1", "N2", "O2", "C2")}
arms["C0_cold3h"] = arm("C0", hours=3)
res = {"arms": arms, "ratios": {}, "spreads": {}}
for metric in ("whole_s_per_fch", "step_median", "step_mean"):
    r = {}
    for i in ("1", "2"):
        n, o, c = arms.get("N" + i), arms.get("O" + i), arms.get("C" + i)
        if n and o:
            r[f"N{i}/O{i}"] = round(n[metric] / o[metric], 4)
        if c and o:
            r[f"C{i}/O{i}"] = round(c[metric] / o[metric], 4)
        if n and c:
            r[f"N{i}/C{i}"] = round(n[metric] / c[metric], 4)
    pooled = {}
    for x in ("N", "O", "C"):
        v = [arms[x + i][metric] for i in ("1", "2") if arms.get(x + i)]
        if v:
            pooled[x] = statistics.mean(v)
        if len(v) == 2:
            res["spreads"][f"{x}1/{x}2 {metric}"] = round(abs(v[0] / v[1] - 1), 4)
    for a, b in (("N", "O"), ("C", "O"), ("N", "C")):
        if a in pooled and b in pooled:
            r[f"pooled {a}/{b}"] = round(pooled[a] / pooled[b], 4)
    res["ratios"][metric] = r
clean = all(arms.get(n) and (arms[n].get("core_gate") or "").startswith("free") for n in ("N1", "O1", "N2", "O2"))
nO = res["ratios"]["whole_s_per_fch"].get("pooled N/O")
res["gate_clean_all_NO_arms"] = clean
res["verdict"] = ("REAL >10 %" if nO and nO > 1.10 and clean else
                  ("> 10 % but gate NOT clean (disclosed)" if nO and nO > 1.10 else
                   ("<= 10 %" if nO else "INCOMPLETE")))
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({"ratios": res["ratios"], "spreads": res["spreads"], "gate_clean": clean, "verdict": res["verdict"]}, indent=1))
