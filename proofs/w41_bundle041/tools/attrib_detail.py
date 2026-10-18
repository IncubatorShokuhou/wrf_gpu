"""Per-arm anatomy of RW3 attribution arms from raw receipts (CPU, reads files) -- critic/manager 14:16-14:17Z asks.

usage: python3 attrib_detail.py ATTRIB_DIR OUT_JSON [ARM ...]      (default arms N1 O1 C1 N2 O2 C2)
Arm name = <kind><rep>, kind N (v0.4.1 bundle) / O (v0.4.0 bundle) / C (snapshot 06ab6c). Per arm:
  wall        consumer wall (session.log 'ARM x kind' -> 'ARM x rc='), process wall (t_end - t0), receipt total and
              forecast-only wall, pre-step = total - forecast-only, AOT prefetch elapsed
  files       E162 original mtimes vs t0: t0 frame, lead 1, last file; whole s/fc-h = last/24
  stepping    d03 hourly publication cadence leads 2-24 (mean, median, leads 2-14 mean); d01 cadence median
  writer      per lead: d01 publication lag behind d02 (the d01 frame is the last of each hour) -> max lag + lead
  host        /usr/bin/time: user, sys, voluntary/involuntary context switches, fs in/out blocks, max RSS
  device      per-PID VRAM peak (nvidia-smi compute-apps 0.5 s); core gate disposition; optional core_watch summary
Ratios N/O per rep and pooled (mean of reps), A/A spreads max/min - 1 over reps of one kind.
"""
import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = sys.argv[3:] or ["N1", "O1", "C1", "N2", "O2", "C2"]
R = A / "runs"
log = (A / "session.log").read_text().splitlines() if (A / "session.log").exists() else []
gate = {}
for line in (A / "core_gate.tsv").read_text().splitlines() if (A / "core_gate.tsv").exists() else []:
    f = line.split("\t")
    gate[f[0]] = "\t".join(f[2:])


def utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def consumer_wall(name):
    t = {}
    for line in log:
        m = re.match(rf"(\S+) ARM {name} (kind|rc=)", line)
        if m:
            t[m.group(2)] = utc(m.group(1))
    return round(t["rc="] - t["kind"], 1) if len(t) == 2 else None


def timev(path):
    txt = path.read_text()
    g = lambda k: (re.search(rf"{k}: ([\d.]+)", txt) or [None, None])[1]
    return dict(user_s=float(g("User time \\(seconds\\)")), sys_s=float(g("System time \\(seconds\\)")),
                vol_ctx=int(g("Voluntary context switches")), invol_ctx=int(g("Involuntary context switches")),
                fs_in_blocks=int(g("File system inputs")), fs_out_blocks=int(g("File system outputs")),
                rss_gib=round(int(g("Maximum resident set size \\(kbytes\\)")) / 2**20, 3))


def arm(name):
    d = R / name
    try:
        t0 = float((d / "t0_unix").read_text()); t1 = float((d / "t_end_unix").read_text())
        mt = json.loads((d / "mtimes.json").read_text())
        rec = json.loads((d / "proof" / "nested_pipeline_run.json").read_text())
    except (OSError, ValueError):
        return None
    dom = {k: sorted(v - t0 for f, v in mt.items() if f.startswith(f"wrfout_{k}_")) for k in ("d01", "d02", "d03")}
    s3 = [dom["d03"][i] - dom["d03"][i - 1] for i in range(2, 25)]
    s1 = [dom["d01"][i] - dom["d01"][i - 1] for i in range(2, 25)]
    lag = [dom["d01"][i] - dom["d02"][i] for i in range(1, 25)]
    peak = 0
    for row in (d / "vram_apps.csv").read_text().splitlines() if (d / "vram_apps.csv").exists() else []:
        f = [x.strip() for x in row.split(",")]
        if len(f) >= 3 and f[2].isdigit():
            peak = max(peak, int(f[2]))
    watch = json.loads((d / "core_watch_summary.json").read_text()) if (d / "core_watch_summary.json").exists() else None
    out = dict(rc=(d / "rc.txt").read_text().strip(), files=len(mt), consumer_wall_s=consumer_wall(name),
               process_wall_s=round(t1 - t0, 1), receipt_total_s=round(rec["wall_clock_total_s"], 1),
               forecast_only_s=round(rec["wall_clock_forecast_only_s"], 1),
               pre_step_s=round(rec["wall_clock_total_s"] - rec["wall_clock_forecast_only_s"], 1),
               aot_prefetch_s=round(rec["metadata"]["aot_prefetch"]["elapsed_s"], 1),
               t0_frame_s=round(min(v[0] for v in dom.values()), 1), lead1_d03_s=round(dom["d03"][1], 1),
               last_file_s=round(max(v[-1] for v in dom.values()), 1),
               whole_s_per_fch=round(max(v[-1] for v in dom.values()) / 24, 4),
               d03_step_mean=round(st.mean(s3), 3), d03_step_median=round(st.median(s3), 3),
               d03_step_mean_leads2_14=round(st.mean(s3[:13]), 3), d01_step_median=round(st.median(s1), 3),
               d03_segments=[round(x, 1) for x in s3],
               d01_lag_max_s=round(max(lag), 2), d01_lag_max_lead=1 + lag.index(max(lag)), d01_lag_median_s=round(st.median(lag), 2),
               vram_pid_peak_mib=peak or None, core_gate=gate.get(name), core_watch=watch)
    out.update(timev(d / "time.txt"))
    return out


arms = {n: arm(n) for n in NAMES}
METRICS = ("consumer_wall_s", "process_wall_s", "whole_s_per_fch", "forecast_only_s", "pre_step_s", "aot_prefetch_s",
           "d03_step_mean", "d03_step_median", "d03_step_mean_leads2_14", "user_s", "invol_ctx", "rss_gib", "vram_pid_peak_mib")
kinds = sorted({n[0] for n in NAMES})
reps = sorted({n[1:] for n in NAMES})
ratios, spreads = {}, {}
for m in METRICS:
    r = {}
    for i in reps:
        n, o = arms.get("N" + i), arms.get("O" + i)
        if n and o and n.get(m) and o.get(m):
            r[f"N{i}/O{i}"] = round(n[m] / o[m], 4)
    pooled = {}
    for k in kinds:
        v = [arms[k + i][m] for i in reps if arms.get(k + i) and arms[k + i].get(m)]
        if v:
            pooled[k] = st.mean(v)
            if len(v) > 1:
                spreads[f"{k} {m}"] = round(max(v) / min(v) - 1, 4)
    for a, b in (("N", "O"), ("C", "O"), ("N", "C")):
        if a in pooled and b in pooled:
            r[f"pooled {a}/{b}"] = round(pooled[a] / pooled[b], 4)
    ratios[m] = r
res = dict(attrib_dir=str(A), arms=arms, ratios=ratios, aa_spreads_max_over_min=spreads)
OUT.write_text(json.dumps(res, indent=1) + "\n")
cols = ("consumer_wall_s", "process_wall_s", "aot_prefetch_s", "t0_frame_s", "forecast_only_s", "whole_s_per_fch",
        "d03_step_mean", "d03_step_median", "d01_lag_max_s", "d01_lag_max_lead", "user_s", "invol_ctx", "rss_gib", "vram_pid_peak_mib")
print("arm " + " ".join(cols))
for n, a in arms.items():
    print(n, " ".join(str(a[c]) if a else "-" for c in cols), "|", (a or {}).get("core_gate"))
print(json.dumps({"ratios": {m: ratios[m] for m in ("whole_s_per_fch", "forecast_only_s", "aot_prefetch_s", "d03_step_mean", "user_s")},
                  "spreads": {k: v for k, v in spreads.items() if any(x in k for x in ("whole", "forecast_only", "d03_step_mean", "prefetch"))}}, indent=1))
