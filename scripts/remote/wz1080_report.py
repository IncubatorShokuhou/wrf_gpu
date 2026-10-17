#!/usr/bin/env python3
"""Summarize the wz1080 priority test (Swiss d01: gpuwrf on the GTX 1080 Ti vs original CPU-WRF on the box CPU).

python wz1080_report.py <dir-with-box-results> [--out report.md]

Expects, per run, the files the probes write next to each output dir: <run>.timing.json, optional <run>.d6.json,
<run>.vram.log and <run>.log (copy them from ~/wrf_gpu2_runs/wz1080/ on the box). Prints a markdown table.
All numbers are [M] on wz1080 (debug box, not a benchmark/gate); D6 = frozen manifest vs the RD11 CPU-WRF reference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

KEY = ("T2", "PSFC", "U10", "V10", "RAINNC", "T", "U", "V", "W", "QVAPOR")


def worst(field: dict) -> float | None:
    vals = [b.get("rmse") for b in field.get("by_lead", []) if b.get("lead_h", 0) > 0 and b.get("rmse") is not None]
    return max(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for tj in sorted(glob.glob(os.path.join(a.dir, "*.timing.json"))):
        run = tj[: -len(".timing.json")]
        name = os.path.basename(run)
        t = json.load(open(tj))
        row = {"run": name, "hours": t.get("forecast_hours"), "whole": t.get("whole_s_per_fch"),
               "steady": t.get("steady_median_s_per_fch"), "segs": t.get("segments_s_per_fch"),
               "first_frame_s": (t.get("frames") or [{}])[0].get("t_s")}
        if os.path.exists(run + ".vram.log"):
            vals = [int(x) for x in open(run + ".vram.log").read().split() if x.strip().isdigit()]
            row["vram_peak_mib"] = max(vals) if vals else None
        if os.path.exists(run + ".d6.json"):
            d = json.load(open(run + ".d6.json"))
            s = d["summaries"]
            row["d6"] = f'{s["verdict"]} ({s["tolerance_failure_count"]} fail / {d["tolerances"]["field_count"]} bounded, ' \
                        f'{s["comparable_field_count"]} compared)'
            row["rmse"] = {k: worst(d["field_summaries"][k]) for k in KEY if k in d["field_summaries"]}
        log = run + ".log"
        if os.path.exists(log):
            txt = open(log, errors="replace").read()
            m = re.findall(r"(Triton support is only enabled[^\n]*|FailedPrecondition[^\n]*|Traceback)", txt)
            if m:
                row["log_flag"] = m[0][:160]
        rows.append(row)
    out = ["| run | fc h | whole s/fc-h | steady s/fc-h | segments | 1st frame s | VRAM peak MiB | D6 vs CPU-WRF ref |",
           "|---|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        out.append(f'| {r["run"]} | {r["hours"]} | {r["whole"]} | {r["steady"]} | {r["segs"]} | {r["first_frame_s"]} | '
                   f'{r.get("vram_peak_mib", "")} | {r.get("d6", "n/a")} |')
    out.append("")
    out.append("Worst-lead RMSE vs RD11 CPU-WRF (D6 bounds: T2/U10/V10 1.5, PSFC 120 Pa, RAINNC 1 mm, T 1.5, U/V 1.8, W 0.30, QV 1e-3):")
    for r in rows:
        if r.get("rmse"):
            out.append(f'- {r["run"]}: ' + ", ".join(f"{k} {v:.4g}" for k, v in r["rmse"].items() if v is not None))
        if r.get("log_flag"):
            out.append(f'- {r["run"]} log: {r["log_flag"]}')
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
