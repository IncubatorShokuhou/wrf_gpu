#!/usr/bin/env python3
"""Wall timing of a forecast from wrfout file mtimes (works for CPU-WRF and gpuwrf output alike).

python wrfout_timing.py <outdir> <t0_epoch> [--domain d01] [--out timing.json]

t0 = process start (epoch seconds, taken by the launcher right before the model command).
Reports: per-frame offset from t0, last-file wall, whole-run s/fc-h (= last-file wall / forecast hours),
per-hour segments and their median after the first forecast hour ("steady"). Lead 0 frames (initial
history) are listed but not counted as forecast hours.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("t0", type=float)
    ap.add_argument("--domain", default="d01")
    ap.add_argument("--out")
    a = ap.parse_args()
    prefix = f"wrfout_{a.domain}_"
    files = sorted(f for f in os.listdir(a.outdir) if f.startswith(prefix))
    if not files:
        print(json.dumps({"ok": False, "error": f"no {prefix}* in {a.outdir}"}))
        return 1
    stamps = [dt.datetime.strptime(f[len(prefix):len(prefix) + 19], "%Y-%m-%d_%H:%M:%S") for f in files]
    init = stamps[0]
    frames = []
    for f, s in zip(files, stamps):
        frames.append({"file": f, "lead_h": (s - init).total_seconds() / 3600.0,
                       "t_s": round(os.stat(os.path.join(a.outdir, f)).st_mtime - a.t0, 3)})
    fc = [fr for fr in frames if fr["lead_h"] > 0]
    hours = fc[-1]["lead_h"] if fc else 0.0
    res = {"ok": bool(fc), "outdir": a.outdir, "t0": a.t0, "frames": frames, "forecast_hours": hours}
    if fc:
        last = fc[-1]["t_s"]
        res["last_file_wall_s"] = last
        res["whole_s_per_fch"] = round(last / hours, 3)
        segs, prev = [], None
        for fr in frames:
            if prev is not None and fr["lead_h"] > 0:
                segs.append(round((fr["t_s"] - prev["t_s"]) / (fr["lead_h"] - prev["lead_h"]), 3))
            prev = fr
        res["segments_s_per_fch"] = segs  # segment k = lead k-1 -> k (first one includes init/compile if no lead-0 file)
        if len(segs) >= 2:
            res["steady_median_s_per_fch"] = round(statistics.median(segs[1:]), 3)
    text = json.dumps(res, indent=1)
    print(text)
    keys = ("forecast_hours", "last_file_wall_s", "whole_s_per_fch", "steady_median_s_per_fch", "segments_s_per_fch")
    print("TIMING_SUMMARY " + json.dumps({k: res.get(k) for k in keys}))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
