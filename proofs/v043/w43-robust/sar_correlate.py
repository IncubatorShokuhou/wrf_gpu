"""Correlate E2 GPU-arm wall time per case (GPU_SERIES_METRICS_V2 attempts) with sar 10-min means over the run window:
core-4 %iowait (the GPU arm's only host core) and <DATA_ROOT> disk nvme3n1 %util/await. sar times are local (WEST = UTC+1)."""
import datetime as dt, json, sys
from pathlib import Path
A = Path(__file__).resolve().parent
UTC_OFF = dt.timedelta(hours=1)
DAY = dt.date(2026, 10, 20)

def parse(path, key_col, want):
    rows = {}
    for line in open(path):
        f = line.split()
        if len(f) < 3 or f[0] in ("Average:", "Linux") or ":" not in f[0]:
            continue
        try:
            t = dt.datetime.combine(DAY, dt.time.fromisoformat(f[0]))
        except ValueError:
            continue
        rows.setdefault(t, {})
        if want(f):
            rows[t][key_col] = f
    return rows

cpu = {t: float(v["c"][5]) for t, v in parse(A / "sar/sa20_cpu4.txt", "c", lambda f: f[1] == "4").items() if "c" in v}
disk = {t: (float(v["d"][8]), float(v["d"][7])) for t, v in parse(A / "sar/sa20_disk.txt", "d", lambda f: f[-1] == "nvme3n1").items() if "d" in v}

def window_mean(series, t0, t1, idx=None):
    # sar row at T covers (T-10min, T]; weight by overlap
    num = den = 0.0
    for T, v in series.items():
        a, b = max(t0, T - dt.timedelta(minutes=10)), min(t1, T)
        w = (b - a).total_seconds()
        if w > 0:
            num += w * (v if idx is None else v[idx]); den += w
    return num / den if den else float("nan")

m = json.load(open(A / "data/R04_try/GPU_SERIES_METRICS_V2.json"))
out = []
for a in m["gpu_usage"]["attempts"]:
    t0 = dt.datetime.fromisoformat(a["started_utc"]).replace(tzinfo=None) + UTC_OFF
    t1 = t0 + dt.timedelta(seconds=a["seconds"])
    out.append({"case": a["case"], "start_utc": a["started_utc"][:19], "wall_s": round(a["seconds"], 1), "rc": a["rc"],
                "core4_iowait_pct": round(window_mean(cpu, t0, t1), 1),
                "nvme3_util_pct": round(window_mean(disk, t0, t1, 0), 1), "nvme3_await_ms": round(window_mean(disk, t0, t1, 1), 1)})
ok = [r for r in out if r["rc"] == 0 or r["wall_s"] > 1000]
import statistics as st
xs, ys = [r["core4_iowait_pct"] for r in ok], [r["wall_s"] for r in ok]
xd = [r["nvme3_util_pct"] for r in ok]
print(json.dumps(out, indent=0))
print("n", len(ok), "pearson(core4 iowait, wall)", round(st.correlation(xs, ys), 3), "pearson(nvme3 util, wall)", round(st.correlation(xd, ys), 3))
json.dump({"rows": out, "n": len(ok), "pearson_iowait_wall": st.correlation(xs, ys), "pearson_util_wall": st.correlation(xd, ys)},
          open(A / "sar_correlation.json", "w"), indent=1)
