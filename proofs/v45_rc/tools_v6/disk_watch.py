"""[v45-rc v0.4.5 copy of the w43 disk_watch.py; ONLY change: devices are given as LABEL=DEV and records are keyed by
the LABEL ("data" = the <DATA_ROOT> disk, resolved by the session from df at run time). Reason [M 2026-10-24]: after the reboot
<DATA_ROOT> is nvme1n1 and / is nvme3n1, so the hard-coded w43 key nvme3n1 would now measure the ROOT disk.]
<DATA_ROOT> I/O latency per ATTRIB2 arm (manager 2026-10-18T16:17:11Z: log await per arm, await > 50 ms = DISCLOSED).
Runs BESIDE the frozen ATTRIB2 session (tools_c9be5bcd32 stays immutable), pinned to core 28.

usage: python3 disk_watch.py sample OUT_JSONL data=DEV [root=DEV ...]
       python3 disk_watch.py summarize OUT_JSONL ATTRIB_DIR OUT_JSON
sample: every 1 s one JSONL line per interval from /proc/diskstats deltas: ios, read/write MB/s, await_ms =
        d(read_ms + write_ms) / d(reads + writes) (iostat r_await/w_await combined), util % = d(io_ticks) / dt.
summarize: for every run dir under ATTRIB_DIR/runs (accepted and aborted) the window t0_unix..t_end_unix (aborted arms:
        dir mtime as end): interval-weighted await = sum d(ms) / sum d(ios), p95 of the 1 s awaits, seconds > 50 ms,
        max 10 s-window await, MB/s; disposition DISCLOSED iff the arm-window await on "data" > 50 ms, else ok.
"""
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read():
    out = {}
    with open("/proc/diskstats") as h:
        for line in h:
            f = line.split()
            out[f[2]] = dict(r=int(f[3]), rs=int(f[5]), rms=int(f[6]), w=int(f[7]), ws=int(f[9]), wms=int(f[10]), tick=int(f[12]))
    return out


def sample(path, devs):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    prev, pt = read(), time.time()
    with open(path, "a") as log:
        while not stop:
            time.sleep(max(0.0, 1.0 - (time.time() - pt)))
            cur, now = read(), time.time()
            dt = now - pt
            rec = dict(t=round(now, 3), utc=datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"), dt=round(dt, 3))
            for label, d in devs:
                a, b = prev[d], cur[d]
                ios = (b["r"] - a["r"]) + (b["w"] - a["w"])
                ms = (b["rms"] - a["rms"]) + (b["wms"] - a["wms"])
                rec[label] = dict(dev=d, ios=ios, ms=ms, await_ms=round(ms / ios, 2) if ios else 0.0,
                              r_mbs=round((b["rs"] - a["rs"]) * 512 / 1e6 / dt, 2), w_mbs=round((b["ws"] - a["ws"]) * 512 / 1e6 / dt, 2),
                              util=round(100.0 * (b["tick"] - a["tick"]) / (1000.0 * dt), 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            prev, pt = cur, now


def summarize(path, attrib, out):
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    devs = [k for k in recs[0] if k not in ("t", "utc", "dt")] if recs else []
    res = {}
    for d in sorted(p for p in (Path(attrib) / "runs").iterdir() if p.is_dir()):
        try:
            t0 = float((d / "t0_unix").read_text())
        except OSError:
            continue
        t1 = float((d / "t_end_unix").read_text()) if (d / "t_end_unix").exists() else d.stat().st_mtime
        win = [r for r in recs if t0 <= r["t"] - r["dt"] and r["t"] <= t1 + 1e-6]
        arm = dict(t0_utc=datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"), dur_s=round(t1 - t0, 1),
                   samples=len(win))
        for dev in devs:
            ios = sum(r[dev]["ios"] for r in win); ms = sum(r[dev]["ms"] for r in win)
            aw = sorted(r[dev]["await_ms"] for r in win if r[dev]["ios"])
            w10 = [sum(r[dev]["ms"] for r in win[i:i + 10]) / max(1, sum(r[dev]["ios"] for r in win[i:i + 10]))
                   for i in range(max(1, len(win) - 9))] if win else [0.0]
            arm[dev] = dict(await_ms=round(ms / ios, 2) if ios else 0.0, p95_1s_ms=aw[int(0.95 * (len(aw) - 1))] if aw else 0.0,
                            seconds_over_50ms=sum(1 for x in aw if x > 50), max_10s_await_ms=round(max(w10), 2),
                            w_mbs=round(sum(r[dev]["w_mbs"] * r["dt"] for r in win) / max(1e-9, sum(r["dt"] for r in win)), 2),
                            r_mbs=round(sum(r[dev]["r_mbs"] * r["dt"] for r in win) / max(1e-9, sum(r["dt"] for r in win)), 2),
                            util_mean=round(sum(r[dev]["util"] for r in win) / max(1, len(win)), 1))
        arm["disposition"] = "DISCLOSED (await > 50 ms)" if arm.get("data", {}).get("await_ms", 0) > 50 else "ok"
        res[d.name] = arm
    Path(out).write_text(json.dumps(dict(log=path, rule="arm-window data (<DATA_ROOT>) await > 50 ms -> DISCLOSED", arms=res), indent=1) + "\n")
    for k, v in res.items():
        m = v.get("data", {})
        print(k, v["dur_s"], "await", m.get("await_ms"), "p95", m.get("p95_1s_ms"), ">50ms s", m.get("seconds_over_50ms"),
              "max10s", m.get("max_10s_await_ms"), "w MB/s", m.get("w_mbs"), v["disposition"])


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        pairs = [tuple(a.split("=", 1)) for a in sys.argv[3:]]
        assert pairs and all(len(x) == 2 for x in pairs) and pairs[0][0] == "data", "usage: sample OUT data=DEV [root=DEV]"
        sample(sys.argv[2], pairs)
    else:
        summarize(sys.argv[2], sys.argv[3], sys.argv[4])
