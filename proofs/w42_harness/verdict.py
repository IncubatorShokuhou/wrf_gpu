"""w42-harness S1H verdict table (CPU, after summarize.py): what the fixed harness changes, and the new v0.4.1 baselines.

usage: python3 verdict.py SESSION_ROOT S1H_RESULT.json OUT.json
Per PROD arm: stepping = median over segments 2..N-1 of bench_summary stepping_s_per_fc_h (period minus output-callback
wall; the last segment carries the end tail) -- insensitive to wrfout writer back-pressure; seg2+ mean / median rate;
proxy (3 h arms) seg 2-3 mean and seg 2 alone; whole-run s/fc-h on three bases (session launch / harness launch /
measured-process t0). Disclosure per arm: <DATA_ROOT> (nvme3n1) interval-weighted await over the arm window, seconds > 50 ms,
max output callback, core_watch sweep + 20/50 shadow, sched_watch CLI runqueue wait + cgroup PSI. 'disk_contended' =
arm-window await > 50 ms (the disk_watch DISCLOSED rule of the w41 sessions). Ratios N/O (mean of the two arms of a kind)
with A/A and B/B = |x1 - x2| / mean, for all arms and for the arms that are not disk_contended.
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

A, RES, OUT = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
_here = Path(__file__).resolve().parent   # frozen tools dir (copy beside it) or the repo (scripts/bench_summary.py)
_bs = _here / "bench_summary.py" if (_here / "bench_summary.py").is_file() else _here.parents[1] / "scripts" / "bench_summary.py"
spec = importlib.util.spec_from_file_location("bs", _bs)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
rows = [json.loads(x) for x in (A / "disk.jsonl").read_text().splitlines() if x.strip()]


def disk(t0, t1):
    win = [r for r in rows if t0 <= r["t"] - r["dt"] and r["t"] <= t1]
    ios = sum(r["nvme3n1"]["ios"] for r in win)
    aw = [r["nvme3n1"]["ms"] / r["nvme3n1"]["ios"] for r in win if r["nvme3n1"]["ios"]]
    return {"await_ms": round(sum(r["nvme3n1"]["ms"] for r in win) / ios, 1) if ios else 0.0,
            "seconds_over_50ms": sum(a > 50 for a in aw), "max_1s_ms": round(max(aw), 1) if aw else 0.0, "seconds": len(win)}


arms = {}
for name, a in RES["arms"].items():
    d = A / "runs" / name
    s = bs.arm_summary(d / "run" / "receipt.json")
    seg = s["segments"]
    step = [x["stepping_s_per_fc_h"] for x in seg[1:-1]] or [x["stepping_s_per_fc_h"] for x in seg[1:]]
    rate = [x["s_per_fc_h"] for x in seg]
    out_max = max((o["s"] for o in json.loads((d / "run" / "receipt.json").read_text())["outputs"]), default=None)
    dk = disk(float((d / "launch_unix").read_text()), float((d / "t_end_unix").read_text()))
    arms[name] = {"hours": a["hours"], "pool": a["allocator_mode"], "stepping_median": round(statistics.median(step), 3),
                  "seg2plus_mean": a["seg2plus_mean"], "seg2plus_median": a["seg2plus_median"],
                  "proxy_seg2_3": a["proxy_seg2_3"], "seg2": round(rate[1], 3) if len(rate) > 1 else None,
                  "k1": round(rate[0], 3), "whole": a["whole_s_per_fc_h"], "disk": dk,
                  "disk_contended": dk["await_ms"] > 50, "max_output_call_s": out_max, "watch": a["watch"],
                  "jax_peak_bytes": (a["jax_memory_stats"] or {}).get("peak_bytes_in_use"),
                  "jax_bytes_limit": (a["jax_memory_stats"] or {}).get("bytes_limit"), "consumer": a["consumer"]}


def pair(kind, key, clean_only):
    def pick(side):
        out = []
        for i in (1, 2):
            a = arms.get(f"{kind}_{side}{i}")
            if a is None or (clean_only and a["disk_contended"]):
                continue
            v = a
            for k in key:
                v = v[k]
            out.append(v)
        return out
    o, n = pick("O"), pick("N")
    sp = lambda x: round(abs(x[0] - x[1]) / statistics.mean(x) * 100, 2) if len(x) == 2 else None  # noqa: E731
    return {"O": o, "N": n, "N_over_O": round(statistics.mean(n) / statistics.mean(o), 4) if o and n else None,
            "AA_pct": sp(o), "BB_pct": sp(n)}


metrics = {"S stepping_median": ("S", ("stepping_median",)), "S seg2plus_mean": ("S", ("seg2plus_mean",)),
           "S whole session_launch": ("S", ("whole", "session_launch")),
           "S whole measured_process_t0": ("S", ("whole", "measured_process_t0")),
           "P proxy_seg2_3": ("P", ("proxy_seg2_3",)), "P seg2": ("P", ("seg2",)), "P stepping_median": ("P", ("stepping_median",))}
ratios = {m: {"all_arms": pair(k, key, False), "disk_clean_arms": pair(k, key, True)} for m, (k, key) in metrics.items()}
res = {"schema": "w42-harness.s1h.verdict.v1", "arms": arms, "ratios": ratios, "identity": RES["identity"],
       "wn3_par3": RES["wn3_par3"], "aborted_tries": RES["aborted_tries"]}
OUT.write_text(json.dumps(res, indent=1) + "\n")
for m, v in ratios.items():
    print(f"{m:30s} all {v['all_arms']}  | disk-clean {v['disk_clean_arms']}")
for n, a in arms.items():
    print(f"{n:5s} {a['pool']:7s} step {a['stepping_median']} seg2+ {a['seg2plus_mean']} proxy {a['proxy_seg2_3']} "
          f"whole {a['whole']['session_launch']} disk {a['disk']['await_ms']} ms{' CONTENDED' if a['disk_contended'] else ''}")
