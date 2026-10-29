"""CPU contention seen BY the arm itself (disclosure beside core_watch; never an abort). v45-rc v0.4.5 variant of the
w43 sched_watch.py: the arm has no own systemd scope any more (one RAM-slot scope holds the whole session), so the cgroup
is given as a directory and the summary reports per-arm DELTAS of the cumulative cgroup counters (last - first sample).

usage: python3 sched_watch.py CGROUP_DIR OUT_JSONL SUMMARY_JSON [PERIOD_S]
Every PERIOD_S (default 1 s) until SIGTERM: cgroup cpu.pressure 'some' total, cpu.stat usage/user/system, and the main-thread
/proc schedstat (run ns, runqueue-wait ns) of every python3 process in the cgroup whose cmdline contains 'gpuwrf.cli'.
Summary: last = per-arm deltas (pressure_some_total_s, usage_s, user_s, system_s), cli_main_thread = the gpuwrf.cli process
with the largest main-thread run time, cli_main_wait_share = its runqueue wait / run time.
"""
import json
import signal
import sys
import time
from datetime import datetime, timezone

cg, out_jsonl, summary_json = sys.argv[1], sys.argv[2], sys.argv[3]
period = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
stop = False
signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("stop", True))
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))


def rd(p):
    try:
        with open(p) as h:
            return h.read()
    except OSError:
        return None


def sample():
    pr, cs, procs = rd(f"{cg}/cpu.pressure"), rd(f"{cg}/cpu.stat"), rd(f"{cg}/cgroup.procs")
    if pr is None:
        return None
    some = {k: float(v) for k, v in (kv.split("=") for kv in pr.splitlines()[0].split()[1:])}
    stat = dict(l.split() for l in (cs or "").splitlines())
    main = {}
    for pid in (procs or "").split():
        if b"gpuwrf.cli" not in ((rd(f"/proc/{pid}/cmdline") or "").encode()):
            continue
        ss = rd(f"/proc/{pid}/schedstat")
        if ss:
            run, wait, n = map(int, ss.split())
            main[pid] = dict(run_s=run / 1e9, wait_s=wait / 1e9, slices=n)
    return dict(utc=datetime.now(timezone.utc).isoformat(), pressure_some_total_s=some["total"] / 1e6,
                usage_s=int(stat.get("usage_usec", 0)) / 1e6, user_s=int(stat.get("user_usec", 0)) / 1e6,
                system_s=int(stat.get("system_usec", 0)) / 1e6, python_main=main)


first, last, n, mains = None, None, 0, {}
t_start = time.time()
with open(out_jsonl, "a") as out:
    while not stop:
        s = sample()
        if s is not None:
            first = first or s
            last, n = s, n + 1
            mains.update(s["python_main"])
            out.write(json.dumps(s) + "\n")
            out.flush()
        time.sleep(period)
cli = max(mains.values(), key=lambda v: v["run_s"], default=None)
delta = {k: round(last[k] - first[k], 3) for k in ("pressure_some_total_s", "usage_s", "user_s", "system_s")} if last else None
summary = dict(cgroup=cg, samples=n, utc_start=datetime.fromtimestamp(t_start, timezone.utc).isoformat(),
               utc_end=datetime.now(timezone.utc).isoformat(), last=delta, cumulative_last=last and {k: v for k, v in last.items() if k != "python_main"},
               cli_main_thread=cli, cli_main_wait_share=round(cli["wait_s"] / cli["run_s"], 4) if cli and cli["run_s"] else None)
with open(summary_json, "w") as h:
    json.dump(summary, h, indent=1)
