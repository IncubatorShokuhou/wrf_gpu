"""CPU contention seen BY the arm itself (disclosure beside core_watch; never an abort).

usage: python3 sched_watch.py UNIT OUT_JSONL SUMMARY_JSON [PERIOD_S]
Every PERIOD_S (default 1 s) while the systemd --user scope UNIT exists: cgroup cpu.pressure 'some' total (us a task of the
arm was runnable but not running = waiting for a CPU, own threads included), cpu.stat usage/user/system, and the main thread
(tid == pid) /proc schedstat of every python3 process in the cgroup (run ns, runqueue-wait ns). The summary keeps the last
cgroup values = whole-arm totals (counters start at 0 with the scope) and, for the python process with the largest main-
thread run time (the CLI; its last values before it exited), the main thread's runqueue-wait share of its run time.
"""
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

unit, out_jsonl, summary_json = sys.argv[1], sys.argv[2], sys.argv[3]
period = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
CG = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}"
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
    pr, cs, procs = rd(f"{CG}/cpu.pressure"), rd(f"{CG}/cpu.stat"), rd(f"{CG}/cgroup.procs")
    if pr is None:
        return None
    some = {k: float(v) for k, v in (kv.split("=") for kv in pr.splitlines()[0].split()[1:])}
    stat = dict(l.split() for l in (cs or "").splitlines())
    main = {}
    for pid in (procs or "").split():
        comm = (rd(f"/proc/{pid}/comm") or "").strip()
        if comm.startswith("python"):
            ss = rd(f"/proc/{pid}/schedstat")
            if ss:
                run, wait, n = map(int, ss.split())
                main[pid] = dict(run_s=run / 1e9, wait_s=wait / 1e9, slices=n)
    return dict(utc=datetime.now(timezone.utc).isoformat(), pressure_some_total_s=some["total"] / 1e6,
                usage_s=int(stat.get("usage_usec", 0)) / 1e6, user_s=int(stat.get("user_usec", 0)) / 1e6,
                system_s=int(stat.get("system_usec", 0)) / 1e6, python_main=main)


last, n, seen, mains = None, 0, False, {}
t_start = time.time()
with open(out_jsonl, "a") as out:
    while not stop:
        s = sample()
        if s is None and seen:
            break                      # scope gone: arm finished
        if s is not None:
            seen, last, n = True, s, n + 1
            mains.update(s["python_main"])   # latest values per python pid (the CLI exits before the scope does)
            out.write(json.dumps(s) + "\n")
            out.flush()
        time.sleep(period)
cli = max(mains.values(), key=lambda v: v["run_s"], default=None)
summary = dict(unit=unit, samples=n, utc_start=datetime.fromtimestamp(t_start, timezone.utc).isoformat(),
               utc_end=datetime.now(timezone.utc).isoformat(), last=last, cli_main_thread=cli,
               cli_main_wait_share=round(cli["wait_s"] / cli["run_s"], 4) if cli and cli["run_s"] else None)
with open(summary_json, "w") as h:
    json.dump(summary, h, indent=1)
