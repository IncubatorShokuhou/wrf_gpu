"""Continuous foreign-load watcher for the GPU-arm host cores DURING an arm (manager 2026-10-18T14:17:09Z).

usage: python3 core_watch.py CORES PROC_PCT LIMIT_S AGG_PCT OWN_ROOT_PID OUT_JSONL SUMMARY_JSON ABORT_FILE [PERIOD_S]
Every PERIOD_S (default 1 s) all threads are read (/proc/<pid>/task/<tid>/stat; ~65 ms CPU per scan on this host). A thread
counts on the gated CORES in an interval when its last CPU (field 39) at the start or the end of the interval is one of
them; its share = CPU ticks in the interval / interval. Threads of OWN_ROOT_PID's process tree (the session, its scopes,
the arm, its samplers) and of this watcher are excluded; kernel threads (ppid 2: writeback, IRQ, nvidia) are tallied
separately and disclosed, never an abort (they may serve the arm's own I/O); everything else is foreign.
  rule P (manager): one foreign PROCESS with > PROC_PCT % of one core on the gated cores in more than LIMIT_S seconds
                    (cumulative within the arm)                                              -> ABORT
  rule A (swarm):   the sum of all foreign on-core shares > AGG_PCT % of one core in more than LIMIT_S seconds
                    (catches sweeps of many short jobs, each under rule P)                   -> ABORT
On the first ABORT the reason is written to ABORT_FILE (the session stops and repeats the arm); the watcher keeps sampling
until SIGTERM so the summary covers the whole arm. Each interval appends one JSONL line (per-core busy % from /proc/stat,
foreign aggregate, offenders > 5 %); SUMMARY_JSON is rewritten every interval and at exit.
"""
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

cores = {int(c) for c in sys.argv[1].split(",")}
proc_pct, limit_s, agg_pct = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
own_root = int(sys.argv[5])
out_jsonl, summary_json, abort_file = sys.argv[6], sys.argv[7], sys.argv[8]
period = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
TICK = os.sysconf("SC_CLK_TCK")
me = os.getpid()
stop = False


def on_term(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)


def parse(raw):
    comm = raw[raw.index("(") + 1:raw.rindex(")")]
    rest = raw[raw.rindex(")") + 2:].split()
    return comm, int(rest[1]), int(rest[11]) + int(rest[12]), int(rest[36])


def snapshot():
    threads, parents, comms = {}, {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        p = int(pid)
        try:
            with open(f"/proc/{pid}/stat") as h:
                comm, ppid, _, _ = parse(h.read())
            parents[p], comms[p] = ppid, comm
            for tid in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{tid}/stat") as h:
                    _, _, ticks, cpu = parse(h.read())
                threads[(p, int(tid))] = (ticks, cpu)
        except (OSError, ValueError, IndexError):
            continue
    return threads, parents, comms


def cpu_busy():
    out = {}
    with open("/proc/stat") as h:
        for line in h:
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                c = int(f[0][3:])
                if c in cores:
                    v = list(map(int, f[1:9]))
                    out[c] = (sum(v) - v[3] - v[4], sum(v))  # busy (minus idle, iowait), total
    return out


def tree(parents, roots):
    kids = {}
    for pid, ppid in parents.items():
        kids.setdefault(ppid, []).append(pid)
    out, frontier = set(roots), list(roots)
    while frontier:
        for k in kids.get(frontier.pop(), []):
            if k not in out:
                out.add(k)
                frontier.append(k)
    return out


def args_of(pid, comm):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:160] or comm
    except OSError:
        return comm


flagged = {}      # pid -> {"s": seconds over proc_pct, "max_pct": .., "comm": .., "args": ..}
agg_s, agg_max, samples, aborted = 0.0, 0.0, 0, None
kthread_s, kthread_max = 0.0, 0.0  # kernel threads (ppid 2) on the cores: disclosed, never an abort (may serve our own IO/GPU)
busy_sum = {c: [0, 0] for c in cores}
t_start = time.time()


def write_summary():
    worst = sorted(flagged.items(), key=lambda kv: -kv[1]["s"])[:10]
    s = dict(utc_start=datetime.fromtimestamp(t_start, timezone.utc).isoformat(), utc_end=datetime.now(timezone.utc).isoformat(),
             cores=sorted(cores), period_s=period, samples=samples, rule_P=dict(proc_pct=proc_pct, limit_s=limit_s),
             rule_A=dict(agg_pct=agg_pct, limit_s=limit_s), aborted=aborted,
             max_proc_seconds=round(max((v["s"] for v in flagged.values()), default=0.0), 1), agg_seconds=round(agg_s, 1),
             agg_max_pct=round(agg_max, 1),
             kthread_seconds_over_proc_pct=round(kthread_s, 1), kthread_max_pct=round(kthread_max, 1),
             core_busy_pct={c: round(100.0 * b / t, 1) if t else None for c, (b, t) in busy_sum.items()},
             within_rule=aborted is None,
             clean=aborted is None and not any(v["s"] > 0 for v in flagged.values()) and agg_s == 0,
             offenders=[dict(pid=p, **{k: (round(x, 1) if isinstance(x, float) else x) for k, x in v.items()}) for p, v in worst])
    tmp = summary_json + ".tmp"
    with open(tmp, "w") as h:
        json.dump(s, h, indent=1)
    os.replace(tmp, summary_json)


prev, _, _ = snapshot()
prev_busy, prev_t = cpu_busy(), time.time()
with open(out_jsonl, "a") as log:
    while not stop:
        time.sleep(max(0.0, period - (time.time() - prev_t)))
        cur, parents, comms = snapshot()
        cur_busy, now = cpu_busy(), time.time()
        dt = now - prev_t
        mine = tree(parents, [me, own_root])
        per_proc, per_kthread = {}, {}
        for key, (ticks, cpu) in cur.items():
            if key[0] in mine or key not in prev:
                continue
            p_ticks, p_cpu = prev[key]
            if not ({cpu, p_cpu} & cores) or ticks <= p_ticks:
                continue
            bucket = per_kthread if key[0] == 2 or parents.get(key[0]) == 2 else per_proc
            bucket[key[0]] = bucket.get(key[0], 0.0) + 100.0 * (ticks - p_ticks) / (TICK * dt)
        kagg = sum(per_kthread.values())
        kthread_max = max(kthread_max, kagg)
        if kagg > proc_pct:
            kthread_s += dt
        agg = sum(per_proc.values())
        agg_max = max(agg_max, agg)
        if agg > agg_pct:
            agg_s += dt
        for pid, pct in per_proc.items():
            if pct > proc_pct:
                f = flagged.setdefault(pid, dict(s=0.0, max_pct=0.0, comm=comms.get(pid, "?"), args=args_of(pid, comms.get(pid, "?"))))
                f["s"] += dt
                f["max_pct"] = max(f["max_pct"], pct)
        core_pct = {}
        for c in cores:
            if c in cur_busy and c in prev_busy:
                b = cur_busy[c][0] - prev_busy[c][0]; t = cur_busy[c][1] - prev_busy[c][1]
                busy_sum[c][0] += b; busy_sum[c][1] += t
                core_pct[c] = round(100.0 * b / t, 1) if t else None
        samples += 1
        if aborted is None:
            worst = max(flagged.items(), key=lambda kv: kv[1]["s"], default=None)
            if worst and worst[1]["s"] > limit_s:
                aborted = dict(rule="P", utc=datetime.now(timezone.utc).isoformat(), pid=worst[0], comm=worst[1]["comm"],
                               seconds=round(worst[1]["s"], 1), max_pct=round(worst[1]["max_pct"], 1), args=worst[1]["args"])
            elif agg_s > limit_s:
                aborted = dict(rule="A", utc=datetime.now(timezone.utc).isoformat(), agg_seconds=round(agg_s, 1),
                               agg_max_pct=round(agg_max, 1))
            if aborted:
                with open(abort_file, "w") as h:
                    h.write(json.dumps(aborted) + "\n")
        log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), dt=round(dt, 2),
                                  core_busy=core_pct, foreign_agg=round(agg, 1), kthread_agg=round(kagg, 1),
                                  top=sorted(((p, round(v, 1), comms.get(p, "?")) for p, v in per_proc.items() if v > 5),
                                             key=lambda x: -x[1])[:5])) + "\n")
        log.flush()
        write_summary()
        prev, prev_busy, prev_t = cur, cur_busy, now
write_summary()
