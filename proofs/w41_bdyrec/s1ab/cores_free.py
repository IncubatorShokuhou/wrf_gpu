"""Are the GPU-arm host cores free of foreign load? (B1 rule, manager 2026-10-18T05:31:27Z)

usage: python3 cores_free.py CORES THRESHOLD_PCT LOG_JSONL [OWN_ROOT_PID] [SAMPLE_S]
Samples every thread three times (/proc/<pid>/task/<tid>/stat at 0, SAMPLE_S/2, SAMPLE_S; default 3 s): a thread is an
offender when its CPU share over the interval exceeds THRESHOLD_PCT of one core and it was on one of CORES at any sample
(field 39, the last CPU). Threads of OWN_ROOT_PID's process tree (our session) and of this checker are excluded. Appends one JSON
line {utc, cores, threshold, free, offenders[...]} to LOG_JSONL; exit 0 = free, 1 = foreign load.
"""
import json, os, sys, time
from datetime import datetime, timezone

cores = {int(c) for c in sys.argv[1].split(",")}
threshold = float(sys.argv[2])
log = sys.argv[3]
own_root = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0
sample = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
tick = os.sysconf("SC_CLK_TCK")


def stat(path):
    with open(path) as handle:
        raw = handle.read()
    comm = raw[raw.index("(") + 1:raw.rindex(")")]
    rest = raw[raw.rindex(")") + 2:].split()
    return comm, int(rest[1]), int(rest[11]) + int(rest[12]), int(rest[36])  # comm, ppid, utime+stime, processor


def snapshot():
    threads, parents = {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            parents[int(pid)] = stat(f"/proc/{pid}/stat")[1]
            for tid in os.listdir(f"/proc/{pid}/task"):
                comm, _, ticks, cpu = stat(f"/proc/{pid}/task/{tid}/stat")
                threads[(int(pid), int(tid))] = (comm, ticks, cpu)
        except (OSError, ValueError, IndexError):
            continue
    return threads, parents


def tree(parents, root):
    out, frontier = {root}, [root]
    while frontier:
        node = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == node and pid not in out:
                out.add(pid)
                frontier.append(pid)
    return out


first, _ = snapshot()
time.sleep(sample / 2)
middle, _ = snapshot()
time.sleep(sample / 2)
second, parents = snapshot()
mine = tree(parents, os.getpid()) | (tree(parents, own_root) if own_root else set())
offenders = []
for key, (comm, ticks, cpu) in second.items():
    if key not in first or key[0] in mine:
        continue
    pct = 100.0 * (ticks - first[key][1]) / (tick * sample)
    seen = {cpu, first[key][2]} | ({middle[key][2]} if key in middle else set())
    if pct > threshold and seen & cores:
        try:
            args = open(f"/proc/{key[0]}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:160]
        except OSError:
            args = comm
        offenders.append({"pid": key[0], "tid": key[1], "cpu_now": cpu, "cpus_seen": sorted(seen), "pct": round(pct, 1), "comm": comm, "args": args})
record = {"utc": datetime.now(timezone.utc).isoformat(), "cores": sorted(cores), "threshold_pct": threshold,
          "sample_s": sample, "free": not offenders, "offenders": sorted(offenders, key=lambda o: -o["pct"])}
with open(log, "a") as handle:
    handle.write(json.dumps(record) + "\n")
print(json.dumps({"free": record["free"], "offenders": [(o["cpu_now"], o["pct"], o["comm"]) for o in record["offenders"]]}))
sys.exit(0 if record["free"] else 1)
