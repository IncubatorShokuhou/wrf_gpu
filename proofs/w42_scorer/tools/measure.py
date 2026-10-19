"""Run CMD and record its process-tree memory (w42-scorer measurement helper, CPU only).
Samples every 0.5 s: sum of VmRSS over the whole descendant tree (true concurrent peak) and per-process VmHWM; at the end
the CPU time of all reaped descendants (user+sys, unaffected by SIGSTOP pauses). Writes OUT_JSON, exit code = CMD's.
usage: python measure.py OUT_JSON CMD ARGS..."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def children_map():
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    return kids


def status(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM", "Name")):
                    k, v = line.split(":", 1)
                    out[k] = v.strip()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            out["cmd"] = f.read().replace(b"\0", b" ").decode(errors="replace")[-160:]
    except OSError:
        return None
    return out


out_path = Path(sys.argv[1])
t0 = time.time()
proc = subprocess.Popen(sys.argv[2:])
peak_sum, per_pid = 0, {}
while proc.poll() is None:
    kids, tree, stack = children_map(), [], [proc.pid]
    while stack:
        p = stack.pop()
        tree.append(p)
        stack.extend(kids.get(p, []))
    total = 0
    for p in tree:
        st = status(p)
        if not st or "VmRSS" not in st:
            continue
        rss = int(st["VmRSS"].split()[0])
        total += rss
        hwm = int(st.get("VmHWM", "0 kB").split()[0])
        key = f"{p} {st['cmd']}"
        per_pid[key] = max(per_pid.get(key, 0), hwm)
    peak_sum = max(peak_sum, total)
    time.sleep(0.5)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
result = {"cmd": sys.argv[2:], "rc": proc.returncode, "wall_s": round(time.time() - t0, 1),
          "cpu_user_s": round(ru.ru_utime, 1), "cpu_sys_s": round(ru.ru_stime, 1),
          "max_single_process_rss_MiB": ru.ru_maxrss // 1024,
          "peak_tree_rss_sum_MiB": peak_sum // 1024,
          "per_process_hwm_MiB": {k: v // 1024 for k, v in sorted(per_pid.items(), key=lambda kv: -kv[1])[:12]}}
out_path.write_text(json.dumps(result, indent=1) + "\n")
print(f"[measure] rc={proc.returncode} peak_tree_rss_sum={peak_sum // 1024} MiB max_single={ru.ru_maxrss // 1024} MiB "
      f"cpu={ru.ru_utime + ru.ru_stime:.1f} s", file=sys.stderr)
sys.exit(proc.returncode)
