"""Is the owner's e2-generator sweep quiet? (ATTRIB2 re-arm, 2026-10-18; the sweep aborted N1 tries 1-3.)

usage: python3 e2_quiet.py [SAMPLE_S] [MAX_PCT]     (defaults 5 s, 20 %)
Sums the CPU share over SAMPLE_S of every process whose cmdline contains 'e2-generator' or whose comm is ngspice
(this process excluded). Exit 0 = quiet (sum <= MAX_PCT), 1 = busy. Prints one JSON line. Reads /proc only.
"""
import json
import os
import sys
import time

sample = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
max_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
tick, me = os.sysconf("SC_CLK_TCK"), os.getpid()


def snap():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            raw = open(f"/proc/{pid}/stat").read()
            comm = raw[raw.index("(") + 1:raw.rindex(")")]
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            if comm == "ngspice" or "e2-generator" in cmd:
                rest = raw[raw.rindex(")") + 2:].split()
                out[int(pid)] = (comm, int(rest[11]) + int(rest[12]))
        except (OSError, ValueError, IndexError):
            continue
    return out


a = snap()
time.sleep(sample)
b = snap()
per = {p: 100.0 * (t - a[p][1]) / (tick * sample) for p, (c, t) in b.items() if p in a}
new = [p for p in b if p not in a]
total = sum(per.values()) + 100.0 * sum(1 for p in new if b[p][0] == "ngspice")  # a fresh ngspice job counts as busy
print(json.dumps({"quiet": total <= max_pct, "e2_pct": round(total, 1), "procs": len(b), "new": len(new),
                  "top": sorted(((p, b[p][0], round(v, 1)) for p, v in per.items() if v > 5), key=lambda x: -x[2])[:4]}))
sys.exit(0 if total <= max_pct else 1)
