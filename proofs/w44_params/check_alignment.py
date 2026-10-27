# Copied unchanged from lane/o5-coast proofs/o5-coast/tools/check_alignment.py (tools_10db38419c, sha 77961823) for
# the w44-params launcher E225 guard (manager 2026-10-24 21:28Z).
"""E225 run-config alignment guard for o5-coast launchers (the CLI --dry-run does NOT check it, measured 2026-10-24).

usage: python check_alignment.py NAMELIST HOURS [HOURS ...]
Refuses (exit 2) unless, for every requested run length, hours*3600 is an exact multiple of every domain's time step
(root time_step + time_step_fract_num/den, children divided by the cumulative parent_time_step_ratio), and the
lateral-boundary interval_seconds is an exact multiple of the root time step.  Exact rational arithmetic."""
import re
import sys
from fractions import Fraction


def values(text, key):
    m = re.search(rf"(?mi)^\s*{key}\s*=\s*([^\n/]*)", text)
    return [v.strip() for v in m.group(1).split(",") if v.strip()] if m else []


text = open(sys.argv[1]).read()
hours = [Fraction(h) for h in sys.argv[2:]]
dt = Fraction(int(values(text, "time_step")[0]))
num = values(text, "time_step_fract_num"); den = values(text, "time_step_fract_den")
if num and den and int(den[0]) != 0:
    dt += Fraction(int(num[0]), int(den[0]))
max_dom = int((values(text, "max_dom") or ["1"])[0])
ratios = [int(r) for r in values(text, "parent_time_step_ratio")] or [1]
parents = [int(p) for p in values(text, "parent_id")] or [0]
dts = {1: dt}
for d in range(2, max_dom + 1):
    dts[d] = dts[parents[d - 1] if d - 1 < len(parents) else 1] / ratios[d - 1]
interval = Fraction(int((values(text, "interval_seconds") or ["0"])[0]))
bad = []
for h in hours:
    for d, step in dts.items():
        if (h * 3600) % step:
            bad.append(f"hours {h}: {h * 3600} s not a multiple of d0{d} dt {step} s")
if interval and interval % dt:
    bad.append(f"interval_seconds {interval} not a multiple of root dt {dt}")
print({"dt": {f"d0{d}": str(s) for d, s in dts.items()}, "hours": [str(h) for h in hours], "interval": str(interval),
       "ok": not bad, "bad": bad})
sys.exit(2 if bad else 0)
