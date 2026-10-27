"""Join scanner sites with the release-program literal inventory (w44-constfold).

A scanner site (file, line..end_line, port f32 bits P, REAL4-fold bits R) is
  CONFIRMED  if a release-jaxpr literal/constvar with bits P is consumed by an equation whose innermost gpuwrf
             frame lies on the site's lines (the f64-folded value IS in the release program there),
  FIXED      if instead bits R appear there (a typed REAL path already selects WRF's value),
  EXECUTED   if the site's lines ran during the step trace but neither value is attributed there
             (e.g. the constant was combined further, or travels through a helper), -> manual review,
  NOT-RUN    if the site's lines never ran while tracing the release step (other scheme / legacy / init path).
Also lists the release literal VALUES that equal some scanner port value anywhere (value-level evidence).

usage: python match_release.py SCAN_JSON OUT_JSON PREFIX [PREFIX ...]   (PREFIX of release_coverage outputs)
"""
import json
import sys
from collections import defaultdict

import numpy as np


def bits32(x):
    return int(np.asarray(np.float32(x)).view(np.uint32))


scan = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
prefixes = sys.argv[3:]
by_line = defaultdict(set)      # (file, line) -> {bits}
value_lines = defaultdict(set)  # bits -> {file:line}
covered = set()
for pre in prefixes:
    for f, n in json.load(open(pre + ".coverage.json")):
        covered.add((f, n))
    for rec in json.load(open(pre + ".literals.json")):
        if rec["dtype"] != "float32":
            continue
        if rec["frame"]:
            f, n = rec["frame"].rsplit(":", 1)
            by_line[(f, int(n))].add(rec["bits"])
            value_lines[rec["bits"]].add(rec["frame"])
        else:
            value_lines[rec["bits"]].add("<constvar>")

rows = []
for s in scan["sites"]:
    p, r = bits32(s["port_f32"]), bits32(s["real4_fold"])
    span = [(s["file"], n) for n in range(s["line"], s["end_line"] + 1)]
    here = set().union(*(by_line.get(k, set()) for k in span))
    ran = any(k in covered for k in span)
    if p in here:
        status = "CONFIRMED"
    elif r in here:
        status = "FIXED"
    elif ran:
        status = "EXECUTED"
    else:
        status = "NOT-RUN"
    rows.append(dict(s, status=status, port_bits=p, real4_bits=r,
                     port_value_anywhere=sorted(value_lines.get(p, ()))[:8],
                     real4_value_anywhere=sorted(value_lines.get(r, ()))[:8]))
summary = defaultdict(int)
for row in rows:
    summary[row["status"]] += 1
json.dump({"prefixes": prefixes, "summary": summary, "rows": rows}, open(out_path, "w"), indent=1)
print(dict(summary))
for row in rows:
    if row["status"] in ("CONFIRMED", "EXECUTED", "FIXED"):
        print(row["status"][:4], row["kind"][:4], row["file"].replace("src/gpuwrf/", ""), row["line"], row["scope"][:24],
              "|", row["expr"][:60], "|", row["ulps"], "anywhere:", len(row["port_value_anywhere"]))
