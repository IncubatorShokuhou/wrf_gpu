"""Diff two pytest junit XML runs (same test set): per-test outcome change OFF -> ON.

usage: python junit_diff.py OFF.xml ON.xml OUT_JSON
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def outcomes(path):
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = f"{case.get('classname')}::{case.get('name')}"
        status, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                status, msg = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag], (el.get("message") or "")[:300]
                break
        out[key] = (status, msg)
    return out


off, on = outcomes(sys.argv[1]), outcomes(sys.argv[2])
changed = {k: {"off": off.get(k, ("missing", ""))[0], "on": on.get(k, ("missing", ""))[0],
               "on_msg": on.get(k, ("", ""))[1], "off_msg": off.get(k, ("", ""))[1]}
           for k in sorted(set(off) | set(on)) if off.get(k, ("missing",))[0] != on.get(k, ("missing",))[0]}
summary = {"off": dict(Counter(v[0] for v in off.values())), "on": dict(Counter(v[0] for v in on.values())),
           "changed": len(changed)}
json.dump({"summary": summary, "changed": changed}, open(sys.argv[3], "w"), indent=1)
print(summary)
for k, v in changed.items():
    print(f"{v['off']:8s} -> {v['on']:8s} {k}  | {v['on_msg'][:160]}")
