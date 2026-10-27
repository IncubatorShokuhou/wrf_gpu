"""Publish once-only immutable manager entry wrappers. Does not launch jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tools", type=Path, required=True)
a = p.parse_args()
common = a.tools / "proofs/w44_perf/night_launch.sh"
assert common.is_file()
lane = Path("<USER_HOME>/wrf_gpu2_lanes/w44-perf")
rows = []
for short, phase, cap in (("p1", "prime32", 25), ("p2", "prime40", 25), ("p3", "compare", 30), ("r43", "radiation", 3)):
    target = lane / f"night_launch_{short}.sh"
    body = ("#!/usr/bin/env bash\nset -euo pipefail\n"
            + "exec bash " + shlex.quote(str(common)) + " " + phase + ' "$@"\n')
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o555)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    rows.append({"phase": phase, "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "cap_minutes": cap})
print(json.dumps({"tools": str(a.tools), "common_sha256": hashlib.sha256(common.read_bytes()).hexdigest(), "launchers": rows}, indent=2))
