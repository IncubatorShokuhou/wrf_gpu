"""Capture original output timestamps immediately after an arm (E162). No JAX."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
r = json.loads((d / "run/receipt.json").read_text())
files = {p.name: p.stat().st_mtime for p in (d / "run/wrfout").glob("wrfout_d0*") if p.is_file()}
assert r["rc"] == 0 and r["device"]["platform"] == "gpu"
assert len(files) == 6, len(files)
launch = float((d / "launch_unix").read_text())
end = float((d / "end_unix").read_text())
assert all(launch <= t <= end for t in files.values())
print(json.dumps({"files": files, "process_launch_unix": launch,
                  "harness_launch_unix": r["launch_wall_utc"],
                  "whole_to_last_file_s": max(files.values()) - launch,
                  "harness_to_last_file_s": max(files.values()) - r["launch_wall_utc"],
                  "rc": 0, "device": r["device"], "valid": True}, indent=2))
