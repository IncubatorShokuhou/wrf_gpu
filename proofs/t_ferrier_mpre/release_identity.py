"""Official s1-port identity algorithm; only its temporary directory changes."""
import hashlib
import json
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:3])
official = root / "proofs/s1_port/release_identity.py"
source = official.read_text()
old = "<USER_HOME>/wrf_gpu2_lanes/s1-port"
assert source.count(old) == 1
adapted = source.replace(old, "<USER_HOME>/wrf_gpu2_lanes/t-ferrier-mpre")
exec(compile(adapted, str(official), "exec"), {"__file__": str(official), "__name__": "__main__"})
record = json.loads(output.read_text())
record["lane_scratch_adaptation"] = dict(official_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                         change="temporary directory only")
output.write_text(json.dumps(record, indent=2) + "\n")
