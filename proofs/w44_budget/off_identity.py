"""Official release identity probe with only its temporary lane path relocated.

No hashing/canonicalization/physics code is changed. Own artifacts stay in this
lane. The frozen Swiss/PROD identities remain assertions, not inferred claims.
"""
import hashlib
import json
from pathlib import Path
import sys

root, out, case = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
official = root / "proofs/s1_port/release_identity.py"
source = official.read_text()
old = "'LANE = Path(\"<USER_HOME>/wrf_gpu2_lanes/s1-port\")'"
new = "'LANE = Path(\"<USER_HOME>/wrf_gpu2_lanes/w44-budget\")'"
assert source.count(old) == 1
source = source.replace(old, new)
sys.argv = [str(official), str(root), str(out), case]
exec(compile(source, str(official), "exec"), {"__file__": str(official), "__name__": "__main__"})
record = json.loads(out.read_text())
expected = "321aa651" if "20260725" in case else "b977712f"
assert record["arms"]["release"]["sha"].startswith(expected), record["arms"]["release"]
record["official_probe_sha256"] = hashlib.sha256(official.read_bytes()).hexdigest()
record["relocated_temp_lane_only"] = True
out.write_text(json.dumps(record, indent=2) + "\n")
