"""Use the common native-identity probe with temporary files in this lane."""
import hashlib
import json
from pathlib import Path
import sys

root, output = (Path(s).resolve() for s in sys.argv[1:3])
probe = root / "proofs/s1_port/release_identity.py"
source = probe.read_text()
old = "<USER_HOME>/wrf_gpu2_lanes/s1-port"
assert source.count(old) == 1
source = source.replace(old, str(output.parent))
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
receipt = json.loads(output.read_text())
receipt["lane_probe"] = {"common_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                         "only_change": "temporary artifact root redirected to this lane"}
output.write_text(json.dumps(receipt, indent=2) + "\n")
assert receipt["arms"]["release"]["sha"].startswith("2bf12cb8"), receipt["arms"]
