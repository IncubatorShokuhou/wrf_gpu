"""Run the official CPU release identity probe with this critic's scratch directory."""
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1]).resolve()
official = root / "proofs/s1_port/release_identity.py"
source = official.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-port")'
new = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix-mynn")'
assert source.count(old) == 1
source = source.replace(old, new)
print("official_probe_sha256", hashlib.sha256(official.read_bytes()).hexdigest(), flush=True)
exec(compile(source, str(official), "exec"), {"__file__": str(official), "__name__": "__main__"})
