"""Use the established native-facade release probe with lane-owned scratch."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
scratch = "<USER_HOME>/wrf_gpu2_lanes/t-camrad/identity"
Path(scratch).mkdir(parents=True, exist_ok=True)
probe = source / "proofs/s1_port/release_identity.py"
body = probe.read_text()
assert body.count("<USER_HOME>/wrf_gpu2_lanes/s1-port") == 1
body = body.replace("<USER_HOME>/wrf_gpu2_lanes/s1-port", scratch)
exec(compile(body, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
