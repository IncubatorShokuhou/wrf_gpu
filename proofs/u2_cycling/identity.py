"""Run the mandatory native-facade identity probe using this lane's artifacts."""
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
probe = root / "proofs/s1_port/release_identity.py"
source = probe.read_text().replace("<USER_HOME>/wrf_gpu2_lanes/s1-port", "<USER_HOME>/wrf_gpu2_lanes/u2-cycling")
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
