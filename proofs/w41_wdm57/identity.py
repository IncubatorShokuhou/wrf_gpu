"""Official default identity probe with temporary files confined to this lane."""
from pathlib import Path
import runpy
import sys
import tempfile

original = tempfile.TemporaryDirectory


def lane_temp(*args, **kwargs):
    kwargs["dir"] = "<USER_HOME>/wrf_gpu2_lanes/w41-wdm57"
    return original(*args, **kwargs)


tempfile.TemporaryDirectory = lane_temp
probe = Path(sys.argv[1]) / "proofs/s1_port/release_identity.py"
sys.argv = [str(probe), *sys.argv[1:]]
runpy.run_path(str(probe), run_name="__main__")
