"""Run the official identity probe with its temporary views in this lane.

Usage is identical to proofs/s1_port/release_identity.py. Only the temporary
directory location changes; the official native-facade/model trace is intact.
"""
from pathlib import Path
import runpy
import sys
import tempfile

original = tempfile.TemporaryDirectory


def lane_temporary_directory(*args, **kwargs):
    kwargs["dir"] = "<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc"
    return original(*args, **kwargs)


tempfile.TemporaryDirectory = lane_temporary_directory
try:
    runpy.run_path(str(Path(sys.argv[1]) / "proofs/s1_port/release_identity.py"), run_name="__main__")
finally:
    tempfile.TemporaryDirectory = original
