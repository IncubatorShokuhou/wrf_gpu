"""Reuse the accepted Swiss CLI identity probe with lane-owned temporary IO.

Only its scratch destination changes. Model loading, release defaults, CPU
Thomas shim, Pallas tracing and location-stripped jaxpr hashing are unchanged.
"""
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
probe = repo / "proofs/o1_nlbind/probe.py"
text = probe.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert text.count(old) == 1
text = text.replace(old, f"LANE = Path({str(out.parent)!r})")
sys.argv = [str(probe), str(repo), str(out), "release"]
exec(compile(text, str(probe), "exec"), {"__name__": "__main__", "__file__": str(probe)})
