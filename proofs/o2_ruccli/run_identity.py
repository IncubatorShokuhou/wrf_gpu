"""Reuse the release identity harness, moving only its scratch directory here.

Args: <immutable source tree> <output.json> [release | physics edits...].
The canonical jaxpr/digest procedure is unchanged from proofs/o1_nlbind/probe.py.
"""
from pathlib import Path
import sys

script = Path(sys.argv[1]) / "proofs/o1_nlbind/probe.py"
source = script.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert source.count(old) == 1
source = source.replace(old, 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o2-ruccli")')
exec(compile(source, str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
