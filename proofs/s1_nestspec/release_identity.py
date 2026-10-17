"""Run the frozen O1 release probe with temporary files inside this lane only."""
from pathlib import Path

probe = Path(__file__).resolve().parents[1] / "o1_nlbind/probe.py"
source = probe.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert source.count(old) == 1
source = source.replace(old, 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-nestspec")')
exec(compile(source, str(probe), "exec"), {"__name__": "__main__", "__file__": str(probe)})
