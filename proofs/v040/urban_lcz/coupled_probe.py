"""Use the existing CPU-safe real CLI probe, with all temporary IO in this lane.

Set O1_CASE for a prepared case, O1_EXEC_STEPS for a coupled smoke, and
O1_INTERPRET_EXEC=1 to interpret the release Pallas kernels on CPU. Arguments
and SHA convention are exactly proofs/o1_nlbind/probe.py's. No GPU claims.
"""
from pathlib import Path

root=Path(__file__).resolve().parents[3]
source=root/'proofs/o1_nlbind/probe.py'
text=source.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
    'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-lcz")')
# Retain the original __file__ so its cpu_safe helper import resolves identically.
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
