"""Run the established real CLI Swiss identity probe with lane-local scratch."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
probe=root/'proofs/o1_nlbind/probe.py'
text=probe.read_text()
old='LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert text.count(old)==1
text=text.replace(old,'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-mpfix")')
exec(compile(text,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
