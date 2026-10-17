"""Run the canonical release probe with temporary files confined to this lane."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
probe = ROOT/'proofs/o1_nlbind/probe.py'
text = probe.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
new = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/trace-case")'
assert text.count(old)==1
text = text.replace(old,new)
Path('<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/trace-case').mkdir(exist_ok=True)
sys.argv = [str(probe),str(ROOT),sys.argv[1],'release']
exec(compile(text,str(probe),'exec'),{'__name__':'__main__','__file__':str(probe)})
