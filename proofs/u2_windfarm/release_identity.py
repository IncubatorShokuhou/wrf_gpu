"""Official portability-aware release probe, with lane-owned temporary files."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
probe=ROOT/'proofs/s1_port/release_identity.py'
source=probe.read_text()
old='<USER_HOME>/wrf_gpu2_lanes/s1-port'
assert source.count(old)==1,'official probe changed; audit before use'
source=source.replace(old,'<USER_HOME>/wrf_gpu2_lanes/u2-windfarm')
out,*case=sys.argv[1:]
sys.argv=[str(probe),str(ROOT),out,*case]
exec(compile(source,str(probe),'exec'),{'__name__':'__main__','__file__':str(probe)})
