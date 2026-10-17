"""Accepted portability/native trace probe; change only the scratch destination."""
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
probe=root/'proofs/s1_port/release_identity.py'
source=probe.read_text()
old='LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-port")'
new='LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-lcz")'
assert old in source
source=source.replace(old,new)
exec(compile(source,str(probe),'exec'),{'__name__':'__main__','__file__':str(probe)})
