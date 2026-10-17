"""Official native-facade identity probe with scratch redirected to this lane."""
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
probe=root/'proofs/s1_port/release_identity.py'
source=probe.read_text()
old='<USER_HOME>/wrf_gpu2_lanes/s1-port'
assert source.count(old)==1
source=source.replace(old,'<USER_HOME>/wrf_gpu2_lanes/s2-glacier')
exec(compile(source,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
