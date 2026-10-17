"""Established identity probe with temporary case links placed in this lane."""
from pathlib import Path
import sys

root=Path(sys.argv[1]);out=Path(sys.argv[2])
scratch=out.parent/'identity_tmp';scratch.mkdir(exist_ok=True)
script=root/'proofs/s1_port/release_identity.py'
source=script.read_text()
assert source.count('<USER_HOME>/wrf_gpu2_lanes/s1-port')==1
source=source.replace('<USER_HOME>/wrf_gpu2_lanes/s1-port',str(scratch))
exec(compile(source,str(script),'exec'),{'__file__':str(script),'__name__':'__main__'})
