"""Run the established native-facade identity probe with lane-owned temp files."""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
path=root/'proofs/s1_port/release_identity.py'
source=path.read_text().replace('<USER_HOME>/wrf_gpu2_lanes/s1-port','<USER_HOME>/wrf_gpu2_lanes/t-fdda')
exec(compile(source,str(path),'exec'),{'__file__':str(path),'__name__':'__main__'})
