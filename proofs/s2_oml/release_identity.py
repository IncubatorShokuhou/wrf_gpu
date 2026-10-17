"""Run the official s1_port release probe with temporary IO confined to this lane."""
from pathlib import Path
import json
import runpy
import sys
import tempfile

root=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2]).resolve()
lane=output.parent
original=tempfile.TemporaryDirectory


def own_temporary_directory(*args,**kwargs):
    if Path(kwargs.get('dir') or '/nonexistent')==Path('<USER_HOME>/wrf_gpu2_lanes/s1-port'):
        kwargs['dir']=lane
    return original(*args,**kwargs)


tempfile.TemporaryDirectory=own_temporary_directory
script=root/'proofs/s1_port/release_identity.py'
sys.argv=[str(script),str(root),str(output),*sys.argv[3:]]
try:
    runpy.run_path(str(script),run_name='__main__')
finally:
    tempfile.TemporaryDirectory=original
result=json.loads(output.read_text())
result['temporary_directory_root']=str(lane)
output.write_text(json.dumps(result,indent=2)+'\n')
