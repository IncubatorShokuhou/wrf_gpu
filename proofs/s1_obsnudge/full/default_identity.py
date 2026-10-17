"""Run the unchanged o1 release-identity probe with lane-local artifact paths."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[3]
probe=root/'proofs/o1_nlbind/probe.py'
out=Path(sys.argv[1]).resolve();out.parent.mkdir(parents=True,exist_ok=True)
source=probe.read_text()
source=source.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',f'LANE = Path({str(out.parent)!r})')
sys.argv=[str(probe),str(root),str(out),'release']
exec(compile(source,str(probe),'exec'),dict(__name__='__main__',__file__=str(probe)))
