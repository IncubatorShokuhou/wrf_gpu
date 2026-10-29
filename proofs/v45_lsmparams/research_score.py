"""Execute the retained pristine-WRF scorer with process-local research admission."""
import argparse
import os
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
a,remaining=p.parse_known_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/w44_lsmparams')]
from gpuwrf.surface_params import controls as sc

assert sc.UNAVAILABLE_SURFACE_CONTROLS==frozenset(sc.LSM_CONTROLS)
saved=sc.UNAVAILABLE_SURFACE_CONTROLS
sc.UNAVAILABLE_SURFACE_CONTROLS=frozenset()
script=a.source/'proofs/w44_lsmparams/score.py'
sys.argv=[str(script),'--repo',str(a.source),*remaining]
try:
    runpy.run_path(str(script),run_name='__main__')
finally:
    sc.UNAVAILABLE_SURFACE_CONTROLS=saved
