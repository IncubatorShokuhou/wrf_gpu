"""Load the frozen oracle assets from an actual clean git archive (no worktree data)."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/archive_gate')
OUT.mkdir(exist_ok=True)
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
paths=['src','data/fixtures','scripts/v025/cpu_guard.py','tests/conftest.py','tests/test_v040_urban_lcz.py','pyproject.toml',
       'proofs/v040/urban_lcz','proofs/v040/urban_slucm/slucm_parity.py']
assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
archive=OUT/'source.tar'
with archive.open('wb') as stream:
    subprocess.run(['git','archive',source_commit,*paths],cwd=ROOT,stdout=stream,check=True,timeout=30)
tree=OUT/'tree';tree.mkdir(exist_ok=True)
with tarfile.open(archive) as stream: stream.extractall(tree,filter='data')
env=dict(os.environ,PYTHONPATH=str(tree/'src'))
assert env.get('JAX_PLATFORMS')=='cpu'
assert not (tree/'data/wrf_pristine').exists()
started=time.monotonic()
run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_v040_urban_lcz.py',
    '-k','fixture_source_and_hashes or pristine_regime_census_and_table_selection or heat_clock_and_ahe_against',
    '--tb=short','--basetemp','<USER_HOME>/wrf_gpu2_lanes/u2-lcz/pytest'],cwd=tree,env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
(OUT/'tests.log').write_text(run.stdout)
report={'commit':source_commit,'source_tree':str(ROOT),'archive_tree':str(tree),
        'external_pristine_present':False,'returncode':run.returncode,'wall_s':time.monotonic()-started}
(OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
assert run.returncode==0,run.stdout[-5000:]
