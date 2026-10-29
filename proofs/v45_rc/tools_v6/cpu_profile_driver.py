"""Approved CPU profile phase: exact requested CLI plan, then real Swiss NVTX work."""
import os,runpy,sys,json
from pathlib import Path
assert os.environ.get('WRF_GPU2_SEQ_SMOKE')=='1' and os.environ.get('JAX_PLATFORMS')=='cpu'
driver=str(Path(__file__).with_name('cli_driver.py'));original=sys.argv[1:]
def run(argv):
 sys.argv=[driver,*argv]
 try:runpy.run_path(driver,run_name='__main__')
 except SystemExit as e:
  assert e.code in (None,0),('CPU CLI stage failed',e.code)
run([*original,'--dry-run'])
root=Path.cwd();swiss=os.environ['W45R_CPU_SWISS_INPUT']
run(['run','--input-dir',swiss,'--output-dir',str(root/'cpu_swiss/wrfout'),'--proof-dir',str(root/'cpu_swiss/proof'),
     '--scratch-dir',str(root/'cpu_swiss/scratch'),'--domains-from-namelist','--hours','3'])
(root/'CPU_PROFILE_SUBSTITUTION.json').write_text(json.dumps(dict(authority='manager09:54:33Z',requested_case='PROD exact CLI dry-run',executed_case='Swiss real CPU first root',GPU_profile='NOT_APPLICABLE'),indent=2)+'\n')
