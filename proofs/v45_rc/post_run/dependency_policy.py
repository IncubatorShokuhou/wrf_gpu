"""Exact closed-artifact/current-plan and terminal receipt allowlist; no lane-prefix exemption."""
import hashlib,json
from pathlib import Path
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc')
RECEIPT_NAMES={'nested_pipeline_run.json','timer.json','RECEIPT.json','PROOF_SUMMARY.json','WRFOUT_FILES.tsv','wrfout_sha.txt','mtimes.json','time.txt','command.txt','REGISTRY_START.json','REGISTRY_END.json','device_measured.json','rc.txt','t0_unix','t_end_unix'}
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def expected(path,policy,current_outputs=()):
 p=Path(path);j=json.loads(Path(policy).read_text())
 if str(p)in j.get('closed_artifacts',{}):
  row=j['closed_artifacts'][str(p)]
  return sha(p)==row['sha256'],'exact closed artifact '+row['reason']
 for receipt in current_outputs:
  q=Path(receipt)
  if q.is_file():
   allow=json.loads(q.read_text()).get('files',{})
   if str(p)in allow and sha(p)==allow[str(p)]:return True,'completed current scoring/controller receipt'
 if p.suffix=='.json' and sha(p)in j.get('current_plan_sha256',[])+j.get('retired_reviewed_plan_sha256',[]):return True,'exact current authorized plan bytes'
 if p.name in RECEIPT_NAMES:
  try:
   a=json.loads((p.parent/'REGISTRY_START.json').read_text());b=json.loads((p.parent/'REGISTRY_END.json').read_text());t=json.loads((p.parent/'timer.json').read_text())
   ok=a['run_id']==b['run_id'] and b['status']in {'DONE_HELD','DONE_CLEAN','FAILED','ABORTED','SUPERSEDED'} and t['teardown']['all_reaped']
   return bool(ok),'terminal arm receipt only'
  except (OSError,ValueError,KeyError,TypeError):pass
 return False,'not a closed receipt or current authorized plan'
