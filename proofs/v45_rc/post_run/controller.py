"""CPU post-window scoring and owned-run cleanup. Default is PLAN_ONLY; no GPU imports."""
import argparse,datetime,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');D=Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-rc');GIB=2**30

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def free(p):s=os.statvfs(p);return s.f_bavail*s.f_frsize

def discover(manifest,window):
 m=json.loads(manifest.read_text());entries={}
 paid=json.loads((Path(m['bundles']['prod']['dir'])/'meta/primes/A1_swiss.json').read_text())
 assert paid['pin']==m['pin'] and paid['mode']=='gpu' and paid['backend']['platform']=='gpu'
 paid_arm=Path(paid['arm']);assert sha(paid['proof'])==paid['proof_sha256'] and sha(paid['sha_list'])==paid['sha_list_sha256']
 for f in (D/'runs').glob('*/ENTRY.json'):
  j=json.loads(f.read_text())
  if j.get('mode')!='gpu' or Path(j.get('manifest','')).resolve()!=manifest.resolve():continue
  carried_swiss=j.get('entry')=='prime_swiss' and paid_arm.parent.parent==D/('session_'+m['pin'][:12]+'_gpu_v6')
  if j.get('window',[None])[0]!=window and not carried_swiss:continue
  name=j['entry'];assert name not in entries,'multiple actual entries: require explicit resolution'
  assert f.parent.parent==D/'runs' and not f.parent.is_symlink()
  assert carried_swiss or j.get('tools_sums_sha256')==sha(Path(m['tools_dir'])/'SHA256SUMS'),'entry toolset mismatch'
  lifecycle=json.loads((f.parent/'LIFECYCLE.json').read_text());assert lifecycle['cleanup']['all_children_reaped'] and not lifecycle['cleanup']['remaining_pids']
  entries[name]=dict(path=str(f.parent),entry=j,lifecycle=lifecycle)
 assert set(entries)=={'prime_swiss','prime_prod','prime_wn3','rw_warm','rw3'},'incomplete run catalog; retain raw'
 short=D/('session_'+m['pin'][:12]+'_gpu_v6');rw3=Path(entries['rw3']['path']);assert short.is_dir() and rw3.is_dir()
 return m,entries,short,rw3

def run_checked(cmd,env,log,remaining):
 assert remaining>0
 with Path(log).open('w')as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
 return p.returncode

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--window-start',required=True);p.add_argument('--deadline',default='2026-10-29T05:30:00Z');p.add_argument('--apply',action='store_true');a=p.parse_args()
 assert os.environ.get('JAX_PLATFORMS')=='cpu'
 stamp=re.sub('[^0-9TZ]','',a.window_start);O=R/'post_run'/stamp;O.mkdir(exist_ok=True);m,entries,S,A=discover(a.manifest,a.window_start);roots=[S,*[Path(v['path'])for v in entries.values()]]
 for root in roots:assert root.resolve().is_relative_to(D) and not root.is_symlink()
 catalog=dict(manifest=str(a.manifest),manifest_sha256=sha(a.manifest),window=a.window_start,entries=entries,delete_exact=[str(p)for p in roots],DATA_free_bytes=free('<DATA_ROOT>'),ROOT_free_bytes=free('/'),method='score/extract first; preserve ROOT; scan; literal own-run deletion only')
 dump(O/'DELETE_EXACT.json',catalog)
 if not a.apply:print('POST PLAN_ONLY',O/'DELETE_EXACT.json');return 0
 deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp();assert deadline-time.time()>=35*60,'not enough post budget; do not delete raw'
 # Explicit synchronous post-GPU handoff: all five model lifecycles are already reaped.
 # Keep the v9 window exclusion flag intact; no new GPU work is started here.
 locks=subprocess.run(['lslocks','-n','-o','PATH'],capture_output=True,text=True,check=True).stdout
 assert '/tmp/wrf_gpu2_gpu.lock' not in locks,'GPU holder still active; retain raw'
 ready=O/'POST_GPU_READY.json';dump(ready,dict(all_model_children_reaped=True,manifest_sha256=sha(a.manifest),window=a.window_start))
 assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET; retain raw'
 assert free('/')>=20*GIB,'ROOT floor; retain raw'
 env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',PYTHONDONTWRITEBYTECODE='1',W45R_FU_INNER='1')
 tools=Path(m['tools_dir']);pin=m['pin'][:12];F=R/'receipts'/('followup_short_'+pin);Q=R/'receipts'/('followup_rw3_'+pin)
 # CPU RW4 key work is prepared before the GPU window, on the same pinned source.
 ak=R/'receipts'/('aotkey_'+pin)
 for name in ['k0227_v_0502_h72','k0227_v_0614_h48','k0227_v_0227in48_h3','ki0227_v_ki0502_h3']:
  j=json.loads((ak/(name+'.json')).read_text());assert j['comparison'] and all(x['shared']for x in j['comparison'].values())
 adapter=Path(__file__).with_name('reference_phase.py')
 cmds=[('reference_prepare',[sys.executable,'-B',str(adapter),'prepare','--manifest',str(a.manifest),'--ready',str(ready)]),('short',['bash',str(tools/'followup/followup_short.sh'),str(a.manifest),str(S)])]
 results=[]
 for name,cmd in cmds:
  assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET; retain raw'
  rc=run_checked(cmd,env,O/(name+'.log'),max(1,deadline-time.time()-180));results.append(dict(phase=name,rc=rc,command=cmd));dump(O/'PHASES.json',results)
  if rc:raise RuntimeError(name+' transport failed; raw retained')
 Q.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'-I',str(tools/'followup/bytes_check.py'),str(Q/'BYTES_RW3.json'),'PREFIX N72 vs A3=prefix:'+str(A/'runs/N72/wrfout_sha.txt')+':'+str(S/'runs/A3_wn3_0227_3h/wrfout_sha.txt')]
 assert run_checked(cmd,env,O/'prefix.log',120)==0,'prefix evidence incomplete; raw retained'
 cmd=[sys.executable,'-B',str(adapter),'score','--manifest',str(a.manifest),'--ready',str(ready),'--gpu-dir',str(A/'runs/N72/wrfout'),'--out',str(Q/'d6_N72')]
 score_rc=run_checked(cmd,env,O/'rw3.log',max(1,deadline-time.time()-180));dump(O/'RW3_SCORE_RC.json',dict(rc=score_rc,meaning='raw D6/floor/integrity result; completeness checked separately'))
 RT=A/'rw3t'
 cmd=[sys.executable,'-I',str(tools/'followup/rw_verdict_v044.py'),str(Q/'RW_VERDICT_v045.json'),str(F/'SHORT_VERDICT.json'),str(RT/'RW3T_VERDICT.json'),str(RT/'RW3T_DETAIL24.json'),str(RT/'RW3T_DETAIL72.json'),str(Q/'d6_N72/FLOOR72.json'),str(Q/'d6_N72'),str(Q/'BYTES_RW3.json'),str(a.manifest.parent/'PREREG.json')]
 assert run_checked(cmd,env,O/'verdict.log',120)==0
 proof=check_scores_paths(F,Q);dump(O/'SCI_COMPLETE.json',proof)
 assert proof['required_score_complete'], 'required evidence missing; no raw deletion: '+str(proof['missing_evidence'])
 closed=[]
 for base,names in [(O,['DELETE_EXACT.json','SCI_COMPLETE.json','POST_GPU_READY.json','PHASES.json','RW3_SCORE_RC.json']), (F,['SHORT_VERDICT.json','BYTES.json']), (Q,['RW_VERDICT_v045.json','BYTES_RW3.json'])]:
  closed += [base/n for n in names if (base/n).is_file()]
 for base in [F/'d6_swiss',Q/'d6_N72']:
  names=['summary.json','resources.json','FLOOR72.json']+[d+suffix for d in ['d01','d02','d03'] for suffix in ['.json','.md','_early_hours.json']]
  closed += [base/n for n in names if (base/n).is_file()]
 dump(O/'CLOSED_CURRENT_RECEIPTS.json',dict(files={str(p):sha(p)for p in closed},basis='only named output artifacts after the actual scientific phase returned and complete coverage was checked'))
 assert deadline-time.time()>=180,'insufficient preservation/delete budget; raw retained'
 from data_ops import remove_literal,registry_deleted
 # All data readers and scientific wrappers completed before this literal operation.
 recovery=remove_literal(roots,O,proof,apply=True)
 registry_deleted(O,a.manifest)
 result=dict(resource_recovery=recovery,scientific_result=proof,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_post_complete=True)
 dump(O/'POST_RESULT.json',result)
 return 0 if recovery['meets_404_5GiB'] else 5

def check_scores_paths(F,Q):
 # Validate existing scientific artifacts in place; do not copy/delete any raw to pass validation.
 d6=json.loads((Q/'d6_N72/summary.json').read_text());floor=json.loads((Q/'d6_N72/FLOOR72.json').read_text());sw=json.loads((F/'d6_swiss/summary.json').read_text());short=json.loads((F/'SHORT_VERDICT.json').read_text());rw=json.loads((Q/'RW_VERDICT_v045.json').read_text())
 assert set(d6['domains'])=={'d01','d02','d03'} and all(x['pairing']['paired_file_count']==73 for x in d6['domains'].values())
 assert d6['all_common_variables_scored'] and d6['all_cpu_variables_required'] and floor['coverage_complete']
 assert sw['domains']['d01']['pairing']['paired_file_count']==25 and sw['all_common_variables_scored']
 from evidence_completeness import complete
 completeness=complete(short,rw)
 return dict(scientific_verdict=rw['overall'],**completeness,raw_D6=d6['pass'],binding72_A3=floor['binding72_A3'],integrity=d6['output_integrity_pass'],short=str(F),rw3=str(Q))
if __name__=='__main__':
 import signal
 def terminated(_sig,_frame):raise TimeoutError('managed post deadline/termination; incomplete evidence keeps raw')
 signal.signal(signal.SIGTERM,terminated)
 try:raise SystemExit(main())
 except Exception as exc:
  window=sys.argv[sys.argv.index('--window-start')+1] if '--window-start' in sys.argv else 'unknown'
  out=R/'post_run'/re.sub('[^0-9TZ]','',window)
  dump(out/'POST_RESULT.json',dict(full_post_complete=False,error=type(exc).__name__+': '+str(exc),scientific_PASS=False,raw_not_deleted_before_required_evidence=True))
  raise
