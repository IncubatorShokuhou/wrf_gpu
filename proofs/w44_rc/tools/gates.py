"""Serial capped CPU gates; waits for GPU window/QUIET, resumes only complete receipts."""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,os,re,shutil,signal,subprocess as sp,sys,time
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc'); T=D/'snapshot'; AL=D/'source_alias'; MAIN=Path('<USER_HOME>/src/wrf_gpu2'); TOOLS=D/'tools'; CFG=json.loads((D/'GATE_CONFIG.json').read_text()); CORES=CFG['cores'];T=Path(CFG.get('snapshot_dir',str(T)))
assert os.environ.get('JAX_PLATFORMS')=='cpu'
for key in list(os.environ):
 if re.search(r'TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$',key,re.I):os.environ.pop(key)
P=argparse.ArgumentParser();P.add_argument('action',choices=['controller','suite-controller','suite','native']);P.add_argument('--suite');P.add_argument('--selectors',nargs='*');P.add_argument('--source');P.add_argument('--case');P.add_argument('--domain');P.add_argument('--label');P.add_argument('--lower-only',action='store_true');a=P.parse_args()
def now():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def write(path,data):path.write_text(json.dumps(data,indent=2,default=str)+'\n')
def wait_clear():
 said=False
 while any(Path(f).exists() for f in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window')):
  if not said:print(now(),'WAIT GPU_WINDOW/QUIET',flush=True);said=True
  time.sleep(15)
 for _ in range(3 if said else 0):
  time.sleep(5)
  if any(Path(f).exists() for f in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window')):return wait_clear()
 if shutil.disk_usage('<DATA_ROOT>').free < 370*1024**3:raise RuntimeError('DATA_FLOOR need370GiB before new gate')
def logexp(eid,hyp,change,metric,result,verdict,artifact):
 row=[now(),'w44-rc','Codex',eid,hyp,change,metric,'frozen base/source receipts',result,verdict,f'artifact={artifact};cache=NONE (GPUWRF_JAX_CACHE=0, private tmp/cache removed);work=CPU8,9,12 NI19 <=12G noSwap CPU-only']
 with (MAIN/'.agent/kernel/EXPERIMENTS.tsv').open('a') as f:f.write('\t'.join(str(x).replace('\n',' ') for x in row)+'\n')
def rgmain(kw,label):
 p=sp.run(['rg','-i',kw,*[str(MAIN/'.agent/kernel'/x) for x in ('FINDINGS.md','EXPERIMENTS.tsv','INBOX.md')]],stdout=sp.PIPE,stderr=sp.STDOUT)
 (D/(label+'_prior.txt')).write_bytes(p.stdout)
def loadrunner():
 spec=importlib.util.spec_from_file_location('rcsuite',T/'proofs/w42_bundle042/tools/rc_suite.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 class SharedPause:
  def exists(self):return any(Path(f).exists() for f in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window'))
  def read_text(self):return 'QUIET/GPU_WINDOW'
 r.QUIET=SharedPause()
 return r
def clean_own_tmp(path):
 while Path('/tmp/wrf_gpu2_quiet').exists():time.sleep(5)
 assert path.resolve().is_relative_to(D.resolve())
 if path.exists():shutil.rmtree(path)
def run_scope(args,label,seconds=14400):
 wait_clear()
 cmd=[str(MAIN/'scripts/with_ram_slot.sh'),'--label','w44-rc-'+label,'--max','12G','--timeout','7200','--','taskset','-c',CORES,'nice','-n','19','timeout','-k','30',str(seconds),sys.executable,str(__file__),*args]
 cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi; exec "$@"','rc-scope',*cmd]
 with (D/(label+'.log')).open('w') as log:
  p=sp.run(cmd,stdout=log,stderr=sp.STDOUT)
 print(now(),label,'rc',p.returncode,flush=True)
 return p.returncode
if a.action=='suite':
 r=loadrunner();root=D/a.suite;root.mkdir(exist_ok=True)
 for selector in a.selectors:
  wait_clear();key=selector.replace('/','__').replace(':','_');out=root/key;out.mkdir(exist_ok=True)
  if (out/'result.json').exists():continue
  tmp=D/'cpu_tmp';tmp.mkdir(exist_ok=True);bt=Path('<USER_HOME>/wrf_gpu2_lanes/w44-rc/pytest')
  env=r.base_env(T,tmp)
  for key in list(env):
   if key.startswith('GPUWRF_'):env.pop(key)
  env.update(GPUWRF_JAX_CACHE='0')
  env.update(GPUWRF_FAST_DEFAULTS='1' if a.suite=='release_defaults' else '0',JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE='false')
  retry=out/'retry_selectors.json'
  actual_selectors=json.loads(retry.read_text()) if retry.exists() else [selector]
  cmd=r.pytest_cmd(T,bt,out/'pytest.xml',actual_selectors,extra=('--tb=short',))
  cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi; exec "$@"','rc-pytest',*cmd]
  result=r.run_paused(['/usr/bin/time','-v','-o',str(out/'time.txt'),*cmd],T,env,out/'pytest.log',1500,lambda s:print(now(),s,flush=True),stall_s=900)
  nodes=r.parse_xml(out/'pytest.xml')
  infra=out/'result.infrastructure.json'
  if infra.exists():
   old=json.loads(infra.read_text());kept={k:v for k,v in old['nodes'].items() if v[0]!='ERROR'}
   assert not (set(kept)&set(nodes)), 'retry repeated completed nodes'
   nodes={**kept,**nodes};result['infrastructure_retry']=dict(raw=str(infra),completed_nodes_reused=len(kept),setup_blocked_only=actual_selectors)
  result.update(selector=selector,head=CFG['candidate'],counts=dict(collections.Counter(o for o,m in nodes.values())),nodes=nodes)
  write(out/'result.json',result)
  print(now(),selector,result['rc'],result['counts'],round(result['active_s'],1),flush=True)
  # No consumer needs disposable pytest/JAX caches beyond XML/log/receipt.
  clean_own_tmp(D/'pytest_home/pytest');(D/'pytest_home/pytest').mkdir(exist_ok=True);clean_own_tmp(tmp)
 sys.exit(0)
if a.action=='native':
 rgmain('E41|E222|E224|RC02|composed.*native',a.label)
 wait_clear();assert not AL.exists() or AL.is_symlink()
 if AL.is_symlink():AL.unlink()
 AL.symlink_to(a.source,target_is_directory=True)
 out=D/'native'/a.label;out.parent.mkdir(exist_ok=True)
 if (out/'receipt.json').exists():
  rec=json.loads((out/'receipt.json').read_text())
  if rec.get('phase') in ('done','lower_done'):sys.exit(0)
  raise RuntimeError('Incomplete prior native arm requires explicit diagnosis/fresh label')
 cmd=[sys.executable,str(TOOLS/'native_identity_gate.py'),'--source',str(AL),'--case',a.case,'--target',CFG['target'],'--out',str(out),'--domain',a.domain,'--schema2-tool',str(TOOLS/'off_identity_probe.py')]
 if a.lower_only:cmd+=['--lower-only']
 env=dict(os.environ);env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',JAX_ENABLE_X64='true',GPUWRF_JAX_CACHE='0',JAX_ENABLE_COMPILATION_CACHE='false',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(AL/'scripts'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for k in list(env):
  if k.startswith('GPUWRF_') and k!='GPUWRF_JAX_CACHE':env.pop(k)
 env['GPUWRF_FAST_DEFAULTS']='1'
 env['GPUWRF_XLA_AUTOTUNE_CACHE']='0'
 env.pop('XLA_FLAGS',None)
 with (D/'native'/(a.label+'.log')).open('w') as log:p=sp.run(['timeout','-k','30','1800',*cmd],env=env,stdout=log,stderr=sp.STDOUT)
 rec=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
 okay=p.returncode==0 and rec.get('phase')==('lower_done' if a.lower_only else 'done')
 if not a.lower_only:okay=okay and rec.get('native_gate')=='PASS' and rec.get('hard_stack_gate')=='PASS' and not rec.get('kinput_transpose_over500')
 logexp(a.label,'composed real product native-first schema2/E41',f"{a.case} {a.domain} lower_only={a.lower_only}",'full constants/native census/E41',f"rc={p.returncode}, phase={rec.get('phase')}, native={rec.get('native_gate')}, STACK={rec.get('max_stack_B')}",'PASS' if okay else 'FAIL/INCOMPLETE',out)
 write(out/'VERDICT.json',{'okay':okay,'rc':p.returncode,'receipt':rec})
 sys.exit(0 if okay else 1)
# Lightweight controller: no slot/job until both flags are gone.
write(D/'CONTROLLER.json',{'pid':os.getpid(),'phase':'WAITING_FOR_WINDOW_END','candidate':CFG['candidate'],'started':now()})
wait_clear();rgmain('composition|release.default|E222|E224|E41','RC02')
assert sp.check_output(['git','rev-parse','HEAD'],cwd=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc'),text=True).strip()==CFG['candidate'] or sp.check_output(['git','rev-parse',CFG['candidate']+':src'],cwd=MAIN,text=True).strip()==CFG['source_tree']
if not T.exists():
 cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"','rc-snapshot','git','-C',str(MAIN),'worktree','add','--detach',str(T),CFG['candidate']]
 sp.run(cmd,check=True)
 (T/'data/wrf_pristine').symlink_to('<USER_HOME>/src/wrf_pristine')
assert sp.check_output(['git','rev-parse','HEAD'],cwd=T,text=True).strip()==CFG['candidate']
for p in ('pytest','native'):(D/p).mkdir(exist_ok=True)
home=Path('<USER_HOME>/wrf_gpu2_lanes/w44-rc');home.mkdir(exist_ok=True)
assert home.is_symlink() and home.resolve()==D/'pytest_home'
assert not (home/'pytest').is_symlink()
(home/'pytest').mkdir(exist_ok=True)
# Compile the changed candidate only; baseline uses the identical real-program lowering builder.
if a.action=='controller':
 for case,domain in [('swiss','d01'),('prod','d01'),('prod','d02'),('wn3','d01'),('wn3','fused_d02')]:
  for variant,source,loweronly in [('candidate',str(T),False),('v043',CFG['baseline_bundle_source'],True)]:
   label=f'RC02-{variant}-{case}-{domain}'
   write(D/'CONTROLLER.json',{'pid':os.getpid(),'phase':label,'candidate':CFG['candidate'],'updated':now()})
   args=['native','--source',source,'--case',CFG['cases'][case],'--domain',domain,'--label',label]
   if loweronly:args+=['--lower-only']
   rc=run_scope(args,label,seconds=2100)
   if rc:
    write(D/'STOP.json',{'phase':label,'rc':rc,'updated':now()});sys.exit(1)
# Suites stay one fresh pytest process per file/selector; each file/selector releases its RAM slot (E220).
for suite,listname in [('tierA','tierA_files.txt'),('release_defaults','release_defaults_nodes.txt')]:
 selectors=(D/listname).read_text().splitlines()
 for k in range(0,len(selectors)):
  batch=selectors[k:k+1];label=f'RC03-{suite}-{k:03d}'
  if all((D/suite/sel.replace('/','__').replace(':','_')/'result.json').exists() for sel in batch):
   print(now(),label,'REUSE completed receipt',flush=True);continue
  rgmain('tier.A|known.red|RC03',label)
  write(D/'CONTROLLER.json',{'pid':os.getpid(),'phase':label,'candidate':CFG['candidate'],'updated':now()})
  rc=run_scope(['suite','--suite',suite,'--selectors',*batch],label,seconds=86400)
  result=[]
  for selector in batch:
   f=D/suite/selector.replace('/','__').replace(':','_')/'result.json'
   result.append(json.loads(f.read_text()) if f.exists() else {'selector':selector,'rc':'MISSING'})
  counts=collections.Counter()
  for x in result:counts.update(x.get('counts',{}))
  logexp(label,'composed candidate has zero new CPU failures',f'{suite} FAST{int(suite=="release_defaults")} {len(batch)} selectors','pytest JUnit',f'wrapper_rc={rc}, counts={dict(counts)}, incomplete={sum(not x.get("counts") for x in result)}','REQUIRES CLASSIFICATION' if rc or counts['FAIL'] or counts['ERROR'] else 'PASS',D/suite)
write(D/'CONTROLLER.json',{'pid':os.getpid(),'phase':'MEASUREMENTS_DONE_CLASSIFICATION_PENDING','candidate':CFG['candidate'],'updated':now()})
print(now(),'MEASUREMENTS_DONE',flush=True)
