"""Run minimal exact-base failure classification after the full suite completes."""
from pathlib import Path
import collections,datetime,importlib.util,json,os,re,shutil,subprocess as sp,sys,time
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc');BASE=D/'baseline_snapshot';CUR=D/'final_snapshot';MAIN=Path('<USER_HOME>/src/wrf_gpu2')
assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert json.loads((D/'CONTROLLER.json').read_text())['phase']=='MEASUREMENTS_DONE_CLASSIFICATION_PENDING'
assert sp.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip().startswith('700a382f8b')
spec=importlib.util.spec_from_file_location('runner',CUR/'proofs/w42_bundle042/tools/rc_suite.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class Pause:
 def exists(self):return any(Path(p).exists() for p in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window'))
 def read_text(self):return 'QUIET/GPU_WINDOW'
r.QUIET=Pause()
def wait():
 while r.QUIET.exists():time.sleep(5)
def now():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def selectors(result):
 file=result['selector'].split('::')[0];prefix=file[:-3].replace('/','.')
 out=[]
 for node,(status,msg) in result['nodes'].items():
  if status not in ('FAIL','ERROR'):continue
  cls,name=node.split('::',1);assert cls==prefix or cls.startswith(prefix+'.')
  suffix=cls[len(prefix):].strip('.').replace('.','::')
  out.append(file+('::'+suffix if suffix else '')+'::'+name)
 return out
jobs=[]
for suite,fast in [('tierA','0'),('release_defaults','1')]:
 selected=[]
 for p in sorted((D/suite).glob('*/result.json')):
  res=json.loads(p.read_text())
  if not res.get('nodes'):continue # unavailable inventory rows handled separately, not PASS
  selected.extend(selectors(res))
 selected=sorted(set(selected))
 if selected:jobs.append(dict(label='base_'+suite+'_same_env',tree=str(BASE),selectors=selected,fast=fast,compilation_cache='false'))
cache_nodes=['tests/test_aot_key_cli.py::test_aot_key_leaves_a_sealed_bundle_byte_identical','tests/test_aot_key_cli.py::test_without_the_redirect_aot_key_writes_the_bundle']
for label,tree in [('base_cache_enabled',BASE),('candidate_cache_enabled',CUR)]:jobs.append(dict(label=label,tree=str(tree),selectors=cache_nodes,fast='0',compilation_cache='true'))
root=D/'classification';root.mkdir(exist_ok=True);(root/'JOBS.json').write_text(json.dumps(jobs,indent=2)+'\n')
for job in jobs:
 wait();out=root/job['label'];out.mkdir(exist_ok=True)
 if (out/'result.json').exists():continue
 tmp=D/'classification_tmp';tmp.mkdir(exist_ok=True);tree=Path(job['tree']);env=r.base_env(tree,tmp)
 for key in list(env):
  if re.search('TOKEN|PASSW|SECRET|API_?KEY|CREDENTIAL|EARTHDATA',key,re.I) or key.startswith('GPUWRF_'):env.pop(key)
 env.update(JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS=job['fast'],JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE=job['compilation_cache'])
 cmd=r.pytest_cmd(tree,Path('<USER_HOME>/wrf_gpu2_lanes/w44-rc/pytest'),out/'pytest.xml',job['selectors'],extra=('--tb=short',))
 cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi; exec "$@"','classify',*cmd]
 result=r.run_paused(['/usr/bin/time','-v','-o',str(out/'time.txt'),*cmd],tree,env,out/'pytest.log',900,lambda x:print(now(),x,flush=True))
 nodes=r.parse_xml(out/'pytest.xml');result.update(job=job,head=sp.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),src_tree=sp.check_output(['git','rev-parse','HEAD:src'],cwd=tree,text=True).strip(),nodes=nodes,counts=dict(collections.Counter(v[0] for v in nodes.values())))
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(now(),job['label'],result['rc'],result['counts'],flush=True)
 wait()
 for p in [D/'pytest_home/pytest',tmp]:
  assert p.resolve().is_relative_to(D)
  if p.exists():shutil.rmtree(p)
 (D/'pytest_home/pytest').mkdir(exist_ok=True)
print(now(),'CLASSIFICATION_MEASUREMENTS_COMPLETE',flush=True)
