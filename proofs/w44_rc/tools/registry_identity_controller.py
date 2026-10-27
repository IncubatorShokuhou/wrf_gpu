"""Serial identity probes on immutable sources; reuse optimized records only for exact compiler input bytes."""
from pathlib import Path
import datetime,importlib.util,json,os,re,subprocess as sp,sys,time
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc');MAIN=Path('<USER_HOME>/src/wrf_gpu2');TOOLS=D/'tools';AL=D/'source_alias';CFG=json.loads((D/'GATE_CONFIG.json').read_text());OUT=D/'registry_identity';OUT.mkdir(exist_ok=True)
assert os.environ.get('JAX_PLATFORMS')=='cpu'
spec=importlib.util.spec_from_file_location('runner',D/'final_snapshot/proofs/w42_bundle042/tools/rc_suite.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class Pause:
 def exists(self):return any(Path(p).exists() for p in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window'))
 def read_text(self):return 'QUIET/GPU_WINDOW'
r.QUIET=Pause()
def wait():
 while r.QUIET.exists():time.sleep(5)
def now():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def run(job):
 label=job['label'];out=OUT/label
 if (out/'receipt.json').exists():return json.loads((out/'receipt.json').read_text())
 wait()
 if AL.is_symlink():AL.unlink()
 assert not AL.exists();AL.symlink_to(job['source'],target_is_directory=True)
 env=dict(os.environ)
 for k in list(env):
  if k.startswith('GPUWRF_') or re.search('TOKEN|PASSW|SECRET|API_?KEY|CREDENTIAL|EARTHDATA',k,re.I):env.pop(k)
 env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE='false',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS='1',GPUWRF_XLA_AUTOTUNE_CACHE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(AL/'scripts'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('XLA_FLAGS',None)
 script=TOOLS/('registry_key_probe.py' if job['kind']=='key' else 'native_identity_gate.py')
 cmd=[sys.executable,str(script),'--source',str(AL),'--case',CFG['cases'][job['case']],'--domain',job['domain'],'--target',CFG['target'],'--out',str(out),'--schema2-tool',str(TOOLS/'off_identity_probe.py')]
 if job['kind']=='lower':cmd.append('--lower-only')
 cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi; exec "$@"','registry-identity',*cmd]
 result=r.run_paused(cmd,TOOLS,env,OUT/(label+'.log'),1800,lambda x:print(now(),x,flush=True))
 (OUT/(label+'.wrapper.json')).write_text(json.dumps(result,indent=2)+'\n')
 assert result['rc']==0,(label,result)
 rec=json.loads((out/'receipt.json').read_text());assert rec['phase'] in ('key_done','lower_done','done'),rec
 return rec
if len(sys.argv)>1 and sys.argv[1]=='child':
 run(json.loads(Path(sys.argv[2]).read_text()));raise SystemExit(0)
def launch(job):
 wait();f=OUT/(job['label']+'.job.json');f.write_text(json.dumps(job,indent=2)+'\n')
 cmd=[str(MAIN/'scripts/with_ram_slot.sh'),'--label','w44-rc-'+job['label'],'--max','12G','--timeout','600','--','taskset','-c','8,9,12','nice','-n','19','timeout','-k','30','86400',sys.executable,__file__,'child',str(f)]
 with (OUT/(job['label']+'.scope.log')).open('w') as log:child=sp.run(cmd,stdout=log,stderr=sp.STDOUT)
 assert child.returncode==0,(job,child.returncode)
 rec=json.loads((OUT/job['label']/'receipt.json').read_text());print(now(),job['label'],rec['phase'],round(rec['wall_s'],1),flush=True)
 row=[now(),'w44-rc','Codex',job['label'],'registry-only delta preserves actual canonical program/key identity',json.dumps(job,sort_keys=True),'actualproductioncall metadata/native-first CUDA IR; optimized reuse only exact input bytes','4b saved compile outputs',rec['phase'],'RECORDED CPU-only; noGPU',f'artifact={OUT/job["label"]};cache=private tmp/triton/cuda/scan caches cleaned by helper;work=CPU8,9,12NI19<=12GnoSwap,fullcase{job["case"]}/{job["domain"]}']
 with (MAIN/'.agent/kernel/EXPERIMENTS.tsv').open('a') as log:log.write('\t'.join(str(x).replace('\n',' ') for x in row)+'\n')
 return rec
before=launch(dict(label='RC22-key-before-prod',kind='key',case='prod',domain='d01',source=str(D/'final_snapshot')))
after=launch(dict(label='RC22-key-after-prod',kind='key',case='prod',domain='d01',source=str(D/'registry_snapshot')))
assert before['key_components']==after['key_components'], 'registry delta changed production-call program key components'
rows=[]
for case,domain in [('swiss','d01'),('prod','d01'),('prod','d02'),('wn3','d01'),('wn3','fused_d02')]:
 label=f'RC22-lower-{case}-{domain}';new=launch(dict(label=label,kind='lower',case=case,domain=domain,source=str(D/'registry_snapshot')))
 old=json.loads((D/'native'/f'RC02-candidate-{case}-{domain}'/'receipt.json').read_text())
 exact=new['raw_full_constant_proto_sha256']==old['raw_full_constant_proto_sha256']
 row=dict(case=case,domain=domain,compiler_input_proto_EXACT=exact,before=str(D/'native'/f'RC02-candidate-{case}-{domain}'),after=str(OUT/label),old_optimized_sha256=old['optimized_hlo_sha256'])
 if exact:
  row['optimized_scope']='CARRY existing measured4b optimizedHLO: exact fullconstant compiler input bytes and unchanged target/compiler/options; no new optimized compile claimed'
 else:
  comp=launch(dict(label=f'RC22-compile-{case}-{domain}',kind='compile',case=case,domain=domain,source=str(D/'registry_snapshot')))
  row['optimized_scope']='FRESH CPU CompileOnly autotune0; actualGPU remains required';row['new_optimized_sha256']=comp['optimized_hlo_sha256'];row['raw_optimized_EXACT']=comp['optimized_hlo_sha256']==old['optimized_hlo_sha256'];row['new_optimized_artifact']=str(OUT/f'RC22-compile-{case}-{domain}')
 rows.append(row);(OUT/'PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
report=dict(registry_program_key_components_EXACT=True,before_key=str(OUT/'RC22-key-before-prod/receipt.json'),after_key=str(OUT/'RC22-key-after-prod/receipt.json'),GPU_exec_key_bytes='UNMEASURED; program key actual deviceless call held; exec_env implementation/target/flags unchanged by JSON',rows=rows,scientific_scope='regression/compiler identity only, not WRF fidelity or GPU execution')
(OUT/'IDENTITY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(now(),'REGISTRY_IDENTITY_DONE',flush=True)
