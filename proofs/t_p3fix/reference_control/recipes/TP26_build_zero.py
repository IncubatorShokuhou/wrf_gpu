from pathlib import Path
import hashlib,json,os,re,signal,subprocess,time
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP26_zero';root.mkdir(exist_ok=False)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
src=Path('<USER_HOME>/src/wrf_pristine/WRF');dest=root/'WRF'
start=time.monotonic()
cmd=['rsync','-a','--exclude=.git','--exclude=*.o','--exclude=*.a','--exclude=*.exe','--exclude=wrfout*','--exclude=wrfrst*','--exclude=wrfinput*','--exclude=wrfbdy*','--exclude=oracle_run*',str(src)+'/',str(dest)+'/']
subprocess.run(cmd,check=True,timeout=300)
original=(src/'configure.wrf').read_text();flags='-finit-real=zero -finit-integer=0 -finit-logical=false -g1'
config,n=re.subn(r'(?m)^(FCBASEOPTS\s*=.*)$',lambda m:m[0]+' '+flags,original);assert n==1
(dest/'configure.wrf').write_text(config)
manifest={}
for p in dest.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in ['.F','.F90','.c','.h']:
  rel=p.relative_to(dest);original_p=src/rel
  h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==hashlib.sha256(original_p.read_bytes()).hexdigest(),str(rel)
  manifest[str(rel)]=h
(root/'source_sha256.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
prefix=Path('<USER_HOME>/miniconda3/envs/wrfbuild')
env={**os.environ,'PATH':str(prefix/'bin')+':'+os.environ['PATH'],'NETCDF':str(prefix),'LD_LIBRARY_PATH':str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','WRF_SRC_ROOT_DIR':str(dest),'J':'-j 2'}
receipt={'scope':'full private originalWRF source replica; compiler-defined ZERO init; FP trapsOFF; no sourcebody edits','source_git':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),'source_manifest_sha256':hashlib.sha256((root/'source_sha256.json').read_bytes()).hexdigest(),'original_config_sha256':hashlib.sha256(original.encode()).hexdigest(),'config_sha256':hashlib.sha256(config.encode()).hexdigest(),'init_flags':flags,'compiler':subprocess.check_output([str(prefix/'bin/gfortran'),'--version'],text=True).splitlines()[0],'stage_s':time.monotonic()-start,'GPU_execution':False,'cache':'NONE','status':'BUILDING'}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (root/'compile.log').open('w') as log:
 p=subprocess.Popen(['timeout','--kill-after=10','3600',str(prefix/'bin/csh'),'-f','./compile','em_real','-j','2'],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:rc=p.wait(timeout=3630)
 except BaseException:
  os.killpg(p.pid,signal.SIGKILL);p.wait();raise
exe=dest/'main/wrf.exe';receipt.update(rc=rc,wall_s=time.monotonic()-start,status='BUILT' if exe.is_file() else 'FAIL')
if exe.is_file():receipt['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
assert exe.is_file() and rc==0
