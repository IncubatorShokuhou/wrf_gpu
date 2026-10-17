from pathlib import Path
import hashlib,json,os,re,shlex,signal,subprocess,time
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP26b';root.mkdir(exist_ok=False)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
work=base/'TP26_zero/WRF';start=time.monotonic();prefix=Path('<USER_HOME>/miniconda3/envs/wrfbuild')
old=(work/'configure.wrf').read_text();(root/'config.before.wrf').write_text(old)
flags='-finit-real=zero -finit-integer=0 -finit-logical=false'
config,n=re.subn(r'(?m)^(FCBASEOPTS_NO_G\s*=.*)$',lambda m:m[0]+' '+flags,old);assert n==1
config,n=re.subn(r'(?m)^FCBASEOPTS\s*=.*$',r'FCBASEOPTS = $(FCBASEOPTS_NO_G) $(FCDEBUG) -g1',config);assert n==1
(work/'configure.wrf').write_text(config);(root/'config.after.wrf').write_text(config)
env={**os.environ,'PATH':str(prefix/'bin')+':'+os.environ['PATH'],'NETCDF':str(prefix),'LD_LIBRARY_PATH':str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','WRF_SRC_ROOT_DIR':str(work),'J':'-j 2'}
commands=[l for l in (base/'TP26_zero/compile.log').read_text().splitlines() if l.startswith('time gfortran -o module_mp_morr_two_moment_aero.o -c ')]
assert len(commands)==1
args=[a for a in shlex.split(commands[0])[2:] if a not in flags.split()]
args=[str(prefix/'bin/gfortran'),*args]
assert args.count('-c')==1 and args[-1]=='module_mp_morr_two_moment_aero.f90'
with (root/'inactive_morrison.log').open('w') as log:subprocess.run(args,cwd=work/'phys',env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
for prefix_name in ['nl_get','nl_set']:
 for i in range(8):(work/f'frame/{prefix_name}_{i}_routines.o').unlink()
record={'scope':'ZERO activeWRF build repair; bodies unchanged; GCC14 blanketSAVE/automatic-array init conflict in UNSELECTED mp40 Morrison; only that module omits finit flags; P3/default8/MYNN/NoahMP/RRTMG+all16generatedgetters/setters ZERO','parent_failed_build':str(base/'TP26_zero/receipt.json'),'initial_config_sha256':hashlib.sha256(old.encode()).hexdigest(),'config_sha256':hashlib.sha256(config.encode()).hexdigest(),'inactive_only_exception':{'module':'module_mp_morr_two_moment_aero','options':[40],'compiler_command':args},'init_flags':flags,'FP_traps':'OFF','cache':'NONE','GPU_execution':False,'status':'LINKING'}
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
with (root/'compile.log').open('w') as log:
 p=subprocess.Popen(['timeout','--kill-after=10','600',str(prefix/'bin/csh'),'-f','./compile','em_real','-j','2'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:rc=p.wait(timeout=630)
 except BaseException:
  os.killpg(p.pid,signal.SIGKILL);p.wait();raise
exe=work/'main/wrf.exe';record.update(rc=rc,wall_s=time.monotonic()-start,status='BUILT' if exe.is_file() else 'FAIL')
if exe.is_file():record['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
assert rc==0 and exe.is_file()
