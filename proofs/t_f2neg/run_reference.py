"""Run a private observed WRF on exact retained inputs; no reference writes."""
import hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
assert os.environ['JAX_PLATFORMS']=='cpu'
ROOT=pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/t-f2neg')
name=sys.argv[1]
assert name in ('release_default','mp16')
arm=ROOT/name
arm.mkdir(exist_ok=False)
plan=json.loads(pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/v-run/PLAN.json').read_text())
config=next(c for c in plan['configs'] if c['config']==name)
reference=pathlib.Path(config['reference'])
for p in reference.iterdir():
    if p.is_symlink() and p.exists() and not p.name.startswith(('wrf','namelist')):
        (arm/p.name).symlink_to(p.resolve())
# Supplement canonical runtime tables only; no symlinks to mutable outputs.
for p in (ROOT/'build/run').iterdir():
    if not p.is_file() or p.name.startswith(('wrf','namelist','ideal','real','README')): continue
    if not (arm/p.name).exists(): (arm/p.name).symlink_to(p.resolve())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
for n in ('namelist.input','wrfinput_d01','wrfbdy_d01'):
    src=pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/v-run')/name/'input'/n
    shutil.copyfile(src,arm/n)
    inputs[n]=sha(arm/n)
    assert inputs[n]==config['matched_inputs'][n]['GPU_sha256']
exe=ROOT/'wrf_observed.exe'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='<USER_HOME>/miniconda3/envs/wrfbuild/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
receipt=dict(case=name,reference=str(reference),input_sha256=inputs,exe_sha256=sha(exe),source_sha256=sha(ROOT/'build/dyn_em/solve_em.f90'),observer_sha256=sha(ROOT/'build/dyn_em/f2_observer.f90'),cache='NONE',core_affinity=sorted(os.sched_getaffinity(0)),started_utc=datetime.now(timezone.utc).isoformat(),scope='24h original WRF with read-only post-RK observer; E26 byte qualification pending')
(arm/'START.json').write_text(json.dumps(receipt,indent=2)+'\n')
start=time.monotonic()
with (arm/'run.log').open('w') as log:
    p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=15','2400',str(exe)],cwd=arm,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    paused=False
    while p.poll() is None:
        quiet=pathlib.Path('/tmp/wrf_gpu2_quiet').exists()
        if quiet!=paused:
            os.killpg(p.pid,signal.SIGSTOP if quiet else signal.SIGCONT)
            paused=quiet
        time.sleep(.5)
receipt.update(rc=p.returncode,wall_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),frames=len(list(arm.glob('wrfout_d01*'))))
(arm/'RUN.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
assert p.returncode==0
