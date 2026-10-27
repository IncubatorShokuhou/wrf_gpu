"""Execute the exact sealed shell with CPU device/lock/profiler shims.

The real child receives the product CUDA env, recorded before the device shim
substitutes CPU. It executes the actual MYNN caller and twin on a frozen WN3
column. No official GPU lock/device or GPU numerical qualification is used.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launcher',type=Path,required=True);p.add_argument('--tools',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    a.out.mkdir(exist_ok=False);(a.out/'bin').mkdir();(a.out/'receipts').mkdir()
    manifest=json.loads((a.tools/'manifest.json').read_text());cache=Path(manifest['cache_root'])
    assert cache.is_relative_to(Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-budget'))
    before={p for p in cache.rglob('*') if p.is_file()} if cache.exists() else set()
    (a.out/'cpu_cache_before.json').write_text(json.dumps(sorted(str(p) for p in before))+'\n')
    python=sys.executable
    def script(name,code):
        target=a.out/'bin'/name;target.write_text('#!'+python+'\n'+code);target.chmod(0o755);return target
    script('nvidia-smi',"print('CPU DEVICE SUBSTITUTION; no NVIDIA query')\n")
    script('gpu_lock_cpu',"""import os,sys
args=sys.argv[1:];assert args[:5]==['--label','w44-budget','--timeout','1','--']
env=dict(os.environ,GPUWRF_GPU_LOCK_HELD='1')
os.execvpe(args[5],args[5:],env)
""")
    script('nsys',"""import json,os,sys
from pathlib import Path
root=Path(os.environ['BUDGET_SMOKE_ROOT']);assert sys.argv[1]=='profile'
env=dict(os.environ);assert env['JAX_PLATFORMS']=='cuda',env.get('JAX_PLATFORMS')
assert env['GPUWRF_FAST_DEFAULTS']=='1' and env['GPUWRF_MYNN_EDMF']=='1'
assert env['GPUWRF_NESTED_AOT']=='1' and env['GPUWRF_AOT_KEEP_TRACE_METADATA']=='1'
keys=['JAX_PLATFORMS','GPUWRF_FAST_DEFAULTS','GPUWRF_MYNN_EDMF','GPUWRF_NESTED_AOT',
      'GPUWRF_AOT_KEEP_TRACE_METADATA','GPUWRF_JAX_CACHE','GPUWRF_JAX_CACHE_DIR',
      'JAX_COMPILATION_CACHE_DIR','XLA_PYTHON_CLIENT_ALLOCATOR','XLA_PYTHON_CLIENT_MEM_FRACTION']
(root/'exact_gpu_child_env.json').write_text(json.dumps({k:env[k] for k in keys},indent=2)+'\\n')
env.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',BUDGET_CPU_LAUNCH_SMOKE='1')
index=sys.argv.index('python');os.execvpe(sys.argv[index],sys.argv[index:],env)
""")
    bootstrap=a.out/'bootstrap.py'
    bootstrap.write_text("""import importlib.util,os,sys
from pathlib import Path
root=Path(os.environ['BUDGET_SMOKE_ROOT'])
source=next(Path(v) for v in sys.argv[1:] if v.endswith('night_supervisor_design_b.py'))
spec=importlib.util.spec_from_file_location('actual_sealed_supervisor',source)
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
N.M.BASE=root/'receipts'
N.M.RU=root/'real_utc_cpu.sh'
N.M.RU.write_text('echo '+os.environ['BUDGET_SMOKE_EPOCH']+chr(10))
original=N.run_group
def device_layer(command,log,limit,*,env=None):
 assert Path(log).name=='gpu.log','CPU smoke must never run GPU E41'
 command=list(command);matches=[i for i,v in enumerate(command) if Path(v).name=='with_gpu_lock.sh']
 assert len(matches)==1;command[matches[0]]=str(root/'bin/gpu_lock_cpu')
 row=original(command,log,limit,env=env)
 marker=Path(log).parent/'gpu_start.txt'
 if marker.exists():marker.rename(marker.with_name('cpu_device_substitution.txt'))
 return row
N.run_group=device_layer
raise SystemExit(N.M.main())
""")
    script('python3',"import os,sys\nos.execv(sys.executable,[sys.executable,os.environ['BUDGET_SMOKE_BOOTSTRAP'],*sys.argv[1:]])\n")
    admission=dt.datetime(2026,10,24,11,30,tzinfo=dt.timezone.utc)
    env=dict(os.environ,PATH=str(a.out/'bin')+os.pathsep+os.environ['PATH'],
        BUDGET_SMOKE_ROOT=str(a.out),BUDGET_SMOKE_BOOTSTRAP=str(bootstrap),
        BUDGET_SMOKE_EPOCH=str(int(admission.timestamp())),
        GA_START_UTC=admission.isoformat(),GA_END_UTC=(admission+dt.timedelta(hours=3)).isoformat())
    with open(a.out/'launcher.log','w') as log:
        rc=subprocess.run([str(a.launcher)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
    receipts=list((a.out/'receipts').glob('NIGHT_*/night_receipt.json'));assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text());run=receipts[0].parent
    assert rc==receipt['rc']==2,receipt  # real GPU/forecast gate correctly remains incomplete
    assert not receipt['gpu_started']
    assert receipt['start_real_utc']==int(admission.timestamp()),receipt
    assert receipt['phases']['gpu']['rc']==0,receipt
    expected=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-budget')/run.name/'probe'
    assert (run/'probe').resolve()==expected,(run/'probe').resolve()
    model=json.loads((run/'probe/result.json').read_text())
    assert model['verdict']=='LAUNCH_SMOKE_MODEL_REACHED' and model['actual_platform']=='cpu'
    assert not model['gpu_executed'] and model['history_fields']==21
    child=json.loads((a.out/'exact_gpu_child_env.json').read_text());assert child['JAX_PLATFORMS']=='cuda'
    result=dict(verdict='LAUNCH_SMOKE_PASS',launcher=str(a.launcher),
        launcher_sha256=hashlib.sha256(a.launcher.read_bytes()).hexdigest(),receipt=str(receipts[0]),
        exact_gpu_child_env=child,model=model,official_gpu_lock_called=False,gpu_executed=False,
        ga_window_only=True,admission_clock_utc=admission.isoformat(),
        run_scratch_unique=True,
        substitution='GPU lock/NVIDIA/profiler device shims; CPU at device layer; virtual admission clock only',
        launcher_rc=rc,note='rc2 is expected: actual GPU/full-forecast gate is not claimed by this CPU smoke')
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    # CPU JAX entries are backend-keyed, but never leave them in the future GPU
    # cache. Remove only exact newly-created regular files from this smoke.
    for path in {p for p in cache.rglob('*') if p.is_file()}-before:
        assert path.is_relative_to(cache) and not path.is_symlink();path.unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
