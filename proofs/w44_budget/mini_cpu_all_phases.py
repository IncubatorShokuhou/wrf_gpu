"""Exercise sealed mini commands, real RAM scopes, checker and real E41 on CPU.

Only device boundaries are substituted. CPU model outputs feed the real pristine
term checker. After CPU execution, the native-export device boundary supplies a
retained REAL CUDA executable to the unchanged E41 scanner. That fixture proves
scanner/command plumbing, never the new mini executable's GPU qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--caps',default='2400,1800,540',help='CPU-only CAP,GPU_CAP,E41_CAP rehearsal allowance (s)')
    args=parser.parse_args()
    manifest=json.loads((args.launcher.parent/args.launcher.stem/'manifest.json').read_text())
    caps=[int(v) for v in args.caps.split(',')];assert len(caps)==3
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    args.out.mkdir(exist_ok=False);(args.out/'bin').mkdir();(args.out/'receipts').mkdir()
    python=sys.executable
    def script(name,code):
        path=args.out/'bin'/name;path.write_text('#!'+python+'\n'+code);path.chmod(0o555)
    script('nvidia-smi',"print('CPU device substitution; no NVIDIA query')\n")
    script('gpu_lock_cpu',"""import os,sys
args=sys.argv[1:];assert args[:5]==['--label','w44-budget','--timeout','1','--']
os.execvpe(args[5],args[5:],dict(os.environ,GPUWRF_GPU_LOCK_HELD='1'))
""")
    script('nsys',"""import hashlib,json,os,subprocess,sys
from pathlib import Path
import zstandard
root=Path(os.environ['BUDGET_MINI_CPU_ROOT'])
assert sys.argv[1]=='profile'
env=dict(os.environ);assert env['JAX_PLATFORMS']=='cuda'
keys=['JAX_PLATFORMS','GPUWRF_FAST_DEFAULTS','GPUWRF_MYNN_EDMF','GPUWRF_NESTED_AOT',
 'GPUWRF_AOT_KEEP_TRACE_METADATA','GPUWRF_JAX_CACHE','GPUWRF_JAX_CACHE_DIR',
 'JAX_COMPILATION_CACHE_DIR','XLA_PYTHON_CLIENT_ALLOCATOR','XLA_PYTHON_CLIENT_MEM_FRACTION']
(root/'product_child_env.json').write_text(json.dumps({k:env[k] for k in keys},indent=2))
group=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
scope=Path('/sys/fs/cgroup')/group.lstrip('/')
memory=(scope/'memory.max').read_text().strip();swap=(scope/'memory.swap.max').read_text().strip()
want=int(os.environ['BUDGET_MINI_MEMORY_MAX'].rstrip('G'))*2**30
assert memory==str(want) and swap=='0',(memory,swap)
(root/'actual_memory_scope.json').write_text(json.dumps(dict(cgroup=group,memory_max=memory,memory_swap_max=swap)))
env.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',BUDGET_CPU_ALL_PHASES='1')
env['XLA_FLAGS']='--xla_cpu_max_isa=AVX'  # same CPU-device setting as P26--35
index=sys.argv.index('python')
rc=subprocess.run(sys.argv[index:],env=env).returncode
if rc==0:
 cache=Path(env['GPUWRF_JAX_CACHE_DIR'])
 fixture=Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget/NIGHT_1792792718433431644_987496/cache/jax/jit__advance_chunk_fori-19377b5ab009a9e182606836351afe1169d95400a37c671a5dbfec75fba38b51-cache')
 raw=fixture.read_bytes();native=zstandard.ZstdDecompressor().decompress(raw,max_output_size=1<<33)[4:]
 donor=root/'cuda_native_fixture.xlaexec';donor.write_bytes(native)
 compressed=root/'cuda_jit_fixture-cache';compressed.write_bytes(raw)
 replacements=[]
 paths=list((cache/'aot').glob('*/*.xlaexec'))+list((cache/'aot').glob('*/*/*.xlaexec'))
 paths+=list(cache.glob('jit__advance_chunk*-cache'))+list(cache.glob('jit_fused_jit*-cache'))
 for path in paths:
  replacements.append(dict(path=str(path),cpu_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  path.unlink();os.link(compressed if path.name.endswith('-cache') else donor,path)
 (root/'native_device_substitution.json').write_text(json.dumps(dict(
  fixture=str(fixture),fixture_sha256=hashlib.sha256(raw).hexdigest(),
  scope='retained actual CUDA fixture only at native-export boundary; NOT new executable qualification',
  replacements=replacements),indent=2))
raise SystemExit(rc)
""")
    bootstrap=args.out/'bootstrap.py'
    bootstrap.write_text("""import importlib.util,json,os,sys
from pathlib import Path
root=Path(os.environ['BUDGET_MINI_CPU_ROOT'])
source=next(Path(v) for v in sys.argv[1:] if v.endswith('night_supervisor_design_b.py'))
spec=importlib.util.spec_from_file_location('sealed_mini',source)
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
N.M.BASE=root/'receipts'
# CPU compilation is slower than CUDA. This hardware-only rehearsal allowance
# changes neither sealed command builders nor the actual 300-second CUDA cap.
N.M.CAP,N.M.GPU_CAP,N.M.E41_CAP=(int(v) for v in os.environ['BUDGET_MINI_CPU_CAPS'].split(','))
N.M.RU=root/'clock.sh';N.M.RU.write_text('echo 1792841400\\n')
original=N.run_group
def device_boundary(command,log,limit,*,env=None):
 command=list(command)
 if Path(log).name=='gpu.log':
  matches=[i for i,v in enumerate(command) if Path(v).name=='with_gpu_lock.sh']
  assert len(matches)==1
  command[matches[0]]=str(root/'bin/gpu_lock_cpu')
 return original(command,log,limit,env=env)
N.run_group=device_boundary
write=N.M.write_json
def cpu_receipt(path,record):
 if path.name=='night_receipt.json':
  record=dict(record,gpu_started=False,cpu_device_boundary_reached=record.get('gpu_started',False),
   actual_gpu_executed=False,official_gpu_lock_called=False,
   device_substitution='CPU model + retained actual CUDA E41 fixture; not GPU gate evidence')
 return write(path,record)
N.M.write_json=cpu_receipt
raise SystemExit(N.M.main())
""")
    script('python3',"import os,sys\nos.execv(sys.executable,[sys.executable,os.environ['BUDGET_MINI_CPU_BOOTSTRAP'],*sys.argv[1:]])\n")
    env=dict(os.environ,PATH=str(args.out/'bin')+os.pathsep+os.environ['PATH'],
        BUDGET_MINI_CPU_ROOT=str(args.out),BUDGET_MINI_CPU_BOOTSTRAP=str(bootstrap),
        GA_START_UTC='2026-10-24T11:30:00Z',GA_END_UTC='2026-10-24T14:30:00Z',BUDGET_MINI_CPU_CAPS=args.caps,
        BUDGET_MINI_MEMORY_MAX=manifest.get('gpu_memory_max','12G'))
    with open(args.out/'launcher.log','w') as log:
        rc=subprocess.run([str(args.launcher)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=caps[0]+100).returncode
    receipts=list((args.out/'receipts').glob('NIGHT_*/night_receipt.json'))
    assert len(receipts)==1,receipts
    receipt=json.loads(receipts[0].read_text());run=receipts[0].parent
    post=json.loads((run/'post_cpu.json').read_text()) if (run/'post_cpu.json').exists() else {}
    probe=json.loads((run/'probe/result.json').read_text()) if (run/'probe/result.json').exists() else {}
    result=dict(schema='wrf_gpu2.w44_budget.mini_cpu.v1',launcher=str(args.launcher),
        launcher_sha256=sha(args.launcher),rc=rc,receipt=str(receipts[0]),phases=receipt['phases'],
        model_verdict=probe.get('verdict'),post_verdict=post.get('verdict'),
        gpu_executed=False,official_gpu_lock_called=False,
        native_fixture_scope='scanner plumbing only; actual mini CUDA E41 still mandatory',
        expected_probe_verdict=manifest['expected_probe_verdict'],caps_s=caps,
        passed=rc==0 and post.get('verdict')=='CPU_GATES_PASS' and probe.get('verdict')==manifest['expected_probe_verdict'])
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
