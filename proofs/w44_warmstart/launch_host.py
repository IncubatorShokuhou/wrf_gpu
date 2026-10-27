"""Shared production CPU child environment and actual all-phase launch smoke."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SMOKE='W44_WARMSTART_LAUNCH_SMOKE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_environment(snap):
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',GPUWRF_JAX_CACHE='0',
             PYTHONPATH=str(Path(snap)/'src'),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('JAX_PLATFORM_NAME',None)
    return env


def cpu_command(snap,label,args):
    # Manager 24 Oct 15:28: CPU post phases must not depend on the common RAM
    # queue. One sequential host child, hard 12 GiB cgroup, no swap; no slot.
    return ['taskset','-c','24,25,28','nice','-n','19',
            'systemd-run','--user','--scope','--quiet','-p','MemoryMax=12G','-p','MemorySwapMax=0','--','bash','-c',
            'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"','w44-cpu',*map(str,args)]


def e41_arguments(snap,cache,out,proof=None,pattern=None):
    args=[sys.executable,str(Path(snap)/'scripts/wave_cubin_gate.py'),str(cache),str(out)]
    if pattern:args.append(pattern)
    if proof:args+=['--run-proof',str(proof)]
    return args


def admitted_component(seal):
    """A supplement cannot rewrite or waive the failed original E41 attempt."""
    path=Path(seal['component_recovery'])
    recovery=json.loads(path.read_text())
    assert recovery['verdict']=='GPU_ORIGINAL_INIT_PASS' and recovery['rc']==0
    original_path=Path(recovery['original_receipt'])
    assert sha(original_path)==recovery['original_receipt_sha256']
    original=json.loads(original_path.read_text())
    assert original['source_commit']==seal['commit']==recovery['source_commit']
    assert original['gpu_started'] and original['phases']['gpu']['rc']==0
    component=original['component']
    assert component['passed'] and component['backend']=='gpu'
    assert len(component['records'])==6 and all(r['passed']for r in component['records'])
    assert recovery['phases']['e41']['rc']==0
    gate_path=Path(recovery['e41_gate_path'])
    assert sha(gate_path)==recovery['e41_gate_sha256']
    gate=json.loads(gate_path.read_text())
    assert gate['e41_gate']=='PASS' and gate['n_entries']>0
    assert gate['coverage']['n_discovered']==gate['coverage']['n_scanned']==gate['n_entries']
    assert {r['path']for r in gate['entries']}==set(recovery['scanned_cache_files'])
    return recovery


def launch_smoke(mode,forecast,warm=None):
    """Run the exact shell/controller and child builders; never publish GPU PASS."""
    from launch_preflight import CPU_PREFLIGHT
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    seal=json.loads(forecast.SEAL.read_text())
    for path,expected in seal['files'].items():assert sha(path)==expected,path
    recovery=admitted_component(seal)
    snap=Path(seal['snapshot'])
    assert subprocess.check_output(['git','-C',str(snap),'rev-parse','HEAD'],text=True).strip()==seal['commit']
    root=forecast.BASE/('LAUNCH_SMOKE_V6_'+mode.upper()+'_'+str(time.time_ns()))
    root.mkdir();(root/'tmp').mkdir()
    for folder in ('cache/jax','cache/triton','cache/cuda'):(root/folder).mkdir(parents=True)
    product,configuration=forecast.child_environment(snap,root)
    if warm:product=warm.child_environment(configuration)
    assert product['JAX_PLATFORMS']=='cuda' and product['CUDA_VISIBLE_DEVICES']=='0'
    dry=dict(product,JAX_PLATFORMS='cpu',**{CPU_PREFLIGHT:'1'})
    difference=sorted(k for k in product.keys()|dry.keys()if product.get(k)!=dry.get(k))
    assert difference==sorted(['JAX_PLATFORMS',CPU_PREFLIGHT])
    output=root/mode
    phases=[]
    def run(name,command,env,cap=120):
        start=time.monotonic()
        with (root/(name+'.log')).open('w')as log:
            child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        row=dict(name=name,command=command,rc=child.returncode,wall_s=time.monotonic()-start,
                 JAX_PLATFORMS=env['JAX_PLATFORMS'],PYTHONPATH=env['PYTHONPATH'])
        phases.append(row)
        assert child.returncode==0,(name,child.returncode,str(root/(name+'.log')))
    run('model',cpu_command(snap,'launch-smoke-model',forecast.child_command(seal,seal['case'],output,mode)),dry)
    # Real E41 CLI, native ELF/HLO parsing and resource scanner on retained GPU
    # cache. Only cache selection differs from the future forecast scan.
    run('e41',cpu_command(snap,'launch-smoke-E41',e41_arguments(snap,recovery['cache'],root/'e41',pattern='jit__lambda*')),
        cpu_environment(snap))
    assert json.loads((root/'e41/wave_gate.json').read_text())['e41_gate']=='PASS'
    run('checker',cpu_command(snap,'launch-smoke-check',[sys.executable,seal['checker'],'launch-smoke',
        str(output/'forecast'),str(root/'CHECK.json'),str(root/'e41/wave_gate.json')]),cpu_environment(snap))
    report=dict(passed=True,mode=mode,phases=phases,source_commit=seal['commit'],
        launcher=os.environ['W44_WARMSTART_LAUNCHER'],launcher_sha256=sha(os.environ['W44_WARMSTART_LAUNCHER']),
        seal_sha256=sha(forecast.SEAL),component_recovery=seal['component_recovery'],
        product_env={k:product[k]for k in configuration},only_model_env_changes=difference,
        GPU_executed=False,scope='CPU mini dispatch + REAL retained-GPU E41; no forecast fidelity claim')
    (root/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'LAUNCH_SMOKE_PASS':True,'receipt':str(root/'receipt.json'),'phases':phases}),flush=True)
    return 0
