"""CUDA dispatch qualification using the frozen production builders and runner.

Only CASE is a real three-nest 24x24 crop. Duration, physics, GPU runner,
arm shell, checker, child environment and command builders stay production.
"""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart')
SEAL=BASE/'CUDA_MINI_SEAL_V3.json'
CPU_PROBE='W44_WARMSTART_MINI_CPU_PROBE'
CAP=900


def main():
    mode=os.environ['W44_WARMSTART_MINI_MODE'];assert mode in ('forecast','warm')
    dry=os.environ.get(CPU_PROBE)=='1'
    seal=json.loads(SEAL.read_text())
    sys.path.insert(0,seal['production_tools'])
    import night_forecast as F
    import night_warm_cache as W
    from launch_host import cpu_environment,cpu_command,e41_arguments,admitted_component
    from launch_preflight import CPU_PREFLIGHT
    for p,h in seal['files'].items():assert F.sha(p)==h,p
    prod=json.loads(F.SEAL.read_text());snap=Path(prod['snapshot'])
    recovered=admitted_component(prod)
    assert subprocess.check_output(['git','-C',str(snap),'rev-parse','HEAD'],text=True).strip()==prod['commit']
    assert not subprocess.check_output(['git','-C',str(snap),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert shutil.disk_usage('<DATA_ROOT>').free>=365*2**30,'disk reserve below365GiB'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    if not dry:
        epoch=lambda key:dt.datetime.fromisoformat(os.environ[key].replace('Z','+00:00'))
        begin,end=epoch('GA_START_UTC'),epoch('GA_END_UTC')
        now=int(subprocess.check_output(['bash','<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'],env=cpu_environment(snap),text=True).strip())
        assert begin.timestamp()<=now and end.timestamp()-now>=CAP,'GA window must cover900s'
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1'
    run=BASE/('CUDA_MINI_'+mode.upper()+('_CPU_'if dry else'_')+str(time.time_ns()));run.mkdir()
    record=dict(schema='w44-warmstart.cuda-mini.v1',mode=mode,CPU_probe=dry,rc=2,verdict='RUNNING',
                phases={},production_launcher=seal['production_launchers'][mode],source_commit=prod['commit'],
                mini_case=seal['mini_case'],grid_only_difference=seal['difference'],receipt_path=str(run/'receipt.json'))
    F.write(run/'receipt.json',record)
    spec=importlib.util.spec_from_file_location('mini_groups',prod['supervisor'])
    groups=importlib.util.module_from_spec(spec);spec.loader.exec_module(groups)
    import ctypes
    assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(number,lambda n,_frame:setattr(groups,'SIGNALLED',n))
    deadline=groups.boot()+(1800 if dry else CAP)-5
    config=None;primary=None
    if mode=='warm'and not dry:
        primary=json.loads((BASE/'CUDA_MINI_FORECAST_RESULT_V3.json').read_text())
        assert primary['rc']==0 and primary['verdict']=='CUDA_MINI_FORECAST_PASS'
        assert primary['source_commit']==prod['commit']and primary['mini_case']==seal['mini_case']
        parent=Path(primary['receipt_path']).parent;config=primary['gpu_environment_config']
    if config is None:
        for name in ('cache/jax','cache/triton','cache/cuda','tmp'):(run/name).mkdir(parents=True)
        env,config=F.child_environment(snap,run)
    else:env=W.child_environment(config)
    if mode=='warm':env=W.child_environment(config)
    env.pop(CPU_PROBE,None);env.pop('W44_WARMSTART_MINI_MODE',None)
    assert env['JAX_PLATFORMS']=='cuda'and env['CUDA_VISIBLE_DEVICES']=='0'
    record.update(gpu_environment_config=config,product_environment_backend='cuda',
                  CPU_post_policy='own sequential MemoryMax12G/noSwap cgroup; NO common RAM slot')
    rc=2
    def phase(name,command,environment,cap):
        result=groups.run_group(command,run/(name+'.log'),min(cap,max(1,deadline-groups.boot())),env=environment)
        record['phases'][name]=dict(result,command=command,JAX_PLATFORMS=environment['JAX_PLATFORMS'],PYTHONPATH=environment.get('PYTHONPATH'))
        F.write(run/'receipt.json',record);return result['rc']
    try:
        label='cold'if mode=='forecast'else'warm';output=run/label
        builder=F.child_command if mode=='forecast'else W.child_command
        command=builder(prod,seal['mini_case'],output,label)
        if dry:
            env=dict(env,JAX_PLATFORMS='cpu',**{CPU_PREFLIGHT:'1'})
            command=cpu_command(snap,'cuda-mini-CPU-model',command)
        else:command=[str(snap/'scripts/with_gpu_lock.sh'),'--label','w44-warmstart','--timeout','1','--',*command]
        rc=phase('model',command,env,90 if dry else 780)
        # Always run the actual native scan, including the warm fallback path.
        if dry:
            cache=recovered['cache'];args=e41_arguments(snap,cache,run/'e41',pattern='jit__lambda*')
        else:
            cache=config['GPUWRF_JAX_CACHE_DIR']
            proof=output/'forecast/proofs/nested_pipeline_run.json'
            args=e41_arguments(snap,cache,run/'e41',proof=proof if proof.exists()else None)
        gate_rc=phase('e41',cpu_command(snap,'cuda-mini-E41',args),cpu_environment(snap),120)
        if rc or gate_rc:rc=rc or gate_rc;return rc
        assert json.loads((run/'e41/wave_gate.json').read_text())['e41_gate']=='PASS'
        if dry:args=[sys.executable,prod['checker'],'launch-smoke',str(output/'forecast'),str(run/'CHECK.json'),str(run/'e41/wave_gate.json')]
        elif mode=='forecast':args=[sys.executable,prod['checker'],'cold',str(output/'forecast'),str(run/'COLD_CHECK.json'),cache]
        else:args=[sys.executable,prod['checker'],'warm',str(output/'forecast'),str(run/'WARM_CHECK.json'),str(parent/'COLD_CHECK.json'),cache]
        rc=phase('checker',cpu_command(snap,'cuda-mini-check',args),cpu_environment(snap),60)
        if rc:return rc
        record['verdict']='CPU_ALL_PHASES_PASS'if dry else'CUDA_MINI_'+mode.upper()+'_PASS'
        record['scope']='CPU dispatch only + real retained CUDA E41'if dry else'real3nest24x24x44/3h miniature; NOT full-grid D6'
        return 0
    finally:
        record['rc']=rc;F.write(run/'receipt.json',record)
        if not dry:F.write(BASE/('CUDA_MINI_'+mode.upper()+'_RESULT_V3.json'),record)
        print(json.dumps(dict(receipt=str(run/'receipt.json'),rc=rc,verdict=record['verdict'])),flush=True)


if __name__=='__main__':raise SystemExit(main())
