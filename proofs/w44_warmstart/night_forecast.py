"""Minimal three-nest 3h cycling job, capped at45min including cold setup.

The independent warm/cache repeat is night_warm_cache.py. The component result
is read from one documented fixed path; only the GA window is supplied by the
sequencer. Input/output hashes, finite/CWP/census and E41 checks ride along.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from launch_preflight import cuda_environment
from launch_host import cpu_environment,cpu_command,e41_arguments,admitted_component,launch_smoke,SMOKE

BASE=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart')
SEAL=BASE/'NIGHT_FORECAST_SEAL_V6.json'
COMPONENT_RESULT=BASE/'NIGHT_COMPONENT_RESULT.json'
FIXED_RESULT=BASE/'NIGHT_FORECAST_RESULT.json'
CAP=2700
CPU_ENV=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(data,indent=2)+'\n');os.replace(pending,path)


def child_environment(snap,run):
    cache=run/'cache'
    configuration=dict(PYTHONPATH=str(snap/'src'),PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',
        GPUWRF_FAST_DEFAULTS='1',GPUWRF_JAX_CACHE='1',GPUWRF_JAX_CACHE_DIR=str(cache/'jax'),
        JAX_COMPILATION_CACHE_DIR=str(cache/'jax'),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES='-1',GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(cache/'jax/xla_gpu_per_fusion_autotune_cache_dir'),
        TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),TMPDIR=str(run/'tmp'),
        XLA_PYTHON_CLIENT_ALLOCATOR='cuda_async',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        GPUWRF_NESTED_PARALLEL_COMPILE='0',GPUWRF_CENSUS='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    base={k:v for k,v in os.environ.items()if not k.startswith(('GPUWRF_','JAX_','XLA_'))and k!=SMOKE}
    return cuda_environment(base,**configuration),configuration


def child_command(seal,case,output,phase):
    return ['bash',seal['arm_shell'],str(case),str(output),phase,seal['runner']]


def forecast_main():
    run=BASE/('NIGHT_FORECAST_'+str(time.time_ns()));run.mkdir()
    started=time.monotonic()
    record={'schema':'w44-warmstart.night-forecast.v1','cap_s':CAP,'phases':{},'verdict':'PREFLIGHT','rc':2}
    receipt=run/'receipt.json';record['receipt_path']=str(receipt)
    write(receipt,record);write(FIXED_RESULT,record);rc=2
    try:
        epoch=lambda s:dt.datetime.fromisoformat(s.replace('Z','+00:00'))
        start,end=map(epoch,(os.environ['GA_START_UTC'],os.environ['GA_END_UTC']))
        assert start.utcoffset()==end.utcoffset()==dt.timedelta(0),'explicit UTC window required'
        now=int(subprocess.check_output(['bash','<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'],env=CPU_ENV,timeout=8,text=True).strip())
        if now<start.timestamp() or end.timestamp()-now<CAP:
            rc=4;record['verdict']='REFUSED_WINDOW';return rc
        if Path('/tmp/wrf_gpu2_quiet').exists():
            rc=3;record['verdict']='REFUSED_QUIET';return rc
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1','call outside an existing GPU lock'
        seal=json.loads(SEAL.read_text());snap=Path(seal['snapshot']);case=Path(seal['case'])
        for path,expected in seal['files'].items():assert sha(path)==expected,path
        component_path=Path(seal['component_recovery'])
        component=admitted_component(seal)
        assert subprocess.check_output(['git','-C',str(snap),'rev-parse','HEAD'],text=True).strip()==seal['commit']
        assert not subprocess.check_output(['git','-C',str(snap),'status','--porcelain','--untracked-files=no'],text=True).strip()
        assert (snap/'data/wrf_pristine').exists()
        manifest=json.loads(Path(seal['case_manifest']).read_text())
        for row in manifest['files']:assert sha(case/row['name'])==row['sha256'],row['name']
        reference=json.loads(Path(seal['cpu_result']).read_text())
        assert reference['status']=='DONE_CLEAN' and reference['input_identity'] and reference['complete_times']
        assert reference['executable_sha256']==seal['cpu_guarded_sha256']
        assert reference['original_gen2_executable_sha256']==seal['cpu_original_sha256']
        assert reference['namelist_diff_keys']==['nproc_x','nproc_y']
        cpu_prep=json.loads(Path(seal['cpu_result']).with_name('PREPARE.json').read_text())
        assert cpu_prep['stage_receipt_sha256']==sha(seal['case_manifest'])
        expected_inputs={r['name']:r['sha256']for r in manifest['files']}
        assert all(row['sha256']==expected_inputs[name]for name,row in cpu_prep['inputs'].items())
        assert cpu_prep['original_namelist_sha256']==expected_inputs['namelist.input']
        assert cpu_prep['all_other_namelist_bytes_exact']
        record.update(source_commit=seal['commit'],component_receipt=str(component_path),component_receipt_sha256=sha(component_path),
            cpu_reference=seal['cpu_result'],cpu_reference_sha256=sha(seal['cpu_result']),case_manifest_sha256=sha(seal['case_manifest']))
        assert shutil.disk_usage('<DATA_ROOT>').free>=365*2**30 and shutil.disk_usage(BASE).free>=20*2**30
        spec=importlib.util.spec_from_file_location('night_groups',seal['supervisor'])
        groups=importlib.util.module_from_spec(spec);spec.loader.exec_module(groups)
        import ctypes
        assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            signal.signal(number,lambda n,_frame:setattr(groups,'SIGNALLED',n))
        cache=run/'cache';(cache/'jax').mkdir(parents=True);(cache/'triton').mkdir();(cache/'cuda').mkdir();(run/'tmp').mkdir()
        deadline=groups.boot()+CAP-30
        # Start from released defaults, avoiding settings left by an earlier
        # sequencer arm. Both phases receive exactly the same compiler flags.
        env,configuration=child_environment(snap,run)
        record.update(gpu_environment_config=configuration,cache_path=str(cache),cold_run=str(run/'cold/forecast'))
        def group(name,command,limit,environment):
            row=groups.run_group(command,run/(name+'.log'),min(limit,max(1,deadline-groups.boot())),env=environment)
            record['phases'][name]=row;write(receipt,record);return row['rc']
        def cpu(name,args,limit):
            if Path('/tmp/wrf_gpu2_quiet').exists():return 3
            return group(name,cpu_command(snap,name,args),limit,cpu_environment(snap))
        def gpu(phase,limit):
            command=[str(snap/'scripts/with_gpu_lock.sh'),'--label','w44-warmstart','--timeout','1','--',
                     *child_command(seal,case,run/phase,phase)]
            return group(phase+'_gpu',command,limit,env)
        def e41(name,phase):
            proof=run/phase/'forecast/proofs/nested_pipeline_run.json'
            args=e41_arguments(snap,cache/'jax',run/name,proof=proof if proof.exists()else None)
            return cpu(name,args,120)
        pending=[]
        rc=gpu('cold',2400)
        if (run/'cold/forecast/PLATFORM.json').exists():
            gate_rc=e41('e41_cold','cold')
            if gate_rc in (3,124,137):pending.append('e41_cold')
            else:rc=rc or gate_rc
        if rc:return rc
        rc=cpu('check_cold',[sys.executable,seal['checker'],'cold',str(run/'cold/forecast'),str(run/'COLD_CHECK.json'),str(cache/'jax')],120)
        if rc in (3,124,137):pending.append('check_cold');rc=0
        elif rc:return rc
        if pending:
            record.update(verdict='GPU_COMPLETE_CPU_GATES_PENDING',pending_cpu_gates=pending,
                reason='QUIET/resource queue/cgroup cap; raw GPU validation kept, no scientific PASS invented')
            rc=3;return rc
        record.update(verdict='THREE_NEST_3H_FINITE_PASS',wrf_fidelity='PENDING D6 vs private guarded CPU reference',
            byte_checks='sealed CPU-identical input set; per-file GPU hashes; optional warm repeat is separate',
            timing_scope='cold setup +3h CENSUS=1 qualification, not warm speed or release24h S1')
        return 0
    except (KeyError,AssertionError,ValueError,OSError,subprocess.SubprocessError)as exc:
        record.update(error=str(exc),verdict='REFUSED_OR_ERROR');rc=4 if isinstance(exc,KeyError)else 2;return rc
    finally:
        record['rc']=rc
        if rc and record['verdict'] not in ('REFUSED_WINDOW','REFUSED_QUIET','REFUSED_OR_ERROR','GPU_COMPLETE_CPU_GATES_PENDING'):
            record['verdict']='FAILED_OR_INCOMPLETE'
        record['elapsed_s']=time.monotonic()-started
        record['cache_disposition']='retained until E41 + warm identity final; worker removes exact owned cache immediately after receipt'
        write(receipt,record);write(FIXED_RESULT,record)
        print(json.dumps({'receipt':str(receipt),'rc':rc,'verdict':record['verdict']}),flush=True)


def main():
    if os.environ.get(SMOKE)=='1':return launch_smoke('cold',sys.modules[__name__])
    return forecast_main()


if __name__=='__main__':raise SystemExit(main())
