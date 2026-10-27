"""Separate optional 15min warm/cache repeat; fixed primary receipt, GA only."""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from night_forecast import BASE,CPU_ENV,sha,write,child_command
from launch_preflight import cuda_environment
from launch_host import cpu_environment,cpu_command,e41_arguments,admitted_component,launch_smoke,SMOKE

SEAL=BASE/'NIGHT_FORECAST_SEAL_V6.json'
PRIMARY=BASE/'NIGHT_FORECAST_RESULT.json'
CAP=900


def child_environment(configuration):
    assert 'GPUWRF_GPU_LOCK_HELD'not in configuration
    assert configuration['JAX_PLATFORMS']=='cuda','primary must use explicit product CUDA backend'
    return cuda_environment({k:v for k,v in os.environ.items()if not k.startswith(('GPUWRF_','JAX_','XLA_'))and k!=SMOKE},**configuration)


def warm_main():
    started=time.monotonic()
    remaining=lambda limit:max(1,min(limit,CAP-20-(time.monotonic()-started)))
    run=BASE/('NIGHT_WARM_CACHE_'+str(time.time_ns()));run.mkdir()
    record={'cap_s':CAP,'rc':2,'verdict':'PREFLIGHT','phases':{}};receipt=run/'receipt.json';write(receipt,record);rc=2
    try:
        start,end=[dt.datetime.fromisoformat(os.environ[k].replace('Z','+00:00'))for k in ('GA_START_UTC','GA_END_UTC')]
        assert start.utcoffset()==end.utcoffset()==dt.timedelta(0)
        now=int(subprocess.check_output(['bash','<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'],env=CPU_ENV,timeout=8,text=True).strip())
        if now<start.timestamp()or end.timestamp()-now<CAP:
            rc=4;record['verdict']='REFUSED_WINDOW';return rc
        assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
        assert shutil.disk_usage('<DATA_ROOT>').free>=365*2**30,'disk reserve below 365GiB'
        primary=json.loads(PRIMARY.read_text());seal=json.loads(SEAL.read_text())
        assert primary['rc']==0 and primary['verdict']=='THREE_NEST_3H_FINITE_PASS'
        assert primary['source_commit']==seal['commit']
        for path,expected in seal['files'].items():assert sha(path)==expected,path
        admitted_component(seal)
        snap=Path(seal['snapshot']);parent=Path(primary['receipt_path']).parent
        assert subprocess.check_output(['git','-C',str(snap),'rev-parse','HEAD'],text=True).strip()==seal['commit']
        assert not subprocess.check_output(['git','-C',str(snap),'status','--porcelain','--untracked-files=no'],text=True).strip()
        manifest=json.loads(Path(seal['case_manifest']).read_text())
        for row in manifest['files']:assert sha(Path(seal['case'])/row['name'])==row['sha256'],row['name']
        assert json.loads((parent/'COLD_CHECK.json').read_text())['passed']
        config=primary['gpu_environment_config'];env=child_environment(config)
        assert Path(config['GPUWRF_JAX_CACHE_DIR']).is_dir(),'primary cache unavailable'
        spec=importlib.util.spec_from_file_location('night_groups',seal['supervisor'])
        groups=importlib.util.module_from_spec(spec);spec.loader.exec_module(groups)
        import ctypes
        assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            signal.signal(number,lambda n,_frame:setattr(groups,'SIGNALLED',n))
        command=[str(snap/'scripts/with_gpu_lock.sh'),'--label','w44-warmstart','--timeout','1','--',
                 *child_command(seal,seal['case'],run/'warm','warm')]
        record.update(primary_receipt=str(PRIMARY),primary_receipt_sha256=sha(PRIMARY),source_commit=seal['commit'])
        row=groups.run_group(command,run/'warm_gpu.log',remaining(600),env=env);record['phases']['warm_gpu']=row;rc=row['rc']
        def failed_e41():
            if not (run/'warm/forecast/PLATFORM.json').exists():return
            if Path('/tmp/wrf_gpu2_quiet').exists():
                record['phases']['e41_changed_warm']={'rc':3,'status':'PENDING_QUIET'};return
            command=cpu_command(snap,'warm-E41',e41_arguments(snap,config['GPUWRF_JAX_CACHE_DIR'],run/'e41_changed_warm',
                proof=parent/'cold/forecast/proofs/nested_pipeline_run.json'))
            record['phases']['e41_changed_warm']=groups.run_group(command,run/'e41_changed_warm.log',remaining(180),env=cpu_environment(snap))
        if rc:
            failed_e41();return rc
        command=cpu_command(snap,'warm-check',[sys.executable,seal['checker'],'warm',str(run/'warm/forecast'),str(run/'WARM_CHECK.json'),
            str(parent/'COLD_CHECK.json'),config['GPUWRF_JAX_CACHE_DIR']])
        row=groups.run_group(command,run/'warm_check.log',remaining(180),env=cpu_environment(snap));record['phases']['check_warm']=row;rc=row['rc']
        if rc in (3,124,137):record['verdict']='GPU_COMPLETE_CPU_CHECK_PENDING';return rc
        if rc:
            failed_e41();return rc
        record.update(verdict='WARM_CACHE_BYTES_PASS',warm_s_per_fc_h=record['phases']['warm_gpu']['elapsed_s']/3,
            compile_count=0,finite_cloud='carried from 12byte-identical primary files',
            e41='carried from byte-identical primary native executables',
            scope='3h CENSUS=1 shared-host proxy/GPU-vs-GPU regression; CPU D6 remains separate')
        return 0
    except (KeyError,AssertionError,ValueError,OSError,subprocess.SubprocessError)as exc:
        record.update(error=str(exc),verdict='REFUSED_OR_ERROR');rc=4 if isinstance(exc,KeyError)else 2;return rc
    finally:
        record['rc']=rc
        record['elapsed_s']=time.monotonic()-started
        if rc and record['verdict']=='PREFLIGHT':record['verdict']='FAILED_OR_INCOMPLETE'
        write(receipt,record);write(BASE/'NIGHT_WARM_RESULT.json',record)
        print(json.dumps({'receipt':str(receipt),'rc':rc,'verdict':record['verdict']}),flush=True)


def main():
    if os.environ.get(SMOKE)=='1':
        import night_forecast
        return launch_smoke('warm',night_forecast,sys.modules[__name__])
    return warm_main()


if __name__=='__main__':raise SystemExit(main())
