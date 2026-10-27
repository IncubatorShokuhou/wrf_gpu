"""CPU-only continuation: wait for the booked GPU job, then score D6 once.

This process never starts a GPU job or changes a scientific gate. It can
score a completed cold GPU run even when host post-gates are still pending;
those pending gates remain explicit and are never called passed here.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

BASE=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart')
MAIN=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
SOURCE=BASE/'WS19_snapshot'
COMMIT='0c315f3108eeb08222b79c30468aea24d5ad7669'
SCORE=BASE/'night_forecast_tools_v3/score_twin.py'
SCORE_SHA='b8f51aebf1388cf68a438deb020a804b949e952c5ec1da850b7292b2c1557861'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def utc():
    return subprocess.check_output(['bash','<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh','--iso'],text=True,timeout=10).strip()


def post(text):
    with (MAIN/'MAILBOX.md').open('a')as f:f.write(utc()+' w44-warmstart: '+text+'\n')


def main():
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert hashlib.sha256(SCORE.read_bytes()).hexdigest()==SCORE_SHA
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    (BASE/'AUTO_D6_WATCHER.json').write_text(json.dumps({'pid':os.getpid(),'source':COMMIT,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status':'WAITING_BOOKED_GPU_RECEIPT','GPU_commands':False,'poll_seconds':15,
        'scope':'CPU-only waiter/scorer; stops on GPU failure, never launches/retries GPU'},indent=2)+'\n')
    proc=None
    def interrupted(number,_frame):
        if proc is not None and proc.poll()is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    while True:
        component=read(BASE/'NIGHT_COMPONENT_RECOVERED_RESULT_V1.json') or read(BASE/'NIGHT_COMPONENT_RESULT.json')
        if (component and component.get('gpu_started')and component.get('rc')not in (0,None)
                and component.get('verdict')!='PREFLIGHT'):
            post('ACTUAL COMPONENT NOTGREEN; CPU D6 watcher stopped withoutGPUselflaunch. '+str(component.get('receipt_path')))
            return 1
        primary=read(BASE/'NIGHT_FORECAST_RESULT.json')
        phase=(primary or {}).get('phases',{}).get('cold_gpu')
        if primary and phase and primary.get('verdict')!='PREFLIGHT':
            assert primary['source_commit']==COMMIT
            if phase['rc']!=0:
                post('ACTUAL PRIMARY GPU failed/incomplete; D6 notstarted. '+str(primary.get('receipt_path')))
                return 1
            break
        time.sleep(15)
    while Path('/tmp/wrf_gpu2_quiet').exists():time.sleep(2)
    output=BASE/'AUTO_D6'
    claim=BASE/'AUTO_D6.claim'
    try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    except FileExistsError:
        post('D6 claim alreadyexists; no duplicate scoring. '+str(claim));return 0
    os.write(fd,(str(os.getpid())+'\n').encode());os.close(fd)
    temporary=BASE/'AUTO_D6_TMP';mpl=BASE/'AUTO_D6_MPL'
    temporary.mkdir();mpl.mkdir()
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',CUDA_VISIBLE_DEVICES='',
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TMPDIR=str(temporary),MPLCONFIGDIR=str(mpl))
    command=['timeout','--signal=TERM','--kill-after=30','3600','taskset','-c','24,25,28','nice','-n','19',
        str(SOURCE/'scripts/with_ram_slot.sh'),'--label','w44-warmstart-D6','--max','12G','--timeout','1800','--','bash','-c',
        'while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 2; done; if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"',
        'w44-d6',sys.executable,str(SCORE),primary['receipt_path'],str(output)]
    start=time.monotonic();post('CPU D6 START once onprimarycold vsretainedguardedCPU;noextraGPU. '+str(output))
    with (BASE/'AUTO_D6.log').open('x')as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rc=proc.wait()
    summary=read(output/'d6/summary.json')
    report={'rc':rc,'wall_s':time.monotonic()-start,'primary_receipt':primary['receipt_path'],
        'D6_pass':None if summary is None else summary['pass'],
        'output_integrity_pass':None if summary is None else summary['output_integrity_pass'],
        'source':COMMIT,'scope':'same-input3hprobe;notrelease24/72h;GPUpostgates/F2justification remain explicit'}
    (BASE/'AUTO_D6_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    while Path('/tmp/wrf_gpu2_quiet').exists():time.sleep(2)
    for path in (temporary,mpl):
        assert path.parent==BASE and path.name in ('AUTO_D6_TMP','AUTO_D6_MPL')
        shutil.rmtree(path)
    with (MAIN/'EXPERIMENTS.tsv').open('a')as f:
        f.write('\t'.join([utc(),'w44-warmstart','gpt-6.1-sol','AUTO-D6-3H',json.dumps(report,separators=(',',':')),
            'artifact='+str(output)+';cache=NONE;work=actualsameCMay4_00GPU3h3nests vsapprovedguardCPU1c/orig1d,standardfrozenD6manifest,CPU24,25,28nice19JAXcpu12Gtimeout3600'])+'\n')
    post('D6 3h probe DONE rc='+str(rc)+' boundedpass='+str(report['D6_pass'])+' integrity='+str(report['output_integrity_pass'])+
         ';NOT24/72hrelease;F2justification/postgatesseparate; '+str(BASE/'AUTO_D6_RESULT.json'))
    return rc


if __name__=='__main__':raise SystemExit(main())
