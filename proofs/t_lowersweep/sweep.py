"""Sequential bounded/zero-swap CPU preflight; append-only experiment receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from process_result import classify

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--target',type=Path,required=True)
p.add_argument('--cases',nargs='*')
p.add_argument('--reuse-from',type=Path,help='Reuse completed rows/receipts without modifying the prior sweep.')
p.add_argument('--rerun',nargs='*',default=[],help='Explicit interrupted/changed cases to rerun in this new output directory.')
p.add_argument('--first',help='Run this interrupted case before other missing cases.')
p.add_argument('--ram-wrapper',type=Path,default=Path('<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh'))
p.add_argument('--ram-max', choices=('6G','12G'), default='6G',
               help='12G requires an explicit manager-authorized singleton window.')
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
main=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
prep=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
tool=Path(__file__).with_name('lower_case.py').resolve()
a.out.mkdir(parents=True,exist_ok=True)
matrix=json.loads((prep/'MATRIX.json').read_text())
progress=json.loads((prep/'GPU_PROGRESS.json').read_text())
ran={r['case'] for r in progress['rows'] if r.get('actual_gpu_receipt')}
cases=sorted(matrix['cases'],key=lambda c:(c['id'] in ran,c['id']))
if a.cases: cases=[c for c in cases if c['id'] in a.cases]
if a.first:cases.sort(key=lambda c:c['id']!=a.first)
assert a.ram_wrapper.is_file(), 'Manager RAM-slot wrapper is required for every heavy child'
previous={}
if a.reuse_from:
    prior_summary=json.loads((a.reuse_from/'RESULTS.json').read_text())
    previous={r['case']:r for r in prior_summary.get('prior_rows',[])+prior_summary['rows']}
(a.out/'MATRIX_SNAPSHOT.json').write_text(json.dumps(matrix,indent=2)+'\n')
(a.out/'GPU_PROGRESS_SNAPSHOT.json').write_text(json.dumps(progress,indent=2)+'\n')
rows=[]
def utc():return datetime.now(timezone.utc).isoformat()
def append(path,text):
    with path.open('a') as f:f.write(text+'\n')
def quiet():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        print('QUIET; no case launch or cleanup',flush=True);time.sleep(15)
def publish():
    (a.out/'RESULTS.json').write_text(json.dumps(dict(utc=utc(),source=str(a.source),
        target=str(a.target),rows=rows,prior_rows=list(previous.values()),total=len(cases),
        scope='CPU preflight; no GPU or fidelity claim'),indent=2)+'\n')
    lines=['| Case | Lower | Compile | Lower s | Peak RSS MiB | Estimated STACK | Exception |',
           '|---|---|---|---:|---:|---|---|']
    for r in rows:
        secs=sum(x.get('lower_s',0) for x in r.get('programs',[]))
        comp=sum(x.get('compile_s',0) for x in r.get('programs',[]))
        rss=r.get('peak_rss_KiB')
        lines.append('| '+ ' | '.join((r['case'],r['lower'],str(round(comp,2))+' s' if r['compile']=='OK' else r['compile'],
            str(round(secs,2)),str(round(rss/1024,1)) if rss is not None else 'UNKNOWN',
            ','.join(str(x.get('stack','UNKNOWN')) for x in r.get('programs',[])) or 'UNKNOWN',
            r.get('exception','').replace('|','/')[:240]))+' |')
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')

for case in cases:
    quiet()
    ident=case['id'];out=a.out/ident;log=a.out/(ident+'.log')
    if ident in previous and ident not in a.rerun:
        r=previous[ident]
        if r['lower'] not in ('RUNNING','NOT_STARTED','TIMEOUT','NO_RECEIPT','RESOURCE_OR_PROCESS_FAIL'):
            r.setdefault('artifact',str(a.reuse_from/ident))
            rows.append(r);publish()
            continue
    if out.exists():
        # Resume only completed, already journaled rows; never overwrite arms.
        r=json.loads((out/'receipt.json').read_text())
        assert r['lower'] not in ('RUNNING','NOT_STARTED')
        rows.append(r);publish()
        continue
    # The different composite / immutable source is explicit in each prior scan.
    prior=subprocess.run(['rg','-n','-i',f'{ident}|t-lowersweep|E156',
        str(main/'FINDINGS.md'),str(main/'EXPERIMENTS.tsv'),str(main/'INBOX.md')],capture_output=True,text=True)
    (a.out/(ident+'.prior.txt')).write_text('\n'.join(prior.stdout.splitlines()[-16:])+'\n')
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',GPUWRF_JAX_CACHE='0',
        GPUWRF_FAST_DEFAULTS='1',GPUWRF_CENSUS='1',GPUWRF_CENSUS_SLIM='0',GPUWRF_AOT_SLIM_MODULE='0',
        GPUWRF_RRTMG_MP_RE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    disposable=a.out/(ident+'_scratch');disposable.mkdir()
    for key,name in (('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),
                     ('JAX_COMPILATION_CACHE_DIR','jax'),('GPUWRF_XLA_AUTOTUNE_CACHE_DIR','autotune')):
        d=disposable/name;d.mkdir();env[key]=str(d)
    child=['taskset','-c','8,9,24,25','nice','-n','19','timeout','--kill-after=10','900',
        'python',str(tool),'--source',str(a.source),'--case',str(prep/ident),
        '--out',str(out),'--target',str(a.target)]
    # Recheck QUIET after obtaining the global slot, not only before queuing.
    guarded='if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; ulimit -c 0; exec '+shlex.join(child)
    command=['bash',str(a.ram_wrapper),'--label','t-lowersweep','--max',a.ram_max,'--',
             'bash','-c',guarded]
    shell='if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; ulimit -c 0; exec '+shlex.join(command)
    print(utc(),'START',ident,case['status'],flush=True)
    started_epoch=time.time()
    with log.open('x') as f:run=subprocess.run(['bash','-c',shell],env=env,stdout=f,stderr=subprocess.STDOUT)
    if run.returncode==3 and not out.exists():
        quiet()
        with log.open('a') as f:run=subprocess.run(['bash','-c',shell],env=env,stdout=f,stderr=subprocess.STDOUT)
    r=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else dict(
        case=ident,lower='NO_RECEIPT',compile='NOT_ATTEMPTED',programs=[])
    r['returncode']=run.returncode;r['artifact']=str(out);r['cache_dir_deleted']=str(disposable)
    if out.exists():
        r['process_result']=classify(r,out,run.returncode,started_epoch,time.time())
    if ident in previous:
        r['prior_attempt']={k:previous[ident].get(k) for k in ('artifact','source_commit','lower','returncode')}
        if previous[ident].get('lower')=='OOM_AT_CAP':
            r['prior_attempt']['classification']='OOM_AT_CAP'
        elif ident=='c66' and previous[ident].get('lower') in ('NOT_STARTED','RUNNING','HOST_REBOOT_INTERRUPTION'):
            r['prior_attempt']['classification']='HOST_REBOOT_INTERRUPTION; NOT_A_REAL_TIMEOUT'
    if run.returncode and r['lower'] in ('RUNNING','NOT_STARTED'):
        r['lower']=r.get('process_result',{}).get('classification','RESOURCE_OR_PROCESS_FAIL')
    if case['status']!='ADMITTED':
        r['control_status']=case['status']
        if r['lower'] in ('INIT_FAIL','FAIL'):r['lower']='CONTROL_REFUSAL_UNVERIFIED'
    rows.append(r);publish()
    outcome=f"[M CPU] lower={r['lower']} compile={r['compile']} wall={r.get('wall_s')}s rss={r.get('peak_rss_KiB')}KiB rc={run.returncode}"
    append(main/'EXPERIMENTS.tsv','\t'.join((utc(),'t-lowersweep','Codex','LS_'+ident,
        'Native CUDA lower of distinct actual composite before GPU allocation',
        'immutable '+r.get('source_commit','UNKNOWN'), 'CLI history-segment signature / native Pallas',
        'prior GPU failures / untested composite',outcome,'KEEP preflight evidence',
        f'artifact={out},{log};cache={disposable} DELETED after arm;work=actualinputSHA manifest '+
        f'FAST1CENSUS1SLIM0RRTMG_MP_RE1 CPU8,9,24,25nice19 MemoryMax{a.ram_max}Swap0 timeout900;NO_GPU')))
    if case['status']=='ADMITTED' and (r['lower']!='OK' or r['compile']=='FAIL'):
        append(main/'MAILBOX.md',f"{utc()} t-lowersweep: FAIL {ident} {outcome}; {r.get('exception','process killed/no final receipt')[:350]}; {out}/receipt.json; product fix owner needed, no product edits.")
    quiet()
    # Only disposable, newly created paths under this lane; retain small receipts.
    assert disposable.parent==a.out and disposable.name==ident+'_scratch'
    shutil.rmtree(disposable)
    if (out/'scan_cache').exists():shutil.rmtree(out/'scan_cache')
    print(utc(),'DONE',ident,outcome,flush=True)
print(utc(),'SWEEP_COMPLETE',len(rows),flush=True)
