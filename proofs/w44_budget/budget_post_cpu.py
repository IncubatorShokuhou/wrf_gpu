"""After GPU unlock: actual E41, pristine same-input oracles, cost summaries.

Accounting gaps stay UNKNOWN. Native E41, physical BYTE and term failures stay
failures. Everything here runs on CPU in the outer supervisor's bounded group.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--run-proof',type=Path,required=True)
    a=p.parse_args();assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    here=Path(__file__).resolve().parent;run=a.out.parent
    assert run.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget'))
    manifest=json.loads((here/'manifest.json').read_text());model=Path(manifest['model'])
    env=dict(os.environ,BUDGET_MODEL_ROOT=str(model),PYTHONPATH=str(model/'src'))
    a.out.mkdir(parents=True,exist_ok=True)
    result=dict(e41_rc=None,oracles={},accounting={},scope=__doc__)
    # QA A1/A3: every inner child gets min(nominal, time left before the outer post cap); a timeout is
    # recorded as 'TIMEOUT' (never an uncaught TimeoutExpired), so later oracles/accounting still run.
    deadline=time.monotonic()+float(os.environ.get('BUDGET_POST_CAP_S',manifest.get('cpu_cap_s',720)))-20
    def bounded(command,nominal,**kwargs):
        limit=max(1.,min(float(nominal),deadline-time.monotonic()))
        try:return subprocess.run(command,timeout=limit,**kwargs).returncode
        except subprocess.TimeoutExpired:return 'TIMEOUT'
    status=run/'post_cpu.json'
    def save():status.write_text(json.dumps(result,indent=2)+'\n')
    command=[sys.executable,str(here/'budget_cubin_gate.py'),str(a.cache),str(a.out),
             '--run-proof',str(a.run_proof)]
    result['e41_rc']=bounded(command,600,env=env);save()
    probe=run/'probe';gpu=probe/'result.json'
    if gpu.exists():
        record=json.loads(gpu.read_text())
        for name,component in record.get('components',{}).items():
            if component.get('verdict')!='TWIN_COMPONENT_2STEP_BYTE':continue
            target=run/'oracles'/name;target.parent.mkdir(exist_ok=True)
            if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit(3)
            command=[sys.executable,str(here/'twin_oracle.py'),'--source',str(model),
                '--probe',str(probe/'components'/name),'--floors',manifest['floor_protocol'],'--out',str(target)]
            with open(run/(name+'_oracle.log'),'w') as log:
                rc=bounded(command,150,env=env,stdout=log,stderr=subprocess.STDOUT)
            result['oracles'][name]=dict(rc=rc,result=str(target/'result.json'));save()
            if (target/'result.json').exists():
                # Keep actual/reference NPZ, caller, provenance and all column
                # bounds. These exact completed-arm binary streams are disposable.
                for leaf in ('step1','step2','step1_conditioning','step2_conditioning'):
                    path=target/leaf
                    if path.exists():assert path.is_dir() and not path.is_symlink();shutil.rmtree(path)
                executable=target/'original/oracle'
                if executable.exists():executable.unlink()
        phases=record.get('phases',{})
        for name in ('replay_off','replay_on','replay_edmf0'):
            segments=phases.get(name,{}).get('segments',[])
            # QA A2: segment 1 of the EDMF-on replays is the nsys node-traced segment (E185) -> separate untraced list.
            traced={1} if name in ('replay_off','replay_on') else set()
            result['accounting'][name]=dict(warm_segment_s_per_fc_h=[s['wall_s']*3600/s['simulated_s'] for s in segments[1:]],
                untraced_warm_segment_s_per_fc_h=[s['wall_s']*3600/s['simulated_s'] for i,s in enumerate(segments) if i>=1 and i not in traced],
                traced_segment_index=sorted(traced),
                rss_max_kib=phases.get(name,{}).get('rss_max_kib'),
                device_memory_stats=phases.get(name,{}).get('device_memory_stats'),
                memory_scope='same GPU process retains shared compiled programs; peak is cumulative, not isolated ON-minus-OFF')
        save()
    # Each cudaProfiler capture becomes a separate Nsight report. Missing or
    # unsupported accounting is reported, never turned into a numerical pass.
    reports=sorted((run/'work').glob('budget_trace*.nsys-rep'),key=lambda p:p.stat().st_mtime_ns) if (run/'work').exists() else []
    result['node_accounting']=[]
    for index,report in enumerate(reports):
        if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit(3)
        database=report.with_suffix('.sqlite')
        try:
            rc=bounded(['nsys','export','--type','sqlite','--force-overwrite','true',
                '--output',str(database),str(report)],150,env=env,stdout=subprocess.DEVNULL)
            assert rc==0 and database.exists()
            connection=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
            tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts={name:int(connection.execute('SELECT count(*) FROM '+table).fetchone()[0]) if table in tables else 0
                    for name,table in [('kernels','CUPTI_ACTIVITY_KIND_KERNEL'),('memcpy','CUPTI_ACTIVITY_KIND_MEMCPY'),('memset','CUPTI_ACTIVITY_KIND_MEMSET')]}
            connection.close();assert counts['kernels']>0
            phase=('replay_off','replay_on')[index] if len(reports)==2 and index<2 else 'UNKNOWN'
            result['node_accounting'].append(dict(report=str(report),phase=phase,counts=counts,total_device_ops=sum(counts.values())))
        except (OSError,AssertionError,sqlite3.Error,subprocess.SubprocessError) as error:
            result['node_accounting'].append(dict(report=str(report),verdict='UNKNOWN',reason=str(error)))
        save()
    if not reports:result['node_accounting']=[dict(verdict='UNKNOWN',reason='no Nsight capture report')]
    expected=len(manifest.get('component_cases',[0]*4))
    result['verdict']='CPU_GATES_PASS' if result['e41_rc']==0 and len(result['oracles'])==expected and all(
        v['rc']==0 for v in result['oracles'].values()) else 'CPU_GATES_FAIL_OR_INCOMPLETE'
    save();print(json.dumps(result),flush=True)
    return 0 if result['verdict']=='CPU_GATES_PASS' else 2


if __name__=='__main__':raise SystemExit(main())
