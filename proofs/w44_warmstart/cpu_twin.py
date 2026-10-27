"""Original CPU-WRF native May cycling twin; no JAX or ALISIOS payload opens.

Run prepare once, commit this launcher, then run inside a 12 GiB systemd scope.
Staging reuses the guarded, hashed native May input set. Every attempt is private.
"""
import argparse
from datetime import datetime, timezone, timedelta
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart')
RUN = ROOT / 'cpu_native_may04_00_3r/cpu'
META = RUN.parent
STAGE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/NATIVE_CASE_MANIFEST.json')
BUILD = Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar')
WENV = Path('<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build')
ORIGINAL_EXE_SHA = '1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b'
EXE_PATH = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/patched/wrf.exe')
EXE_SHA = '1c1269fb10b7aa135bf42417f1aa7f7a36f7137ea40cd82769316e8ac5bb4918'
GUARD_PROOFS = [Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/runs') / name / 'RESULT.json'
                for name in ('E9_A_nocyc_orig_h1', 'E10_A_nocyc_patched_h1')]
MB = Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md')
EXP = MB.with_name('EXPERIMENTS.tsv')
CPUS = [8, 9, 12]
ENV = dict(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OMPI_MCA_opal_cuda_support='0',
           OMPI_MCA_accelerator='null', OMPI_MCA_rcache='grdma', OMPI_MCA_pml='ob1',
           OMPI_MCA_btl='self,sm,tcp', OMPI_MCA_coll='^ucc,hcoll', UCX_TLS='self,sm,tcp',
           HWLOC_COMPONENTS='-cuda,-nvml')


def utc():
    return subprocess.check_output(['bash','<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh','--iso'],text=True).strip()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(name, data):
    (META / name).write_text(json.dumps(data, indent=2) + '\n')


def post(message):
    line = f'{utc()} w44-warmstart: {message}'
    print(line, flush=True)
    with MB.open('a') as f:
        f.write(line + '\n')


def experiment(result):
    with EXP.open('a')as f:
        f.write('\t'.join([utc(),'w44-warmstart','gpt-6.1-sol','CPU-TWIN02-GUARDED',
            'originalCPU-WRF same native cycled inputs;onlynproc2x2->3x1',result,
            'artifact='+str(META)+';cache=NONE;work=WN3May4_00cyclingTrue3h3nestsdt54/18/6;CPU8,9,12 physicalnice19 E219 MemoryMax12GiB timeout7200;noGPU'])+'\n')


def safe(p):
    p = p.resolve(strict=True)
    assert not str(p).startswith(('<DATA_ROOT>/server/work/', '<DATA_ROOT>/alisios/')), p
    return p


def prepare():
    import shutil
    assert not RUN.exists(), 'fresh attempt required'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'QUIET'
    assert shutil.disk_usage(ROOT).free >= 200 * 2**30
    receipt=json.loads(STAGE.read_text());source=Path(receipt['case'])
    for row in receipt['files']: assert sha(source/row['name'])==row['sha256'],row['name']
    original=(source/'namelist.input').read_text();text=original
    for key,value in [('nproc_x',3),('nproc_y',1)]:
        text,n=re.subn(r'(?m)^(\s*'+key+r'\s*=\s*)2,',lambda m:m[1]+str(value)+',',text)
        assert n==1,key
    # Direct reversion of precisely the two changed value spans proves every
    # other byte (cycling, clocks, restart_interval, physics) is identical.
    reversed_text=text
    for key,value in [('nproc_x',3),('nproc_y',1)]:
        reversed_text,n=re.subn(r'(?m)^(\s*'+key+r'\s*=\s*)'+str(value)+',',lambda m:m[1]+'2,',reversed_text)
        assert n==1,key
    assert reversed_text==original
    assert re.search(r'(?m)^\s*run_hours\s*=\s*3,',text)
    assert re.search(r'(?m)^\s*cycling\s*=\s*\.true\.',text)
    assert sha(BUILD/'run/wrf.exe')==ORIGINAL_EXE_SHA
    assert sha(EXE_PATH)==EXE_SHA
    proofs=[json.loads(p.read_text()) for p in GUARD_PROOFS]
    assert all(p['status']=='STARTUP_PASS' for p in proofs)
    assert len(proofs[0]['outputs'])==6 and proofs[0]['outputs']==proofs[1]['outputs']
    RUN.mkdir(parents=True)
    inputs={}
    for row in receipt['files']:
        name=row['name']
        if name=='namelist.input':continue
        target=RUN/name;shutil.copyfile(source/name,target);assert sha(target)==row['sha256']
        inputs[name]=dict(source=str(source/name),sha256=row['sha256'])
    tables={}
    for p in sorted((BUILD/'run').iterdir()):
        if p.name in ('namelist.input','wrf.exe','real.exe','ndown.exe','tc.exe') or p.name.startswith(('wrfinput','wrfbdy','wrflowinp')):continue
        if p.is_file():
            target=safe(p);(RUN/p.name).symlink_to(target);tables[p.name]=dict(path=str(target),sha256=sha(target))
    (RUN/'wrf.exe').symlink_to(safe(EXE_PATH))
    reused_tables={}
    for name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):
        cached=ROOT/'startup_A_3r/cpu'/name
        # Acceleration only: absent caches are recomputed by thompson_init (E160).
        if cached.is_file() and not (RUN/name).exists():
            (RUN/name).symlink_to(cached)
            reused_tables[name]=dict(path=str(cached),sha256=sha(cached))
    (RUN/'namelist.input').write_text(text)
    (META/'original_namelist.input').write_text(original)
    (META/'namelist.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),'GPU_same_inputset','CPU_3x1')))
    rank=RUN/'rank.sh'
    rank.write_text('#!/bin/bash\nset -eu\nCPUS=(8 9 12)\nexec taskset -c "${CPUS[$OMPI_COMM_WORLD_RANK]}" nice -n 19 ./wrf.exe\n')
    rank.chmod(0o755)
    write('PREPARE.json',dict(stage_receipt=str(STAGE),stage_receipt_sha256=sha(STAGE),
        original_namelist_sha256=sha(source/'namelist.input'),twin_namelist_sha256=sha(RUN/'namelist.input'),
        inputs=inputs,tables=tables,executable_sha256=EXE_SHA,original_decomposition=[2,2],twin_decomposition=[3,1],
        original_gen2_executable_sha256=ORIGINAL_EXE_SHA,executable_path=str(EXE_PATH),
        guarded_reference_family='gen2 WRF4.7.1 + package-allocated cycling diagnostic reset guards only',
        guard_inert_proofs=[dict(path=str(p),sha256=sha(p))for p in GUARD_PROOFS],
        normal_start_output_identity='6/6 original-vs-guarded wrfout BYTE (t0+1h,3nests); manager approved',
        reused_thompson_tables=reused_tables,
        namelist_diff_keys=['nproc_x','nproc_y'],all_other_namelist_bytes_exact=True,
        expected_wrfout=12,rank_cpus=CPUS,env=ENV,prepared_utc=utc(),cache='NONE',
        tooling_source='<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/run_twin.py'))
    print('PREPARED',RUN,flush=True)


def ranks():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / 'cwd').resolve() == RUN and (p / 'exe').resolve() == (RUN / 'wrf.exe').resolve():
                found.append(int(p.name))
        except (OSError, RuntimeError):
            pass
    return sorted(found)


def inspect_ranks():
    rows = []
    for pid in ranks():
        p = Path('/proc') / str(pid)
        maps = (p / 'maps').read_text()
        devices = []
        for fd in (p / 'fd').iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if '/dev/nvidia' in target:
                devices.append(target)
        env = dict(x.split('=', 1) for x in (p / 'environ').read_text().split('\0') if '=' in x)
        rank = int(env['OMPI_COMM_WORLD_RANK'])
        row = dict(pid=pid, rank=rank, affinity=sorted(os.sched_getaffinity(pid)),
                   libcuda_mappings=maps.count('libcuda'), nvidia_fds=devices,
                   env={k: env.get(k) for k in ENV}, cgroup=(p / 'cgroup').read_text())
        assert row['affinity'] == [CPUS[rank]], row
        assert row['libcuda_mappings'] == 0 and not devices, row
        assert row['env'] == ENV, row
        rows.append(row)
    assert len(rows) == 3, rows
    return dict(utc=utc(), ranks=rows)


def teardown(proc):
    # PRTE can put ranks into separate process groups; clean every group in our session.
    groups = set()
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if os.getsid(int(p.name)) == proc.pid:
                    groups.add(os.getpgid(int(p.name)))
            except ProcessLookupError:
                pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM and groups:
            time.sleep(5)
    proc.wait(timeout=30)


def finish(rc, started, t0):
    from netCDF4 import Dataset,chartostring
    import math
    rows=[];expected=set();read_errors=[]
    for domain,dt in ((1,54),(2,18),(3,6)):
        for hour in range(4):
            seconds=math.ceil(hour*3600/dt)*dt
            stamp=(datetime(2026,5,4)+timedelta(seconds=seconds)).strftime('%Y-%m-%d_%H:%M:%S')
            expected.add((f'd{domain:02}',stamp))
    actual=set()
    for p in sorted(RUN.glob('wrfout_d0*')):
        row=dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
        try:
            with Dataset(p)as ds:
                times=chartostring(ds['Times'][:]).tolist();assert len(times)==1
                fields={name:dict(dtype=str(v.dtype),dimensions=list(v.dimensions))for name,v in ds.variables.items()}
            actual.add((p.name.split('_')[1],times[0]));row.update(times=times,fields=fields)
        except OSError as exc:
            row['read_error']=str(exc);read_errors.append(p.name)
        rows.append(row)
    write('CPU_OUTPUTS.json',rows)
    restart_rows=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(RUN.glob('wrfrst_d0*'))]
    write('CPU_RESTART_OUTPUTS.json',restart_rows)
    success=rc==0 and not read_errors and 'SUCCESS COMPLETE WRF' in (RUN/'rsl.out.0000').read_text()and actual==expected
    prep=json.loads((META/'PREPARE.json').read_text())
    identity=sha(RUN/'wrf.exe')==EXE_SHA and sha(RUN/'namelist.input')==prep['twin_namelist_sha256']
    identity=identity and all(sha(RUN/n)==r['sha256']for n,r in prep['inputs'].items())
    identity=identity and all(sha(Path(r['path']))==r['sha256']for r in prep['tables'].values())
    identity=identity and all(sha(Path(r['path']))==r['sha256']for r in prep['reused_thompson_tables'].values())
    result=dict(rc=rc,status='DONE_CLEAN'if success and identity else'FAIL',started_utc=started,ended_utc=utc(),
        wall_s=time.monotonic()-t0,wrfout=len(rows),expected_wrfout=12,complete_times=actual==expected,
        input_identity=identity,output_manifest_sha256=sha(META/'CPU_OUTPUTS.json'),rank_cpus=CPUS,
        executable_sha256=EXE_SHA,cache='NONE',namelist_diff_keys=prep['namelist_diff_keys'])
    result.update(original_gen2_executable_sha256=ORIGINAL_EXE_SHA,
                  restart_files=len(restart_rows),restart_manifest_sha256=sha(META/'CPU_RESTART_OUTPUTS.json'))
    mpi=json.loads((META/'MPI_EXIT.json').read_text())
    result.update(model_wall_s=mpi['wall_s'],quiet_pause_s=mpi['quiet_pause_s'],
                  unpaused_model_wall_s=mpi['wall_s']-mpi['quiet_pause_s'],
                  timing_scope='model wall includes QUIET pauses; supervisor wall also includes hashing/queue')
    result['output_read_errors']=read_errors
    write('cpu_run.json',result)
    post(f"END rc={rc} status={result['status']} wrfout={len(rows)}/12 wall={result['wall_s']:.1f}s run={RUN}; RAW KEPT")
    experiment('[M]'+json.dumps(result,separators=(',',':')))
    return 0 if result['status']=='DONE_CLEAN'else 1


def run():
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'QUIET'
    assert not list(RUN.glob('wrfout_*')), 'refusing restart/overwrite'
    assert sha(RUN / 'wrf.exe') == EXE_SHA
    assert sorted(os.sched_getaffinity(0)) == CPUS
    cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    memory_max = (cgroup / 'memory.max').read_text().strip()
    assert memory_max != 'max' and int(memory_max) >= 12 * 2**30, memory_max
    started, t0 = utc(), time.monotonic()
    env = os.environ.copy()
    env.update(ENV)
    env['LD_LIBRARY_PATH'] = str(WENV / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(WENV / 'bin') + ':' + env['PATH']
    cmd = ['timeout', '--signal=TERM', '--kill-after=30', '7200', str(WENV / 'bin/mpirun'),
           '--use-hwthread-cpus', '--bind-to', 'none', '-np', '3', './rank.sh']
    write('LAUNCH.json', dict(started_utc=started, command=cmd, env=ENV, memory_max=memory_max,
                             cgroup=str(cgroup), launcher_sha256=sha(__file__), budget_s=7200))
    proc = None
    quiet_pauses=[]
    def quiet_yield():
        if not Path('/tmp/wrf_gpu2_quiet').exists():
            return
        paused=time.monotonic();pids=ranks()
        for pid in pids:
            try:os.kill(pid,signal.SIGSTOP)
            except ProcessLookupError:pass
        while Path('/tmp/wrf_gpu2_quiet').exists() and proc.poll() is None:
            time.sleep(.5)
        for pid in pids:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        quiet_pauses.append(dict(seconds=time.monotonic()-paused,pids=pids))
        write('QUIET_PAUSES.json',quiet_pauses)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (META / 'cpu.log').open('xb') as log:
            proc = subprocess.Popen(cmd, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write('PROCESS.json', dict(supervisor_pid=os.getpid(), pid=proc.pid, sid=proc.pid, started_utc=started))
            deadline = time.monotonic() + 120
            while len(ranks()) != 3:
                quiet_yield()
                assert proc.poll() is None, 'MPI exited before three ranks'
                assert time.monotonic() < deadline, 'rank startup timed out'
                time.sleep(1)
            for _ in range(20):
                quiet_yield();time.sleep(.5)
            write('CPU_ONLY_RANKS.json', inspect_ranks())
            post(f'START ranks=3 cores=8,9,12 libcuda=0x3 nvidia_fds=0x3 decomposition=3x1(sameinputset2x2) '
                 f'budget=7200s memcg=12GiB nice19 run={RUN}; native inputs byte-exact; namelist diff ONLY nproc_x/nproc_y; '
                 f'guardedexe={EXE_SHA} originalgen2={ORIGINAL_EXE_SHA}')
            experiment('[M] RUNNING CPU_ONLY_RANKS.json libcuda0x3; no completion claim')
            next_check = time.monotonic() + 60
            checked_history = False
            while proc.poll() is None:
                quiet_yield();time.sleep(.5)
                if not checked_history and len(list(RUN.glob('wrfout_d0*'))) >= 3:
                    write('CPU_ONLY_AFTER_HISTORY.json', inspect_ranks())
                    checked_history = True
                if time.monotonic() >= next_check:
                    write('PROGRESS.json', dict(utc=utc(), wall_s=time.monotonic()-t0,
                          wrfout=len(list(RUN.glob('wrfout_d0*'))), memory_current=(cgroup / 'memory.current').read_text().strip(),
                          memory_peak=(cgroup / 'memory.peak').read_text().strip(),
                          rsl_tail=(RUN / 'rsl.error.0000').read_text()[-1500:]))
                    assert os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= 200 * 2**30, 'DATA floor'
                    next_check = time.monotonic() + 60
            rc = proc.returncode
            write('MPI_EXIT.json',dict(rc=rc,utc=utc(),wall_s=time.monotonic()-t0,
                  quiet_pause_s=sum(p['seconds']for p in quiet_pauses)))
    finally:
        if proc is not None:
            teardown(proc)
    while Path('/tmp/wrf_gpu2_quiet').exists():time.sleep(.5)
    return finish(rc, started, t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--attempt',default='cpu_native_may04_00_3r')
    parser.add_argument('--stage',default=str(STAGE),help='hashed native case manifest (default: original May case)')
    args = parser.parse_args()
    STAGE=Path(args.stage)
    assert re.fullmatch(r'cpu_native_may04_00_[a-zA-Z0-9_]+',args.attempt)
    RUN=ROOT/args.attempt/'cpu';META=RUN.parent
    if args.action == 'prepare':
        prepare()
    else:
        raise SystemExit(run())
