"""One sealed builder for the full forecast, CUDA miniature and CPU rehearsal."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

LANE=Path('<USER_HOME>/wrf_gpu2_lanes/w44-topo')
SPEC=LANE/'FORECAST_REQUEST_v11.json'
GROUP=LANE/'night_tools_111bc8fb6b10/night_driver.py'
GROUP_SHA='3da459287de21489509d2e500dd7fe5127375a51ac504329fa90912887914923'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def load_group():
    assert sha(GROUP)==GROUP_SHA
    spec=importlib.util.spec_from_file_location('topo_owned_process_group',GROUP)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def preflight():
    spec=json.loads(SPEC.read_text());tree=Path(spec['tree']);tools=Path(spec['tools'])
    assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()==spec['pin']
    assert not subprocess.check_output(['git','-C',str(tree),'status','--porcelain','--untracked-files=no'])
    assert (tree/'data/wrf_pristine').is_dir()
    for name,digest in spec['tool_hashes'].items():assert sha(tools/name)==digest,name
    for name,digest in spec['dependency_hashes'].items():assert sha(name)==digest,name
    for profile in spec['profiles'].values():
        for name,digest in profile['input_hashes'].items():assert sha(Path(profile['input_dir'])/name)==digest,name
        assert_run_length(profile)
    return spec,tree,tools


def assert_run_length(profile):
    """E187: refuse before any GPU work unless hours fit whole root steps."""
    import re
    text=(Path(profile['input_dir'])/'namelist.input').read_text()
    dt=float(re.search(r'^\s*time_step\s*=\s*([0-9.]+)',text,re.M|re.I).group(1))
    steps=float(profile['hours'])*3600.0/dt
    assert int(profile['hours'])%3==0 and abs(steps-round(steps))<1e-9,(profile['hours'],dt,steps)


def scoped(command,memory):
    """E233: each model child in its OWN user scope (no swap); never through with_ram_slot."""
    return ['systemd-run','--user','--scope','--quiet','-p',f'MemoryMax={memory}','-p','MemorySwapMax=0','--',*command]


PROBE=('import json,pathlib;r=pathlib.Path("/proc/self/cgroup").read_text().strip().split("::",1)[-1];'
    'd=pathlib.Path("/sys/fs/cgroup"+r);print(json.dumps(dict(cgroup=r,memory_max=(d/"memory.max").read_text().strip(),'
    'swap_max=(d/"memory.swap.max").read_text().strip())))')


def scope_probe(memory):
    """Refuse before any GPU work unless the user manager grants the scope (E231: needs DBUS env)."""
    out=subprocess.run(scoped([sys.executable,'-I','-c',PROBE],memory),capture_output=True,text=True,timeout=30)
    assert out.returncode==0,('user scope refused',out.returncode,out.stderr[-500:])
    record=json.loads(out.stdout.strip().splitlines()[-1])
    assert record['memory_max']==str(int(memory[:-1])*1024**3) and record['swap_max']=='0',record
    return record


def e41_record(path):
    """E230: the E41 verdict is <out>/wave_gate.json, never the tool's rc. INCOMPLETE = compact AOT
    programs whose kInput HLO is unknown (E201) with STACK and callee/kInput checks clean: acceptable."""
    path=Path(path)/'wave_gate.json'
    if not path.is_file():return dict(path=str(path),e41_gate='MISSING',acceptable=False)
    gate=json.loads(path.read_text());verdict,hard,flags=gate['e41_gate'],gate['hard_stack_gate'],bool(gate['conservative_flags'])
    assert verdict in ('PASS','INCOMPLETE','FAIL') and hard in ('PASS','FAIL') and (verdict=='FAIL')==(hard=='FAIL' or flags),path
    entries=gate['entries'];status=lambda r:str(r.get('kinput_transpose_status',''))
    return dict(path=str(path),sha256=sha(path),e41_gate=verdict,hard_stack_gate=hard,conservative_flags=flags,
        n_entries=gate['n_entries'],aot_programs=gate['coverage']['aot_programs'],
        unknown_hlo=sorted({status(r) for r in entries if status(r).startswith('UNKNOWN')}),
        max_stack_B=max(e['max_stack_B'] for r in entries for e in r['elfs']),acceptable=verdict in ('PASS','INCOMPLETE'))


def build_work(spec,profile,tree,tools,output,caps,memory):
    """Identical commands and model flags; profile changes only grid/length/caps."""
    common={k:v for k,v in os.environ.items() if not k.startswith(('GPUWRF_','JAX_','XLA_','TRITON_','CUDA_','_GPUWRF_','TOPO_')) and k!='PYTHONPATH'}
    common.update(JAX_PLATFORMS='cuda',GPUWRF_FAST_DEFAULTS='1',GPUWRF_JAX_CACHE='1',GPUWRF_AOT_SLIM_MODULE='0',
        XLA_PYTHON_CLIENT_ALLOCATOR='cuda_async',XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_CLIENT_MEM_FRACTION='.30',
        OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    bundle=Path(spec['old_bundle']);base=Path(profile['input_dir'])
    for folder in (('new_cache','native/cache') if 'new' in profile['phases'] else ('native/cache',)):
        for name in ('jax','autotune','triton','cuda','tmp'):(output/folder/name).mkdir(parents=True)
    (output/'old_tmp').mkdir()
    def cache_env(root):
        return dict(GPUWRF_JAX_CACHE_DIR=str(root/'jax'),JAX_COMPILATION_CACHE_DIR=str(root/'jax'),
            GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(root/'autotune'),TRITON_CACHE_DIR=str(root/'triton'),CUDA_CACHE_PATH=str(root/'cuda'),TMPDIR=str(root/'tmp'))
    old_env={**common,**cache_env(bundle),'TMPDIR':str(output/'old_tmp'),'BENCH_SRC_REPO':str(bundle/'src'),'PYTHONPATH':str(bundle/'src/src')}
    new_env={**common,**cache_env(output/'new_cache'),'BENCH_SRC_REPO':str(tree),'PYTHONPATH':str(tree/'src')}
    native_env={**common,**cache_env(output/'native/cache'),'PYTHONPATH':str(tree/'src'),'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
        'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS':'0','JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES':'0'}
    def dispatch(seconds):
        return [sys.executable,str(tools/'gpu_dispatch.py'),'--lock',str(tree/'scripts/with_gpu_lock.sh'),'--seconds',str(seconds),'--',
            sys.executable,str(tools/'device_layer.py'),'--entry']
    native=scoped(dispatch(caps['native'])+[str(tools/'gpu_probe.py'),'--manifest',profile['native_manifest'],'--out',str(output/'native/operator')],memory)
    work=dict(native=(native,native_env))
    if 'new' in profile['phases']:
        new=scoped(dispatch(caps['new'])+[str(tools/'forecast_harness.py'),'--input-dir',str(base),'--out',str(output/'new'),
            '--hours',str(profile['hours']),'--tag','w44-topo-new'],memory)
        work['new']=(new,new_env)
    # Only the CUDA child owns the GPU lock. No E41 on the sealed donor: the bundle's own
    # dependency-hashed E41 covers its programs and the warm run must compile none (driver checks).
    old_body=('source "$1/bundle.env" || exit; '
        'python "$2/gpu_dispatch.py" --lock "$5/scripts/with_gpu_lock.sh" --seconds "$7" -- '
        'python "$2/device_layer.py" --entry "$2/forecast_harness.py" --input-dir "$3" --out "$4/old" --hours "$6" --tag w44-topo-old')
    old=scoped([sys.executable,'-I',str(tree/'scripts/release/bundle_consumer.py'),'--bundle',str(bundle),'--layer',str(output/'old_layer'),
        '--','bash','-c',old_body,'topo-old',str(bundle),str(tools),str(base),str(output),str(tree),str(profile['hours']),str(caps['old'])],memory)
    work['old']=(old,old_env)
    return work


def clean(folder,parent):
    if not folder.exists():return
    assert not folder.is_symlink() and folder.resolve().is_relative_to(parent.resolve())
    for p in sorted(folder.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_file() or p.is_symlink():p.unlink()
        elif p.is_dir():p.rmdir()
    folder.rmdir()


def main():
    group=load_group()
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(number,group.stop_signal)
    args=set(sys.argv[1:]);assert args<={'--cuda-mini','--launch-smoke','--dry-run','--check-window'}
    mini='--cuda-mini' in args;smoke='--launch-smoke' in args;cap=7200 if smoke else 290 if mini else 2400
    if Path('/tmp/wrf_gpu2_quiet').exists():print('QUIET');return 3
    if '--check-window' in args:
        now=group.utc_now();return 0 if group.epoch(os.environ['GA_START_UTC'])<=now and group.epoch(os.environ['GA_END_UTC'])-now>=cap else 5
    spec,tree,tools=preflight();profile=spec['profiles']['mini' if mini else 'production']
    memory=spec['scope_memory']['smoke' if smoke else 'gpu']
    if '--dry-run' in args:
        # Exercise the actual profile caps even for a smoke launcher: smoke_caps
        # contain unused phases and concealed the v9 mini's caps['new'] crash.
        with tempfile.TemporaryDirectory(prefix='dry_work_v11_',dir=LANE) as temporary:
            work=build_work(spec,profile,tree,tools,Path(temporary),profile['caps'],memory)
            assert set(work)==set(profile['phases'])-{'checker'},sorted(work)
            assert all(env['JAX_PLATFORMS']=='cuda' for _,env in work.values())
            print('DRY BUILDER PASS: actual GPU caps, phases',sorted(work))
        print('DRY PASS: content seals, same builder/env, user scope',scope_probe(memory),'for every model child, own12GiB CPU-post cap');return 0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
    output=LANE/('cuda_mini_cpu_v11' if smoke and mini else 'launch_cpu_v11' if smoke else 'cuda_mini_v11' if mini else 'forecast_receipts_v11')/stamp
    output.mkdir(parents=True)
    result=dict(source=spec['pin'],profile='mini' if mini else 'production',CPU_rehearsal=smoke,cap_s=cap,rc=1,passed=False,
        request_sha256=sha(SPEC),input_hashes=profile['input_hashes'],
        RAM=f'model children: own systemd user scope MemoryMax={memory} MemorySwapMax=0 (E233); CPU post RLIMIT_AS12GiB',
        rc_meaning={'0':'all phases done, every E41 verdict acceptable (PASS, or INCOMPLETE with STACK PASS)','2':'all phases done, an E41 verdict FAIL (recorded)',
            '6':'all phases done, an E41 verdict MISSING (tool error/timeout; cache retained for a post-window CPU scan)'},
        scope='CPU model/tool admission only; no CUDA qualification' if smoke else 'CUDA miniature qualification; not full-grid fidelity' if mini else '0227 v0.4.3/candidate effect includes non-topo changes (TP59); not topo attribution or CPU-WRF fidelity')
    def save():(output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    def finish(rc):result['rc']=int(rc);return int(rc)
    try:
        now=group.utc_now();begin=group.epoch(os.environ['GA_START_UTC']);end=group.epoch(os.environ['GA_END_UTC'])
        if now<begin or end-now<cap:result['refusal']='GA window too short/outside';return finish(5)
        if shutil.disk_usage('<DATA_ROOT>').free<365*1024**3:result['refusal']='DISK below365GiB';return finish(3)
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1'
        try:result['scope_probe']=scope_probe(memory)
        except Exception as exc:result['refusal']=f'no user systemd scope: {exc!r}';return finish(4)
        caps=spec['smoke_caps'] if smoke else profile['caps']
        stop=min(end,now+cap-5);work=build_work(spec,profile,tree,tools,output,caps,memory);bundle=Path(spec['old_bundle'])
        def run(name,command,env,seconds):
            if smoke:env={**env,'TOPO_LAUNCH_SMOKE':'1'}
            value=group.run_group(command,env,output/(name+'.log'),min(seconds,max(1,stop-group.utc_now())),stop)
            result[name]=value;save();return value['rc']
        def model(name):
            command,env=work[name]
            assert env['JAX_PLATFORMS']=='cuda' and not env.get('JAX_PLATFORM_NAME')
            # old: the device phase gets caps['old']; the bundle consumer's view + P checks run after it.
            return run(name,command,env,caps[name]+(caps['old_post'] if name=='old' else 0)+5)
        def gate(name,cache,seconds,proof=None,pattern=None,substitution=None):
            """Real wave_cubin_gate; its rc never aborts a measurement (E230): the verdict JSON is the record."""
            command=[sys.executable,str(tools/'cpu_post.py'),'e41','--tree',str(tree),'--cache',str(cache),'--out',str(output/f'e41_{name}.json')]
            if proof:command+=['--proof',str(proof)]
            if pattern:command+=['--pattern',pattern]
            env={**work['native'][1],'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0','PYTHONPATH':str(tree/'src')}
            tool_rc=run('e41_'+name,command,env,seconds)
            record=dict(e41_record(output/f'e41_{name}.json'),tool_rc=tool_rc,cache=str(cache),pattern=pattern or 'default',
                        proof=str(proof) if proof else None,cache_substitution=substitution)
            result.setdefault('e41',{})[name]=record;save();return record
        if rc:=model('native'):return finish(rc)
        if smoke:
            # CPU caches hold no CUDA ELF: scan retained real CUDA executables instead (JIT-only, as the native cache).
            result['native_cache_pattern_hits']=len(list((output/'native/cache/jax').glob('jit_one*-cache')))
            gate('native',Path(spec['cpu_gate_fixture']),caps['e41'],pattern='jit__lambda*',substitution='retained CUDA JIT fixture (CPU run has no CUDA ELF)')
        else:gate('native',output/'native/cache/jax',caps['e41'],pattern='jit_one*')
        if rc:=model('old'):return finish(rc)
        official=[json.loads(p.read_text()) for p in (output/'old_layer/runs').glob('*.json') if not p.name.endswith('.view.json')]
        official=[r for r in official if 'bundle_check_after' in r]
        assert len(official)==1 and official[0]['bundle_check_after'].startswith('BUNDLE_OK'),official
        result['old_bundle_receipt']=official[0];written=official[0]['written_by_this_run']
        compiled_new=len(written['compiled'])+len(written['changed_sealed'])
        # The donor's E41 is the sealed bundle receipt (sha in dependency_hashes); a warm GPU run that compiled
        # anything would run unscanned programs -> not acceptable, layer kept for a scan. CPU programs are not CUDA.
        sealed=e41_record(bundle/'e41')
        result.setdefault('e41',{})['old']=dict(sealed,covered_by='sealed v0.4.3 bundle E41 (dependency-hashed); no rescan of the donor',
            compiled_new=compiled_new,acceptable=sealed['acceptable'] and (smoke or compiled_new==0),
            note='CPU rehearsal: compiled_new counts CPU programs, not CUDA executables' if smoke else None)
        save()
        result['phases']=phases=profile['phases'];checker_rc=0
        if 'new' in phases:
            if rc:=model('new'):return finish(rc)
            command=[sys.executable,str(tools/'cpu_post.py'),'checker','--tree',str(tree),'--tools',str(tools),'--run',str(output),
                '--hours',str(profile['hours']),'--report-start',str(profile['report_start']),'--out',str(output/'EFFECT.json')]
            env={**work['new'][1],'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0','PYTHONPATH':str(tree/'src')}
            checker_rc=run('checker',command,env,caps['checker'])
            # After the checker, so a slow E41 can never cost the effect; MISSING keeps the cache (E177).
            seconds=min(caps['e41_new'],max(1,stop-group.utc_now()))
            if smoke:gate('new',bundle/'jax',seconds,substitution='real AOT-bearing v0.4.3 bundle cache, read-only (CPU run has no CUDA ELF)')
            else:gate('new',output/'new_cache/jax',seconds,proof=output/'new/cli_stdout.json')
        elif smoke:
            # Mini phase set (native+old): validate the real-init old smoke directly.
            record=json.loads((output/'old/LAUNCH_SMOKE_MODEL.json').read_text())
            assert record['status']=='CPU_LAUNCH_SMOKE_PASS' and record['model_hook']=='real (no substitution)',record
        else:
            assert json.loads((output/'old/cli_stdout.json').read_text()).get('all_domains_finite'),'old mini forecast not finite'
        e41=result['e41']
        if smoke:
            # Tool admission: the verdict path must have been exercised exactly as on the GPU.
            assert e41['native']['e41_gate']=='PASS',e41['native']
            assert e41['old']['e41_gate']=='INCOMPLETE' and e41['old']['hard_stack_gate']=='PASS' and e41['old']['acceptable'],e41['old']
            if 'new' in e41:
                assert e41['new']['e41_gate']=='INCOMPLETE' and e41['new']['acceptable'] and e41['new']['aot_programs'] \
                    and e41['new']['unknown_hlo']==['UNKNOWN_AOT_METADATA'] and e41['new']['tool_rc']==2,e41['new']
        if checker_rc:return finish(checker_rc)
        missing=sorted(n for n,v in e41.items() if v['e41_gate']=='MISSING');failed=sorted(n for n,v in e41.items() if not v['acceptable'] and n not in missing)
        result['e41_summary']=dict(failed=failed,missing=missing,acceptable=not failed and not missing)
        rc=2 if failed else 6 if missing else 0
        result['passed']=not smoke and rc==0;result['CPU_all_phases']='PASS' if smoke else None
        result['gate']=('CPU_ALL_PHASES_PASS' if smoke else 'CUDA_MINI_PASS' if mini else 'FORECAST_EFFECT_PASS')+('' if rc==0 else '_E41_FAIL' if rc==2 else '_E41_DEFERRED')
        if not Path('/tmp/wrf_gpu2_quiet').exists():
            keep={'native/cache':not e41['native']['acceptable'],'new_cache':'new' in e41 and not e41['new']['acceptable'],
                  'old_layer/upper':not e41['old']['acceptable']}
            for folder in ('native/cache','new_cache','old_tmp','old_layer/upper','old_layer/work','old/wrfout','old/scratch','new/wrfout','new/scratch'):
                if not keep.get(folder):clean(output/folder,output)
            result['retained_for_e41']=sorted(k for k,v in keep.items() if v)
            result['disposable_caches_outputs_cleaned']=True
        return finish(rc)
    except BaseException as exc:result['error']=repr(exc);raise
    finally:
        result['ended_utc']=datetime.now(timezone.utc).isoformat();save();print(str(output/'RESULT.json'),flush=True)


if __name__=='__main__':raise SystemExit(main())
