"""Pipeline CPU executable gates and full D6 while subsequent GPU arms run."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp28_followup')
MAIN = Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
plan = json.loads((ROOT / 'RUN_PLAN.json').read_text())
SOURCE = Path(plan.get('source', plan['configs'][0]['source']))
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.sched_getaffinity(0) == {9}
GUARD = 'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"'

def utc():
    return datetime.now(timezone.utc).isoformat()

def append(file, message):
    with (MAIN / file).open('a') as f:
        f.write(message + '\n')

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def prior(case, suffix):
    with (case / f'{suffix}_prior.txt').open('w') as f:
        subprocess.run(['rg', '-i', case.name + '|E41|E166|E177|D6',
            *[str(MAIN / n) for n in ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]], stdout=f)

def guarded(cmd, logfile, seconds):
    while True:
        while Path('/tmp/wrf_gpu2_quiet').exists():
            time.sleep(5)
        with logfile.open('w') as log:
            r = subprocess.run(['<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh', '--label', 'v-run', '--max', '12G', '--timeout', '1800', '--', 'bash', '-c', GUARD, 'v-run-cpu', 'timeout', '--signal=TERM',
                '--kill-after=15', str(seconds), *cmd], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 3:
            return r.returncode

def cleanup(case, runtime, gate):
    assert gate['coverage']['status'] == 'PASS'
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(5)
    mailbox = (MAIN / 'MAILBOX.md').read_text().splitlines()[-150:]
    holds = [x for x in mailbox if re.search(r'\bRETAIN\b', x, re.I) and
             ('v-run' in x or str(runtime / 'cache') in x)]
    if holds:
        write(case / 'cleanup.json', {'status': 'RETAIN_REQUEST', 'requests': holds})
        return
    pins = case / 'e41/pins'
    pins.mkdir(exist_ok=True)
    rows = []
    for i, e in enumerate(gate['entries']):
        original = Path(e['path'])
        assert original.resolve().is_relative_to(runtime / 'cache')
        dest = pins / f'{i}_{original.name}'
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        shutil.copyfile(original, dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
        rows.append({'original_cache_path': str(original), 'pin': str(dest),
                     'sha256': digest, 'bytes': dest.stat().st_size})
    write(case / 'e41/PIN_MAP.json', {'pins': rows, 'e41_gate': gate['e41_gate'],
        'original_cache_deleted': True, 'consumer': 'Full E41 scan complete; exact critic PINs retained'})
    # Only bounded, declared lane-owned disposable directories; never reference paths.
    for name in ('cache', 'tmp'):
        path = runtime / name
        assert runtime in (case, Path('<DATA_ROOT>/wrf_gpu2/scratch/v-run/round3/runtime/mp28_followup'))
        assert path.resolve() == path
        if path.exists():
            shutil.rmtree(path)
    write(case / 'cleanup.json', {'status': 'DONE', 'utc': utc(), 'cache_deleted': True})

previous = json.loads((ROOT / 'STATUS.json').read_text()) if (ROOT / 'STATUS.json').exists() else {}
done = previous.get('done', {})
pending = [c['config'] for c in plan['configs'] if c['config'] not in done]
while pending:
    for name in list(pending):
        case = ROOT / name
        config = next(c for c in plan['configs'] if c['config'] == name)
        runtime = Path(config.get('runtime', str(case)))
        if not (case / 'GPU_DONE.json').exists():
            continue
        process = json.loads((case / 'GPU_DONE.json').read_text())
        gpu_file = case / 'gpu_run.json'
        if not gpu_file.exists():
            done[name] = {'verdict': 'GPU_FAILED', 'rc': process['rc'], 'no_process_receipt': True}
            pending.remove(name)
            append('MAILBOX.md', f'{utc()} v-run: {name} GPU_FAILED rc{process["rc"]}; no full D6; {case}/gpu_cuda.log')
            continue
        gpu = json.loads(gpu_file.read_text())
        append('EXPERIMENTS.tsv', f'{utc()}\tv-run\tGPT Sol\tVR-FU28-GPU-{name}\tCorrected current-main GPU24h matches original WRF\tno model edits; actual frozen source {gpu["source_rev"]}\t25frames/finite/D6/E41\tround2 source8afb\t[M GPU]rc{gpu["rc"]},wall{gpu["wall_s"]:.6f}s,sampledVRAM{gpu["peak_process_vram_mib_sampled"]}MiB\tKEEP measured receipt\tartifact={gpu_file};cache={runtime}/cache;work=Swiss42x42x44 dt18 hours24 output60min FASTdefaults/CENSUS1/SLIM0/RRTMG_MP_RE{gpu["measurement_flags"]["GPUWRF_RRTMG_MP_RE"]},officiallock v-run+matched59599+GPUassert timeout1170 arm1200 RAM12GSwap0,concurrent cold sharedCPU/IO NOT S1')
        cache = runtime / 'cache/jax'
        gate = None
        if cache.exists():
            prior(case, 'E41')
            cmd = ['python', str(SOURCE / 'scripts/wave_cubin_gate.py'), str(cache), str(case / 'e41')]
            proof = case / 'gpu/proofs/nested_pipeline_run.json'
            if proof.exists():
                cmd += ['--run-proof', str(proof)]
            rc = guarded(cmd, case / 'e41.log', 900)
            if (case / 'e41/wave_gate.json').exists():
                gate = json.loads((case / 'e41/wave_gate.json').read_text())
                cleanup(case, runtime, gate)
            append('EXPERIMENTS.tsv', f'{utc()}\tv-run\tGPT Sol\tVR-FU28-E41-{name}\tActual new cold executable meets original E41\tfull scanner before cache cleanup\tSTACK256B/nonleaf1000SASS/kInput500HLO\tround2 source8afb\t[M CPU]rc{rc},E41={gate["e41_gate"] if gate else "REFUSED"}\tKEEP actual verdict\tartifact={case}/e41/wave_gate.json;cache={cache},minimal PINs retained after full scan;work=CPU9nice19OMP1JAXcpu timeout900,source={gpu["source_rev"]},actualGPU24h executable')
        if gpu['rc'] != 0:
            done[name] = {'verdict': 'GPU_FAILED', 'rc': gpu['rc'], 'error': gpu['error'],
                          'E41': gate['e41_gate'] if gate else 'UNKNOWN'}
            pending.remove(name)
            append('MAILBOX.md', f'{utc()} v-run: {name} GPU_FAILED {gpu["error"]}; {gpu_file}')
            continue
        prior(case, 'D6')
        rc = guarded(['python', str(ROOT / 'tools/score.py'), str(SOURCE), str(case)], case / 'score.log', 1800)
        if (case / 'compare.json').exists():
            audit_rc = guarded(['python', str(ROOT / 'tools/audit_writer.py'), str(case)], case / 'writer_audit.log', 900)
            assert audit_rc == 0, ('writer audit failed', name, audit_rc)
        if not (case / 'grid_gate.json').exists():
            done[name] = {'verdict': 'SCORER_FAILED', 'rc': rc, 'E41': gate['e41_gate'] if gate else 'UNKNOWN'}
        else:
            d6 = json.loads((case / 'grid_gate.json').read_text())
            missing = sorted({x for frame in d6['by_frame'] for x in frame['missing_vs_cpu']})
            done[name] = {'verdict': d6['verdict'], 'bounded_D6': 'FAIL' if d6['failures'] else 'PASS',
                'paired_frames': d6['paired_frames'], 'cpu_field_count': d6['cpu_field_count'],
                'bounded_field_count': d6['bounded_field_count'], 'coverage_complete': d6['coverage_complete'],
                'first_failure': d6['first_failure'], 'missing_CPU_fields': missing,
                'raw_gpu_nonfinite': d6['raw_gpu_nonfinite'], 'E41': gate['e41_gate'] if gate else 'UNKNOWN',
                'cold_s_per_fc_h': gpu['cold_s_per_fc_h'],
                'host_peak_rss_kib': gpu.get('host_peak_rss_kib'),
                'sampled_peak_vram_MiB': gpu['peak_process_vram_mib_sampled']}
            first = json.dumps(d6['first_failure'], separators=(',', ':')) if d6['first_failure'] else 'none'
            append('MAILBOX.md', f'{utc()} v-run: {name} {d6["verdict"]} boundedD6={done[name]["bounded_D6"]} {d6["paired_frames"]}/25frames first_failure={first} missingCPU={missing} E41={done[name]["E41"]}; {case}/grid_gate.json')
            append('EXPERIMENTS.tsv', f'{utc()}\tv-run\tGPT Sol\tVR-FU28-D6-{name}\tCorrected GPU24h matches unchanged originalCPUWRF\tunchanged frozen full-coverage scorer\tfrozen D6 perfield/perframe+allCPU inventory+rawfinite\tround2 source8afb\t[M CPU]{json.dumps(done[name],separators=(",",":"))}\tKEEP truthful final gate\tartifact={case}/grid_gate.json,{case}/compare.json;cache=NONE;work=CPU9nice19OMP1JAXcpu timeout1800,Swiss dt18 25frames hours0..24,boundarywidth5,manifest9df4fab8 noGPU/no newmodel')
        pending.remove(name)
        write(case / 'RESULT.json', done[name])
        print(utc(), name, json.dumps(done[name], separators=(',', ':')), flush=True)
    write(ROOT / 'STATUS.json', {'utc': utc(), 'done': done, 'pending': pending})
    if pending:
        time.sleep(5)
write(ROOT / 'CONSUMERS_DONE.json', {'utc': utc(), 'done': done})
