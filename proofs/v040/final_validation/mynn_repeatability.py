"""Private, serial original-WRF MYNN repeatability checks (CPU only).

Never writes a case input/reference or a foreign lane. Each heavy arm acquires
the official RAM slot; outputs from every arm are retained in distinct dirs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
MAIN = Path('<USER_HOME>/src/wrf_gpu2')
HERE = Path(__file__).resolve().parent
BASE = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/mynn-repeat-20261014')
QUIET = Path('/tmp/wrf_gpu2_quiet')
ORIGINAL = Path('<USER_HOME>/src/wrf_pristine/WRF')
ZERO = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP26_zero/WRF')
EXES = {'original': (ORIGINAL, '52f914cb38e0899d8a1da54bd52f3e08d81ba0f6a7015d3159e48393f5226bd6'),
        'zero': (ZERO, '9a67a93dd891115c095f64a95ce8f50e7ff0942ef1abf4185faf20bd7065742a')}
SELECTED = ('c00', 'c17')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def append(path, line):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, (line + '\n').encode())
    finally:
        os.close(fd)


def wait_quiet():
    while QUIET.exists():
        time.sleep(5)


def before(ident):
    # Grep MAIN again immediately before every experiment, not a branch copy.
    result = subprocess.run(['grep', '-iE', 'MYNN|rmol|repeatab|VP99|VP100',
        *[str(MAIN / '.agent/kernel' / p) for p in
          ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]], capture_output=True, text=True)
    return dict(utc=utc(), experiment=ident, grep_returncode=result.returncode,
                matching_lines=len(result.stdout.splitlines()),
                grep_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                different='Two complete 24h reruns of D6-PASS c00/c17, same original exe and cwd; existing P3 probes were short and different MP cases.')


def matrix():
    return json.loads((ROOT / 'MATRIX.json').read_text())['cases']


def inventory():
    rows = []
    for row in matrix():
        if row['options']['bl_pbl_physics'] != 5:
            continue
        domains = {}
        for domain in row['domains']:
            p = ROOT / row['id'] / f'D6_{domain}.json'
            d = json.loads(p.read_text()) if p.exists() else {}
            domains[domain] = dict(verdict=d.get('verdict', 'UNRUN'),
                paired_frames=d.get('paired_frames', 0), bounded_misses=len(d.get('failures', [])),
                nonfinite=len(d.get('nonfinite', [])),
                missing_cpu_fields=sorted({v for frame in d.get('by_frame', [])
                                          for v in frame.get('missing_vs_cpu', [])}),
                receipt=str(p) if p.exists() else None,
                sha256=sha(p) if p.exists() else None)
        eligible = row.get('gpu_eligible', True)
        run_path = ROOT / row['id'] / 'gpu_run.json'
        reuse = ROOT / row['id'] / 'GPU_REUSE.json'
        if reuse.exists():
            run_path = Path(json.loads(reuse.read_text())['producer_receipt'])
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        zero_evidence = None
        zero_reference = None
        if row['id'] in ('c21', 'c22', 'c23', 'c24'):
            reference = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP32') / f"{row['options']['mp_physics']}a"
            if (reference / 'receipt.json').exists():
                zr = json.loads((reference / 'receipt.json').read_text())
                if zr['status'] == 'COMPLETE':
                    assert zr['source_executable_sha256'] == EXES['zero'][1]
                    assert zr['input_sha256'] == row['input_sha256']
                    assert sha(reference / 'outputs.json') == zr['output_manifest_sha256']
                    zero_reference = dict(directory=str(reference), receipt_sha256=sha(reference / 'receipt.json'),
                        output_manifest_sha256=zr['output_manifest_sha256'], frames=zr['frames'], status=zr['status'])
            z = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP33_zero') / row['id']
            if (z / 'receipt.json').exists():
                evidence = json.loads((z / 'receipt.json').read_text())
                if 'input_sha256' in evidence:
                    assert evidence['input_sha256'] == row['input_sha256']
                    assert evidence['zero_source_executable_sha256'] == EXES['zero'][1]
                    zero_evidence = dict(receipt=str(z / 'receipt.json'), sha256=sha(z / 'receipt.json'),
                        verdict=evidence['verdict'], paired_frames=evidence['paired_frames'],
                        bounded_misses=evidence['bounded_failures'], nonfinite=evidence['nonfinite'])
                else:
                    # CPU-ready / no-GPU marker is not a scored D6 receipt.
                    assert evidence['case'] == row['id'] and evidence['verdict'] == 'INCOMPLETE'
                    assert evidence['actual_gpu_frames'] == 0 and zero_reference is not None
                    assert evidence['zero_receipt_sha256'] == zero_reference['receipt_sha256']
                    zero_evidence = dict(receipt=str(z / 'receipt.json'), sha256=sha(z / 'receipt.json'),
                        verdict='INCOMPLETE', paired_frames=0, bounded_misses=None, nonfinite=None,
                        reason=evidence['reason'])
        rows.append(dict(case=row['id'], active=eligible, options=row['options'],
            geography=row['geography'], domains=domains,
            gpu_source_commit=run.get('source_commit', run.get('source_rev')),
            actual_gpu_receipt=str(run_path) if run_path.exists() else None,
            actual_gpu_receipt_sha256=sha(run_path) if run_path.exists() else None,
            zero_reference=zero_reference,
            reference_of_record=(zero_reference['directory'] if zero_reference else 'ZERO_REFERENCE_PENDING')
                if row['id'] in ('c21', 'c22', 'c23', 'c24') else str(ROOT / row['id'] / 'cpu'),
            zero_reference_d6=zero_evidence,
            scope_reason=row.get('control_reason', row.get('exclusion', row.get('qualification', row['status']))),
            full_d6_pass=eligible and all(d['verdict'] == 'PASS' for d in domains.values()),
            zero_reference_required=eligible and any(d['verdict'] == 'FAIL' for d in domains.values()),
            preserved_cpu_executable_sha256=json.loads((ROOT / row['id'] / 'cpu_run.json').read_text()).get('executable', {}).get('original_sha256')
                if (ROOT / row['id'] / 'cpu_run.json').exists() else None))
    report = dict(schema='v-prep-mynn-reference-inventory-v1', utc=utc(),
        authority='MAILBOX 2026-10-14T22:51:14Z P3/MYNN decision; frozen D6 bounds unchanged',
        scope='Historical D6 at cited measured sources; schema INCOMPLETE is not a numerical PASS or FAIL. Excluded controls listed separately.',
        active_count=sum(r['active'] for r in rows), excluded_count=sum(not r['active'] for r in rows),
        selected_repeatability_cases=list(SELECTED), zero_reference_owner={'c21': 't-p3fix', 'c22': 't-p3fix', 'c23': 't-p3fix', 'c24': 't-p3fix'}, rows=rows)
    save(HERE / 'MYNN_REFERENCES.json', report)
    lines = ['# MYNN references', '', f"Inventory: {report['utc']}. {report['active_count']} active composites and {report['excluded_count']} excluded controls.", '',
        'D6 evidence is historical and source-bound. `INCOMPLETE` can mean missing writer fields, a compile failure, or a stopped forecast; it is not a full numerical PASS.', '',
        'Manager policy: a MYNN D6 numerical FAIL uses original WRF zero-init build `9a67a93d` as reference of record, with original-vs-zero spread disclosed. Original references remain intact and D6 bounds remain frozen. t-p3fix owns c21–c24 replacement references and rescoring. c00 and c17 receive two full original-build repeats here; any non-repeatable case receives a separate zero-init reference.', '',
        '| Case | MP / CU | Recorded case D6 (paired frames; bounded misses) | Separate zero-init rescore | Missing recorded fields | Scope |',
        '|---|---|---|---|---|---|']
    for r in rows:
        ds = '; '.join(f"{k} {d['verdict']} ({d['paired_frames']}; {d['bounded_misses']})" for k, d in r['domains'].items())
        missing = sorted({v for d in r['domains'].values() for v in d['missing_cpu_fields']})
        scope = 'active' if r['active'] else 'excluded: ' + r['scope_reason']
        z = r['zero_reference_d6']
        ztext = f"{z['verdict']} ({z['paired_frames']}; {z['bounded_misses'] if z['bounded_misses'] is not None else 'unmeasured'})" if z else '—'
        lines.append(f"| {r['case']} | {r['options']['mp_physics']} / {r['options']['cu_physics']} | {ds} | {ztext} | {', '.join(missing) or '—'} | {scope} |")
    (HERE / 'MYNN_REFERENCES.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


def stage(case, mode):
    reference, expected = EXES[mode]
    assert sha(reference / 'main/wrf.exe') == expected
    directory = BASE / case / mode
    runtime = directory / 'runtime'
    diag = ROOT / ('rp' + case[1:] + ('z' if mode == 'zero' else ''))
    if not runtime.exists():
        assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3
        directory.mkdir(parents=True, exist_ok=True)
        runtime.mkdir()
        diag.mkdir(exist_ok=False)
        # This helper changes only two diagnostic OPEN literals, then reverses
        # the patches and asserts exact original binary identity.
        from prepare import private_executable
        private_executable(diag, reference)
        shutil.copyfile(ROOT / case / 'input/namelist.input', runtime / 'namelist.input')
        assert not (runtime / 'namelist.input').is_symlink()
        assert (runtime / 'namelist.input').resolve().is_relative_to(runtime)
        for p in (ROOT / case / 'cpu').iterdir():
            if p.name.startswith(('wrfout_', 'rsl.', 'wrfrst_', 'wrfdiag_')) or p.name in ('namelist.input', 'namelist.output', 'wrf.exe'):
                continue
            if p.is_symlink():
                (runtime / p.name).symlink_to(p.resolve(strict=True))
            elif p.is_file():
                # Copy any generated MP lookup table; never let WRF rewrite a
                # table in the canonical reference directory through a link.
                shutil.copyfile(p, runtime / p.name)
        (runtime / 'wrf.exe').symlink_to(diag / 'wrf.exe')
        save(directory / 'STAGING.json', dict(utc=utc(), canonical_case=str(ROOT / case),
            runtime=str(runtime), diagnostic_directory=str(diag),
            executable=json.loads((diag / 'CPU_executable.json').read_text()),
            namelist_sha256=sha(runtime / 'namelist.input'),
            original_input_sha256={p.name: sha(p) for p in (ROOT / case / 'input').iterdir() if p.is_file()}))
    return directory, runtime, diag


def arm(case, mode, number):
    assert case in SELECTED and os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert os.environ.get('GPUWRF_JAX_CACHE') == '0'
    # systemd may close the wrapper's flock fd in the scoped child. Verify the
    # actual cgroup cap instead; entrypoint sequence always uses with_ram_slot.
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2].lstrip('/')
    assert (cg / 'memory.max').read_text().strip() == str(6 * 1024**3)
    assert (cg / 'memory.swap.max').read_text().strip() == '0'
    wait_quiet()
    prereg = before(f'{case}-{mode}-{number}')
    directory, runtime, diag = stage(case, mode)
    out = directory / f'run-{number}'
    out.mkdir(exist_ok=False)
    save(out / 'PREREGISTRATION.json', prereg)
    row = next(r for r in matrix() if r['id'] == case)
    actual = {p.name: sha(runtime / p.name) for p in (ROOT / case / 'input').iterdir() if p.is_file()}
    assert actual == row['input_sha256'], (actual, row['input_sha256'])
    assert not list(runtime.glob('wrfout_*')), 'Refuse overwriting previous run'
    start = time.monotonic()
    started = utc()
    pauses = []
    with (out / 'cpu.log').open('w') as log:
        proc = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10', '5400', './wrf.exe'],
            cwd=runtime, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        paused = False
        try:
            while proc.poll() is None:
                quiet = QUIET.exists()
                if quiet != paused:
                    os.killpg(proc.pid, signal.SIGSTOP if quiet else signal.SIGCONT)
                    paused = quiet
                    if quiet:
                        pauses.append({'start': utc()})
                    else:
                        pauses[-1]['end'] = utc()
                time.sleep(1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGCONT)
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
    wall = time.monotonic() - start
    wait_quiet()
    for p in list(runtime.iterdir()):
        if p.name.startswith(('wrfout_', 'rsl.')) or p.name == 'namelist.output':
            assert p.is_file() and not p.is_symlink()
            p.rename(out / p.name)
    from netCDF4 import Dataset
    import numpy as np
    files, nf = [], []
    for p in sorted(out.glob('wrfout_*')):
        wait_quiet()
        fields = {}
        with Dataset(p) as ds:
            ds.set_auto_maskandscale(False)
            for name, var in ds.variables.items():
                a = np.asarray(var[:])
                fields[name] = dict(shape=list(a.shape), dtype=str(a.dtype), dimensions=list(var.dimensions),
                                    sha256=hashlib.sha256(a.tobytes()).hexdigest())
                if np.issubdtype(a.dtype, np.number) and np.any(~np.isfinite(a)):
                    nf.append(dict(file=p.name, field=name, count=int(np.count_nonzero(~np.isfinite(a)))))
            files.append(dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p), fields=fields))
    text = (out / 'cpu.log').read_text(errors='replace')
    for p in out.glob('rsl.*'):
        text += p.read_text(errors='replace')
    after = {p.name: sha(runtime / p.name) for p in (ROOT / case / 'input').iterdir() if p.is_file()}
    receipt = dict(schema='v-prep-mynn-24h-repeat-arm-v1', case=case, mode=mode, repeat=number,
        started=started, finished=utc(), wall_s=wall, rc=proc.returncode,
        success_marker='SUCCESS COMPLETE WRF' in text, frames=len(files), nonfinite=nf,
        input_before=actual, input_after=after, original_inputs_intact=after == actual,
        executable=json.loads((diag / 'CPU_executable.json').read_text()),
        launcher_sha256=sha(__file__),
        resources='CPU core0/nice19; official RAMslot MemoryMax6G MemorySwapMax0; timeout5400+10s; no JAX/GPU/cache',
        quiet_pauses=pauses, files=files)
    receipt['status'] = 'COMPLETE' if receipt['rc'] == 0 and receipt['success_marker'] and len(files) == 25 and not nf and after == actual else 'FAILED'
    save(out / 'RECEIPT.json', receipt)
    append(MAIN / '.agent/kernel/EXPERIMENTS.tsv', '\t'.join((utc(), 'v-prep', 'GPT Sol', f'VP99-{case}-{mode}-{number}',
        'Full24h D6-PASS MYNN original repeatability' if mode == 'original' else 'Defined-zero original MYNN replacement; preserve raw reference',
        receipt['executable']['original_sha256'], 'WRF_SUCCESS+25finiteframes+sealedinputSHA+allfieldhashes',
        f"[M CPU] {receipt['status']} rc{proc.returncode} frames{len(files)} NF{len(nf)} wall{wall:.3f}s",
        'KEEP independent reference outputs; repeatability verdict follows pair comparison',
        f'artifact={out}/RECEIPT.json;cache=NONE GPUWRF0;work=CPU0NI19 RAMslot6GSwap0 timeout5400 hours24 namelistSHA{actual["namelist.input"]} options{json.dumps(row["options"],separators=(",", ":"))}')))
    print(f'{utc()} {case} {mode} #{number}: {receipt["status"]} {wall:.1f}s', flush=True)
    assert receipt['status'] == 'COMPLETE'


def compare(case, mode='original'):
    wait_quiet()
    directory = BASE / case / mode
    save(directory / 'COMPARE_PREREGISTRATION.json', before(f'{case}-{mode}-compare'))
    a = json.loads((directory / 'run-1/RECEIPT.json').read_text())
    b = json.loads((directory / 'run-2/RECEIPT.json').read_text())
    assert a['status'] == b['status'] == 'COMPLETE'
    assert a['input_before'] == b['input_before'] and a['executable'] == b['executable']
    left = {p['file']: p for p in a['files']}
    right = {p['file']: p for p in b['files']}
    assert left.keys() == right.keys()
    differences, baseline_differences = [], []
    from netCDF4 import Dataset
    import numpy as np
    for name, f in left.items():
        wait_quiet()
        assert f['fields'].keys() == right[name]['fields'].keys()
        for field, info in f['fields'].items():
            if info != right[name]['fields'][field]:
                with Dataset(directory / 'run-1' / name) as x, Dataset(directory / 'run-2' / name) as y:
                    x.set_auto_maskandscale(False); y.set_auto_maskandscale(False)
                    xa, ya = np.asarray(x[field][:]), np.asarray(y[field][:])
                    delta = np.asarray(xa, dtype='float64') - np.asarray(ya, dtype='float64') if np.issubdtype(xa.dtype, np.number) else None
                    differences.append(dict(file=name, field=field,
                        max_abs=float(np.max(np.abs(delta))) if delta is not None else None,
                        rmse=float(np.sqrt(np.mean(delta**2))) if delta is not None else None))
        # Also check that the preserved canonical CPU reference is reproducible,
        # rather than silently accepting two identical but different trajectories.
        with Dataset(ROOT / case / 'cpu' / name) as ds:
            ds.set_auto_maskandscale(False)
            for field, var in ds.variables.items():
                raw = np.asarray(var[:])
                oldsha = hashlib.sha256(raw.tobytes()).hexdigest()
                if field not in f['fields'] or oldsha != f['fields'][field]['sha256']:
                    baseline_differences.append(dict(file=name, field=field))
    result = dict(schema='v-prep-mynn-repeat-pair-v1', utc=utc(), case=case, mode=mode,
        same_executable_and_cwd=True, frames=len(left), fields_per_frame=len(next(iter(left.values()))['fields']),
        pair_array_byte_equal=not differences, pair_differences=differences,
        canonical_array_byte_equal=not baseline_differences, canonical_differences=baseline_differences,
        pair_file_byte_equal=all(f['sha256'] == right[n]['sha256'] for n, f in left.items()),
        inputs=a['input_before'], executable=a['executable'],
        verdict='REPEATABLE' if not differences and not baseline_differences else 'NON_REPEATABLE',
        replacement_required=bool(differences or baseline_differences), cache='NONE')
    save(directory / 'COMPARISON.json', result)
    append(MAIN / '.agent/kernel/EXPERIMENTS.tsv', '\t'.join((utc(), 'v-prep', 'GPT Sol', f'VP100-{case}-{mode}',
        'Compare allfields of full24h original repeat pair and canonical preserved reference', a['executable']['original_sha256'],
        'Raw numeric+character array bytes including signedzero; schema+input+exe identity',
        f"[M CPU] {result['verdict']} frames{len(left)} pair_differences{len(differences)} canonical_differences{len(baseline_differences)} fileBYTE{result['pair_file_byte_equal']}",
        'KEEP truthful reference qualification; zero replacement if nonrepeatable',
        f'artifact={directory}/COMPARISON.json;cache=NONE;work=full24h allfields sealedNML{a["input_before"]["namelist.input"]}')))
    print(json.dumps({k: v for k, v in result.items() if k not in ('pair_differences', 'canonical_differences', 'executable', 'inputs')}, indent=2), flush=True)
    return result


def cleanup(case, mode):
    # Exact owned disposable files only. All numerical outputs/receipts survive.
    wait_quiet()
    runtime = BASE / case / mode / 'runtime'
    diag = ROOT / ('rp' + case[1:] + ('z' if mode == 'zero' else ''))
    assert runtime.resolve().is_relative_to(BASE.resolve())
    for p in runtime.iterdir():
        assert not p.name.startswith('wrfout_'), 'Outputs must have been archived'
        assert p.is_symlink() or p.is_file()
        p.unlink()
    runtime.rmdir()
    for p in diag.iterdir():
        assert p.name in ('wrf.exe', 'CPU_executable.json', 'd0', 'd1') and p.is_file()
        p.unlink()
    diag.rmdir()


def sequence():
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    BASE.mkdir(parents=True, exist_ok=True)
    for case in SELECTED:
        for n in (1, 2):
            wait_quiet()
            subprocess.run([str(MAIN / 'scripts/with_ram_slot.sh'), '--label', 'v-prep', '--timeout', '14400', '--',
                'bash', '-c', 'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec taskset -c 0 nice -n 19 timeout --kill-after=20 5700 "$@"', 'v-prep-repeat',
                sys.executable, str(Path(__file__).resolve()), 'arm', '--case', case, '--mode', 'original', '--number', str(n)], check=True)
        result = compare(case)
        cleanup(case, 'original')
        if result['replacement_required']:
            wait_quiet()
            subprocess.run([str(MAIN / 'scripts/with_ram_slot.sh'), '--label', 'v-prep', '--timeout', '14400', '--',
                'bash', '-c', 'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec taskset -c 0 nice -n 19 timeout --kill-after=20 5700 "$@"', 'v-prep-zero',
                sys.executable, str(Path(__file__).resolve()), 'arm', '--case', case, '--mode', 'zero', '--number', '1'], check=True)
            cleanup(case, 'zero')
        append(MAIN / '.agent/kernel/MAILBOX.md', f'{utc()} v-prep: MYNN original24h {case} {result["verdict"]}: pair differences={len(result["pair_differences"])} canonical differences={len(result["canonical_differences"])}; outputs preserved {BASE/case}; zero replacement={result["replacement_required"]}.')
    save(BASE / 'DONE.json', dict(utc=utc(), cases=list(SELECTED), status='COMPLETE', GPU=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inventory', 'sequence', 'arm', 'compare'))
    parser.add_argument('--case', choices=SELECTED)
    parser.add_argument('--mode', choices=EXES, default='original')
    parser.add_argument('--number', type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.action == 'inventory': inventory()
    elif args.action == 'sequence': sequence()
    elif args.action == 'arm': arm(args.case, args.mode, args.number)
    elif args.action == 'compare': compare(args.case, args.mode)
