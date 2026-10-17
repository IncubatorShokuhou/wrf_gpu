"""Four private serial original-WRF references, pausing on official QUIET."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from prepare import ROOT, sha, write_json

MAIN = Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
QUIET = Path('/tmp/wrf_gpu2_quiet')


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, line):
    # One O_APPEND write; never rewrite another lane's rows.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, (line + '\n').encode())
    finally:
        os.close(fd)


def inventory(case, domains, hours):
    import numpy as np
    from gpuwrf.io.netcdf_lock import Dataset
    records, nonfinite = [], []
    for domain in domains:
        frames = sorted((case / 'cpu').glob(f'wrfout_{domain}_*'))
        for p in frames:
            try:
                with Dataset(p) as ds:
                    ds.set_auto_mask(False)
                    fields = {name: dict(dimensions=list(v.dimensions), shape=list(v.shape), dtype=str(v.dtype))
                              for name, v in ds.variables.items()}
                    for name, var in ds.variables.items():
                        if np.issubdtype(var.dtype, np.number):
                            n = int(np.count_nonzero(~np.isfinite(var[:])))
                            if n:
                                nonfinite.append(dict(file=p.name, field=name, count=n))
                    records.append(dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p),
                                        times=[b''.join(t).decode() for t in ds['Times'][:]],
                                        fields=fields,
                                        globals={key: str(ds.getncattr(key)) for key in ds.ncattrs()}))
            except (OSError, RuntimeError, KeyError) as exc:
                records.append(dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p),
                                    inventory_error=repr(exc)))
        if len(frames) != hours + 1:
            return records, nonfinite, False
    return records, nonfinite, not nonfinite and not any('inventory_error' in r for r in records)


def finish(job):
    case, row = job['case'], job['row']
    job['log'].close()
    text = (case / 'cpu.log').read_text(errors='replace')
    for rsl in sorted((case / 'cpu').glob('rsl.*.0000')):
        text += '\n' + rsl.read_text(errors='replace')
    marker = 'SUCCESS COMPLETE WRF' in text
    frames, nonfinite, complete = inventory(case, row['domains'], row['hours'])
    elapsed = time.monotonic() - job['start']
    receipt = dict(schema='v040-original-cpu-reference-v1', case=row['id'],
        started_utc=job['utc'], finished_utc=utc(), monotonic_wall_s=elapsed,
        utc_wall_s=datetime.now(timezone.utc).timestamp() - job['epoch'],
        rc=job['process'].returncode, success_marker=marker,
        status='COMPLETE' if job['process'].returncode == 0 and marker and complete else 'FAILED',
        core=job['core'], nice=19, ranks=1, timeout_s=job['timeout'],
        source_commit=job['source_commit'], launcher_sha256=sha(__file__),
        input_sha256=row['input_sha256'],
        actual_input_sha256={p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()},
        executable=json.loads((case / 'CPU_executable.json').read_text()),
        quiet_pauses=job['pauses'], nonfinite=nonfinite, frame_count=len(frames),
        inventory_errors=[r for r in frames if 'inventory_error' in r],
        output_manifest=str(case / 'CPU_OUTPUTS.json'),
        timing_scope='original serial WRF, shared nice19 CPU; not GPU S1 or four-rank CPU benchmark')
    assert receipt['input_sha256'] == receipt['actual_input_sha256']
    write_json(case / 'CPU_OUTPUTS.json', frames)
    receipt['output_manifest_sha256'] = sha(case / 'CPU_OUTPUTS.json')
    write_json(case / 'cpu_run.json', receipt)
    detail = 'COMPLETE' if receipt['status'] == 'COMPLETE' else text[-1000:].replace('\n', ' ')
    append(MAIN / 'EXPERIMENTS.tsv', '\t'.join((utc(), 'v-prep', 'GPT Sol', 'VP02-' + row['id'],
        'Original WRF composite24h reference', job['source_commit'],
        '25hourlyframes/domain+allCPUfieldsfinite+actualinputSHA',
        f"[M CPU]rc{receipt['rc']} frames{len(frames)} NF{len(nonfinite)} wall{elapsed:.3f}s {receipt['status']}",
        'KEEP original reference' if receipt['status'] == 'COMPLETE' else 'FAILED reference retained; no GPU qualification',
        f'artifact={case}/cpu_run.json,{case}/CPU_OUTPUTS.json;cache=NONE GPUWRF0;work=serialCPU{job["core"]}NI19timeout{job["timeout"]}hours24options{json.dumps(row["options"],separators=(",", ":"))}')))
    print(f"{utc()} {row['id']} {receipt['status']} frames={len(frames)} wall={elapsed:.1f}s {detail[:400]}", flush=True)
    # Executables are reproducible from retained pristine bytes and patches.
    # Remove only the private binary after its last execution, preserve receipts.
    (case / 'cpu/wrf.exe').unlink()
    (case / 'wrf.exe').unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--cores', default='0,1,2,3')
    parser.add_argument('--timeout', type=int, default=5400)
    args = parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not QUIET.exists()
    cores = [int(x) for x in args.cores.split(',')]
    assert set(cores) <= {0, 1, 2, 3, 16, 17, 18, 19}
    matrix = json.loads((ROOT / 'MATRIX.json').read_text())
    queue = [r for r in matrix['cases'] if not r['admission_errors']
             and (not args.cases or r['id'] in args.cases)]
    queue = [r for r in queue if not (ROOT / r['id'] / 'cpu_run.json').exists()]
    active, completed = [], []
    def terminate(signum, frame):
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGCONT)
                os.killpg(job['process'].pid, signal.SIGTERM)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while queue or active:
        quiet = QUIET.exists()
        for job in active:
            if job['process'].poll() is not None:
                continue
            if quiet != job['paused']:
                os.killpg(job['process'].pid, signal.SIGSTOP if quiet else signal.SIGCONT)
                job['paused'] = quiet
                if quiet:
                    job['pauses'].append({'start_utc': utc()})
                else:
                    job['pauses'][-1]['end_utc'] = utc()
        if not quiet:
            for job in list(active):
                if job['process'].poll() is not None:
                    completed.append(finish(job))
                    active.remove(job)
            for core in cores:
                if not queue or any(j['core'] == core for j in active):
                    continue
                row = queue.pop(0)
                case = ROOT / row['id']
                assert not list((case / 'cpu').glob('wrfout_*')), 'Refuse overwriting original frames'
                log = (case / 'cpu.log').open('w')
                job = dict(case=case, row=row, core=core, log=log, utc=utc(), epoch=time.time(),
                           start=time.monotonic(), paused=False, pauses=[], timeout=args.timeout,
                           source_commit=matrix['source_commit'])
                job['process'] = subprocess.Popen(['taskset', '-c', str(core), 'nice', '-n', '19',
                    'timeout', '--signal=TERM', '--kill-after=10', str(args.timeout), './wrf.exe'],
                    cwd=case / 'cpu', stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append(job)
                print(f"{utc()} START {row['id']} cpu{core} pid{job['process'].pid}", flush=True)
        time.sleep(.5)
    write_json(ROOT / 'CPU_QUEUE.json', completed)


if __name__ == '__main__':
    main()
