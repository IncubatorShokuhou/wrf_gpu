"""Continue bounded cohorts; pause for a changed main product or resource gate."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
MAIN = Path('<USER_HOME>/src/wrf_gpu2')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep')
POINTER = ROOT / 'CONTINUOUS_SOURCE.json'
CLOSE = datetime(2026, 10, 15, 18, 45, tzinfo=timezone.utc)
GROUPS = [('P02', ['c24', 'c10', 'c27']),
          ('B11', ['c83', 'c47', 'c56']),
          ('B12', ['c57', 'c73', 'c08']),
          ('B13', ['c19', 'c20']),
          ('S02', ['c28']), ('S03', ['c65']), ('S04', ['c60']),
          ('S05', ['c84']), ('S06', ['c85'])]
parser = argparse.ArgumentParser()
parser.add_argument('--after', nargs='+', default=['c21', 'c22', 'c23'])
parser.add_argument('--after-label', default='P01')
parser.add_argument('--start', default='P02', choices=[name for name, _ in GROUPS])
args = parser.parse_args()
GROUPS = GROUPS[[name for name, _ in GROUPS].index(args.start):]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def append(path, line):
    with path.open('a') as f:
        f.write(line + '\n')


def quiet():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(5)


def post(name, cases, rc):
    until = time.monotonic() + 600
    while time.monotonic() < until:
        if all((ROOT / c / 'QUALIFICATION.json').exists() for c in cases):
            break
        time.sleep(5)
    results = {}
    for c in cases:
        q = read(ROOT / c / 'QUALIFICATION.json')
        results[c] = dict(rc=read(ROOT / c / 'gpu_launch.json').get('process_rc'),
                         D6=q.get('d6', 'PENDING'), E41=q.get('e41', 'PENDING'),
                         frames=len(list((ROOT / c / 'gpu').glob('wrfout*'))))
    if rc is None:
        rc = 0 if all(r['rc'] == 0 for r in results.values()) else 'see case rc'
    line = f'{datetime.now(timezone.utc).isoformat()} v-prep: COHORT {name} rc={rc} {json.dumps(results,separators=(",",":"))}; actual-source-bound, actual E41 retained/F2OPEN; CPU consumers pipeline with next GPU cohort.'
    append(MAIN / '.agent/kernel/MAILBOX.md', line)
    print(line, flush=True)


# P01 was already submitted under its official lock before this continuation.
while not all((ROOT / c / 'gpu_launch.json').exists() for c in args.after):
    if datetime.now(timezone.utc) >= CLOSE:
        raise SystemExit(3)
    time.sleep(5)
with ThreadPoolExecutor(max_workers=1) as observer:
    observer.submit(post, args.after_label, args.after, None)
    for name, cases in GROUPS:
        if datetime.now(timezone.utc) + timedelta(minutes=30) > CLOSE:
            break
        quiet()
        while True:
            source = Path(read(POINTER)['source'])
            different = subprocess.check_output(['git', 'diff', '--name-only',
                subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                'main', '--', 'src', 'data', 'scripts', 'pyproject.toml'], cwd=MAIN, text=True)
            if not different.strip():
                break
            (ROOT / 'NEEDS_CURRENT_SOURCE.json').write_text(json.dumps(dict(
                utc=datetime.now(timezone.utc).isoformat(), cohort=name, files=different.splitlines()), indent=2)+'\n')
            time.sleep(5)
        assert all(not (ROOT / c / 'gpu_launch.json').exists() for c in cases), cases
        # Mandatory main-copy search, retained even when there are no matches.
        search = subprocess.run(['rg', '-i', '|'.join(cases),
            *[str(MAIN / '.agent/kernel' / n) for n in ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]],
            capture_output=True, text=True)
        (ROOT / f'{name}-MAIN_SEARCH.txt').write_text(search.stdout)
        end = min(datetime.now(timezone.utc) + timedelta(minutes=89), CLOSE)
        log = ROOT / f'CONT-{name}.log'
        cmd = ['bash', str(source / 'proofs/v040/final_validation/gpu_window.sh'),
               end.strftime('%Y-%m-%d %H:%M:%S UTC'), *cases]
        wrapped = [str(MAIN / 'scripts/with_ram_slot.sh'), '--label', 'v-prep', '--max', '40G',
                   '--timeout', '5400', '--', 'timeout', '--kill-after=15', '5400', *cmd]
        append(MAIN / '.agent/kernel/EXPERIMENTS.tsv',
            f'{datetime.now(timezone.utc).isoformat()}\tv-prep\tGPT Sol\tVP184-{name}\tContinue final-current composite cohort\tQUEUED {cases}\tPENDING D6/E41\tartifact={log};cache=DATAarms case/cache;workSOURCE={source}/officiallock/1770child1800cohort/3parallel40GSwap0/fresh89minwindow')
        with log.open('w') as f:
            rc = subprocess.run(wrapped, cwd=source, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, 'GPUWRF_FINALVAL_PARALLEL': '3'}, timeout=5500).returncode
        observer.submit(post, name, cases, rc)
        print(name, 'GPU released', rc, flush=True)
