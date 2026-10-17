"""Six explicitly authorized reruns on the exact t-wsmdz final product."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, choices=(1, 2), default=1)
parser.add_argument('--observe-first', action='store_true')
args = parser.parse_args()
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
MAIN = Path('<USER_HOME>/src/wrf_gpu2')
PLAN = json.loads((ROOT / 'GPU23_SOURCE.json').read_text())
SOURCE = Path(PLAN['source'])
HELPERS = SOURCE.parent / 'GPU23-tools'
END = datetime.fromisoformat(PLAN['window_end_utc'])
LAST_START = datetime.fromisoformat(PLAN['last_start_utc'])
assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PLAN['source_commit']
assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
for name, digest in PLAN['helper_sha256'].items():
    assert hashlib.sha256((HELPERS / name).read_bytes()).hexdigest() == digest


def append(name, line):
    with (MAIN / '.agent/kernel' / name).open('a') as f:
        f.write(line + '\n')


def post(label, cases, rc):
    until = time.monotonic() + 2400
    for case in cases:
        path = ROOT / case
        while not (path / 'QUALIFICATION.json').exists() and time.monotonic() < until:
            time.sleep(3)
        q = json.loads((path / 'QUALIFICATION.json').read_text()) if (path / 'QUALIFICATION.json').exists() else {}
        launch = json.loads((path / 'gpu_launch.json').read_text()) if (path / 'gpu_launch.json').exists() else {}
        if q:
            assert q['gpu_process'] == launch
        result = dict(case=case, cohort_rc=rc, process_rc=launch.get('process_rc'),
                      source=PLAN['source_commit'], D6=q.get('d6', 'PENDING'),
                      E41=q.get('e41', 'PENDING'), frames=len(list((path / 'gpu').glob('wrfout*'))),
                      cache_remaining=q.get('cache_remaining'))
        (ROOT / f'GPU23-{label}-{case}.json').write_text(json.dumps(result, indent=2) + '\n')
        line = f'{datetime.now(timezone.utc).isoformat()} v-prep: COHORT {label} {json.dumps(result,separators=(",",":"))}; originalCPU/frozenD6; F2OPEN; previousattemptSHAarchived.'
        append('MAILBOX.md', line)
        print(line, flush=True)


with ThreadPoolExecutor(max_workers=2) as observers:
    if args.observe_first:
        observers.submit(post, 'WSM01', PLAN['cohorts'][0], 'see original receipts')
    for index, cases in enumerate(PLAN['cohorts'][args.start-1:], args.start):
        if datetime.now(timezone.utc) > LAST_START:
            raise SystemExit('Last24h start passed; no arm queued')
        while Path('/tmp/wrf_gpu2_quiet').exists():
            time.sleep(3)
        assert all(not (ROOT / c / 'gpu_launch.json').exists() for c in cases)
        label = f'WSM{index:02}'
        search = subprocess.run(['rg', '-i', '|'.join(cases),
            *[str(MAIN / '.agent/kernel' / n) for n in ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]],
            capture_output=True, text=True)
        (ROOT / f'GPU23-{label}-MAIN_SEARCH.txt').write_text(search.stdout)
        end = min(datetime.now(timezone.utc) + timedelta(minutes=89), END)
        log = ROOT / f'GPU23-{label}.log'
        append('EXPERIMENTS.tsv', f'{datetime.now(timezone.utc).isoformat()}\tv-prep\tGPT Sol\tVP208-{label}\tFinal WSM/WDM caller+radius full24h qualification\tQUEUED {cases}\tPENDING D6/E41\tartifact={log};cache=DATA/arms/cXX/cache;work={SOURCE}@{PLAN["source_commit"]}/originalCPU/1770child1800cohort/3parallel40GSwap0/fresh89minqueue/hard1845laststart1815')
        cmd = [str(MAIN / 'scripts/with_ram_slot.sh'), '--label', 'v-prep', '--max', '40G',
               '--timeout', '5400', '--', 'timeout', '--kill-after=15', '5400',
               'bash', str(HELPERS / 'gpu_window.sh'), end.strftime('%Y-%m-%d %H:%M:%S UTC'), *cases]
        with log.open('w') as f:
            rc = subprocess.run(cmd, cwd=SOURCE, stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, 'GPUWRF_FINALVAL_PARALLEL': '3',
                     'GPUWRF_GPU_ARM_CPUS': '3,19,13,29'}, timeout=5500).returncode
        observers.submit(post, label, cases, rc)
        print(label, 'released', rc, flush=True)
