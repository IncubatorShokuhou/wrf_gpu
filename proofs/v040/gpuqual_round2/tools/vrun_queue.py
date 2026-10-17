"""Daytime GPU queue: bounded pairs, releasing the shared lock between pairs."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-run')
MAIN = Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')
plan = json.loads((ROOT / 'PLAN.json').read_text())
source = Path(plan.get('source', plan['configs'][0]['source']))
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.sched_getaffinity(0) == {9}
def utc():
    return datetime.now(timezone.utc)
def append(file, message):
    with (MAIN / file).open('a') as f:
        f.write(message + '\n')
start = utc().replace(hour=11, minute=30, second=0, microsecond=0)
while utc() < start or Path('/tmp/wrf_gpu2_quiet').exists():
    time.sleep(min(5, max(.2, (start - utc()).total_seconds())))
for index, names in enumerate(plan['groups']):
    names = [n for n in names if not (ROOT / n / 'GPU_DONE.json').exists()]
    if not names:
        continue
    # 30 min execution + bounded wait must fit before 19:15; no new arms after 18:40.
    now = utc()
    last_start = now.replace(hour=18, minute=40, second=0, microsecond=0)
    wait_budget = min(7200, int((last_start - now).total_seconds()))
    if wait_budget <= 0:
        append('MAILBOX.md', f'{now.isoformat()} v-run: BLOCKED daytime cutoff; unrun configs {names} and later groups; no invented results.')
        break
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(5)
    for name in names:
        case = ROOT / name
        config = next(c for c in plan['configs'] if c['config'] == name)
        runtime = Path(config.get('runtime', str(case)))
        with (case / 'gpu_prior.txt').open('w') as f:
            subprocess.run(['rg', '-i', name + '|t-mprain|t-camuw|t-fdda|t-mpout',
                *[str(MAIN / n) for n in ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]], stdout=f)
        append('EXPERIMENTS.tsv', f'{utc().isoformat()}\tv-run\tGPT Sol\tVR-GPU-{name}-pre\tMerged correction on current main closes prior {name} gate\timmutable{plan["source_rev"]},no model edits; corrected CPUFDDA reference where applicable\t25frames/allCPUfields/frozenD6/E41/F2+wall/VRAM\tround1 FAIL/INCOMPLETE; release glacier changed\tPRE-REGISTER\tRUN\tartifact={case}/gpu_run.json;cache={runtime}/cache(jax,autotune,triton,cuda,xdg);work=Swiss42x42x44dt18hours24output60minFASTdefaults/CENSUS1/SLIM0 same matched namelist,RRTMG_MP_RE={next(c["GPUWRF_RRTMG_MP_RE"] for c in plan["configs"] if c["config"]==name)},group={names},officiallock v-run+bridge+GPUassert timeout1500; sharedIO/cold not S1')
    cmd = ['bash', '-c', 'source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env; exec "$@"',
        'v-run-bridge', 'timeout', '--signal=TERM', '--kill-after=30', str(wait_budget + 1860),
        str(source / 'scripts/with_gpu_lock.sh'), '--timeout', str(wait_budget), '--label', 'v-run',
        '--', 'timeout', '--signal=TERM', '--kill-after=15', '1530', sys.executable,
        str(ROOT / 'tools/batch.py'), *names]
    print(utc().isoformat(), 'launch', names, flush=True)
    with (ROOT / f'group_{index}.log').open('w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    for name in names:
        case = ROOT / name
        if not (case / 'GPU_DONE.json').exists():
            (case / 'GPU_DONE.json').write_text(json.dumps({'utc': utc().isoformat(), 'rc': rc,
                'group_failed_before_receipt': True}) + '\n')
    print(utc().isoformat(), 'group finished rc', rc, flush=True)
    if rc != 0:
        (ROOT / 'QUEUE_STOP.json').write_text(json.dumps({'utc': utc().isoformat(),
            'failed_group': names, 'rc': rc, 'reason': 'Stop repeated launcher/model failure; inspect before retry'}) + '\n')
        break
(ROOT / 'GPU_QUEUE_DONE.json').write_text(json.dumps({'utc': utc().isoformat()}) + '\n')
