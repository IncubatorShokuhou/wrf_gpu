"""One manager-requested c07 rerun after the final WSM3 c04 consumers."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
OWN = Path(__file__).resolve().parent
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep')
MAIN = Path('<USER_HOME>/src/wrf_gpu2')
PLAN = json.loads((OWN / 'C07_SHINHONG_RERUN_PLAN.json').read_text())
LAST = datetime.fromisoformat(PLAN['last_start_utc'])
END = datetime.fromisoformat(PLAN['window_end_utc'])


def now():
    return datetime.now(timezone.utc)


def append(name, line):
    with (MAIN / '.agent/kernel' / name).open('a') as f:
        f.write(now().isoformat() + ' v-prep: ' + line + '\n')


while True:
    p = ROOT / 'c04/QUALIFICATION.json'
    if p.exists():
        q = json.loads(p.read_text())
        if q['gpu_process'].get('source_commit') == '061d53020e64b9946f2791984a130ecfe8b4dde4' and q['cache_remaining'] is False:
            break
    assert now() <= LAST, 'Last24h start passed'
    time.sleep(3)
while Path('/tmp/wrf_gpu2_quiet').exists():
    time.sleep(3)
search = subprocess.run(['rg', '-i', 'c07|Shin.Hong|44b7b226b',
    *[str(MAIN / '.agent/kernel' / n) for n in ('FINDINGS.md', 'EXPERIMENTS.tsv', 'INBOX.md')]],
    capture_output=True, text=True)
(ROOT / 'GPU25-MAIN_SEARCH.txt').write_text(search.stdout)
assert shutil.disk_usage(DATA).free >= 120 * 1024**3
source = DATA / 'gpu25-tree'
subprocess.run(['git', '-C', str(MAIN), 'worktree', 'add', '--detach', str(source), PLAN['source_commit']], check=True)
(source / 'data/wrf_pristine').symlink_to('<USER_HOME>/src/wrf_pristine', target_is_directory=True)
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
subprocess.run(['python', str(OWN / 'archive_attempt.py'), 'before_ShinHong_44b7', 'c07'], check=True)
old_pid = int((ROOT / 'CPU212_PID').read_text())
assert b'VP212-tools/consume_ready.py' in Path(f'/proc/{old_pid}/cmdline').read_bytes()
assert subprocess.run(['pgrep', '-P', str(old_pid)], capture_output=True).returncode == 1
os.kill(old_pid, signal.SIGTERM)
tools = DATA / 'VP215-tools'
tools.mkdir(exist_ok=False)
for p in (DATA / 'VP212-tools').glob('*.py'):
    shutil.copyfile(p, tools / p.name)
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tools.glob('*.py')}
assert files == json.loads((DATA / 'VP212-tools/TOOL_SEAL.json').read_text())['files']
(tools / 'TOOL_SEAL.json').write_text(json.dumps(dict(source=str(source), files=files), indent=2) + '\n')
with (ROOT / 'CPU215-CONSUMER.log').open('w') as f:
    consumer = subprocess.Popen(['python', '-u', str(tools / 'consume_ready.py'), str(source)],
        cwd=source, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'CPU215_PID').write_text(str(consumer.pid) + '\n')
helpers = DATA / 'GPU25-tools'
helpers.mkdir(exist_ok=False)
for name in ('gpu_window.sh', 'gpu_inside.sh', 'gpu_authorization.py'):
    shutil.copyfile(OWN / name, helpers / name)
subprocess.run(['bash', '-n', str(helpers / 'gpu_window.sh'), str(helpers / 'gpu_inside.sh')], check=True)
# Keep source24 only if its old official wrapper is still MAIN's pinned copy.
old_source = DATA / 'gpu24-tree'
pin = (MAIN / 'scripts/with_gpu_lock.sha256').read_text().split()[0]
main_matches = hashlib.sha256((MAIN / 'scripts/with_gpu_lock.sh').read_bytes()).hexdigest() == pin
legacy = old_source / 'scripts/with_gpu_lock.sh'
env = {**os.environ, 'GPUWRF_GPU_ARM_CPUS': '3,19,13,29', 'GPUWRF_FINALVAL_PARALLEL': '3'}
if main_matches:
    subprocess.run(['git', '-C', str(MAIN), 'worktree', 'remove', str(old_source)], check=True)
else:
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == pin
    env['GPUWRF_PINNED_LOCK_WRAPPER'] = str(legacy)
receipt = dict(PLAN, utc=now().isoformat(), source=str(source), consumer_pid=consumer.pid,
    official_legacy_wrapper_dependency=None if main_matches else str(legacy),
    helper_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers.iterdir()})
(ROOT / 'GPU25_SOURCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
append('EXPERIMENTS.tsv', '\tGPT Sol\tVP215-C07\tOne changed-ShinHong full24h rerun\tQUEUED exact44b7b226b\tartifact=ROOT/GPU25-C07.log;cache=DATA/arms/c07/cache,tmp;work=originalCPU/frozenD6/E41/12GSwap0/1770child1800cohort/CORE3,19,13,29/fresh89minwindow')
append('MAILBOX.md', 'GPU25 c07 QUEUED exact44b7b226b afterWSM03consumers; oldPASS SHAarchived before_ShinHong_44b7; CPU215PID' + str(consumer.pid) + '/source25/helperBYTE212; officialpinnedwrapper/correctcores/one24h1770sarm. Source24 retained only if legacyofficialwrapper dependency.')
assert now() <= LAST
end = min(now() + timedelta(minutes=89), END)
cmd = [str(MAIN / 'scripts/with_ram_slot.sh'), '--label', 'v-prep', '--max', '12G', '--timeout', '5400', '--',
       'timeout', '--kill-after=15', '5400', 'bash', str(helpers / 'gpu_window.sh'), end.strftime('%Y-%m-%d %H:%M:%S UTC'), 'c07']
with (ROOT / 'GPU25-C07.log').open('w') as f:
    rc = subprocess.run(cmd, cwd=source, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
until = time.monotonic() + 2400
while not (ROOT / 'c07/QUALIFICATION.json').exists() and time.monotonic() < until:
    time.sleep(3)
q = json.loads((ROOT / 'c07/QUALIFICATION.json').read_text()) if (ROOT / 'c07/QUALIFICATION.json').exists() else {}
result = dict(case='c07', source=PLAN['source_commit'], cohort_rc=rc, D6=q.get('d6', 'PENDING'),
              E41=q.get('e41', 'PENDING'), cache_remaining=q.get('cache_remaining'))
(ROOT / 'GPU25-C07.json').write_text(json.dumps(result, indent=2) + '\n')
append('MAILBOX.md', 'COHORT SHINHONG ' + json.dumps(result, separators=(',', ':')) + '; originalCPU/frozenD6/F2OPEN; previousPASSretained.')
print(result, flush=True)
