"""Yield one real v-prep cohort between the two authorized pairs."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round4_qhmass')
MAIL=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md')
assert os.environ['JAX_PLATFORMS']=='cpu' and os.sched_getaffinity(0)=={9}
def note(s):
    stamp=datetime.now(timezone.utc).isoformat()
    with MAIL.open('a') as f: f.write(f'{stamp} v-run: {s}\n')
    print(stamp,s,flush=True)
deadline=time.monotonic()+7200
first=ROOT/'mp18_mp18_cu5_WRAPPER_DONE.json'
while not first.exists():
    assert time.monotonic()<deadline,'pair1 deadline'
    time.sleep(5)
result=json.loads(first.read_text())
assert result['rc']==0,result
assert all(json.loads((ROOT/n/'GPU_DONE.json').read_text())['rc']==0 for n in ('mp18','mp18_cu5'))
prior=json.loads((ROOT/'mp18_mp18_cu5_COMMAND.json').read_text())['observed_predecessor']
note('Firstpair18/18cu5 GPU DONE/released; E41/D6 consumer active, notyet scientificverdict. Yielding real v-prep cohort before pair24/26 per05:09 order.')
marker=Path('/tmp/wrf_gpu2_gpu.next')
if not marker.exists(): marker.write_text('v-prep\n')
while True:
    assert time.monotonic()<deadline,'v-prep alternation deadline'
    holder=Path('/tmp/wrf_gpu2_gpu.lock.holder').read_text().strip()
    if holder.startswith('holder=v-prep ') and holder!=prior:
        break
    time.sleep(5)
while Path('/tmp/wrf_gpu2_quiet').exists():
    assert time.monotonic()<deadline,'QUIET deadline'
    time.sleep(5)
assert not marker.exists() or marker.read_text().split()[0]=='v-run','unexpected next marker'
marker.write_text('v-run\n')
(ROOT/'SECOND_PAIR_ORDER.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'intervening_active_vprep_holder':holder,'first_pair_wrapper':result,'next':'v-run'},indent=2)+'\n')
note('Alternation observed '+holder+'; NOW queueing24/26 behindthatcohort; NEXT=v-run/fresh3600lockwait/1800pair/1770children RAM12GSwap0. Source remains immutable627, no older PASS reused.')
rc=subprocess.run(['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"','v-run-next',sys.executable,str(ROOT/'tools/run_pair.py'),'mp24','mp26']).returncode
print('SECOND_PAIR_WRAPPER',rc,flush=True)
sys.exit(rc)
