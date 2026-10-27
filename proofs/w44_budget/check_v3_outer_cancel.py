"""CPU-only cancellation of the frozen v3 supervisor's actual top group.

Only the official-lock command is replaced, by a private dummy-flock worker.
The frozen admission, signals, run_group, subreaper, receipts and cleanup run.
No official GPU lock, GPU command, JAX import or model execution is permitted.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


BASE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget')
TOOLS = BASE/'night_tools_v3c_71b8edf1e6'
EXPECTED = __import__('json').loads((TOOLS/'manifest.json').read_text())['files']['night_supervisor_design_b.py']


def check(out):
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    assert hashlib.sha256((TOOLS/'night_supervisor_design_b.py').read_bytes()).hexdigest() == EXPECTED
    with tempfile.TemporaryDirectory(prefix='outer_cancel_',dir=BASE) as temp:
        root = Path(temp)
        worker = root/'dummy_worker.py'
        worker.write_text('''import fcntl,os,signal,time
from pathlib import Path
root=Path(__file__).parent
lock=open(root/'dummy.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
(root/'worker.pid').write_text(str(os.getpid()))
if os.fork()==0:
 os.setsid();signal.signal(signal.SIGTERM,signal.SIG_IGN)
 (root/'sampler.pid').write_text(str(os.getpid()))
 while True:time.sleep(.05)
while True:time.sleep(.05)
''')
        driver = root/'drive.py'
        driver.write_text('''import importlib.util,os,sys,time
from pathlib import Path
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('frozen',Path(TOOLS)/'night_supervisor_design_b.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
M=N.M;M.BASE=root;M.real_utc=lambda:10000
run=N.run_group
def shim(command,log,limit,*,env=None):
 assert Path(log).name=='gpu.log','Unexpected E41 or other command'
 assert Path(command[0]).name=='with_ram_slot.sh'
 split=command.index('--')
 lock=command[split+1:]
 assert Path(lock[0]).name=='with_gpu_lock.sh'
 assert lock[1:6]==['--label','w44-budget','--timeout','1','--']
 command=command[:split+1]+['timeout','--kill-after=2','120',sys.executable,str(root/'dummy_worker.py')]
 return run(command,log,limit,env=env)
N.run_group=shim
os.environ.update(GA_START_UTC='1970-01-01T02:40:00Z',GA_END_UTC='1970-01-01T05:00:00Z')
raise SystemExit(M.main())
'''.replace('Path(TOOLS)','Path('+repr(str(TOOLS))+')'))
        known = []
        record = None
        start = time.monotonic()
        with open(root/'driver.log','w') as stream:
            process = subprocess.Popen([sys.executable,str(driver)],start_new_session=True,
                                       stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+12
                while not (root/'sampler.pid').exists():
                    assert process.poll() is None,(root/'driver.log').read_text()
                    assert time.monotonic()<deadline,'dummy worker did not start'
                    time.sleep(.05)
                known = [int((root/name).read_text()) for name in ('worker.pid','sampler.pid')]
                with open(root/'dummy.lock','a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        raise AssertionError('worker did not hold dummy flock')
                # Exactly the manager's cancellation: only the TOP group gets TERM.
                os.killpg(process.pid,signal.SIGTERM)
                rc = process.wait(timeout=15)
                receipt = json.loads(next(root.glob('NIGHT_*/night_receipt.json')).read_text())
                assert rc == receipt['rc'] == 143,receipt
                assert not receipt['gpu_started'] and receipt['protocol_version']==3
                cleanup = receipt['phases']['gpu']['cleanup']
                assert not cleanup['live_remaining'],cleanup
                assert all(not Path('/proc',str(pid)).exists() for pid in known),known
                with open(root/'dummy.lock','a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                record=dict(verdict='V3_RAM_SCOPE_OUTER_TERM_NO_ORPHANS',rc=rc,elapsed_s=time.monotonic()-start,
                            frozen_supervisor_sha256=EXPECTED,receipt=receipt,
                            dummy_flock_reacquired=True,worker_and_escaped_sampler_reaped=True,
                            scope=__doc__)
            finally:
                # Exact fixture PIDs only; a failed control cannot leak its own worker.
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                for pid in known:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
        out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='receipt'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
