"""CPU-only TOP-PG cancellation with nested timeout and an escaped lock holder."""
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

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert os.environ.get("JAX_PLATFORMS")=="cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
outer=Path(__file__).with_name("outer_supervise.py")
base=Path("<USER_HOME>/wrf_gpu2_lanes/w44-thparams")
record=None
with tempfile.TemporaryDirectory(prefix="cancel_",dir=base) as temp:
    root=Path(temp)
    worker=root/"worker.py"
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
    known=[]
    started=time.monotonic()
    with (root/"outer.log").open("w") as stream:
        process=subprocess.Popen([sys.executable,str(outer),"--seconds","30","--",
                                  "timeout","--kill-after=2","30",sys.executable,str(worker)],
                                 start_new_session=True,stdout=stream,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+10
            while not (root/"sampler.pid").exists():
                assert process.poll() is None,(root/"outer.log").read_text()
                assert time.monotonic()<deadline,"fixture worker did not start"
                time.sleep(.05)
            known=[int((root/name).read_text()) for name in ("worker.pid","sampler.pid")]
            with (root/"dummy.lock").open("a") as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: pass
                else: raise AssertionError("dummy lock was not held")
            os.killpg(process.pid,signal.SIGTERM)  # Manager cancels only TOP PG.
            rc=process.wait(timeout=15)
            assert rc==143,(rc,(root/"outer.log").read_text())
            assert all(not Path("/proc",str(pid)).exists() for pid in known),known
            with (root/"dummy.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record=dict(verdict="PASS",rc=rc,wall_s=time.monotonic()-started,
                        outer_sha256=hashlib.sha256(outer.read_bytes()).hexdigest(),
                        nested_timeout=True,escaped_sampler_ignored_TERM=True,
                        worker_and_sampler_reaped=True,dummy_flock_reacquired=True,
                        gpu_executed=False,official_lock_used=False,log=(root/"outer.log").read_text())
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            for pid in known:
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
args.out.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
