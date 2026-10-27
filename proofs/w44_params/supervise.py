"""Keep the official lock until this command and all its descendants are gone.

Reuse the release consumer's tested subreaper teardown; this process imports no
JAX and owns no CUDA pool. In particular, an Nsight timeout must not orphan a
collector holding fd 9 after the wrapper reports completion.
"""
import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=150)
parser.add_argument("--selftest", action="store_true")
parser.add_argument("--whole-arm", action="store_true")
parser.add_argument("--deadline-epoch", type=float)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
assert args.seconds > 0
source = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("params_teardown", source / "scripts/release/bundle_consumer.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0, "cannot become child subreaper"
if args.selftest:
    assert not args.command and os.environ.get("JAX_PLATFORMS") == "cpu"
    command = [sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"]
    fds = ()
else:
    if args.whole_arm:
        # This parent owns no CUDA pool; the child shell acquires its own official lock.
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD", "0") != "1"
        fds = ()
    else:
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
        fd = int(os.environ["GPUWRF_GPU_LOCK_FD"])
        assert fd >= 3
        actual, expected = os.fstat(fd), os.stat(os.environ["GPUWRF_GPU_LOCK_FILE"])
        assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
        fds = (fd,)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command


def interrupted(signum, frame):
    raise cleanup.Terminated(signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupted)
rc = 1
try:
    clock = lambda: time.clock_gettime(time.CLOCK_BOOTTIME)  # Includes host suspension.
    budget = args.seconds
    if args.deadline_epoch is not None:
        anchor = clock()
        utc = float(subprocess.check_output(
            ['<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'], text=True, timeout=5,
            env={**os.environ, 'JAX_PLATFORMS': 'cpu'}))
        budget = min(budget, args.deadline_epoch - utc - (clock() - anchor) - 5.)
    if budget <= 0:
        raise subprocess.TimeoutExpired(command, 0)
    deadline = clock() + budget
    child_env = dict(os.environ)
    if args.whole_arm:
        child_env['PARAMS_ARM_SUPERVISOR_PID'] = str(os.getpid())
    child = subprocess.Popen(command, start_new_session=True, pass_fds=fds, env=child_env)
    while True:
        left = deadline - clock()
        if left <= 0:
            raise subprocess.TimeoutExpired(command, budget)
        try:
            rc = child.wait(timeout=min(1., left))
            break
        except subprocess.TimeoutExpired:
            continue
except subprocess.TimeoutExpired:
    rc = 124
except cleanup.Terminated as exc:
    rc = 128 + exc.signum
finally:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    result = cleanup.teardown(grace=2.)
    assert not cleanup.descendants(os.getpid())
if args.selftest:
    assert rc == 0 and result["termed"] >= 1, result
    print("subreaper orphan-cleanup selftest PASS")
raise SystemExit(rc)
