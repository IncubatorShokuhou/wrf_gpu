"""Keep the official lock until this command and all its descendants are gone.

Reuse the release consumer's tested subreaper teardown; this process imports no
JAX and owns no CUDA pool. In particular, an Nsight timeout must not orphan a
collector holding fd 9 after the wrapper reports completion.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=150)
parser.add_argument("--selftest", action="store_true")
parser.add_argument("--whole-step", action="store_true", help="one process group for the entire night step")
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
elif args.whole_step:
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD", "0") != "1"
    fds = ()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command
else:
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    fd = int(os.environ["GPUWRF_GPU_LOCK_FD"])
    assert fd >= 3
    actual, expected = os.fstat(fd), os.stat(os.environ["GPUWRF_GPU_LOCK_FILE"])
    assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    fds = (fd,)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command


child = None


def interrupted(signum, frame):
    if args.whole_step:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise cleanup.Terminated(signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupted)
rc = 1
try:
    # Only the outer controller creates a group. It stays in its caller's
    # group so cancellation through an enclosing RAM scope/timeout reaches it.
    child = subprocess.Popen(command, start_new_session=args.whole_step, pass_fds=fds)
    rc = child.wait(timeout=args.seconds)
except subprocess.TimeoutExpired:
    rc = 124
except cleanup.Terminated as exc:
    rc = 128 + exc.signum
finally:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    result = cleanup.teardown(grace=2.)
    assert not cleanup.descendants(os.getpid())
    print(json.dumps({"supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgrp(),
                      "payload_pgid": child.pid if child is not None and args.whole_step else None,
                      "rc": rc, "teardown": result}), flush=True)
if args.selftest:
    assert rc == 0 and result["termed"] >= 1, result
    print("subreaper orphan-cleanup selftest PASS")
elif args.whole_step and rc == 0 and (result.get("termed", 0) or result.get("killed", 0)):
    rc = 125  # A nominally successful step left live descendants; do not certify it.
raise SystemExit(rc)
