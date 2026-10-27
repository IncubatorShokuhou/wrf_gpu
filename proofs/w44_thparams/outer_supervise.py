"""Own the whole night arm so top-group TERM reaches nested timeout sessions.

This parent never imports JAX or owns a CUDA pool/official GPU lease. The
child acquires its lease itself. The release consumer's tested subreaper
teardown owns only descendants, including setsid/TERM-ignoring samplers.
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

parser=argparse.ArgumentParser()
parser.add_argument("--seconds",type=float,required=True)
parser.add_argument("command",nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==["--"] else args.command
assert args.seconds>0 and command
source=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("thparams_outer_teardown",source/"scripts/release/bundle_consumer.py")
cleanup=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(36,1,0,0,0)==0, "cannot become whole-arm subreaper"


def interrupted(signum,frame):
    raise cleanup.Terminated(signum)


for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
    signal.signal(sig,interrupted)
rc=1
try:
    child=subprocess.Popen(command,start_new_session=True)
    rc=child.wait(timeout=args.seconds)
except subprocess.TimeoutExpired:
    rc=124
except cleanup.Terminated as exc:
    rc=128+exc.signum
finally:
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,signal.SIG_IGN)
    result=cleanup.teardown(grace=2.)
    assert not cleanup.descendants(os.getpid()),result
    print(json.dumps({"outer_rc":rc,"cleanup":result}),flush=True)
raise SystemExit(rc)
