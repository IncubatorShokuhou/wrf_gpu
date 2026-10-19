"""Run a CPU script (scripts/wn3_score.py and its wn3_fast_compare children) inside an alisios_guard.guarded_session
(lane w41-nestqual). The scorer reads only our private copies (CPU twin output + GPU frames staged once under the guard
by stage_gpu.py, both outside the ALISIOS roots); the session admits exactly one ALISIOS file, the frozen twin-threshold
metadata the scorer hashes for its summary (no values used), and the read spy refuses every other ALISIOS open, also
in the child compare processes (each python child is re-run through this wrapper).
usage: JAX_PLATFORMS=cpu PYTHONPATH=<tree>/src python guard_run.py <receipt_dir> <script.py> [args ...]
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

from gpuwrf.assimilation.letkf.alisios_guard import guarded_session

THRESHOLDS = Path("<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_WN3_TWIN_THRESHOLDS_v1.json")
ISSUE_DAY, VALID_DAY = "20260707", "20260708"   # G2 0707 arms: issue day + first valid day (guard scan ALLOW)

original = subprocess.Popen


def guarded_child(command, *args, **kwargs):
    if isinstance(command, list) and command[0] == sys.executable and len(command) > 1 and command[1].endswith(".py"):
        command = [command[0], __file__, receipts, *command[1:]]
    return original(command, *args, **kwargs)


receipts = sys.argv[1]
subprocess.Popen = guarded_child
target = sys.argv[2]
sys.argv = sys.argv[2:]
with guarded_session({THRESHOLDS: VALID_DAY}, [ISSUE_DAY], Path(receipts) / str(os.getpid()),
                     purpose="wrf_gpu2 w41-nestqual: D6 nest gate scoring of private CPU-twin / guarded GPU copies; "
                             "frozen twin-threshold metadata hashed only"):
    runpy.run_path(target, run_name="__main__")
