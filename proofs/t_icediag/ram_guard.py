"""Stop this proof's process group before the official RAM cgroup can OOM.

The official RAM-slot wrapper remains authoritative. This extra early stop
produces a resource receipt instead of an Owner desktop OOM notification.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit-gib', type=float, default=4.75)
    p.add_argument('--max-gib', type=int, choices=(6,12), default=6)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert command
    group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().split('0::', 1)[1].strip().lstrip('/')
    assert a.limit_gib < a.max_gib
    assert (group / 'memory.max').read_text().strip() == str(a.max_gib * 1024**3), group
    process = subprocess.Popen(command, start_new_session=True)
    def terminate_child(signum, _frame):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate_child)
    signal.signal(signal.SIGINT, terminate_child)
    start = time.monotonic(); peak = 0; stopped = False
    while process.poll() is None:
        current = int((group / 'memory.current').read_text())
        peak = max(peak, current)
        if current > a.limit_gib * 1024**3:
            stopped = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(.05)
    rc = process.wait()
    receipt = dict(command=command, child_rc=rc, early_resource_stop=stopped,
                   sampled_cgroup_peak_B=peak, stop_threshold_GiB=a.limit_gib,
                   cgroup=str(group), wall_s=time.monotonic()-start)
    a.out.write_text(json.dumps(receipt, indent=2)+'\n')
    print('RAM_GUARD', json.dumps(receipt), flush=True)
    raise SystemExit(75 if stopped else rc)


if __name__ == '__main__':
    main()
