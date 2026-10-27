"""w44-rc: per-arm wall-clock cap WITHOUT a new process group or session (the entry's arm_supervisor.py owns the
ONLY setsid of the launcher; GNU timeout is never used because it moves the command into its own process group).

usage: python3 -I arm_timer.py --seconds S --receipt OUT.json [--term-grace 25] [--kill-grace 10] -- CMD [ARGS...]
- Child subreaper (PR_SET_CHILD_SUBREAPER): orphans of the arm are re-parented here, so the descendant set is the
  complete arm tree (parent links only; never a whole process group or session, which the session shares).
- E36/E215: the official GPU-lock fd (GPUWRF_GPU_LOCK_FD, normally 9) is passed to the child only after it is proven to
  be open on GPUWRF_GPU_LOCK_FILE in this process; nothing else is forwarded (close_fds).
- On the cap, on TERM/INT/HUP to this process, and for any descendant left behind after a normal exit: SIGTERM every
  descendant, wait --term-grace s (bundle_consumer.py tears its tree down in 15 s), then SIGKILL every survivor and reap
  until none is left. Receipt: rc, timed_out, signal, elapsed (CLOCK_BOOTTIME: a suspend counts, E60/E180), stragglers.
Exit: the command's rc; 124 cap reached; 128+N after signal N; 125 when a descendant survived SIGKILL.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def boot():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def table():
    rows = {}
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            v = p.read_text().rsplit(")", 1)[1].split()
            rows[int(p.parent.name)] = (v[0], int(v[1]))
        except (OSError, ValueError, IndexError):
            pass
    return rows


def descendants(root):
    rows, out = table(), set()
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for pid, (_, ppid) in rows.items():
            if ppid == parent and pid not in out:
                out.add(pid)
                frontier.append(pid)
    return {pid: rows[pid][0] for pid in out if pid in rows}


def reap_zombies(skip):
    for pid, state in descendants(os.getpid()).items():
        if state == "Z" and pid != skip:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def teardown(proc, term_grace, kill_grace):
    seen = set()
    for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, kill_grace)):
        end = boot() + grace
        while True:
            live = {p for p, s in descendants(os.getpid()).items() if s != "Z"}
            seen |= live
            if not live:
                break
            for p in live:
                try:
                    os.kill(p, sig)
                except ProcessLookupError:
                    pass
            proc.poll()
            reap_zombies(proc.pid)
            if boot() >= end:
                break
            time.sleep(0.1)
        if not {p for p, s in descendants(os.getpid()).items() if s != "Z"}:
            break
    proc.poll()
    reap_zombies(proc.pid)
    left = {p: s for p, s in descendants(os.getpid()).items() if s != "Z"}
    return dict(signalled=sorted(seen), left=sorted(left), all_reaped=not left)


def lock_fd():
    fd, path = os.environ.get("GPUWRF_GPU_LOCK_FD"), os.environ.get("GPUWRF_GPU_LOCK_FILE")
    if not fd or not path:
        return ()
    fd = int(fd)
    st, want = os.fstat(fd), os.stat(path)   # runtime identity check only; never stored in a receipt
    if (st.st_dev, st.st_ino) != (want.st_dev, want.st_ino):
        raise SystemExit(f"arm_timer: GPUWRF_GPU_LOCK_FD {fd} is not open on {path}")
    return (fd,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--term-grace", type=float, default=25.0)
    ap.add_argument("--kill-grace", type=float, default=10.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd or a.seconds <= 0:
        ap.error("positive --seconds and a command are required")
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot become child subreaper")
    stop = [0]
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda n, _f: stop.__setitem__(0, n))
    fds = lock_fd()
    t0, rec = boot(), dict(cmd=cmd, cap_s=a.seconds, lock_fd_passed=list(fds), pid=os.getpid(),
                           pgid=os.getpgrp(), start_unix=time.time())
    proc = subprocess.Popen(cmd, pass_fds=fds, start_new_session=False)
    rec["child_pid"] = proc.pid
    while proc.poll() is None and not stop[0] and boot() - t0 < a.seconds:
        reap_zombies(proc.pid)
        time.sleep(0.2)
    timed_out = proc.returncode is None and not stop[0]
    rec["teardown"] = teardown(proc, a.term_grace, a.kill_grace)
    if proc.returncode is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    orig = proc.returncode
    rc = 128 + stop[0] if stop[0] else 124 if timed_out else (128 - orig if orig is not None and orig < 0 else orig)
    if not rec["teardown"]["all_reaped"]:
        rc = 125
    rec.update(rc=rc, command_rc=orig, timed_out=timed_out, signal=stop[0], elapsed_s=round(boot() - t0, 3),
               end_unix=time.time(), stragglers_after_exit=bool(rec["teardown"]["signalled"]) and not timed_out and not stop[0])
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(rec, indent=1) + "\n")
    raise SystemExit(rc if rc is not None else 125)


if __name__ == "__main__":
    main()
