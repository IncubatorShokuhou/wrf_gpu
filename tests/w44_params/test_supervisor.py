"""Cancel only the outer process group; nested/escaped children must release the lock."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SUPERVISOR = Path(__file__).resolve().parents[2] / 'proofs/w44_params/supervise.py'


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def outer_cancel(tmp_path, *, ram_scope):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600)
fcntl.flock(fd,fcntl.LOCK_EX)
code="import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGHUP,signal.SIG_IGN); time.sleep(60)"
sampler=subprocess.Popen([sys.executable,'-c',code],pass_fds=(fd,),start_new_session=True)
Path(sys.argv[2]).write_text(json.dumps({'worker':os.getpid(),'sampler':sampler.pid}))
time.sleep(60)
''')
    lock, receipt = tmp_path / 'dummy.lock', tmp_path / 'pids.json'
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'GPUWRF_GPU_LOCK_HELD': '0'}
    payload = ['timeout', '20', sys.executable, str(worker), str(lock), str(receipt)]
    if ram_scope:
        payload = [str(SUPERVISOR.parents[2] / 'scripts/with_ram_slot.sh'), '--label',
                   'w44-params-cancel-fixture', '--max', '512M', '--timeout', '10', '--', *payload]
    command = [sys.executable, str(SUPERVISOR), '--whole-arm', '--seconds', '45', '--', *payload]
    process = subprocess.Popen(command, start_new_session=True, env=env)
    pids = {}
    try:
        limit = time.monotonic() + 18
        while not receipt.exists() and process.poll() is None and time.monotonic() < limit:
            time.sleep(.02)
        assert receipt.exists(), 'fixture did not reach the nested lock holder'
        pids = json.loads(receipt.read_text())
        time.sleep(.1)  # Let the sampler install its TERM-ignoring handler.
        assert all(alive(pid) for pid in pids.values())
        os.killpg(process.pid, signal.SIGTERM)  # Deliberately no direct child signalling here.
        assert process.wait(timeout=10) == 143
        assert not any(alive(pid) for pid in pids.values()), pids
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        # A failing control must not leak its own fixture workers.
        for pid in pids.values():
            if alive(pid):
                os.kill(pid, signal.SIGKILL)


def test_outer_term_reaps_nested_timeout_and_escaped_lock_holder(tmp_path):
    outer_cancel(tmp_path, ram_scope=False)


def test_outer_term_also_reaps_actual_ram_scope_payload(tmp_path):
    # The payload takes a ticket in the SHARED RAM-slot FIFO (scripts/with_ram_slot.sh, fixed /tmp queue). With other
    # tickets waiting it cannot get a slot inside its 10 s and must not jump the queue; during a GPU window no new
    # tickets are allowed at all (_WAVE_V044_COMMON). Skip loudly instead of failing on host load (critic A2).
    queue = Path('/tmp/wrf_gpu2_ram_slots/queue')
    if Path('/tmp/wrf_gpu2_gpu_window').exists() or (queue.is_dir() and any(queue.iterdir())):
        pytest.skip('shared RAM-slot queue busy or GPU window open: RAM-scope cancellation control not runnable now')
    outer_cancel(tmp_path, ram_scope=True)


def test_expired_window_does_not_spawn_the_command(tmp_path):
    marker = tmp_path / 'spawned'
    command = [sys.executable, str(SUPERVISOR), '--whole-arm', '--seconds', '10',
               '--deadline-epoch', '1', '--', sys.executable, '-c',
               'from pathlib import Path; Path(' + repr(str(marker)) + ').touch()']
    result = subprocess.run(command, env={**os.environ, 'JAX_PLATFORMS': 'cpu',
                            'GPUWRF_GPU_LOCK_HELD': '0'}, timeout=10)
    assert result.returncode == 124 and not marker.exists()
