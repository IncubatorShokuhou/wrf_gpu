"""CPU-only controls. No production lock, GPU command, CUDA import or model run."""
import ctypes
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('w44_night', HERE / 'night_supervisor.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class NightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='night_controls_', dir='<USER_HOME>/wrf_gpu2_lanes/w44-budget')
        self.root = Path(self.temp.name)
        M.SIGNALLED = 0
        self.assertEqual(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0), 0)

    def tearDown(self):
        self.temp.cleanup()

    def admission(self, start, end):
        env = {k: v for k, v in os.environ.items() if k not in ('GA_START_UTC', 'GA_END_UTC', 'GPUWRF_GPU_LOCK_HELD')}
        if start is not None:
            env['GA_START_UTC'] = start
        if end is not None:
            env['GA_END_UTC'] = end
        with patch.object(M, 'BASE', self.root), patch.object(M, 'real_utc', return_value=10000), \
             patch.dict(os.environ, env, clear=True), patch.object(M, 'run_group') as child:
            rc = M.main()
            child.assert_not_called()
        receipt = json.loads(next(self.root.glob('NIGHT_*/night_receipt.json')).read_text())
        self.assertFalse(receipt['gpu_started'])
        self.assertEqual(receipt['rc'], rc)
        return rc, receipt

    @staticmethod
    def stamp(epoch):
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()

    def test_missing_window_writes_refusal(self):
        self.assertEqual(self.admission(None, None)[0], 2)

    def test_short_window_never_takes_lock(self):
        self.assertEqual(self.admission(self.stamp(9990), self.stamp(10000 + M.CAP - 1))[0], 4)

    def test_early_window_never_takes_lock(self):
        self.assertEqual(self.admission(self.stamp(10001), self.stamp(20000))[0], 4)

    def test_naive_window_refuses(self):
        self.assertEqual(self.admission('2026-10-23T00:00:00', '2026-10-24T00:00:00')[0], 2)

    def test_one_second_busy_lock_rc_propagates_without_gpu(self):
        env = dict(os.environ, GA_START_UTC=self.stamp(9990), GA_END_UTC=self.stamp(15000))
        env.pop('GPUWRF_GPU_LOCK_HELD', None)
        row = dict(rc=124, cleanup=dict(signalled_pids=[], live_remaining=[]))
        with patch.object(M, 'BASE', self.root), patch.object(M, 'real_utc', return_value=10000), \
             patch.dict(os.environ, env, clear=True), patch.object(M, 'run_group', return_value=row) as child:
            self.assertEqual(M.main(), 124)
            command = child.call_args.args[0]
            self.assertEqual(command[1:6], ['--label', 'w44-budget', '--timeout', '1', '--'])
            self.assertEqual(child.call_count, 1)
        r = json.loads(next(self.root.glob('NIGHT_*/night_receipt.json')).read_text())
        self.assertFalse(r['gpu_started'])
        self.assertEqual(r['rc'], 124)

    def test_signal_cancels_child_group(self):
        M.SIGNALLED = signal.SIGTERM
        row = M.run_group([sys.executable, '-c', 'import time;time.sleep(30)'], self.root / 'signal.log', 5)
        self.assertEqual(row['rc'], 143)
        self.assertFalse(row['timed_out'])
        self.assertEqual(row['cleanup']['live_remaining'], [])
        M.SIGNALLED = 0

    def test_child_rc_and_private_group(self):
        log = self.root / 'child.log'
        row = M.run_group([sys.executable, '-c', 'import os,sys;print(os.getpid(),os.getpgrp(),os.getsid(0));sys.exit(7)'], log, 3)
        self.assertEqual(row['rc'], 7)
        self.assertEqual(list(map(int, log.read_text().split())), [row['pid']] * 3)
        self.assertEqual(row['cleanup']['live_remaining'], [])

    def test_timeout_kills_escaping_descendant_and_releases_dummy_flock(self):
        # Grandchild creates another SESSION, ignores TERM and inherits a flock.
        # Parent is killed on timeout; subreaper must adopt and kill grandchild.
        code = '''import fcntl,os,signal,time
f=open("LOCKPATH","w");fcntl.flock(f,fcntl.LOCK_EX)
pid=os.fork()
if pid==0:
 os.setsid();signal.signal(signal.SIGTERM,signal.SIG_IGN)
 open("PIDPATH","w").write(str(os.getpid()))
 while True:time.sleep(.1)
while True:time.sleep(.1)
'''.replace('LOCKPATH', str(self.root / 'dummy.lock')).replace('PIDPATH', str(self.root / 'grandchild.pid'))
        row = M.run_group([sys.executable, '-c', code], self.root / 'timeout.log', .5)
        self.assertEqual(row['rc'], 124)
        self.assertTrue(row['timed_out'])
        self.assertEqual(row['cleanup']['live_remaining'], [])
        pid = int((self.root / 'grandchild.pid').read_text())
        self.assertFalse(Path('/proc', str(pid)).exists())
        with open(self.root / 'dummy.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_successful_parent_cannot_leave_orphan(self):
        code = 'import os,time; pid=os.fork(); time.sleep(30) if pid==0 else None'
        row = M.run_group([sys.executable, '-c', code], self.root / 'orphan.log', 2)
        self.assertEqual(row['rc'], 125)
        self.assertEqual(row['cleanup']['live_remaining'], [])

    def test_gpu_body_has_no_nested_timeout(self):
        self.assertNotIn('timeout ', M.GPU_BODY)
        self.assertIn('exec env -u JAX_PLATFORMS', M.GPU_BODY)


if __name__ == '__main__':
    unittest.main()
