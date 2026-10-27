"""Backend-dark controls for the post-unlock CPU phase (launcher QA A1/A2/A3): no JAX, no GPU, no real children."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget')


class PostCpu(unittest.TestCase):
    def setUp(self):
        (LANE / 'pytest').mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='postcpu_', dir=LANE / 'pytest'))
        tools = self.root / 'tools'; tools.mkdir()
        shutil.copy(HERE / 'budget_post_cpu.py', tools)
        (tools / 'manifest.json').write_text(json.dumps(dict(model=str(self.root / 'model'), floor_protocol=str(self.root / 'floors'),
            component_cases=[[1, 'day'], [0, 'day']], cpu_cap_s=100)))
        self.run_dir = self.root / 'NIGHT_test'; (self.run_dir / 'probe/components/a').mkdir(parents=True)
        (self.run_dir / 'probe/components/b').mkdir()
        seg = lambda s: dict(wall_s=s, simulated_s=3600.)
        (self.run_dir / 'probe/result.json').write_text(json.dumps(dict(
            components={'a': dict(verdict='TWIN_COMPONENT_2STEP_BYTE'), 'b': dict(verdict='TWIN_COMPONENT_2STEP_BYTE')},
            phases={'replay_on': dict(segments=[seg(400.), seg(20.), seg(6.)]), 'replay_edmf0': dict(segments=[seg(300.), seg(5.), seg(5.5)])})))
        spec = importlib.util.spec_from_file_location('post_under_test', tools / 'budget_post_cpu.py')
        self.P = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.P)
        self.limits = []

    def tearDown(self):
        shutil.rmtree(self.root)

    def fake_run(self, command, timeout=None, **kwargs):
        self.limits.append((Path(command[1]).name if len(command) > 1 else command[0], timeout))
        if 'budget_cubin_gate.py' in ' '.join(map(str, command)):
            raise subprocess.TimeoutExpired(command, timeout)
        if 'twin_oracle.py' in ' '.join(map(str, command)):
            out = Path(command[command.index('--out') + 1]); out.mkdir(parents=True); (out / 'result.json').write_text('{}')
        return subprocess.CompletedProcess(command, 0)

    def call(self):
        argv = ['budget_post_cpu.py', str(self.run_dir / 'cache'), str(self.run_dir / 'e41'), '--run-proof', str(self.run_dir / 'probe')]
        with patch.object(self.P.subprocess, 'run', self.fake_run), patch('sys.argv', argv), \
             patch.dict(os.environ, {'JAX_PLATFORMS': 'cpu'}):
            try:
                return self.P.main()
            except subprocess.TimeoutExpired as error:   # QA A1: the old post crashed here and lost the rest
                self.fail(f'post phase crashed on an inner timeout: {error}')

    def test_e41_timeout_is_recorded_and_oracles_and_accounting_still_run(self):
        rc = self.call()
        result = json.loads((self.run_dir / 'post_cpu.json').read_text())
        self.assertEqual(rc, 2)
        self.assertEqual(result['e41_rc'], 'TIMEOUT')
        self.assertEqual({k: v['rc'] for k, v in result['oracles'].items()}, {'a': 0, 'b': 0})
        self.assertEqual(result['verdict'], 'CPU_GATES_FAIL_OR_INCOMPLETE')
        self.assertIn('replay_on', result['accounting'])

    def test_every_child_limit_fits_inside_the_post_cap(self):
        self.call()
        self.assertTrue(self.limits)
        for name, limit in self.limits:
            self.assertIsNotNone(limit, name)
            self.assertLessEqual(limit, 100 - 20 + 1e-6, name)   # cpu_cap_s 100 minus the 20 s margin

    def test_traced_segment_is_excluded_from_untraced_cost(self):
        self.call()
        acc = json.loads((self.run_dir / 'post_cpu.json').read_text())['accounting']
        self.assertEqual(acc['replay_on']['untraced_warm_segment_s_per_fc_h'], [6.])
        self.assertEqual(acc['replay_on']['traced_segment_index'], [1])
        self.assertEqual(acc['replay_edmf0']['untraced_warm_segment_s_per_fc_h'], [5., 5.5])

    def test_actual_outer_limit_from_the_supervisor_wins(self):
        with patch.dict(os.environ, {'BUDGET_POST_CAP_S': '60'}):
            self.call()
        for name, limit in self.limits:
            self.assertLessEqual(limit, 60 - 20 + 1e-6, name)



class PostAdmission(unittest.TestCase):
    """QA 01:21Z: the post is shortened when the GPU phase ran long, and skipped only below the admission threshold."""
    def setUp(self):
        spec = importlib.util.spec_from_file_location('common_under_test', HERE / 'night_common_v6.py')
        self.C = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.C)

    def test_production_long_gpu_phase_gets_a_shortened_post(self):
        self.assertEqual(self.C.post_limit(3000., 720, 1500), 1500)
        self.assertEqual(self.C.post_limit(1000., 720, 1500), 1000.)   # v10 would have skipped E41 + oracles here
        self.assertIsNone(self.C.post_limit(700., 720, 1500))

    def test_mini_threshold_follows_its_cpu_cap(self):
        self.assertEqual(self.C.post_limit(650., min(720, 360), 360), 360)


if __name__ == '__main__':
    unittest.main()
