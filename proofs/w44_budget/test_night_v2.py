"""CPU admission/control checks; never call the official lock or model."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('w44_v2',HERE/'night_supervisor_v2.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)
spec = importlib.util.spec_from_file_location('w44_probe',HERE/'night_probe_v2.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


class Controls(unittest.TestCase):
    def test_paid_source_and_first_step_proof(self):
        r = P.prereq()
        self.assertEqual(r['byte_fields'],183)
        self.assertEqual(r['total_fields'],183)
        self.assertEqual(r['platform'],'gpu')

    def test_e41_environment_is_pinned_and_cpu_only(self):
        self.assertEqual(N.M.CPU_ENV['JAX_PLATFORMS'],'cpu')
        self.assertEqual(N.M.CPU_ENV['PYTHONPATH'],str(N.M.SNAP/'src'))
        self.assertEqual(N.M.CPU_ENV['GPUWRF_AOT_KEEP_TRACE_METADATA'],'1')

    def test_one_shared_paid_cache_and_no_nested_timeout(self):
        self.assertEqual(N.M.CAP,3600)
        self.assertEqual(N.M.GPU_CAP,3000)
        self.assertIn('GPUWRF_JAX_CACHE_DIR='+N.PAID+'/cache/jax',N.M.GPU_BODY)
        self.assertIn('JAX_COMPILATION_CACHE_DIR='+N.PAID+'/cache/jax',N.M.GPU_BODY)
        self.assertNotIn('timeout ',N.M.GPU_BODY)

    def test_e41_requires_actual_priming_program_inventory(self):
        with patch.object(N,'run_group',return_value={'rc':0}) as child:
            N.run_group_v2(['scan','cache','out'],Path('/job/e41.log'),480,env=N.M.CPU_ENV)
            command=child.call_args.args[0]
            self.assertEqual(command[-2:],['--run-proof','/job/probe/compiled_program_inventory.json'])

    def test_actual_v2_admission_rejects_short_window_before_lock(self):
        with tempfile.TemporaryDirectory(prefix='v2_control_',dir='<USER_HOME>/wrf_gpu2_lanes/w44-budget') as work:
            env=dict(os.environ,GA_START_UTC='1970-01-01T02:40:00Z',GA_END_UTC='1970-01-01T03:46:39Z')
            with patch.object(N.M,'BASE',Path(work)),patch.object(N.M,'real_utc',return_value=10000), \
                 patch.dict(os.environ,env,clear=True),patch.object(N.M,'run_group') as child:
                self.assertEqual(N.M.main(),4)
                child.assert_not_called()
            r=json.loads(next(Path(work).glob('NIGHT_*/night_receipt.json')).read_text())
            self.assertEqual(r['protocol_version'],2)
            self.assertEqual(r['cap_s'],3600)
            self.assertFalse(r['gpu_started'])
            self.assertEqual(r['phases'],{})


if __name__=='__main__':
    unittest.main()
