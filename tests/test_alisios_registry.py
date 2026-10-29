"""Host integration tests: REAL ALISIOS schema, state checks, flock and append.

Run with pytest under the root conftest, or unittest. All stores, including
the immutable stand-in for production, are PRIVATE fixtures; no real ATLAS reads.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get('V05_REGISTRY_HELPER', Path(__file__).resolve().parents[1] / 'scripts/alisios_registry.py'))
spec = importlib.util.spec_from_file_location('tested_alisios_registry', SOURCE)
hooks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hooks
spec.loader.exec_module(hooks)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get('V05_REGISTRY_TEST_ROOT', '<USER_HOME>/wrf_gpu2_lanes/v05-registry/test_tmp'))
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.sentinel = tempfile.TemporaryDirectory(dir=cls.root)
        cls.addClassCleanup(cls.sentinel.cleanup)
        cls.private_production = Path(cls.sentinel.name) / 'production-fixture'
        cls.private_production.mkdir()
        cls.real_paths = [cls.private_production / name for name in
                          ('WRF_RUN_REGISTRY.jsonl', 'DATA_REGISTRY.jsonl')]
        for path in cls.real_paths:
            path.write_text('{"fixture":"immutable private sentinel"}\n')
        cls.real_before = [digest(p) for p in cls.real_paths]
        private_production = patch.object(hooks, 'PRODUCTION_ATLAS', cls.private_production)
        private_production.start()
        cls.addClassCleanup(private_production.stop)

    @classmethod
    def tearDownClass(cls):
        # Preserve the unchanged-store assertion on a PRIVATE production stand-in.
        if [digest(p) for p in cls.real_paths] != cls.real_before:
            raise AssertionError('Private production sentinel changed during private tests')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.atlas = self.base / 'atlas'
        self.reg = hooks.Registry(atlas_dir=self.atlas, smoke=True)
        self.start = dict(run_id='v05-registry:test:20260227_18z', by='v05-registry',
                          producer='v05-registry', purpose='Private hook integration; NO_SCORE',
                          family='registry_unit', forcing='WN3', engine='GPU wrf_gpu test',
                          nests='9/3/1 km', lead_h=78, issue='20260227_18z',
                          run_dir=str(self.base / 'model'), retention='raw_delete_after_extract')
        self.end = dict(run_id=self.start['run_id'], by='v05-registry', rc=0,
                        status='DONE_CLEAN', corpus='held', raw_deleted=True)
        self.output = io.StringIO()
        capture = contextlib.ExitStack()
        capture.enter_context(contextlib.redirect_stdout(self.output))
        capture.enter_context(contextlib.redirect_stderr(self.output))
        self.addCleanup(capture.close)

    def events(self):
        return self.reg._events()

    def metadata(self, value):
        path = self.base / 'metadata.json'
        path.write_text(json.dumps(value))
        return path

    def cli(self, action, value):
        return hooks.main([action, str(self.metadata(value)), '--atlas-dir', str(self.atlas), '--smoke'])

    def test_real_main_start_end_rc_zero_and_disk_events(self):
        self.assertEqual(self.cli('start', self.start), 0)
        self.assertEqual(self.cli('end', self.end), 0)
        events = self.events()
        self.assertEqual([e['event'] for e in events], ['start', 'end'])
        self.assertTrue(all(e['schema'] == 'alisios.wrf_run_registry.v1' for e in events))
        self.assertEqual(events[-1]['status'], 'DONE_CLEAN')
        self.assertTrue((self.atlas / 'WRF_RUN_REGISTRY.jsonl.lock').exists())

    def test_bad_forcing_rc_two_no_append(self):
        self.assertEqual(self.cli('start', dict(self.start, forcing='ERA5 v043 E2W300I')), 2)
        self.assertEqual(self.events(), [])
        self.assertIn('forcing:', self.output.getvalue())

    def test_missing_issue_rc_two_no_append(self):
        start = dict(self.start)
        del start['issue']
        self.assertEqual(self.cli('start', start), 2)
        self.assertEqual(self.events(), [])
        self.assertIn("'issue'", self.output.getvalue())

    def test_all_forcing_enums_and_combinations_real_schema(self):
        for forcing in ['WN3', 'WN2', 'AIFS', 'ERA5', 'GFS', 'IFS', 'HARMONIE', 'OTHER', 'WN3+AIFS3D']:
            with self.subTest(forcing=forcing):
                self.reg.start(dict(self.start, run_id=self.start['run_id'] + ':' + forcing, forcing=forcing))
        self.assertEqual(len(self.events()), 9)

    def test_duplicate_start_refused_by_real_state_check(self):
        self.reg.start(self.start)
        before = (self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes()
        self.assertEqual(self.cli('start', self.start), 2)
        self.assertEqual((self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes(), before)
        self.assertIn('bereits registriert', self.output.getvalue())

    def test_duplicate_end_refused_by_real_state_check(self):
        self.reg.start(self.start)
        self.reg.end(self.end)
        before = (self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes()
        self.assertEqual(self.cli('end', self.end), 2)
        self.assertEqual((self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes(), before)
        self.assertIn('schon beendet', self.output.getvalue())

    def test_end_without_start_refused(self):
        self.assertEqual(self.cli('end', self.end), 2)
        self.assertEqual(self.events(), [])
        self.assertIn('keinen Start-Eintrag', self.output.getvalue())

    def test_precheck_refuses_before_storage_transport(self):
        # The transport is a boundary spy, not a fake validator. Removing the
        # actual pre-check must reach this spy and fail this assertion.
        with patch.object(self.reg, '_call', return_value=0) as transport:
            try:
                self.reg.start(dict(self.start, forcing='ERA5 v043 E2W300I'))
            except hooks.RegistryError as exc:
                self.assertEqual(exc.rc, 2)
            self.assertEqual(transport.call_count, 0, 'schema precheck bypassed: storage transport called')

    def test_success_finally_end_is_clean(self):
        with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=True)) as outcome:
            self.assertEqual([e['event'] for e in self.events()], ['start'])
            outcome.rc = 0
        self.assertEqual([e['event'] for e in self.events()], ['start', 'end'], 'finally-end missing')
        self.assertEqual(self.events()[-1]['status'], 'DONE_CLEAN')

    def test_success_with_retained_raw_is_held(self):
        with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)) as outcome:
            outcome.rc = 0
        self.assertEqual(self.events()[-1]['status'], 'DONE_HELD')

    def test_crash_finally_writes_failed_end(self):
        with self.assertRaisesRegex(RuntimeError, 'model crashed'):
            with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)):
                raise RuntimeError('model crashed')
        self.assertEqual([e['event'] for e in self.events()], ['start', 'end'], 'finally-end missing on crash')
        self.assertEqual(self.events()[-1]['status'], 'FAILED')
        self.assertEqual(self.events()[-1]['rc'], 1)

    def test_child_nonzero_status_and_rc_preserved(self):
        with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)) as outcome:
            outcome.rc = 37
        self.assertEqual((self.events()[-1]['status'], self.events()[-1]['rc']), ('FAILED', 37))

    def test_called_process_error_rc_preserved(self):
        with self.assertRaises(subprocess.CalledProcessError):
            with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)):
                raise subprocess.CalledProcessError(42, ['model'])
        self.assertEqual((self.events()[-1]['status'], self.events()[-1]['rc']), ('FAILED', 42))

    def test_keyboard_interrupt_is_aborted(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)):
                raise KeyboardInterrupt()
        self.assertEqual((self.events()[-1]['status'], self.events()[-1]['rc']), ('ABORTED', 130))

    def test_no_end_if_start_refused(self):
        with self.assertRaises(hooks.RegistryError):
            with self.reg.run(dict(self.start, forcing='INVALID'), end=dict(corpus='held', raw_deleted=False)):
                self.fail('model must not run after a rejected start')
        self.assertEqual(self.events(), [])

    def test_bad_terminal_metadata_refused_before_model(self):
        with self.assertRaises(hooks.RegistryError):
            with self.reg.run(self.start, end=dict(raw_deleted=False)) as outcome:
                self.fail('model must not run with malformed terminal metadata')
        self.assertEqual(self.events(), [])

    def test_end_storage_failure_is_not_silenced(self):
        with patch.object(self.reg, '_call', side_effect=[0, 2]):
            with self.assertRaises(hooks.RegistryError):
                with self.reg.run(self.start, end=dict(corpus='held', raw_deleted=False)) as outcome:
                    outcome.rc = 0

    def test_smoke_and_live_use_same_exact_argv(self):
        observed = []
        real_main = self.reg.reg.main
        def record(argv):
            observed.append((tuple(argv), os.environ['ALISIOS_ATLAS_DIR']))
            return real_main(argv)
        live = hooks.Registry(atlas_dir=self.base / 'live-private')
        with patch.object(self.reg.reg, 'main', record):
            for registry in (self.reg, live):
                registry.start(self.start)
                registry.end(self.end)
        self.assertEqual([v[0] for v in observed[:4]], [v[0] for v in observed[4:]])
        self.assertEqual([v[1] for v in observed[:4]], [str(self.atlas)] * 4)
        self.assertNotEqual(observed[0][1], observed[4][1])

    def test_seq_smoke_requires_private_atlas_even_without_smoke_flag(self):
        with patch.dict(os.environ, {'WRF_GPU2_SEQ_SMOKE': '1'}):
            with self.assertRaises(hooks.RegistryError):
                hooks.Registry()
            for unsafe in [hooks.PRODUCTION_ATLAS, hooks.PRODUCTION_ATLAS.parent, hooks.PRODUCTION_ATLAS / 'child']:
                with self.assertRaises(hooks.RegistryError):
                    hooks.Registry(atlas_dir=unsafe)

    def test_smoke_rejects_symlinked_file_before_append(self):
        target = self.base / 'outside.jsonl'
        target.write_text('')
        self.atlas.mkdir()
        (self.atlas / 'WRF_RUN_REGISTRY.jsonl').symlink_to(target)
        with self.assertRaises(hooks.RegistryError):
            self.reg.start(self.start)
        self.assertEqual(target.read_bytes(), b'')

    def test_update_requires_authorization_and_never_duplicates_end(self):
        self.reg.start(self.start)
        self.reg.end(dict(self.end, rc=1, status='FAILED', raw_deleted=False))
        with self.assertRaises(hooks.RegistryError):
            self.reg.update(self.end, authorization='')
        self.reg.update(self.end, authorization='manager MAILBOX approved CPU post repair')
        self.assertEqual([e['event'] for e in self.events()], ['start', 'end', 'update'])
        self.assertEqual(self.reg.reg.rl.fold_wrf(self.events())[self.start['run_id']]['status'], 'DONE_CLEAN')
        self.assertIn('Authorized repair:', self.events()[-1]['notes'])

    def test_update_cannot_reopen_ended_run(self):
        self.reg.start(self.start)
        self.reg.end(self.end)
        with self.assertRaises(hooks.RegistryError):
            self.reg.update(dict(run_id=self.start['run_id'], by='v05-registry', status='RUNNING'),
                            authorization='test owner decision')
        self.assertEqual(len(self.events()), 2)

    def test_exact_launcher_template_runs_real_gate_with_private_child_env(self):
        template = SOURCE.with_name('alisios_registry_launcher_example.py')
        start_file = self.base / 'start.json'
        end_file = self.base / 'end.json'
        start_file.write_text(json.dumps(self.start))
        end_file.write_text(json.dumps(dict(corpus='held', raw_deleted=False)))
        env = dict(os.environ, JAX_PLATFORMS='cpu', WRF_GPU2_SEQ_SMOKE='1',
                   REGISTRY_PRIVATE_ATLAS=str(self.atlas), PYTHONDONTWRITEBYTECODE='1')
        child = [sys.executable, str(hooks.REGISTER), 'wrf', 'check', '--run-dir', self.start['run_dir']]
        # Execute the EXACT template, redirecting only its production-path sentinel
        # before import. The real register.py child still checks our private atlas.
        shim = ("import sys,runpy; sys.path.insert(0, sys.argv[1]); import alisios_registry as h; "
                "from pathlib import Path; h.PRODUCTION_ATLAS=Path(sys.argv[2]); "
                "sys.argv=sys.argv[3:]; runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, '-c', shim, str(SOURCE.parent), str(self.private_production),
                                 str(template), str(start_file), str(end_file), *child],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([e['event'] for e in self.events()], ['start', 'end'])
        self.assertEqual((self.events()[-1]['rc'], self.events()[-1]['status']), (0, 'DONE_HELD'))
        self.assertEqual(len(template.read_text().splitlines()), 20)

    def receipt(self):
        evidence = self.base / 'MODEL_RECEIPT.json'
        evidence.write_text('{"model_rc":0,"raw_deleted":true}\n')
        return dict(start=dict(self.start, started_utc='2026-10-26T13:00:00Z'),
                    end=dict(self.end, ended_utc='2026-10-26T14:00:00Z'),
                    evidence={str(evidence): digest(evidence)})

    def test_backfill_is_idempotent_and_preserves_timestamps(self):
        receipt = self.receipt()
        self.assertEqual(self.reg.backfill(receipt), 'recorded')
        before = (self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes()
        self.assertEqual(self.reg.backfill(receipt), 'already-recorded')
        self.assertEqual((self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes(), before)
        self.assertEqual(self.events()[0]['started_utc'], receipt['start']['started_utc'])
        self.assertEqual(self.events()[1]['ended_utc'], receipt['end']['ended_utc'])

    def test_backfill_repairs_partial_start(self):
        receipt = self.receipt()
        self.reg.start(receipt['start'])
        self.assertEqual(self.reg.backfill(receipt), 'recorded')
        self.assertEqual([e['event'] for e in self.events()], ['start', 'end'])

    def test_backfill_conflicts_refused(self):
        receipt = self.receipt()
        self.reg.backfill(receipt)
        before = (self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes()
        for variant in [dict(receipt, start=dict(receipt['start'], forcing='ERA5')),
                        dict(receipt, end=dict(receipt['end'], rc=1, status='FAILED'))]:
            with self.assertRaises(hooks.RegistryError):
                self.reg.backfill(variant)
        self.assertEqual((self.atlas / 'WRF_RUN_REGISTRY.jsonl').read_bytes(), before)

    def test_backfill_checks_both_schemas_before_start(self):
        receipt = self.receipt()
        del receipt['end']['rc']
        with self.assertRaises(hooks.RegistryError):
            self.reg.backfill(receipt)
        self.assertEqual(self.events(), [])

    def test_backfill_requires_known_end_and_pinned_evidence(self):
        receipt = self.receipt()
        for variant in [dict(receipt, evidence={}), dict(receipt, end=dict(receipt['end'], ended_utc=None)),
                        dict(receipt, evidence={next(iter(receipt['evidence'])): '0' * 64})]:
            with self.assertRaises(hooks.RegistryError):
                self.reg.backfill(variant)
        self.assertEqual(self.events(), [])


if __name__ == '__main__':
    unittest.main()
