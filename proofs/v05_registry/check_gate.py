#!/usr/bin/env python3
"""CPU gate on an immutable source-only git archive; private storage only."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu', 'CPU only'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'QUIET: retry later'
    args.out.mkdir(parents=True, exist_ok=True)
    helper = args.source / 'scripts/alisios_registry.py'
    spec = importlib.util.spec_from_file_location('gate_registry', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reg = module.load_register()
    stores = [module.PRODUCTION_ATLAS / name for name in (reg.rl.WRF_REGISTRY_NAME, reg.rl.DATA_REGISTRY_NAME)]
    before = {str(p): sha(p) for p in stores}
    report = dict(commit=args.commit, source_sha256={str(p.relative_to(args.source)): sha(p) for p in
                  [helper, helper.with_name('alisios_registry_launcher_example.py'),
                   args.source / 'tests/test_alisios_registry.py',
                   args.source / 'proofs/v05_registry/check_gate.py']},
                  register_sha256=sha(module.REGISTER), registry_lib_sha256=sha(Path(reg.rl.__file__)),
                  production_before=before, jax_imported=False, model_executed=False)
    try:
        with tempfile.TemporaryDirectory(prefix='gate-', dir=args.out) as directory:
            private = Path(directory)
            env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
                       V05_REGISTRY_TEST_ROOT=str(private / 'test_tmp'))
            env.pop('V05_REGISTRY_HELPER', None)
            test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                                   str(args.source / 'tests'), '-p', 'test_alisios_registry.py', '-v'],
                                  env=env, capture_output=True, text=True, timeout=60)
            (args.out / 'tests.log').write_text(test.stdout + test.stderr)
            assert test.returncode == 0 and 'Ran 28 tests' in test.stderr and 'OK' in test.stderr, test.stderr
            report['tests'] = dict(passed=28, failures=0, errors=0, skipped=0)
            mutations = [
                ('remove_finally_end', '            self.end(terminal)', '            pass  # removed finally end',
                 ['test_success_finally_end_is_clean', 'test_crash_finally_writes_failed_end']),
                ('remove_schema_precheck', '            rc = self._schema_check(argv)', '            rc = 0',
                 ['test_precheck_refuses_before_storage_transport']),
            ]
            report['mutations'] = []
            text = helper.read_text()
            for name, old, new, selectors in mutations:
                assert text.count(old) == 1, ('literal mutation must be unique', name)
                mutant = private / (name + '.py')
                mutant.write_text(text.replace(old, new))
                mutant_env = dict(env, V05_REGISTRY_HELPER=str(mutant))
                command = [sys.executable, '-m', 'unittest', '-v'] + [
                    'test_alisios_registry.RegistryTests.' + selector for selector in selectors]
                result = subprocess.run(command, env=mutant_env, cwd=args.source / 'tests',
                                        capture_output=True, text=True, timeout=30)
                log = result.stdout + result.stderr
                (args.out / (name + '.log')).write_text(log)
                assert result.returncode == 1 and 'AssertionError' in log and '\nERROR:' not in log, log
                assert f'FAILED (failures={len(selectors)})' in log, log
                report['mutations'].append(dict(name=name, failures=len(selectors), errors=0,
                                               assertion_failure=True, mutant_sha256=sha(mutant)))
            # Exercise a private COPY of current production state, preserving
            # its prefix bytes; never copy unrelated ATLAS data or extracts.
            atlas = private / 'seeded_atlas'
            atlas.mkdir()
            for source in stores:
                if source.exists():
                    shutil.copyfile(source, atlas / source.name)
            original = (atlas / reg.rl.WRF_REGISTRY_NAME).read_bytes()
            registry = module.Registry(atlas_dir=atlas, smoke=True)
            samples = []
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for index in range(5):
                    start = dict(run_id='v05-registry:private:' + uuid.uuid4().hex,
                                 by='v05-registry', producer='v05-registry', purpose='PRIVATE hook gate NO_SCORE',
                                 family='registry_hook_gate', forcing='WN3', engine='GPU wrf_gpu gate (no model)',
                                 nests='9/3/1 km', lead_h=78, issue='20260227_18z',
                                 run_dir=str(private / ('model' + str(index))), retention='raw_delete_after_extract')
                    t0 = time.perf_counter()
                    with registry.run(start, end=dict(corpus='held', raw_deleted=False)) as outcome:
                        outcome.rc = 0
                    samples.append(time.perf_counter() - t0)
            updated = (atlas / reg.rl.WRF_REGISTRY_NAME).read_bytes()
            assert updated.startswith(original)
            added = [json.loads(s) for s in updated[len(original):].splitlines()]
            assert len(added) == 10 and [e['event'] for e in added] == ['start', 'end'] * 5
            report['seeded_copy'] = dict(seed_bytes=len(original), private_events=10, prior_bytes_exact=True)
            report['bench'] = dict(scope='host hooks only; private copy of production state; no forecast',
                                  start_end_seconds=samples, median_seconds=statistics.median(samples),
                                  child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                                  driver_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                  compile_count=0, gpu_commands=0, timestep_ops_added=0)
            assert 'jax' not in sys.modules
            report['status'] = 'PASS'
    finally:
        report['production_after'] = {str(p): sha(p) for p in stores}
        report['production_bytes_exact'] = report['production_after'] == before
        report['private_atlases_and_mutants_removed'] = True
        (args.out / 'GATE.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['production_bytes_exact'], 'Production stores changed: reconcile external writers'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
