"""Deletion checks: only assertion failures count as mutant kills (E202)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root = Path(os.environ.get('W44_PRECOMPILE_TEST_SOURCE', Path(__file__).resolve().parents[2]))
out = Path('<USER_HOME>/wrf_gpu2_lanes/w44-precompile/mutants')
out.mkdir(parents=True, exist_ok=True)
source = (root/'src/gpuwrf/runtime/precompile_cli.py').read_text()
cases = [
 ('cache_integrity', '    return all((cache / name).is_relative_to(cache)',
  '    return True\n    return all((cache / name).is_relative_to(cache)',
  'test_cache_requires_complete_identical_native_artifacts'),
 ('namelist_binding', 'shutil.copyfile(namelist, view / "namelist.input")',
  'shutil.copyfile(input_dir / "namelist.input", view / "namelist.input")',
  'test_namelist_is_bound_to_actual_case'),
 ('lock_fd_inheritance', 'pass_fds=(fd,), start_new_session=True', 'start_new_session=True',
  'test_timeout_reaps_own_process_group'),
 ('external_tables', '"tables_sha256": digest(tables)', '"tables_sha256": "ignored"',
  'test_external_runtime_tables_invalidate_receipts'),
]
receipts = []
for name, old, new, test in cases:
    assert source.count(old) == 1, (name, source.count(old))
    with tempfile.TemporaryDirectory(prefix=name+'-', dir=out) as temporary:
        directory = Path(temporary)
        changed = source.replace(old, new)
        plugin = directory / 'mutation_plugin.py'
        plugin.write_text('import cpu_guard\nimport gpuwrf.runtime.precompile_cli as pc\n'
                          f'exec(compile({changed!r}, pc.__file__, "exec"), pc.__dict__)\n')
        env = dict(os.environ, GPUWRF_JAX_CACHE='0',
                   PYTHONPATH=os.pathsep.join((str(directory), str(root/'scripts/v025'), str(root/'src'))))
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'mutation_plugin',
                   f'tests/test_precompile_cli.py::{test}', '--basetemp',
                   '<USER_HOME>/wrf_gpu2_lanes/w44-precompile/pytest']
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
        log = result.stdout + result.stderr
        (out/f'{name}.log').write_text(log)
        killed = result.returncode == 1 and 'AssertionError' in log and 'ERROR at setup' not in log and 'ERROR collecting' not in log
        receipts.append(dict(name=name, returncode=result.returncode, killed=killed, test=test))
(out/'receipt.json').write_text(json.dumps(receipts, indent=2)+'\n')
assert all(r['killed'] for r in receipts), receipts
