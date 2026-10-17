"""Exercise the real collection hooks in isolated, tiny pytest sessions."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('args,code,summary', [
    ([], 0, '1 passed, 1 skipped'),
    (['-m', 'heavy'], 0, '1 skipped, 1 deselected'),
    (['--run-heavy'], 1, '1 failed, 1 passed'),
    (['--run-heavy', '-m', 'heavy'], 1, '1 failed, 1 deselected'),
    (['--run-heavy', '-m', 'not heavy'], 0, '1 passed, 1 deselected'),
])
def test_heavy_requires_explicit_opt_in(tmp_path, args, code, summary):
    (tmp_path / 'conftest.py').write_text(
        'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("suite_hooks", {str(ROOT / "tests/conftest.py")!r})\n'
        'hooks = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(hooks)\n'
        'pytest_addoption = hooks.pytest_addoption\n'
        'pytest_collection_modifyitems = hooks.pytest_collection_modifyitems\n'
    )
    path = tmp_path / 'test_tier.py'
    path.write_text('import pytest\n'
                    'def test_ordinary(): pass\n'
                    '@pytest.mark.heavy\n'
                    'def test_heavy(): assert False, "heavy sentinel executed"\n')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-rs', '-c', str(ROOT / 'pyproject.toml'),
         '--confcutdir', str(tmp_path), '--basetemp', str(tmp_path / 'child-base'),
         str(path), *args], cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert summary in result.stdout, result.stdout + result.stderr
    if '--run-heavy' not in args:
        assert 'heavy CPU test needs ~17 GiB; run with --run-heavy in a 24G RAM slot' in result.stdout
    if code == 1:
        assert 'AssertionError: heavy sentinel executed' in result.stdout
        assert 'ERROR at setup' not in result.stdout
