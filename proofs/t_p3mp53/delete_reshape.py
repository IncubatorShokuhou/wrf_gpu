"""Delete the actual MP53 reshape branch in-process; require an assertion kill."""
import inspect
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts/v025'))
import cpu_guard  # noqa: E402,F401; before JAX, as in tests/conftest.py
import gpuwrf.physics.p3.process_support as support
import pytest

source = inspect.getsource(support._expand)
needle = 'if MP53_PREDICATE_RESHAPE.get() and value.dtype == jnp.bool_:'
assert source.count(needle) == 1
exec(compile(source.replace(needle, 'if False:'), support.__file__, 'exec'),
     support.__dict__)
sys.exit(pytest.main([str(root / 'tests/test_p3_mp53_compile.py'),
                     '-k', 'predicate_singleton', '-q', '--basetemp', sys.argv[1],
                     '--junitxml=' + sys.argv[2]]))
