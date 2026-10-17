"""Delete the admission predicate in memory; require genuine assertion kills."""
import inspect
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts/v025"))
import cpu_guard  # CPU protection must precede gpuwrf/JAX, like tests/conftest.py.
import pytest
from gpuwrf.io import cammg_config

source = inspect.getsource(cammg_config.selection_errors)
needle = 'if (backend or target_backend()) != "cpu" and any(v == 11 for v in values):'
assert source.count(needle) == 1
exec(source.replace(needle, "if False:"), cammg_config.__dict__)
raise SystemExit(pytest.main([
    str(root / "tests/test_cammg_gpu_admission.py"),
    "-q", "-k", "gpu_selection_refused or cpu_and_other_microphysics_admitted",
    "--basetemp", sys.argv[1], "--junitxml", sys.argv[2],
]))
