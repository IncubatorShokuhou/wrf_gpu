"""Delete the lakeini top-temperature assignment and run the real parity gate.

In-process source mutant only. Original numerical fixture and assertion are
unchanged; accept a kill only on AssertionError with zero setup errors.
"""
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo / "scripts/v025"))
import cpu_guard  # noqa: E402,F401 -- must precede JAX/gpuwrf for pytest

import gpuwrf.physics.lake_initialization as module
import pytest

source = Path(module.__file__).read_text()
needle = "    temp = temp.at[..., 0].set(skin)\n"
assert source.count(needle) == 1
exec(compile(source.replace(needle, ""), module.__file__, "exec"), module.__dict__)
raise SystemExit(pytest.main([
    str(repo / "tests/test_s1_lake_initialization.py") + "::test_lakeini_pristine_all_fields[open_deep]",
    "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/s1-lake/pytest", "-q", "--tb=short",
]))
