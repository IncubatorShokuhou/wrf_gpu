"""Remove actual LSM exclusion; require the unchanged partition assertion kill."""
from pathlib import Path
import sys
repo=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(repo/'scripts/v025'))
import cpu_guard  # noqa: E402,F401
import gpuwrf.coupling.lake_surface_hook as module
import pytest

source=Path(module.__file__).read_text()
needle='    if not bundle.indices:\n'
assert source.count(needle)==2
exec(compile(source.replace(needle,'    if True:\n',1),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main([str(repo/'tests/test_s1_lake_wiring.py')+'::test_noahmp_call_excludes_actual_lake_columns',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s1-lake/pytest','-q','--tb=short']))
