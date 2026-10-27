"""Delete only the MMINLU refusal; require genuine DID NOT RAISE, not setup errors."""
import inspect
import sys
from pathlib import Path

# Match pytest's mandatory import order, even though this launcher patches the
# production function before pytest loads conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v025'))
import cpu_guard  # noqa: F401,E402

import pytest
from gpuwrf.io import noahmp_land_init as init

source = inspect.getsource(init._build_noahmp_land_state)
needle = 'if dataset != "MODIFIED_IGBP_MODIS_NOAH":'
assert source.count(needle) == 1
namespace = dict(vars(init))
exec(compile(source.replace(needle, 'if False:'), init.__file__, 'exec'), namespace)
init._build_noahmp_land_state = namespace['_build_noahmp_land_state']
sys.exit(pytest.main([
    'tests/test_noahmp_landuse_admission.py::test_real_input_with_usgs_metadata_is_refused',
    '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/w44-lutab/pytest_mutant', '-q',
]))
