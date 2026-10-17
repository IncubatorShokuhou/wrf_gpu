"""Remove each required coupling fix; unchanged seam tests must assert."""
from pathlib import Path
import sys

repo=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(repo/'scripts/v025'))
import cpu_guard  # noqa: E402,F401
import gpuwrf.coupling.lake_surface_hook as module
import pytest

variants={
    'exner':('**_mynn_surface_exner_kwargs(view,f32)','**{}',
             'test_lake_mynn_forwards_independent_original_wrf_exner[True]'),
    'empty_history':('carry = carry._replace(background=background)','carry = carry',
                     'test_zero_lake_domain_keeps_changing_land_history'),
}
needle,replacement,test=variants[sys.argv[2]]
source=Path(module.__file__).read_text()
assert source.count(needle)==1
exec(compile(source.replace(needle,replacement),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main([str(repo/'tests/test_s1_lake_wiring.py')+'::'+test,
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s1-lake/pytest','-q','--tb=short']))
