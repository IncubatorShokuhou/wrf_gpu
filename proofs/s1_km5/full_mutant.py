"""Delete the actual implicit solve; require a numerical oracle assertion kill."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v025')]
import cpu_guard
import pytest
from gpuwrf.dynamics import sms_vertical
sms_vertical.solve=lambda a,b,c,d:d
OUT=Path(sys.argv[1])
rc=pytest.main(['-q','tests/dynamics/test_sms3dtke_full.py::test_sms_complete_pristine_operator[real_mixed]',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s1-km5/pytest',
    '-o','cache_dir=<USER_HOME>/wrf_gpu2_lanes/s1-km5/pytest_cache',
    '--junitxml='+str(OUT.with_suffix('.xml'))])
from xml.etree import ElementTree
x=ElementTree.parse(OUT.with_suffix('.xml'))
failures=x.findall('.//failure');errors=x.findall('.//error')
assert rc==1 and failures and not errors and any('AssertionError' in ''.join(f.itertext()) for f in failures)
OUT.write_text(json.dumps(dict(mutant='actual Thomas solve returns unsolved RHS',
    numeric_assertion_kills=len(failures),setup_errors=len(errors)),indent=1))
