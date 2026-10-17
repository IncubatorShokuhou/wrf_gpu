"""Verify repaired mocks preserve variant assertions; no production file mutation."""
from pathlib import Path
import sys
sys.path.insert(0,'<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/NSSL/candidate/scripts/v025')
import cpu_guard
import inspect,json
import xml.etree.ElementTree as ET
import pytest
from gpuwrf.runtime import operational_mode as op
out=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/NSSL')
source=inspect.getsource(op._apply_post_rk_microphysics)
old='variant={"variant":mp_opt}'
assert source.count(old)==1
exec(compile(source.replace(old,'variant={"variant":19}'),op.__file__,'exec'),op.__dict__)
nodes=['tests/v040/test_nssl_one_moment_wiring.py::test_actual_post_rk_variant_and_cold_start[1-21]',
       'tests/v040/test_nssl_one_moment_wiring.py::test_actual_post_rk_variant_and_cold_start[2-21]']
rc=pytest.main([*nodes,'-q','--tb=short','-p','no:cacheprovider','--basetemp',str(out.parent/'pytest'),
               '--junitxml='+str(out/'RN03-mutant.xml')])
root=ET.parse(out/'RN03-mutant.xml').getroot()
failures=root.findall('.//failure');errors=root.findall('.//error')
assertions=[f for f in failures if 'AssertionError' in (f.text or '') and '19 == 21' in (f.text or '')]
record={'rc':rc,'failures':len(failures),'errors':len(errors),'intended_variant_assertion_kills':len(assertions),
        'mutation':'mp21 incorrectly dispatches variant19, process-local only'}
(out/'RN03-mutant.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
assert rc==1 and len(assertions)==2 and not errors
