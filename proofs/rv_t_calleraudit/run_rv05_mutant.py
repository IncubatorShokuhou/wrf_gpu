"""Revert only the F1 carry seed in this process; genuine numeric assertions required."""
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from gpuwrf.runtime import operational_mode as op

out=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit')
source=inspect.getsource(op._initial_carry_for_run)
old='and int(namelist.mp_physics) != 0:'
new='and int(namelist.mp_physics) in (DEFAULT_MP_PHYSICS, 5, 7, 15, 38):'
assert source.count(old)==1
exec(compile(source.replace(old,new),op.__file__,'exec'),op.__dict__)
rc=pytest.main([str(out/'test_rv03_two_step.py'),'-q','--tb=short','-p','no:cacheprovider',
               '-k','test_two_step_real_dispatch_to_surface_forcing',
               '--basetemp',str(out/'pytest'),'--junitxml='+str(out/'RV07-mutant-final.xml')])
root=ET.parse(out/'RV07-mutant-final.xml').getroot()
cases=root.findall('.//testcase')
failures=[n for c in cases for n in c.findall('failure')]
errors=root.findall('.//error')
numeric=[f for f in failures if 'AssertionError' in (f.text or '') and 'Not equal to tolerance' in (f.text or '')]
result=dict(rc=rc,tests=len(cases),failures=len(failures),errors=len(errors),numeric_assertion_kills=len(numeric),
            mutation='in-memory original seed set; production bytes unchanged',source=op.__file__)
(out/'RV07-mutant-final.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert rc==1 and len(numeric)==13 and not errors
