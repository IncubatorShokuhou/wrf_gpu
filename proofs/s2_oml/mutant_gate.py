"""Genuine deletion-sensitive assertion: remove OML evolution in-process."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

root=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(root/'src'),str(root/'scripts/v025')]
import cpu_guard
import gpuwrf.physics.ocean_mixed_layer as oml
oml.step=lambda carry,**kw:carry
import pytest

xml=out.with_suffix('.xml')
code=pytest.main(['-q',str(root/'tests/test_ocean_mixed_layer.py')+'::test_pristine_real_columns',
                  '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-oml/pytest-mutant','--tb=short',f'--junitxml={xml}'])
tree=ET.parse(xml).getroot()
failures=list(tree.iter('failure'));errors=list(tree.iter('error'))
signature=bool(failures and 'AssertionError' in (failures[0].text or '') and 'Arrays are not equal' in (failures[0].text or ''))
killed=code==1 and len(failures)==1 and not errors and signature
record={'source_root':str(root),'mutation':'step returns unchanged carry','pytest_code':int(code),
        'assertion_failures':len(failures),'setup_errors':len(errors),'killed':killed,
        'failure_type':None if not failures else failures[0].get('type'),'assertion_signature':signature}
out.write_text(json.dumps(record,indent=2)+'\n')
assert killed,record
