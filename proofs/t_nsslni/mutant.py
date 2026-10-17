"""Exercise actual PBL producers with the old gate or all QNI sources deleted."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT, OUT, MODE = Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),sys.argv[3]
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(ROOT/'scripts/v025'))
import cpu_guard
os.environ.setdefault('GPUWRF_FAST_DEFAULTS','0')
sys.path.insert(0,str(ROOT/'src'))
from gpuwrf.runtime import operational_mode as op
import pytest

if MODE=='old':
    op._pbl_qni_active=lambda mp:int(mp) not in (5,7,15,14,16,26)
    selection='real_pbl_dispatch and (19 or 21)'
    expected=4
elif MODE=='delete':
    op._pbl_qni_active=lambda mp:False
    selection='real_pbl_dispatch and (8 or 18 or 22)'
    expected=6
else:
    raise ValueError(MODE)
xml=OUT.with_suffix('.xml')
rc=pytest.main([str(ROOT/'tests/test_nsslni_physics_sources.py'),'-k',selection,
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/t-nsslni/pytest','-q','--tb=short','--junitxml='+str(xml)])
tests=ET.parse(xml).findall('.//testcase')
failures=[t for t in tests if t.find('failure') is not None]
errors=[t for t in tests if t.find('error') is not None]
assert rc==1 and len(failures)==len(tests)==expected and not errors
assert all('assert ' in t.find('failure').get('message','').lower() for t in failures)
OUT.write_text(json.dumps(dict(mode=MODE,killed=True,assertion_failures=len(failures),setup_errors=0,
                              cases=[t.get('name') for t in failures]),indent=2)+'\n')
