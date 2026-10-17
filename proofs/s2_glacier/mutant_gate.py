"""A genuine deletion of the actual operational glacier-dispatch branch."""
import ast
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v025'))
import cpu_guard  # establish backend guard before importing JAX/gpuwrf
from gpuwrf.physics import noahmp_coupler as cp

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
tree=ast.parse(inspect.getsource(cp.noahmp_surface_adapter));hits=0
for node in ast.walk(tree):
    if isinstance(node,ast.If) and ast.unparse(node.test)=="getattr(static, 'glacier', None) is not None":
        node.test=ast.Constant(False);hits+=1
assert hits==1
ast.fix_missing_locations(tree);namespace=dict(cp.__dict__)
exec(compile(tree,cp.__file__,'exec'),namespace)
cp.noahmp_surface_adapter=namespace['noahmp_surface_adapter']
report=out/'mutant.xml'
code=pytest.main(['-q',str(root/'tests/test_s2_glacier_wiring.py')+'::test_operational_adapter_routes_glacier_tile',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-glacier/pytest',f'--junitxml={report}'])
doc=ET.parse(report);failures=doc.findall('.//failure');errors=doc.findall('.//error')
assert code==1 and len(failures)==1 and not errors
assert 'natural path selected instead of glacier routing' in (failures[0].text or '')
verdict=dict(status='KILLED',assertion_failures=1,setup_errors=0,mutation='operational glacier branch replaced with False')
(out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
print(json.dumps(verdict))
