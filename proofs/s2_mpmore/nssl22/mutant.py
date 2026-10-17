"""Remove the physical no-hail creation guard and demand an oracle assertion."""
from pathlib import Path
import inspect,json,sys
from xml.etree import ElementTree

root,output=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
sys.path[:0]=[str(root/'src'),str(root/'scripts/v025')]
import cpu_guard
from gpuwrf.physics.nssl2mom import gs_c
import pytest

original=gs_c.gs_part_c
source=inspect.getsource(original)
start=source.index('    if not hail_enabled:')
end=source.index('    alpha = g["alpha"]',start)
removed=source[start:end]
assert 'conv = jnp.zeros_like(conv)' in removed
scope={}
exec(compile(source[:start]+source[end:],'<removed-nohail-guard>','exec'),original.__globals__,scope)
gs_c.gs_part_c=scope['gs_part_c']
rc=pytest.main([str(root/'tests/v040/test_nssl_nohail22.py')+'::test_complete_original22_column[fp64-16]',
               '-q','--tb=short','--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-mpmore/pytest',
               '--junitxml='+str(output.with_suffix('.xml'))])
s=ElementTree.parse(output.with_suffix('.xml')).getroot().find('testsuite')
assert rc==1 and s.get('failures')=='1' and s.get('errors')=='0',(rc,s.attrib)
failure=s.find('testcase/failure')
assert 'AssertionError' in failure.get('message',''),failure.get('message')
output.write_text(json.dumps(dict(killed=True,scope='physical no-hail creation guard',
                                 assertion=failure.get('message')),indent=2)+'\n')
