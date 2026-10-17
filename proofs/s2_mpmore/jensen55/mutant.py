"""Disable actual homogeneous freezing; require numerical assertion failures."""
from pathlib import Path
import inspect,json,os,sys,textwrap
from xml.etree import ElementTree

tree,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and os.environ['JAX_PLATFORMS']=='cpu'
sys.path[:0]=[str(tree/'src'),str(tree/'scripts/v025')]
import cpu_guard
from gpuwrf.physics.jensen_ishmael import processes,column
import pytest

original=processes.processes
source=textwrap.dedent(inspect.getsource(original))
needle='    riming=rain_ice=freeze_qc=splinters=wet_grow=True'
assert source.count(needle)==1
scope={}
exec(compile(source.replace(needle,needle+'\n    freeze_qc=False'),'<disabled-Jensen-freezing>','exec'),original.__globals__,scope)
column.processes=scope['processes']
code=pytest.main([str(tree/'tests/v040/test_jensen_ishmael_column.py'),'-q','--tb=short',
                 '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-mpmore/pytest',
                 '--junitxml='+str(out.with_suffix('.xml'))])
suite=ElementTree.parse(out.with_suffix('.xml')).getroot().find('testsuite')
assert code==1 and int(suite.get('failures'))>0 and suite.get('errors')=='0',(code,suite.attrib)
assert all('AssertionError' in (f.text or '') for f in suite.findall('testcase/failure'))
out.write_text(json.dumps(dict(mutant='disable actual homogeneous cloud/rain freezing',killed=True,
                             failures=int(suite.get('failures')),setup_errors=0),indent=2)+'\n')
