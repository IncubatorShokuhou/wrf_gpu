"""Independent wrong-species or traceback deletion; genuine assertion failures only."""
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,'<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/candidate/scripts/v025')
import cpu_guard  # enforce CPU before any gpuwrf/JAX import, matching current conftest
import pytest

out=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18')
mode=sys.argv[1]
if mode=='species':
    from gpuwrf.kernels import phys_mynn_plume as module
    source=inspect.getsource(module.dmp_mf_columns_native)
    old='lane = dict(kwargs, qni=jnp.asarray(values, dtype=jnp.float32))'
    assert source.count(old)==1
    source=source.replace(old,'lane = dict(kwargs)')
    node='tests/test_t_calleraudit_mynn_passive_native.py::test_native_entry_passive_fluxes_match_pristine_dmp_mf'
else:
    assert mode=='traceback'
    from gpuwrf.runtime import domain_tree as module
    source=inspect.getsource(module._lower_error_text)
    old='    traceback.print_exception(exc, file=sys.stderr)\n'
    assert source.count(old)==1
    source=source.replace(old,'')
    node='tests/test_t_calleraudit_lower_error_traceback.py::test_lower_error_text_prints_the_traceback'
exec(compile(source,module.__file__,'exec'),module.__dict__)
name='RP04-'+mode+'-mutant'
rc=pytest.main([node,'-q','--tb=short','-p','no:cacheprovider','--basetemp',str(out.parent/'pytest'),
               '--junitxml='+str(out/(name+'.xml'))])
root=ET.parse(out/(name+'.xml')).getroot()
failures=root.findall('.//failure');errors=root.findall('.//error')
assertions=[f for f in failures if 'AssertionError' in (f.text or '')]
record=dict(mode=mode,rc=rc,failures=len(failures),errors=len(errors),assertion_kills=len(assertions),
            scope='in-memory deletion, production bytes unchanged')
(out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(record)
assert rc==1 and len(assertions)==1 and not errors
