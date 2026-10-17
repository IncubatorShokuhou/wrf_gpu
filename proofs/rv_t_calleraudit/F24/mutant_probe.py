"""Independent in-memory deletion on final caller/surface tests; assertion kills only."""
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest
from gpuwrf.runtime import operational_mode as op
import gpuwrf.coupling.noahclassic_surface_hook as noah

out=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/F24')
mode=sys.argv[1]
if mode=='seed':
    source=inspect.getsource(op._land_precipitation_seed)
    needle='if not bool(namelist.run_physics):'
    assert source.count(needle)==1
    source=source.replace(needle,'if not bool(namelist.run_physics) or int(namelist.mp_physics) == 0:')
    exec(compile(source,op.__file__,'exec'),op.__dict__)
    nodes=[str(out/'test_rc02_caller.py')+'::test_real_cu_adapter_two_step_noah_forcing['+str(c)+'-0]' for c in (2,3,5,6,16,93)]
    required=6
else:
    assert mode=='sr'
    source=inspect.getsource(noah.noahclassic_precipitation)
    needle='jnp.asarray(precipitation.sr, jnp.float32)'
    assert source.count(needle)==1
    exec(compile(source.replace(needle,'jnp.zeros_like(rate)'),noah.__file__,'exec'),noah.__dict__)
    nodes=[str(out/'test_rc04_surface.py')+'::test_active_carry_reaches_real_lsm_solver[2]']
    required=1
name='RC05-seed-mutant' if mode=='seed' else 'RC06-sr-mutant'
rc=pytest.main([*nodes,'-q','--tb=short','-p','no:cacheprovider',
    '--basetemp',str(out.parent/'pytest'),'--junitxml='+str(out/(name+'.xml'))])
root=ET.parse(out/(name+'.xml')).getroot()
failures=root.findall('.//failure');errors=root.findall('.//error')
assertions=[f for f in failures if 'AssertionError' in (f.text or '')]
record=dict(mode=mode,rc=rc,failures=len(failures),errors=len(errors),assertion_kills=len(assertions),
            scope='actual caller/surface kernel inputs, in-memory mutation; source bytes unchanged')
(out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(record)
assert rc==1 and len(assertions)==required and not errors
