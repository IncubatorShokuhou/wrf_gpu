"""Delete the actual Liu crop evolution; the untouched original numeric test must fail."""
import json,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
TREE,OUT=map(Path,sys.argv[1:3]);sys.path.insert(0,str(TREE/'src'))
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_COLUMN_KERNELS='0')
sys.path.insert(0,str(TREE/'scripts/v025'))
import cpu_guard
import jax
import jax.numpy as jnp
import pytest
from gpuwrf.physics.noahmp import crop
assert jax.devices()[0].platform=='cpu'


def deleted_evolution(state,p,lai,sai,*args,**kwargs):
    return state,lai,sai,{key:jnp.zeros_like(lai) for key in ('GPP','NPP','NEE','AUTORS','HETERS','TOTSC','TOTLB')}


crop.crop_flux=deleted_evolution
xml=OUT.with_suffix('.xml')
code=pytest.main(['-q',str(TREE/'tests/test_u2_crop_options.py'),'--basetemp','<USER_HOME>/wrf_gpu2_lanes/u2-land/pytest','--junitxml',str(xml)])
cases=ET.parse(xml).findall('.//testcase')
failures=[c for c in cases if c.find('failure') is not None];errors=[c for c in cases if c.find('error') is not None]
report=dict(platform='cpu',mutant='actual crop.crop_flux returns unchanged state/zero fluxes',pytest_rc=code,
    failures=len(failures),errors=len(errors),numeric_assertions=sum('AssertionError' in c.find('failure').get('message','') for c in failures))
OUT.write_text(json.dumps(report,indent=2)+'\n')
assert code==1 and len(failures)==1 and not errors and report['numeric_assertions']==1,report
