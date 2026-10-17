"""Delete actual surface rates at the production seam; demand a numerical failure."""
from pathlib import Path
import json,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'scripts/v025'))
import cpu_guard
cpu_guard.force_cpu()
import jax
import jax.numpy as jnp
import pytest
from gpuwrf.assimilation import fdda_surface as surface

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
original=surface.sfddagd
def remove_surface_rates(*args,**kwargs):
    rates,sda=original(*args,**kwargs)
    return tuple(jnp.zeros_like(a) for a in rates),sda
surface.sfddagd=remove_surface_rates
jax.clear_caches()
try:
    code=pytest.main([str(root/'tests/v040/sfdda/test_surface_oracle.py')+'::test_full_pristine_surface_chain[swiss_sfc1_plain_r4.npz]',
        '-q','--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-sfdda/pytest','--junitxml',str(out/'mutant.xml')])
finally:
    surface.sfddagd=original;jax.clear_caches()
tree=ET.parse(out/'mutant.xml').getroot()
failures=tree.findall('.//failure');errors=tree.findall('.//error')
assert code==1 and len(failures)==1 and not errors
assert 'AssertionError' in (failures[0].text or '') and 'unc' in (failures[0].text or '')
result={'mutant':'production SFDDAGD rates deleted','killed':True,'numerical_failures':1,'errors':0,
        'model_source_edited':False,'pytest_returncode':int(code)}
(out/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
