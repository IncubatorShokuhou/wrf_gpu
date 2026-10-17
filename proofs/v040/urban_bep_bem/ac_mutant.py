"""Genuine lost-AC-exhaust source mutant versus the complete pristine driver."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import bem_driver_parity as parity
from gpuwrf.physics import urban_bem_model as model

p=argparse.ArgumentParser();p.add_argument('--save',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
original=inspect.getsource(model.bem_column_step)
before='result.sensible_ventilation[...,k]-result.sensible_exhaust[...,k]'
assert original.count(before)==1
mutated=original.replace(before,'result.sensible_ventilation[...,k]')
namespace=dict(vars(model));exec(compile(mutated,'<lost_ac_exhaust_source_mutant>','exec'),namespace)
parity.bem_column_step=namespace['bem_column_step']
actual=a.out.with_name(a.out.stem+'_fields.json')
parity.run(a.save,actual,['fp64'],1)
results=json.loads(actual.read_text())
assert results['fp64_ac_off']['violations']==0,results['fp64_ac_off']
heat=[x for x in results['fp64_ac_on']['fields'] if x['field']=='B_T_BEP']
assert heat and heat[0]['violations']>0
try:
    assert results['fp64_ac_on']['violations']==0,'complete driver differs from pristine WRF (numerical gate)'
except AssertionError as e:
    receipt={'status':'KILLED','numerical_assertions':1,'setup_errors':0,
        'message':str(e),'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'mutant_sha256':hashlib.sha256(mutated.encode()).hexdigest(),
        'deleted_source':before,'replacement':'result.sensible_ventilation[...,k]',
        'active_heat_source_gate':heat[0],'AC_off_unchanged_violations':0}
    a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
else:raise RuntimeError('lost AC exhaust mutant survived')
