"""Delete urban drag/moisture forcing at the actual PBL entry, require numerics."""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import pbl_parity as gate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--budget',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
record={'numerical_assertions':0,'setup_errors':0,'mutant':'zero actual urban a_u/a_v/b_u/b_v/a_q/b_q at PBL entry'}
try:
    def mutated(fn):
        def call(*args,**kwargs):
            urban=kwargs['urban']
            kwargs['urban']=urban._replace(**{name:jnp.zeros_like(getattr(urban,name))
                for name in ('a_u','a_v','b_u','b_v','a_q','b_q')})
            return fn(*args,**kwargs)
        return call
    gate.ysu_columns=mutated(gate.ysu_columns)
    gate.boulac_columns=mutated(gate.boulac_columns)
    fields=a.out.with_name(a.out.stem+'_fields.json')
    gate.run(a.root,fields,precisions=('fp32',),budget=a.budget)
    results=json.loads(fields.read_text())
    record['violations']={scheme:sum(row['violations'] for row in results[scheme]['fp32']['fields']
        if row['field'] in ('u','v','qv')) for scheme in results}
    for scheme,count in record['violations'].items():
        try:
            assert count==0,(scheme,count)
        except AssertionError:
            record['numerical_assertions']+=1
    assert record['numerical_assertions']==2,record
except Exception:
    record['setup_errors']+=1
    raise
finally:
    a.out.write_text(json.dumps(record,indent=2)+'\n')
print(record)
