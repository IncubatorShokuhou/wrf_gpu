"""Requalify only the source bodies touched by the native profile adapter."""
import json,os,runpy,sys
from pathlib import Path
tree,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
assert os.environ['JAX_PLATFORMS']=='cpu'
gates=('carbon','crop','groundwater','equilibrium','prognostic','runoff','advanced_water','water')
summary={}
for family in gates:
    script=tree/'proofs/u2_land'/f'{family}_gate.py';target=out/f'{family}.json'
    sys.argv=[str(script),str(tree),str(target)]
    if family=='water':sys.argv+=['run3_inf2','run4_inf1','run4_inf2']
    runpy.run_path(str(script),run_name='__main__')
    data=json.loads(target.read_text());rows=list(data['results'].values()) if 'results' in data else [data]
    values=sum(row['values'] for row in rows);literal=sum(row['literal_byte'] for row in rows)
    assert values==literal,(family,values,literal)
    summary[family]=dict(values=values,literal_byte=literal,
        compiled_within=sum(row['compiled_within'] for row in rows),
        max_excess=max(row['max_excess'] for row in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
