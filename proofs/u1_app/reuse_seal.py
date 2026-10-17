"""Bind reused positive gates to exact unchanged sources after composition."""
import ast
import json
from pathlib import Path
import subprocess
import sys

PREFIX='src/gpuwrf/post/apps/'
def source(ref,path):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],text=True)
def blob(ref,path):
    return subprocess.check_output(['git','rev-parse',f'{ref}:{path}'],text=True).strip()

rows=[]
for name,ref,receipt in [('_numeric.py','e319b719f','AP04'),('solar.py','e319b719f','AP04'),
                         ('fire.py','e319b719f','AP04'),('hazards.py','9453acf51','AP14'),
                         ('wind.py','6959a3db1','AP18')]:
    path=PREFIX+name
    old,new=blob(ref,path),blob('HEAD',path)
    rows.append(dict(path=path,reference=ref,receipt=receipt,old_blob=old,new_blob=new,exact=old==new))
    assert old==new,path
path=PREFIX+'dataset.py'
old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source('9453acf51',path)).body if isinstance(n,ast.FunctionDef)}
new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source('HEAD',path)).body if isinstance(n,ast.FunctionDef)}
changed=[name for name in old.keys()|new.keys() if old.get(name)!=new.get(name)]
assert changed==['_accumulated_rain'],changed
receipt=dict(candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             exact_family_blobs=rows,dataset_changed_functions=changed,
             changed_function_gate='AP20 dataset FWI bucket/noon/CLI checks; pure FWI fire.py unchanged,108reference cases not replayed',
             interpretation='Source-bound numerical validation reuse, not a new numerical run or GPU fidelity claim')
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True))
