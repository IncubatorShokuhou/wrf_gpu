"""Independent static audit, no JAX backend or production mutations."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

root=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/candidate')
out=root.parent
wrf=Path('<USER_HOME>/src/wrf_pristine/WRF')
driver=wrf/'phys/module_microphysics_driver.F'
registry=wrf/'Registry/Registry.EM_COMMON'
packages={int(m[2]):m[1].upper() for m in re.finditer(r'(?m)^package\s+(\w+)\s+mp_physics==(\d+)\s',registry.read_text())}
lines=driver.read_text().splitlines()
cases=[(i,m[1].upper()) for i,line in enumerate(lines) if (m:=re.match(r'\s*CASE\s*\(\s*(\w+)\s*\)',line))]
slots={}
for n,(start,name) in enumerate(cases):
    end=cases[n+1][0] if n+1<len(cases) else len(lines)
    body=''.join(line.split('!')[0] for line in lines[start:end]).upper().replace(' ','').replace('&','')
    slots[name]={'line':start+1,'slots':[s for s in ('RAINNCV','SNOWNCV','GRAUPELNCV','HAILNCV') if re.search(rf'={s}[,)]',body)]}
catalog=ast.parse((root/'src/gpuwrf/io/scheme_catalog.py').read_text())
for node in ast.walk(catalog):
    if isinstance(node,ast.Dict):
        for k,v in zip(node.keys,node.values):
            if isinstance(k,ast.Constant) and k.value=='mp_physics' and isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='frozenset':
                admitted=sorted(ast.literal_eval(v.args[0]));break
        else:continue
        break
old=subprocess.check_output(['git','show','93dfdac853^:src/gpuwrf/runtime/operational_mode.py'],cwd=root,text=True)
new=(root/'src/gpuwrf/runtime/operational_mode.py').read_text()
def branch(source,token):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_apply_post_rk_microphysics')
    for n in ast.walk(fn):
        if isinstance(n,ast.If) and ast.unparse(n.test)==token:
            return ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False)
    raise ValueError(token)
retained={token:branch(old,token)==branch(new,token) for token in
          ('mp_opt in (5, 15)','mp_opt == 7','mp_opt == 38','mp_opt == DEFAULT_MP_PHYSICS')}
assert all(retained.values())
routes={0:'dry: no MP precipitation',**{c:'generic public-slot helper' for c in (1,2,3,4,6,10,13,14,16,24,26,40,97)},
    5:'Ferrier existing slots',15:'Ferrier existing slots',7:'Goddard 4-ice existing slots',8:'Thompson existing slots',
    9:'Milbrandt total/hail override',11:'CAM-MG existing amounts, now seeded',
    **{c:'NSSL public slots' for c in (17,18,19,21,22)},28:'Thompson aerosol optional amounts',
    38:'Thompson hail existing slots',**{c:'P3 existing amounts, now seeded' for c in (50,51,52,53)},
    55:'Jensen existing amounts, now seeded',56:'NTU existing public rain-only override, now seeded'}
assert set(admitted)==set(routes)
rows=[]
for code in admitted:
    name=packages.get(code,packages[18] if code in (17,19,21,22) else None)
    rows.append(dict(code=code,wrf_case=name,wrf_alias=code in (17,19,21,22),route=routes[code],**slots.get(name,{})))
result={'source':'052f16d2b69e9f0c1299e6ae0049c742e112a765','admitted_count':len(admitted),
        'admitted':rows,'retained_branch_ast_equal':retained,
        'reference_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (driver,registry)},
        'scope':'static routing and retained branch identity; not numerical/GPU fidelity'}
(out/'RV04-source.json').write_text(json.dumps(result,indent=2)+'\n')
print('admitted',len(admitted),'retained AST',retained)
