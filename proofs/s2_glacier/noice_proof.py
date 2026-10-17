"""Exact source reduction and unchanged pytree proof for every ice-free input.

The source theorem covers all 54 WN3 domains. The separate official PROD d01
trace is a direct witness; this script does not perform 54 measured traces.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys

root=Path(sys.argv[1]).resolve();control=sys.argv[2];target=Path(sys.argv[3])
assert os.environ['JAX_PLATFORMS']=='cpu'


def old(path):return subprocess.check_output(['git','show',f'{control}:{path}'],cwd=root,text=True)
def function(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)


class IceFree(ast.NodeTransformer):
    def visit_If(self,node):
        condition=ast.unparse(node.test)
        if condition=="getattr(static, 'glacier', None) is not None":return self.visit(node.orelse[0])
        if condition=='ice_points.size':return None
        return self.generic_visit(node)
    def visit_ImportFrom(self,node):
        return None if node.module=='gpuwrf.io.noahmp_glacier_options' else node
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='ice_points' for t in node.targets):return None
        return self.generic_visit(node)


paths={'coupler':'src/gpuwrf/physics/noahmp_coupler.py','init':'src/gpuwrf/io/noahmp_land_init.py',
       'contracts':'src/gpuwrf/contracts/noahmp_state.py'}
sources={key:((root/path).read_text(),old(path)) for key,path in paths.items()}
checks={}
for key,name in [('coupler','noahmp_surface_adapter'),('init','_build_noahmp_land_state')]:
    candidate,baseline=sources[key]
    reduced=IceFree().visit(copy.deepcopy(function(candidate,name)))
    checks[key+'_reduced_ast_byte_exact']=ast.dump(reduced,include_attributes=False)==ast.dump(function(baseline,name),include_attributes=False)
    assert checks[key+'_reduced_ast_byte_exact'],key
for name,extra in [('NoahMPLandState','glacier_qsnow'),('NoahMPStatic','glacier')]:
    classes=[]
    for text in sources['contracts']:
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==name)
        slots=ast.literal_eval(next(n.value for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__slots__' for t in n.targets)))
        ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        packed=next(n for n in ast.walk(ctor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='zip').args[1]
        names=tuple(ast.unparse(n) for n in packed.elts)
        classes.append((slots,ctor,names,cls))
    candidate,baseline=classes
    assert candidate[0][:-1]==baseline[0] and candidate[0][-1]==extra
    assert candidate[2][:-1]==baseline[2] and candidate[2][-1]==extra
    assert [x.arg for x in candidate[1].args.args[:-1]]==[x.arg for x in baseline[1].args.args]
    assert isinstance(candidate[1].args.defaults[-1],ast.Constant) and candidate[1].args.defaults[-1].value is None
    flatten=ast.unparse(next(n for n in candidate[3].body if isinstance(n,ast.FunctionDef) and n.name=='tree_flatten'))
    assert 'self.__slots__[:-1]' in flatten
    checks[name]=dict(prefix_slots=list(baseline[0]),new_optional=extra,default=None,
        ice_free_children=len(baseline[0]),ice_free_aux=None)

# Every existing arithmetic module and the operational/pipeline drivers remain
# byte-identical to the control; only the reduced seams above are different.
unchanged=subprocess.check_output(['git','ls-tree','-r','--name-only',control,'src/gpuwrf/physics/noahmp','src/gpuwrf/kernels/phys_noahmp_columns.py',
    'src/gpuwrf/kernels/phys_noahmp_layers.py','src/gpuwrf/kernels/phys_noahmp_radiation.py',
    'src/gpuwrf/runtime/operational_mode.py','src/gpuwrf/runtime/operational_state.py',
    'src/gpuwrf/integration/nested_pipeline.py','src/gpuwrf/integration/daily_pipeline.py'],cwd=root,text=True).splitlines()
module_hashes={}
for path in unchanged:
    a=(root/path).read_bytes();b=subprocess.check_output(['git','show',f'{control}:{path}'],cwd=root)
    assert a==b,path
    module_hashes[path]=hashlib.sha256(a).hexdigest()
census=json.loads(Path('<USER_HOME>/wrf_gpu2_lanes/s2-glacier/G01_impact.json').read_text())
wn3=[r for r in census['rows'] if '<DATA_ROOT>/server/' in r['path']]
assert len(wn3)==54 and all(r['glacier_caller_count']==0 and r['isice']==15 for r in wn3)
receipt=dict(control_commit=control,candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    proof='ice-free source AST reduces exactly to control; constructor/flatten prefixes and None auxiliary data preserved; unchanged arithmetic modules',
    checks=checks,unchanged_module_sha256=module_hashes,wn3_domains=wn3,
    scope='Universal source/pytree proof for all 54 zero-ISICE WN3 domains, not 54 measured native executions; direct PROD graph witness G19',
    gpu_executed=False)
target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(checks=list(checks),unchanged_modules=len(module_hashes),wn3_domains=len(wn3)),indent=2))
