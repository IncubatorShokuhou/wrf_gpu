"""Source-bound follow-up review, CPU only, no candidate edits."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types

root=Path(sys.argv[1]);out=Path(__file__).parent
sys.path.insert(0,str(root/'src'))
old='9a40424b0dfcd82f39ca4d14919007738523ba72'
new='3ec98090e1'
def source(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root,text=True)
def function(text,name):return next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def nodoc(node):
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
    return node
routepath='src/gpuwrf/kernels/_triton_kernel_call.py'
a,b=source(old,routepath),source(new,routepath)
lo,ln=nodoc(function(a,'_lowering')),nodoc(function(b,'_lowering'))
for node in ast.walk(ln):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='stock_compile':
        assert len(node.args)==6 and isinstance(node.args[-1],ast.Call) and node.args[-1].func.id=='stock_compiler_identity'
        node.args.pop()
assert dump(lo)==dump(ln)
def stages(text):
    body=function(text,'stock_compile').body
    index=next(i for i,n in enumerate(body) if isinstance(n,ast.ImportFrom) and n.module=='triton._C.libtriton')
    return ast.dump(ast.Module(body=body[index:],type_ignores=[]),include_attributes=False)
assert stages(a)==stages(b)
kernelpath='src/gpuwrf/kernels/phys_cammg.py'
assert dump(nodoc(function(source(old,kernelpath),'full_column')))==dump(nodoc(function(source(new,kernelpath),'full_column')))
physics=subprocess.check_output(['git','diff','--name-only',old,new,'--','src/gpuwrf/physics/cammg'],cwd=root,text=True).splitlines()
assert physics==[]

from gpuwrf.runtime import operational_mode as om
ns=dict(vars(om))
oldfn=function(source(old,'src/gpuwrf/runtime/operational_mode.py'),'_initial_carry_for_run')
oldfn.name='review_original_initial_carry'
exec(compile(ast.fix_missing_locations(ast.Module(body=[oldfn],type_ignores=[])), '<original-9a404-init>', 'exec'),ns)
fixture=runpy.run_path(str(root/'tests/test_rrtm_lw_operational_wiring.py'))
grid=fixture['_grid'](nz=8,ny=3,nx=4);state=fixture['_state'](grid)
nl=om.OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=None,dt_s=18.,
    acoustic_substeps=4,use_noahmp=True,run_physics=True,mp_physics=11,
    bl_pbl_physics=9,sf_sfclay_physics=5,sf_surface_physics=4,ra_sw_physics=0,ra_lw_physics=0,cu_physics=0)
before=ns['review_original_initial_carry'](state,nl)
after=om._initial_carry_for_run(state,nl)
assert before.noahmp_precipitation is None
assert after.noahmp_precipitation is not None
init={'old_noahmp_precipitation':None,'new_noahmp_precipitation_fields':list(after.noahmp_precipitation._fields),
      'c89_mp11_noahmp_initial_carry_BYTE':False,
      'scope':'actual old/new init bodies, same tiny CPU state and mp11+bl9+NoahMP configuration; no forecast execution'}

from gpuwrf.kernels import _triton_kernel_call as route
tests=runpy.run_path(str(root/'tests/v040/test_cammg_kernel_call_route.py'))
with tempfile.TemporaryDirectory(dir=out) as tmp:
    tests['test_tree_digest_is_content_addressed'](Path(tmp))
with tempfile.TemporaryDirectory(dir=out) as tmp:
    route.tree_digest=lambda root:'metadata-only-mutant'
    try:tests['test_tree_digest_is_content_addressed'](Path(tmp))
    except AssertionError:mutant={'result':'KILLED','failure':'AssertionError','setup_errors':0}
    else:raise AssertionError('content-hash mutant survived')

missing=[];bad=[];count=0
for line in (root/'proofs/t_cammgws/RECEIPTS.sha256').read_text().splitlines():
    sha,name=line.split(maxsplit=1);path=root/'proofs/t_cammgws/receipts'/name;count+=1
    if not path.exists():missing.append(name)
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=sha:bad.append(name)
assert not missing and not bad,(missing,bad)
result={'utc':datetime.now(timezone.utc).isoformat(),'old_commit':old,'new_commit':new,
        'stock_lowering_AST_equal_except_identity_argument':True,'stock_compile_pipeline_AST_equal':True,
        'full_column_AST_equal':True,'cammg_physics_files_changed':physics,
        'initial_carry':init,'content_hash_mutant':mutant,'receipt_manifest':{'files':count,'missing':missing,'bad_sha256':bad}}
(out/'FIX_SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
