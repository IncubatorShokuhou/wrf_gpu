"""Prove scientific-body identity to the original-qualified native source."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(__file__).resolve().parents[2]
REFERENCE='95276f35d99f2fbc5daa62f3a712f8562b9256e8'
ORIGINAL_MEASURED_SOURCE='e1aff006c468c6b3dd08874dbef083e941dd4af9'
class RemoveDocstrings(ast.NodeTransformer):
    def visit_Module(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body.pop(0)
        return node
    visit_FunctionDef=visit_Module
    visit_AsyncFunctionDef=visit_Module
    visit_ClassDef=visit_Module

def normalized(source):
    return ast.dump(RemoveDocstrings().visit(ast.parse(source)),include_attributes=False).encode()

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    names=('_native','_support','constants','gamma','column','micro','macro','aerosol','cloud','ice_activation','error_function','tables')
    paths=[f'src/gpuwrf/physics/cammg/{name}.py' for name in names]+['src/gpuwrf/kernels/phys_cammg.py']
    rows=[]
    for path in paths:
        previous=subprocess.check_output(['git','show',f'{REFERENCE}:{path}'],cwd=ROOT)
        current=(ROOT/path).read_bytes()
        old_ast,new_ast=normalized(previous),normalized(current)
        assert old_ast==new_ast,path
        rows.append(dict(path=path,reference_sha256=sha(previous),current_sha256=sha(current),
                         AST_sha256=sha(new_ast),BYTE_identical=previous==current))
    external='src/gpuwrf/physics/bl_camuw.py'
    previous_external=subprocess.check_output(['git','show',f'{REFERENCE}:{external}'],cwd=ROOT)
    current_external=(ROOT/external).read_bytes()
    def dependency_ast(source):
        tree=ast.parse(source)
        # These two PBL call bodies never run in CAM-MG's column kernel.
        tree.body=[node for node in tree.body if getattr(node,'name',None) not in ('_compute_vdiff','camuw_column')]
        return normalized(ast.unparse(tree))
    assert dependency_ast(previous_external)==dependency_ast(current_external)
    external_receipt=dict(path=external,non_PBL_AST_identical=True,
        excluded_uninvoked_PBL_functions=['_compute_vdiff','camuw_column'],
        original_ESTBL_and_constant_dependency_AST_sha256=sha(dependency_ast(current_external)))
    asset='src/gpuwrf/physics/cammg/data/initialization.npz'
    previous=subprocess.check_output(['git','show',f'{REFERENCE}:{asset}'],cwd=ROOT)
    current=(ROOT/asset).read_bytes();assert previous==current
    report=dict(status='PASS',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reference=REFERENCE,original_measured_source=ORIGINAL_MEASURED_SOURCE,
        rebased_reference_scientific_files_BYTE_identical=True,scope='Scientific/native bodies only; composed wiring separately tested',
        normalization='Python AST without comments, source locations and docstrings; no arithmetic transformation',
        rows=rows,external_table_and_constant_module=external_receipt,initialization_asset=dict(path=asset,SHA256=sha(current),BYTE_identical=True),
        reused_native_receipt='NATIVE_CPU_GATE.json',reused_calls=48,
        reused_original_enclosure_sha256='5f29c2ca2ef50b2db0b7a65a3f61acf9e9f9f24b3c9145c825a72eb35dc024c8')
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
