"""Delete the real adapter's ze retention (replace by a same-shaped wrong output)."""
import ast
import inspect
import os
from pathlib import Path
import json

def pytest_configure(config):
    from gpuwrf.physics.p3 import adapter
    source=inspect.getsource(adapter.p3_adapter);tree=ast.parse(source)
    class Delete(ast.NodeTransformer):
        count=0
        def visit_Subscript(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='out' and isinstance(node.slice,ast.Constant) and node.slice.value=='ze':
                self.count+=1;node.slice=ast.Constant('re_cloud')
            return self.generic_visit(node)
    edit=Delete();tree=edit.visit(tree);assert edit.count==1
    ast.fix_missing_locations(tree);exec(compile(tree,str(adapter.__file__)+'::<ze-retention-deleted>','exec'),vars(adapter))
    Path(os.environ['T_WRITER_MUTANT_RECEIPT']).write_text(json.dumps({'deletions':1,'mutant_ast':ast.unparse(tree)},indent=2)+'\n')
