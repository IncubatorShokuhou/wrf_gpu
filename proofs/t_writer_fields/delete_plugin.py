"""Delete one real source operation in-process; immutable disk source stays clean.

Run with T_WRITER_MUTANT=rimi|goddard|ccn and pytest -p delete_plugin. A kill
requires a genuine value/field assertion and zero setup errors (E202).
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path


def pytest_configure(config):
    kind=os.environ['T_WRITER_MUTANT']
    if kind in ('rimi','goddard'):
        from gpuwrf.coupling import scan_adapters, goddard_radiation
        module=scan_adapters if kind=='rimi' else goddard_radiation
        name='sbu_ylin_adapter' if kind=='rimi' else 'paired_radiation'
        keyword='rimi' if kind=='rimi' else 'radius_inputs'
        source=inspect.getsource(getattr(module,name))
        tree=ast.parse(source)
        removed=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                before=len(node.keywords)
                node.keywords=[k for k in node.keywords if k.arg!=keyword]
                removed += before-len(node.keywords)
    elif kind=='ccn':
        from gpuwrf.io import producer_history as module
        name='inactive_names'
        source=inspect.getsource(getattr(module,name));tree=ast.parse(source)
        class Delete(ast.NodeTransformer):
            removed=0
            def visit_Expr(self,node):
                if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                        and node.value.func.attr=='extend' and node.value.args
                        and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=='CCN_FIELDS'):
                    self.removed+=1
                    return ast.copy_location(ast.Pass(),node)
                return self.generic_visit(node)
        edit=Delete();tree=edit.visit(tree);removed=edit.removed
    else:
        raise ValueError(kind)
    assert removed==1, f'expected one real source deletion, got {removed}'
    ast.fix_missing_locations(tree)
    # Keep the actual module globals: copying the dictionary would shield
    # helper monkeypatches in a retention test and change more than the delete.
    namespace=vars(module);exec(compile(tree,str(module.__file__)+'::<deleted>', 'exec'),namespace)
    setattr(module,name,namespace[name])
    record=dict(kind=kind,module=str(module.__file__),function=name,operations_deleted=removed,
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        mutant_ast=ast.unparse(tree),scope='actual source deletion; no disk mutation')
    Path(os.environ['T_WRITER_MUTANT_RECEIPT']).write_text(json.dumps(record,indent=2)+'\n')
