"""Revert only the actual producer's interface geometry; expect value ASSERT."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path


def pytest_configure(config):
    from gpuwrf.coupling import physics_couplers as pc
    source=inspect.getsource(pc._aerosol_surface_emission_columns)
    tree=ast.parse(source)
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='z' for t in node.targets):
            node.value=ast.parse('jnp.moveaxis(_mass_level_height_columns(state)[..., :2], -1, 0)',mode='eval').body
            changed+=1
    assert changed==1
    ast.fix_missing_locations(tree)
    exec(compile(tree,str(pc.__file__)+'::<interface-fix-deleted>','exec'),vars(pc))
    Path(os.environ['T_WRITER_MUTANT_RECEIPT']).write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        mutations=changed,mutant_ast=ast.unparse(tree)),indent=2)+'\n')
