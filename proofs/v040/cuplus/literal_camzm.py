"""Qualification-only NumPy/libm executor for actual candidate CAM equations."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
from literal_kfcup import Literal,RealBackend,binary,where,get,set_,fori,while_,tree_map


def executor(source):
    real=RealBackend()
    fake=SimpleNamespace(lax=SimpleNamespace(fori_loop=fori,while_loop=while_,
        select=lambda pred,yes,no:np.where(pred,yes,no),
        cond=lambda pred,yes,no,arg:(yes if bool(pred) else no)(arg)),
        tree=SimpleNamespace(map=tree_map),
        tree_util=SimpleNamespace(tree_map=tree_map),enable_x64=lambda value:nullcontext())
    ns=dict(jnp=real,jax=fake,_r32=np.float32,_r64=np.float64,_binary=binary,
            _where=where,_get=get,_set=set_,_roll=lambda a,n:np.roll(a,int(n)),
            _get_vector=lambda a,i:a[i])
    path=source/'src/gpuwrf/kernels/phys_camzm_column.py'
    tree=ast.parse(path.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign):nodes.append(Literal(double=True,respect_with=True).visit(node))
        elif isinstance(node,ast.FunctionDef) and node.name.startswith('_') and node.name!='_get_vector':
            node.decorator_list=[];nodes.append(Literal(double=True,respect_with=True).visit(node))
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns['_camzm_cps']
