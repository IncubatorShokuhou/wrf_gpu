"""Budget-only MYNN twin; production kernels, adapters and returns are untouched.

The caller passes the exact operands also given to the production column call.
Only history arrays leave this module. The production implementation supplies
the equations; a private plume copy adds ENT observation. No twin state is ever
returned to the prognostic carry. Importing this module does not enable it.
"""
from __future__ import annotations

import ast
import hashlib
import inspect

import jax
import jax.numpy as jnp
from gpuwrf.kernels import phys_mynn_columns as production
from gpuwrf.kernels import phys_mynn_budget_plume as _plume_projection
from gpuwrf.physics import mynn_budget as history


def _compile(function, namespace, edit):
    source=inspect.getsource(function)
    tree=ast.parse(source)
    node=tree.body[0]
    assert isinstance(node,ast.FunctionDef) and not node.decorator_list
    edit(node)
    ast.fix_missing_locations(tree)
    exec(compile(tree,__file__,'exec'),namespace)
    return namespace[node.name],hashlib.sha256(source.encode()).hexdigest()


def _build():
    # Copy the namespace, never patch a production function/module/global.
    scope=dict(production.__dict__)
    scope.update(__name__=__name__,_history=history)
    def observe_plume(node):
        imports=[n for n in ast.walk(node) if isinstance(n,ast.ImportFrom)
                 and n.module=='gpuwrf.kernels.phys_mynn_plume']
        assert len(imports)==1
        imports[0].module='gpuwrf.kernels.phys_mynn_budget_plume'
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='dmp_mf_columns_native']
        assert len(calls)==1
    plume,plume_sha=_compile(production._native_edmf_arrays,scope,observe_plume)
    scope['_native_edmf_arrays']=plume
    def observe_history(node):
        ret=node.body[-1]
        assert isinstance(ret,ast.Return) and isinstance(ret.value,ast.Tuple) and len(ret.value.elts)==2
        removed=[n for n in node.body if isinstance(n,ast.Delete) and any(
            isinstance(v,ast.Name) and v.id=='qdiss' for v in n.targets)]
        assert len(removed)==1
        node.body.remove(removed[0])
        node.body.insert(1,ast.parse('_budget_before = state').body[0])
        pos=node.body.index(ret)
        node.body[pos:pos+1]=[
            ast.Assign(targets=[ast.Name('_budget_after',ast.Store())],value=ret.value.elts[0]),
            ast.parse('return (_history.tke_history(state, 2.0*_budget_before.tke, qke_new, turb, _qwt, qdiss, flux, dt, mf=mf, pdk=_pdk) + _history.edmf_history(state,mf) + _history.mean_tendencies(_budget_before,_budget_after,dt))').body[0]]
    advance,advance_sha=_compile(production._advance_native,scope,observe_history)
    scope['_advance_native']=advance
    # Preserve the public native call's casts and any selected runtime operands.
    native,native_sha=_compile(production.step_mynn_columns_native,scope,lambda _node:None)
    return native,dict(plume=plume_sha,advance=advance_sha,native=native_sha,
                       plume_equations=_plume_projection.PRODUCTION_SOURCE_SHA256)


_native,PRODUCTION_SOURCE_SHA256=_build()


def step(state,dt,*,debug=False,surface=None,edmf=False,dx=1000.,barrier='output',**kwargs):
    """Compute only history, with one extra column pass in opt-in programs.

    Output-only barrier is the first design (E204). ``barrier='input'`` is an
    explicit experimental fallback if a compiled identity gate still detects
    neighbouring-fusion changes. Neither mode changes operand values.
    """
    if barrier not in ('none','output','input'):
        raise ValueError('unknown budget twin barrier')
    if barrier=='input':
        state,surface=jax.lax.optimization_barrier((state,surface))
    values=_native(state,dt,debug=debug,surface=surface,edmf=edmf,dx=dx,**kwargs)
    if barrier=='output':
        values=jax.lax.optimization_barrier(values)
    return values
