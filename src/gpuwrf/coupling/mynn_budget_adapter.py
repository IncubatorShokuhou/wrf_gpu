"""Opt-in MYNN caller projection; production adapter source stays unchanged.

Prepare/initialize once. The production call and diagnostic twin receive the
same argument objects. Only the production call supplies MynnPBLSourceLeaves;
the twin supplies a separate history group, never prognostic State updates.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect

from gpuwrf.coupling import physics_couplers as production
from gpuwrf.diagnostics.mynn_budget import from_columns


def _column_pair(production_call,twin_call,*args,**kwargs):
    """Evaluate argument expressions once, then forward their identical values."""
    physical,pblh=production_call(*args,**kwargs)
    history=twin_call(*args,**kwargs)
    return physical,pblh,history


@lru_cache(maxsize=4)
def _build(edmf,observe):
    scope=dict(production.__dict__)
    scope.update(__name__=__name__,_MYNN_EDMF=edmf)
    function=production.mynn_adapter_with_source_leaves
    source=inspect.getsource(function)
    tree=ast.parse(source);node=tree.body[0]
    assert isinstance(node,ast.FunctionDef) and not node.decorator_list
    if observe:
        from gpuwrf.physics.mynn_budget_twin import step
        call=scope['step_mynn_pbl_column_with_pblh']
        def pair(*args,**kwargs):return _column_pair(call,step,*args,**kwargs)
        scope['_budget_column_pair']=pair
        matches=[n for n in node.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Name) and n.value.func.id=='step_mynn_pbl_column_with_pblh']
        assert len(matches)==1,'MYNN production column-call seam changed'
        assign=matches[0]
        assert len(assign.targets)==1 and isinstance(assign.targets[0],ast.Tuple)
        assert [n.id for n in assign.targets[0].elts]==['out_b','pblh_b']
        assign.targets[0].elts.append(ast.Name('_budget_values',ast.Store()))
        assign.value.func.id='_budget_column_pair'
        ret=node.body[-1]
        assert isinstance(ret,ast.Return) and isinstance(ret.value,ast.Call)
        assert isinstance(ret.value.func,ast.Name) and ret.value.func.id=='MynnPBLSourceLeaves'
        ret.value=ast.Tuple([ret.value,ast.Name('_budget_values',ast.Load())],ast.Load())
        # Keep all production postprocessing and fields exactly as written. The
        # public wrapper below packs history without touching the source result.
    ast.fix_missing_locations(tree)
    exec(compile(tree,__file__,'exec'),scope)
    return scope[function.__name__],hashlib.sha256(source.encode()).hexdigest()


def mynn_sources_with_history(state,dt,grid=None,*,budget_config,**kwargs):
    """Separate caller for selected histories or the nondefault EDMF-off mode."""
    config=budget_config
    observe=bool(config['tke_budget'] or config['bl_mynn_output'])
    edmf=bool(config['bl_mynn_edmf'])
    if not observe and edmf:
        return production.mynn_adapter_with_source_leaves(state,dt,grid,**kwargs),None
    if observe and edmf and not production._MYNN_EDMF:
        raise ValueError('MYNN budget bl_mynn_edmf=1 conflicts with the legacy module switch physics_couplers._MYNN_EDMF=False; restore it or select bl_mynn_edmf=0')
    if (production._mynn_real_dtype()!=production.jnp.float32
            or not production._mynn_env_bool('GPUWRF_MYNN_FP32_COLUMNS',False)):
        raise ValueError('MYNN budgets/EDMF-off require native fp32 MYNN columns')
    adapter,_source_sha=_build(edmf,observe)
    if not observe:return adapter(state,dt,grid,**kwargs),None
    physical,values=adapter(state,dt,grid,**kwargs)
    ny,nx=state.theta.shape[1:]
    return physical,from_columns(values,ny,nx,config)
