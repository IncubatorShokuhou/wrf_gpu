"""Budget-only projection of production plume equations, plus ENT observation.

Never mutate production functions/globals. Project the current wrappers too,
so selected runtime operands (e.g. entrainment) reach the SAME scalar stack as
production instead of becoming duplicated budget-only constants.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from gpuwrf.kernels import phys_mynn_plume as production


def _build(*, _surface_ent=.001):
    scope=dict(production.__dict__)
    scope['__name__']=__name__
    hashes={}
    def project(name,edit=lambda _node:None):
        source=inspect.getsource(getattr(production,name))
        tree=ast.parse(source);node=tree.body[0]
        assert isinstance(node,ast.FunctionDef) and not node.decorator_list
        edit(node);ast.fix_missing_locations(tree)
        exec(compile(tree,__file__,'exec'),scope)
        hashes[name]=hashlib.sha256(source.encode()).hexdigest()
    def level(node):
        ret=node.body[-1]
        assert isinstance(ret,ast.Return) and isinstance(ret.value,ast.Tuple) and len(ret.value.elts)==3
        ret.value.elts[1]=ast.parse("emits + (jnp.where(still,ent,c['c0']),)",mode='eval').body
    project('_plume_level',level)
    def kernel(node):
        counts=[n for n in node.body if isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=='n_sums' for t in n.targets)]
        assert len(counts)==1
        counts[0].value=ast.BinOp(counts[0].value,ast.Add(),ast.Constant(1))
        loops=[n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
               and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='zip']
        assert len(loops)==1
        assert isinstance(loops[0].iter.args[1],ast.Name) and loops[0].iter.args[1].id=='values'
        loops[0].iter.args[1]=ast.parse('values + (upa*emits[-1],)',mode='eval').body
    project('_plume_sums_kernel',kernel)
    def sums(node):
        calls=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='outs' for t in n.targets)]
        assert len(calls)==1
        node.body.insert(calls[0],ast.parse("names=names+('aent',)").body[0])
    project('fused_plume_sums',sums)
    def assemble(node):
        assert isinstance(node.body[-1],ast.Return)
        # WRF :5806 initializes ENT=.001, including kts (no rise update there).
        # Private factory argument is a deletion-test seam, not a public knob.
        extra=ast.parse(f"""aent0=jnp.sum(upa0*jnp.asarray({_surface_ent!r},upa0.dtype),axis=-1)[None,:]
aent=jnp.concatenate((aent0,sums['aent'],zero_top),axis=0)
result['edmf_ent']=keep(jnp.where(has_a,aent*adjustment/safe,0.)).T
""").body
        node.body[-1:-1]=extra
    project('_assemble_from_sums',assemble)
    def impl(node):
        node.body.insert(1,ast.parse("""if not (plume_sums_enabled() or qni is not None):
    raise ValueError('MYNN budget twin requires the native summed-plume path')
""").body[0])
    project('_dmp_mf_columns_impl',impl)
    project('dmp_mf_columns_native')
    return scope['dmp_mf_columns_native'],hashes


dmp_mf_columns_native,PRODUCTION_SOURCE_SHA256=_build()
