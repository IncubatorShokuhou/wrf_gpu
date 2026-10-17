"""Proof-only structural CAM-MG native-kernel variants (lane t-cammgws).

Each variant changes ONLY how the native (Pallas/Triton) column is lowered:
loop/branch structure or Triton compiler parameters. Physical formulas,
operation order inside each body, dtypes and the CPU literal reference path are
untouched. install() must run before the first trace; it monkeypatches module
globals in the measuring process only (product files stay byte-identical).

Variants (comma-combinable):
  stages1  - Triton software pipelining off (num_stages=1; default 3 on CUDA)
  inline1  - native jump loops with a STATIC trip count of 1 (every ncol=1 loop)
             are emitted inline behind the same jump guard (no scf.while)
  forguard - native jump loops are emitted as a fixed-trip fori (scf.for) whose
             body runs behind lax.cond(jump==0); identical iteration sequence
             because no iteration after a jump can clear the jump flag
"""
import types

VARIANTS=('stages1','inline1','forguard','actvec','distinct','kcall','nofma')

# nofma - stock-Triton route with LLVM FP contraction off (enable_fp_fusion=False).

# kcall - compile ROUTE only: the cammg_mp11_column pallas_call takes JAX's existing
#   fallback lowering (jax/_src/pallas/triton/pallas_call_registration.py, the
#   branch after `if True:`): TTIR -> PTX via the cuda PJRT plugin Triton compiler
#   (compile_triton_to_asm) -> jaxlib `triton_kernel_call` custom call, instead of
#   XLA's in-graph `__gpu$xla.gpu.triton` CompileTritonToLLVM. Kernel jaxpr/TTIR
#   are unchanged; every other pallas_call keeps the default route.

# distinct - every native lax.cond condition gets a unique always-true conjunct
#   (itimestep anchor != -1-n, itimestep>=1) that no folder/CSE can prove equal,
#   so MLIR scf.if canonicalization (CombineIfs etc.) cannot merge the hundreds of
#   adjacent `_jump==0` ifs into one region. Branch bodies unchanged.

# actvec - the LTRUE activity block (WRF module_cam_mp_cldwat2m_micro.F:1307-1312:
#   do i=1,ncol; ltrue(i)=0; do k=1,pver; if (qc>=qsmall .or. qi>=qsmall .or.
#   cmei>=qsmall) ltrue(i)=1 ...) is emitted on the native path as the same f64
#   predicates reduced over the real (unpadded) levels, guarded by the same
#   _jump==0 condition through selects; post-loop i/k = ncol/pver as the port's
#   _loop leaves them. The literal CPU reference keeps the original statements.
_ACTVEC='''
if _cammgws_native.active.get() and int(v['ncol'])>=1 and int(v['pver'])>=1:
    _qs=v['qsmall']
    _hit=xp.logical_or(xp.logical_or(v['qc']>=_qs,v['qi']>=_qs),v['cmei']>=_qs)
    _lev=xp.arange(v['qc'].shape[-1],dtype=xp.int32)<v['pver']
    _act=xp.max(xp.where(xp.logical_and(_hit,_lev[None,:]),xp.int32(1),xp.int32(0)),axis=1)
    _col=xp.arange(v['ltrue'].shape[0],dtype=xp.int32)<v['ncol']
    _go=v['_jump']==0
    v['ltrue']=xp.where(xp.logical_and(_go,_col),_act.astype(v['ltrue'].dtype),v['ltrue'])
    v['i']=xp.where(_go,xp.int32(v['ncol']),v['i']).astype(v['i'].dtype)
    v['k']=xp.where(_go,xp.int32(v['pver']),v['k']).astype(v['k'].dtype)
else:
    pass
'''


def _patch_activity(micro,_native):
    import ast,inspect
    source=ast.parse(inspect.getsource(micro.mmicro_pcond));function=source.body[0]
    body=function.body
    def assigns(n,name):
        return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    start=next(i for i,n in enumerate(body) if assigns(n,'_outer117'))
    call=next(i for i in range(start,len(body)) if assigns(body[i],'_result') and
              isinstance(body[i].value,ast.Call) and any(isinstance(a,ast.Name) and a.id=='_yes117' for a in body[i].value.args))
    end=call+1
    assert ast.unparse(body[end]).startswith("v.update(zip(('i', 'k', 'ltrue'), _result))"),ast.unparse(body[end])
    block=ast.parse(_ACTVEC).body[0]
    block.orelse=body[start:end+1]
    function.body=body[:start]+[block]+body[end+1:]
    micro.__dict__['_cammgws_native']=_native
    exec(compile(ast.fix_missing_locations(source),'<t-cammgws-actvec-micro>','exec'),micro.__dict__)
    return dict(replaced_statements=end+1-start,first=start)


def _static_int(x):
    import numpy as np
    return isinstance(x,(int,np.integer)) and not isinstance(x,bool)


def install(spec):
    names=tuple(n for n in spec.split(',') if n and n!='none')
    unknown=[n for n in names if n not in VARIANTS]
    if unknown:raise ValueError(f'unknown variants {unknown}')
    import jax
    import jax.numpy as jnp
    from jax import lax
    from gpuwrf.physics.cammg import _support,_native
    from gpuwrf.physics.cammg import micro,macro,aerosol,cloud,gamma,ice_activation,error_function
    from gpuwrf.kernels import phys_cammg
    original_loop=_support._loop
    report={'variants':names,'patched_modules':[],'physical_math_changed':False}

    def loop(start,end,step,body,carry,xp,jump_index=None):
        literal=xp is not jnp or getattr(xp,'_literal_numpy',False)
        if literal or not _native.active.get() or jump_index is None:
            return original_loop(start,end,step,body,carry,xp,jump_index=jump_index)
        static=all(_static_int(v) for v in (start,end,step))
        if 'inline1' in names and static and max(0,(int(end)-int(start))//int(step)+1)==1:
            def run(c):return _support._materialize(body(jnp.int32(int(start)),c),jnp.int32(0))
            return _support._cond(carry[jump_index]==0,run,lambda c:c,carry,xp)
        if 'forguard' in names:
            count=jnp.maximum(0,(end-start)//step+1)
            def advance(i,c):
                def run(cc):return body(start+i*step,cc)
                return _support._materialize(_support._cond(c[jump_index]==0,run,lambda cc:cc,c,xp),i)
            return lax.fori_loop(0,count,advance,carry)
        return original_loop(start,end,step,body,carry,xp,jump_index=jump_index)

    if 'inline1' in names or 'forguard' in names:
        _support._loop=loop
        for module in (micro,macro,aerosol,cloud,gamma,ice_activation,error_function):
            if getattr(module,'_loop',None) is original_loop:
                module._loop=loop;report['patched_modules'].append(module.__name__)
    if 'distinct' in names:
        original_cond=_support._cond
        counter=[0]
        def cond(condition,yes,no,carry,xp):
            anchor=_native.anchor.get()
            if (xp is jnp and not getattr(xp,'_literal_numpy',False) and _native.active.get() and anchor is not None
                    and isinstance(condition,(jax.Array,jax.core.Tracer))):
                counter[0]+=1
                condition=jnp.logical_and(condition,anchor!=jnp.int32(-1-counter[0]))
            return original_cond(condition,yes,no,carry,xp)
        _support._cond=cond
        for module in (micro,macro,aerosol,cloud,gamma,ice_activation,error_function):
            if getattr(module,'_cond',None) is original_cond:
                module._cond=cond;report['patched_modules'].append(module.__name__+'._cond')
        report['distinct_counter']=counter
    if 'nofma' in names:
        from gpuwrf.kernels import _triton_kernel_call as route
        real_params=route.KernelCallCompilerParams
        phys_cammg.KernelCallCompilerParams=lambda **kw:real_params(**{**kw,'enable_fp_fusion':False})
        report['patched_modules'].append('phys_cammg KernelCallCompilerParams(enable_fp_fusion=False)')
    if 'kcall' in names:
        import inspect
        import jax._src.pallas.triton.pallas_call_registration as reg
        import jax._src.pallas.core as pcore
        from jax._src.pallas.triton import core as triton_core
        source=inspect.getsource(reg.pallas_call_lowering)
        assert source.count('  if True:')==1,'unexpected pallas_call_lowering source'
        source=source.replace('  if True:','  if False:',1).replace('def pallas_call_lowering(','def _cammgws_kcall_lowering(',1)
        namespace=dict(reg.__dict__);exec(compile(source,'<t-cammgws-kcall>','exec'),namespace)
        fallback=namespace['_cammgws_kcall_lowering'];default=reg.pallas_call_lowering
        seen=report.setdefault('kcall_routed',[])
        def rule(ctx,*in_nodes,jaxpr,**kw):
            name=getattr(jaxpr.debug_info,'func_name','') or ''
            if 'cammg_mp11_column' in name:
                seen.append(name);return fallback(ctx,*in_nodes,jaxpr=jaxpr,**kw)
            return default(ctx,*in_nodes,jaxpr=jaxpr,**kw)
        assert pcore._backend_lowering_rules[triton_core.CompilerParams][0] is default
        pcore._backend_lowering_rules[triton_core.CompilerParams]=(rule,'gpu')
        report['patched_modules'].append('pallas lowering rule (cammg_mp11_column -> triton_kernel_call)')
    if 'actvec' in names:
        report['actvec']=_patch_activity(micro,_native);report['patched_modules'].append('micro.mmicro_pcond[activity]')
    if 'stages1' in names:
        real=phys_cammg.plt
        proxy=types.SimpleNamespace(**{k:getattr(real,k) for k in dir(real) if not k.startswith('__')})
        proxy.CompilerParams=lambda **kw:real.CompilerParams(**{**kw,'num_stages':1})
        phys_cammg.plt=proxy;report['patched_modules'].append('gpuwrf.kernels.phys_cammg.plt')
    return report
