"""Literal REAL operations: native RN32; CPU barriers forbid FMA/reassociation.

CPU transcendental evaluation widens only for rounding emulation: XLA's REAL
approximations differ from scalar WRF libm. The GPU path uses libdevice REAL,
and all prognostic/temporary arrays remain REAL32 there.
"""
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.backends.pallas import cuda_native
from jax.experimental.pallas import triton as pt
from jax.extend import core
from jax._src.pallas.triton import lowering
from jax._src.interpreters import mlir
from jaxlib.triton import dialect as tt
from jaxlib.mlir import ir


_rn=core.Primitive('milbrandt_rn32')
_rn.def_abstract_eval(lambda a,b,op:jax.core.ShapedArray(a.shape,jnp.float32))
_libmath=core.Primitive('milbrandt_libdevice')
_libmath.def_abstract_eval(lambda x,name:x)


@lowering.register_lowering(_libmath)
def _libmath_lower(ctx,x,*,name):
    aval=ctx.avals_out[0]
    x=lowering._ensure_ir_value(x,ctx.avals_in[0])
    x=lowering._bcast_to(x,aval.shape)
    typ=lowering._dtype_to_ir_type(aval.dtype)
    if aval.shape:typ=ir.RankedTensorType.get(aval.shape,typ)
    symbol='__nv_'+name+('f' if aval.dtype==jnp.float32 else '')
    return tt.extern_elementwise(typ,[x],libname='',libpath='',symbol=symbol,pure=True)


def real(x,dtype):
    # Materialize static literals on the host, so Pallas never receives an
    # unlowerable neg(TypedNdArray). Dynamic REAL conversions stay on device.
    if isinstance(x,(int,float,np.number,np.ndarray)):
        return np.asarray(x,dtype)
    return jnp.asarray(x,dtype)


def _not(x):
    if isinstance(x,(bool,np.bool_,np.ndarray)):
        return np.logical_not(x)
    return jnp.logical_not(x)


@lowering.register_lowering(_rn)
def _rn_lower(ctx,a,b,*,op):
    # Pallas's stock inline_asm lowering does not materialize literal operands.
    # MY2 has scalar constant arithmetic as well as profile/profile operations.
    a=lowering._ensure_ir_value(a,ctx.avals_in[0])
    b=lowering._ensure_ir_value(b,ctx.avals_in[1])
    shape=ctx.avals_out[0].shape
    a=lowering._bcast_to(a,shape);b=lowering._bcast_to(b,shape)
    typ=lowering._dtype_to_ir_type(jnp.float32)
    if shape:typ=ir.RankedTensorType.get(shape,typ)
    return tt.ElementwiseInlineAsmOp([typ],
        op+'.rn.f32 $0, $1, $2;', '=f,f,f', pure=True,packed_element=1,args=[a,b]).results[0]


def _op(op,a,b):
    dtype=jnp.result_type(a,b)
    a,b=jnp.broadcast_arrays(jnp.asarray(a,dtype),jnp.asarray(b,dtype))
    if cuda_native() and dtype==jnp.float32:
        return _rn.bind(a,b,op=op)
    value={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[op]()
    if cuda_native(): return value  # WRF's explicit DBLE rain-intercept expression
    return jax.lax.optimization_barrier(jnp.nextafter(value,value))


def _trans(name,x):
    if cuda_native():
        return _libmath.bind(x,name=name)
    return getattr(jnp,name)(x.astype(jnp.float64)).astype(x.dtype)


def _exp(x): return _trans('exp',x)
def _log(x): return _trans('log',x)
def _log10(x): return _trans('log10',x)


def _pow(a,b):
    if isinstance(b,int):
        if b==0: return jnp.ones_like(a)
        if b==1: return a
        if b<0: return _op('div',jnp.ones_like(a),_pow(a,-b))
        # gfortran's integer exponentiation uses exponentiation by squaring.
        half=_pow(a,b//2)
        square=_op('mul',half,half)
        return _op('mul',square,a) if b%2 else square
    if cuda_native(): return a**b
    a,b=jnp.broadcast_arrays(jnp.asarray(a),jnp.asarray(b))
    return (a.astype(jnp.float64)**b.astype(jnp.float64)).astype(a.dtype)
