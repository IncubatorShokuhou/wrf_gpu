"""Trace-local register views for one CAM-MG column; no model carry/state flags."""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import jax.numpy as jnp

active=ContextVar('cammg_native_column',default=False)
levels=ContextVar('cammg_physical_levels',default=0)
anchor=ContextVar('cammg_dynamic_anchor',default=None)


@contextmanager
def column(nz,dynamic_anchor=None):
    a=active.set(True);n=levels.set(nz);q=anchor.set(dynamic_anchor)
    try:yield
    finally:anchor.reset(q);levels.reset(n);active.reset(a)


def width(n):
    n=int(n)
    if active.get() and n==levels.get():n+=1
    return 1<<(n-1).bit_length() if n>0 else 1


def storage(shape):return tuple(width(n) for n in shape)


def gather_axis(value,indices,axis):
    from gpuwrf.kernels.phys_thompson_sedimentation import _gather
    # Static outer dimensions are small (1, 2, 4, 16); gather operates on the
    # existing register axis, never a domain-sized intermediate.
    if value.ndim==1:return _gather.bind(value,indices)
    if axis!=0:
        order=(axis,)+tuple(i for i in range(value.ndim) if i!=axis)
        inverse=tuple(order.index(i) for i in range(value.ndim))
        return jnp.transpose(gather_axis(jnp.transpose(value,order),indices,0),inverse)
    trailing=value.shape[1:]
    # A new gather lowering for arbitrary rank avoids slice/concatenate tiles.
    return _gather_nd.bind(value,indices)


from jax.extend import core
from jax.interpreters import mlir
from jax._src.pallas.triton import lowering
from jaxlib.triton import dialect as tt
import jax

_gather_nd=core.Primitive('cammg_register_gather_axis0')
_gather_nd.def_impl(lambda x,i:jnp.take(x,i,axis=0,mode='clip'))
_gather_nd.def_abstract_eval(lambda x,i:jax.core.ShapedArray(i.shape+x.shape[1:],x.dtype))
mlir.register_lowering(_gather_nd,mlir.lower_fun(lambda x,i:jnp.take(x,i,axis=0,mode='clip'),multiple_results=False))


@lowering.register_lowering(_gather_nd)
def _lower_gather(ctx,x,i):
    # Triton tt.gather requires a same-rank index tile.
    from jaxlib.mlir import ir
    shape=ctx.avals_in[0].shape
    for _ in shape[1:]:i=tt.expand_dims(i,axis=len(ir.RankedTensorType(i.type).shape))
    target=ir.RankedTensorType.get((ctx.avals_in[1].shape[0],*shape[1:]),ir.IntegerType.get_signless(32))
    i=tt.broadcast(target,i)
    return tt.gather(x,i,axis=0)


def flip(value,n):
    index=jnp.arange(value.shape[0],dtype=jnp.int32)
    src=jnp.asarray(n-1,jnp.int32)-index
    result=gather_axis(value,jnp.clip(src,0,value.shape[0]-1),0)
    return jnp.where(index<n,result,jnp.zeros_like(result))


def get(value,indices):
    for axis in range(len(indices)-1,-1,-1):
        index=indices[axis]
        size=value.shape[axis]
        if isinstance(index,slice):
            start=0 if index.start is None else index.start
            stop=size if index.stop is None else index.stop
            if index.step not in (None,1):raise ValueError('CAM native slice step must be one')
            j=jnp.arange(size,dtype=jnp.int32)
            shifted=value if not isinstance(start,jax.core.Tracer) and int(start)==0 else gather_axis(value,jnp.clip(j+start,0,size-1),axis)
            shape=[1]*value.ndim;shape[axis]=size
            value=jnp.where(j.reshape(shape)<stop-start,shifted,jnp.zeros_like(shifted))
        else:
            shape=[1]*value.ndim;shape[axis]=size
            select=jnp.arange(size,dtype=jnp.int32).reshape(shape)==index
            if value.dtype==jnp.bool_:
                value=jnp.max(jnp.where(select,value.astype(jnp.int32),jnp.int32(0)),axis=axis)!=0
            else:value=jnp.sum(jnp.where(select,value,jnp.zeros_like(value)),axis=axis,dtype=value.dtype)
    return value


def put(old,indices,value):
    select=jnp.ones(old.shape,dtype=jnp.bool_)
    kept=tuple(size for axis,size in enumerate(old.shape) if axis>=len(indices) or isinstance(indices[axis],slice))
    result=jnp.broadcast_to(value,kept)
    # Expand reduced scalar-index axes back to the destination rank first.
    for axis,index in enumerate(indices):
        if not isinstance(index,slice):result=jnp.expand_dims(result,axis)
    for axis,index in enumerate(indices):
        size=old.shape[axis];shape=[1]*old.ndim;shape[axis]=size
        j=jnp.arange(size,dtype=jnp.int32)
        if isinstance(index,slice):
            start=0 if index.start is None else index.start;stop=size if index.stop is None else index.stop
            select=select&((j.reshape(shape)>=start)&(j.reshape(shape)<stop))
            if result.ndim and result.shape[axis]>1:
                result=gather_axis(result,jnp.clip(j-start,0,result.shape[axis]-1),axis)
        else:select=select&(j.reshape(shape)==index)
    return jnp.where(select,result,old)
