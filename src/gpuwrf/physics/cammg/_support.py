"""Literal CAM-MG control/array operations (CPU reference; native column backend next).

Fortran indices are one-based. Conditional/loop state is limited to the modified
variables, and every assignment preserves the declared WRF kind. Public inputs
remain REAL4; the original CAM r8 procedures are explicit DOUBLE islands.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from . import _native


def _dtype(kind,xp):
    return dict(f64=xp.float64,f32=xp.float32,i32=xp.int32,bool=xp.bool_)[kind]


def _cast(value,kind,xp):return xp.asarray(value,dtype=_dtype(kind,xp))


def _static(value,kind):return np.asarray(value,dtype=_dtype(kind,np))


def _r(x):
    if isinstance(x,(jax.Array,jax.core.Tracer)):return jnp.asarray(x,dtype=jnp.float32)
    return np.asarray(x,dtype=np.float32)


def _d(x):
    if isinstance(x,(jax.Array,jax.core.Tracer)):return jnp.asarray(x,dtype=jnp.float64)
    return np.asarray(x,dtype=np.float64)


def _int(x):
    if isinstance(x,(jax.Array,jax.core.Tracer)):return jnp.asarray(x,dtype=jnp.int32)
    return np.asarray(x,dtype=np.int32)


def _slice(lower,upper,step):
    return slice(0 if lower is None else lower-1,upper,step)


def _indices(indices):
    return tuple(i if isinstance(i,slice) else i-1 for i in indices)


def _get(array,indices):
    if getattr(array,'_cammg_table_ref',False):return array[_indices(indices)]
    if np.ndim(array)==0:return array  # WRF sequence-associated length-one argument
    if _native.active.get():return _native.get(array,_indices(indices))
    return array[_indices(indices)]


def _put(v,name,indices,value):
    old=v[name]
    if isinstance(old,str):
        v[name]=value;return
    if not hasattr(old,'dtype'):
        old=np.asarray(old,dtype=np.bool_ if isinstance(old,bool) else np.int32 if isinstance(old,int) else np.float64)
    xp=jnp if isinstance(old,(jax.Array,jax.core.Tracer)) or isinstance(value,(jax.Array,jax.core.Tracer)) else np
    value=xp.asarray(value,dtype=old.dtype)
    if indices is None:
        v[name]=xp.broadcast_to(value,old.shape) if old.shape else value
    elif _native.active.get():v[name]=_native.put(old,_indices(indices),value)
    elif xp is np:
        result=old.copy();result[_indices(indices)]=value;v[name]=result
    else:
        v[name]=old.at[_indices(indices)].set(value)


def _cond(condition,yes,no,carry,xp):
    if xp is np or getattr(xp,'_literal_numpy',False):return (yes if bool(condition) else no)(carry)
    if not isinstance(condition,(jax.Array,jax.core.Tracer)):
        return (yes if bool(condition) else no)(carry)
    if _native.active.get():
        anchor=_native.anchor.get()
        if anchor is not None:condition=lax.select(anchor==anchor,condition,condition)
        # The installed Triton cond emitter yields branch literals without
        # materializing them as MLIR Values. An identity select preserves all
        # bits while producing an IR value; Triton folds it after lowering.
        def materialize(branch):
            def run(value):
                result=branch(value)
                return jax.tree_util.tree_map(lambda x:lax.select(jnp.broadcast_to(condition,jnp.shape(x)),x,x),result)
            return run
        return lax.cond(condition,materialize(yes),materialize(no),carry)
    return lax.cond(condition,yes,no,carry)


def _materialize(values,anchor):
    if not _native.active.get():return values
    return jax.tree_util.tree_map(lambda x:lax.select(jnp.broadcast_to(anchor==anchor,jnp.shape(x)),x,x),values)


def _neg(value):
    if _native.active.get() and _native.anchor.get() is not None:
        value=_materialize(value,_native.anchor.get())
    return -value


def _loop(start,end,step,body,carry,xp,jump_index=None):
    if xp is np or getattr(xp,'_literal_numpy',False):
        for k in range(int(start),int(end)+(1 if step>0 else -1),int(step)):
            if jump_index is not None and carry[jump_index]!=0:break
            carry=body(k,carry)
        return carry
    count=jnp.maximum(0,(end-start)//step+1)
    if jump_index is not None:
        def cond(c):return (c[0]<count)&(c[1][jump_index]==0)
        def advance(c):return c[0]+1,_materialize(body(start+c[0]*step,c[1]),c[0])
        return lax.while_loop(cond,advance,(jnp.int32(0),carry))[1]
    return lax.fori_loop(0,count,lambda i,c:_materialize(body(start+i*step,c),i),carry)


def _pow(x,y,xp):
    return xp.power(x,y)


def _powi(x,n):
    if n==0:return x*0+1
    if n<0:return 1/_powi(x,-n)
    if n==1:return x
    if n%2:return _powi(x,n-1)*x
    half=_powi(x,n//2)
    return half*half


def _retry(target,body,carry,xp,jump_index):
    def clear(c):return tuple(_int(0) if i==jump_index else x for i,x in enumerate(c))
    if xp is np or getattr(xp,'_literal_numpy',False):
        first=True
        while first or carry[jump_index]==target:
            first=False;carry=body(clear(carry))
        return carry
    return lax.while_loop(lambda c:c[0]|(c[1][jump_index]==target),
                          lambda c:_materialize((jnp.bool_(False),body(clear(c[1]))),c[0]),
                          (jnp.bool_(True),carry))[1]


def _zeros(shape,kind,xp):
    if _native.active.get():shape=_native.storage(shape)
    return xp.zeros(shape,dtype=_dtype(kind,xp))


def _literal_array(values,kind,xp):
    if not _native.active.get():return xp.asarray(values,dtype=_dtype(kind,xp))
    index=jnp.arange(_native.width(len(values)),dtype=jnp.int32)
    out=jnp.zeros(index.shape,dtype=_dtype(kind,jnp))
    for i,value in enumerate(values):out=jnp.where(index==i,jnp.asarray(value,dtype=out.dtype),out)
    return out
