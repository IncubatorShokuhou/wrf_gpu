"""Small literal operators shared by generated NTU scalar routines."""
import numpy as np


def maximum(*values, xp=np):
    result = values[0]
    for value in values[1:]:
        result = xp.maximum(result, value)
    return result


def minimum(*values, xp=np):
    result = values[0]
    for value in values[1:]:
        result = xp.minimum(result, value)
    return result


_max = maximum
_min = minimum


def _int(value):
    return np.asarray(value, dtype=np.int32)


def nint(value, xp=np):
    """Fortran NINT: ties away from zero, unlike NumPy/JAX round."""
    return xp.asarray(xp.sign(value)*xp.floor(xp.abs(value)+np.dtype(value.dtype).type(.5)),dtype=np.int32)


def any_true(value,*,xp=np):
    """Resident Boolean reduction supported by the pinned Triton lowerer."""
    return xp.max(xp.asarray(value,dtype=np.int32))>np.int32(0)


def masked_call(selected,routine,inputs,*,xp=np,c=None):
    """CALL a generated routine only when some lane merges its outputs.

    The emitter merges every output of a masked CALL as where(selected, out, old)
    (or a masked put), so with no lane selected nothing the routine returns is
    read and skipping it is exact, like WRF's IF around the CALL. The literal
    NumPy reference always calls.
    """
    if xp is np or getattr(xp,'literal',False):
        return routine(inputs,xp=xp,c=c)
    import jax
    call=lambda d,flag:routine(d,xp=xp,c=c)
    flag=xp.asarray(any_true(selected,xp=xp),dtype=np.int32)
    shapes=jax.eval_shape(call,inputs,flag)
    # Fillers derive from the traced flag (0 in this branch): Triton cannot yield literals.
    unread=lambda d,flag:jax.tree.map(lambda s:xp.broadcast_to(xp.asarray(flag,dtype=s.dtype),s.shape),shapes)
    return jax.lax.cond(flag>np.int32(0),call,unread,inputs,flag)


def _get(value, index):
    indices=index if isinstance(index,tuple) else (index,)
    if isinstance(value,np.ndarray) and any(hasattr(i,'aval') for i in indices):
        import jax.numpy as jnp
        return jnp.asarray(value)[index]
    return value[index]


def get_local(value,index,*,xp=np):
    """Select a trailing aerosol mode from a per-level register tile."""
    if not isinstance(index,tuple):index=(index,)
    from . import _native
    if _native.active.get() and not getattr(value,'_ntu_table_ref',False):
        selected=value
        for i in reversed(index):
            mode=xp.arange(selected.shape[-1],dtype=np.int32)
            selected=xp.where(mode==i,selected,xp.zeros_like(selected))
            if selected.dtype==np.dtype('bool'):
                selected=xp.max(xp.asarray(selected,dtype=np.int32),axis=-1)>np.int32(0)
            else:selected=xp.sum(selected,axis=-1)
        return selected
    # All mode indices are the original short static loops, not level indices.
    return value[(Ellipsis,)+index]


def _set_local(value,index,updated,active,*,xp=np):
    shape=value.shape[-len(index):]
    selector=xp.ones(shape,dtype=bool)
    for axis,i in enumerate(index):
        idxshape=[1]*len(index);idxshape[axis]=shape[axis]
        selector=selector & (xp.arange(shape[axis],dtype=np.int32).reshape(idxshape)==i)
    active=xp.asarray(active);updated=xp.asarray(updated)
    for _ in index:active=active[...,None];updated=updated[...,None]
    return xp.where(selector & active,updated,value)


def get_column(value,index,*,xp=np):
    if getattr(value,'_ntu_table_ref',False):return value.column(index)
    from . import _native
    if _native.active.get():return get_local(value,index,xp=xp)
    return value[...,index]


def put_value(old,new,active,*,xp=np):
    active=xp.asarray(active)
    while active.ndim>0 and active.ndim<new.ndim:active=active[...,None]
    return xp.where(active,new,old)
