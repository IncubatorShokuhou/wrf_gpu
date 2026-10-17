"""Trace-local NTU column geometry, independent of the model state/carry."""
from contextlib import contextmanager
from contextvars import ContextVar

active=ContextVar('ntu_native_column',default=False)
levels=ContextVar('ntu_physical_levels',default=0)


@contextmanager
def column(nz):
    a=active.set(True);b=levels.set(nz)
    try:yield
    finally:levels.reset(b);active.reset(a)


def configuration():return active.get(),levels.get()


def extent(n):return 1<<(n-1).bit_length() if active.get() else n


def model_index(k,nz):
    """Reverse real levels; keep masked padding outside the positive bounds.

    Negative masked store indices wrap in the pinned CPU interpreter and can
    overwrite real outputs with its uninitialized values. This is not a clamp:
    padded indices remain outside the array, and the mask prevents all writes.
    """
    import jax.numpy as jnp
    return jnp.where(k<nz,nz-1-k,k)


def previous(value):
    import jax.numpy as jnp
    from gpuwrf.kernels.phys_thompson_sedimentation import _gather
    k=jnp.arange(value.shape[0],dtype=jnp.int32)
    return _gather.bind(value,(k-1)%value.shape[0])


def point(value,index):
    import jax.numpy as jnp
    k=jnp.arange(value.shape[0],dtype=jnp.int32)
    return jnp.sum(jnp.where(k==index,value,jnp.zeros_like(value)),axis=0)
