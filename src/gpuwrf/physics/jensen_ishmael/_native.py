"""Trace-local controls for the per-column Pallas route, never model carries."""
from contextlib import contextmanager
from contextvars import ContextVar

active=ContextVar('jensen_native_column',default=False)
levels=ContextVar('jensen_physical_levels',default=0)


@contextmanager
def column(nz):
    first=active.set(True);second=levels.set(nz)
    try:yield
    finally:levels.reset(second);active.reset(first)


def configuration():return (active.get(),levels.get())


def stack4(values):
    # Installed Triton joins only two singleton final dimensions. A register
    # select constructs four categories directly without join/slice buffers.
    import jax.numpy as jnp
    a,b,c,d=values
    category=jnp.arange(4,dtype=jnp.int32)
    return jnp.where(category==0,a[...,None],jnp.where(category==1,b[...,None],jnp.where(category==2,c[...,None],d[...,None])))
