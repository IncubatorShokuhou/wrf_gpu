"""Host evaluation of the *unchanged port bodies*, with scalar libm REAL4.

Diagnostic backend only: no JAX/XLA approximation, FMA, FTZ, or reassociation.
This isolates source ordering from backend arithmetic. Compiled JAX results
must independently pass full_compare.py; this is not a runtime implementation.
"""
from contextlib import contextmanager
import ctypes
from types import SimpleNamespace

import jax
import numpy as np

_libm = ctypes.CDLL('libm.so.6')
for _name in ('expf', 'logf', 'tanhf', 'powf'):
    _fn = getattr(_libm, _name)
    _fn.restype = ctypes.c_float
    _fn.argtypes = [ctypes.c_float] * (2 if _name == 'powf' else 1)


def wrap(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).view(Real)
    return value


def libm(name, *args):
    arrays = np.broadcast_arrays(*(np.asarray(a, np.float32) for a in args))
    out = np.empty(arrays[0].shape, np.float32)
    for index in np.ndindex(out.shape):
        out[index] = getattr(_libm, name)(*(float(a[index]) for a in arrays))
    return wrap(out)


class Real(np.ndarray):
    __array_priority__ = 1000

    def __getitem__(self, key):
        return wrap(super().__getitem__(key))

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        args = tuple(np.asarray(a) if isinstance(a, Real) else a for a in args)
        if any(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in args):
            # JAX promotes integer/weak scalar operands to the REAL operand.
            args = tuple(np.asarray(a, np.float32) if np.asarray(a).dtype.kind in 'ifu' else a for a in args)
        if ufunc is np.power and method == '__call__':
            exponent = np.asarray(args[1])
            if np.all(exponent == 2): return wrap(args[0]*args[0])
            if np.all(exponent == 3): return wrap(args[0]*args[0]*args[0])
            if np.all(exponent == .5): return wrap(np.sqrt(args[0]))
            return libm('powf', *args)
        result = getattr(ufunc, method)(*args, **kwargs)
        return tuple(wrap(x) for x in result) if isinstance(result, tuple) else wrap(result)

    @property
    def at(self): return At(self)


class At:
    def __init__(self, value, key=None): self.value, self.key = value, key
    def __getitem__(self, key): return At(self.value, key)
    def set(self, value):
        result = self.value.copy()
        result[self.key] = value
        return result
    def add(self, value): return self.set(self.value[self.key]+value)


class NumpyReal:
    def __getattr__(self, name):
        if name in ('float32', 'int32', 'bool_', 'integer'): return getattr(np, name)
        if name in ('exp', 'log', 'tanh'): return lambda *a: libm(name+'f', *a)
        fn = getattr(np, name)
        def call(*args, **kwargs):
            # enable_x64(False): even a where with two Python float literals
            # has REAL32 output. NumPy otherwise gives these literals fp64.
            if name == 'where':
                args=tuple(np.float32(x) if isinstance(x,float) else x for x in args)
            if name == 'arange' and 'dtype' not in kwargs: kwargs['dtype'] = np.int32
            if name == 'sum' and args[0].dtype == np.int32: kwargs.setdefault('dtype', np.int32)
            result=fn(*args, **kwargs)
            if name == 'asarray' and result.dtype == np.float64: result=result.astype(np.float32)
            return wrap(result)
        return call


def fori(lower, upper, body, carry):
    for index in range(int(lower), int(upper)): carry = body(np.int32(index), carry)
    return carry


def while_loop(test, body, carry):
    while bool(test(carry)): carry = body(carry)
    return carry


def scan(body, carry, xs):
    leaves = jax.tree.leaves(xs)
    outputs = []
    for index in range(len(leaves[0])):
        carry, out = body(carry, jax.tree.map(lambda x: x[index], xs))
        outputs.append(out)
    return carry, jax.tree.map(lambda *x: wrap(np.stack(x)), *outputs)


def vmap(fn):
    return lambda xs: jax.tree.map(lambda *x: wrap(np.stack(x)), *(fn(x) for x in xs))


@contextmanager
def source_backend():
    """Redirect only module globals; execute the same Python source functions."""
    from gpuwrf.physics import pbl_keps, pbl_eeps
    modules = (pbl_keps, pbl_eeps)
    saved = [(m.jax, m.jnp) for m in modules]
    proxy = SimpleNamespace(enable_x64=jax.enable_x64, tree=jax.tree, vmap=vmap,
        lax=SimpleNamespace(fori_loop=fori, while_loop=while_loop, scan=scan,
                            cond=lambda pred, yes, no, operand: (yes if bool(pred) else no)(operand),
                            optimization_barrier=lambda x: x))
    try:
        for m in modules: m.jax, m.jnp = proxy, NumpyReal()
        with np.errstate(all='ignore'): yield
    finally:
        for m, original in zip(modules, saved):
            m.jax, m.jnp = original
