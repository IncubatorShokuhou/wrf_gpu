"""Execute the shared Ferrier source order with literal host REAL arithmetic.

This is an oracle diagnostic, not the forecast implementation. It shares the
port's equation code objects, then compares its outputs with independent WRF
Fortran fixtures. NumPy performs elementary REAL operations; glibc expf/logf
match the original gfortran routine's scalar transcendental calls. It neither
uses JAX execution nor changes GPU/runtime precision.
"""
import ctypes
from types import FunctionType

import numpy as np

from gpuwrf.physics.ferrier import column, saturation


_LIBM = ctypes.CDLL("libm.so.6")
for _name in ("exp", "log"):
    for _suffix, _ctype in (("f", ctypes.c_float), ("", ctypes.c_double)):
        _fn = getattr(_LIBM, _name + _suffix)
        _fn.argtypes = [_ctype]
        _fn.restype = _ctype
for _suffix, _ctype in (("f", ctypes.c_float), ("", ctypes.c_double)):
    _fn = getattr(_LIBM, "pow" + _suffix)
    _fn.argtypes = [_ctype, _ctype]
    _fn.restype = _ctype


def _wrap(value):
    if isinstance(value, tuple):
        return tuple(_wrap(v) for v in value)
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).view(_RealArray)
    return value


class _RealArray(np.ndarray):
    """Fortran REAL32 + INTEGER32 is REAL32; NumPy normally promotes to f64."""
    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        values = [np.asarray(v) if isinstance(v, _RealArray) else v for v in inputs]
        dtypes = [np.asarray(v).dtype for v in values if isinstance(v, (np.ndarray, np.generic))]
        if any(t == np.float64 for t in dtypes):
            target = np.float64
        elif any(t == np.float32 for t in dtypes):
            target = np.float32
        else:
            target = None
        if target is not None:
            values = [np.asarray(v, dtype=target) for v in values]
        if ufunc is np.power and method == "__call__" and target is not None:
            a, b = np.broadcast_arrays(*values)
            fn = _LIBM.powf if target == np.float32 else _LIBM.pow
            return _wrap(np.asarray([fn(float(x), float(y)) for x, y in zip(a.ravel(), b.ravel())],
                                    dtype=target).reshape(a.shape))
        # The shared equation body uses JAX's immutable ``running &= ...``.
        # A NumPy in-place ``out`` would also mutate the aliased phase mask
        # (existing/remaining_rain), changing the algorithm. Return a new value.
        kwargs.pop("out", None)
        return _wrap(getattr(ufunc, method)(*values, **kwargs))


class _LiteralNumpy:
    def __getattr__(self, name):
        fn = getattr(np, name)
        if name in ("float32", "float64", "int32", "int64", "bool_"):
            return fn
        return lambda *args, **kwargs: _wrap(fn(*args, **kwargs))

    @staticmethod
    def where(mask, a, b):
        dtypes = [np.asarray(v).dtype for v in (a, b) if isinstance(v, (np.ndarray, np.generic))]
        target = (np.float64 if np.dtype(np.float64) in dtypes else
                  np.float32 if np.dtype(np.float32) in dtypes else
                  np.int32 if np.dtype(np.int32) in dtypes else None)
        return _wrap(np.where(mask, np.asarray(a, dtype=target), np.asarray(b, dtype=target)))

    @staticmethod
    def exp(value):
        return _wrap(_libm("exp", value))

    @staticmethod
    def log(value):
        return _wrap(_libm("log", value))


def _libm(name, value):
    array = np.asarray(value)
    fn = getattr(_LIBM, name + ("f" if array.dtype == np.float32 else ""))
    return np.asarray([fn(float(x)) for x in array.ravel()], dtype=array.dtype).reshape(array.shape)


class _LiteralLax:
    @staticmethod
    def while_loop(condition, body, carry):
        while condition(carry):
            carry = body(carry)
        return carry

    @staticmethod
    def fori_loop(lower, upper, body, carry):
        for i in range(lower, upper):
            carry = body(i, carry)
        return carry

    @staticmethod
    def scan(body, carry, fields):
        output = []
        for i in range(fields[0].shape[0]):
            carry, result = body(carry, tuple(field[i] for field in fields))
            output.append(result)

        def stack(values):
            if isinstance(values[0], tuple):
                return tuple(stack([v[i] for v in values]) for i in range(len(values[0])))
            return np.stack(values)

        return carry, stack(output)


class _LiteralWork(column._Work):
    def put(self, mask, **fields):
        for name, value in fields.items():
            self.values[name] = _LiteralNumpy.where(mask, value, self.values.get(name, np.zeros_like(value)))


def _copy(fn, env):
    copy = FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
    copy.__kwdefaults__ = fn.__kwdefaults__
    return copy


def literal_column(*args, **kwargs):
    numpy = _LiteralNumpy()
    sat_env = dict(vars(saturation), jnp=numpy, lax=_LiteralLax)
    for name in ("fpvs", "_ws", "_adjust", "condense", "deposit"):
        sat_env[name] = _copy(getattr(saturation, name), sat_env)
    env = dict(vars(column), jnp=numpy, lax=_LiteralLax, _Work=_LiteralWork)
    env.update({name: sat_env[name] for name in ("fpvs", "condense", "deposit")})
    for name in ("_table", "_vel_rime", "_growth", "_rain_dsd", "_partition", "prepare_level",
                 "prepare", "finish_level", "finish", "initial_flux", "level", "reference_column"):
        env[name] = _copy(getattr(column, name), env)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        return env["reference_column"](*args, **kwargs)
