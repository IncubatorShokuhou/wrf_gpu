"""Test-only literal REAL4 evaluation of the actual port operator bodies.

NumPy ufuncs round each arithmetic operation; libm expf/powf match the
pristine gfortran oracle's scalar intrinsics. This is not a forecast backend.
"""
import ctypes
import ctypes.util

import numpy as np

_libm = ctypes.CDLL(ctypes.util.find_library('m'))
_expf = _libm.expf
_expf.argtypes = [ctypes.c_float]
_expf.restype = ctypes.c_float
_powf = _libm.powf
_powf.argtypes = [ctypes.c_float, ctypes.c_float]
_powf.restype = ctypes.c_float


def real4(value):
    if isinstance(value, np.ndarray) and value.dtype == np.float32:
        return value.view(Real4)
    return value


class Real4(np.ndarray):
    def __pow__(self, exponent):
        left, right = np.broadcast_arrays(np.asarray(self), np.asarray(exponent))
        values = [_powf(float(a), float(b)) for a, b in zip(left.flat, right.flat)]
        return np.asarray(values, dtype=np.float32).reshape(left.shape).view(Real4)


class LiteralNumpy:
    """Small jnp-compatible namespace for literal source-association tests."""

    def __getattr__(self, name):
        if name in ('int32', 'float32'):
            return getattr(np, name)
        if name == 'exp':
            return lambda a: np.asarray([_expf(float(v)) for v in np.asarray(a).flat],
                                        dtype=np.float32).reshape(np.shape(a)).view(Real4)
        function = getattr(np, name)
        return lambda *args, **kwargs: real4(function(*args, **kwargs))
