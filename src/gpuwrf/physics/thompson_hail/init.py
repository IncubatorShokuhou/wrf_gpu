"""Original Thompson missing-aerosol initialization, executed once on the host.

The pristine initializer (:493-549) does not read its ORHO argument. These
values are assigned directly to the aerosol scalar storage. NWFA2D is a held
surface tendency; it must not be diagnosed again after the model evolves.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import numpy as np


_LIBM = ctypes.CDLL(ctypes.util.find_library("m") or None)
for _name in ("expf", "logf", "cosf"):
    _function = getattr(_LIBM, _name)
    _function.argtypes = [ctypes.c_float]
    _function.restype = ctypes.c_float


def _unary32(name, values):
    """Scalar libm calls match gfortran's literal REAL/no-FMA intrinsics."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    function = getattr(_LIBM, name)
    for index, value in enumerate(values.flat):
        result.flat[index] = function(float(value))
    return result


def initial_profiles(hgt):
    """Return original NWFA, NIFA and held NWFA2D for mass-level HGT (..., z).

    The caller independently applies MAXVAL(field)<eps for NWFA and NIFA.
    Existing positive aerosol data and supplied surface emissions are held.
    NIFA2D is not assigned by pristine thompson_init; its cold default is zero.
    """
    hgt = np.asarray(hgt, dtype=np.float32)
    if hgt.ndim < 1 or hgt.shape[-1] < 2 or not np.isfinite(hgt).all():
        raise ValueError("Thompson aerosol initialization requires finite HGT with at least two mass levels")
    f = np.float32
    h0 = hgt[..., :1]
    middle = f(.8) * _unary32("cosf", h0 * f(.001) - f(1.))
    h01 = np.where(h0 <= f(1000.), f(.8), np.where(h0 >= f(2500.), f(.01), middle))
    ni_ccn3 = -f(1.) * f(_LIBM.logf(float(f(50.e6) / f(300.e6)))) / h01
    ni_in3 = -f(1.) * f(_LIBM.logf(float(f(.5e6) / f(1.5e6)))) / h01
    dz = hgt - h0
    dz1 = hgt[..., 1:2] - h0
    # Original k=1 intentionally uses k=2's height offset.
    dz = np.concatenate((dz1, dz[..., 1:]), axis=-1)
    nwfa = f(50.e6) + f(300.e6) * _unary32("expf", -(dz / f(1000.)) * ni_ccn3)
    nifa = f(.5e6) + f(1.5e6) * _unary32("expf", -(dz / f(1000.)) * ni_in3)
    nwfa2d = nwfa[..., 0] * f(.000196) * (f(50.) / dz1[..., 0])
    return nwfa, nifa, nwfa2d
