"""WRF digital-filter weights, literal REAL port of ``share/dfi.F`` (WRF 4.7.1).

``optfil_hcoeff`` reproduces ``optfil_driver`` + ``dfcoef`` (dfi.F:775-1103) and the window
routines ``UNIFORM`` / ``LANCZOS`` / ``HAMMING`` / ``BLACKMAN`` / ``KAISER`` (+ ``BESSI0``) /
``POTTER2`` / ``DOLPHWIN`` / ``DOLPH`` / ``CHEBY`` / ``NORMLZ`` in WRF ``REAL`` (IEEE binary32)
arithmetic, operation by operation in Fortran evaluation order.  Scalar transcendentals call the
C library's single-precision functions (``sinf``/``cosf``/``sqrtf``/``powf``/``log10f``/``expf``)
exactly as gfortran does for ``REAL`` intrinsics, so the weights are bit-identical to the
pristine routines compiled like the CPU-WRF reference build (tests/test_o2_dfi_filter.py against
``proofs/o2_dfi/dfcoef_oracle.json``).

Refused filters: ``dfi_nfilter=8`` (``RHOFIL``: indexes ``FILTER(0:100,0:100)`` with rows up to
``2*nsteps`` and writes ``FROW`` past its declared size -- out of bounds in WRF itself for any
window longer than 50 steps) and ``-1`` (``debug``: ``dfcoef`` then windows an uninitialised
``HH``).  Pure NumPy (no JAX import): usable by the pre-JAX namelist admission.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

__all__ = ["SUPPORTED_NFILTER", "dfcoef", "optfil_hcoeff", "optfil_nstep2"]

#: dfi_nfilter values the port reproduces (WRF Registry default 7 = Dolph).
SUPPORTED_NFILTER = (0, 1, 2, 3, 4, 5, 6, 7)
#: WRF Registry dfi_time_dim default (size of grid%hcoeff).
DFI_TIME_DIM = 1000

_f32 = np.float32
_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
for _name in ("sinf", "cosf", "sqrtf", "log10f", "expf", "atanf"):
    getattr(_libm, _name).restype = ctypes.c_float
    getattr(_libm, _name).argtypes = [ctypes.c_float]
_libm.powf.restype = ctypes.c_float
_libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]


def _sin(x):
    return _f32(_libm.sinf(float(x)))


def _cos(x):
    return _f32(_libm.cosf(float(x)))


def _sqrt(x):
    return _f32(_libm.sqrtf(float(x)))


def _pow(x, y):
    return _f32(_libm.powf(float(x), float(y)))


def _r(n) -> np.float32:
    """Fortran integer -> REAL conversion."""

    return _f32(n)


# REAL pi variants of the source: ``PI=4*ATAN(1.)`` (REAL atan) and ``PI = 4*ATAN(1.D0)`` assigned to
# a REAL (double pi rounded once).  Both round to the same binary32 value.
_PI_REAL = _f32(_f32(4.0) * _f32(_libm.atanf(1.0)))
_PI_FROM_DOUBLE = _f32(4.0 * math.atan(1.0))


def _uniform(nsteps: int) -> np.ndarray:
    return np.ones(nsteps + 1, dtype=_f32)


def _lanczos(nsteps: int) -> np.ndarray:
    power = _f32(1.0)
    pi = _PI_REAL
    ww = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        if n == 0:
            w = _f32(1.0)
        else:
            arg = _f32(_f32(_r(n) * pi) / _r(nsteps + 1))
            w = _f32(_sin(arg) / arg)
        ww[n] = _pow(w, power)
    return ww


def _hamming(nsteps: int) -> np.ndarray:
    alpha = _f32(0.54)
    pi = _PI_REAL
    ww = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        if n == 0:
            ww[n] = _f32(1.0)
        else:
            c = _cos(_f32(_f32(_r(n) * pi) / _r(nsteps)))
            ww[n] = _f32(alpha + _f32(_f32(_f32(1.0) - alpha) * c))
    return ww


def _blackman(nsteps: int) -> np.ndarray:
    pi = _PI_REAL
    ww = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        if n == 0:
            ww[n] = _f32(1.0)
        else:
            c1 = _cos(_f32(_f32(_r(n) * pi) / _r(nsteps)))
            c2 = _cos(_f32(_f32(_r(2 * n) * pi) / _r(nsteps)))
            ww[n] = _f32(_f32(_f32(0.42) + _f32(_f32(0.50) * c1)) + _f32(_f32(0.08) * c2))
    return ww


def _bessi0(x: np.float32) -> np.float32:
    p = (1.0, 3.5156230, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2)
    q = (0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1,
         0.2635537e-1, -0.1647633e-1, 0.392377e-2)
    if abs(x) < _f32(3.75):
        y = float(_f32(_f32(x / _f32(3.75)) ** 2))  # REAL (X/3.75)**2 assigned to REAL(8) Y
        val = p[0] + y * (p[1] + y * (p[2] + y * (p[3] + y * (p[4] + y * (p[5] + y * p[6])))))
        return _f32(val)
    ax = _f32(abs(x))
    y = float(_f32(_f32(3.75) / ax))
    poly = q[0] + y * (q[1] + y * (q[2] + y * (q[3] + y * (q[4] + y * (q[5] + y * (q[6] + y * (q[7]
                                                                                        + y * q[8])))))))
    scale = _f32(_f32(_libm.expf(float(ax))) / _sqrt(ax))  # (EXP(AX)/SQRT(AX)) in REAL
    return _f32(float(scale) * poly)


def _kaiser(nsteps: int) -> np.ndarray:
    alpha = _f32(1.0)
    xi0a = _bessi0(alpha)
    ww = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        xn = _r(n)
        ratio = _f32(xn / _r(nsteps))
        as_ = _f32(alpha * _sqrt(_f32(_f32(1.0) - _f32(ratio * ratio))))
        ww[n] = _f32(_bessi0(as_) / xi0a)
    return ww


def _potter2(nsteps: int) -> np.ndarray:
    d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165], dtype=_f32)
    pi = _PI_REAL
    ck = _f32(1.0)
    ww = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        if n == nsteps:
            ck = _f32(0.5)
        arg = _f32(_f32(pi * _r(n)) / _r(nsteps))
        arg = _f32(arg / _f32(2.0))
        total = d[0]
        for ip in range(1, 4):
            total = _f32(total + _f32(_f32(_f32(2.0) * d[ip]) * _cos(_f32(arg * _r(ip)))))
        ww[n] = _f32(ck * total)
    return ww


def _cheby_last(nm1: int, x: np.float32) -> np.float32:
    """``CHEBY(T, NM1, X)``; returns ``T(NM1)``."""

    t0, t1 = _f32(1.0), _f32(x)
    if nm1 < 2:
        return t0 if nm1 == 0 else t1
    for _ in range(2, nm1 + 1):
        t0, t1 = t1, _f32(_f32(_f32(_f32(2.0) * x) * t1) - t0)
    return t1


def _dolph_terms(n: int, x0: np.float32):
    nm1 = n - 1
    root = _sqrt(_f32(_f32(x0 * x0) - _f32(1.0)))
    term1 = _pow(_f32(x0 + root), _r(nm1))
    term2 = _pow(_f32(x0 - root), _r(nm1))
    return term1, term2


def _dolphwin(m: int) -> np.ndarray:
    pi = _PI_FROM_DOUBLE
    thetas = _f32(_f32(_f32(2.0) * pi) / _r(m))
    n = 2 * m + 1
    nm1 = n - 1
    x0 = _f32(_f32(1.0) / _cos(_f32(thetas / _f32(2.0))))
    term1, term2 = _dolph_terms(n, x0)
    rr = _f32(_f32(0.5) * _f32(term1 + term2))
    window = np.empty(m + 1, dtype=_f32)
    for nt in range(m + 1):
        total = rr
        for i in range(1, m + 1):
            arg = _f32(x0 * _cos(_f32(_f32(_r(i) * pi) / _r(n))))
            t_last = _cheby_last(nm1, arg)
            t2 = _cos(_f32(_f32(_f32(_r(2 * nt) * pi) * _r(i)) / _r(n)))
            total = _f32(total + _f32(_f32(_f32(2.0) * t_last) * t2))
        window[nt] = _f32(total / _r(n))
    return window


def _dolph(deltat: float, taus: float, m: int) -> np.ndarray:
    pi = _PI_FROM_DOUBLE
    n = 2 * m + 1
    nm1 = n - 1
    thetas = _f32(_f32(_f32(2.0) * pi) * _f32(abs(_f32(_f32(deltat) / _f32(taus)))))
    x0 = _f32(_f32(1.0) / _cos(_f32(thetas / _f32(2.0))))
    term1, term2 = _dolph_terms(n, x0)
    rr = _f32(_f32(0.5) * _f32(term1 + term2))
    r = _f32(_f32(1.0) / rr)
    w = np.empty(m + 1, dtype=_f32)
    for nt in range(m + 1):
        total = _f32(1.0)
        for i in range(1, m + 1):
            arg = _f32(x0 * _cos(_f32(_f32(_r(i) * pi) / _r(n))))
            t_last = _cheby_last(nm1, arg)
            t2 = _cos(_f32(_f32(_f32(_r(2 * nt) * pi) * _r(i)) / _r(n)))
            total = _f32(total + _f32(_f32(_f32(r * _f32(2.0)) * t_last) * t2))
        w[nt] = _f32(total / _r(n))
    # WINDOW(NT) = W2(NT) with W2(M-NT) = W(NT): edge first, centre last.
    return w[::-1].copy()


def dfcoef(nsteps: int, dt: float, tauc: float, nfilt: int) -> np.ndarray:
    """``DFCOEF(NSTEPS,DT,TAUC,NFILT,H)``: returns ``H(1:NSTEPS+1)`` as float32."""

    if nfilt not in SUPPORTED_NFILTER:
        raise ValueError(f"dfi_nfilter={nfilt} is not reproduced (supported {SUPPORTED_NFILTER})")
    if nsteps < 1:
        raise ValueError("DFI window must span at least two time steps")
    if nfilt == 7:
        return _dolph(dt, tauc, nsteps)
    window = {0: _uniform, 1: _lanczos, 2: _hamming, 3: _blackman, 4: _kaiser, 5: _potter2,
              6: _dolphwin}[nfilt](nsteps)
    pi = _PI_REAL
    deltat = _f32(abs(_f32(dt)))
    omegac = _f32(_f32(_f32(2.0) * pi) / _f32(tauc))
    hh = np.empty(nsteps + 1, dtype=_f32)
    for n in range(nsteps + 1):
        if n == 0:
            x = _f32(_f32(omegac * deltat) / pi)
        else:
            x = _f32(_sin(_f32(_f32(_r(n) * omegac) * deltat)) / _f32(_r(n) * pi))
        hh[n] = _f32(x * window[n])
    total = hh[0]
    for n in range(1, nsteps + 1):
        total = _f32(total + _f32(_f32(2.0) * hh[n]))
    hh = (hh / total).astype(_f32)
    return hh[::-1].copy()


def optfil_nstep2(rundfi_s: int, dt_s: float) -> int:
    """``nstep2 = ceiling((1.0 + real(rundfi)/timestep) / 2.0)`` in REAL."""

    timestep = _f32(abs(_f32(dt_s)))
    span = _f32(_f32(1.0) + _f32(_r(abs(int(rundfi_s))) / timestep))
    return int(math.ceil(float(_f32(span / _f32(2.0)))))


def optfil_hcoeff(rundfi_s: int, dt_s: float, tauc_s: float, nfilter: int) -> np.ndarray:
    """``optfil_driver``: the stage's ``grid%hcoeff(1:2*nstep2)`` (float32, trailing zeros kept).

    Element ``k`` (0-based) weights the state after ``k`` completed steps of the stage
    (``hn = hcoeff(itimestep+1)``, frame/module_integrate.F dfi_accumulate calls).
    """

    timestep = _f32(abs(_f32(dt_s)))
    rundfi = abs(int(rundfi_s))
    nstep2 = optfil_nstep2(rundfi, dt_s)
    if 2 * nstep2 > DFI_TIME_DIM:
        raise ValueError(f"DFI window needs {2 * nstep2} weights > dfi_time_dim={DFI_TIME_DIM}")
    h = np.zeros(2 * nstep2, dtype=_f32)
    h[:nstep2] = dfcoef(nstep2 - 1, dt_s, tauc_s, nfilter)
    span = int(_f32(_f32(1.0) + _f32(_r(rundfi) / timestep)))  # Fortran INT truncation
    if span % 2 != 0:
        for i in range(1, nstep2):
            h[2 * nstep2 - i - 1] = h[i - 1]
    else:
        for i in range(1, nstep2 + 1):
            h[2 * nstep2 - i] = h[i - 1]
    return h
