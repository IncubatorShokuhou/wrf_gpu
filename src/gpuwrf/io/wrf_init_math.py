"""Backend-dark WRF REAL libm/profile helpers, shared by native ingest and cycling.

Moved verbatim from integration.d02_replay; scalar arithmetic and C-array
helper remain identical. No JAX/model import and no device allocation.
"""
from __future__ import annotations

import numpy as np

_WRF32_R_D = np.float32(287.0)

_WRF32_CP = np.float32(7.0) * _WRF32_R_D / np.float32(2.0)

_WRF32_CV = _WRF32_CP - _WRF32_R_D

_WRF32_CVPM = -(_WRF32_CV / _WRF32_CP)

_WRF32_RDOCP = _WRF32_R_D / _WRF32_CP

_WRF32_CPOVCV = _WRF32_CP / _WRF32_CV

_WRF32_P1000MB = np.float32(100000.0)

_WRF32_T0 = np.float32(300.0)

_WRF32_G = np.float32(9.81)

class _WRFInitLibm32:
    """float32 ``expf``/``logf``/``powf`` bit-matched to the CPU-WRF build's libm.

    The CPU-WRF truth is gfortran ``-O2`` REAL(4) calling scalar glibc
    ``expf``/``logf``/``powf`` (math-errno blocks vectorization, no fast-math, no
    FMA).  NumPy's float32 SIMD ``exp``/``log`` differ from glibc by 1-4 ulp and
    glibc ``powf(x, 0.5)`` differs from ``sqrtf`` by 1 ulp on rare inputs; those
    ulps are amplified ~50x by the hypsometric layer-thickness division into
    ~Pa-level base/perturbation pressure errors (proofs/v014/
    mythos_kernel_fix_260609.*).  Calling the same libm closes the chain
    bit-exactly.  Initialization-only host helper; never on the device hot path.
    Falls back to float64-computed correctly-rounded float32 when libm is not
    loadable (residual then bounded ~2.3 Pa worst-case instead of ~0.04 Pa).
    """

    def __init__(self, *, native_arrays: bool = True) -> None:
        self._native_arrays = native_arrays
        self._libm = None
        self.provider = "float64-rounded-fallback"
        try:
            import ctypes
            import ctypes.util

            path = ctypes.util.find_library("m")
            libm = ctypes.CDLL(path) if path else ctypes.CDLL(None)
            for name, nargs in (("expf", 1), ("logf", 1), ("powf", 2)):
                fn = getattr(libm, name)
                fn.restype = ctypes.c_float
                fn.argtypes = [ctypes.c_float] * nargs
            self._ctypes = ctypes
            self._libm = libm
            self.provider = "glibc-libm-float32"
        except Exception:  # pragma: no cover - non-glibc fallback
            self._libm = None

    def _map1(self, name: str, x: np.ndarray) -> np.ndarray:
        x32 = np.asarray(x, dtype=np.float32)
        if self._libm is None:
            fn64 = np.exp if name == "expf" else np.log
            return fn64(x32.astype(np.float64)).astype(np.float32)
        fn = getattr(self._libm, name)
        c_float = self._ctypes.c_float
        out = np.empty_like(x32)
        if self._native_arrays and x32.flags.c_contiguous:
            from gpuwrf.integration._init_libm_arrays import load
            native = load()
            if native is not None:
                ptr = self._ctypes.POINTER(self._ctypes.c_float)
                native.map1(self._ctypes.cast(fn,self._ctypes.c_void_p),
                            x32.ctypes.data_as(ptr),out.ctypes.data_as(ptr),x32.size)
                return out
        fi, fo = x32.ravel(), out.ravel()
        for i in range(fi.size):
            fo[i] = fn(c_float(float(fi[i])))
        return out

    def exp(self, x: np.ndarray) -> np.ndarray:
        return self._map1("expf", x)

    def log(self, x: np.ndarray) -> np.ndarray:
        return self._map1("logf", x)

    def pow(self, x: np.ndarray, y: np.float32) -> np.ndarray:
        x32 = np.asarray(x, dtype=np.float32)
        y32 = np.float32(y)
        if self._libm is None:
            return np.power(x32.astype(np.float64), np.float64(y32)).astype(np.float32)
        c_float = self._ctypes.c_float
        c_y = c_float(float(y32))
        out = np.empty_like(x32)
        if self._native_arrays and x32.flags.c_contiguous:
            from gpuwrf.integration._init_libm_arrays import load
            native = load()
            if native is not None:
                ptr = self._ctypes.POINTER(c_float)
                native.map2(self._ctypes.cast(self._libm.powf,self._ctypes.c_void_p),
                            x32.ctypes.data_as(ptr),c_y,out.ctypes.data_as(ptr),x32.size)
                return out
        fi, fo = x32.ravel(), out.ravel()
        for i in range(fi.size):
            fo[i] = self._libm.powf(c_float(float(fi[i])), c_y)
        return out

    def sqrt_via_pow(self, x: np.ndarray) -> np.ndarray:
        """WRF ``(...)**0.5`` compiles to ``powf(x, 0.5)`` (not ``sqrtss``)."""

        if self._libm is None:
            # float64 sqrt rounds to the correctly-rounded float32 sqrt.
            return np.sqrt(np.asarray(x, dtype=np.float32).astype(np.float64)).astype(np.float32)
        return self.pow(x, np.float32(0.5))

_WRF_INIT_LIBM32 = _WRFInitLibm32()

def _wrf_start_domain_t_init_alb32(pb: np.ndarray, s: dict[str, np.float32]) -> tuple[np.ndarray, np.ndarray]:
    """``start_em.F`` reference-profile ``T_INIT``/``ALB`` from REAL(4) ``PB`` (WRF grouping, WRF libm)."""

    m = _WRF_INIT_LIBM32
    p00, t00, a = s["p00"], s["t00"], s["a"]
    tiso, a_strat, p_strat = s["tiso"], s["a_strat"], s["p_strat"]
    # temp = MAX(tiso, t00 + A*LOG(pb/p00)); stratosphere branch when p_strat > 0
    temp = np.maximum(tiso, (t00 + a * m.log((pb / p00).astype(np.float32))).astype(np.float32))
    if float(p_strat) > 0.0:
        strat_temp = (tiso + a_strat * m.log((pb / p_strat).astype(np.float32))).astype(np.float32)
        temp = np.where(pb < p_strat, strat_temp, temp).astype(np.float32)
    t_init = (temp * m.pow((p00 / pb).astype(np.float32), _WRF32_RDOCP) - _WRF32_T0).astype(np.float32)
    alb = (
        _WRF32_R_D / _WRF32_P1000MB * (t_init + _WRF32_T0) * m.pow((pb / _WRF32_P1000MB).astype(np.float32), _WRF32_CVPM)
    ).astype(np.float32)
    return t_init, alb
