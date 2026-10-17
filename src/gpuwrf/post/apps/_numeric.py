"""Small NumPy-only helpers; never initialize a model/JAX backend here."""
from __future__ import annotations

import numpy as np


def finite(name, value, *, minimum=None, maximum=None):
    """Reject missing/nonfinite inputs before reductions or thresholding."""
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite (no NaN/Inf)")
    if minimum is not None and np.any(a < minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and np.any(a > maximum):
        raise ValueError(f"{name} must be <= {maximum}")
    return a


def thermo(temperature_k, pressure_pa, mixing_ratio):
    """Bolton (1980) saturation over liquid; q is kg vapor / kg dry air.

    RH is bounded at saturation for diagnostic use (not a prognostic repair).
    Returns dry-air density kg/m3, RH percent, vapor-pressure deficit hPa.
    """
    t = finite("temperature_k", temperature_k, minimum=150, maximum=350)
    p = finite("pressure_pa", pressure_pa, minimum=1)
    q = finite("mixing_ratio", mixing_ratio, minimum=0)
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    e = p * q / (0.622 + q)
    return (p / (287.05 * t * (1 + q / 0.622)),
            np.minimum(100.0, 100 * e / es), np.maximum(0.0, es - e) / 100)
