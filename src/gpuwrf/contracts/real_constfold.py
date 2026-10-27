"""WRF REAL constant folding for REAL consumers (w44-constfold).

WRF is built with RWORDSIZE=4: gfortran folds a REAL constant expression such as ``(t0c-6.) - tice`` or
``VKC*VKC`` per operation in REAL (every leaf a REAL literal/PARAMETER, every operation rounded to REAL).
The same expression written in Python is folded in float64 and rounded ONCE when it meets a float32 field,
which can land 1-3 REAL ulps away from WRF's constant (27.15 vs WRF 27.149994).

``GPUWRF_WRF_REAL_CONSTFOLD=1`` (default OFF; the v0.4.4 RC switches it on together with the other identity changes) makes the audited sites use WRF's own REAL fold for float32
(REAL) consumers; float64 consumers (legacy / DOUBLE paths) keep the Python value, so they are unchanged.
The REAL values are written as explicit ``np.float32`` arithmetic in WRF's source order next to each site and
are pinned bit-for-bit to the pristine gfortran oracle in tests/test_w44_constfold.py.
Audit + oracle: proofs/w44_constfold/CONSTFOLD.md.
"""

from __future__ import annotations

import os

import numpy as np

# Import-time flag: registered in runtime.aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS (E80).
ENABLED = os.environ.get("GPUWRF_WRF_REAL_CONSTFOLD", "0").strip().lower() in {"1", "true", "yes", "on"}

R4 = np.float32


def real_constant(real_value, legacy_value, dtype=np.float32):
    """WRF's REAL fold for a float32 consumer when enabled, else the unchanged Python value."""
    if ENABLED and np.dtype(dtype) == np.dtype(np.float32):
        return real_value
    return legacy_value


_HOST_SCALAR = (int, float, np.integer, np.floating)


def is_host_scalar(value) -> bool:
    return isinstance(value, _HOST_SCALAR)


def real_scalar_product(factor, value, dtype=np.float32):
    """WRF REAL run-time product ``factor*value`` of REAL scalars (a constant and/or a namelist REAL such as the time
    step or dampcoef), operands in WRF's order.

    Flag ON and a REAL consumer: two host scalars give the exact REAL product (IEEE binary32 multiply of the REAL
    operands); a traced operand (e.g. a w44-params runtime parameter) is never converted on the host - it is cast to
    float32 with jnp (no-op when already REAL) and multiplied in REAL, one rounding like WRF.
    Flag OFF or a float64 consumer: the unchanged Python/JAX product.
    """
    if not (ENABLED and np.dtype(dtype) == np.dtype(np.float32)):
        return factor * value
    if is_host_scalar(factor) and is_host_scalar(value):
        return R4(factor) * R4(value)
    import jax.numpy as jnp
    real = lambda x: x if is_host_scalar(x) else jnp.asarray(x, jnp.float32)  # noqa: E731
    return real(factor) * real(value)
