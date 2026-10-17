"""Finite-first matched-input WRF compiler-spread check; no fitted JAX bound."""
from __future__ import annotations

import numpy as np


def evaluate(candidate,reference,native_fma,*,absolute=3.e-6,ulps=16):
    arrays=tuple(np.asarray(x) for x in (candidate,reference,native_fma))
    if any(x.dtype!=np.float32 for x in arrays):
        raise ValueError('Compiler-spread arrays must be original float32')
    if any(x.shape!=arrays[0].shape for x in arrays) or arrays[0].ndim!=3:
        raise ValueError('Matched row/profile/field geometry required')
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError('Non-finite candidate, pristine reference or FMA reference')
    if not np.isfinite(absolute) or absolute<0 or ulps!=16:
        raise ValueError('Invalid original allowance')
    candidate,reference,native_fma=(x.astype(np.float64) for x in arrays)
    spread=np.max(np.abs(native_fma-reference),axis=1,keepdims=True)
    original=ulps*np.spacing(np.abs(arrays[1])).astype(np.float64)+absolute
    error=np.abs(candidate-reference)
    bound=spread+original
    return dict(pass_=bool(np.all(error<=bound)),outside=int(np.count_nonzero(error>bound)),
                strict_outside=int(np.count_nonzero(error>original)),
                max_error=float(error.max()),max_ratio=float(np.max(error/bound)))
