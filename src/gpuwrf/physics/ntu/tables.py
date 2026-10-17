"""Tiny initialized WRF aerosol table banks; raw host NumPy only is cached.

Every bank was produced by original AERO_CONST for its explicit CCNTY and
REAL kind. Conversion to JAX arrays happens at use, never in this cache.
The operational admission/key registration is added with the column adapter.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .constants import constants

DATA=Path(__file__).with_name('data')


@lru_cache(maxsize=2)
def host_tables(kind):
    if kind not in ('r4','r8'):raise ValueError(f'NTU REAL kind: {kind}')
    with np.load(DATA/f'aerosol_{kind}.npz',allow_pickle=False) as f:
        result={k:f[k].copy() for k in f.files}
    for value in result.values():value.flags.writeable=False
    return result


def selected_tables(dtype=np.float32,ccnty=2):
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(f'NTU dtype: {dtype}')
    if ccnty not in (1,2,3,4):raise ValueError(f'NTU CCNTY must be 1..4: {ccnty}')
    raw=host_tables('r4' if dtype.itemsize==4 else 'r8')
    prefix=f'ccnty{ccnty}_'
    result=dict(constants(dtype))
    result.update({k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)})
    return result
