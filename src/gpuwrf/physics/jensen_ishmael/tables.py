"""Packaged original WRF tables; key the actual cached host namespace.

Only the raw provider is cached. Typed promotion is derived at each trace,
so a cached REAL8 derivative cannot outlive changed raw values. Production
REAL4 arrays share the cached storage; no table IO occurs in compiled steps.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np

from gpuwrf.assets import asset_path, ensure_asset

# > 10 MB: repo copy in a private checkout, else fetched on first use (gpuwrf.assets, fail-closed).
_ASSET=asset_path('src/gpuwrf/physics/jensen_ishmael/data/lookup_tables.npz')


@lru_cache(maxsize=1)
def _host_tables():
    with np.load(ensure_asset(_ASSET),allow_pickle=False) as data:values={n:data[n] for n in data.files}
    expected={
        'itab':((51,51,51,11,2),'float32'),'itabr':((51,51,51,11,6),'float32'),
        'gamma_tab':((505001,),'float32'),'igrdata':((60,),'float64'),
        'coltab_fp32':((60,60,35),'float32'),'coltabn_fp32':((60,60,35),'float32'),
        'coltab_fp64':((60,60,35),'float64'),'coltabn_fp64':((60,60,35),'float64')}
    if set(values)!=set(expected):raise ValueError('ISHMAEL table inventory differs from original WRF')
    for n,a in values.items():
        shape,dtype=expected[n]
        if a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():
            raise ValueError('Invalid original ISHMAEL table: '+n)
        a.flags.writeable=False
    return values


def get_tables(dtype):
    R=np.dtype(dtype)
    if R not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('ISHMAEL requires native REAL4/REAL8 arguments')
    mode='fp32' if R==np.dtype('float32') else 'fp64'
    raw=_host_tables()
    result={n:raw[n].astype(R,copy=False) for n in ('itab','itabr','gamma_tab','igrdata')}
    result.update(coltab=raw['coltab_'+mode],coltabn=raw['coltabn_'+mode])
    return result
