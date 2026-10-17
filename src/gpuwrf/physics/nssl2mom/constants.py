"""Module constants of WRF ``module_mp_nssl_2mom.F`` after ``nssl_2mom_init`` (default mp=18).

Values are frozen from the PRISTINE init (``data/init_constants.json``, built by
``proofs/v034/oracle/nssl2mom/make_constants_asset.py``) for both the WRF REAL=fp32 build and the
``-fdefault-real-8`` fp64 oracle build.  Arrays keep FORTRAN indexing: an array declared
``x(lc:lqmx)`` becomes a numpy array of length ``lqmx + 1`` with ``x[il]`` == Fortran ``x(il)``
(entries below the lower bound are zero).  REAL values carry the build's REAL dtype
(np.float32 for fp32), DOUBLE PRECISION values np.float64.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ASSET = Path(__file__).resolve().parent / "data" / "init_constants.json"


def _bound(tok: str, ns: dict) -> int:
    tok = tok.strip()
    try:
        return int(tok)
    except ValueError:
        pass
    sign = -1 if tok.startswith("-") else 1
    return sign * int(ns[tok.lstrip("-")])


@lru_cache(maxsize=None)
def get_constants(mode: str) -> SimpleNamespace:
    """Return the module constants of one build (``mode`` in {"fp32", "fp64"})."""
    raw = json.loads(_ASSET.read_text())
    vals = raw[mode]
    decls = raw["decls"]
    R = np.float32 if mode == "fp32" else np.float64
    ns: dict = {}
    # scalars first (integer bounds such as lc, lqmx, ngm0 are scalars)
    for name, v in vals.items():
        typ, kind, _dims = decls[name]
        if kind != "scalar":
            continue
        x = v[0]
        if typ == "real":
            ns[name] = R(x)
        elif typ == "doubleprecision":
            ns[name] = np.float64(x)
        elif typ == "integer":
            ns[name] = int(x)
        else:
            ns[name] = bool(x)
    for name, v in vals.items():
        typ, kind, dims = decls[name]
        if kind == "scalar":
            continue
        bounds = []
        for d in dims.split(","):
            lo, sep, hi = d.partition(":")
            bounds.append((_bound(lo, ns), _bound(hi, ns)) if sep else (1, _bound(lo, ns)))
        shape = [hi - lo + 1 for lo, hi in bounds]
        dt = {"real": R, "doubleprecision": np.float64, "integer": np.int64}.get(typ, bool)
        arr = np.array(v, dtype=np.float64 if dt is not bool else bool).reshape(shape, order="F")
        full = np.zeros([hi + 1 for lo, hi in bounds], dtype=dt)
        full[tuple(slice(lo, hi + 1) for lo, hi in bounds)] = arr.astype(dt)
        ns[name] = full
    ns["mode"] = mode
    ns["R"] = R
    return SimpleNamespace(**ns)


@lru_cache(maxsize=None)
def get_one_moment_constants(mode: str, variant: int) -> SimpleNamespace:
    """Original legacy19/21 init with canonical padded internal species slots.

    Active masses keep their WRF indices3..8 (19) or3..7 (21). Absent
    moment slots are zeros; the sole graupel-volume and CCNA indices are
    remapped to the existing stack. No active REAL constants are inferred.
    """
    if variant not in (19, 21):
        raise ValueError(variant)
    from .indices import LVH, LCCNA
    base = get_constants(mode)
    ns = dict(vars(base))
    raw = json.loads(_ASSET.with_name('one_moment_constants.json').read_text())
    vals = raw[str(variant)][mode]
    R = np.float32 if mode == 'fp32' else np.float64
    for name, values in vals.items():
        typ, kind, dims = raw['decls'][name]
        if kind == 'scalar':
            cast = {'real': R, 'doubleprecision': np.float64, 'integer': int}.get(typ, bool)
            ns[name] = cast(values[0])
    for name, values in vals.items():
        typ, kind, dims = raw['decls'][name]
        if kind == 'scalar':
            continue
        bounds = []
        for d in dims.split(','):
            lo, sep, hi = d.partition(':')
            bounds.append((_bound(lo, ns), _bound(hi, ns)) if sep else (1, _bound(lo, ns)))
        dt = {'real': R, 'doubleprecision': np.float64, 'integer': np.int64}.get(typ, bool)
        arr = np.asarray(values, dtype=dt).reshape([hi-lo+1 for lo, hi in bounds], order='F')
        full = np.zeros([hi+1 for lo, hi in bounds], dtype=dt)
        full[tuple(slice(lo, hi+1) for lo, hi in bounds)] = arr
        # The absent hail padding keeps safe default18 constants. It never
        # contributes mass/volume or caller output for21.
        if variant == 21 and typ in ('real', 'doubleprecision') and full.shape == getattr(base, name).shape:
            if len(bounds) == 1 and bounds[0] == (3, 30):
                full[8] = getattr(base, name)[8]
            elif len(bounds) == 2 and bounds == [(3, 30), (3, 30)]:
                full[8, :] = getattr(base, name)[8, :]
                full[:, 8] = getattr(base, name)[:, 8]
        ns[name] = full
    ns['variant'] = variant
    ns['lvol'] = np.zeros_like(base.lvol)
    ns['denscale'] = np.zeros_like(base.denscale)
    if variant == 19:
        ns['lvol'][7] = LVH
        ns['denscale'][LVH] = 1
        ns['denscale'][LCCNA] = 1
    return SimpleNamespace(**ns)
