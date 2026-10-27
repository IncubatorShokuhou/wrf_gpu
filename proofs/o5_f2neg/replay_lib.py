"""Same-operand classification of port negatives against the pristine WRF chain.

PRE-REGISTERED (o5-f2neg, 2026-10-24, before any replay data was inspected):
  tol(cell) = max(8 * E(cell), 64 * eps32 * S(cell))
    E = |pristine(fp-contract=fast) - pristine(off)| on the same operands (rounding envelope)
    S = max |input field| over the 7x7 horizontal x 3 level stencil footprint of the cell
        (step-start field, stage field and preload), the scale every chain term is rounded at.
  For every port-negative cell (port < 0):
    FAITHFUL          |port - wrf| <= tol            (WRF gives the same value within rounding;
                                                       sub-class wrf >= 0 = sign decided at rounding level)
    LARGER_IN_PORT    port < wrf - tol                (port more negative beyond rounding -> bug candidate)
    SMALLER_IN_PORT   port > wrf + tol                (port less negative beyond rounding)
  plus WRF_ONLY: wrf < -tol while port >= 0.
  Species verdict FAITHFUL iff LARGER_IN_PORT == SMALLER_IN_PORT == WRF_ONLY == 0.
NaN/Inf in any compared array is a hard failure (E200).
"""
from __future__ import annotations

import numpy as np

EPS32 = float(np.finfo(np.float32).eps)  # 2**-23
K_ENV = 8.0
K_EPS = 64.0


def finite_or_raise(name, *arrays):
    for a in arrays:
        if not np.isfinite(np.asarray(a, np.float64)).all():
            raise FloatingPointError(f"non-finite values in {name}")


def footprint_max(*fields, r=3, rk=1):
    """max |f| over a (2rk+1) x (2r+1) x (2r+1) box, edge-clipped, for each cell."""
    a = np.max([np.abs(np.asarray(f, np.float64)) for f in fields], axis=0)
    out = a.copy()
    nz, ny, nx = a.shape
    pad = np.pad(a, ((rk, rk), (r, r), (r, r)), mode="edge")
    for dk in range(2 * rk + 1):
        for dj in range(2 * r + 1):
            for di in range(2 * r + 1):
                np.maximum(out, pad[dk:dk + nz, dj:dj + ny, di:di + nx], out=out)
    return out


def ring_index(shape):
    nz, ny, nx = shape
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    return np.broadcast_to(np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))[None], shape)


def compare_field(port, wrf_off, wrf_fast=None):
    """Field-level difference statistics (any stage)."""
    port = np.asarray(port, np.float64)
    off = np.asarray(wrf_off, np.float64)
    finite_or_raise("compare_field", port, off)
    d = port - off
    ref = float(np.abs(off).max())
    rec = {"max_abs_diff": float(np.abs(d).max()), "max_abs_ref": ref,
           "rel_max": float(np.abs(d).max() / ref) if ref > 0 else (0.0 if not d.any() else float("inf")),
           "bitwise_equal_fraction": float(np.mean(np.asarray(port, np.float32) == np.asarray(off, np.float32)))}
    inner = ring_index(port.shape) > 0 if port.ndim == 3 else np.ones(port.shape, bool)
    if inner.any():
        refi = float(np.abs(off[inner]).max())
        rec["rel_max_interior"] = float(np.abs(d[inner]).max() / refi) if refi > 0 else float(np.abs(d[inner]).max())
    if wrf_fast is not None:
        fast = np.asarray(wrf_fast, np.float64)
        finite_or_raise("compare_field fast", fast)
        rec["fma_envelope_max"] = float(np.abs(fast - off).max())
    return rec


def classify(port, wrf_off, wrf_fast, scale, ring_min=0):
    """ring_min > 0 restricts every count to cells with lateral ring index >= ring_min."""
    port = np.asarray(port, np.float64)
    off = np.asarray(wrf_off, np.float64)
    fast = np.asarray(wrf_fast, np.float64)
    finite_or_raise("classify", port, off, fast, scale)
    env = np.abs(fast - off)
    tol = np.maximum(K_ENV * env, K_EPS * EPS32 * np.asarray(scale, np.float64))
    d = port - off
    ring = ring_index(port.shape)
    keep = ring >= ring_min
    pneg = (port < 0) & keep
    faithful = pneg & (np.abs(d) <= tol)
    larger = pneg & (d < -tol)
    smaller = pneg & (d > tol)
    wrf_only = (off < -tol) & ~pneg & keep
    wneg = (off < 0) & keep
    out = {
        "ring_min": int(ring_min),
        "port_negative": int(pneg.sum()), "port_negative_interior": int((pneg & (ring > 0)).sum()),
        "wrf_negative": int(wneg.sum()),
        "port_min": float(port[pneg].min()) if pneg.any() else None,
        "wrf_min": float(off[wneg].min()) if wneg.any() else None,
        "FAITHFUL": int(faithful.sum()),
        "FAITHFUL_wrf_nonnegative_at_rounding": int((faithful & (off >= 0)).sum()),
        "LARGER_IN_PORT": int(larger.sum()), "SMALLER_IN_PORT": int(smaller.sum()), "WRF_ONLY": int(wrf_only.sum()),
    }
    if pneg.any():
        r = np.abs(d[pneg]) / np.maximum(EPS32 * np.asarray(scale, np.float64)[pneg], 1e-300)
        out["diff_over_eps_scale_at_port_neg"] = {"max": float(r.max()), "p99": float(np.percentile(r, 99)),
                                                   "median": float(np.median(r))}
        rel = np.abs(d[pneg]) / np.maximum(np.abs(port[pneg]), 1e-300)
        out["rel_diff_at_port_neg"] = {"max": float(rel.max()), "p99": float(np.percentile(rel, 99)),
                                       "median": float(np.median(rel))}
        imin = np.unravel_index(int(np.argmin(np.where(pneg, port, np.inf))), port.shape)
        out["port_min_cell"] = {"kji": [int(x) for x in imin], "port": float(port[imin]), "wrf_off": float(off[imin]),
                                "wrf_fast": float(fast[imin]), "tol": float(tol[imin]), "ring": int(ring[imin])}
    for label, mask in (("LARGER_IN_PORT", larger), ("SMALLER_IN_PORT", smaller), ("WRF_ONLY", wrf_only)):
        if mask.any():
            w = np.unravel_index(int(np.argmax(np.where(mask, np.abs(d) - tol, -np.inf))), port.shape)
            out[label + "_worst"] = {"kji": [int(x) for x in w], "port": float(port[w]), "wrf_off": float(off[w]),
                                     "wrf_fast": float(fast[w]), "tol": float(tol[w]), "ring": int(ring[w])}
    # Band split (critic cr-o5-f2neg F1, added after the pre-registered rule; reporting only):
    # R0 |port| <= 1 eps*S (sign set by rounding), R1 1..64 eps*S, R2 > 64 eps*S.
    unit = EPS32 * np.asarray(scale, np.float64)
    mag = np.abs(port) / np.maximum(unit, 1e-300)
    bands = {}
    for name, lo, hi in (("R0", -1.0, 1.0), ("R1", 1.0, 64.0), ("R2", 64.0, np.inf)):
        m = pneg & (mag > lo) & (mag <= hi)
        same = m & (off < 0)
        rec = {"n": int(m.sum()), "wrf_same_sign": int(same.sum()), "wrf_nonnegative": int((m & (off >= 0)).sum())}
        if same.any():
            rec["max_rel_diff_same_sign"] = float((np.abs(d[same]) / np.abs(port[same])).max())
        bands[name] = rec
    out["bands"] = bands
    out["WRF_ONLY_above_1epsS"] = int(((off < 0) & (np.abs(off) > unit) & ~(port < 0) & keep).sum())
    out["wrf_negative_R0"] = int(((off < 0) & (np.abs(off) <= unit) & keep).sum())
    out["verdict"] = "FAITHFUL" if out["LARGER_IN_PORT"] == out["SMALLER_IN_PORT"] == out["WRF_ONLY"] == 0 else "DIFFERS"
    return out


def allcell_diff(port, wrf_off, *fields):
    """All-cell max |port-wrf|/(eps*S) with the pre-registered 7x7x3 and a 7x7x5 footprint (critic F6)."""
    d = np.abs(np.asarray(port, np.float64) - np.asarray(wrf_off, np.float64))
    out = {}
    for label, rk in (("S7x7x3", 1), ("S7x7x5", 2)):
        s = footprint_max(*fields, rk=rk) * EPS32
        r = d / np.maximum(s, 1e-300)
        r[d == 0] = 0.0
        out[label] = float(r.max())
    fm = float(np.abs(np.asarray(wrf_off, np.float64)).max())
    out["max_abs_over_field_max"] = float(d.max() / fm) if fm > 0 else float(d.max())
    return out


def birth(qold, pre, fin, mass1, dt, parts):
    """Where port-negative cells are born: pre-add (pre < 0) vs advection/update (pre >= 0).

    For pre-add-born cells, the sc part with the most negative increment dt*part/mass is reported."""
    fin = np.asarray(fin, np.float64)
    pre = np.asarray(pre, np.float64)
    neg = fin < 0
    bp = neg & (pre < 0)
    ba = neg & (pre >= 0)
    out = {"negative": int(neg.sum()), "born_in_preadd": int(bp.sum()), "born_in_advection_or_update": int(ba.sum()),
           "qold_negative": int((neg & (np.asarray(qold) < 0)).sum()),
           "min_born_preadd": float(fin[bp].min()) if bp.any() else None,
           "min_born_advection": float(fin[ba].min()) if ba.any() else None}
    if ba.any():
        # scale of advection-born negatives relative to the local field magnitude
        rel = np.abs(fin[ba]) / np.maximum(footprint_max(qold, pre)[ba], 1e-300)
        out["advection_born_rel_to_footprint"] = {"max": float(rel.max()), "median": float(np.median(rel))}
    if bp.any() and parts:
        incr = {k: dt * np.asarray(v, np.float64) / np.asarray(mass1, np.float64) for k, v in parts.items()}
        names = list(incr)
        stack = np.stack([incr[k][bp] for k in names])
        dom = np.argmin(stack, axis=0)
        out["preadd_dominant_negative_sc_part"] = {k: int((dom == i).sum()) for i, k in enumerate(names)}
        imin = np.unravel_index(int(np.argmin(np.where(bp, fin, np.inf))), fin.shape)
        out["preadd_min_cell"] = {"kji": [int(x) for x in imin], "qold": float(np.asarray(qold)[imin]),
                                  "pre": float(pre[imin]), "fin": float(fin[imin]),
                                  "increments": {k: float(incr[k][imin]) for k in names}}
    return out
