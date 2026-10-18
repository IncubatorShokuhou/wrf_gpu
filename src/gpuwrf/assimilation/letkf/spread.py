"""Ensemble spread maintenance after the LETKF transform — OPT-IN (lane w41-da-ens; R3 items 4/5).

R3 (proofs/a1_da/R3_SPREAD.md) measured three defects: (a) RTPS 0.8 relaxes the posterior toward the RAW prior
spread, so a multiplicative prior inflation ρ applied inside the transform never reaches the written members
(T2 prior 0.065 K, posterior 0.140 K without RTPS, 0.081 K with RTPS 0.8); (b) the 6-h forecasts lose 40–75 % of
the T/V spread, so the deficit compounds cycle to cycle (ρ 2.35 → 3.06); (c) N = 10 sampling noise. This module
supplies the posterior-side tools; nothing here touches the forecast program.

* :func:`rtps` — relaxation to prior spread (Whitaker & Hamill 2012) with an optional TARGET inflation:
  σ_target = √ρ·σ_b, factor = 1 + α (σ_target − σ_a)/σ_a. ``rho=1`` is the classical RTPS (identical arithmetic to
  :func:`core.rtps`); ``rho`` may be a scalar or any array broadcastable to one member (e.g. a per-level or
  per-(variable, band) field from :mod:`inflation`).
* :func:`rtpp` — relaxation to prior perturbations (Zhang et al. 2004): x′_a ← (1 − α) x′_a + α x′_b.
* :func:`posterior_inflation` — multiplicative inflation of the analysis deviations about their mean.
* :func:`additive_inflation` — seeded, centred additive inflation from a pool of perturbation samples (archive
  deviations, forecast differences, spun-up member deviations): member i gets a·η_i with
  η_i = Σ_m z_im (p_m − p̄)/√(M − 1), z ~ N(0, 1) i.i.d. (so Cov(η) = Cov(pool) in expectation and every η_i is a
  linear combination of balanced pool states), then η is re-centred over the members (the ensemble mean is
  unchanged to round-off). QVAPOR/Q2 work in LOG space (multiplicative, re-centred: positive by construction and the
  member mean kept exactly). The perturbation is multiplied by the analysis boundary taper (0 in the
  specified/relaxation zone → those cells keep their bytes).
* :func:`spread_by_band` — the R3 diagnostic: interior member spread (std, ddof 1) per variable × height band, so
  that survival can be measured at IC, after writing, after 1 h and at the end of a cycle.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

LOG_FIELDS = ("QVAPOR", "Q2")
STAGGER = {"U": "x", "V": "y"}
BANDS = ((0.0, 1000.0), (1000.0, 2000.0), (2000.0, 5000.0), (5000.0, np.inf))
_Q_FLOOR = 1.0e-12
G = 9.81


def rtps(xb, xa, alpha, *, rho=1.0):
    """Relaxation to (optionally inflated) prior spread, pointwise over the leading member axis.

    Points with σ_a == 0 keep their analysis (factor 1). ``rho`` ≥ 0 broadcastable to ``xb.shape[1:]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("RTPS alpha must lie in [0, 1]")
    xb = np.asarray(xb, dtype=np.float64)
    xa = np.asarray(xa, dtype=np.float64)
    mean = xa.mean(axis=0)
    pa = xa - mean
    sb = xb.std(axis=0, ddof=1)
    if not (np.isscalar(rho) and rho == 1.0):
        rho = np.asarray(rho, dtype=np.float64)
        if (rho < 0).any() or not np.isfinite(rho).all():
            raise ValueError("RTPS target inflation must be finite and >= 0")
        sb = np.sqrt(rho) * sb
    sa = pa.std(axis=0, ddof=1)
    safe = sa > 0.0
    factor = np.where(safe, 1.0 + alpha * (sb - sa) / np.where(safe, sa, 1.0), 1.0)
    return mean[None] + pa * factor[None]


def rtpp(xb, xa, alpha):
    """Relaxation to prior perturbations: analysis mean kept, deviations (1 − α) x′_a + α x′_b."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("RTPP alpha must lie in [0, 1]")
    xb = np.asarray(xb, dtype=np.float64)
    xa = np.asarray(xa, dtype=np.float64)
    mean = xa.mean(axis=0)
    return mean[None] + (1.0 - alpha) * (xa - mean[None]) + alpha * (xb - xb.mean(axis=0)[None])


def posterior_inflation(xa, factor):
    """Multiplicative inflation of the deviations about the member mean (``factor`` broadcastable, >= 0)."""
    xa = np.asarray(xa, dtype=np.float64)
    mean = xa.mean(axis=0)
    return mean[None] + np.asarray(factor, np.float64) * (xa - mean[None])


def pool_from_states(states: Mapping[str, np.ndarray], names: Iterable[str] | None = None) -> dict:
    """Centred pool deviations from M model states (name → (M, …)); LOG_FIELDS become log-deviations."""
    out = {}
    for name in (states if names is None else names):
        x = np.asarray(states[name], np.float64)
        if name in LOG_FIELDS:
            x = np.log(np.maximum(x, _Q_FLOOR))
        out[name] = x - x.mean(axis=0)[None]
    return out


def gaussian_combinations(n_members: int, n_pool: int, seed) -> np.ndarray:
    """(N, M) weights z/√(M − 1), z ~ N(0, 1) from ``numpy.random.default_rng(seed)`` (recorded by the caller)."""
    if n_pool < 2:
        raise ValueError("additive inflation needs a pool of at least 2 perturbation samples")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_members, n_pool)) / np.sqrt(n_pool - 1.0)


def additive_inflation(members: Mapping[str, np.ndarray], pool: Mapping[str, np.ndarray], amplitude, *, seed,
                       names: Iterable[str] | None = None, taper: Mapping[str, np.ndarray] | None = None,
                       weights: np.ndarray | None = None):
    """Return (new member dict, receipt). ``pool``: name → (M, …) CENTRED deviations (:func:`pool_from_states`;
    log-deviations for LOG_FIELDS). ``amplitude`` scalar or name → scalar. ``taper``: name → 2-D (ny, nx[+1])
    weights (0 keeps the member bytes). The same weights z couple every variable (multivariate structure kept).
    Fields not listed are returned unchanged (same object)."""
    names = tuple(pool if names is None else names)
    n = int(np.shape(members[names[0]])[0])
    m = int(np.shape(pool[names[0]])[0])
    z = gaussian_combinations(n, m, seed) if weights is None else np.asarray(weights, np.float64)
    if z.shape != (n, m):
        raise ValueError(f"weights shape {z.shape} != {(n, m)}")
    out = dict(members)
    receipt = {"seed": seed if weights is None else "explicit weights", "n_members": n, "n_pool": m, "fields": {}}
    for name in names:
        a = float(amplitude[name] if isinstance(amplitude, Mapping) else amplitude)
        x = np.asarray(members[name])
        p = np.asarray(pool[name], np.float64)
        if p.shape[1:] != x.shape[1:]:
            raise ValueError(f"{name}: pool shape {p.shape[1:]} != member shape {x.shape[1:]}")
        eta = np.einsum("im,m...->i...", z, p)
        eta -= eta.mean(axis=0)[None]
        eta *= a
        w = None
        if taper is not None and name in taper:
            w = np.asarray(taper[name], np.float64)
            eta *= w
        x64 = x.astype(np.float64)
        before = x64.std(axis=0, ddof=1)
        if name in LOG_FIELDS:
            factors = np.exp(eta)
            mean = x64.mean(axis=0)
            new = np.maximum(x64, 0.0) * factors
            scale = np.where(new.mean(axis=0) > 0, mean / np.where(new.mean(axis=0) > 0, new.mean(axis=0), 1.0), 1.0)
            new = new * scale[None]
        else:
            new = x64 + eta
        stored = new.astype(x.dtype)
        if w is not None:
            stored = np.where(np.broadcast_to(w == 0, x.shape[-2:]), x, stored)
        out[name] = stored
        after = stored.astype(np.float64).std(axis=0, ddof=1)
        receipt["fields"][name] = {
            "amplitude": a, "space": "log" if name in LOG_FIELDS else "linear",
            "spread_rms_before": float(np.sqrt(np.mean(before ** 2))),
            "spread_rms_after": float(np.sqrt(np.mean(after ** 2))),
            "mean_change_absmax": float(np.abs(stored.astype(np.float64).mean(axis=0) - x64.mean(axis=0)).max()),
            "negative_after": int((stored < 0).sum()) if name in LOG_FIELDS else None}
    return out, receipt


def taper_fields(names: Iterable[str], ny: int, nx: int, *, spec_bdy_width: int = 5, ramp: int = 5) -> dict:
    """name → analysis boundary taper on the field's own (staggered) 2-D grid (``state_io.boundary_taper``)."""
    from .state_io import boundary_taper
    return {n: np.asarray(boundary_taper(ny, nx, spec_bdy_width, ramp, STAGGER.get(n)), np.float64) for n in names}


def _mass(name, x):
    """(N, nz, ny, nx) mass-grid version of a member field (U/V face-averaged)."""
    x = np.asarray(x, np.float64)
    if STAGGER.get(name) == "x":
        return 0.5 * (x[..., 1:] + x[..., :-1])
    if STAGGER.get(name) == "y":
        return 0.5 * (x[..., 1:, :] + x[..., :-1, :])
    return x


def height_agl(frames: Mapping[str, np.ndarray], geo: Mapping) -> np.ndarray:
    """(nz, ny, nx) member-mean mass-level height above ground (m)."""
    z = (np.asarray(frames["PH"], np.float64) + np.asarray(frames["PHB"], np.float64)).mean(axis=0) / G
    return 0.5 * (z[1:] + z[:-1]) - np.asarray(geo["HGT"], np.float64)[None]


def band_index(frames: Mapping[str, np.ndarray], geo: Mapping) -> np.ndarray:
    """(nz, ny, nx) band index of every mass level from the member-mean geopotential AGL."""
    return band_of_height(height_agl(frames, geo))


def band_of_height(zm: np.ndarray) -> np.ndarray:
    """Band index (``BANDS``) of heights AGL (m), any shape."""
    zm = np.asarray(zm, np.float64)
    idx = np.full(zm.shape, len(BANDS) - 1, dtype=np.int64)
    for b, (lo, hi) in reversed(list(enumerate(BANDS))):
        idx = np.where((zm >= lo) & (zm < hi), b, idx)
    return idx


def spread_by_band(frames: Mapping[str, np.ndarray], geo: Mapping, names: Iterable[str] = ("T", "QVAPOR", "U", "V"),
                   *, interior: int = 15, surface: Iterable[str] = ("T2", "Q2", "U10", "V10")) -> dict:
    """R3 metric: rms of the member spread (std, ddof 1) over interior cells (≥ ``interior`` rows from every edge)
    per variable × height band (+ 2-D surface diagnostics). QVAPOR/Q2 reported in g/kg."""
    band = band_index(frames, geo)
    nz, ny, nx = band.shape
    j, i = np.mgrid[0:ny, 0:nx]
    inner = np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i)) >= interior
    if not inner.any():
        raise ValueError(f"no interior cells at {interior} rows from the edge on a {ny}x{nx} grid")
    out = {"interior_rows": interior, "interior_cells": int(inner.sum()), "bands_m": [list(b) for b in BANDS]}
    for name in names:
        if name not in frames:
            continue
        s = _mass(name, frames[name]).std(axis=0, ddof=1)
        if name in ("QVAPOR",):
            s = s * 1000.0
        rows = {}
        for b, (lo, hi) in enumerate(BANDS):
            sel = (band == b) & inner[None]
            rows[f"{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km"] = (
                float(np.sqrt(np.mean(s[sel] ** 2))) if sel.any() else None)
        out[name] = rows
    for name in surface:
        if name in frames:
            s = np.asarray(frames[name], np.float64).std(axis=0, ddof=1) * (1000.0 if name == "Q2" else 1.0)
            out[name] = float(np.sqrt(np.mean(s[inner] ** 2)))
    return out


def survival(before: Mapping, after: Mapping) -> dict:
    """Ratio after/before of two :func:`spread_by_band` results (same keys)."""
    out = {}
    for name, row in after.items():
        ref = before.get(name)
        if isinstance(row, dict) and isinstance(ref, dict):
            out[name] = {k: (row[k] / ref[k] if row.get(k) and ref.get(k) else None) for k in row}
        elif isinstance(row, float) and isinstance(ref, float) and ref > 0:
            out[name] = row / ref
    return out


# --------------------------------------------------------------------------- analysis.analyse maintenance hook

def write_pool_npz(path, deviations: Mapping[str, np.ndarray], provenance: Mapping) -> dict:
    """Store CENTRED pool deviations (name → (M, …); log-deviations for LOG_FIELDS) + provenance JSON, no clobber."""
    import hashlib
    import json
    import os
    from pathlib import Path
    path = Path(path)
    if path.exists():
        raise ValueError(f"{path}: refusing to overwrite a pool file")
    arrays = {n: np.asarray(v, np.float32) for n, v in deviations.items()}
    m = {a.shape[0] for a in arrays.values()}
    if len(m) != 1:
        raise ValueError("all pool fields need the same number of samples")
    meta = {"schema": "w41.additive_pool.v1", "n_samples": m.pop(), "fields": sorted(arrays),
            "log_fields": [n for n in arrays if n in LOG_FIELDS], "provenance": dict(provenance)}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, __meta__=np.frombuffer(json.dumps(meta, default=str).encode(), np.uint8), **arrays)
    os.link(tmp, path)
    tmp.unlink()
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def load_pool_npz(path) -> dict:
    import json
    with np.load(path) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        if meta.get("schema") != "w41.additive_pool.v1":
            raise ValueError(f"{path}: not an additive pool file")
        out = {n: np.asarray(z[n], np.float64) for n in meta["fields"]}
    for name, a in out.items():
        if np.abs(a.mean(axis=0)).max() > 1e-4 * max(1.0, float(np.abs(a).max())):
            raise ValueError(f"{path}: pool field {name} is not centred")
    return out


def make_maintenance(spec: Mapping | None = None, *, pool: Mapping | None = None, amplitude=None, seed=None):
    """``callable(name, xb, xa) -> xa`` for ``analysis.analyse(maintenance=...)``.

    spec: {'method': 'rtps'|'rtpp'|'multiplicative'|'none', 'alpha', 'target': 'raw'|'inflated', 'rho', 'factor'}
    (default RTPS 0.8 toward the raw prior = the probe behaviour). With ``pool``: seeded centred additive inflation
    after the relaxation for every pooled field, ONE weight matrix for all fields of the analysis (drawn at the first
    call from ``seed``; the member count comes from ``xa``). QVAPOR/Q2: the analysis passes them in log space under
    the ``log`` moisture policy (all values < 0, since q < 1 kg/kg) — the pooled LOG-deviations are then added
    linearly; in physical space they act multiplicatively and are re-centred (positive, mean kept).
    """
    spec = dict(spec or {})
    method = spec.get("method", "rtps")
    alpha = float(spec.get("alpha", 0.8))
    rho_spec = spec.get("rho", 1.0) if spec.get("target", "raw") == "inflated" else 1.0
    rho_label = "calibrated" if callable(rho_spec) else float(rho_spec)
    factor = float(spec.get("factor", 1.0))
    state = {"z": None, "calls": 0}

    def maintenance(name, xb, xa):
        if method == "rtps":
            out = rtps(xb, xa, alpha, rho=float(rho_spec()) if callable(rho_spec) else float(rho_spec))
        elif method == "rtpp":
            out = rtpp(xb, xa, alpha)
        elif method == "multiplicative":
            out = posterior_inflation(xa, factor)
        elif method == "none":
            out = np.asarray(xa, np.float64)
        else:
            raise ValueError(f"unknown maintenance method {method!r}")
        if pool is not None and name in pool:
            if state["z"] is None:
                m = int(np.shape(next(iter(pool.values())))[0])
                state["z"] = gaussian_combinations(int(out.shape[0]), m, seed)
            amp = amplitude[name] if isinstance(amplitude, Mapping) else amplitude
            if name in LOG_FIELDS and np.all(out < 0):          # log-space moisture policy: linear in log q
                eta = np.einsum("im,m...->i...", state["z"], np.asarray(pool[name], np.float64))
                out = out + float(amp) * (eta - eta.mean(axis=0)[None])
            else:
                out = additive_inflation({name: out}, {name: pool[name]}, amp, seed=None, weights=state["z"])[0][name]
                out = np.asarray(out, np.float64)
        state["calls"] += 1
        return out

    maintenance.__name__ = (f"w41:{method}(alpha={alpha}, target_rho={rho_label})"
                            + ("" if pool is None else f"+additive(seed={seed})"))
    return maintenance
