"""Balanced initial/boundary ensemble perturbations — OPT-IN (lane w41-da-ens; R3 item 5, PLAN2 W3).

R3 measured that 36 % of the wind spread aloft disappears in the FIRST forecast hour of the archive-perturbed members
(V 2–5 km 3.80 → 2.43 m/s) and attributed it to an unbalanced part of the perturbations [I]. The archive
construction scales same-phase deviations of states from different days (x_ctrl + s·(x_i − x̄)) for U/V/T/QVAPOR
only, FREEZES the column mass MU and re-derives P/PH hydrostatically at fixed MU. The wind deviation of a model state
is in (near-)geostrophic balance with ITS mass field, of which the frozen-MU members keep only the T part: the
surface-pressure part of that balance is dropped, and the taper adds a second source. This module provides:

* :func:`combination_perturbations` — member deviations as seeded Gaussian combinations of centred pool deviations
  (each member a linear combination of model states: linearly balanced; Cov = pool covariance in expectation),
  MU INCLUDED when the pool has it; LOG_FIELDS (QVAPOR) in log space.
* :func:`forecast_deviations` — deviations of N forecasts valid at the same time (spun-up / lagged / NMC-pair
  members): model-balanced and flow-dependent by construction; recentring them on a control is the standard
  breeding/lagged-forecast cold start.
* :func:`rebalance_with_mu` — the fixed-QV hydrostatic rebalance of ``state_io.rebalance_increment`` extended to a
  MU increment, using WRF real.exe's top-down pressure recursion (module_initialize_real.F:3923–3953, total moisture
  qtot, ``c1f·MU_2`` mass coupling) and the HYPSOMETRIC_OPT 2 geopotential integration (:3974–3986) for both the
  background and the perturbed column, applied as an increment (the background's own non-hydrostatic residual is
  kept). With ΔMU = 0 it reproduces ``rebalance_increment`` to round-off (test).
* :func:`balance_diagnostics` — perturbation divergence/vorticity rms ratio per height band (a cheap imbalance
  indicator: geostrophically balanced perturbations are vorticity-dominated) and the mass-coupled share.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .spread import LOG_FIELDS, gaussian_combinations, pool_from_states

MOIST_SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL")


def combination_perturbations(pool_states: Mapping[str, np.ndarray], n_members: int, *, seed,
                              names: Iterable[str] = ("U", "V", "T", "QVAPOR", "MU"), weights=None,
                              centre: bool = True) -> tuple[dict, np.ndarray]:
    """(deviations name → (N, …) centred over members, weights (N, M)). Log-deviations for LOG_FIELDS.
    ``centre=False`` keeps each row's own combination (e.g. a single OSSE truth draw)."""
    names = tuple(n for n in names if n in pool_states)
    pool = pool_from_states(pool_states, names)
    m = int(np.shape(pool[names[0]])[0])
    z = gaussian_combinations(n_members, m, seed) if weights is None else np.asarray(weights, np.float64)
    out = {}
    for name in names:
        eta = np.einsum("im,m...->i...", z, pool[name])
        out[name] = eta - eta.mean(axis=0)[None] if centre else eta
    return out, z


def forecast_deviations(member_states: Mapping[str, np.ndarray], names: Iterable[str] | None = None) -> dict:
    """Centred deviations of N forecasts valid at the same time (log-deviations for LOG_FIELDS)."""
    return pool_from_states(member_states, names)


def recentre(control: Mapping[str, np.ndarray], deviations: Mapping[str, np.ndarray], scale=1.0, *,
             taper: Mapping[str, np.ndarray] | None = None, centre: bool = True) -> dict:
    """Member fields control + scale·taper·δ (LOG_FIELDS: control·exp(scale·taper·δ) re-centred multiplicatively,
    positive, member mean = control). ``scale`` scalar or name → scalar; ``taper`` name → 2-D weights or absent
    (untapered: required when the members also get matched perturbed LBCs). ``centre=False``: no re-centring
    over the member axis (single draws: control + δ, LOG_FIELDS control·exp(δ))."""
    out = {}
    for name, dev in deviations.items():
        s = float(scale[name] if isinstance(scale, Mapping) else scale)
        ref = np.asarray(control[name], np.float64)
        d = s * np.asarray(dev, np.float64)
        if taper is not None and name in taper:
            d = d * np.asarray(taper[name], np.float64)
        if name in LOG_FIELDS:
            f = np.exp(d)
            out[name] = np.maximum(ref, 0.0)[None] * (f / f.mean(axis=0)[None] if centre else f)
        else:
            out[name] = ref[None] + (d - d.mean(axis=0)[None] if centre else d)
    return out


# --------------------------------------------------------------------------- hydrostatic rebalance incl. MU

def _col(geo, name):
    return np.asarray(geo[name], np.float64).reshape(-1)[:, None, None]


def total_moisture(frame: Mapping[str, np.ndarray], qv=None) -> np.ndarray:
    """qtot of the real.exe recursion: Σ moist species present in ``frame`` (QVAPOR replaced by ``qv`` if given)."""
    q = np.asarray(frame["QVAPOR"] if qv is None else qv, np.float64).copy()
    for name in MOIST_SPECIES[1:]:
        if name in frame:
            q = q + np.asarray(frame[name], np.float64)
    return q


def pressure_recursion(mu, qtot, geo) -> np.ndarray:
    """WRF real.exe perturbation pressure p′ (nz, ny, nx) from the top down (module_initialize_real.F:3934/3953):
    p(kte−1) = −0.5 (c1f MU_2 + qvf1 (c1f MUB + c2f)) / rdnw / qvf2, then
    p(kk) = p(kk+1) − (c1f MU_2 + qvf1 (c1f MUB + c2f)) / qvf2 / rdn(kk+1), qtot averaged to the face below."""
    mu = np.asarray(mu, np.float64)
    mub = np.asarray(geo["MUB"], np.float64)
    q = np.asarray(qtot, np.float64)
    c1f, c2f = (np.asarray(geo[n], np.float64).reshape(-1) for n in ("C1F", "C2F"))
    rdnw, rdn = (np.asarray(geo[n], np.float64).reshape(-1) for n in ("RDNW", "RDN"))
    nz = q.shape[0]
    p = np.empty_like(q)
    qvf2 = 1.0 / (1.0 + q[nz - 1])
    qvf1 = q[nz - 1] * qvf2
    p[nz - 1] = -0.5 * ((c1f[nz] * mu) + qvf1 * (c1f[nz] * mub + c2f[nz])) / rdnw[nz - 1] / qvf2
    for kk in range(nz - 2, -1, -1):
        k = kk + 1
        qt = 0.5 * (q[kk] + q[kk + 1])
        qvf2 = 1.0 / (1.0 + qt)
        qvf1 = qt * qvf2
        p[kk] = p[kk + 1] - ((c1f[k] * mu) + qvf1 * (c1f[k] * mub + c2f[k])) / qvf2 / rdn[k]
    return p


def _hyps2(mu, geo) -> np.ndarray:
    """HYPSOMETRIC_OPT 2 layer factor phm·ln(pfd/pfu) with MU0 = MUB + MU (real.exe :3978–3981)."""
    mut = np.asarray(geo["MUB"], np.float64) + np.asarray(mu, np.float64)
    p_top = float(geo["P_TOP"])
    c3f, c4f = _col(geo, "C3F"), _col(geo, "C4F")
    pfu = c3f[1:] * mut + c4f[1:] + p_top
    pfd = c3f[:-1] * mut + c4f[:-1] + p_top
    phm = _col(geo, "C3H") * mut + _col(geo, "C4H") + p_top
    return phm * np.log(pfd / pfu)


def rebalance_with_mu(bg: Mapping, an: Mapping, geo: Mapping, *, return_increments: bool = False) -> dict:
    """Hydrostatic rebalance for analysed T/QVAPOR AND MU (HYPSOMETRIC_OPT 2 only), as increments on ``bg``.

    Δp = p_rec(MU_a, qtot_a) − p_rec(MU_b, qtot_b) (condensate from ``bg``, unchanged); α from the EOS
    (``state_io.eos_alpha``) at p_b + Δp; Δφ(k) = Σ_{k'<k} [α_a F(MU_a) − α_b F(MU_b)] with φ(surface) fixed;
    AL = AL_b + Δα; P_HYD = P_HYD_b + Δp. Returns P, PH, MU (+ P_HYD/AL when in ``bg``) in the background dtypes.
    """
    from .state_io import eos_alpha

    if int(geo["HYPSOMETRIC_OPT"]) != 2:
        raise ValueError("rebalance_with_mu supports HYPSOMETRIC_OPT 2 only")
    qv_a = np.asarray(an["QVAPOR"], np.float64)
    mu_a = np.asarray(an.get("MU", bg["MU"]), np.float64)
    mu_b = np.asarray(bg["MU"], np.float64)
    if not (np.isfinite(qv_a).all() and np.isfinite(np.asarray(an["T"], np.float64)).all() and np.isfinite(mu_a).all()):
        raise ValueError("non-finite analysed T/QVAPOR/MU")
    if (qv_a < 0).any():
        raise ValueError(f"negative analysed QVAPOR in {int((qv_a < 0).sum())} cells")
    if ((np.asarray(geo["MUB"], np.float64) + mu_a) <= 0).any():
        raise ValueError("non-positive analysed dry column mass")
    dp = pressure_recursion(mu_a, total_moisture(bg, qv_a), geo) - pressure_recursion(mu_b, total_moisture(bg), geo)
    p_b = np.asarray(bg["P"], np.float64)
    alpha_b = eos_alpha(bg["T"], bg["QVAPOR"], p_b, geo)
    alpha_a = eos_alpha(an["T"], qv_a, p_b + dp, geo)
    if not ((p_b + dp + np.asarray(geo["PB"], np.float64) > 0).all() and (alpha_a > 0).all()):
        raise ValueError("non-positive analysed pressure or inverse density")
    layer = alpha_a * _hyps2(mu_a, geo) - alpha_b * _hyps2(mu_b, geo)
    dph = np.zeros((layer.shape[0] + 1,) + layer.shape[1:])
    np.cumsum(layer, axis=0, out=dph[1:])
    out = {}
    for name, inc in (("P", dp), ("PH", dph)):
        ref = np.asarray(bg[name])
        out[name] = (ref.astype(np.float64) + inc).astype(ref.dtype)
    out["MU"] = mu_a.astype(np.asarray(bg["MU"]).dtype)
    if "P_HYD" in bg:
        ref = np.asarray(bg["P_HYD"])
        out["P_HYD"] = (ref.astype(np.float64) + dp).astype(ref.dtype)
    if "AL" in bg:
        ref = np.asarray(bg["AL"])
        out["AL"] = (ref.astype(np.float64) + (alpha_a - alpha_b)).astype(ref.dtype)
    if return_increments:
        out["increments"] = {"P": dp, "PH": dph, "ALPHA": alpha_a - alpha_b, "MU": mu_a - mu_b}
    return out


def hydrostatic_residual_change(bg: Mapping, an: Mapping, out: Mapping, geo: Mapping) -> float:
    """max |r_a − r_b| / layer thickness of the model-form (opt 2) residual Δφ − α F(MU)."""
    from .state_io import eos_alpha

    def resid(t, qv, p, ph, mu):
        ph = np.asarray(ph, np.float64)
        phb = np.asarray(geo["PHB"], np.float64)
        return (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]) - eos_alpha(t, qv, p, geo) * _hyps2(mu, geo)

    r_b = resid(bg["T"], bg["QVAPOR"], bg["P"], bg["PH"], bg["MU"])
    r_a = resid(an["T"], an["QVAPOR"], out["P"], out["PH"], out.get("MU", an.get("MU", bg["MU"])))
    thick = np.abs(np.diff(np.asarray(out["PH"], np.float64) + np.asarray(geo["PHB"], np.float64), axis=0))
    return float((np.abs(r_a - r_b) / thick).max())


# --------------------------------------------------------------------------- diagnostics

def _div_vort(u, v, dx, dy):
    """C-grid divergence (mass points) and relative vorticity (interior corner points), per level."""
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    div = (u[..., 1:] - u[..., :-1]) / dx + (v[..., 1:, :] - v[..., :-1, :]) / dy
    vort = (v[..., 1:-1, 1:] - v[..., 1:-1, :-1]) / dx - (u[..., 1:, 1:-1] - u[..., :-1, 1:-1]) / dy
    return div, vort


def balance_diagnostics(deviations: Mapping[str, np.ndarray], geo: Mapping, frames: Mapping | None = None) -> dict:
    """rms(divergence)/rms(vorticity) of the member wind deviations (all members, per band when ``frames`` with
    PH/PHB is given) and the relative rms of the MU deviation; smaller divergence ratio = more rotational
    (geostrophic-like) perturbation. ``geo`` needs DX/DY (m)."""
    from .spread import BANDS, band_of_height, height_agl
    dx, dy = float(geo["DX"]), float(geo["DY"])
    du = np.asarray(deviations["U"], np.float64)
    dv = np.asarray(deviations["V"], np.float64)
    div, vort = _div_vort(du, dv, dx, dy)
    out = {"div_over_vort_all": float(np.sqrt(np.mean(div[..., 1:-1, 1:-1] ** 2)) / max(np.sqrt(np.mean(vort ** 2)), 1e-30))}
    if frames is not None:
        zm = height_agl(frames, geo)
        band_div = band_of_height(zm[:, 1:-1, 1:-1])             # interior mass points (div[..., 1:-1, 1:-1])
        # interior corner (j+1/2, i+1/2) of the vorticity: band of the mean height of its 4 mass points
        band_vort = band_of_height(0.25 * (zm[:, :-1, :-1] + zm[:, 1:, :-1] + zm[:, :-1, 1:] + zm[:, 1:, 1:]))
        rows = {}
        for b, (lo, hi) in enumerate(BANDS):
            sel_d, sel_w = band_div == b, band_vort == b
            if sel_d.any() and sel_w.any():
                d = div[..., 1:-1, 1:-1][..., sel_d]
                w = vort[..., sel_w]
                rows[f"{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km"] = float(
                    np.sqrt(np.mean(d ** 2)) / max(np.sqrt(np.mean(w ** 2)), 1e-30))
        out["div_over_vort_by_band"] = rows
    if "MU" in deviations:
        out["mu_dev_rms_pa"] = float(np.sqrt(np.mean(np.asarray(deviations["MU"], np.float64) ** 2)))
    return out
