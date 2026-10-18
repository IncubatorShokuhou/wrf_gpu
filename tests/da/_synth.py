"""Synthetic WRF-like ensembles for the gpuwrf da product tests (no model, no files).

The observation operator is injected (point sampling of a field at grid indices), so the LETKF/analysis tests
check the algebra and the hooks independently of ``operators``.
"""
from __future__ import annotations

import types

import numpy as np

from gpuwrf.assimilation.letkf import analysis as an
from gpuwrf.assimilation.letkf import obs as ob


def _smooth(a):
    return (a + np.roll(a, 1, -1) + np.roll(a, -1, -1) + np.roll(a, 1, -2) + np.roll(a, -1, -2)) / 5.0


def ensemble(seed=0, n=10, nz=6, ny=30, nx=34, qv=0.008, qv_spread=0.001, qv_lognormal=False):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(27.0 + 0.08 * np.arange(ny), -18.0 + 0.09 * np.arange(nx), indexing="ij")
    eta_full = np.linspace(1.0, 0.0, nz + 1)
    p_top = 5000.0
    mub = np.full((ny, nx), 95000.0)
    geo = {"XLAT": lat, "XLONG": lon, "HGT": np.zeros((ny, nx)), "C3F": eta_full, "C4F": np.zeros(nz + 1),
           "P_TOP": p_top, "MUB": mub}
    p_full_levels = eta_full[:, None, None] * mub + p_top
    pb = 0.5 * (p_full_levels[:-1] + p_full_levels[1:])
    frames = {
        "T": 10 + np.stack([_smooth(_smooth(rng.normal(size=(nz, ny, nx)))) for _ in range(n)]),
        "QVAPOR": (qv * np.exp(qv_spread * np.stack([_smooth(rng.normal(size=(nz, ny, nx))) for _ in range(n)]))
                   if qv_lognormal else qv + qv_spread * np.stack([_smooth(rng.normal(size=(nz, ny, nx)))
                                                                   for _ in range(n)])),
        "U": rng.normal(size=(n, nz, ny, nx + 1)), "V": rng.normal(size=(n, nz, ny + 1, nx)),
        "P": np.zeros((n, nz, ny, nx)), "PB": np.broadcast_to(pb, (n, nz, ny, nx)).copy(),
        "PSFC": np.full((n, ny, nx), mub[0, 0] + p_top), "MU": np.zeros((n, ny, nx)),
    }
    return frames, geo


def obs_at(points, values, sigma, frames, geo, *, var=ob.T):
    """Observations at mass points (k, j, i); ``var`` T/QV (sonde) or T2 (surface, k ignored)."""
    k, j, i = (np.array(x) for x in zip(*points))
    kind = ob.VAR_KIND[var]
    o = types.SimpleNamespace(kind=np.full(len(k), kind, int), var=np.full(len(k), var, int),
                              value=np.asarray(values, float), error=np.asarray(sigma, float) * np.ones(len(k)),
                              k=k, j=j, i=i, qc=np.zeros(len(k), int))
    o.lat = geo["XLAT"][j, i]
    o.lon = geo["XLONG"][j, i]
    o.p_pa = frames["PB"][0][k, j, i] if kind == ob.SONDE else np.full(len(k), np.nan)
    return o


def concat(*parts):
    out = types.SimpleNamespace()
    for name in ("kind", "var", "value", "error", "k", "j", "i", "qc", "lat", "lon", "p_pa"):
        setattr(out, name, np.concatenate([getattr(p, name) for p in parts]))
    return out


def hx(obs, frames, geo):
    """Point sampling: T/QV at (k, j, i); surface T2 samples T at the lowest level."""
    out = np.empty((next(iter(frames.values())).shape[0], len(obs.value)))
    for n, (v, k, j, i) in enumerate(zip(obs.var, obs.k, obs.j, obs.i)):
        if v == ob.QV:
            out[:, n] = np.asarray(frames["QVAPOR"])[:, k, j, i]
        elif v == ob.T2:
            out[:, n] = np.asarray(frames["T"])[:, 0, j, i]
        else:
            out[:, n] = np.asarray(frames["T"])[:, k, j, i]
    return out


PARAMS = an.LetkfParams(stride_h=1, stride_v=1, rtps_alpha=0.0, rebalance=False, variables=("T", "QVAPOR", "U", "V"),
                        diagnostic_variables=(), loc_h_km=40.0, loc_v_lnp=0.2, loc_v_lnp_low=0.2, loc_v_sfc_lnp=0.2,
                        pmax=50)


def hydrostatic_case(nz=20, ny=12, nx=14, p_top=5000.0, seed=0):
    """Hybrid-coordinate geometry + a near-hydrostatic control frame (the tests/a1_da synthetic_geo construction),
    enough for ``state_io.rebalance_increment`` and ``analysis_mean_fields``."""
    from gpuwrf.assimilation.letkf.state_io import balance_state
    from gpuwrf.init.real_init.types import CP, CVPM, P1000MB, R_D, T0

    g = 9.81
    znw = np.linspace(1.0, 0.0, nz + 1) ** 1.3
    znw[0], znw[-1] = 1.0, 0.0
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = znw[1:] - znw[:-1]
    dn = np.concatenate(([dnw[0]], 0.5 * (dnw[1:] + dnw[:-1])))
    jj, ii = np.mgrid[0:ny, 0:nx]
    ht = 1500.0 * np.exp(-(((ii - nx / 2) / 3.0) ** 2 + ((jj - ny / 2) / 2.5) ** 2))
    p00, t00, a, tiso = 1.0e5, 290.0, 50.0, 200.0
    p_surf = p00 * np.exp(-t00 / a + np.sqrt((t00 / a) ** 2 - 2.0 * g * ht / a / R_D))
    c3h, c3f = znu[:, None, None], znw[:, None, None]
    pb = c3h * (p_surf - p_top) + p_top
    temp = np.maximum(tiso, t00 + a * np.log(pb / p00))
    t_init = temp * (p00 / pb) ** (R_D / CP) - T0
    alb = (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM
    mub = p_surf - p_top
    phb = np.empty((nz + 1, ny, nx))
    phb[0] = ht * g
    for k in range(1, nz + 1):
        pfu, pfd = c3f[k, 0, 0] * mub + p_top, c3f[k - 1, 0, 0] * mub + p_top
        phb[k] = phb[k - 1] + alb[k - 1] * (c3h[k - 1, 0, 0] * mub + p_top) * np.log(pfd / pfu)
    ones_f, zeros_f = np.ones(nz + 1), np.zeros(nz + 1)
    geo = {"ZNW": znw, "ZNU": znu, "DNW": dnw, "RDNW": 1.0 / dnw, "DN": dn, "RDN": 1.0 / dn,
           "FNP": np.zeros(nz), "FNM": np.zeros(nz), "C1F": ones_f, "C2F": zeros_f, "C3F": znw, "C4F": zeros_f,
           "C1H": np.ones(nz), "C2H": np.zeros(nz), "C3H": znu, "C4H": np.zeros(nz),
           "MUB": mub, "PB": pb, "PHB": phb, "ALB": alb, "HGT": ht}
    geo = {k: np.asarray(v, np.float32) for k, v in geo.items()}
    geo.update({"P_TOP": p_top, "CF1": 0.0, "CF2": 0.0, "CF3": 0.0, "CFN": 0.0, "CFN1": 0.0,
                "HYBRID_OPT": 2, "HYPSOMETRIC_OPT": 2})
    rng = np.random.default_rng(seed)
    theta = t_init + 0.3 * rng.standard_normal((nz, ny, nx))
    qv = 0.012 * znu[:, None, None] ** 3 * (1 + 0.1 * rng.random((nz, ny, nx)))
    mu = 300.0 * np.sin(np.arange(nx) / 3.0)[None, :] * np.ones((ny, 1))
    bal = balance_state(theta, qv, mu, geo)
    ctrl = {"T": theta, "QVAPOR": qv, "MU": mu, "P": bal["P"], "PH": bal["PH"], "P_HYD": bal["P"] + geo["PB"]}
    return geo, {k: np.asarray(v, np.float32) for k, v in ctrl.items()}
