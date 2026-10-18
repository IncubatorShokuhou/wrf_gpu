"""Observation operators H(x) on WRF history frames for the LETKF probe (lane a1-obs).

``model_equivalents(obs, frames, geo) -> (N, n_obs)`` for ONE valid time. Host NumPy fp64.

frames: dict wrfout name -> (N, ...) member arrays without the Time axis
    U (N,nz,ny,nx+1), V (N,nz,ny+1,nx), T (N,nz,ny,nx) dry theta - 300 K, QVAPOR, P, PB,
    CLDFRA (N,nz,ny,nx), T2, Q2, U10, V10, PSFC (N,ny,nx). Only the fields needed by the
    observation kinds present are required.
geo: dict XLAT, XLONG, HGT, COSALPHA, SINALPHA, LANDMASK (ny,nx mass grid) and DX [m].

Horizontal location: the containing four-node lat/lon quadrilateral, inverted with the
same Newton method as ``gpuwrf.verify.sampling.StationSampler`` (isoparametric bilinear,
locally unwrapped; affine-invariant, so one projection serves all obs). Out-of-domain
observations get NaN and QC_OUTSIDE (no nearest fallback).

- SONDE T/QV/U/V: bilinear horizontal at the obs point on each model level (U/V on their
  own staggered grids), ln p column = bilinear of ln(P + PB), linear interpolation in
  ln p between the bracketing mass levels. Outside [p_top_mass, p_lowest_mass] -> NaN +
  QC_OUTSIDE. T obs compare with temperature = (T + 300) (p/1e5)^(R_d/c_p) on levels
  (R_d/c_p = 287/1004.5, WRF). Winds are rotated grid -> earth with the bilinear
  COSALPHA/SINALPHA at the obs point (``verify.sampling.earth_wind`` convention).
- SURFACE T2/Q2/U10/V10/PSFC: bilinear (land-aware: only nodes of the station's land/sea
  class, renormalised, like StationSampler); U10/V10 rotated to earth on the full grid
  before sampling (as ``verify.wrfout.extract_wrfout``). T2 lapse-rate correction
  H = T2 - lapse (z_stn - z_model); PSFC reduced hydrostatically to z_stn with
  Tv = (T2 - lapse dz/2)(1 + 0.608 Q2). |z_stn - z_model| > dz_max -> NaN + QC_DZ
  (default 150 m, DESIGN_v2; unknown station height -> QC_DZ). Q2 has no height correction.
- CLOUD CLDMASK: bilinear of the column-maximum CLDFRA.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .obs import (CLDMASK, CLOUD, G, LAPSE_RATE, PSFC, Q2, QC_DZ, QC_OUTSIDE, QV, R_D, SONDE, SURFACE,
                  T, T2, U, U10, V, V10)

RCP = 287.0 / 1004.5  # WRF r_d / cp (cp = 7 r_d / 2)
P0 = 1.0e5


def _lon_delta(a, b):
    return (np.asarray(a, float) - b + 180.0) % 360.0 - 180.0


class GridLocator:
    """Fractional mass-grid indices (i along x, j along y) of lat/lon points."""

    def __init__(self, xlat, xlong):
        from scipy.spatial import cKDTree

        self.lat = np.asarray(xlat, dtype=float)
        self.lon = np.asarray(xlong, dtype=float)
        if self.lat.ndim != 2 or self.lat.shape != self.lon.shape or min(self.lat.shape) < 2:
            raise ValueError("XLAT/XLONG must share a >= 2x2 mass grid")
        if not (np.isfinite(self.lat).all() and np.isfinite(self.lon).all()):
            raise ValueError("XLAT/XLONG must be finite")
        self.shape = self.lat.shape
        self.lat0 = float(self.lat.mean())
        self.lon0 = float(self.lon[self.shape[0] // 2, self.shape[1] // 2])
        self.coslat = np.cos(np.deg2rad(self.lat0))
        self.x = _lon_delta(self.lon, self.lon0) * self.coslat
        self.y = self.lat - self.lat0
        self.tree = cKDTree(np.column_stack((self.x.ravel(), self.y.ravel())))

    def _project(self, lat, lon):
        return _lon_delta(lon, self.lon0) * self.coslat, np.asarray(lat, float) - self.lat0

    def locate(self, lat, lon):
        """Return (i_frac, j_frac, inside) arrays; outside points have NaN indices."""
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        px, py = self._project(lat, lon)
        ny, nx = self.shape
        _, flat = self.tree.query(np.column_stack((px, py)))
        jn, inn = np.unravel_index(flat, self.shape)
        ifrac = np.full(len(lat), np.nan)
        jfrac = np.full(len(lat), np.nan)
        done = np.zeros(len(lat), bool)
        # Same candidate order as StationSampler: y in (j-1, j), x in (i-1, i).
        for dj in (-1, 0):
            for di in (-1, 0):
                j0, i0 = jn + dj, inn + di
                ok = ~done & (j0 >= 0) & (j0 < ny - 1) & (i0 >= 0) & (i0 < nx - 1)
                if not ok.any():
                    continue
                jj, ii = j0[ok], i0[ok]
                s, t, valid = _inverse_bilinear(
                    (self.x[jj, ii], self.y[jj, ii]), (self.x[jj, ii + 1], self.y[jj, ii + 1]),
                    (self.x[jj + 1, ii], self.y[jj + 1, ii]), (self.x[jj + 1, ii + 1], self.y[jj + 1, ii + 1]),
                    px[ok], py[ok])
                idx = np.flatnonzero(ok)[valid]
                ifrac[idx] = ii[valid] + s[valid]
                jfrac[idx] = jj[valid] + t[valid]
                done[idx] = True
        return ifrac, jfrac, done


def _inverse_bilinear(p00, p10, p01, p11, px, py):
    """Vectorised Newton inversion of p00 + a s + b t + c s t = P (StationSampler tolerances)."""
    ax, ay = p10[0] - p00[0], p10[1] - p00[1]
    bx, by = p01[0] - p00[0], p01[1] - p00[1]
    cx, cy = p11[0] - p10[0] - p01[0] + p00[0], p11[1] - p10[1] - p01[1] + p00[1]
    s = np.full(len(px), 0.5)
    t = np.full(len(px), 0.5)
    for _ in range(15):
        rx = p00[0] + ax * s + bx * t + cx * s * t - px
        ry = p00[1] + ay * s + by * t + cy * s * t - py
        if np.max(np.abs(np.concatenate((rx, ry))), initial=0.0) < 1e-10:
            break
        j11, j12 = ax + cx * t, bx + cx * s
        j21, j22 = ay + cy * t, by + cy * s
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-14, np.nan, det)
        s = s - (j22 * rx - j12 * ry) / det
        t = t - (-j21 * rx + j11 * ry) / det
    rx = p00[0] + ax * s + bx * t + cx * s * t - px
    ry = p00[1] + ay * s + by * t + cy * s * t - py
    with np.errstate(invalid="ignore"):
        valid = (np.maximum(np.abs(rx), np.abs(ry)) <= 1e-8) & (s >= -1e-8) & (s <= 1 + 1e-8) & \
                (t >= -1e-8) & (t <= 1 + 1e-8)
    return np.clip(np.nan_to_num(s), 0, 1), np.clip(np.nan_to_num(t), 0, 1), valid


def _stencil(xf, yf, ny, nx):
    """Corner indices/weights for fractional (x, y) on an (ny, nx) array."""
    valid = np.isfinite(xf) & np.isfinite(yf) & (xf >= -1e-9) & (xf <= nx - 1 + 1e-9) & \
        (yf >= -1e-9) & (yf <= ny - 1 + 1e-9)
    x = np.clip(np.nan_to_num(xf), 0, nx - 1)
    y = np.clip(np.nan_to_num(yf), 0, ny - 1)
    i0 = np.minimum(np.floor(x).astype(int), nx - 2)
    j0 = np.minimum(np.floor(y).astype(int), ny - 2)
    wx, wy = x - i0, y - j0
    w = np.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy))  # (4, n)
    jj = np.stack((j0, j0, j0 + 1, j0 + 1))
    ii = np.stack((i0, i0 + 1, i0, i0 + 1))
    return jj, ii, w, valid


def _apply(field, jj, ii, w):
    """field (..., ny, nx) -> (..., n) with stencil weights w (4, n); corners are cast to float64 first."""
    return np.sum(field[..., jj, ii].astype(np.float64) * w, axis=-2)


def grid_to_earth(u, v, cosalpha, sinalpha):
    """Grid-relative -> earth-relative (``verify.sampling.earth_wind`` COS/SINALPHA form)."""
    return u * cosalpha - v * sinalpha, v * cosalpha + u * sinalpha


def earth_to_grid(u, v, cosalpha, sinalpha):
    """Inverse of :func:`grid_to_earth` for a unit (cos, sin) pair."""
    return u * cosalpha + v * sinalpha, v * cosalpha - u * sinalpha


def _members(frames, name, ndim):
    if name not in frames:
        raise KeyError(f"frames need {name}")
    arr = np.asarray(frames[name])  # native dtype: never copy whole member fields; gathers are cast to float64
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64)
    if arr.ndim == ndim - 1:
        arr = arr[None]
    if arr.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dims (member axis first), got {arr.shape}")
    return arr


_FIELD_NDIM = {"U": 3, "V": 3, "T": 3, "QVAPOR": 3, "P": 3, "PB": 3, "PH": 3, "PHB": 3, "CLDFRA": 3,
               "T2": 2, "Q2": 2, "U10": 2, "V10": 2, "PSFC": 2, "HGT": 2}


def _member_count(frames):
    for name, nd in _FIELD_NDIM.items():
        if name in frames:
            arr = np.asarray(frames[name])
            if arr.ndim == nd + 1:
                return arr.shape[0]
            if arr.ndim == nd:
                return 1
            raise ValueError(f"{name}: unexpected shape {arr.shape}")
    raise ValueError("frames contain no known wrfout field")


def _corners(frames, name, jj, ii):
    """Member field (N, [nz,] ny, nx) gathered at stencil corners -> (N, [nz,] 4, n)."""
    arr = _members(frames, name, _FIELD_NDIM[name] + 1)
    return arr[..., jj, ii].astype(np.float64)


def vertical_interp_lnp(col_f, col_lnp, lnp_obs):
    """Linear-in-ln p interpolation. col_f/col_lnp (N, nz, n) with ln p decreasing in k.

    Returns (values (N, n), valid (N, n)); outside the column -> NaN, invalid.
    """
    nz = col_lnp.shape[1]
    target = lnp_obs[None, None, :]
    k = np.sum(col_lnp >= target, axis=1) - 1
    top_hit = (k == nz - 1) & (col_lnp[:, -1, :] == lnp_obs[None, :])
    k = np.where(top_hit, nz - 2, k)
    valid = (k >= 0) & (k <= nz - 2) & np.isfinite(lnp_obs)[None, :]
    kc = np.clip(k, 0, nz - 2)[:, None, :]
    l0 = np.take_along_axis(col_lnp, kc, axis=1)[:, 0]
    l1 = np.take_along_axis(col_lnp, kc + 1, axis=1)[:, 0]
    f0 = np.take_along_axis(col_f, kc, axis=1)[:, 0]
    f1 = np.take_along_axis(col_f, kc + 1, axis=1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (lnp_obs[None, :] - l0) / (l1 - l0)
        out = f0 + w * (f1 - f0)
    return np.where(valid, out, np.nan), valid


DZ_MAX_M = 150.0  # DESIGN_v2 section 5: T2 lapse-rate correction only for |dz| <= 150 m


def default_dz_max(geo):
    """DESIGN_v2: one threshold for every domain (v1 used 300 m d01 / 200 m nests)."""
    return DZ_MAX_M


@dataclasses.dataclass(frozen=True)
class OperatorResult:
    """DESIGN_v2 section 4 operator contract for one valid time.

    hx (N, n_obs) float64 with NaN where undefined; valid (N, n_obs); reason (n_obs,) QC bits
    (QC_OUTSIDE / QC_DZ) to OR into ``obs.qc``; lnp_vert (n_obs,) the vertical coordinate actually
    used (ln p_obs for sondes, ln of the member-mean model PSFC at the station for surface obs,
    NaN for cloud); dz_m (n_obs,) z_station - z_model (surface, else NaN); corrections dict.
    """

    hx: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    lnp_vert: np.ndarray
    dz_m: np.ndarray
    corrections: dict


def model_equivalents(obs, frames, geo, *, return_qc=False, **kwargs):
    """H(x) (N, n_obs) for every observation from ONE valid-time ensemble frame (NaN where undefined).

    With ``return_qc`` returns (hx, qc_flags); see :func:`apply_operators` for the full result.
    """
    res = apply_operators(obs, frames, geo, **kwargs)
    return (res.hx, res.reason) if return_qc else res.hx


def apply_operators(obs, frames, geo, *, land_aware=True, surface_landmask=1, lapse_rate=LAPSE_RATE,
                    dz_max_m=None, locator=None):
    """Evaluate all operators; returns :class:`OperatorResult`.

    ``surface_landmask`` = land/sea class of surface stations (scalar or (n_obs,), 1 = land).
    """
    n_obs = len(obs)
    locator = locator or GridLocator(geo["XLAT"], geo["XLONG"])
    ny, nx = locator.shape
    nmem = _member_count(frames)
    hx = np.full((nmem, n_obs), np.nan)
    qc = np.zeros(n_obs, np.int32)
    lnp_vert = np.full(n_obs, np.nan)
    dz_out = np.full(n_obs, np.nan)
    dz_max = default_dz_max(geo) if dz_max_m is None else float(dz_max_m)
    corrections = {"t2_lapse_rate_K_per_m": lapse_rate, "dz_max_m": dz_max, "land_aware": bool(land_aware),
                   "psfc_reduction": "hydrostatic, Tv=(T2-lapse*dz/2)(1+0.608 Q2)",
                   "sonde_vertical": "linear in ln p between bracketing mass levels (bilinear ln(P+PB))",
                   "winds": "grid->earth with COSALPHA/SINALPHA (verify.sampling.earth_wind convention)"}

    def result():
        valid = np.isfinite(hx)
        return OperatorResult(hx=hx, valid=valid, reason=qc, lnp_vert=lnp_vert, dz_m=dz_out, corrections=corrections)

    if n_obs == 0:
        return result()
    ifr, jfr, inside = locator.locate(obs.lat, obs.lon)
    qc[~inside] |= QC_OUTSIDE

    def rotation():  # only wind operators need the map rotation
        return np.asarray(geo["COSALPHA"], float), np.asarray(geo["SINALPHA"], float)

    # --- radiosonde-type profiles -------------------------------------------------
    sonde = inside & (obs.kind == SONDE)
    if sonde.any():
        idx = np.flatnonzero(sonde)
        pts = np.stack((ifr[idx], jfr[idx]), axis=1)
        uniq, loc = np.unique(pts, axis=0, return_inverse=True)
        loc = loc.ravel()
        xi, yj = uniq[:, 0], uniq[:, 1]
        jj, ii, w, _ = _stencil(xi, yj, ny, nx)
        # Gather the four stencil corners first; all per-field arithmetic stays on corners.
        lnp_c = np.log(_corners(frames, "P", jj, ii) + _corners(frames, "PB", jj, ii))   # (N, nz, 4, n_loc)
        col_lnp = np.sum(lnp_c * w, axis=-2)[..., loc]                                      # (N, nz, n_sonde)
        lnp_obs = np.log(obs.p_pa[idx])
        var = obs.var[idx]
        out = np.full((col_lnp.shape[0], len(idx)), np.nan)
        valid_all = np.zeros((col_lnp.shape[0], len(idx)), bool)
        if np.isin(var, (T,)).any():
            temp_c = (_corners(frames, "T", jj, ii) + 300.0) * np.exp(RCP * (lnp_c - np.log(P0)))
            sel = var == T
            vals, ok = vertical_interp_lnp(np.sum(temp_c * w, axis=-2)[..., loc[sel]], col_lnp[..., sel],
                                           lnp_obs[sel])
            out[:, sel], valid_all[:, sel] = vals, ok
        if np.isin(var, (QV,)).any():
            sel = var == QV
            qcol = np.sum(_corners(frames, "QVAPOR", jj, ii) * w, axis=-2)[..., loc[sel]]
            vals, ok = vertical_interp_lnp(qcol, col_lnp[..., sel], lnp_obs[sel])
            out[:, sel], valid_all[:, sel] = vals, ok
        if np.isin(var, (U, V)).any():
            uu = _members(frames, "U", 4)
            vv = _members(frames, "V", 4)
            if uu.shape[-1] != nx + 1 or vv.shape[-2] != ny + 1:
                raise ValueError("U/V must be on their staggered grids (nx+1 / ny+1)")
            sel = np.isin(var, (U, V))
            ju, iu, wu, _ = _stencil(xi + 0.5, yj, ny, nx + 1)
            jv, iv, wv, _ = _stencil(xi, yj + 0.5, ny + 1, nx)
            ug = _apply(uu, ju, iu, wu)[..., loc[sel]]
            vg = _apply(vv, jv, iv, wv)[..., loc[sel]]
            cos_a, sin_a = rotation()
            ca = _apply(cos_a, jj, ii, w)[loc[sel]]
            sa = _apply(sin_a, jj, ii, w)[loc[sel]]
            ue, ve = grid_to_earth(ug, vg, ca[None, None, :], sa[None, None, :])
            comp = np.where((var[sel] == U)[None, None, :], ue, ve)
            vals, ok = vertical_interp_lnp(comp, col_lnp[..., sel], lnp_obs[sel])
            out[:, sel], valid_all[:, sel] = vals, ok
        hx[:, idx] = out
        lnp_vert[idx] = lnp_obs
        bad = ~valid_all.all(axis=0)
        qc[idx[bad]] |= QC_OUTSIDE

    # --- surface stations -----------------------------------------------------------
    surf = inside & (obs.kind == SURFACE)
    if surf.any():
        idx = np.flatnonzero(surf)
        jj, ii, w, _ = _stencil(ifr[idx], jfr[idx], ny, nx)
        if land_aware:
            land = np.asarray(geo["LANDMASK"], float)
            klass = np.broadcast_to(np.asarray(surface_landmask, float), (n_obs,))[idx]
            w = w * (land[jj, ii] == klass[None, :])
            total = w.sum(axis=0)
            nomatch = total <= 1e-12
            qc[idx[nomatch]] |= QC_OUTSIDE
            w = w / np.where(nomatch, 1.0, total)[None, :]
        else:
            nomatch = np.zeros(len(idx), bool)
        z_model = _apply(np.asarray(geo["HGT"], float), jj, ii, w)
        dz = obs.height_m[idx] - z_model
        dz_out[idx] = dz
        if "PSFC" in frames:
            ps_mean = _apply(_members(frames, "PSFC", 3), jj, ii, w).mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lnp_vert[idx] = np.log(ps_mean)
        too_far = ~(np.abs(dz) <= dz_max)
        qc[idx[too_far & ~nomatch]] |= QC_DZ
        reject = nomatch | too_far
        var = obs.var[idx]
        res = np.full((nmem, len(idx)), np.nan)
        need = set(var.tolist())
        t2s = _apply(_members(frames, "T2", 3), jj, ii, w) if need & {T2, PSFC} else None
        q2s = _apply(_members(frames, "Q2", 3), jj, ii, w) if need & {Q2, PSFC} else None
        if T2 in need:
            sel = var == T2
            res[:, sel] = t2s[:, sel] - lapse_rate * dz[sel][None, :]
        if Q2 in need:
            sel = var == Q2
            res[:, sel] = q2s[:, sel]
        if need & {U10, V10}:
            cos_a, sin_a = rotation()
            ue, ve = grid_to_earth(_members(frames, "U10", 3), _members(frames, "V10", 3), cos_a, sin_a)
            for code, comp in ((U10, ue), (V10, ve)):
                sel = var == code
                if sel.any():
                    res[:, sel] = _apply(comp, jj, ii, w)[:, sel]
        if PSFC in need:
            sel = var == PSFC
            ps = _apply(_members(frames, "PSFC", 3), jj, ii, w)[:, sel]
            d = dz[sel][None, :]
            tv = (t2s[:, sel] - 0.5 * lapse_rate * d) * (1.0 + 0.608 * q2s[:, sel])
            res[:, sel] = ps * np.exp(-G * d / (R_D * tv))
        res[:, reject] = np.nan
        hx[:, idx] = res

    # --- cloud mask -------------------------------------------------------------------
    cloud = inside & (obs.kind == CLOUD)
    if cloud.any():
        idx = np.flatnonzero(cloud)
        if not np.all(obs.var[idx] == CLDMASK):
            raise ValueError("unsupported cloud variable")
        jj, ii, w, _ = _stencil(ifr[idx], jfr[idx], ny, nx)
        cmax = _corners(frames, "CLDFRA", jj, ii).max(axis=1)              # (N, 4, n) column max
        hx[:, idx] = np.sum(cmax * w, axis=-2)
    return result()


def sonde_layer_superob(obs, frames, geo, *, error_mode="keep", locator=None):
    """Thin every sonde launch to <= 1 super-ob per (variable, model layer, role) (DESIGN section 4).

    Layers = ln p midpoints of the member-mean model column (P + PB) at the launch site (bilinear ln p,
    as the operator). Inputs are kept flagged QC_THINNED. Returns the new batch.
    """
    from .obs import SONDE as _SONDE, superob

    sonde = (obs.kind == _SONDE) & obs.usable
    if not sonde.any():
        return obs
    locator = locator or GridLocator(geo["XLAT"], geo["XLONG"])
    ny, nx = locator.shape
    bins = np.full(len(obs), -1, dtype=np.int64)
    keys = np.array([f"{a}|{b}" for a, b in zip(obs.station, obs.launch)], dtype=object)
    for code, key in enumerate(sorted(set(keys[sonde]))):
        members = np.flatnonzero(sonde & (keys == key))
        fi, fj, inside = locator.locate(obs.lat[members[:1]], obs.lon[members[:1]])
        if not inside[0]:
            continue
        jj, ii, w, _ = _stencil(fi, fj, ny, nx)
        lnp_c = np.log(_corners(frames, "P", jj, ii) + _corners(frames, "PB", jj, ii))      # (N, nz, 4, 1)
        col = np.sum(lnp_c * w, axis=-2)[..., 0].mean(axis=0)                             # (nz,)
        edges = 0.5 * (col[1:] + col[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            layer = np.searchsorted(-edges, -np.log(obs.p_pa[members]), side="right")
        bins[members] = code * 1000 + layer
    return superob(obs, bins, error_mode=error_mode)


def apply_operator_qc(obs, qc_flags):
    """OR operator QC bits into the batch."""
    return obs.replace(qc=obs.qc | np.asarray(qc_flags, np.int32))


def frame_index(obs, frame_dt_s, *, half_window_s=1800.0):
    """Nearest frame per observation (4D-LETKF hourly bins); -1 if none within the half window."""
    frame_dt_s = np.asarray(frame_dt_s, dtype=float)
    if frame_dt_s.size == 0:
        return np.full(len(obs), -1)
    off = np.abs(obs.dt_s[:, None] - frame_dt_s[None, :])
    best = np.argmin(off, axis=1)
    return np.where(off[np.arange(len(obs)), best] <= half_window_s, best, -1)


def model_equivalents_4d(obs, frames_list, frame_dt_s, geo, *, half_window_s=1800.0, **kwargs):
    """Evaluate each observation on its nearest-time frame. Returns (hx, qc_flags, frame_idx).

    ``frames_list`` items are frames dicts or zero-argument callables returning one (lazy: a frame without
    observations is never loaded).
    """
    from .obs import QC_OUTSIDE as _OUT

    which = frame_index(obs, frame_dt_s, half_window_s=half_window_s)
    locator = kwargs.pop("locator", None) or GridLocator(geo["XLAT"], geo["XLONG"])
    hx = qc = None
    for f, frames in enumerate(frames_list):
        sel = np.flatnonzero(which == f)
        if not len(sel):
            continue
        frames = frames() if callable(frames) else frames
        part, flags = model_equivalents(obs.subset(sel), frames, geo, return_qc=True, locator=locator, **kwargs)
        if hx is None:
            hx = np.full((part.shape[0], len(obs)), np.nan)
            qc = np.zeros(len(obs), np.int32)
        hx[:, sel] = part
        qc[sel] |= flags
    if hx is None:
        hx = np.full((1, len(obs)), np.nan)
        qc = np.zeros(len(obs), np.int32)
    qc[which < 0] |= _OUT
    return hx, qc, which
