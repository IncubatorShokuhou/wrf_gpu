"""Analysis grid, R-localisation tables and weight interpolation for the LETKF.

LETKF weights vary smoothly, so they are computed on a sub-sampled set of mass
points (every ``stride_h`` columns, every ``stride_v`` model levels, always
including the last index) and interpolated linearly in index space to every
point of each analysed field (Yang et al. 2009, QJRMS 135, 251). With
``stride_h = stride_v = 1`` the interpolation is the identity.

Horizontal distances are chord lengths on a sphere of radius 6370 km (equal to
the great-circle distance to < 0.1 % below 1000 km); vertical distances are
|ln p| differences. Localisation is R-localisation with a Gaspari-Cohn taper
per observation: rho = GC(dh / c_h) * GC(dv / c_v), c = sqrt(10/3) L. An
observation with c_v = inf (column observation) has no vertical taper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .core import apply_transform, gaspari_cohn, letkf_transform

EARTH_RADIUS_KM = 6370.0


def lonlat_to_xyz_km(lat, lon):
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64))
    return EARTH_RADIUS_KM * np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def subsample_indices(n, stride):
    if n < 1 or stride < 1:
        raise ValueError("size and stride must be >= 1")
    return np.unique(np.r_[np.arange(0, n, stride), n - 1]).astype(np.int64)


@dataclass(frozen=True)
class AnalysisGrid:
    """Coarse analysis points on the mass grid (k, j, i index space)."""

    kc: np.ndarray
    jc: np.ndarray
    ic: np.ndarray
    xyz: np.ndarray       # (n_pts, 3) km
    lnp: np.ndarray       # (n_pts,) ln(Pa) of the background-mean full pressure
    mass_shape: tuple     # (nz, ny, nx)

    @property
    def shape(self):
        return (len(self.kc), len(self.jc), len(self.ic))

    @property
    def n_points(self):
        return int(np.prod(self.shape))

    @classmethod
    def from_mass_grid(cls, lat, lon, lnp, *, stride_h=3, stride_v=2):
        lnp = np.asarray(lnp, dtype=np.float64)
        nz, ny, nx = lnp.shape
        if np.shape(lat) != (ny, nx) or np.shape(lon) != (ny, nx):
            raise ValueError("lat/lon must be (ny, nx) on the mass grid")
        kc, jc, ic = subsample_indices(nz, stride_v), subsample_indices(ny, stride_h), subsample_indices(nx, stride_h)
        xyz2 = lonlat_to_xyz_km(np.asarray(lat)[np.ix_(jc, ic)], np.asarray(lon)[np.ix_(jc, ic)])
        xyz = np.broadcast_to(xyz2[None], (len(kc), len(jc), len(ic), 3)).reshape(-1, 3)
        return cls(kc, jc, ic, np.ascontiguousarray(xyz), lnp[np.ix_(kc, jc, ic)].reshape(-1),
                   (nz, ny, nx))


def _candidates(tree, query, k, radius):
    k = min(k, tree.n)
    dist, idx = tree.query(query, k=k, distance_upper_bound=radius)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    truncated = np.isfinite(dist[:, -1]) & (k < tree.n)
    return idx, truncated


def local_obs_table(pts_xyz, pts_lnp, obs_xyz, obs_lnp, obs_ch, obs_cv, *, pmax=400, kquery=None,
                    chunk=8192, workers=1):
    """Per analysis point: up to ``pmax`` observations with the largest rho > 0.

    obs_ch: horizontal GC half-widths (km); obs_cv: vertical half-widths (ln p),
    inf for column observations (no vertical taper). Returns
    (idx (n_pts, pmax) int64, rho (n_pts, pmax) float64, n_truncated) where
    padding slots have rho = 0 and idx = 0. ``n_truncated`` counts points whose
    candidate search hit the ``kquery`` limit (possible missed observations).
    """
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
    pts_lnp = np.asarray(pts_lnp, dtype=np.float64)
    obs_xyz = np.asarray(obs_xyz, dtype=np.float64)
    obs_lnp = np.asarray(obs_lnp, dtype=np.float64)
    obs_ch = np.asarray(obs_ch, dtype=np.float64)
    obs_cv = np.asarray(obs_cv, dtype=np.float64)
    n_pts, n_obs = len(pts_xyz), len(obs_xyz)
    idx_out = np.zeros((n_pts, pmax), dtype=np.int64)
    rho_out = np.zeros((n_pts, pmax), dtype=np.float64)
    if n_obs == 0:
        return idx_out, rho_out, 0
    if np.any(~np.isfinite(obs_ch) | (obs_ch <= 0)) or np.any(np.isnan(obs_cv) | (obs_cv <= 0)):
        raise ValueError("localisation half-widths must be positive (c_v may be inf)")
    kquery = int(kquery or 4 * pmax)
    groups = []
    level = np.isfinite(obs_cv)
    if level.any():
        sel = np.flatnonzero(level)
        if np.any(~np.isfinite(obs_lnp[sel])):
            raise ValueError("level observations need finite ln p")
        sh, sv = obs_ch[sel].max(), obs_cv[sel].max()
        tree = cKDTree(np.column_stack((obs_xyz[sel] / sh, obs_lnp[sel] / sv)))
        groups.append((sel, tree, lambda p, sh=sh, sv=sv: np.column_stack((p[0] / sh, p[1][:, None] / sv)),
                       2.0 * np.sqrt(2.0) * (1 + 1e-12)))
    if (~level).any():
        sel = np.flatnonzero(~level)
        sh = obs_ch[sel].max()
        tree = cKDTree(obs_xyz[sel] / sh)
        groups.append((sel, tree, lambda p, sh=sh: p[0] / sh, 2.0 * (1 + 1e-12)))
    n_truncated = 0
    for start in range(0, n_pts, chunk):
        stop = min(start + chunk, n_pts)
        pxyz, plnp = pts_xyz[start:stop], pts_lnp[start:stop]
        cand_idx, cand_rho = [], []
        for sel, tree, scale, radius in groups:
            local, truncated = _candidates(tree, scale((pxyz, plnp)), kquery, radius)
            n_truncated += int(truncated.sum())
            valid = local < tree.n
            gidx = np.where(valid, sel[np.minimum(local, tree.n - 1)], 0)
            dh = np.linalg.norm(pxyz[:, None, :] - obs_xyz[gidx], axis=-1)
            rho = gaspari_cohn(dh / obs_ch[gidx])
            cv = obs_cv[gidx]
            vert = np.isfinite(cv)
            dv = np.abs(plnp[:, None] - np.where(vert, obs_lnp[gidx], 0.0))
            rho = rho * np.where(vert, gaspari_cohn(dv / np.where(vert, cv, 1.0)), 1.0)
            rho = np.where(valid, rho, 0.0)
            cand_idx.append(gidx)
            cand_rho.append(rho)
        gidx = np.concatenate(cand_idx, axis=1)
        rho = np.concatenate(cand_rho, axis=1)
        if gidx.shape[1] > pmax:
            top = np.argpartition(-rho, pmax - 1, axis=1)[:, :pmax]
            gidx = np.take_along_axis(gidx, top, axis=1)
            rho = np.take_along_axis(rho, top, axis=1)
        width = gidx.shape[1]
        idx_out[start:stop, :width] = np.where(rho > 0, gidx, 0)
        rho_out[start:stop, :width] = rho
    return idx_out, rho_out, n_truncated


def analysis_weights(yb, yo, sigma_o, idx, rho_loc, *, inflation=1.0, chunk=4096):
    """LETKF transforms at every analysis point.

    yb: (N, n_obs) model equivalents; yo, sigma_o: (n_obs,); idx/rho_loc from
    ``local_obs_table``. ``inflation`` = scalar or one prior inflation per analysis point (n_pts,).
    Returns T (n_pts, N, N).
    """
    yb = np.asarray(yb, dtype=np.float64)
    ybar = yb.mean(axis=0)
    ypert = (yb - ybar).T                                  # (n_obs, N)
    d = np.asarray(yo, dtype=np.float64) - ybar
    rinv = 1.0 / np.asarray(sigma_o, dtype=np.float64) ** 2
    n_pts, n_members = idx.shape[0], yb.shape[0]
    per_point = np.ndim(inflation) > 0
    if per_point:
        inflation = np.asarray(inflation, dtype=np.float64).reshape(-1)
        if inflation.shape != (n_pts,) or not (np.isfinite(inflation).all() and (inflation > 0).all()):
            raise ValueError(f"per-point inflation must be ({n_pts},) finite and > 0")
    if yb.shape[1] == 0:  # no assimilated observations: identity transform everywhere (rho = inflation)
        if per_point:
            _, _, t = letkf_transform(np.zeros((n_pts, 1, n_members)), np.zeros((n_pts, 1)), np.zeros((n_pts, 1)),
                                      rho=inflation)
            return t
        _, _, t = letkf_transform(np.zeros((1, 1, n_members)), np.zeros((1, 1)), np.zeros((1, 1)), rho=inflation)
        return np.broadcast_to(t[0], (n_pts, n_members, n_members)).copy()
    out = np.empty((n_pts, n_members, n_members), dtype=np.float64)
    for start in range(0, n_pts, chunk):
        sl = slice(start, min(start + chunk, n_pts))
        i = idx[sl]
        _, _, t = letkf_transform(ypert[i], d[i], rho_loc[sl] * rinv[i], rho=inflation[sl] if per_point else inflation)
        out[sl] = t
    return out


def _axis_interp(coarse, fine_coord):
    coarse = np.asarray(coarse, dtype=np.float64)
    x = np.clip(np.asarray(fine_coord, dtype=np.float64), coarse[0], coarse[-1])
    if len(coarse) == 1:
        z = np.zeros(x.shape, dtype=np.int64)
        return z, z, np.zeros(x.shape)
    lo = np.clip(np.searchsorted(coarse, x, side="right") - 1, 0, len(coarse) - 2)
    frac = (x - coarse[lo]) / (coarse[lo + 1] - coarse[lo])
    return lo, lo + 1, frac


def transform_field(xb, t_coarse, grid: AnalysisGrid, *, stagger=None):
    """Apply interpolated transforms to one field of the ensemble.

    xb: (N, nz, ny, nx[+1]) or (N, ny, nx) background members on the field's
    own grid; t_coarse: (nkc, njc, nic, N, N); stagger in {None, 'x', 'y'}.
    2-D fields use the lowest analysis level's weights. Returns xa (N, …).
    """
    xb = np.asarray(xb, dtype=np.float64)
    t_coarse = np.asarray(t_coarse)
    two_d = xb.ndim == 3
    field = xb[:, None] if two_d else xb
    _, nz, ny, nx = field.shape
    jf = np.arange(ny, dtype=np.float64) - (0.5 if stagger == "y" else 0.0)
    if_ = np.arange(nx, dtype=np.float64) - (0.5 if stagger == "x" else 0.0)
    jlo, jhi, fj = _axis_interp(grid.jc, jf)
    ilo, ihi, fi = _axis_interp(grid.ic, if_)
    if two_d:
        klo, khi, fk = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1)
    else:
        if nz != grid.mass_shape[0]:
            raise ValueError("vertically staggered fields are not analysed")
        klo, khi, fk = _axis_interp(grid.kc, np.arange(nz, dtype=np.float64))
    out = np.empty_like(field)
    for k in range(nz):
        tk = (1.0 - fk[k]) * t_coarse[klo[k]] + fk[k] * t_coarse[khi[k]]          # (njc, nic, N, N)
        ti = tk[:, ilo] * (1.0 - fi)[None, :, None, None] + tk[:, ihi] * fi[None, :, None, None]
        tj = ti[jlo] * (1.0 - fj)[:, None, None, None] + ti[jhi] * fj[:, None, None, None]
        out[:, k] = apply_transform(field[:, k], tj)
    return out[:, 0] if two_d else out
