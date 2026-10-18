"""Vertical remap of a d01 WRF case (wrfinput_d01 + wrfbdy_d01) to a refined eta set (lane a1-io, task VR).

Roadblock R1 test: does more vertical resolution in the trade-wind inversion layer (≈ 0.4–2.5 km) let WRF keep the
inversion? The refined set keeps EVERY original full eta level, the top (η = 0) and p_top, and only inserts levels.

* 1-D coordinate: real.exe formulas (``init.real_init.vertical_coord.compute_vertical_coord`` with explicit
  eta_levels, hybrid_opt/etac from the case; C1F/C1H in fp32 via ``wrf_fp32_c1_from_c3`` = real.exe bit-exact).
* Base state PB/ALB/T_INIT/MUB/PHB: ``init.real_init.base_state.compute_base_state`` (WRF start_em/real.exe form).
* Fields: linear interpolation in ln(dry hydrostatic pressure) p_d = c3·μ_d + c4 + p_top of each column (μ_d = MUB+MU,
  FIXED; U/V faces use WRF calc_mu_staggered face mass). Old and new levels share the η bracket in every column,
  no extrapolation (bottom and top layers are never split).
* P/PH: fixed-MU / fixed-QV real.exe hydrostatic balance (module_initialize_real.F :3923-4036, hypsometric_opt 2, ALL
  moist species in the loading) on the NEW levels + the old non-hydrostatic residual (P − B_old, PH − B_old)
  interpolated → the identity remap reproduces P/PH to rounding. AL = start_em form from PH; P_HYD = P + PB.
* wrfbdy: every 3-D bdy value and tendency, record by record, in WRF's coupled space: decouple with the old
  (c1·μ_d + c2) of WRF ``couple`` (u/v: c1h with face μ; t/moist/scalars: c1h; ph/w: c1f), interpolate, recouple
  with the new coefficients (the column map factor cancels). Tendency = (remap(B + Δt·BT) − remap(B)) / Δt with the
  record's μ and μ + Δt·MU_BT. Boundary PH gets the same hydrostatic treatment as the interior.
* Every variable without a vertical dimension is copied byte-identically. Host NumPy; float64 inside, file dtype out.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gpuwrf.init.real_init.base_state import compute_base_state
from gpuwrf.init.real_init.types import CPOVCV, CVPM, P1000MB, R_D, RVOVRD, T0, RealInitConfig
from gpuwrf.init.real_init.vertical_coord import compute_vertical_coord, wrf_fp32_c1_from_c3
from gpuwrf.io.netcdf_lock import Dataset

G = 9.81
MOIST = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
COORD_1D = ("ZNU", "ZNW", "DNW", "RDNW", "DN", "RDN", "FNP", "FNM", "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F")
COORD_0D = ("CF1", "CF2", "CF3", "CFN", "CFN1")
BASE_FIELDS = ("PB", "PHB", "ALB", "T_INIT")
DERIVED = ("P", "PH", "AL", "P_HYD")
VERT_DIMS = ("bottom_top", "bottom_top_stag")
BDY_COUPLING = {"U": "u", "V": "v", "W": "f", "PH": "f"}  # default: 't' (c1h/c2h with mass μ)
SIDES = ("BXS", "BXE", "BYS", "BYE")


# --------------------------------------------------------------------------- eta set

def refine_eta(znw, z_full, *, z_lo: float = 400.0, z_hi: float = 2500.0, dz_max: float = 75.0) -> tuple[np.ndarray, dict]:
    """Insert levels into every layer that overlaps (z_lo, z_hi) so that dz ≤ dz_max there (equal η sub-steps).

    ``z_full``: representative full-level heights AGL (m) of the original levels (e.g. ocean mean). Original levels,
    η = 1, η = 0 are kept bitwise; bottom and top layers are never split."""
    znw = np.asarray(znw, np.float64)
    z = np.asarray(z_full, np.float64)
    out, splits = [znw[0]], []
    for k in range(len(znw) - 1):
        dz = z[k + 1] - z[k]
        n = 1
        if 0 < k < len(znw) - 2 and z[k + 1] > z_lo and z[k] < z_hi:
            n = max(1, math.ceil(dz / dz_max))
        for s in range(1, n):
            out.append(znw[k] + (znw[k + 1] - znw[k]) * s / n)
        out.append(znw[k + 1])
        splits.append(n)
    new = np.asarray(out, np.float64)
    zn = np.interp(-new, -znw, z)  # approximate heights of the new levels (piecewise linear in η)
    band = (zn[1:] > z_lo) & (zn[:-1] < z_hi)
    info = {"e_vert_old": int(len(znw)), "e_vert_new": int(len(new)), "z_lo": z_lo, "z_hi": z_hi, "dz_max": dz_max,
            "splits_per_layer": splits, "inserted": int(len(new) - len(znw)),
            "approx_new_full_level_heights_m": [round(float(v), 1) for v in zn],
            "max_dz_in_band_m": float(np.diff(zn)[band].max()) if band.any() else None,
            "levels_in_band_old": int(((z > z_lo) & (z < z_hi)).sum()), "levels_in_band_new": int(((zn > z_lo) & (zn < z_hi)).sum())}
    return new, info


def vertical_coord(znw, p_top: float, etac: float, hybrid_opt: int = 2, reference: dict | None = None) -> dict:
    """real.exe 1-D coordinate arrays for ``znw`` as float32 (C1 via the fp32 finite-difference recipe).

    ``reference`` = the original file's 1-D arrays: at every level whose (float32) η exists there, C3/C4 are taken
    from the file (real.exe's gfortran fp32 rounding of the Klemp polynomial, not portably reproducible); inserted
    levels get the polynomial value. C1/C2 then follow from the fp32 recipe, so an identity set is bit-exact."""
    znw = np.asarray(znw, np.float64)
    cfg = RealInitConfig(nz=len(znw) - 1, p_top_pa=float(p_top), hybrid_opt=int(hybrid_opt), etac=float(etac),
                         eta_levels=tuple(float(v) for v in znw))
    vc = compute_vertical_coord(cfg)
    f = np.float32
    out = {"ZNW": vc.znw, "ZNU": vc.znu, "DNW": vc.dnw, "RDNW": vc.rdnw, "DN": vc.dn, "RDN": vc.rdn, "FNP": vc.fnp,
           "FNM": vc.fnm, "C3F": vc.c3f, "C4F": vc.c4f, "C3H": vc.c3h, "C4H": vc.c4h}
    out = {k: np.asarray(v, np.float64).astype(f) for k, v in out.items()}
    if reference is not None:
        for eta, names in (("ZNW", ("C3F", "C4F")), ("ZNU", ("C3H", "C4H"))):
            ref_eta = np.asarray(reference[eta], f)
            lookup = {float(v): j for j, v in enumerate(ref_eta)}
            for i, v in enumerate(out[eta]):
                j = lookup.get(float(v))
                if j is not None:
                    for n in names:
                        out[n][i] = np.asarray(reference[n], f)[j]
    c1f, c1h = wrf_fp32_c1_from_c3(out["C3F"], out["C3H"], out["ZNW"], out["ZNU"], hybrid_opt)
    out["C1F"], out["C1H"] = c1f, c1h
    out["C2F"] = ((f(1.0) - c1f) * f(P1000MB - p_top)).astype(f)
    out["C2H"] = ((f(1.0) - c1h) * f(P1000MB - p_top)).astype(f)
    for k, v in (("CF1", vc.cf1), ("CF2", vc.cf2), ("CF3", vc.cf3), ("CFN", vc.cfn), ("CFN1", vc.cfn1)):
        out[k] = f(v)
    out["_cfg"] = cfg
    return out


BASE_CONSTS = {"T00": "base_temp", "P00": "base_pres", "TLP": "base_lapse", "TISO": "iso_temp",
               "P_STRAT": "base_pres_strat", "TLP_STRAT": "base_lapse_strat"}


def base_consts(ds) -> dict:
    """real.exe base-state constants as written to the wrfinput scalars (WRF start_em checks them vs the namelist)."""
    return {cfg: float(np.asarray(ds[name][0]).reshape(-1)[0]) for name, cfg in BASE_CONSTS.items()}


def base_state(vc: dict, hgt, consts: dict) -> dict:
    """PB/ALB/T_INIT/MUB/PHB (float64) for the coordinate ``vc`` (file-dtype coefficients) — compute_base_state."""
    from dataclasses import replace
    from types import SimpleNamespace
    cfg = replace(vc["_cfg"], **consts)
    v64 = SimpleNamespace(**{k.lower(): np.asarray(vc[k], np.float64) for k in COORD_1D})
    b = compute_base_state(cfg, v64, hgt)
    return {"PB": b.pb, "ALB": b.alb, "T_INIT": b.t_init, "MUB": b.mub, "PHB": b.phb}


# --------------------------------------------------------------------------- hydrostatic balance (real.exe)

def _c(vc, name, nd):
    return np.asarray(vc[name], np.float64).reshape((-1,) + (1,) * nd)


def hydrostatic(theta_m_pert, qv, qtot, mu, mub, pb, alb, phb, vc, p_top: float) -> dict:
    """real.exe :3923-4036 (hypsometric_opt 2): P, PH, AL for fixed μ′ and moisture; arrays (nz[,…]) float64.

    ``theta_m_pert`` is the θ entering the EOS times (1+Rv/Rd·qv) (i.e. θ_m − T0); ``qtot`` = Σ moist species
    (loading), ``qv`` only sets nothing else here because the EOS uses θ_m directly."""
    th = np.asarray(theta_m_pert, np.float64)
    qt = np.asarray(qtot, np.float64)
    mu, mub = np.asarray(mu, np.float64), np.asarray(mub, np.float64)
    pb, alb, phb = (np.asarray(a, np.float64) for a in (pb, alb, phb))
    nd = th.ndim - 1
    nz = th.shape[0]
    c1f, c2f = _c(vc, "C1F", nd), _c(vc, "C2F", nd)
    rdnw, rdn = _c(vc, "RDNW", nd), _c(vc, "RDN", nd)
    c3f, c4f, c3h, c4h = _c(vc, "C3F", nd), _c(vc, "C4F", nd), _c(vc, "C3H", nd), _c(vc, "C4H", nd)
    p = np.empty_like(th)
    alt = np.empty_like(th)
    q = qt[nz - 1]
    qvf2, qvf1 = 1.0 / (1.0 + q), q / (1.0 + q)
    p[nz - 1] = -0.5 * ((c1f[nz] * mu) + qvf1 * (c1f[nz] * mub + c2f[nz])) / rdnw[nz - 1] / qvf2
    for kk in range(nz - 2, -1, -1):
        q = 0.5 * (qt[kk] + qt[kk + 1])
        qvf2, qvf1 = 1.0 / (1.0 + q), q / (1.0 + q)
        p[kk] = p[kk + 1] - ((c1f[kk + 1] * mu) + qvf1 * (c1f[kk + 1] * mub + c2f[kk + 1])) / qvf2 / rdn[kk + 1]
    alt = (R_D / P1000MB) * (th + T0) * (((p + pb) / P1000MB) ** CVPM)
    mu0 = mub + mu
    pfu = c3f[1:] * mu0 + c4f[1:] + p_top
    pfd = c3f[:-1] * mu0 + c4f[:-1] + p_top
    phm = c3h * mu0 + c4h + p_top
    ph_tot = np.concatenate((phb[:1], phb[:1] + np.cumsum(alt * phm * np.log(pfd / pfu), axis=0)), axis=0)
    ph = ph_tot - phb
    al = (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]) / phm / np.log(pfd / pfu) - alb
    p = P1000MB * ((R_D * (T0 + th)) / (P1000MB * (al + alb))) ** CPOVCV - pb
    return {"P": p, "PH": ph, "AL": al}


# --------------------------------------------------------------------------- interpolation

def _brackets(eta_old, eta_new) -> np.ndarray:
    """Index k with eta_old[k] ≥ eta_new ≥ eta_old[k+1] (η decreasing); k ≤ len−2."""
    eo, en = np.asarray(eta_old, np.float64), np.asarray(eta_new, np.float64)
    tol = 1e-6
    if en.max() > eo[0] + tol or en.min() < eo[-1] - tol:
        raise ValueError("new levels outside the old range: no extrapolation allowed")
    k = np.searchsorted(-eo, -en, side="right") - 1
    return np.clip(k, 0, len(eo) - 2)


def lnp_dry(vc, level: str, mu_d, p_top: float):
    nd = np.ndim(mu_d)
    c3, c4 = (_c(vc, "C3H", nd), _c(vc, "C4H", nd)) if level == "h" else (_c(vc, "C3F", nd), _c(vc, "C4F", nd))
    return np.log(c3 * np.asarray(mu_d, np.float64) + c4 + p_top)


def interp_vertical(x, lnp_old, lnp_new, k) -> np.ndarray:
    """Linear in ln p_d, column-wise; ``k`` = shared η brackets (len nz_new)."""
    x = np.asarray(x, np.float64)
    lo, hi = lnp_old[k], lnp_old[k + 1]
    w = (lnp_new - lo) / (hi - lo)
    return (1.0 - w) * x[k] + w * x[k + 1]


def face_mu(mu_d, axis: int) -> np.ndarray:
    """WRF calc_mu_staggered along ``axis`` (−1: x faces, −2: y faces): interior means, edge copies."""
    mu = np.moveaxis(np.asarray(mu_d, np.float64), axis, -1)
    out = np.concatenate((mu[..., :1], 0.5 * (mu[..., 1:] + mu[..., :-1]), mu[..., -1:]), axis=-1)
    return np.moveaxis(out, -1, axis)


class Remapper:
    """Old/new coordinates of one case; ``remap(x, level, mu_d)`` for any (nz, …) column array."""

    def __init__(self, vc_old: dict, vc_new: dict, p_top: float):
        self.old, self.new, self.p_top = vc_old, vc_new, float(p_top)
        self.k_h = _brackets(vc_old["ZNU"], vc_new["ZNU"])
        self.k_f = _brackets(vc_old["ZNW"], vc_new["ZNW"])

    def remap(self, x, level: str, mu_d):
        lo, ln = lnp_dry(self.old, level, mu_d, self.p_top), lnp_dry(self.new, level, mu_d, self.p_top)
        return interp_vertical(x, lo, ln, self.k_h if level == "h" else self.k_f)

    def remap_1d(self, x, level: str):
        eo = np.asarray(self.old["ZNU" if level == "h" else "ZNW"], np.float64)
        en = np.asarray(self.new["ZNU" if level == "h" else "ZNW"], np.float64)
        return np.interp(-en, -eo, np.asarray(x, np.float64))


def _moist_total(get, names):
    tot = 0.0
    for n in MOIST:
        if n in names:
            tot = tot + np.asarray(get(n), np.float64)
    return tot


# --------------------------------------------------------------------------- netCDF writing

def _copy_var_def(src_var, dst, dims, name=None):
    filt = src_var.filters() or {}
    kw = {}
    if filt.get("zlib"):
        kw.update(zlib=True, complevel=filt.get("complevel", 4), shuffle=filt.get("shuffle", False))
    fill = getattr(src_var, "_FillValue", None)
    var = dst.createVariable(name or src_var.name, src_var.dtype, dims, fill_value=fill, **kw)
    var.setncatts({k: src_var.getncattr(k) for k in src_var.ncattrs() if k != "_FillValue"})
    var.set_auto_mask(False)
    return var


def _raw(var):
    var.set_auto_mask(False)
    return np.array(var[:], copy=True)


def remap_wrfinput(src_path, out_path, znw_new, *, extra_attrs=None) -> dict:
    """Write the remapped wrfinput (new file). Returns a report with per-variable treatment."""
    src_path, out_path = Path(src_path), Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} exists")
    with Dataset(str(src_path)) as src:
        src.set_auto_mask(False)
        p_top = float(np.asarray(src["P_TOP"][0]).reshape(-1)[0])
        etac, hyb = float(src.getncattr("ETAC")), int(src.getncattr("HYBRID_OPT"))
        if int(src.getncattr("HYPSOMETRIC_OPT")) != 2 or int(src.getncattr("USE_THETA_M")) != 1:
            raise ValueError("vremap supports hypsometric_opt 2 and use_theta_m 1 (WN3)")
        znw_old = _raw(src["ZNW"])[0]
        vc_old = {n: _raw(src[n])[0] for n in COORD_1D}
        for n in COORD_0D:
            vc_old[n] = _raw(src[n])[0]
        vc_new = vertical_coord(znw_new, p_top, etac, hyb, reference=vc_old)
        rm = Remapper(vc_old, vc_new, p_top)
        hgt = _raw(src["HGT"])[0]
        mub = _raw(src["MUB"])[0].astype(np.float64)
        mu = _raw(src["MU"])[0].astype(np.float64)
        mu_d = mub + mu
        base_old = {n: _raw(src[n])[0].astype(np.float64) for n in BASE_FIELDS}
        consts = base_consts(src)
        base_new = base_state(vc_new, hgt, consts)
        get = lambda n: _raw(src[n])[0]  # noqa: E731
        names = set(src.variables)
        qtot_o = _moist_total(get, names)
        thm_o = get("THM").astype(np.float64)
        hyd_o = hydrostatic(thm_o, get("QVAPOR"), qtot_o, mu, mub, base_old["PB"], base_old["ALB"], base_old["PHB"],
                            vc_old, p_top)
        res_p = get("P").astype(np.float64) - hyd_o["P"]
        res_ph = get("PH").astype(np.float64) - hyd_o["PH"]
        new = {}
        treat = {}
        for name in src.variables:
            dims = src[name].dimensions
            if not any(d in VERT_DIMS for d in dims):
                continue
            if name in COORD_1D:
                new[name] = vc_new[name][None]
                treat[name] = "real.exe 1-D coordinate (new eta set)"
                continue
            if name in BASE_FIELDS:
                continue
            if name in DERIVED:
                continue
            x = get(name)
            level = "f" if "bottom_top_stag" in dims else "h"
            if x.ndim == 1:  # *_BASE ideal profiles
                new[name] = rm.remap_1d(x, level).astype(x.dtype)[None]
                treat[name] = "1-D, linear in eta"
                continue
            if "west_east_stag" in dims:
                m = face_mu(mu_d, -1)
            elif "south_north_stag" in dims:
                m = face_mu(mu_d, -2)
            else:
                m = mu_d
            new[name] = rm.remap(x, level, m).astype(x.dtype)[None]
            treat[name] = f"linear in ln p_d ({'full' if level == 'f' else 'mass'} levels" + (
                ", face mu)" if m is not mu_d else ")")
        qtot_n = _moist_total(lambda n: new[n][0], names)
        thm_n = new["THM"][0].astype(np.float64)
        hyd_n = hydrostatic(thm_n, new["QVAPOR"][0], qtot_n, mu, mub, base_new["PB"], base_new["ALB"], base_new["PHB"],
                            vc_new, p_top)
        p_n = hyd_n["P"] + rm.remap(res_p, "h", mu_d)
        ph_n = hyd_n["PH"] + rm.remap(res_ph, "f", mu_d)
        ph_n[0] = get("PH")[0]  # surface geopotential perturbation unchanged
        mut = mub + mu
        c3f, c4f, c3h, c4h = (_c(vc_new, n, 2) for n in ("C3F", "C4F", "C3H", "C4H"))
        pfu = c3f[1:] * mut + c4f[1:] + p_top
        pfd = c3f[:-1] * mut + c4f[:-1] + p_top
        phm = c3h * mut + c4h + p_top
        al_n = (ph_n[1:] - ph_n[:-1] + base_new["PHB"][1:] - base_new["PHB"][:-1]) / phm / np.log(pfd / pfu) - base_new["ALB"]
        dt = {n: src[n].dtype for n in DERIVED + BASE_FIELDS}
        # P_HYD: keep the frame's (P_HYD − P − PB) offset (wrfout-derived inputs: hydrostatic diag ≠ P + PB)
        hyd_off = get("P_HYD").astype(np.float64) - get("P").astype(np.float64) - base_old["PB"]
        p_hyd_n = p_n + base_new["PB"] + rm.remap(hyd_off, "h", mu_d)
        new.update({"P": p_n.astype(dt["P"])[None], "PH": ph_n.astype(dt["PH"])[None], "AL": al_n.astype(dt["AL"])[None],
                    "P_HYD": p_hyd_n.astype(dt["P_HYD"])[None]})
        for n in BASE_FIELDS:
            new[n] = base_new[n].astype(dt[n])[None]
        treat.update({"P": "real.exe hydrostatic (fixed MU, all moist species) on new levels + interpolated old residual",
                      "PH": "same as P (surface PH kept)", "AL": "start_em form from new PH/MU",
                      "P_HYD": "P + PB + interpolated old (P_HYD − P − PB)",
                      **{n: "base state recomputed (compute_base_state)" for n in BASE_FIELDS}})
        mub_new = base_new["MUB"]
        nz_new = len(znw_new) - 1
        # ---- write
        with Dataset(str(out_path), "w", format=src.data_model) as dst:
            for dname, dim in src.dimensions.items():
                size = None if dim.isunlimited() else (nz_new if dname == "bottom_top" else
                                                     nz_new + 1 if dname == "bottom_top_stag" else len(dim))
                dst.createDimension(dname, size)
            attrs = {k: src.getncattr(k) for k in src.ncattrs()}
            attrs["BOTTOM-TOP_GRID_DIMENSION"] = np.int32(nz_new + 1)
            dst.setncatts(attrs)
            if extra_attrs:
                dst.setncatts(dict(extra_attrs))
            copied = []
            for name, var in src.variables.items():
                out = _copy_var_def(var, dst, var.dimensions)
                if name in new:
                    out[:] = new[name]
                elif name in COORD_0D:
                    out[:] = np.asarray(vc_new[name], var.dtype).reshape(var.shape)
                    treat[name] = "real.exe extrapolation coefficient (new eta set)"
                else:
                    out[:] = _raw(var)
                    copied.append(name)
    return {"e_vert_old": int(len(znw_old)), "e_vert_new": int(nz_new + 1), "p_top": p_top, "etac": etac,
            "base_consts": consts,
            "treatment": treat, "copied_byte_identical": sorted(copied),
            "mub_recomputed_minus_file_max_abs": float(np.abs(mub_new - mub).max()),
            "hydrostatic_residual_old_max_abs": {"P": float(np.abs(res_p).max()), "PH": float(np.abs(res_ph).max())}}


# --------------------------------------------------------------------------- wrfbdy

def _strip_index(side: str, arr2d: np.ndarray, width: int, stag: str | None) -> np.ndarray:
    """Values of a full 2-D (ny, nx) field arranged like a WRF bdy strip (width, n) for ``side``."""
    a = np.asarray(arr2d, np.float64)
    if side == "BXS":
        s = a[:, :width].T
    elif side == "BXE":
        s = a[:, ::-1][:, :width].T
    elif side == "BYS":
        s = a[:width, :]
    else:
        s = a[::-1, :][:width, :]
    return s


def _strip_mu(side, mub2d, mu_strip, width, base: str = ""):
    """Total dry mass at the points of a bdy array (width, n) for ``side``: mass points, or WRF calc_mu_staggered
    face mass for U (x faces) / V (y faces). Faces NORMAL to the strip (U on X sides, V on Y sides): face w lies
    between mass w−1 and w counted from the boundary (edge copy at w = 0 = domain edge); faces ALONG the strip:
    interior means with edge copies."""
    mut = _strip_index(side, mub2d, width, None) + np.asarray(mu_strip, np.float64)  # (width, n_mass)
    normal = (base == "U" and side in ("BXS", "BXE")) or (base == "V" and side in ("BYS", "BYE"))
    along = (base == "U" and side in ("BYS", "BYE")) or (base == "V" and side in ("BXS", "BXE"))
    if normal:
        f = np.empty_like(mut)
        f[0] = mut[0]
        f[1:] = 0.5 * (mut[1:] + mut[:-1])
        return f
    if along:
        return face_mu(mut, -1)
    return mut


def _coupling(vc, kind, mu_d):
    nd = np.ndim(mu_d)
    if kind == "f":
        c1, c2 = _c(vc, "C1F", nd), _c(vc, "C2F", nd)
    else:
        c1, c2 = _c(vc, "C1H", nd), _c(vc, "C2H", nd)
    return c1 * np.asarray(mu_d, np.float64) + c2


def remap_wrfbdy(src_path, out_path, wrfinput_src, znw_new, *, extra_attrs=None) -> dict:
    """Remap every 3-D bdy value/tendency record by record in coupled space; byte-copy the rest."""
    src_path, out_path = Path(src_path), Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} exists")
    with Dataset(str(wrfinput_src)) as wi:
        wi.set_auto_mask(False)
        p_top = float(np.asarray(wi["P_TOP"][0]).reshape(-1)[0])
        etac, hyb = float(wi.getncattr("ETAC")), int(wi.getncattr("HYBRID_OPT"))
        vc_old = {n: _raw(wi[n])[0] for n in COORD_1D}
        mub2d = _raw(wi["MUB"])[0].astype(np.float64)
        hgt = _raw(wi["HGT"])[0]
        base_old2d = {n: _raw(wi[n])[0].astype(np.float64) for n in BASE_FIELDS}
        consts = base_consts(wi)
    vc_new = vertical_coord(znw_new, p_top, etac, hyb, reference=vc_old)
    base_new2d = base_state(vc_new, hgt, consts)
    rm = Remapper(vc_old, vc_new, p_top)
    nz_new = len(znw_new) - 1
    report = {"remapped": [], "copied_byte_identical": [], "records": None}
    with Dataset(str(src_path)) as src:
        src.set_auto_mask(False)
        width = len(src.dimensions["bdy_width"])
        times = [bytes(np.asarray(t)).decode() for t in src["Times"][:]]
        stamps = [datetime.strptime(t, "%Y-%m-%d_%H:%M:%S") for t in times]
        dts = [(stamps[i + 1] - stamps[i]).total_seconds() for i in range(len(stamps) - 1)]
        interval = float(dts[0]) if dts else float(src.getncattr("DT"))
        if any(abs(d - interval) > 1e-6 for d in dts):
            raise ValueError("irregular wrfbdy record interval")
        report["records"], report["interval_s"] = times, interval
        mu_b = {s: _raw(src[f"MU_B{s[1:]}"]) .astype(np.float64) for s in SIDES}
        mu_bt = {s: _raw(src[f"MU_BT{s[1:]}"]).astype(np.float64) for s in SIDES}
        names = set(src.variables)
        new = {}
        bases = sorted({v.rsplit("_B", 1)[0] for v in names if any(d in VERT_DIMS for d in src[v].dimensions)})
        for side in SIDES:
            for rec in range(len(times)):
                for phase in ("now", "next"):
                    mu_s = mu_b[side][rec] + (interval * mu_bt[side][rec] if phase == "next" else 0.0)
                    vals = {}
                    for base in bases:
                        vb, vt = f"{base}_B{side[1:]}", f"{base}_BT{side[1:]}"
                        b = _raw(src[vb])[rec].astype(np.float64)  # (width, nz, n)
                        if phase == "next":
                            b = b + interval * _raw(src[vt])[rec].astype(np.float64)
                        vals[base] = np.moveaxis(b, 1, 0)  # (nz, width, n)
                    out = _remap_strip(vals, side, width, mu_s, mub2d, base_old2d, base_new2d, vc_old, vc_new, rm,
                                       p_top, {b: src[f"{b}_B{side[1:]}"].dimensions for b in bases})
                    for base, arr in out.items():
                        new.setdefault((base, side, phase), [None] * len(times))[rec] = np.moveaxis(arr, 0, 1)
        with Dataset(str(out_path), "w", format=src.data_model) as dst:
            for dname, dim in src.dimensions.items():
                size = None if dim.isunlimited() else (nz_new if dname == "bottom_top" else
                                                     nz_new + 1 if dname == "bottom_top_stag" else len(dim))
                dst.createDimension(dname, size)
            attrs = {k: src.getncattr(k) for k in src.ncattrs()}
            if "BOTTOM-TOP_GRID_DIMENSION" in attrs:
                attrs["BOTTOM-TOP_GRID_DIMENSION"] = np.int32(nz_new + 1)
            dst.setncatts(attrs)
            if extra_attrs:
                dst.setncatts(dict(extra_attrs))
            for name, var in src.variables.items():
                out = _copy_var_def(var, dst, var.dimensions)
                if any(d in VERT_DIMS for d in var.dimensions):
                    base, tag = name.rsplit("_B", 1)  # tag: XS | TXS …
                    side = "B" + tag[-2:]
                    now = np.stack(new[(base, side, "now")])
                    if tag.startswith("T"):
                        nxt = np.stack(new[(base, side, "next")])
                        out[:] = ((nxt - now) / interval).astype(var.dtype)
                    else:
                        out[:] = now.astype(var.dtype)
                    report["remapped"].append(name)
                else:
                    out[:] = _raw(var)
                    report["copied_byte_identical"].append(name)
    report["e_vert_new"] = int(nz_new + 1)
    return report


def _remap_strip(vals, side, width, mu_strip, mub2d, base_old2d, base_new2d, vc_old, vc_new, rm, p_top, dims):
    """Remap one side/record (coupled values (nz, width, n)); PH via the hydrostatic route like the interior."""
    out = {}
    decoupled = {}
    mu_mass = _strip_mu(side, mub2d, mu_strip, width)
    for base, b in vals.items():
        kind = BDY_COUPLING.get(base, "t")
        m = _strip_mu(side, mub2d, mu_strip, width, base) if base in ("U", "V") else mu_mass
        if m.shape != b.shape[1:]:
            raise ValueError(f"{base}_{side}: mass/face shape {m.shape} != array {b.shape[1:]}")
        level = "f" if kind == "f" else "h"
        x = b / _coupling(vc_old, kind, m)
        decoupled[base] = (x, m, level, kind)
    # interior-consistent PH: hydrostatic on new levels + interpolated old residual (θ_m = T_B decoupled, use_theta_m 1)
    mub_s = _strip_index(side, mub2d, width, None)
    mu_p = mu_mass - mub_s
    strip = {n: np.stack([_strip_index(side, base_old2d[n][k], width, None) for k in range(base_old2d[n].shape[0])])
             for n in BASE_FIELDS}
    strip_new = {n: np.stack([_strip_index(side, base_new2d[n][k], width, None) for k in range(base_new2d[n].shape[0])])
                 for n in BASE_FIELDS}
    have = all(n in decoupled for n in ("T", "QVAPOR", "PH"))
    for base, (x, m, level, kind) in decoupled.items():
        xn = rm.remap(x, level, m)
        if base == "PH" and have:
            thm_o, qv_o = decoupled["T"][0], decoupled["QVAPOR"][0]
            qt_o = sum(decoupled[n][0] for n in MOIST if n in decoupled)
            hyd_o = hydrostatic(thm_o, qv_o, qt_o, mu_p, mub_s, strip["PB"], strip["ALB"], strip["PHB"], vc_old, p_top)
            thm_n = rm.remap(thm_o, "h", mu_mass)
            qt_n = sum(rm.remap(decoupled[n][0], "h", mu_mass) for n in MOIST if n in decoupled)
            hyd_n = hydrostatic(thm_n, None, qt_n, mu_p, mub_s, strip_new["PB"], strip_new["ALB"], strip_new["PHB"],
                                vc_new, p_top)
            xn = hyd_n["PH"] + rm.remap(x - hyd_o["PH"], "f", m)
            xn[0] = x[0]
        out[base] = xn * _coupling(vc_new, kind, m)
    return out


# --------------------------------------------------------------------------- case

def _edit_namelist(text: str, e_vert_new: int, znw_new, time_step: int | None) -> str:
    import re
    lines, out, in_domains, inserted = text.splitlines(), [], False, False
    eta = ", ".join(f"{v:.6f}" for v in znw_new)
    for line in lines:
        s = line.strip().lower()
        if s.startswith("&domains"):
            in_domains = True
        if in_domains and re.match(r"\s*e_vert\s*=", line, re.I):
            n = len([t for t in line.split("=", 1)[1].split(",") if t.strip()])
            line = f" e_vert                              = {', '.join([str(e_vert_new)] * max(n, 1))},"
        if in_domains and re.match(r"\s*eta_levels\s*=", line, re.I):
            continue  # replaced below
        if time_step is not None and in_domains and re.match(r"\s*time_step\s*=", line, re.I):
            line = f" time_step                           = {time_step},"
        if in_domains and s.startswith("/") and not inserted:
            out.append(f" eta_levels                          = {eta},")
            inserted = True
            in_domains = False
        out.append(line)
    return "\n".join(out) + "\n"


def remap_case(case_dir, out_dir, znw_new, *, time_step: int | None = None, provenance=None) -> dict:
    """New case dir: remapped wrfinput_d01 + wrfbdy_d01, wrflowinp_d01 byte copy, namelist (e_vert, eta_levels,
    optional time_step), static symlinks recreated as symlinks (targets not opened)."""
    import shutil
    case_dir, out_dir = Path(case_dir), Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"{out_dir} exists")
    tmp = out_dir.parent / f".{out_dir.name}.building"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    attrs = {"GPUWRF_VREMAP": json.dumps({"source_case": str(case_dir), "e_vert_new": len(znw_new),
                                          "created_utc": datetime.now(timezone.utc).isoformat(),
                                          **dict(provenance or {})}, sort_keys=True)}
    rep_in = remap_wrfinput(case_dir / "wrfinput_d01", tmp / "wrfinput_d01", znw_new, extra_attrs=attrs)
    rep_bdy = remap_wrfbdy(case_dir / "wrfbdy_d01", tmp / "wrfbdy_d01", case_dir / "wrfinput_d01", znw_new,
                           extra_attrs=attrs)
    for p in sorted(case_dir.iterdir()):
        if p.name in ("wrfinput_d01", "wrfbdy_d01", "namelist.input"):
            continue
        if p.is_symlink():
            os.symlink(os.readlink(p), tmp / p.name)
        elif p.name == "wrflowinp_d01":
            shutil.copyfile(p, tmp / p.name)
        elif p.name == "cycle_case_manifest.json":
            shutil.copyfile(p, tmp / "source_cycle_case_manifest.json")
    (tmp / "namelist.input").write_text(_edit_namelist((case_dir / "namelist.input").read_text(),
                                                       len(znw_new), znw_new, time_step))
    os.rename(tmp, out_dir)
    return {"case": str(out_dir), "wrfinput": rep_in, "wrfbdy": rep_bdy}
