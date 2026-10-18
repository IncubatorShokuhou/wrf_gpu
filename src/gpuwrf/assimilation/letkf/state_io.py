"""LETKF state IO: member frames, incremental hydrostatic rebalance, boundary taper, analysis wrfinput.

Lane a1-io (DESIGN §3.1, §6.1 steps 2/6/7, §13). Host NumPy only: no JAX and no
``gpuwrf.runtime/integration/dynamics/physics/kernels`` import, so the analysis
stays an offline step between forecasts.

Conventions (pristine WRF 4.7.1, checked on the CPU-WRF WN3 0502 files):

* wrfout and wrfinput ``T`` = DRY theta - 300 K under both ``use_theta_m`` values; ``THM`` = the
  prognostic (moist theta_m - 300 for ``use_theta_m=1``; real.exe :4918-4928 converts at the very
  end, Registry ``th_phy_m_t0`` -> "T", ``t`` -> "THM"). wrfout ``P_HYD`` != ``P + PB`` (model
  non-hydrostatic p); wrfinput ``P_HYD`` == ``P + PB``. ``AL``/``ALB``/``T_INIT`` exist only in wrfinput.
* WRF ``start_em`` (non-restart, :794-842) re-derives AL from PH/MU (hypsometric form) and P from
  AL, ``t_1`` = THM and qv, so PH, MU, THM/T and QVAPOR are the fields that set the warm start.

Fixed-QV / fixed-MU incremental rebalance (DESIGN_v2 §3.1; ``rebalance_increment``). MU is frozen,
so the dry hydrostatic pressure and the hypsometric coefficients (phm, ln(pfd/pfu)) are unchanged:

  Δp′  real.exe top-down moisture-loading recursion (:3923-3953) applied to Δqv. That recursion is
       LINEAR in the summed moisture (c1f·μ′ + q̄·(c1f·μ_d + c2f)), so with hydrometeors frozen the
       qv-only increment is exact (real.exe sums all ``moist`` species; their loading cancels);
  Δα   EOS α = (R_d/p0)·(θ+T0)·(1+R_v/R_d·qv)·((p′+pb)/p0)^cvpm (real.exe :3937/:3955) at the ACTUAL
       background p′ (model wrfout P) and at p′+Δp′ with the analysed θ, qv;
  ΔPH  integrated from Δα in the model's hypsometric form (HYPSOMETRIC_OPT 2: α·phm·ln(pfd/pfu),
       real.exe :3976-3983 / start_em :629-637; opt 1: −dnw·(c1h·μ_d+c2h)·α), ΔPH(surface) = 0;
  ΔAL  = Δα (ALB fixed); P_HYD gets Δp′.

So the discrete hydrostatic residual of the written (P, PH, T, QV) changes by round-off only
(``hydrostatic_residual``), and a zero increment returns the background bit-for-bit (Δq = 0,
identical EOS inputs). ``real_init.hydrostatic.balance`` is NOT used by the adapter (it re-derives
QV from RH with caps/floors); :func:`balance_state` keeps it as an independent cross-check oracle.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

import numpy as np

from gpuwrf.init.real_init.hydrostatic import balance
from gpuwrf.init.real_init.types import (
    BaseStateColumns,
    CP,
    CPOVCV,
    CVPM,
    DynamicsInit,
    P1000MB,
    R_D,
    RVOVRD,
    T0,
    RealInitConfig,
    VerticalCoord1D,
)
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_theta_convention import parse_use_theta_m, thm_source

ANALYSIS_VARIABLES = ("U", "V", "T", "QVAPOR", "MU")
HX_VARIABLES = ("P", "PB", "PH", "PHB", "T2", "Q2", "U10", "V10", "PSFC", "CLDFRA")
FRAME_VARIABLES = ANALYSIS_VARIABLES + HX_VARIABLES
PROVENANCE_ATTRIBUTE = "GPUWRF_LETKF_ANALYSIS"
WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

_VCOORD_1D = ("ZNW", "ZNU", "DNW", "RDNW", "DN", "RDN", "FNP", "FNM",
              "C1F", "C2F", "C3F", "C4F", "C1H", "C2H", "C3H", "C4H")
_SCALARS = ("P_TOP", "CF1", "CF2", "CF3", "CFN", "CFN1", "P00", "T00", "TLP", "TISO", "P_STRAT", "TLP_STRAT")
_REQUIRED_2D = ("XLAT", "XLONG", "HGT", "MUB")
_OPTIONAL_2D = ("COSALPHA", "SINALPHA", "LANDMASK", "XLAND", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY",
                "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V")
_BASE_3D = ("PB", "PHB")
_ATTRS_REQUIRED = ("DX", "DY", "HYBRID_OPT", "HYPSOMETRIC_OPT", "USE_THETA_M")
_ATTRS_OPTIONAL = ("MAP_PROJ", "GRID_ID", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON")
GRID_IDENTITY = ("XLAT", "XLONG", "HGT", "ZNU", "ZNW")


# --------------------------------------------------------------------------- time / files

def wrf_time(value) -> str:
    """Normalise a datetime or a WRF/ISO-8601 string (``Z`` or ``±hh:mm`` offsets → UTC) to WRF ``Times`` form."""
    if not isinstance(value, datetime):
        text = str(value).strip()
        try:
            return datetime.strptime(text, WRF_TIME_FORMAT).strftime(WRF_TIME_FORMAT)
        except ValueError:
            value = datetime.fromisoformat(text.replace("_", "T", 1).replace("Z", "+00:00"))
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime(WRF_TIME_FORMAT)


def parse_wrf_time(value) -> datetime:
    return datetime.strptime(wrf_time(value), WRF_TIME_FORMAT)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_times(ds) -> list[str]:
    """``Times`` rows of an open dataset (never inferred from the filename)."""
    if "Times" not in ds.variables:
        raise ValueError("Times is required; timestamps are never inferred from filenames")
    rows = []
    for row in np.asarray(ds["Times"][:]):
        rows.append(b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").strip("\0 "))
    return [wrf_time(r) for r in rows]


def _time_index(ds, valid_time, tolerance_s: float = 0.0) -> tuple[int, str]:
    times = frame_times(ds)
    want = parse_wrf_time(valid_time)
    hits = [i for i, t in enumerate(times) if abs((parse_wrf_time(t) - want).total_seconds()) <= tolerance_s]
    if len(hits) != 1:
        raise ValueError(f"need exactly one frame within {tolerance_s:g} s of {wrf_time(want)}; file has {times}")
    return hits[0], times[hits[0]]


def _raw(var, index=None) -> np.ndarray:
    var.set_auto_mask(False)
    data = var[index] if index is not None else var[:]
    return np.array(data, copy=True)


def _frame_value(ds, name: str, index: int) -> np.ndarray:
    var = ds[name]
    if var.dimensions[:1] == ("Time",):
        return _raw(var, index)
    return _raw(var)


def read_frame(path, valid_time, names: Iterable[str] = FRAME_VARIABLES, *, tolerance_s: float = 0.0
               ) -> dict[str, np.ndarray]:
    """Arrays (file dtype, Time axis dropped) of ``names`` at ``valid_time``.

    Exact ``Times`` match by default; a missing frame/variable or a non-finite value refuses.
    ``tolerance_s`` > 0 is only for TES offset frames (hh:00:18/hh:00:36 with dt 54 s).
    """
    out = {}
    with Dataset(str(path)) as ds:
        index, _ = _time_index(ds, valid_time, tolerance_s)
        for name in names:
            if name not in ds.variables:
                raise ValueError(f"{path}: variable {name} missing")
            value = _frame_value(ds, name, index)
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{path}: {name} has non-finite values")
            out[name] = value
    return out


def read_frame_time(path, valid_time, *, tolerance_s: float = 0.0) -> str:
    with Dataset(str(path)) as ds:
        return _time_index(ds, valid_time, tolerance_s)[1]


def read_frame_index(path, valid_time, *, tolerance_s: float = 0.0) -> tuple[int, str]:
    with Dataset(str(path)) as ds:
        return _time_index(ds, valid_time, tolerance_s)


def frame_path(run_dir, domain: str, valid_time) -> Path:
    """Exact-time wrfout of ``domain`` at ``valid_time`` in ``run_dir`` (must exist)."""
    path = Path(run_dir) / f"wrfout_{domain}_{wrf_time(valid_time)}"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: exact-time frame required")
    return path


def _namelist_spec_bdy_width(namelist) -> tuple[int | None, str, bool]:
    """(width, namelist path or 'absent', unset): a namelist without the key runs WRF's Registry default 5."""
    if namelist is None or not Path(namelist).is_file():
        return None, "absent", False
    from gpuwrf.io.namelist_parser import parse_namelist
    value = parse_namelist(Path(namelist)).get("bdy_control", {}).get("spec_bdy_width")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return (5 if value is None else int(value)), str(namelist), value is None


def _base_alb(pb: np.ndarray, scalars: Mapping[str, float]) -> np.ndarray:
    """WRF start_em/real.exe base inverse density from PB (:610-620); used only without a wrfinput ALB."""
    p00, t00, a = scalars["P00"], scalars["T00"], scalars["TLP"]
    tiso, p_strat, a_strat = scalars["TISO"], scalars["P_STRAT"], scalars["TLP_STRAT"]
    temp = np.maximum(tiso, t00 + a * np.log(pb / p00))
    temp = np.where(pb < p_strat, tiso + a_strat * np.log(pb / max(p_strat, 1e-30)), temp)
    t_init = temp * (p00 / pb) ** (R_D / CP) - T0
    return (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM


def read_geometry(path, *, base_wrfinput=None, namelist="auto", time_index: int = 0) -> dict:
    """Static geometry + vertical coordinate + dry base state of one domain (wrfout or wrfinput).

    ``ALB`` comes from ``path`` if present, else from ``base_wrfinput`` (exact real.exe value; its
    PB must equal ``path``'s PB bitwise), else the WRF base-state formula (labelled). ``namelist``
    = ``"auto"`` reads ``namelist.input`` next to ``path`` for ``spec_bdy_width`` (None if absent).
    """
    geo: dict = {"source": str(path)}
    with Dataset(str(path)) as ds:
        for name in _REQUIRED_2D + _VCOORD_1D + _BASE_3D:
            if name not in ds.variables:
                raise ValueError(f"{path}: geometry variable {name} missing")
            geo[name] = _frame_value(ds, name, time_index)
        for name in _OPTIONAL_2D:
            if name in ds.variables:
                geo[name] = _frame_value(ds, name, time_index)
        scalars = {}
        for name in _SCALARS:
            if name not in ds.variables:
                raise ValueError(f"{path}: scalar {name} missing")
            scalars[name] = float(np.asarray(_frame_value(ds, name, time_index)).reshape(-1)[0])
        geo.update(scalars)
        for name in _ATTRS_REQUIRED:
            if name not in ds.ncattrs():
                raise ValueError(f"{path}: global attribute {name} missing")
            geo[name] = ds.getncattr(name)
        for name in _ATTRS_OPTIONAL:
            if name in ds.ncattrs():
                geo[name] = ds.getncattr(name)
        if "ALB" in ds.variables:
            geo["ALB"], geo["ALB_source"] = _frame_value(ds, "ALB", time_index), str(path)
    if "ALB" not in geo and base_wrfinput is not None:
        with Dataset(str(base_wrfinput)) as base:
            if not np.array_equal(_frame_value(base, "PB", 0), geo["PB"]):
                raise ValueError(f"{base_wrfinput}: PB differs from {path}; ALB would be inconsistent")
            geo["ALB"], geo["ALB_source"] = _frame_value(base, "ALB", 0), str(base_wrfinput)
    if "ALB" not in geo:
        geo["ALB"] = _base_alb(np.asarray(geo["PB"], np.float64), scalars).astype(np.asarray(geo["PB"]).dtype)
        geo["ALB_source"] = "recomputed:WRF base-state formula from PB"
    if namelist == "auto":
        namelist = Path(path).parent / "namelist.input"
    geo["SPEC_BDY_WIDTH"], geo["SPEC_BDY_WIDTH_source"], geo["SPEC_BDY_WIDTH_unset"] = _namelist_spec_bdy_width(namelist)
    geo["nz"] = int(np.asarray(geo["ZNU"]).shape[-1])
    for name in _VCOORD_1D:  # 1-D (Time dropped); guard against (1, n) storage
        geo[name] = np.asarray(geo[name]).reshape(-1)
    return geo


def assert_same_grid(a, b, names: Iterable[str] = GRID_IDENTITY) -> None:
    """Refuse two files whose grid/geometry differ (exact arrays + DX/DY)."""
    with Dataset(str(a)) as da, Dataset(str(b)) as db:
        for attr in ("DX", "DY"):
            if da.getncattr(attr) != db.getncattr(attr):
                raise ValueError(f"grid differs: {attr} {a} vs {b}")
        for name in names:
            if not np.array_equal(_frame_value(da, name, 0), _frame_value(db, name, 0)):
                raise ValueError(f"grid differs: {name} {a} vs {b}")


# --------------------------------------------------------------------------- ensemble container

class EnsembleFrames(dict):
    """dict name -> (N, ...) member arrays at ONE valid time (only arrays are items).

    DESIGN_v2 §4 mapping (probe nt = 1, time axis implicit): ``member_ids`` (N str), ``times`` (actual
    UTC ``Times`` of the analysis frame), ``valid_time``, ``members`` (per-member provenance: source
    file(s), frame index, SHA256), ``control_file`` (passive fields come from this frame; None for
    real-member ensembles), ``theta_convention``, ``meta`` (construction record). Staggers follow the
    WRF names (U x, V y, PH/W z). Do not add non-array items: consumers iterate ``.items()``.
    """

    theta_convention = "T = dry theta - 300 K (wrfout); THM = prognostic per USE_THETA_M"

    def __init__(self, fields: Mapping[str, np.ndarray], *, member_ids, valid_time, members=(), meta=None,
                 times=None, control_file=None):
        super().__init__(fields)
        self.member_ids = tuple(str(m) for m in member_ids)
        self.valid_time = wrf_time(valid_time)
        self.times = tuple(times) if times is not None else (self.valid_time,)
        self.control_file = None if control_file is None else str(control_file)
        self.members = list(members)
        self.meta = dict(meta or {})
        n = len(self.member_ids)
        for name, value in self.items():
            if np.shape(value)[:1] != (n,):
                raise ValueError(f"{name}: leading axis {np.shape(value)[:1]} != N={n}")

    @property
    def n_members(self) -> int:
        return len(self.member_ids)

    def member(self, m: int) -> dict[str, np.ndarray]:
        return {k: v[m] for k, v in self.items()}

    def mean(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, np.float64).mean(axis=0) for k, v in self.items()}


def read_member_frames(paths, valid_time, names: Iterable[str] = FRAME_VARIABLES, *, member_ids=None
                       ) -> EnsembleFrames:
    """Stack the exact-time frames of real member wrfout files (archive/real ensembles)."""
    paths = [Path(p) for p in paths]
    names = tuple(names)
    for p in paths[1:]:
        assert_same_grid(paths[0], p)
    frames = [read_frame(p, valid_time, names) for p in paths]
    fields = {n: np.stack([f[n] for f in frames]) for n in names}
    ids = member_ids or [p.name for p in paths]
    members = [{"file": str(p), "frame_index": read_frame_index(p, valid_time)[0], "sha256": sha256(p)}
               for p in paths]
    return EnsembleFrames(fields, member_ids=ids, valid_time=valid_time, members=members,
                          meta={"source": "read_member_frames"})


# --------------------------------------------------------------------------- taper

def boundary_taper(ny: int, nx: int, spec_bdy_width: int = 5, ramp: int = 5, stagger=None) -> np.ndarray:
    """Increment weight on the MASS grid (ny, nx) or its U/V stagger (ny, nx+1) / (ny+1, nx).

    0 in the WRF specified+relaxation zone (rows/faces 0..spec_bdy_width-1 from each edge, so the
    analysis never touches a wrfbdy-forced point), cosine ramp over ``ramp`` rows, 1 inside.
    Ramp row r (0-based) = 0.5*(1 - cos(pi*(r+1)/(ramp+1))).
    """
    if spec_bdy_width < 0 or ramp < 0 or ny < 1 or nx < 1:
        raise ValueError("invalid taper geometry")
    stagger = {None: None, "x": "x", "U": "x", "y": "y", "V": "y"}[stagger]
    nxs, nys = nx + (stagger == "x"), ny + (stagger == "y")
    i, j = np.arange(nxs), np.arange(nys)
    di = np.minimum(i, nxs - 1 - i)
    dj = np.minimum(j, nys - 1 - j)
    d = np.minimum(dj[:, None], di[None, :]).astype(np.float64)
    r = d - spec_bdy_width
    w = 0.5 * (1.0 - np.cos(np.pi * (r + 1.0) / (ramp + 1.0)))
    return np.where(r < 0, 0.0, np.where(r >= ramp, 1.0, w))


# --------------------------------------------------------------------------- hydrostatic rebalance

def _balance_inputs(geo: Mapping) -> tuple[RealInitConfig, VerticalCoord1D, BaseStateColumns]:
    hybrid, hypso = int(geo["HYBRID_OPT"]), int(geo["HYPSOMETRIC_OPT"])
    # The port's balance() selects the hypsometric_opt=1 equations by hybrid_opt == 0 (real.exe
    # selects by hypsometric_opt). Only the two consistent pairings are admitted.
    if not ((hypso == 2 and hybrid != 0) or (hypso == 1 and hybrid == 0)):
        raise ValueError(f"rebalance supports (hybrid_opt!=0, hypsometric_opt=2) or (0, 1); got ({hybrid}, {hypso})")
    f = lambda n: np.asarray(geo[n], np.float64).reshape(-1)  # noqa: E731
    p_top = float(geo["P_TOP"])
    cfg = RealInitConfig(nz=int(f("ZNU").size), p_top_pa=p_top, hybrid_opt=hybrid, etac=0.0)
    vc = VerticalCoord1D(znw=f("ZNW"), znu=f("ZNU"), dnw=f("DNW"), rdnw=f("RDNW"), dn=f("DN"), rdn=f("RDN"),
                         fnp=f("FNP"), fnm=f("FNM"), c1f=f("C1F"), c2f=f("C2F"), c3f=f("C3F"), c4f=f("C4F"),
                         c1h=f("C1H"), c2h=f("C2H"), c3h=f("C3H"), c4h=f("C4H"), cf1=float(geo["CF1"]),
                         cf2=float(geo["CF2"]), cf3=float(geo["CF3"]), cfn=float(geo["CFN"]),
                         cfn1=float(geo["CFN1"]), p_top_pa=p_top)
    pb = np.asarray(geo["PB"], np.float64)
    base = BaseStateColumns(pb=pb, alb=np.asarray(geo["ALB"], np.float64), t_init=np.zeros(0),
                            mub=np.asarray(geo["MUB"], np.float64), phb=np.asarray(geo["PHB"], np.float64))
    return cfg, vc, base


def balance_state(theta_pert, qv, mu, geo) -> dict[str, np.ndarray]:
    """Cross-check ORACLE only: P, PH, AL of the port's real.exe balance (float64; P from its EOS step
    with the given qv, before balance's RH re-derivation of qv). Not used by :func:`rebalance_increment`."""
    cfg, vc, base = _balance_inputs(geo)
    theta = np.asarray(theta_pert, np.float64)
    qv = np.asarray(qv, np.float64)
    mu = np.asarray(mu, np.float64)
    empty = np.zeros(0)
    seed = DynamicsInit(u=empty, v=empty, w=empty, theta=theta, qv=qv, mu=mu, mu0=mu + base.mub,
                        p=base.pb, ph=empty, al=empty, alt=empty, p_hyd=empty)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = balance(cfg, vc, base, seed)
    al = np.asarray(out.al, np.float64)
    qvf = 1.0 + RVOVRD * qv
    p = P1000MB * ((R_D * (T0 + theta) * qvf) / (P1000MB * (al + base.alb))) ** CPOVCV - base.pb
    return {"P": p, "PH": np.asarray(out.ph, np.float64), "AL": al}


def _col(geo, name) -> np.ndarray:
    return np.asarray(geo[name], np.float64).reshape(-1)[:, None, None]


def _hypsometric_factor(mu, geo) -> np.ndarray:
    """Per-layer dφ/dα (nz, ny, nx): phm·ln(pfd/pfu) (opt 2) or −dnw·(c1h·μ_d + c2h) (opt 1)."""
    mut = np.asarray(geo["MUB"], np.float64) + np.asarray(mu, np.float64)
    if int(geo["HYPSOMETRIC_OPT"]) == 1:
        return -_col(geo, "DNW") * (_col(geo, "C1H") * mut + _col(geo, "C2H"))
    p_top = float(geo["P_TOP"])
    c3f, c4f = _col(geo, "C3F"), _col(geo, "C4F")
    pfu = c3f[1:] * mut + c4f[1:] + p_top
    pfd = c3f[:-1] * mut + c4f[:-1] + p_top
    phm = _col(geo, "C3H") * mut + _col(geo, "C4H") + p_top
    return phm * np.log(pfd / pfu)


def eos_alpha(t_pert, qv, p_pert, geo) -> np.ndarray:
    """real.exe :3937/:3955 inverse dry density (float64)."""
    pfull = np.asarray(p_pert, np.float64) + np.asarray(geo["PB"], np.float64)
    return ((R_D / P1000MB) * (np.asarray(t_pert, np.float64) + T0) * (1.0 + RVOVRD * np.asarray(qv, np.float64))
            * (pfull / P1000MB) ** CVPM)


def moisture_loading_increment(dqv, mu, geo) -> np.ndarray:
    """Δp′ of the real.exe top-down recursion (:3923-3953) for a moisture increment at fixed MU."""
    dqv = np.asarray(dqv, np.float64)
    mut = np.asarray(geo["MUB"], np.float64) + np.asarray(mu, np.float64)
    c1f, c2f = np.asarray(geo["C1F"], np.float64).reshape(-1), np.asarray(geo["C2F"], np.float64).reshape(-1)
    rdnw, rdn = np.asarray(geo["RDNW"], np.float64).reshape(-1), np.asarray(geo["RDN"], np.float64).reshape(-1)
    nz = dqv.shape[0]
    dp = np.empty_like(dqv)
    dp[nz - 1] = -0.5 * dqv[nz - 1] * (c1f[nz] * mut + c2f[nz]) / rdnw[nz - 1]
    for kk in range(nz - 2, -1, -1):
        dq = 0.5 * (dqv[kk] + dqv[kk + 1])
        dp[kk] = dp[kk + 1] - dq * (c1f[kk + 1] * mut + c2f[kk + 1]) / rdn[kk + 1]
    return dp


def diagnose_al(ph, mu, geo) -> np.ndarray:
    """WRF start_em :794-812 inverse density from PH and MU (float64)."""
    ph = np.asarray(ph, np.float64)
    mu = np.asarray(mu, np.float64)
    mub = np.asarray(geo["MUB"], np.float64)
    phb = np.asarray(geo["PHB"], np.float64)
    alb = np.asarray(geo["ALB"], np.float64)
    if int(geo["HYPSOMETRIC_OPT"]) == 1:
        c1h, c2h, rdnw = _col(geo, "C1H"), _col(geo, "C2H"), _col(geo, "RDNW")
        return -1.0 / ((c1h * mub + c2h) + c1h * mu) * (alb * (c1h * mu) + rdnw * (ph[1:] - ph[:-1]))
    return (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]) / _hypsometric_factor(mu, geo) - alb


def hydrostatic_residual(t_pert, qv, p_pert, ph, mu, geo) -> np.ndarray:
    """Discrete model-form residual per layer: (Δφ_total − α_EOS·factor) for HYPSOMETRIC_OPT 2;
    opt 1 uses perturbation φ with the alb·c1h·μ′ term. Background frames are not zero (non-hydrostatic
    model state); the analysis must change it by round-off only."""
    ph = np.asarray(ph, np.float64)
    alpha = eos_alpha(t_pert, qv, p_pert, geo)
    if int(geo["HYPSOMETRIC_OPT"]) == 1:
        alb = np.asarray(geo["ALB"], np.float64)
        mu = np.asarray(mu, np.float64)
        return (ph[1:] - ph[:-1]) - _hypsometric_factor(mu, geo) * (alpha - alb) + _col(geo, "DNW") * (
            _col(geo, "C1H") * mu) * alb
    phb = np.asarray(geo["PHB"], np.float64)
    return (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]) - alpha * _hypsometric_factor(mu, geo)


HYDROMETEORS = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL")


def rebalance_increment(bg: Mapping, an: Mapping, geo: Mapping, *, return_increments: bool = False
                        ) -> dict[str, np.ndarray]:
    """Fixed-QV/fixed-MU incremental rebalance (module docstring) in the background dtype.

    ``bg``/``an``: one member's wrfout-name arrays (T, QVAPOR, MU; bg also P, PH, optional P_HYD, AL).
    Refuses a MU increment (frozen in the probe), negative/non-finite analysed qv and non-positive
    pressure/α. Returns P, PH (+ P_HYD if in bg) and AL (AL_b + Δα if bg has AL, else start_em-diagnosed
    from PH_a when ``geo`` has ALB); ``return_increments`` adds float64 ``increments`` {P, PH, ALPHA}.
    """
    hypso = int(geo["HYPSOMETRIC_OPT"])
    if hypso not in (1, 2):
        raise ValueError(f"HYPSOMETRIC_OPT {hypso} unsupported")
    if not np.array_equal(np.asarray(an["MU"]), np.asarray(bg["MU"])):
        raise ValueError("MU increment: the fixed-MU rebalance (DESIGN_v2 §3.1) refuses an analysed MU")
    changed = [k for k in HYDROMETEORS if k in an and k in bg and not np.array_equal(np.asarray(an[k]), np.asarray(bg[k]))]
    if changed:   # WRF's hydrostatic recursion loads every moisture species; this rebalance carries only dQV
        raise ValueError(f"analysed {changed} changed: the fixed-QV rebalance only carries QVAPOR increments")
    qv_a = np.asarray(an["QVAPOR"], np.float64)
    if not (np.isfinite(qv_a).all() and np.isfinite(np.asarray(an["T"], np.float64)).all()):
        raise ValueError("non-finite analysed T/QVAPOR")
    if (qv_a < 0).any():
        raise ValueError(f"negative analysed QVAPOR in {int((qv_a < 0).sum())} cells: admissibility first")
    mu = np.asarray(bg["MU"], np.float64)
    p_b = np.asarray(bg["P"], np.float64)
    dp = moisture_loading_increment(qv_a - np.asarray(bg["QVAPOR"], np.float64), mu, geo)
    alpha_b = eos_alpha(bg["T"], bg["QVAPOR"], p_b, geo)
    alpha_a = eos_alpha(an["T"], qv_a, p_b + dp, geo)
    if not ((p_b + dp + np.asarray(geo["PB"], np.float64) > 0).all() and (alpha_a > 0).all()):
        raise ValueError("non-positive analysed pressure or inverse density")
    dalpha = alpha_a - alpha_b
    dph = np.zeros((dalpha.shape[0] + 1,) + dalpha.shape[1:])
    np.cumsum(dalpha * _hypsometric_factor(mu, geo), axis=0, out=dph[1:])
    out = {}
    for name, inc in (("P", dp), ("PH", dph)):
        ref = np.asarray(bg[name])
        out[name] = (ref.astype(np.float64) + inc).astype(ref.dtype)
    if "P_HYD" in bg:
        ref = np.asarray(bg["P_HYD"])
        out["P_HYD"] = (ref.astype(np.float64) + dp).astype(ref.dtype)
    if "AL" in bg:
        ref = np.asarray(bg["AL"])
        out["AL"] = (ref.astype(np.float64) + dalpha).astype(ref.dtype)
    elif "ALB" in geo:
        out["AL"] = diagnose_al(out["PH"], mu, geo).astype(np.asarray(bg["P"]).dtype)
    if return_increments:
        out["increments"] = {"P": dp, "PH": dph, "ALPHA": dalpha}
    return out


def rebalance_report(bg: Mapping, an: Mapping, out: Mapping, geo: Mapping) -> dict:
    """V3b numbers for one rebalanced state (``out`` = :func:`rebalance_increment` result, stored dtype):
    model-form hydrostatic residual change relative to the layer thickness, positivity, increment extrema."""
    r_b = hydrostatic_residual(bg["T"], bg["QVAPOR"], bg["P"], bg["PH"], bg["MU"], geo)
    r_a = hydrostatic_residual(an["T"], an["QVAPOR"], out["P"], out["PH"], an["MU"], geo)
    total = np.asarray(out["PH"], np.float64) + np.asarray(geo["PHB"], np.float64)
    thick = np.abs(np.diff(total, axis=0))
    p_full = np.asarray(out["P"], np.float64) + np.asarray(geo["PB"], np.float64)
    alpha = eos_alpha(an["T"], an["QVAPOR"], out["P"], geo)
    d = {n: np.asarray(out[n], np.float64) - np.asarray(bg[n], np.float64) for n in ("P", "PH")}
    return {"residual_change_rel_max": float((np.abs(r_a - r_b) / thick).max()),
            "min_full_pressure_pa": float(p_full.min()), "min_alpha": float(alpha.min()),
            "min_qv": float(np.asarray(an["QVAPOR"], np.float64).min()),
            "dP_min_max": [float(d["P"].min()), float(d["P"].max())],
            "dPH_min_max": [float(d["PH"].min()), float(d["PH"].max())],
            "dPH_surface_max_abs": float(np.abs(d["PH"][0]).max())}


def analysis_mean_fields(control: Mapping, background: Mapping, analysis: Mapping, geo: Mapping,
                         names: Iterable[str] = ("U", "V", "T", "QVAPOR"), *, negative_qv: str | None = None,
                         moisture=None, counters: dict | None = None) -> dict[str, np.ndarray]:
    """Fields for the deterministic analysis-mean wrfinput: control + mean_m(xa_m − xb_m) per analysed name
    (cells whose member increments are all 0 — the tapered spec/relax zone — stay the control BITWISE; averaging
    float32 members would perturb them at round-off), then P/PH (+AL/P_HYD) from :func:`rebalance_increment`
    of that mean against the control (averaging member P/PH breaks the residual gate: 2.8e-6 vs 1.2e-7 measured
    on 0503 d01). ``control`` = one wrfout frame dict; ``background``/``analysis`` = (N, ...) member arrays.

    Moisture: ``moisture`` (``moisture.MoisturePolicy``; or the policy name ``negative_qv``, default limits) is
    applied to the mean QVAPOR with the same counters and water-budget gate as the members. ``counters`` receives
    ``admissible``/``refusal_reasons``; an inadmissible mean returns the fields WITHOUT rebalance (caller refuses)."""
    from . import moisture as mst

    if moisture is None:
        moisture = mst.MoisturePolicy(negative="clip" if negative_qv is None else negative_qv)
    counters = {} if counters is None else counters
    out = {}
    for name in names:
        ref = np.asarray(control[name])
        inc = (np.asarray(analysis[name], np.float64) - np.asarray(background[name], np.float64)).mean(axis=0)
        out[name] = (ref.astype(np.float64) + inc).astype(ref.dtype)
    reasons = []
    if "QVAPOR" in out:
        qv, raw, neg = mst.negative_step(np.asarray(out["QVAPOR"])[None], moisture, np.asarray(control["QVAPOR"])[None])
        counters["mean_QVAPOR_negative"] = neg["negative_cells"]
        counters["mean_QVAPOR_negative_min"] = neg["negative_min_kgkg"]
        nonfinite = int((~np.isfinite(raw)).sum())
        budget = None
        if moisture.negative != "refuse":
            counters["mean_QVAPOR_negative_clipped"] = neg["negative_cells"]
            mass, label = mst.layer_dry_mass(geo, np.asarray(control["MU"])[None],
                                             (np.asarray(control["P"], np.float64) + np.asarray(geo["PB"]))[None])
            budget = mst.water_budget(raw, qv, mass, geo)
            counters["mean_budget"] = {**budget, "dry_mass": label}
        ok, reasons = mst.gate(moisture, {"negative_cells": neg["negative_cells"], "nonfinite_cells": nonfinite}, budget)
        out["QVAPOR"] = qv[0].astype(np.asarray(out["QVAPOR"]).dtype)
    counters["admissible"] = not reasons
    counters["refusal_reasons"] = reasons
    if reasons:
        return out
    an = {**{k: control[k] for k in ("T", "QVAPOR", "MU")}, **out}
    out.update({k: v for k, v in rebalance_increment(control, an, geo).items() if k in ("P", "PH")})
    return out


# --------------------------------------------------------------------------- analysis wrfinput

def _theta_m_pert(t_pert, qv, use_theta_m: int) -> np.ndarray:
    theta_dry = np.asarray(t_pert, np.float64) + T0
    theta_m = theta_dry * (1.0 + RVOVRD * np.asarray(qv, np.float64))
    return thm_source(theta_m, theta_dry, use_theta_m) - T0


def _set_times(ds, stamp: str) -> None:
    var = ds["Times"]
    width = var.shape[-1]
    row = np.frombuffer(stamp.encode("ascii").ljust(width, b"\0")[:width], dtype="S1")
    var[0] = row


def _start_attributes(ds, when: datetime) -> dict:
    stamp = when.strftime(WRF_TIME_FORMAT)
    attrs = {"START_DATE": stamp, "SIMULATION_START_DATE": stamp,
             "GMT": np.float32(when.hour + when.minute / 60.0 + when.second / 3600.0),
             "JULYR": np.int32(when.year), "JULDAY": np.int32(when.timetuple().tm_yday)}
    for name, value in attrs.items():
        if name in ds.ncattrs():
            ds.setncattr(name, value)
    return {k: (v if isinstance(v, str) else v.item()) for k, v in attrs.items()}


def write_analysis_wrfinput(template_wrfinput, member_wrfout, valid_time, analysis_fields: Mapping[str, np.ndarray],
                            out_path, *, land_cycle: bool = True, extra_provenance: Mapping | None = None,
                            source_labels: Mapping[str, str] | None = None) -> dict:
    """Write a NEW warm-start wrfinput valid at ``valid_time`` (atomic, never overwrites).

    1. copy the case ``template_wrfinput`` (statics, ALB, T_INIT, ...);
    2. every template variable the member frame at ``valid_time`` carries with identical
       dims/dtype is taken from the frame (time-varying state incl. W, hydrometeors, QKE, land);
    3. ``analysis_fields`` overwrite (U, V, T, QVAPOR, P, PH, ... in wrfout names/units; MU frozen in the
       probe, an unchanged MU is accepted);
    4. derived: THM = THM_b + θm(T_a, qv_a) − θm(T_b, qv_b) (``wrfout_theta_convention``),
       P_HYD = P_HYD_b + (P_a − P_b) unless given, AL from PH/MU (WRF start_em) unless given;
    5. ``Times`` and START_DATE/SIMULATION_START_DATE/GMT/JULYR/JULDAY set to ``valid_time``;
       provenance JSON attribute ``GPUWRF_LETKF_ANALYSIS`` with input SHA256s and the field inventory
       (analysed / recomputed / copied / template / reset-by-warm-start lists);
    6. Noah-MP land via :func:`gpuwrf.cycling.core.cycle` (land marker + atomic no-clobber publish).
    Zero increments reproduce the member frame bitwise on every copied/analysed variable.
    ``source_labels`` {resolved path: recorded name}: inputs read from a governance stage are recorded under their
    original path (``inputs_staged`` marks it; the SHA-256s are those of the bytes actually read).
    """
    template, member, out = (Path(p).resolve() for p in (template_wrfinput, member_wrfout, out_path))
    if out in (template, member) or out.exists():
        raise ValueError("out_path must be a new file, distinct from both inputs")
    stamp = wrf_time(valid_time)
    template_sha, member_sha = sha256(template), sha256(member)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".a1io-analysis-", suffix=".nc", dir=out.parent)
    os.close(fd)
    staging = Path(temporary)
    try:
        shutil.copyfile(template, staging)
        try:
            assert_same_grid(template, member, ("XLAT", "XLONG", "HGT", "PB", "PHB", "MUB"))
        except ValueError as exc:
            raise ValueError(f"{exc}: template wrfinput and member frame base states differ (on WN3 nests WRF's "
                             "nest init blends terrain: d02/d03 HGT/PB/PHB/MUB change); nested analysis wrfinputs "
                             "need a parent/child-consistent writer (DESIGN_v2: deferred)") from exc
        labels = dict(source_labels or {})
        record: dict = {"schema": 1, "valid_time": stamp, "template": labels.get(str(template), str(template)),
                        "template_sha256": template_sha, "member_wrfout": labels.get(str(member), str(member)),
                        "member_sha256": member_sha, "created_utc": datetime.now(timezone.utc).isoformat()}
        if str(template) in labels or str(member) in labels:
            record["inputs_staged"] = True
        with Dataset(str(member)) as src, Dataset(str(staging), "r+") as dst:
            index, _ = _time_index(src, stamp)
            record["member_frame_index"] = index
            use_theta_m = parse_use_theta_m(int(src.getncattr("USE_THETA_M")))
            if "USE_THETA_M" in dst.ncattrs() and parse_use_theta_m(int(dst.getncattr("USE_THETA_M"))) != use_theta_m:
                raise ValueError("USE_THETA_M differs between template and member")
            frame, copied, template_only = {}, [], []
            for name, var in dst.variables.items():
                if name == "Times":
                    continue
                if (name in src.variables and src[name].dimensions == var.dimensions
                        and src[name].dtype == var.dtype):
                    value = _frame_value(src, name, index)
                    frame[name] = value
                    var[0] = value
                    copied.append(name)
                else:
                    template_only.append(name)
            fields: dict[str, np.ndarray] = {}
            stats: dict[str, dict] = {}
            for name, value in analysis_fields.items():
                if name not in dst.variables or dst[name].dimensions[:1] != ("Time",):
                    raise ValueError(f"analysis field {name} is not a time-varying template variable")
                var = dst[name]
                value = np.asarray(value)
                if value.shape != var.shape[1:]:
                    raise ValueError(f"analysis field {name}: shape {value.shape} != {var.shape[1:]}")
                if not np.isfinite(value).all():
                    raise ValueError(f"analysis field {name}: non-finite values")
                stored = value.astype(var.dtype)
                fields[name] = stored
                ref = frame.get(name)
                if ref is not None:
                    inc = stored.astype(np.float64) - ref.astype(np.float64)
                    stats[name] = {"n_changed": int(np.count_nonzero(inc)), "max_abs_increment": float(np.abs(inc).max()),
                                   "rms_increment": float(np.sqrt(np.mean(inc ** 2)))}
                else:
                    stats[name] = {"n_changed": None, "note": "template-only variable overwritten"}
                var[0] = stored
            derived = {}
            if ("T" in fields or "QVAPOR" in fields) and "THM" in frame and "THM" not in fields:
                t_b, q_b = frame["T"], frame["QVAPOR"]
                t_a, q_a = fields.get("T", t_b), fields.get("QVAPOR", q_b)
                inc = _theta_m_pert(t_a, q_a, use_theta_m) - _theta_m_pert(t_b, q_b, use_theta_m)
                fields["THM"] = (frame["THM"].astype(np.float64) + inc).astype(frame["THM"].dtype)
                dst["THM"][0] = fields["THM"]
                derived["THM"] = f"incremental thm_source(use_theta_m={use_theta_m})"
            if "P" in fields and "P_HYD" in frame and "P_HYD" not in fields:
                inc = fields["P"].astype(np.float64) - frame["P"].astype(np.float64)
                fields["P_HYD"] = (frame["P_HYD"].astype(np.float64) + inc).astype(frame["P_HYD"].dtype)
                dst["P_HYD"][0] = fields["P_HYD"]
                derived["P_HYD"] = "P_HYD_b + (P_a - P_b)"
            if "AL" in dst.variables and "AL" not in fields:
                geo = {n: _frame_value(dst, n, 0) for n in ("MUB", "PHB", "ALB", "P_TOP") + _VCOORD_1D}
                geo["HYPSOMETRIC_OPT"] = int(dst.getncattr("HYPSOMETRIC_OPT"))
                ph = fields.get("PH", _frame_value(dst, "PH", 0))
                mu = fields.get("MU", _frame_value(dst, "MU", 0))
                fields["AL"] = diagnose_al(ph, mu, geo).astype(dst["AL"].dtype)
                dst["AL"][0] = fields["AL"]
                derived["AL"] = "diagnosed from PH/MU (WRF start_em hypsometric form)"
            _set_times(dst, stamp)
            record["start_attributes"] = _start_attributes(dst, parse_wrf_time(stamp))
            frame_only = sorted(set(src.variables) - set(dst.variables) - {"Times"})
            record.update({"analysis_fields": stats, "derived": derived, "copied_from_frame": sorted(copied),
                           "template_only": sorted(template_only), "frame_only": frame_only,
                           "use_theta_m": use_theta_m})
            record["inventory"] = {  # DESIGN_v2 §3 field inventory
                "analysed": sorted(n for n, v in stats.items() if v.get("n_changed")),
                "written_unchanged": sorted(n for n, v in stats.items() if v.get("n_changed") == 0),
                "recomputed": sorted(derived),
                "copied_from_frame": sorted(set(copied) - set(stats) - set(derived)),
                "template_static_or_initial_time": sorted(template_only),
                "reset_by_warm_start_not_in_wrfinput": frame_only}
            record["warm_start_notes"] = {  # what the consumers do with the written fields (pristine start_em)
                "W": "copied from the frame; WRF start_em with use_input_w=.false. (default) replaces it by "
                     "set_w_surface (:1475-1530), the port keeps it (FINDINGS E121)",
                "P/AL": "WRF start_em re-derives AL from PH/MU and P from AL+THM (:794-842); the port reads P",
                "THM/T": "WRF uses THM (t_2); the port re-derives theta_m from T and QVAPOR"}
            if extra_provenance:
                record["extra"] = dict(extra_provenance)
            dst.setncattr(PROVENANCE_ATTRIBUTE, json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                            default=str))
            dst.sync()
        if land_cycle:
            from gpuwrf.cycling.core import cycle
            report = cycle(member, staging, out)
            record["land_cycle"] = {k: report[k] for k in ("land_cells", "optional_absent", "policy")}
            record["land_cycle"]["fields"] = sorted(report["fields"])
        else:
            with open(staging, "rb") as stream:
                os.fsync(stream.fileno())
            os.link(staging, out)  # atomic no-clobber publication
        if sha256(template) != template_sha or sha256(member) != member_sha:
            out.unlink(missing_ok=True)
            raise ValueError("input changed while writing the analysis")
    finally:
        staging.unlink(missing_ok=True)
    os.chmod(out, 0o644)  # mkstemp staging is 0600; the published input is read by other tools/lanes
    record["output"], record["output_sha256"] = str(out), sha256(out)
    return record
