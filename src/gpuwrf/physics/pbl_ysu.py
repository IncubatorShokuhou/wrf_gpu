"""WRF YSU PBL column kernel for the v0.6.0 PBL lane.

This module ports the scalar-column path of WRF ``bl_ysu_run`` used by
``module_bl_ysu.F`` for the v0.6.0 savepoint gate:

* no BEP/BEM urban canopy forcing,
* no chemical/passive tracer mixing,
* QC3D/QI3D mixing, ysu_topdown_pblmix and the CTOPO vconv form (lane t-ysuqc),
* WRF REAL operation order (pristine public-call oracle proofs/t_ysuqc).

The public entry returns the frozen S0 ``PhysicsStepResult`` with tendencies for
``u``, ``v``, ``theta``, and ``qv`` plus YSU diagnostics. The implementation is
column-level and CPU-parity oriented; dispatcher/domain batching is a later
manager-owned integration task.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency


configure_jax_x64()


G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
XLV = 2.5e6
ROVCP = R_D / CP
EP1 = R_V / R_D - 1.0
EP2 = R_D / R_V
KARMAN = 0.4

XKZMINM = 0.1
XKZMINH = 0.01
XKZMAX = 1000.0
RIMIN = -100.0
RLAM = 30.0
PRMIN = 0.25
PRMAX = 4.0
BRCR_UB = 0.0
BRCR_SB = 0.25
CORI = 1.0e-4
AFAC = 6.8
BFAC = 6.8
PFAC = 2.0
PFAC_Q = 2.0
PHIFAC = 8.0
SFCFRAC = 0.1
D1 = 0.02
D2 = 0.05
D3 = 0.001
H1 = 0.33333335
H2 = 0.6666667
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
TMIN = 1.0e-2
GAMCRT = 3.0
GAMCRQ = 2.0e-3
RCL = 1.0


@jax.tree_util.register_pytree_node_class
class YSUColumnState:
    """Pytree for one independent YSU PBL column on mass levels."""

    __slots__ = ("u", "v", "temperature", "qv", "pressure", "pressure_interface", "exner", "dz")

    def __init__(self, u, v, temperature, qv, pressure, pressure_interface, exner, dz) -> None:
        self.u = u
        self.v = v
        self.temperature = temperature
        self.qv = qv
        self.pressure = pressure
        self.pressure_interface = pressure_interface
        self.exner = exner
        self.dz = dz

    def replace(self, **updates) -> "YSUColumnState":
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, YSUColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


@dataclass(frozen=True)
class YSUDiagnostics:
    pblh: jax.Array
    kpbl: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array
    wstar: jax.Array
    delta: jax.Array


def _leaves(state: YSUColumnState) -> Iterable[jax.Array]:
    return (getattr(state, name) for name in YSUColumnState.__slots__)


def _as1d(value, *, length: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} length {arr.shape[0]} does not match {length}")
    return arr.copy()


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas solve matching WRF ``tridin_ysu``/``tridi2n`` indexing."""

    n = rhs.shape[0]
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n - 1):
        denom = diag[k] - lower[k] * cp[k - 1]
        cp[k] = upper[k] / denom
        dp[k] = (rhs[k] - lower[k] * dp[k - 1]) / denom
    denom = diag[n - 1] - lower[n - 1] * cp[n - 2]
    dp[n - 1] = (rhs[n - 1] - lower[n - 1] * dp[n - 2]) / denom

    out = np.zeros(n, dtype=np.float64)
    out[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        out[k] = dp[k] - cp[k] * out[k + 1]
    return out


def _build_matrix(
    nz: int,
    delp: np.ndarray,
    p: np.ndarray,
    dza: np.ndarray,
    coeff: np.ndarray,
    dt2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    diag[0] = 1.0
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        dsdz2 = dsig * coeff[k] * rdz * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        diag[k] = diag[k] - upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
    return lower, diag, upper


def _first_pbl_guess(thv: np.ndarray, thermal: float, za: np.ndarray, br: float, brcr: float, u: np.ndarray, v: np.ndarray):
    brup = br
    brdn = br
    stable = False
    kpbl = 1
    for k in range(1, thv.shape[0]):
        if not stable:
            brdn = brup
            spdk2 = max(u[k] * u[k] + v[k] * v[k], 1.0)
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
            kpbl = k + 1
            stable = brup > brcr
    return kpbl, brdn, brup


def _interp_pblh(kpbl: int, brdn: float, brup: float, brcr: float, za: np.ndarray, zq: np.ndarray):
    if brdn >= brcr:
        brint = 0.0
    elif brup <= brcr:
        brint = 1.0
    else:
        brint = (brcr - brdn) / (brup - brdn)
    k0 = max(kpbl - 1, 1)
    hpbl = za[k0 - 1] + brint * (za[k0] - za[k0 - 1])
    if hpbl < zq[1]:
        kpbl = 1
    pblflg = kpbl > 1
    return hpbl, kpbl, pblflg


def _ysu_numpy(
    state: YSUColumnState,
    *,
    psfc: float,
    znt: float,
    ust: float,
    hfx: float,
    qfx: float,
    wspd: float,
    br: float,
    psim: float,
    psih: float,
    dt: float,
    xland: float,
    u10: float,
    v10: float,
    uoce: float,
    voce: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | np.ndarray]]:
    u = _as1d(state.u, name="u")
    v = _as1d(state.v, length=u.shape[0], name="v")
    tx = _as1d(state.temperature, length=u.shape[0], name="temperature")
    qv = _as1d(state.qv, length=u.shape[0], name="qv")
    p = _as1d(state.pressure, length=u.shape[0], name="pressure")
    pdi = _as1d(state.pressure_interface, length=u.shape[0] + 1, name="pressure_interface")
    pi = _as1d(state.exner, length=u.shape[0], name="exner")
    dz = _as1d(state.dz, length=u.shape[0], name="dz")
    nz = u.shape[0]

    th = tx / pi
    thli = th.copy()
    thv = th * (1.0 + EP1 * qv)
    rhox = psfc / (R_D * tx[0] * (1.0 + EP1 * qv[0]))
    govrth = G / th[0]

    zq = np.zeros(nz + 1, dtype=np.float64)
    rhox2 = np.zeros(nz, dtype=np.float64)
    for k in range(nz):
        zq[k + 1] = dz[k] + zq[k]
        rhox2[k] = p[k] / (R_D * tx[k] * (1.0 + EP1 * qv[k]))
    za = 0.5 * (zq[:-1] + zq[1:])
    delp = pdi[:-1] - pdi[1:]
    dza = np.empty(nz, dtype=np.float64)
    dza[0] = za[0]
    dza[1:] = za[1:] - za[:-1]

    xkzh = np.zeros(nz, dtype=np.float64)
    xkzm = np.zeros(nz, dtype=np.float64)
    xkzq = np.zeros(nz, dtype=np.float64)
    xkzhl = np.zeros(nz, dtype=np.float64)
    xkzml = np.zeros(nz, dtype=np.float64)
    xkzom = np.full(nz, XKZMINM, dtype=np.float64)
    xkzoh = np.full(nz, XKZMINH, dtype=np.float64)
    xkzom[-1] = 0.0
    xkzoh[-1] = 0.0
    exch_h = np.zeros(nz, dtype=np.float64)
    exch_m = np.zeros(nz, dtype=np.float64)
    zfac = np.zeros(nz, dtype=np.float64)
    zfacent = np.zeros(nz, dtype=np.float64)
    entfac = np.full(nz, 1.0e30, dtype=np.float64)
    wscalek = np.zeros(nz, dtype=np.float64)
    wscalek2 = np.zeros(nz, dtype=np.float64)

    dt2 = 2.0 * dt
    rdt = 1.0 / dt2
    cont = CP / G
    conpr = BFAC * KARMAN * SFCFRAC

    wspd1 = np.sqrt((u[0] - uoce) * (u[0] - uoce) + (v[0] - voce) * (v[0] - voce)) + 1.0e-9
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * th[0]
    sfcflg = not (br > 0.0)
    thermal = float(thv[0])
    thermalli = float(thli[0])
    zl1 = float(za[0])

    kpbl, brdn, brup = _first_pbl_guess(thv, thermal, za, br, BRCR_UB, u, v)
    hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    fm = psim
    fh = psih
    zol1 = max(br * fm * fm / fh, RIMIN)
    zol1 = min(zol1, -ZFMIN) if sfcflg else max(zol1, ZFMIN)
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    if sfcflg:
        phim = (1.0 - APHI16 * hol1) ** (-0.25)
        phih = (1.0 - APHI16 * hol1) ** (-0.5)
        bfx0 = max(sflux, 0.0)
        wstar3 = govrth * bfx0 * hpbl
        wstar = wstar3**H1
    else:
        phim = 1.0 + APHI5 * hol1
        phih = phim
        wstar = 0.0
        wstar3 = 0.0

    ust3 = ust**3
    wscale = (ust3 + PHIFAC * KARMAN * wstar3 * 0.5) ** H1
    wscale = min(wscale, ust * APHI16)
    wscale = max(wscale, ust / APHI5)

    hgamt = 0.0
    hgamq = 0.0
    hgamu = 0.0
    hgamv = 0.0
    wstar3_2 = 0.0
    if sfcflg and sflux > 0.0:
        gamfac = BFAC / rhox / wscale
        hgamt = min(gamfac * hfx / CP, GAMCRT)
        hgamq = min(gamfac * qfx, GAMCRQ)
        vpert = (hgamt + EP1 * th[0] * hgamq) / BFAC * AFAC
        thermal = thermal + max(vpert, 0.0) * min(za[0] / (SFCFRAC * hpbl), 1.0)
        thermalli = thermalli + max(vpert, 0.0) * min(za[0] / (SFCFRAC * hpbl), 1.0)
        hgamt = max(hgamt, 0.0)
        hgamq = max(hgamq, 0.0)
        brint = -15.9 * ust * ust / wspd * wstar3 / (wscale**4)
        hgamu = brint * u[0]
        hgamv = brint * v[0]
    else:
        pblflg = False

    if pblflg:
        kpbl = 1
        hpbl = zq[0]
        kpbl, brdn, brup = _first_pbl_guess(thv, thermal, za, br, BRCR_UB, u, v)
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    if (not sfcflg) and hpbl < zq[1]:
        stable = False
        brup = br
    else:
        stable = True
        brup = br

    brcr = BRCR_UB
    if (not stable) and ((xland - 1.5) >= 0.0):
        wspd10 = np.sqrt(u10 * u10 + v10 * v10)
        ross = wspd10 / (CORI * znt)
        brcr = min(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
    elif not stable:
        brcr = BRCR_SB

    for k in range(1, nz):
        if not stable:
            brdn = brup
            spdk2 = max(u[k] * u[k] + v[k] * v[k], 1.0)
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
            kpbl = k + 1
            stable = brup > brcr

    if (not sfcflg) and hpbl < zq[1]:
        hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, brcr, za, zq)

    cloudflg = False
    wm2 = 0.0
    we = 0.0
    hfxpbl = 0.0
    qfxpbl = 0.0
    ufxpbl = 0.0
    vfxpbl = 0.0
    delta = 0.0
    prpbl = 1.0
    if pblflg:
        k = kpbl - 2
        wm3 = wstar3 + 5.0 * ust3
        wm2 = wm3**H2
        bfxpbl = -0.15 * thv[0] / G * wm3 / hpbl
        dthvx = max(thv[k + 1] - thv[k], TMIN)
        we = max(bfxpbl / dthvx, -np.sqrt(wm2))
        dthx = max(th[k + 1] - th[k], TMIN)
        dqx = min(qv[k + 1] - qv[k], 0.0)
        hfxpbl = we * dthx
        qfxpbl = we * dqx
        dux = u[k + 1] - u[k]
        dvx = v[k + 1] - v[k]
        if dux > TMIN:
            ufxpbl = max(prpbl * we * dux, -ust * ust)
        elif dux < -TMIN:
            ufxpbl = min(prpbl * we * dux, ust * ust)
        if dvx > TMIN:
            vfxpbl = max(prpbl * we * dvx, -ust * ust)
        elif dvx < -TMIN:
            vfxpbl = min(prpbl * we * dvx, ust * ust)
        delb = govrth * D3 * hpbl
        delta = min(D1 * hpbl + D2 * wm2 / delb, 100.0)

    for k in range(nz):
        if pblflg and (k + 1) >= kpbl:
            entfac[k] = ((zq[k + 1] - hpbl) / delta) ** 2

    for k in range(nz):
        if (k + 1) < kpbl:
            zfac[k] = min(max(1.0 - (zq[k + 1] - zl1) / (hpbl - zl1), ZFMIN), 1.0)
            zfacent[k] = (1.0 - zfac[k]) ** 3
            wscalek[k] = (ust3 + PHIFAC * KARMAN * wstar3 * (1.0 - zfac[k])) ** H1
            wscalek2[k] = (PHIFAC * KARMAN * wstar3_2 * zfac[k]) ** H1
            if sfcflg:
                prfac = conpr
                denom = 1.0 + 4.0 * KARMAN * (wstar3 + wstar3_2) / ust3
                prfac2 = 15.9 * (wstar3 + wstar3_2) / ust3 / denom
                prnumfac = -3.0 * max(zq[k + 1] - SFCFRAC * hpbl, 0.0) ** 2 / hpbl**2
            else:
                prfac = 0.0
                prfac2 = 0.0
                prnumfac = 0.0
                phim8z = 1.0 + APHI5 * zol1 * zq[k + 1] / zl1
                wscalek[k] = max(ust / phim8z, 0.001)
            prnum0 = max(min(phih / phim + prfac, PRMAX), PRMIN)
            xkzm[k] = (
                wscalek[k] * KARMAN * zq[k + 1] * zfac[k] ** PFAC
                + wscalek2[k] * KARMAN * (hpbl - zq[k + 1]) * (1.0 - zfac[k]) ** PFAC
            )
            if (k + 1) == kpbl - 1 and cloudflg and we < 0.0:
                xkzm[k] = 0.0
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzq[k] = xkzm[k] / prnum * zfac[k] ** (PFAC_Q - PFAC)
            prnum0 = prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC)
            prnum = 1.0 + (prnum0 - 1.0) * np.exp(prnumfac)
            xkzh[k] = xkzm[k] / prnum
            xkzm[k] = min(xkzm[k] + xkzom[k], XKZMAX)
            xkzh[k] = min(xkzh[k] + xkzoh[k], XKZMAX)
            xkzq[k] = min(xkzq[k] + xkzoh[k], XKZMAX)

    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            ss = ((u[k + 1] - u[k]) ** 2 + (v[k + 1] - v[k]) ** 2) / (dza[k + 1] ** 2) + 1.0e-9
            govrthv = G / (0.5 * (thv[k + 1] + thv[k]))
            ri = govrthv * (thv[k + 1] - thv[k]) / (ss * dza[k + 1])
            zk = KARMAN * zq[k + 1]
            rlamdz = min(max(0.1 * dza[k + 1], RLAM), 300.0)
            rlamdz = min(dza[k + 1], rlamdz)
            rl2 = (zk * rlamdz / (rlamdz + zk)) ** 2
            dk = rl2 * np.sqrt(ss)
            if ri < 0.0:
                ri = max(ri, RIMIN)
                sri = np.sqrt(-ri)
                xkzm[k] = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.746 * sri))
                xkzh[k] = dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.286 * sri))
            else:
                xkzh[k] = dk / (1.0 + 5.0 * ri) ** 2
                prnum = min(1.0 + 2.1 * ri, PRMAX)
                xkzm[k] = xkzh[k] * prnum
            xkzm[k] = min(xkzm[k] + xkzom[k], XKZMAX)
            xkzh[k] = min(xkzh[k] + xkzoh[k], XKZMAX)
            xkzml[k] = xkzm[k]
            xkzhl[k] = xkzh[k]

    # Heat solve.
    rhs = np.zeros(nz, dtype=np.float64)
    rhs[0] = th[0] - 300.0 + hfx / cont / delp[0] * dt2
    lower = np.zeros(nz, dtype=np.float64)
    diag = np.zeros(nz, dtype=np.float64)
    upper = np.zeros(nz, dtype=np.float64)
    diag[0] = 1.0
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzh[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzt = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[k] / xkzh[k])
            rhs[k] = rhs[k] + dtodsd * dsdzt
            rhs[k + 1] = th[k + 1] - 300.0 - dtodsu * dsdzt
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzh[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzh[k] = np.sqrt(xkzh[k] * xkzhl[k])
            xkzh[k] = min(max(xkzh[k], xkzoh[k]), XKZMAX)
            rhs[k + 1] = th[k + 1] - 300.0
        else:
            rhs[k + 1] = th[k + 1] - 300.0
        tem1 = dsig * xkzh[k] * rdz
        dsdz2 = tem1 * rdz
        upper[k] = -dtodsd * dsdz2
        lower[k + 1] = -dtodsu * dsdz2
        diag[k] = diag[k] - upper[k]
        diag[k + 1] = 1.0 - lower[k + 1]
        exch_h[k + 1] = xkzh[k]
    theta_sol = _solve_tridiagonal(lower, diag, upper, rhs)
    theta_tend = (theta_sol - th + 300.0) * rdt

    # Water vapor solve.
    for k in range(nz - 1):
        if (k + 1) >= kpbl:
            xkzq[k] = xkzh[k]
    rhs_q = np.zeros(nz, dtype=np.float64)
    lower_q = np.zeros(nz, dtype=np.float64)
    diag_q = np.zeros(nz, dtype=np.float64)
    upper_q = np.zeros(nz, dtype=np.float64)
    diag_q[0] = 1.0
    rhs_q[0] = qv[0] + qfx * G / delp[0] * dt2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzq = tem1 * (-qfxpbl * zfacent[k] / xkzq[k])
            rhs_q[k] = rhs_q[k] + dtodsd * dsdzq
            rhs_q[k + 1] = qv[k + 1] - dtodsu * dsdzq
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzq[k] = -we * dza[kpbl - 1] * np.exp(-entfac[k])
            xkzq[k] = np.sqrt(xkzq[k] * xkzhl[k])
            xkzq[k] = min(max(xkzq[k], xkzoh[k]), XKZMAX)
            rhs_q[k + 1] = qv[k + 1]
        else:
            rhs_q[k + 1] = qv[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        dsdz2 = tem1 * rdz
        upper_q[k] = -dtodsd * dsdz2
        lower_q[k + 1] = -dtodsu * dsdz2
        diag_q[k] = diag_q[k] - upper_q[k]
        diag_q[k + 1] = 1.0 - lower_q[k + 1]
    qv_sol = _solve_tridiagonal(lower_q, diag_q, upper_q, rhs_q)
    qv_tend = (qv_sol - qv) * rdt

    # Momentum solve.
    rhs_u = np.zeros(nz, dtype=np.float64)
    rhs_v = np.zeros(nz, dtype=np.float64)
    lower_m = np.zeros(nz, dtype=np.float64)
    diag_m = np.zeros(nz, dtype=np.float64)
    upper_m = np.zeros(nz, dtype=np.float64)
    fric = ust * ust / wspd1 * rhox * G / delp[0] * dt2 * (wspd1 / wspd) ** 2
    diag_m[0] = 1.0 + fric
    rhs_u[0] = u[0] + uoce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2
    rhs_v[0] = v[0] + voce * ust * ust * rhox * G / delp[0] * dt2 / wspd1 * (wspd1 / wspd) ** 2
    for k in range(nz - 1):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p[k] - p[k + 1]
        rdz = 1.0 / dza[k + 1]
        tem1 = dsig * xkzm[k] * rdz
        if pblflg and (k + 1) < kpbl:
            dsdzu = tem1 * (-hgamu / hpbl - ufxpbl * zfacent[k] / xkzm[k])
            dsdzv = tem1 * (-hgamv / hpbl - vfxpbl * zfacent[k] / xkzm[k])
            rhs_u[k] = rhs_u[k] + dtodsd * dsdzu
            rhs_u[k + 1] = u[k + 1] - dtodsu * dsdzu
            rhs_v[k] = rhs_v[k] + dtodsd * dsdzv
            rhs_v[k + 1] = v[k + 1] - dtodsu * dsdzv
        elif pblflg and (k + 1) >= kpbl and entfac[k] < 4.6:
            xkzm[k] = prpbl * xkzh[k]
            xkzm[k] = np.sqrt(xkzm[k] * xkzml[k])
            xkzm[k] = min(max(xkzm[k], xkzom[k]), XKZMAX)
            rhs_u[k + 1] = u[k + 1]
            rhs_v[k + 1] = v[k + 1]
        else:
            rhs_u[k + 1] = u[k + 1]
            rhs_v[k + 1] = v[k + 1]
        tem1 = dsig * xkzm[k] * rdz
        dsdz2 = tem1 * rdz
        upper_m[k] = -dtodsd * dsdz2
        lower_m[k + 1] = -dtodsu * dsdz2
        diag_m[k] = diag_m[k] - upper_m[k]
        diag_m[k + 1] = 1.0 - lower_m[k + 1]
        exch_m[k + 1] = xkzm[k]

    u_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_u)
    v_sol = _solve_tridiagonal(lower_m, diag_m.copy(), upper_m, rhs_v)
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt

    tendencies = {
        "u": u_tend,
        "v": v_tend,
        "theta": theta_tend,
        "qv": qv_tend,
    }
    diagnostics = {
        "pblh": float(hpbl),
        "kpbl": int(kpbl),
        "exch_h": exch_h,
        "exch_m": exch_m,
        "wstar": float(wstar),
        "delta": float(delta),
    }
    return tendencies, diagnostics




# ---------------------------------------------------------------------------
# jit/vmap-traceable YSU column kernel: literal transcription of WRF 4.7.1
# ``physics_mmm/bl_ysu.F90:bl_ysu_run`` as called by ``module_bl_ysu.F`` from
# ``module_pbl_driver.F:1208-1251`` (lane t-ysuqc, proofs/t_ysuqc).
#
# * QC3D/QI3D enter the theta-li profile, the top-down cloud entrainment, the
#   in-cloud Richardson number (imvdif=1) and are mixed with the moisture
#   matrix (RQCBLTEN/RQIBLTEN, :1186-1220).
# * ``ysu_topdown_pblmix`` (Registry default 1) adds the theta-li PBL
#   enhancement (:733-753) and the cloud-top radiative entrainment (:840-899).
# * CTOPO is always present in the driver call: the momentum surface diagonal
#   uses the vconv/get_pblh form (:1256-1319), CTOPO2 the 10 m wind blend.
# * WRF REAL order throughout (left-to-right products, REAL exponents via powf,
#   integer exponents as products, reciprocal-pivot ``tridin_ysu``,
#   RTHBLTEN = TTEND/PI with TTEND = rate*PI, wrapper :463). In REAL4 the
#   runtime constant products are formed in single precision like gfortran.
# PBL-height searches are vectorised (first/last level masks) instead of
# per-level scans; only the three implicit solves are level scans.
# ---------------------------------------------------------------------------


class _YSUConstants:
    """Driver constants (module_model_constants) and runtime REAL products."""

    def __init__(self, dtype) -> None:
        F = np.float32 if jnp.dtype(dtype) == jnp.float32 else np.float64
        g, rd, rv, xlv, karman = F(9.81), F(287.0), F(461.6), F(2.5e6), F(0.4)
        cp = F(7.0) * rd / F(2.0)
        values = dict(
            G=g, R_D=rd, R_V=rv, XLV=xlv, KARMAN=karman, CP=cp,
            ROVCP=rd / cp, EP1=rv / rd - F(1.0), EP2=rd / rv,
            CONT=cp / g,                                  # :382
            CONPR=F(6.8) * karman * F(0.1),               # bfac*karman*sfcfrac :386
            PK=F(8.0) * karman,                           # phifac*karman
            K4=F(4.0) * karman,                           # 4.*karman :949
            GG=g * g, XLV2=xlv * xlv, CPXLV=cp / xlv, XLVCP=xlv / cp, EP2XLV=rd / rv * xlv,
            C611=F(100.0) * F(6.112), C16=F(0.2) * F(8.0),
        )
        for key, value in values.items():
            setattr(self, key, float(value))


def _thomas_scan(lower, diag, upper, rhs):
    """WRF ``tridin_ysu`` (bl_ysu.F90:1515-1583) in literal REAL order:
    reciprocal pivot ``fk``, ``au=fk*cu``, ``f=fk*(r-cl*f(k-1))``, back
    substitution ``f(k)-au(k)*f(k+1)``. ``lower[k]`` is row k's ``cl`` (the
    caller's ``al(k-1)``). ``rhs`` may carry leading right-hand-side axes that
    share ``au`` (``tridi2n`` with equal diagonals, multi-species moisture)."""
    n = rhs.shape[-1]
    fk0 = 1.0 / diag[0]
    au0 = fk0 * upper[0]
    f0 = fk0 * rhs[..., 0]

    def fwd(carry, k):
        au_prev, f_prev = carry
        fk = 1.0 / (diag[k] - lower[k] * au_prev)
        f_k = fk * (rhs[..., k] - lower[k] * f_prev)
        au_k = fk * upper[k]
        return (au_k, f_k), (au_k, f_k)

    _, (au_rest, f_rest) = jax.lax.scan(fwd, (au0, f0), jnp.arange(1, n))
    au = jnp.concatenate([au0[None], au_rest])
    f = jnp.concatenate([f0[None], f_rest])

    def bwd(x_next, k):
        x_k = f[k] - au[k] * x_next
        return x_k, x_k

    x_last = f[n - 1]
    _, x_rest = jax.lax.scan(bwd, x_last, jnp.arange(n - 2, -1, -1))
    return jnp.moveaxis(jnp.concatenate([x_rest[::-1], x_last[None]]), 0, -1)


def _seq_cumsum(x):
    """Sequential REAL running sum in WRF loop order (static unrolled adds)."""
    acc, out = None, []
    for k in range(x.shape[-1]):
        acc = x[..., k] if acc is None else acc + x[..., k]
        out.append(acc)
    return jnp.stack(out, axis=-1)


def _wrf_diag(first, upper, lower):
    """WRF ``ad`` assembly: ``ad(k+1)=1-al(k)`` then ``ad(k)=ad(k)-au(k)``."""
    base = jnp.concatenate([jnp.reshape(first, (1,)), 1.0 - lower])
    return base - jnp.concatenate([upper, jnp.zeros(1, upper.dtype)])


def _wrf_rhs(x, first, flux, dtodsd, dtodsu, active):
    """WRF rhs: ``f(k+1)=x(k+1)-dtodsu*dsdz(k)`` then ``f(k)=f(k)+dtodsd*dsdz(k)``."""
    lower_part = jnp.where(active, x[1:] - dtodsu * flux, x[1:])
    base = jnp.concatenate([jnp.reshape(first, (1,)), lower_part])
    upper_part = jnp.where(active, base[:-1] + dtodsd * flux, base[:-1])
    return jnp.concatenate([upper_part, base[-1:]])


def _bulk_search(brup_all, brcr, brstart):
    """WRF ``do k=2,klpbl; if(.not.stable) ...`` (:625-635): the first level
    with ``brup>brcr`` freezes kpbl/brup/brdn; without one kpbl=kte."""
    n = brup_all.shape[0]
    kidx = jnp.arange(n)
    hit = jnp.logical_and(kidx >= 1, brup_all > brcr)
    kc = jnp.where(jnp.any(hit), jnp.argmax(hit), n - 1).astype(jnp.int32)
    brup = brup_all[kc]
    brdn = jnp.where(kc == 1, brstart, brup_all[jnp.maximum(kc - 1, 0)])
    return kc + 1, brdn, brup


def _interp_pblh(kpbl, brdn, brup, brcr, za, zq):
    """WRF :637-648 (and :755-769, :813-827); returns hpbl, kpbl, kpbl>1."""
    brint = jnp.where(brdn >= brcr, 0.0,
                      jnp.where(brup <= brcr, 1.0, (brcr - brdn) / (brup - brdn)))
    k = jnp.clip(kpbl, 2, za.shape[0])
    hpbl = za[k - 2] + brint * (za[k - 1] - za[k - 2])
    kpbl = jnp.where(hpbl < zq[1], jnp.asarray(1, jnp.int32), kpbl).astype(jnp.int32)
    return hpbl, kpbl, kpbl > 1


def _get_pblh(thetav, qke, zw, dzq, xland):
    """WRF ``get_pblh`` (bl_ysu.F90:1586-1693) with its literal loop exits: the
    'k==kte-1' safeguard overwrites a value found at k=kte-2; a zero
    theta-v height keeps searching."""
    n = thetav.shape[0]
    j = jnp.arange(n)
    low = jnp.logical_and(j >= 1, zw[:n] <= 500.0)
    qtke = jnp.maximum(qke, 0.0)
    maxqke = jnp.max(jnp.where(low, qtke, 0.0))
    jtke = jnp.where(maxqke > 0.0, jnp.argmax(jnp.logical_and(low, qtke == maxqke)), 0)
    minthv = jnp.min(jnp.where(low, thetav, 9.0e9))
    jthv = jnp.where(jnp.any(low), jnp.argmax(jnp.logical_and(low, thetav == minthv)), 0)
    tkeeps = jnp.minimum(jnp.maximum(maxqke / 40.0, 0.025), 0.25)
    delt = jnp.where((xland - 1.5) >= 0.0, 0.75, 1.5)
    thv_m1 = jnp.concatenate([thetav[:1], thetav[:-1]])
    dz_m1 = jnp.concatenate([dzq[:1], dzq[:-1]])
    target = minthv + delt
    zi_j = zw[:n] - dz_m1 * jnp.minimum((thetav - target) / jnp.maximum(thetav - thv_m1, 1.0e-6), 1.0)
    found = jnp.logical_and(jnp.logical_and(j >= jthv + 1, j <= n - 3),
                            jnp.logical_and(thetav >= target, zi_j != 0.0))
    jz = jnp.argmax(found)
    zi = jnp.where(jnp.logical_and(jnp.any(found), jz < n - 3), zi_j[jz], zw[1])
    qh = jnp.maximum(qke / 2.0, 0.0)
    qh_m1 = jnp.concatenate([qh[:1], qh[:-1]])
    pt_j = jnp.maximum(zw[:n] - dz_m1 * jnp.minimum((tkeeps - qh) / jnp.maximum(qh_m1 - qh, 1.0e-6), 1.0), zw[1])
    found_t = jnp.logical_and(jnp.logical_and(j >= jtke + 1, j <= n - 3), qh <= tkeeps)
    jt = jnp.argmax(found_t)
    pblh_tke = jnp.where(jnp.logical_and(jnp.any(found_t), jt < n - 3), pt_j[jt], zw[1])
    wt = 0.5 * jnp.tanh((zi - 200.0) / 400.0) + 0.5
    return pblh_tke * (1.0 - wt) + zi * wt


def _ysu_column_traceable(
    u, v, tx, qv, p, pdi, pi, dz,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt, xland, u10, v10, uoce, voce,
    qc=None, qi=None, rthraten=None, ctopo=1.0, ctopo2=1.0, topdown=True, urban=None,
):
    """One WRF YSU column (``bl_ysu_run``). Inputs ``(nz,)`` (``pdi`` ``(nz+1,)``)
    or scalars. ``qc``/``qi`` None = FLAG_QC/FLAG_QI false (zero mixing ratio,
    zero rate). Returns ``(u, v, theta, qv, qc, qi)`` rates, ``exch_h``,
    ``exch_m``, ``pblh``, ``kpbl``, ``wstar``, ``delta``, ``u10``, ``v10``."""

    c = _YSUConstants(u.dtype)
    G, CP, EP1, KARMAN = c.G, c.CP, c.EP1, c.KARMAN
    n = u.shape[0]
    kidx = jnp.arange(n)
    kp1 = kidx + 1  # WRF level index k
    zeros = jnp.zeros_like(u)
    f_qc, f_qi = qc is not None, qi is not None
    qcxl = qc if f_qc else zeros  # :390-416
    qixl = qi if f_qi else zeros
    rthraten = zeros if rthraten is None else rthraten

    thx = tx / pi
    thlix = (tx - c.XLV * qcxl / CP - 2.834e6 * qixl / CP) / pi
    tvcon = 1.0 + EP1 * qv
    thvx = thx * tvcon
    rhox = psfc / (c.R_D * tx[0] * tvcon[0])
    govrth = G / thx[0]

    zq = jnp.concatenate([jnp.zeros(1, u.dtype), _seq_cumsum(dz)])  # :511-517
    rhox2 = p / (c.R_D * tx * tvcon)
    za = 0.5 * (zq[:-1] + zq[1:])
    dzq = zq[1:] - zq[:-1]
    dl = pdi[:-1] - pdi[1:]
    dza = jnp.concatenate([za[:1], za[1:] - za[:-1]])

    wspd1 = jnp.sqrt((u[0] - uoce) * (u[0] - uoce) + (v[0] - voce) * (v[0] - voce)) + 1.0e-9
    dt2 = 2.0 * dt
    rdt = 1.0 / dt2
    zl1 = za[0]
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * thx[0]
    sfcflg = jnp.logical_not(br > 0.0)

    # first guess of the PBL height (:619-649)
    spdk2 = jnp.maximum(u * u + v * v, 1.0)
    thermal = thvx[0]
    thermalli = thlix[0]
    kpbl, brdn, brup = _bulk_search((thvx - thermal) * (G * za / thvx[0]) / spdk2, BRCR_UB, br)
    hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)

    fm, fh = psim, psih  # :651-679
    zol1 = jnp.maximum(br * fm * fm / fh, RIMIN)
    zol1 = jnp.where(sfcflg, jnp.minimum(zol1, -ZFMIN), jnp.maximum(zol1, ZFMIN))
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    unstable_base = 1.0 - APHI16 * hol1
    bfx0 = jnp.maximum(sflux, 0.0)
    wstar3 = jnp.where(sfcflg, govrth * bfx0 * hpbl, 0.0)
    phim = jnp.where(sfcflg, unstable_base ** (-0.25), 1.0 + APHI5 * hol1)
    phih = jnp.where(sfcflg, unstable_base ** (-0.5), phim)
    wstar = jnp.where(sfcflg, wstar3 ** H1, 0.0)
    ust3 = ust ** 3.0
    wscale = (ust3 + c.PK * wstar3 * 0.5) ** H1
    wscale = jnp.maximum(jnp.minimum(wscale, ust * APHI16), ust / APHI5)

    # surface-driven thermal (:684-700)
    conv = jnp.logical_and(sfcflg, sflux > 0.0)
    gamfac = BFAC / rhox / wscale
    hgamt_c = jnp.minimum(gamfac * hfx / CP, GAMCRT)
    hgamq_c = jnp.minimum(gamfac * qfx, GAMCRQ)
    vpert = (hgamt_c + EP1 * thx[0] * hgamq_c) / BFAC * AFAC
    bump = jnp.maximum(vpert, 0.0) * jnp.minimum(za[0] / (SFCFRAC * hpbl), 1.0)
    thermal = jnp.where(conv, thermal + bump, thermal)
    thermalli = jnp.where(conv, thermalli + bump, thermalli)
    hgamt = jnp.where(conv, jnp.maximum(hgamt_c, 0.0), 0.0)
    brint = -(15.9 * ust * ust / wspd * wstar3 / (wscale ** 4.0))
    hgamu = jnp.where(conv, brint * u[0], 0.0)
    hgamv = jnp.where(conv, brint * v[0], 0.0)
    pblflg = jnp.logical_and(pblflg, conv)

    # enhanced PBL height with the thermal (:704-729)
    kpbl_e, brdn_e, brup_e = _bulk_search((thvx - thermal) * (G * za / thvx[0]) / spdk2, BRCR_UB, br)
    kpbl = jnp.where(pblflg, kpbl_e, kpbl)
    brdn = jnp.where(pblflg, brdn_e, brdn)
    brup = jnp.where(pblflg, brup_e, brup)

    if topdown:  # theta-li enhancement (:733-753); brcr is brcr_ub here
        bruptmp = (thlix - thermalli) * (G * za / thlix[0]) / spdk2
        active = jnp.logical_and(kp1 >= kpbl, kp1 <= n - 1)
        unst = jnp.logical_and(active, jnp.logical_not(bruptmp >= BRCR_UB))
        last = jnp.max(jnp.where(unst, kidx, -1))
        last_def = jnp.max(jnp.where(jnp.logical_and(unst, kp1 <= n - 2), kidx, -1))
        brdn = jnp.where(last >= 0, bruptmp[jnp.maximum(last, 0)], brdn)
        kpbl = jnp.where(last_def >= 0, last_def + 2, kpbl).astype(jnp.int32)
        brup = jnp.where(last_def >= 0, bruptmp[jnp.maximum(last_def + 1, 0)], brup)
        pblflg = jnp.logical_or(pblflg, last >= 0)

    hpbl_e, kpbl_ei, _ = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)  # :755-769
    hpbl = jnp.where(pblflg, hpbl_e, hpbl)
    kpbl = jnp.where(pblflg, kpbl_ei, kpbl)
    pblflg = jnp.logical_and(pblflg, kpbl > 1)

    # stable boundary layer (:773-827)
    sbl = jnp.logical_and(jnp.logical_not(sfcflg), hpbl < zq[1])
    wspd10 = jnp.sqrt(u10 * u10 + v10 * v10)
    ross = wspd10 / (CORI * znt)
    brcr_sbro = jnp.minimum(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
    brcr = jnp.where((xland - 1.5) >= 0.0, brcr_sbro, BRCR_SB)
    kpbl_s, brdn_s, brup_s = _bulk_search((thvx - thermal) * (G * za / thvx[0]) / spdk2, brcr, br)
    hpbl_s, kpbl_s, _ = _interp_pblh(kpbl_s, brdn_s, brup_s, brcr, za, zq)
    hpbl = jnp.where(sbl, hpbl_s, hpbl)
    kpbl = jnp.where(sbl, kpbl_s, kpbl)
    pblflg = jnp.where(sbl, jnp.logical_and(pblflg, kpbl > 1), pblflg)

    # entrainment parameters (:831-926); WRF k = kpbl-1
    ke = jnp.clip(kpbl - 2, 0, n - 2)
    wm3 = wstar3 + 5.0 * ust3
    wm2 = wm3 ** H2
    bfxpbl = -(0.15 * thvx[0] / G * wm3 / hpbl)
    dthvx = jnp.maximum(thvx[ke + 1] - thvx[ke], TMIN)
    we = jnp.maximum(bfxpbl / dthvx, -jnp.sqrt(wm2))
    wstar3_2 = jnp.zeros_like(wstar3)
    cloudflg = jnp.asarray(False)
    if topdown:
        cloudflg = jnp.logical_and(jnp.logical_and(pblflg, kpbl >= 2), (qcxl[ke] + qixl[ke]) > 0.01e-3)
        ke2 = jnp.minimum(ke + 2, n - 1)
        pk1 = pdi[ke + 1]
        templ = thlix[ke] * (pk1 / 100000.0) ** c.ROVCP
        rvls = c.C611 * jnp.exp(17.67 * (templ - 273.16) / (templ - 29.65)) * (c.EP2 / pk1)
        qtot = qv[ke] + qcxl[ke]
        temps = templ + (qtot - rvls) / (c.CPXLV + c.EP2XLV * rvls / (c.R_D * (templ * templ)))
        rvls = c.C611 * jnp.exp(17.67 * (temps - 273.15) / (temps - 29.65)) * (c.EP2 / pk1)
        rcldb = jnp.maximum(qtot - rvls, 0.0)
        dthvx_c = ((thlix[ke2] + thx[ke2] * EP1 * (qv[ke2] + qcxl[ke2]))
                   - (thlix[ke] + thx[ke] * EP1 * (qv[ke] + qcxl[ke])))
        dthvx_c = jnp.maximum(dthvx_c, 0.1)
        tmp1 = c.XLVCP * rcldb / (pi[ke] * dthvx_c)
        ent_eff = c.C16 * tmp1 + 0.2
        radflux = rthraten * pi
        radflux = radflux * CP / G * dl
        radsum = _seq_cumsum(jnp.where(jnp.logical_and(kp1 <= kpbl - 1, radflux < 0.0),
                                       jnp.abs(radflux), 0.0))[-1]
        radsum = jnp.maximum(radsum, 0.0)
        wm3_c = govrth * bfx0 * hpbl + 5.0 * ust3
        wm2_c = wm3_c ** H2
        we_c = jnp.maximum(-(0.15 * thvx[0] / G * wm3_c / hpbl) / dthvx, -jnp.sqrt(wm2_c))
        bfx0_r = jnp.maximum(radsum / rhox2[ke] / CP, 0.0)
        wm3_r = G / thvx[ke] * bfx0_r * hpbl
        wm2_c = wm2_c + wm3_r ** H2
        dthvx_r = jnp.maximum(thvx[ke + 1] - thvx[ke], 0.1)
        we_c = we_c + jnp.maximum(-(ent_eff * bfx0_r) / dthvx_r, -jnp.sqrt(wm3_r ** H2))
        wstar3_2c = G / thvx[ke] * bfx0_r * hpbl
        wscale_c = (ust3 + c.PK * (wstar3 + wstar3_2c) * 0.5) ** H1
        wscale_c = jnp.maximum(jnp.minimum(wscale_c, ust * APHI16), ust / APHI5)
        hgamt_cc = jnp.minimum(BFAC / rhox / wscale_c * hfx / CP, GAMCRT)
        hgamt2 = jnp.minimum(BFAC / rhox2[ke] / wscale_c * radsum / CP, GAMCRT)
        hgamt_cc = jnp.maximum(hgamt_cc, 0.0) + jnp.maximum(hgamt2, 0.0)
        brint_c = -(15.9 * ust * ust / wspd * (wstar3 + wstar3_2c) / (wscale_c ** 4.0))
        wm2 = jnp.where(cloudflg, wm2_c, wm2)
        we = jnp.where(cloudflg, we_c, we)
        wstar3_2 = jnp.where(cloudflg, wstar3_2c, wstar3_2)
        hgamt = jnp.where(cloudflg, hgamt_cc, hgamt)
        hgamu = jnp.where(cloudflg, brint_c * u[0], hgamu)
        hgamv = jnp.where(cloudflg, brint_c * v[0], hgamv)
    dthx = jnp.maximum(thx[ke + 1] - thx[ke], TMIN)
    dqx = jnp.minimum(qv[ke + 1] - qv[ke], 0.0)
    hfxpbl = jnp.where(pblflg, we * dthx, 0.0)
    qfxpbl = jnp.where(pblflg, we * dqx, 0.0)
    dux = u[ke + 1] - u[ke]
    dvx = v[ke + 1] - v[ke]
    ust2 = ust * ust

    def _flux(d):  # prpbl = 1.0 (:909-922)
        return jnp.where(d > TMIN, jnp.maximum(we * d, -ust2),
                         jnp.where(d < -TMIN, jnp.minimum(we * d, ust2), 0.0))

    ufxpbl = jnp.where(pblflg, _flux(dux), 0.0)
    vfxpbl = jnp.where(pblflg, _flux(dvx), 0.0)
    delb = govrth * D3 * hpbl
    delta = jnp.where(pblflg, jnp.minimum(D1 * hpbl + D2 * wm2 / delb, 100.0), 0.0)
    we = jnp.where(pblflg, we, 0.0)
    cloudflg = jnp.logical_and(cloudflg, pblflg)

    entfac = jnp.where(jnp.logical_and(pblflg, kp1 >= kpbl),
                       ((zq[1:] - hpbl) / jnp.where(pblflg, delta, 1.0)) ** 2.0, 1.0e30)

    # diffusion coefficients below the PBL (:940-980)
    zfac = jnp.minimum(jnp.maximum(1.0 - (zq[1:] - zl1) / (hpbl - zl1), ZFMIN), 1.0)
    zfacent = (1.0 - zfac) ** 3.0
    wscalek = (ust3 + c.PK * wstar3 * (1.0 - zfac)) ** H1
    wscalek2 = (c.PK * wstar3_2 * zfac) ** H1
    w32 = wstar3 + wstar3_2
    prfac = jnp.where(sfcflg, c.CONPR, 0.0)
    prfac2 = jnp.where(sfcflg, 15.9 * w32 / ust3 / (1.0 + c.K4 * w32 / ust3), 0.0)
    prnumfac = jnp.where(sfcflg, -(3.0 * jnp.maximum(zq[1:] - SFCFRAC * hpbl, 0.0) ** 2.0 / hpbl ** 2.0), 0.0)
    phim8z = 1.0 + APHI5 * zol1 * zq[1:] / zl1
    wscalek = jnp.where(sfcflg, wscalek, jnp.maximum(ust / phim8z, 0.001))
    prnum0 = jnp.maximum(jnp.minimum(phih / phim + prfac, PRMAX), PRMIN)
    xkzm_p = (wscalek * KARMAN * zq[1:] * zfac ** PFAC
              + wscalek2 * KARMAN * (hpbl - zq[1:]) * (1.0 - zfac) ** PFAC)
    xkzm_p = jnp.where(jnp.logical_and(jnp.logical_and(kp1 == kpbl - 1, cloudflg), we < 0.0), 0.0, xkzm_p)
    prnum = 1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)
    xkzq_p = xkzm_p / prnum * zfac ** (PFAC_Q - PFAC)
    prnum0 = prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC)
    prnum = 1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)
    xkzh_p = xkzm_p / prnum
    xkzm_p = jnp.minimum(xkzm_p + XKZMINM, XKZMAX)
    xkzh_p = jnp.minimum(xkzh_p + XKZMINH, XKZMAX)
    xkzq_p = jnp.minimum(xkzq_p + XKZMINH, XKZMAX)

    # free atmosphere (:984-1030), faces k = 1..kte-1
    du = u[1:] - u[:-1]
    dv = v[1:] - v[:-1]
    dza1 = dza[1:]
    ss = (du * du + dv * dv) / (dza1 * dza1) + 1.0e-9
    govrthv = G / (0.5 * (thvx[1:] + thvx[:-1]))
    ri = govrthv * (thvx[1:] - thvx[:-1]) / (ss * dza1)
    cloud = (qcxl + qixl) > 0.01e-3
    qmean = 0.5 * (qv[:-1] + qv[1:])
    tmean = 0.5 * (tx[:-1] + tx[1:])
    alph = c.XLV * qmean / c.R_D / tmean
    chi = c.XLV2 * qmean / CP / c.R_V / tmean / tmean
    ri_cloud = (1.0 + alph) * (ri - c.GG / ss / tmean / CP * ((chi - alph) / (1.0 + chi)))
    ri = jnp.where(jnp.logical_and(cloud[:-1], cloud[1:]), ri_cloud, ri)
    zk = KARMAN * zq[1:n]
    rlamdz = jnp.minimum(dza1, jnp.minimum(jnp.maximum(0.1 * dza1, RLAM), 300.0))
    rl2 = zk * rlamdz / (rlamdz + zk)
    dk = rl2 * rl2 * jnp.sqrt(ss)
    rin = jnp.maximum(ri, RIMIN)
    sri = jnp.sqrt(jnp.where(ri < 0.0, -rin, 0.0))
    t5 = 1.0 + 5.0 * ri
    xkzh_s = dk / (t5 * t5)
    xkzm_l = jnp.where(ri < 0.0, dk * (1.0 + 8.0 * (-rin) / (1.0 + 1.746 * sri)),
                       xkzh_s * jnp.minimum(1.0 + 2.1 * ri, PRMAX))
    xkzh_l = jnp.where(ri < 0.0, dk * (1.0 + 8.0 * (-rin) / (1.0 + 1.286 * sri)), xkzh_s)
    xkzml = jnp.minimum(xkzm_l + XKZMINM, XKZMAX)
    xkzhl = jnp.minimum(xkzh_l + XKZMINH, XKZMAX)

    face = kidx[:-1] + 1  # WRF k of face k..k+1
    in_pbl = face < kpbl
    xkzh = jnp.where(in_pbl, xkzh_p[:-1], xkzhl)
    xkzm = jnp.where(in_pbl, xkzm_p[:-1], xkzml)
    pbl_face = jnp.logical_and(pblflg, in_pbl)
    ent_face = jnp.logical_and(jnp.logical_and(pblflg, jnp.logical_not(in_pbl)), entfac[:-1] < 4.6)
    ent_k = -we * dza[jnp.clip(kpbl - 1, 0, n - 1)] * jnp.exp(-entfac[:-1])
    zfe = zfacent[:-1]

    if urban is not None:
        sfk = urban.surface[:-2]
        dtodsd = sfk * dt2 / dl[:-1]
        dtodsu = sfk * dt2 / dl[1:]
        vlk = urban.volume[:-1]
        bep = 1.0
    else:
        dtodsd = dt2 / dl[:-1]
        dtodsu = dt2 / dl[1:]
        vlk = None
        bep = 0.0
    dsig = p[:-1] - p[1:]
    rdz = 1.0 / dza1

    def _coefficients(kz):
        tem1 = dsig * kz * rdz
        dsdz2 = tem1 * rdz
        au = -(dtodsd * dsdz2)
        al = -(dtodsu * dsdz2)
        if vlk is not None:
            au, al = au / vlk, al / vlk
        return au, al

    # heat (:1034-1112)
    dsdzt = dsig * xkzh * rdz * (-hgamt / hpbl - hfxpbl * zfe / xkzh)
    xkzh_e = jnp.minimum(jnp.maximum(jnp.sqrt(ent_k * xkzhl), XKZMINH), XKZMAX)
    xkzh = jnp.where(ent_face, xkzh_e, xkzh)
    au, al = _coefficients(xkzh)
    ad = _wrf_diag(1.0, au, al)
    first = thx[0] - 300.0 + (1.0 - bep) * hfx / c.CONT / dl[0] * dt2
    f1 = _wrf_rhs(thx - 300.0, first, dsdzt, dtodsd, dtodsu, pbl_face)
    if urban is not None:
        ad = ad - urban.a_t * dt2
        f1 = f1 + urban.b_t * dt2
    lower = jnp.concatenate([jnp.zeros(1, al.dtype), al])
    upper = jnp.concatenate([au, jnp.zeros(1, au.dtype)])
    theta_sol = _thomas_scan(lower, ad, upper, f1)
    ttend = (theta_sol - thx + 300.0) * rdt * pi
    theta_tend = ttend / pi
    exch_h = jnp.concatenate([jnp.zeros(1, xkzh.dtype), xkzh])

    # water vapour, cloud water, cloud ice (:1115-1220)
    xkzq = jnp.where(in_pbl, xkzq_p[:-1], xkzh)
    dsdzq = dsig * xkzq * rdz * (-qfxpbl * zfe / xkzq)
    au, al = _coefficients(xkzq)
    ad = _wrf_diag(1.0, au, al)
    first = qv[0] + (1.0 - bep) * qfx * G / dl[0] * dt2
    rhs_qv = _wrf_rhs(qv, first, dsdzq, dtodsd, dtodsu, pbl_face)
    lower = jnp.concatenate([jnp.zeros(1, al.dtype), al])
    upper = jnp.concatenate([au, jnp.zeros(1, au.dtype)])
    if urban is not None:
        qv_sol = _thomas_scan(lower, ad - urban.a_q * dt2, upper, rhs_qv + urban.b_q * dt2)
        qc_sol, qi_sol = _thomas_scan(lower, ad, upper, jnp.stack([qcxl, qixl]))
    else:
        qv_sol, qc_sol, qi_sol = _thomas_scan(lower, ad, upper, jnp.stack([rhs_qv, qcxl, qixl]))
    qv_tend = (qv_sol - qv) * rdt
    qc_tend = (qc_sol - qcxl) * rdt if f_qc else zeros
    qi_tend = (qi_sol - qixl) * rdt if f_qi else zeros

    # momentum (:1241-1399), TKE diagnostic for vconv (:1256-1304)
    shear = xkzm * ((-hgamu / hpbl + du / dza1) * du / dza1 + (-hgamv / hpbl + dv / dza1) * dv / dza1)
    buoy = xkzh * G * (1.0 / thx[:-1]) * (-hgamt / hpbl + (thx[1:] - thx[:-1]) / dza1)
    rlam_t = jnp.where(in_pbl, 150.0, rlamdz)
    el_ysu = zk * rlam_t / (rlam_t + zk)
    tke = 16.6 * el_ysu * (shear - buoy)
    tke = jnp.where(tke <= 0.0, 0.0, jnp.where(tke > 0.0, tke, 1.0) ** 0.66)
    tke = jnp.concatenate([tke, jnp.zeros(1, tke.dtype)])  # tke_ysu(kte) is never read
    pblh_ysu = _get_pblh(thvx, tke, zq, dzq, xland)
    vconv = jnp.where(xland < 1.5, (G / thvx[0] * pblh_ysu * jnp.maximum(sflux, 0.0)) ** 0.33, 0.0)
    ratio = wspd1 / wspd
    fric = ust * ust / wspd1 * rhox * G / dl[0] * dt2 * (ratio * ratio)
    vconvlim = jnp.minimum(0.9 * vconv + 1.5 * jnp.maximum((pblh_ysu - 500.0) / 1000.0, 0.0), 1.0)
    first_ad = 1.0 + fric * vconvlim + ctopo * fric * (1.0 - vconvlim)
    if urban is not None:
        first_ad = first_ad - bep * urban.fraction * (fric * vconvlim + ctopo * fric * (1.0 - vconvlim))
    dsdzu = dsig * xkzm * rdz * (-hgamu / hpbl - ufxpbl * zfe / xkzm)
    dsdzv = dsig * xkzm * rdz * (-hgamv / hpbl - vfxpbl * zfe / xkzm)
    xkzm_e = jnp.minimum(jnp.maximum(jnp.sqrt(xkzh * xkzml), XKZMINM), XKZMAX)
    xkzm = jnp.where(ent_face, xkzm_e, xkzm)
    au, al = _coefficients(xkzm)
    ad = _wrf_diag(first_ad, au, al)
    ocean = lambda cur: cur * (ust * ust) * rhox * G / dl[0] * dt2 / wspd1 * (ratio * ratio)
    rhs_u = _wrf_rhs(u, u[0] + ocean(uoce), dsdzu, dtodsd, dtodsu, pbl_face)
    rhs_v = _wrf_rhs(v, v[0] + ocean(voce), dsdzv, dtodsd, dtodsu, pbl_face)
    lower = jnp.concatenate([jnp.zeros(1, al.dtype), al])
    upper = jnp.concatenate([au, jnp.zeros(1, au.dtype)])
    if urban is not None:
        from gpuwrf.physics.urban_bep_pbl import ysu_tridi2n
        u_sol, v_sol = ysu_tridi2n(lower, ad - urban.a_u * dt2, ad - urban.a_v * dt2, upper,
                                   rhs_u + urban.b_u * dt2, rhs_v + urban.b_v * dt2)
    else:
        u_sol, v_sol = _thomas_scan(lower, ad, upper, jnp.stack([rhs_u, rhs_v]))
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt
    exch_m = jnp.concatenate([jnp.zeros(1, xkzm.dtype), xkzm])
    u10_out = ctopo2 * u10 + (1.0 - ctopo2) * u[0]
    v10_out = ctopo2 * v10 + (1.0 - ctopo2) * v[0]
    return (u_tend, v_tend, theta_tend, qv_tend, qc_tend, qi_tend, exch_h, exch_m,
            hpbl, kpbl.astype(jnp.int32), wstar, delta, u10_out, v10_out)


_YSU_OUT_KEYS = ("u", "v", "theta", "qv", "qc", "qi", "exch_h", "exch_m",
                 "pblh", "kpbl", "wstar", "delta", "u10", "v10")


def ysu_columns(
    u, v, temperature, qv, pressure, pressure_interface, exner, dz,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt, xland, u10, v10, uoce=None, voce=None,
    qc=None, qi=None, rthraten=None, ysu_topdown_pblmix=1, ctopo=1.0, ctopo2=1.0,
    urban=None, sf_urban_physics=0,
):
    """Batched, jit/vmap-traceable YSU over ``(ncol,)`` columns.

    Profile inputs are ``(ncol, nz)`` (``pressure_interface`` ``(ncol, nz+1)``);
    surface inputs ``(ncol,)``. ``qc``/``qi`` are QC3D/QI3D (None = FLAG false,
    no rate returned), ``rthraten`` the held radiative theta tendency,
    ``ysu_topdown_pblmix`` the namelist switch (Registry default 1). Returns
    ``(ncol, nz)`` rates (``theta`` = RTHBLTEN, dry) and diagnostics."""

    if sf_urban_physics in (2, 3):
        if urban is None:
            raise ValueError('BEP/BEM YSU requires urban multilayer sources')
        from gpuwrf.physics.urban_bep_pbl import ysu_urban_columns
        kwargs = dict(psfc=psfc, znt=znt, ust=ust, hfx=hfx, qfx=qfx, wspd=wspd, br=br, psim=psim, psih=psih,
                      dt=dt, xland=xland, u10=u10, v10=v10, uoce=uoce, voce=voce, urban=urban, qc=qc, qi=qi,
                      rthraten=rthraten, ysu_topdown_pblmix=ysu_topdown_pblmix, ctopo=ctopo, ctopo2=ctopo2)
        if u.dtype == jnp.float32:
            # WRF REAL urban path must also remain REAL inside a wide parent
            # trace; implicit zeros in the retained column core follow scope.
            with jax.enable_x64(False):
                return ysu_urban_columns(u, v, temperature, qv, pressure, pressure_interface, exner, dz, **kwargs)
        return ysu_urban_columns(u, v, temperature, qv, pressure, pressure_interface, exner, dz, **kwargs)
    if urban is not None:
        raise ValueError('YSU urban sources require sf_urban_physics=2 or 3')
    ncol = u.shape[0]
    dtype = u.dtype
    zcol = jnp.zeros(ncol, dtype=dtype)
    uoce = zcol if uoce is None else uoce
    voce = zcol if voce is None else voce
    rthraten = jnp.zeros_like(u) if rthraten is None else rthraten
    col = lambda value: jnp.broadcast_to(jnp.asarray(value, dtype), (ncol,))
    f_qc, f_qi = qc is not None, qi is not None
    topdown = int(ysu_topdown_pblmix) == 1

    def column(u_, v_, t_, qv_, p_, pdi_, pi_, dz_, qc_, qi_, rth_, psfc_, znt_, ust_, hfx_, qfx_, wspd_,
               br_, psim_, psih_, dt_, xland_, u10_, v10_, uoce_, voce_, ctopo_, ctopo2_):
        return _ysu_column_traceable(
            u_, v_, t_, qv_, p_, pdi_, pi_, dz_, psfc=psfc_, znt=znt_, ust=ust_, hfx=hfx_, qfx=qfx_,
            wspd=wspd_, br=br_, psim=psim_, psih=psih_, dt=dt_, xland=xland_, u10=u10_, v10=v10_,
            uoce=uoce_, voce=voce_, qc=qc_ if f_qc else None, qi=qi_ if f_qi else None, rthraten=rth_,
            ctopo=ctopo_, ctopo2=ctopo2_, topdown=topdown)

    zeros2d = jnp.zeros_like(u)
    out = jax.vmap(column)(
        u, v, temperature, qv, pressure, pressure_interface, exner, dz,
        qc if f_qc else zeros2d, qi if f_qi else zeros2d, rthraten,
        psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, col(dt), xland, u10, v10, uoce, voce,
        col(ctopo), col(ctopo2),
    )
    result = dict(zip(_YSU_OUT_KEYS, out))
    if not f_qc:
        result.pop("qc")
    if not f_qi:
        result.pop("qi")
    return result


def step_ysu_column(
    u, v, temperature, qv, pressure, pressure_interface, exner, dz,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt,
    xland=1.0, u10=0.0, v10=0.0, uoce=0.0, voce=0.0,
    qc=None, qi=None, rthraten=None, ysu_topdown_pblmix=1,
) -> PhysicsStepResult:
    """Run one WRF-YSU column and return frozen S0 tendency/diagnostics.

    Inputs follow the WRF ``module_bl_ysu.F`` wrapper naming: ``temperature`` is
    absolute temperature, ``exner`` is ``pi3d``, ``pressure_interface`` is the
    length ``nz+1`` pressure-at-interface column, and surface fluxes are the
    already-computed surface-layer values consumed by YSU.
    """

    f64 = lambda value: None if value is None else jnp.asarray(value, jnp.float64)
    out = _ysu_column_traceable(
        f64(u), f64(v), f64(temperature), f64(qv), f64(pressure), f64(pressure_interface), f64(exner), f64(dz),
        psfc=f64(psfc), znt=f64(znt), ust=f64(ust), hfx=f64(hfx), qfx=f64(qfx), wspd=f64(wspd), br=f64(br),
        psim=f64(psim), psih=f64(psih), dt=f64(dt), xland=f64(xland), u10=f64(u10), v10=f64(v10),
        uoce=f64(uoce), voce=f64(voce), qc=f64(qc), qi=f64(qi), rthraten=f64(rthraten),
        topdown=int(ysu_topdown_pblmix) == 1,
    )
    (u_t, v_t, th_t, qv_t, qc_t, qi_t, exch_h, exch_m, pblh, kpbl, wstar, delta, _u10, _v10) = out
    tendencies = {"u": u_t, "v": v_t, "theta": th_t, "qv": qv_t}
    if qc is not None:
        tendencies["qc"] = qc_t
    if qi is not None:
        tendencies["qi"] = qi_t
    diagnostics = {
        "pblh": pblh,
        "kpbl": kpbl.astype(jnp.int32),
        "exch_h": exch_h,
        "exch_m": exch_m,
        "wstar": wstar,
        "delta": delta,
    }
    tendency = PhysicsTendency(state_tendencies=tendencies)
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        diagnostics=PhysicsDiagnostics(pbl=diagnostics),
    )


__all__ = ["YSUColumnState", "YSUDiagnostics", "step_ysu_column", "ysu_columns"]
