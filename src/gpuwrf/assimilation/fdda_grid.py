"""WRF grid analysis nudging (``grid_fdda = 1``, PSU ``fddagd``) -- literal port.

Sources (pristine WRF v4, ``<USER_HOME>/src/wrf_pristine/WRF``):

* ``phys/module_fddagd_driver.F`` -- driver control: at ``itimestep == 1`` the OLD
  analysis is overwritten by the model state; ``fgdtzero`` zeroing on
  ``mod(itimestep-1, STEPFG) == 0``; recomputation on ``itimestep == 1 .or.
  mod(itimestep, STEPFG) == 0`` (tendencies reset to 0 first).
* ``phys/module_fdda_psufddagd.F:31-690`` (``fddagd``) -- time interpolation
  between the bracketing analyses, PBL weights ``wpbl``, level weights ``wzfac``,
  ramping ``tfac`` and the uncoupled tendencies RUNDGDTEN/RVNDGDTEN/RTHNDGDTEN/
  RQVNDGDTEN (m s-2, K s-1, kg kg-1 s-1).
* ``dyn_em/module_em.F:2420-2470`` (``calculate_phy_tend``) -- in-place coupling
  with the step-entry masses ``c1h*muu+c2h`` / ``c1h*muv+c2h`` / ``c1h*mut+c2h``.
* ``phys/module_physics_addtendc.F`` (``phy_fg_ten``: ``add_a2a``, ``add_c2c_u``,
  ``add_c2c_v``) -- specified/nested domains skip the outer ring.
* ``dyn_em/module_big_step_utilities_em.F:5338-5390`` (``phy_prep_part2``) -- the
  held rates are decoupled with the end-of-RK masses ``muts``/``muus``/``muvs``,
  so between recomputations the COUPLED rate persists (like KF/RTHRATEN).

Arithmetic follows the Fortran expression order (left to right, no reassociation)
in the requested float dtype: ``float32`` = WRF REAL (RWORDSIZE=4), ``float64`` =
the ``-fdefault-real-8`` build. Spectral nudging (``grid_fdda = 2``) lives in
``fdda_spectral``; station observation nudging lives in ``obs_driver``.
Surface FDDA (``grid_sfdda``) and its FASDAS atmospheric branch are implemented
in ``fdda_surface`` and selected by a resident surface-input subtype.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


#: Pristine phys/module_fdda_psufddagd.F constants (Fortran REAL literals).
_THETA_OFFSET = 300.0  # ``th3d - 300.0`` / ``val_analysis - th3d + 300.0``
_WPBL_RAMP = 0.1  # ``wpbl = 0.1`` / ``wzfac = 0.1`` on the first level above the cut
_HALF = 0.5


@dataclass(frozen=True)
class GridFddaOptions:
    """Static ``&fdda`` options of one domain (hashable -> part of the trace key).

    ``interval_m`` and ``end_h`` are WRF ``auxinput10_interval_m`` /
    ``auxinput10_end_h`` (= ``gfdda_interval_m`` / ``gfdda_end_h`` through
    ``dyn_em/namelist_remappings_em.h``).  ``end_h_sfc`` is ``auxinput9_end_h``
    (= ``sgfdda_end_h``); it enters the early-return test even without surface
    nudging (``module_fdda_psufddagd.F:236-244``).
    """

    guv: float = 0.0
    gt: float = 0.0
    gq: float = 0.0
    interval_m: int = 0
    end_h: int = 0
    end_h_sfc: int = 0
    stepfg: int = 1
    fgdtzero: int = 0
    if_no_pbl_nudging_uv: int = 0
    if_no_pbl_nudging_t: int = 0
    if_no_pbl_nudging_q: int = 0
    if_zfac_uv: int = 0
    k_zfac_uv: int = 0
    if_zfac_t: int = 0
    k_zfac_t: int = 0
    if_zfac_q: int = 0
    k_zfac_q: int = 0
    if_ramping: int = 0
    dtramp_min: float = 0.0

    def __post_init__(self) -> None:
        if int(self.interval_m) <= 0:
            raise ValueError("In grid FDDA, gfdda_interval_m must be > 0")  # psufddagd.F:255
        for name in ("guv", "gt", "gq"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"In grid FDDA, {name} must be positive.")  # fddagdinit
        if int(self.stepfg) < 1:
            raise ValueError("STEPFG must be >= 1 (physics_init: max(nint(fgdt*60/dt),1))")


def stepfg_from_fgdt(fgdt_min: float, dt_s: float) -> int:
    """``STEPFG = max(nint(FGDT*60./DT), 1)`` (phys/module_physics_init.F:4751)."""

    value = np.float32(np.float32(fgdt_min) * np.float32(60.0)) / np.float32(dt_s)
    # Fortran NINT rounds half away from zero.
    rounded = int(np.floor(abs(float(value)) + 0.5)) * (1 if value >= 0 else -1)
    return max(rounded, 1)


FDDA_FIELDS = ("u", "v", "t", "q")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GridFddaInputs:
    """Resident ``wrffdda_d0N`` records + options (``OperationalNamelist.data_assimilation``).

    ``*_old``/``*_new`` are stacked over the records read by the run: record ``r``
    (``*_NDG_OLD``/``*_NDG_NEW``) is the auxinput10 read at the entry of the
    zero-based step ``record_steps[r]`` (first step at/after its alarm time).  u: (R, nz, ny, nx+1), v: (R, nz, ny+1, nx), t/q: (R, nz, ny, nx);
    ``t`` is the dry potential-temperature perturbation ``th_phy - 300``.
    """

    u_old: Any
    u_new: Any
    v_old: Any
    v_new: Any
    t_old: Any
    t_new: Any
    q_old: Any
    q_new: Any
    options: GridFddaOptions = GridFddaOptions(interval_m=1)
    record_steps: tuple[int, ...] = ()

    def tree_flatten(self):
        children = tuple(getattr(self, f"{n}_{s}") for n in FDDA_FIELDS for s in ("old", "new"))
        return children, (self.options, tuple(int(m) for m in self.record_steps))

    @classmethod
    def tree_unflatten(cls, aux, children):
        names = [f"{n}_{s}" for n in FDDA_FIELDS for s in ("old", "new")]
        return cls(**dict(zip(names, children)), options=aux[0], record_steps=aux[1])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GridFddaCarry:
    """Per-domain FDDA carry (WRF restart fields fdda3d *_NDG_OLD/NEW + R*NDGDTEN).

    ``*_old``/``*_new`` = the current analysis pair (the auxinput10 cursor).  The
    R*NDGDTEN leaves are UNCOUPLED at step boundaries (after ``phy_prep_part2``);
    during a step :func:`fdda_step_coupled` stores the coupled copy and
    :func:`finish_fdda_rates` decouples it with the final masses.
    """

    u_old: Any
    u_new: Any
    v_old: Any
    v_new: Any
    t_old: Any
    t_new: Any
    q_old: Any
    q_new: Any
    rundgdten: Any
    rvndgdten: Any
    rthndgdten: Any
    rqvndgdten: Any

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    def replace(self, **updates) -> "GridFddaCarry":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(updates)
        return type(self)(**values)

    @property
    def tendencies(self) -> tuple:
        return (self.rundgdten, self.rvndgdten, self.rthndgdten, self.rqvndgdten)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GridFddaHistoryCarry(GridFddaCarry):
    """PSU grid-only Registry history, without changing spectral carry schema."""

    hfx_fdda: Any = None


def initial_fdda_carry(inputs: GridFddaInputs, dtype, *, with_history=False) -> GridFddaCarry:
    """fddagdinit (non-restart): R*NDGDTEN = 0; pair = record 0 (read before step 1)."""

    first = {f"{n}_{s}": jnp.asarray(getattr(inputs, f"{n}_{s}")[0], dtype)
             for n in FDDA_FIELDS for s in ("old", "new")}
    zeros = dict(
        rundgdten=jnp.zeros_like(first["u_old"]),
        rvndgdten=jnp.zeros_like(first["v_old"]),
        rthndgdten=jnp.zeros_like(first["t_old"]),
        rqvndgdten=jnp.zeros_like(first["q_old"]),
    )
    base = GridFddaHistoryCarry(**first, **zeros, hfx_fdda=jnp.zeros_like(first["t_old"])) if with_history else GridFddaCarry(**first, **zeros)
    from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaInputs, seed_surface_carry
    return seed_surface_carry(inputs, base, dtype) if isinstance(inputs, SurfaceGridFddaInputs) else base


# --------------------------------------------------------------------------- #
# WRF clocks (REAL)                                                           #
# --------------------------------------------------------------------------- #
def _wrf_div(a, b, dtype):
    """Fortran ``a/b`` in ``dtype`` (REAL or REAL(8)) with a fenced divisor.

    XLA's algebraic simplifier rewrites a division by a compile-time constant into a
    multiplication by its rounded reciprocal (FINDINGS E123/E130: ``(240-180)/60``
    became 0.99999982, and ``FLOOR(curr_mins2/interval)`` can drop an interval at an
    exact analysis time).  The optimization barrier keeps a true division, which on
    CPU is the IEEE quotient gfortran forms; XLA GPU lowers f32 division to
    ``div.full.f32`` (<= 2 ulp, E84) like every other REAL division of the port.
    """

    return jnp.asarray(a, dtype) / jax.lax.optimization_barrier(jnp.asarray(b, dtype))


def wrf_minutes(lead_seconds, dtype):
    """``(xtime, curr_mins2)`` of WRF for a run that starts at the simulation start.

    xtime = REAL(days)*24.*60. + REAL(seconds)/60. (+ fraction) from WRF's integer clock
    (frame/module_domain.F:2186), REAL arithmetic; curr_mins2 = REAL(curr_secs2_r8/60.0d0)
    (dyn_em/solve_em.F:339) is WRF's only DOUBLE island here: one f64 quotient rounded to REAL.
    """

    lead = jnp.asarray(lead_seconds, jnp.float64)  # the port's step clock (seconds)
    total = lead.astype(jnp.int32)  # integer clock seconds (lead >= 0: truncation == floor)
    days = total // 86400
    seconds = total - days * 86400
    one = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    xtime = (one(days) * one(24.0)) * one(60.0) + _wrf_div(one(seconds), 60.0, dtype)
    # Fractional-second time steps only (Sn/Sd of the WRF clock); exactly 0 for integer dt.
    frac = jnp.asarray(lead - total.astype(jnp.float64), dtype)
    xtime = jnp.where(frac > 0, xtime + _wrf_div(frac, 60.0, dtype), xtime)
    curr_mins2 = jnp.asarray(lead / jax.lax.optimization_barrier(jnp.asarray(60.0, jnp.float64)), dtype)
    return xtime, curr_mins2


def fdda_time_factors(options: GridFddaOptions, lead_seconds, dtype):
    """``(early_off, coef, tfac)`` of ``fddagd`` at this step (psufddagd.F:225-258, 491-547)."""

    one = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    xtime, curr_mins2 = wrf_minutes(lead_seconds, dtype)
    end_h, end_h_sfc = int(options.end_h), int(options.end_h_sfc)
    dtramp = one(options.dtramp_min)
    actual_end = one(end_h) * one(60.0)
    if int(options.if_ramping) == 1 and float(options.dtramp_min) > 0.0:
        actual_end = one(end_h) * one(60.0) + jnp.abs(dtramp)
    # Single-line IF quirk: the sfc end ALWAYS adds |dtramp_min| (psufddagd.F:244).
    actual_end_sfc = one(end_h_sfc) * one(60.0) + jnp.abs(dtramp)
    early_off = (xtime > actual_end) & (xtime > actual_end_sfc)

    interval = int(options.interval_m)
    xtime_old = one(jnp.floor(_wrf_div(curr_mins2, one(interval), dtype)) * one(interval)) * one(1.0)
    xtime_new = xtime_old + one(interval)
    coef = _wrf_div(curr_mins2 - xtime_old, xtime_new - xtime_old, dtype)  # int4 == 1

    tfac = one(1.0)
    if int(options.if_ramping) == 1 and abs(float(options.dtramp_min)) > 0.0:
        if float(options.dtramp_min) <= 0.0:
            actual_end_r = one(end_h) * one(60.0)
        else:
            actual_end_r = one(end_h) * one(60.0) + dtramp
        absd = jnp.abs(dtramp)
        before = xtime < actual_end_r - absd
        inside = (xtime >= actual_end_r - absd) & (xtime <= actual_end_r)
        tfac = jnp.where(before, one(1.0), jnp.where(inside, _wrf_div(actual_end_r - xtime, absd, dtype), one(0.0)))
        if float(options.dtramp_min) > 0.0:
            coef = jnp.where(inside, one(0.0), coef)
    return early_off, coef, jnp.asarray(tfac, dtype)


# --------------------------------------------------------------------------- #
# Weights                                                                     #
# --------------------------------------------------------------------------- #
def _kpbl_weights(zpbl, zagl_w, dtype):
    """wpbl(k) for one column family: zagl_w (nz+1, ...) heights AGL at w levels."""

    one = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    bot, top = zagl_w[:-1], zagl_w[1:]
    hit = (zpbl[None] >= bot) & (zpbl[None] < top)
    nz = bot.shape[0]
    k1 = jnp.arange(1, nz + 1, dtype=jnp.int32).reshape((nz,) + (1,) * (bot.ndim - 1))
    first = jnp.argmax(hit, axis=0).astype(jnp.int32) + 1
    kpbl = jnp.where(jnp.any(hit, axis=0), first, 1)
    return jnp.where(k1 <= kpbl[None], one(0.0), jnp.where(k1 == kpbl[None] + 1, one(_WPBL_RAMP), one(1.0)))


def _wzfac(nz: int, enabled: int, k_zfac: int, dtype):
    k1 = np.arange(1, nz + 1)
    w = np.where(k1 <= k_zfac, 0.0, np.where(k1 == k_zfac + 1, _WPBL_RAMP, 1.0)) if enabled == 1 else np.ones(nz)
    return jnp.asarray(w.astype(np.dtype(dtype)), dtype)


def _face_avg_x(a):
    """``0.5 * ( a(i-1) + a(i) )`` on interior u faces i = ids+1 .. ide-1."""

    return jnp.asarray(_HALF, a.dtype) * (a[..., :-1] + a[..., 1:])


def _face_avg_y(a):
    return jnp.asarray(_HALF, a.dtype) * (a[..., :-1, :] + a[..., 1:, :])


def _agl_x(z_at_w, ht):
    """``0.5 * ( z_at_w(i-1)-ht(i-1) + z_at_w(i)-ht(i) )`` in Fortran order."""

    h = jnp.asarray(_HALF, z_at_w.dtype)
    return h * (((z_at_w[..., :-1] - ht[None, :, :-1]) + z_at_w[..., 1:]) - ht[None, :, 1:])


def _agl_y(z_at_w, ht):
    h = jnp.asarray(_HALF, z_at_w.dtype)
    return h * (((z_at_w[..., :-1, :] - ht[None, :-1, :]) + z_at_w[..., 1:, :]) - ht[None, 1:, :])


# --------------------------------------------------------------------------- #
# fddagd                                                                      #
# --------------------------------------------------------------------------- #
def fddagd(options: GridFddaOptions, pair: GridFddaCarry, u, v, th_phy, qv, pblh, ht, z_at_w,
           lead_seconds, dtype, *, surface_context=None, _with_early_off=False):
    """Uncoupled R*NDGDTEN of one ``fddagd`` call (tendencies reset to 0 first).

    u (nz, ny, nx+1), v (nz, ny+1, nx), th_phy (dry, K) / qv (nz, ny, nx),
    pblh/ht (ny, nx), z_at_w (nz+1, ny, nx).  Returns full-shape arrays with WRF's
    untouched points (boundary u/v faces) equal to zero.
    """

    one = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    cast = lambda a: jnp.asarray(a, dtype)  # noqa: E731
    u, v, th_phy, qv, pblh, ht, z_at_w = map(cast, (u, v, th_phy, qv, pblh, ht, z_at_w))
    nz = th_phy.shape[0]
    early_off, coef, tfac = fdda_time_factors(options, lead_seconds, dtype)

    if int(options.if_no_pbl_nudging_uv) == 1 or surface_context is not None:
        wpbl_u = _kpbl_weights(_face_avg_x(pblh), _agl_x(z_at_w, ht), dtype)
        wpbl_v = _kpbl_weights(_face_avg_y(pblh), _agl_y(z_at_w, ht), dtype)
    else:
        wpbl_u = wpbl_v = one(1.0)
    zagl_mass = z_at_w - ht[None]
    wpbl_t = _kpbl_weights(pblh, zagl_mass, dtype) if int(options.if_no_pbl_nudging_t) == 1 or surface_context is not None else one(1.0)
    wpbl_q = _kpbl_weights(pblh, zagl_mass, dtype) if int(options.if_no_pbl_nudging_q) == 1 or surface_context is not None else one(1.0)
    wz_uv = _wzfac(nz, int(options.if_zfac_uv), int(options.k_zfac_uv), dtype)[:, None, None]
    wz_t = _wzfac(nz, int(options.if_zfac_t), int(options.k_zfac_t), dtype)[:, None, None]
    wz_q = _wzfac(nz, int(options.if_zfac_q), int(options.k_zfac_q), dtype)[:, None, None]

    def val(old, new):
        return cast(old) * (one(1.0) - coef) + cast(new) * coef

    def strength(g, wpbl, wz):
        return ((one(g) * wpbl) * wz) * tfac

    zero = one(0.0)
    initial_u = initial_v = initial_t = initial_q = zero
    sda = None
    if surface_context is not None:
        from gpuwrf.assimilation.fdda_surface import sfddagd
        t_phy, p_phy, regime, znt = surface_context
        initial, sda = sfddagd(options, pair.surface, u,v,th_phy,qv,t_phy,p_phy,pblh,ht,z_at_w,
                              regime,znt,lead_seconds,dtype)
        initial_u, initial_v = initial[0][...,1:-1], initial[1][...,1:-1,:]
        initial_t, initial_q = initial[2:]
    # u: i = ids+1 .. ide-1 (interior faces), every mass row; v: j = jds+1 .. jde-1.
    run_i = initial_u + strength(options.guv, wpbl_u, wz_uv) * (val(pair.u_old, pair.u_new)[..., 1:-1] - u[..., 1:-1])
    rvn_i = initial_v + strength(options.guv, wpbl_v, wz_uv) * (val(pair.v_old, pair.v_new)[..., 1:-1, :] - v[..., 1:-1, :])
    rth = initial_t + strength(options.gt, wpbl_t, wz_t) * ((val(pair.t_old, pair.t_new) - th_phy) + one(_THETA_OFFSET))
    rqv = initial_q + strength(options.gq, wpbl_q, wz_q) * (val(pair.q_old, pair.q_new) - qv)
    run = jnp.zeros_like(u).at[..., 1:-1].set(jnp.broadcast_to(run_i, u[..., 1:-1].shape))
    rvn = jnp.zeros_like(v).at[..., 1:-1, :].set(jnp.broadcast_to(rvn_i, v[..., 1:-1, :].shape))
    rth = jnp.broadcast_to(rth, th_phy.shape)
    rqv = jnp.broadcast_to(rqv, qv.shape)
    rates = tuple(jnp.where(early_off, jnp.zeros_like(a), a) for a in (run, rvn, rth, rqv))
    if surface_context is not None:
        return rates, sda, early_off
    return (rates, early_off) if _with_early_off else rates


# --------------------------------------------------------------------------- #
# Driver control (fddagd_driver) + auxinput10 cursor                          #
# --------------------------------------------------------------------------- #
def _read_record(carry: GridFddaCarry, inputs: GridFddaInputs, itimestep, dtype) -> GridFddaCarry:
    """auxinput10 alarm (med_before_solve_io): load record ``r`` at the entry of its step."""

    steps = np.asarray(inputs.record_steps, dtype=np.int32)
    if steps.size == 0:
        return carry
    hit = jnp.asarray(itimestep, jnp.int32) - 1 == jnp.asarray(steps)
    index = jnp.argmax(hit).astype(jnp.int32)

    def load(c):
        updates = {f"{n}_{s}": jnp.asarray(jax.lax.dynamic_index_in_dim(
            getattr(inputs, f"{n}_{s}"), index, axis=0, keepdims=False), dtype)
            for n in FDDA_FIELDS for s in ("old", "new")}
        return c.replace(**updates)

    return jax.lax.cond(jnp.any(hit), load, lambda c: c, carry)


def fdda_driver_tendencies(carry: GridFddaCarry, inputs: GridFddaInputs, u, v, th_phy, qv, pblh, ht,
                           z_at_w, lead_seconds, itimestep, dtype, *, surface_context=None, p_phy=None) -> GridFddaCarry:
    """One ``fddagd_driver`` call (plus the auxinput10 read before it): UNCOUPLED rates."""

    options = inputs.options
    carry = _read_record(carry, inputs, itimestep, dtype)
    if surface_context is not None:
        from gpuwrf.assimilation.fdda_surface import read_surface_record
        carry = read_surface_record(carry, inputs, itimestep, dtype)
    itimestep = jnp.asarray(itimestep, jnp.int32)
    stepfg = int(options.stepfg)

    def first_step_old(c):
        # fddagd_driver loops i_start..min(i_end,ide-1): the east u / north v boundary faces keep the record.
        return c.replace(
            u_old=c.u_old.at[..., :-1].set(jnp.asarray(u, dtype)[..., :-1]),
            v_old=c.v_old.at[..., :-1, :].set(jnp.asarray(v, dtype)[..., :-1, :]),
            t_old=jnp.asarray(th_phy, dtype) - jnp.asarray(_THETA_OFFSET, dtype),
            q_old=jnp.asarray(qv, dtype))

    carry = jax.lax.cond(itimestep == 1, first_step_old, lambda c: c, carry)
    tend = carry.tendencies
    if int(options.fgdtzero) == 1:
        zero_now = jnp.mod(itimestep - 1, stepfg) == 0
        tend = tuple(jnp.where(zero_now, jnp.zeros_like(a), a) for a in tend)
    recompute = (itimestep == 1) | (jnp.mod(itimestep, stepfg) == 0)
    def history_hfx(c, rates, off, pressure):
        # WRF FDDAGD:655-656, final total surface + grid temperature rate.
        rho_air = jnp.asarray(pressure, dtype) / (jnp.asarray(287.0, dtype) * jnp.asarray(th_phy, dtype))
        dz8w = jnp.asarray(z_at_w, dtype)[1:] - jnp.asarray(z_at_w, dtype)[:-1]
        value = ((rho_air * jnp.asarray(1004.0, dtype)) * rates[2]) * dz8w
        return jnp.where(off, c.hfx_fdda, value)

    if surface_context is not None:
        def update(c):
            rates, sda, off = fddagd(options,c,u,v,th_phy,qv,pblh,ht,z_at_w,lead_seconds,dtype,
                                     surface_context=surface_context)
            result = c.replace(rundgdten=rates[0],rvndgdten=rates[1],rthndgdten=rates[2],rqvndgdten=rates[3])
            if options.surface_code == 2:
                result = result.replace(sda_hfx=jnp.where(off,c.sda_hfx,sda[0]),
                                        sda_qfx=jnp.where(off,c.sda_qfx,sda[1]))
            return result.replace(hfx_fdda=history_hfx(c, rates, off, surface_context[1]))
        carry = carry.replace(rundgdten=tend[0],rvndgdten=tend[1],rthndgdten=tend[2],rqvndgdten=tend[3])
        return jax.lax.cond(recompute,update,lambda c:c,carry)
    def compute(c):
        rates, early_off = fddagd(options, c, u, v, th_phy, qv, pblh, ht, z_at_w, lead_seconds, dtype,
                                 _with_early_off=True)
        hfx = getattr(c, "hfx_fdda", None)
        if p_phy is not None and isinstance(c, GridFddaHistoryCarry):
            # Literal module_fdda_psufddagd.F:655-656. WRF uses dry potential
            # temperature here, even though this intermediate is named rho_air.
            # phy_prep.F:4876 dz8w = z_at_w(k+1)-z_at_w(k).
            rho_air = jnp.asarray(p_phy, dtype) / (jnp.asarray(287.0, dtype) * jnp.asarray(th_phy, dtype))
            dz8w = jnp.asarray(z_at_w, dtype)[1:] - jnp.asarray(z_at_w, dtype)[:-1]
            value = ((rho_air * jnp.asarray(1004.0, dtype)) * rates[2]) * dz8w
            # WRF's early return zeros the tendencies but leaves HFX_FDDA held.
            hfx = jnp.where(early_off, c.hfx_fdda, value)
        return (*rates, hfx)

    result = jax.lax.cond(
        recompute,
        compute,
        lambda c: (*tuple(jnp.asarray(a, dtype) for a in tend), getattr(c, "hfx_fdda", None)),
        carry,
    )
    return carry.replace(rundgdten=result[0], rvndgdten=result[1], rthndgdten=result[2],
                         rqvndgdten=result[3],
                         **({"hfx_fdda": result[4]} if isinstance(carry, GridFddaHistoryCarry) else {}))


# --------------------------------------------------------------------------- #
# Coupling / decoupling / tendency ownership                                  #
# --------------------------------------------------------------------------- #
def wrf_muu_muv(mu_2, mub):
    """calc_mu_uv interior faces: 0.5*(MU(i)+MU(i-1)+MUB(i)+MUB(i-1)) (shape (ny, nx-1)/(ny-1, nx))."""

    h = jnp.asarray(_HALF, mu_2.dtype)
    muu = h * (((mu_2[:, 1:] + mu_2[:, :-1]) + mub[:, 1:]) + mub[:, :-1])
    muv = h * (((mu_2[1:, :] + mu_2[:-1, :]) + mub[1:, :]) + mub[:-1, :])
    return muu, muv


def wrf_muus_muvs(muts):
    """calc_mu_uv_1 interior faces: 0.5*(MU(i)+MU(i-1))."""

    h = jnp.asarray(_HALF, muts.dtype)
    return h * (muts[:, 1:] + muts[:, :-1]), h * (muts[1:, :] + muts[:-1, :])


def _scale_faces(tend, mass_faces, axis_x: bool, op):
    """Apply ``op(tend, c1h*mu+c2h)`` on interior faces only (WRF itsu/jtsv loops)."""

    if axis_x:
        inner = op(tend[..., 1:-1], mass_faces)
        return tend.at[..., 1:-1].set(inner)
    inner = op(tend[..., 1:-1, :], mass_faces)
    return tend.at[..., 1:-1, :].set(inner)


def _mass(c1, c2, mu):
    return c1[:, None, None] * mu[None] + c2[:, None, None]


def couple_fdda_rates(carry: GridFddaCarry, mut, mu_2, mub, c1h, c2h, dtype) -> GridFddaCarry:
    """calculate_phy_tend FDDA block: in-place coupling with the step-entry masses."""

    c1, c2 = jnp.asarray(c1h, dtype), jnp.asarray(c2h, dtype)
    muu, muv = wrf_muu_muv(jnp.asarray(mu_2, dtype), jnp.asarray(mub, dtype))
    mul = lambda t, m: m * t  # noqa: E731  -- (c1(k)*muu+c2(k))*RUNDGDTEN
    run = _scale_faces(carry.rundgdten, _mass(c1, c2, muu), True, mul)
    rvn = _scale_faces(carry.rvndgdten, _mass(c1, c2, muv), False, mul)
    mass_t = _mass(c1, c2, jnp.asarray(mut, dtype))
    return carry.replace(rundgdten=run, rvndgdten=rvn, rthndgdten=mass_t * carry.rthndgdten,
                         rqvndgdten=mass_t * carry.rqvndgdten)


def finish_fdda_rates(carry: GridFddaCarry, muts, c1h, c2h, dtype) -> GridFddaCarry:
    """phy_prep_part2 FDDA block: decouple the held rates with the end-of-RK masses."""

    c1, c2 = jnp.asarray(c1h, dtype), jnp.asarray(c2h, dtype)
    muts = jnp.asarray(muts, dtype)
    muus, muvs = wrf_muus_muvs(muts)
    div = lambda t, m: t / m  # noqa: E731
    run = _scale_faces(carry.rundgdten, _mass(c1, c2, muus), True, div)
    rvn = _scale_faces(carry.rvndgdten, _mass(c1, c2, muvs), False, div)
    mass_t = _mass(c1, c2, muts)
    return carry.replace(rundgdten=run, rvndgdten=rvn, rthndgdten=carry.rthndgdten / mass_t,
                         rqvndgdten=carry.rqvndgdten / mass_t)


def owned_fdda_tendf(carry: GridFddaCarry, *, specified_or_nested: bool, periodic_x: bool = False):
    """phy_fg_ten: coupled rates restricted like add_a2a / add_c2c_u / add_c2c_v.

    Returns ``(ru, rv, t, qv)`` to ADD to ru_tendf, rv_tendf, t_tendf and
    moist_tend(P_QV).  Specified/nested: mass i in [ids+1, ide-2], j in
    [jds+1, jde-2]; u faces i in [ids+1, ide-1]; v faces j in [jds+1, jde-1].
    """

    run, rvn, rth, rqv = carry.tendencies
    if not specified_or_nested:
        return run, rvn, rth, rqv

    def keep(a, ys, xs):
        return jnp.zeros_like(a).at[..., ys, xs].set(a[..., ys, xs])

    inner = slice(1, -1)
    # Only add_a2a has the periodic_x override (i over all mass columns).
    mx = slice(None) if periodic_x else inner
    return (keep(run, inner, inner), keep(rvn, inner, inner),
            keep(rth, inner, mx), keep(rqv, inner, mx))
