"""WRF v4 spectral nudging operator (module_fdda_spnudging.F).

The horizontal filter uses FFTPACK5's low-pass convention, including its
extra, zeroed east/north points. Unlike PSU grid FDDA, spectral nudging keeps
the first OLD analysis, uses linear dk ramps and extrapolates during a positive
end ramp. This module is an operator port; operational admission is separate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.fdda_grid import (
    GridFddaOptions, GridFddaInputs, GridFddaCarry, FDDA_FIELDS,
    initial_fdda_carry, _agl_x, _agl_y, _face_avg_x, _face_avg_y,
    _wrf_div, wrf_minutes,
)


@dataclass(frozen=True)
class SpectralFddaOptions(GridFddaOptions):
    """Per-domain &fdda controls; level indices are Fortran one-based."""

    gph: float = 0.0
    xwavenum: int = 0
    ywavenum: int = 0
    if_no_pbl_nudging_ph: int = 0
    if_zfac_ph: int = 0
    k_zfac_ph: int = 0
    dk_zfac_uv: int = 1
    dk_zfac_t: int = 1
    dk_zfac_ph: int = 1
    dk_zfac_q: int = 1
    ktrop: int = 0

    def __post_init__(self):
        super().__post_init__()
        for name in ("guv", "gt", "gph", "gq"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"In spectral nudging, {name} must be finite and nonnegative")
        if not np.isfinite(self.dtramp_min):
            raise ValueError("dtramp_min must be finite")
        for suffix in ("uv", "t", "ph", "q"):
            if (getattr(self, f"if_no_pbl_nudging_{suffix}") == 1
                    or getattr(self, f"if_zfac_{suffix}") == 1):
                if getattr(self, f"dk_zfac_{suffix}") < 1:
                    raise ValueError(f"dk_zfac_{suffix} must be greater or equal than 1")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralFddaInputs(GridFddaInputs):
    """Resident auxinput10 records; PH_NDG is on vertically staggered levels."""

    options: SpectralFddaOptions = SpectralFddaOptions(interval_m=1)
    ph_old: object = None
    ph_new: object = None

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return (*children, self.ph_old, self.ph_new), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        names = [f"{n}_{s}" for n in FDDA_FIELDS for s in ("old", "new")] + ["ph_old", "ph_new"]
        return cls(**dict(zip(names, children)), options=aux[0], record_steps=aux[1])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralFddaCarry(GridFddaCarry):
    """Grid FDDA carry plus three REAL PH leaves, with an explicit schema tag."""

    ph_old: object = None
    ph_new: object = None
    rphndgdten: object = None

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)), 1

    @classmethod
    def tree_unflatten(cls, aux, children):
        if aux != 1 or len(children) != 15:
            raise ValueError("spectral FDDA restart carry schema mismatch (E78)")
        return cls(*children)


def initial_spectral_carry(inputs: SpectralFddaInputs, dtype):
    base = initial_fdda_carry(inputs, dtype)
    ph_old, ph_new = (jnp.asarray(getattr(inputs, n)[0], dtype) for n in ("ph_old", "ph_new"))
    return SpectralFddaCarry(**{f.name: getattr(base, f.name) for f in fields(base)},
                             ph_old=ph_old, ph_new=ph_new, rphndgdten=jnp.zeros_like(ph_old))


def spectral_driver_tendencies(carry, inputs, u, v, th_phy, ph, qv, pblh, ht,
                              z_at_w, lead_seconds, itimestep, dtype):
    """auxinput10 cursor + fddagd_driver SPNUDGING branch (OLD stays intact)."""

    if not isinstance(carry, SpectralFddaCarry) or any(
            getattr(carry, n) is None for n in ("ph_old", "ph_new", "rphndgdten")):
        raise ValueError("active spectral FDDA needs its complete restart carry (E78)")
    names = [f"{n}_{s}" for n in (*FDDA_FIELDS, "ph") for s in ("old", "new")]
    steps = tuple(inputs.record_steps)
    if steps:
        hit = jnp.asarray(itimestep, jnp.int32) - 1 == jnp.asarray(steps, jnp.int32)
        index = jnp.argmax(hit).astype(jnp.int32)
        def load(c):
            return c.replace(**{n: jnp.asarray(jax.lax.dynamic_index_in_dim(
                getattr(inputs, n), index, axis=0, keepdims=False), dtype) for n in names})
        carry = jax.lax.cond(jnp.any(hit), load, lambda c: c, carry)
    # Spectral branch intentionally does not perform the PSU step-1 OLD overwrite.
    tendencies = (*carry.tendencies, carry.rphndgdten)
    if inputs.options.fgdtzero == 1:
        zero = jnp.mod(itimestep - 1, inputs.options.stepfg) == 0
        tendencies = tuple(jnp.where(zero, jnp.zeros_like(a), a) for a in tendencies)
    recompute = (itimestep == 1) | (jnp.mod(itimestep, inputs.options.stepfg) == 0)
    def compute(c):
        ru, rv, rt, rp, rq = spectral_nudging(
            inputs.options, {n: getattr(c, n) for n in names}, u, v, th_phy, ph, qv,
            pblh, ht, z_at_w, lead_seconds, dtype)
        return ru, rv, rt, rq, rp
    ru, rv, rt, rq, rp = jax.lax.cond(recompute, compute, lambda c: tendencies, carry)
    return carry.replace(rundgdten=ru, rvndgdten=rv, rthndgdten=rt, rqvndgdten=rq, rphndgdten=rp)


def lowpass_axis(a, nwave: int, axis: int):
    """FFTPACK5 forward/mask/inverse; n=0 retains DC, n<0 removes all modes.

    WRF masks packed REAL entries after min(max(1+2*nwave,0),N). That is
    exactly complex bins 0..nwave (including Nyquist when N is even).
    Forward/backward normalization cancels; JAX's default pair has that same
    normalization. Both transforms retain the input REAL precision.
    """

    n = a.shape[axis]
    spectrum = jnp.fft.rfft(a, axis=axis)
    shape = [1] * a.ndim
    shape[axis] = n // 2 + 1
    keep = (jnp.arange(n // 2 + 1) <= int(nwave)).reshape(shape)
    return jnp.fft.irfft(jnp.where(keep, spectrum, jnp.zeros_like(spectrum)), n=n, axis=axis)


def spectral_filter(difference, xwavenum: int, ywavenum: int):
    """Filter mass-domain (k,j,i) differences as WRF's 3dx then 3dy calls.

    The transform lengths are mass nx+1, ny+1. 3dx sets the last x point to
    zero; 3dy sets the last y point to zero AFTER the x filter. No taper or
    periodic extension is present in the pristine implementation.
    """

    sheet = jnp.pad(difference, ((0, 0), (0, 1), (0, 1)))
    sheet = lowpass_axis(sheet, xwavenum, -1)
    sheet = sheet.at[:, -1, :].set(jnp.asarray(0, difference.dtype))
    return lowpass_axis(sheet, ywavenum, -2)[:, :-1, :-1]


def spectral_time_factors(options: SpectralFddaOptions, lead_seconds, dtype):
    """Literal spectral_nudging clock/ramp, distinct from psufddagd's ramp."""

    one = lambda a: jnp.asarray(a, dtype)
    xtime, minutes = wrf_minutes(lead_seconds, dtype)
    interval = one(options.interval_m)
    old = jnp.floor(_wrf_div(minutes, interval, dtype)) * interval
    new = old + interval
    coef = _wrf_div(minutes - old, new - old, dtype)
    end = one(options.end_h) * one(60.0)
    if options.if_ramping == 1 and options.dtramp_min > 0:
        end = end + jnp.abs(one(options.dtramp_min))
    early_off = xtime > end
    tfac = one(1.0)
    if options.if_ramping == 1 and abs(options.dtramp_min) > 0:
        ramp = jnp.abs(one(options.dtramp_min))
        inside = (xtime >= end - ramp) & (xtime <= end)
        tfac = jnp.where(xtime < end - ramp, one(1.0),
                         jnp.where(inside, _wrf_div(end - xtime, ramp, dtype), one(0.0)))
        if options.dtramp_min > 0:
            coef = jnp.where(inside, _wrf_div(minutes - old + interval, interval, dtype), coef)
    return early_off, coef, tfac


def _pbl_index(pblh, zagl_w):
    hit = (pblh[None] >= zagl_w[:-1]) & (pblh[None] < zagl_w[1:])
    return jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0) + 1, 1)


def _ramp(nz, cut, dk, dtype):
    k = jnp.arange(1, nz + 1, dtype=jnp.int32)[:, None, None]
    return jnp.maximum(jnp.minimum(_wrf_div((k - cut).astype(dtype), dk, dtype),
                                   jnp.asarray(1, dtype)), jnp.asarray(0, dtype))


def _weights(options, suffix, nz, pblh, ht, z_at_w, dtype, face=None):
    """WRF PBL precedes (rather than multiplies) zfac; PH uses its own loop."""

    if getattr(options, f"if_no_pbl_nudging_{suffix}") == 1:
        if face == "u":
            cut = _pbl_index(_face_avg_x(pblh), _agl_x(z_at_w, ht))
        elif face == "v":
            cut = _pbl_index(_face_avg_y(pblh), _agl_y(z_at_w, ht))
        elif suffix == "ph":
            # Pristine loop_kph exits on the FIRST level with zpbl >= zagl.
            # Do not substitute the mass-level bracket search used for T/Q.
            hit = pblh[None] >= z_at_w[:nz] - ht[None]
            cut = jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0) + 1, 1)
        else:
            cut = _pbl_index(pblh, z_at_w - ht[None])
        weight = _ramp(nz, cut, getattr(options, f"dk_zfac_{suffix}"), dtype)
        # WRF only fills PBL weights on interior wind faces; west/south stays1.
        if face == "u":
            weight = jnp.pad(weight, ((0, 0), (0, 0), (1, 0)), constant_values=1)
        elif face == "v":
            weight = jnp.pad(weight, ((0, 0), (1, 0), (0, 0)), constant_values=1)
    elif getattr(options, f"if_zfac_{suffix}") == 1:
        weight = _ramp(nz, getattr(options, f"k_zfac_{suffix}"),
                       getattr(options, f"dk_zfac_{suffix}"), dtype)
    else:
        weight = jnp.asarray(1, dtype)
    if suffix in ("t", "q") and options.ktrop > 0:
        k = jnp.arange(1, nz + 1)[:, None, None]
        lid = jnp.where(k < options.ktrop, jnp.asarray(1, dtype),
                        jnp.where(k == options.ktrop, jnp.asarray(.5, dtype), jnp.asarray(0, dtype)))
        weight = weight * lid
    return weight


def spectral_nudging(options: SpectralFddaOptions, pair, u, v, th_phy, ph, qv,
                     pblh, ht, z_at_w, lead_seconds, dtype):
    """Uncoupled (RU,RV,RTH,RPH,RQV) from pristine spectral_nudging.

    ``pair`` maps u/v/t/ph/q + _old/_new to analysis arrays. Physical layouts:
    u(nz,ny,nx+1), v(nz,ny+1,nx), t/q(nz,ny,nx), ph/z_at_w(nz+1,ny,nx).
    PH input is PERTURBATION geopotential (the caller passes grid%ph_2).
    The caller's kte=min(k_end,kde-1) leaves the top PH tendency zero.
    """

    cast = lambda a: jnp.asarray(a, dtype)
    u, v, th_phy, ph, qv, pblh, ht, z_at_w = map(cast, (u, v, th_phy, ph, qv, pblh, ht, z_at_w))
    nz, ny, nx = th_phy.shape
    if options.ktrop > nz + 1:
        raise ValueError("KTROP is set above the model top")
    early_off, coef, tfac = spectral_time_factors(options, lead_seconds, dtype)
    one = lambda a: jnp.asarray(a, dtype)

    def rate(name, model, g, suffix, face=None):
        if g <= 0:
            return jnp.zeros((nz, ny, nx), dtype)
        # All driver tile differences are defined only over the mass footprint.
        old, new = cast(pair[f"{name}_old"]), cast(pair[f"{name}_new"])
        if face == "u":
            old, new, model = old[..., :-1], new[..., :-1], model[..., :-1]
        elif face == "v":
            old, new, model = old[..., :-1, :], new[..., :-1, :], model[..., :-1, :]
        elif name == "ph":
            old, new, model = old[:nz], new[:nz], model[:nz]
        analysis = old * (one(1) - coef) + new * coef
        difference = analysis - model
        if name == "t":
            difference = difference + one(300)
        filtered = spectral_filter(difference, options.xwavenum, options.ywavenum)
        weight = _weights(options, suffix, nz, pblh, ht, z_at_w, dtype, face)
        return ((one(g) * weight) * tfac) * filtered

    def compute(_):
        ru = rate("u", u, options.guv, "uv", "u")
        rv = rate("v", v, options.guv, "uv", "v")
        rt = rate("t", th_phy, options.gt, "t")
        rp = rate("ph", ph, options.gph, "ph")
        rq = rate("q", qv, options.gq, "q")
        return (jnp.pad(ru, ((0, 0), (0, 0), (0, 1))),
                jnp.pad(rv, ((0, 0), (0, 1), (0, 0))), rt,
                jnp.pad(rp, ((0, 1), (0, 0), (0, 0))), rq)

    # WRF returns before transforms after the configured end. A select on
    # already computed rates would execute every FFT throughout the remaining
    # forecast. Keep the genuine early return as a device-resident branch.
    return jax.lax.cond(early_off, lambda _: tuple(jnp.zeros_like(a) for a in (u, v, th_phy, ph, qv)),
                        compute, operand=None)
