"""Resident WRF ``prec_acc_dt`` buckets (module_diag_misc.F:431–459).

Amounts are millimetres per own step, not differences of lifetime totals.
The clock is REAL seconds at step entry, as passed by solve_em.F. A reset is
WRF's exact MOD==0 test, not detection of a crossed wall-clock interval.
"""
from typing import NamedTuple

import jax.numpy as jnp


class PrecipitationBuckets(NamedTuple):
    prec_acc_c: object
    prec_acc_nc: object
    snow_acc_nc: object


def seed_buckets(state):
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    return PrecipitationBuckets(zero, zero, zero)


def update_buckets(buckets, raincv, rainncv, snowncv, t2, curr_secs2, prec_acc_dt):
    """Literal REAL arithmetic and reset/clamp order of pristine WRF.

    WRF clamps both rain buckets, but clamps snow only inside the cold-T2
    branch. The latter is intentional source parity, not a missing guard.
    NaNs propagate for the output-boundary finite checks to reject.
    """
    real = lambda x: jnp.asarray(x, jnp.float32)
    reset = jnp.fmod(real(curr_secs2), real(60.0) * real(prec_acc_dt)) == real(0.0)
    c, nc, snow = (jnp.where(reset, real(0.0), real(x)) for x in buckets)
    c = jnp.maximum(c + real(raincv), real(0.0))
    nc = jnp.maximum(nc + real(rainncv), real(0.0))
    snow = snow + real(snowncv)
    snow = jnp.where(real(t2) < real(273.15), jnp.maximum(snow + real(raincv), real(0.0)), snow)
    return PrecipitationBuckets(c, nc, snow)


def thompson_amounts(precip, *, spec_zone=0):
    """WRF module_mp_thompson.F:1298/1301; preserve REAL association."""
    p = {k: jnp.asarray(precip[k], jnp.float32) for k in ("rain", "snow", "graupel", "ice")}
    total = ((p["rain"] + p["snow"]) + p["graupel"]) + p["ice"]
    if "hail" in precip:
        total = total + jnp.asarray(precip["hail"], jnp.float32)
    if "total" in precip:
        total = jnp.asarray(precip["total"], jnp.float32)
    snow = p["snow"] + p["ice"]
    if spec_zone:
        ny, nx = total.shape
        y, x = jnp.arange(ny)[:, None], jnp.arange(nx)[None, :]
        inside = (y >= spec_zone) & (y < ny - spec_zone) & (x >= spec_zone) & (x < nx - spec_zone)
        total, snow = (jnp.where(inside, a, jnp.float32(0.0)) for a in (total, snow))
    return total, snow


def history_fields(buckets):
    """WRF's NetCDF backend writes uppercase names, despite Registry spelling."""
    return {} if buckets is None else {name.upper(): value for name, value in zip(buckets._fields, buckets)}


def validate_restart_buckets(buckets, interval, version, shape=None):
    """E78: active accumulation cannot silently resume with empty buckets."""
    if buckets is None and float(interval or 0.0) == 0.0:
        return
    if version != 1 or not isinstance(buckets, PrecipitationBuckets):
        raise ValueError("restart precipitation buckets schema is missing (E78)")
    if any(x is None or getattr(x, "dtype", None) != jnp.float32 for x in buckets):
        raise ValueError("restart precipitation buckets lack REAL leaves (E78)")
    if len({x.shape for x in buckets}) != 1 or len(buckets.prec_acc_c.shape) != 2:
        raise ValueError("restart precipitation buckets have inconsistent shapes (E78)")
    if shape is not None and buckets.prec_acc_c.shape != tuple(shape):
        raise ValueError("restart precipitation buckets differ from the surface grid (E78)")


def configured_interval(namelist):
    from collections.abc import Mapping
    if isinstance(namelist, Mapping):
        return float(namelist.get("physics", namelist).get("prec_acc_dt", 0.0))
    return float(getattr(namelist, "prec_acc_dt", 0.0))
