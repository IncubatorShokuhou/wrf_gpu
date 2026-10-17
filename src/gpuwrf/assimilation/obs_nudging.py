"""Pristine-WRF observation-nudging primitives.

This staged port covers independent single-level observations at the boundary
between ``errob`` and ``nudob`` in ``phys/module_fddaobs_rtfdda.F``. Innovations,
grid coordinates, observation pressure and interpolated base pressure must come
from WRF's observation preprocessing/error calculation, not gridded targets.
Arrays on a j-slab use (k, i) order; grid indices and PBL indices are WRF 1-based.
Pressure is in centibars except the model base/perturbation arrays (Pa).

The resident operational driver supplies grouped soundings and negative-QV
scaling through the private profile parameters. The independent single-level
API remains useful for pristine operator checks. GPU qualification is pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


def time_weight(xtime, dtmin, twindo, scalef, obtime):
    """WRF ``get_timewt``: model minutes; half-window/observation time in hours.

    The reference time is XTIME-DTMIN, not XTIME. Full weight occupies the
    inner half-window; the outer half tapers linearly to zero. Inputs must
    have positive ``twindo`` and ``scalef``.
    """
    dtype = jnp.asarray(xtime).dtype
    r = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    fdtim = r(xtime) - r(dtmin)
    tw1 = r(twindo) / r(2) * r(60) * r(scalef)
    tw2 = r(twindo) * r(60) * r(scalef)
    tconst = r(1) / tw1
    ttim = r(obtime) * r(60)
    dift = jnp.abs(fdtim - ttim)
    ramp = jnp.where(fdtim < ttim, (fdtim - (ttim - tw2)) * tconst,
                     ((ttim + tw2) - fdtim) * tconst)
    return jnp.where(dift <= tw1, r(1), jnp.where(dift <= tw2, ramp, r(0)))


def vif_heights(fullr, rampr, pblh, hmax=3000.0, fullmin=50.0, rampmin=50.0):
    """WRF ``get_vif_hts_slab``: lower/full and upper/zero heights in metres AGL.

    Negative full/ramp parameters encode a PBL-relative height with -5000 bias.
    WRF computes ht2 *before* capping ht1, an ordering retained here.
    """
    dtype = jnp.asarray(pblh).dtype
    r = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    fullr, rampr, pblh = r(fullr), r(rampr), r(pblh)
    ht1 = jnp.where(fullr >= r(0), fullr, pblh - (fullr + r(5000)))
    ht1 = jnp.maximum(r(fullmin), ht1)
    ht2 = jnp.where(rampr >= r(0), ht1 + jnp.maximum(r(rampmin), rampr),
                    jnp.maximum(ht1 + r(rampmin), pblh - (rampr + r(5000))))
    return jnp.minimum(ht1, r(hmax) - r(rampmin)), jnp.minimum(ht2, r(hmax))


def regime_vif_heights(vif, regime, pblh, hmax=3000.0, fullmin=50.0, rampmin=50.0):
    """WRF ``compute_VIH`` for validated regimes 1, 2, 3 or 4.

    Regimes 3 and 4 share the final parameter pair. An invalid regime yields
    NaN so it cannot silently select a physical regime; ingestion must refuse
    invalid regimes before a forecast (WRF calls wrf_error_fatal).
    """
    dtype = jnp.asarray(pblh).dtype
    vif = jnp.asarray(vif, dtype)
    idx = jnp.where(regime == 1, 0, jnp.where(regime == 2, 2, 4))
    valid = (regime == 1) | (regime == 2) | (regime == 3) | (regime == 4)
    full = jnp.where(valid, vif[idx], jnp.asarray(jnp.nan, dtype))
    ramp = jnp.where(valid, vif[idx + 1], jnp.asarray(jnp.nan, dtype))
    return vif_heights(full, ramp, pblh, hmax, fullmin, rampmin)


def dynamic_init_factor(xtime, fdaend, dtramp):
    """WRF fddaobs_driver's idynin=1 ramp (all inputs in minutes; |dtramp| > 0)."""
    dtype = jnp.asarray(xtime).dtype
    xtime, fdaend, dtramp = (jnp.asarray(x, dtype) for x in (xtime, fdaend, dtramp))
    dtr = jnp.abs(dtramp)
    tconst = jnp.asarray(1, dtype) / dtr
    return jnp.where(xtime < fdaend - dtr, jnp.asarray(1, dtype),
                     jnp.where(xtime <= fdaend, (fdaend - xtime) * tconst, jnp.asarray(0, dtype)))


class SingleLevelObservations(NamedTuple):
    """One entry per independent observation, already processed by WRF errob.

    ``innovation`` is ERRF(ivar,n), including WRF's >9e4 missing sentinel;
    ``base_pressure_cb`` is ERRF(6/7/8,n), appropriate to the variable's grid;
    ``pbl_level`` is ERRF(5,n), the PBL index at the observation. rio/rjo use
    WRF's non-staggered coordinates (mass indices plus .5). rko is fractional
    model level; NINT(rko+.45)==1 selects the surface branch.
    """

    rio: object
    rjo: object
    rko: object
    time_hours: object
    pressure_cb: object
    innovation: object
    base_pressure_cb: object
    pbl_level: object


@dataclass(frozen=True)
class SingleLevelNudgingOptions:
    """Static nudob options for one root-domain j-slab (nscan=0).

    The caller supplies the coefficient and switch appropriate to IVAR. WRF
    init sets rinfmn=1, rinfmx=2, pfree=50 cb; obs_dpsmx gives dcon=1/dpsmx.
    """

    ivar: int
    j: int
    jde: int
    dx: float
    rinxy: float
    twindo: float
    dtmin: float
    coefficient: float
    ionf: int = 1
    enabled: int = 1
    nudge_pbl: bool = True
    sfcfact: float = 1.0
    sfcfacr: float = 1.0
    dpsmx: float = 7.5
    sfc_scheme_horiz: int = 0
    sfc_scheme_vert: int = 0

    def __post_init__(self):
        import math

        if self.ivar not in (1, 2, 3, 4):
            raise ValueError("nudob IVAR must be U=1, V=2, theta=3 or QV=4 (pressure nudging was removed in WRF)")
        if self.jde < 4 or not 1 <= self.j < self.jde:
            raise ValueError("j must be a 1-based mass-grid slab index below jde")
        if self.ionf < 1 or self.enabled not in (0, 1):
            raise ValueError("ionf must be positive and enabled must be 0 or 1")
        for key in ("dx", "rinxy", "twindo", "dtmin", "sfcfact", "sfcfacr", "dpsmx"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if self.sfc_scheme_horiz not in (0, 1) or self.sfc_scheme_vert not in (0, 1):
            raise ValueError("surface horizontal/vertical schemes must be 0 or 1")


def single_level_nudob(options, obs, aten, saved_weights, *, ktau, xtime, mu, c1h, c2h,
                       msfx, msfy, pbase, pp, kpbl, vih1, vih2, terrain, z,
                       tfaci=1.0, restart=False, ktaur=0,
                       _profiles=None, _qv=None, _scale_negative_qv=0, _maxsnd_gap=20.0,
                       _start=0, _end=None, _j=None):
    """Single-level branches of pristine ``nudob``; return (ATEN, SAVWT(ivar)).

    ATEN/saved_weights/pbase/pp/z have shape (nz,nx); the tile includes all nx
    mass columns, with WRF ide=nx+1 and kte=nz. Wind ATEN is rho/map coupled,
    theta/QV ATEN uses c1h*mu+c2h. The caller owns the physical tendency reset.
    Other than SAVWT, no state is held: non-ionf steps reuse the saved weighted
    innovation with this step's masses and TFACI. Boundary slabs return both
    arrays unchanged, as WRF does.

    The operational driver supplies grouped profiles through ``_profiles``.
    A station loop carries two slab arrays; no obs-by-grid tensor is allocated.
    """
    o = options
    if _scale_negative_qv not in (0,1):
        raise ValueError("WRF obs_scl_neg_qv_innov must be 0 or 1")
    if _scale_negative_qv and (_qv is None or _profiles is None):
        raise ValueError("negative-QV scaling requires model QV and profile ERRF(10)")
    if _j is None and ((o.ivar < 3 and (o.j <= 2 or o.j >= o.jde - 1)) or (
            o.ivar >= 3 and (o.j <= 2 or o.j >= o.jde - 2))):
        return aten, saved_weights
    slab_j = o.j if _j is None else _j
    dtype = aten.dtype
    r = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    nz, nx = aten.shape
    if nz < 3:
        raise ValueError("nudob requires at least three model levels")
    if any(jnp.shape(value) != (len(obs.rio),) for value in obs):
        raise ValueError("observation arrays must be one-dimensional with matching lengths")
    ide = nx + 1
    ii = jnp.arange(1, nx + 1)
    kk = jnp.arange(1, nz + 1)[:, None]
    pind = r(.001) * (pbase + pp)
    rindx = r(o.rinxy) * r(1000) / r(o.dx)
    dcon = r(1) / r(o.dpsmx)
    gridx = r(0 if o.ivar == 1 else .5)
    gridy = r(0 if o.ivar == 2 else .5)
    search = jnp.trunc(rindx * r(2) + r(.99)).astype(jnp.int32)

    def recompute(_):
        if len(obs.rio) == 0:
            return jnp.zeros_like(aten)
        def station(n, accum):
            wt, wt2err = accum
            ra, rb = r(obs.rio[n]) - gridx, r(obs.rjo[n]) - gridy
            innovation = r(obs.innovation[n])
            psurf = r(obs.base_pressure_cb[n])
            pob = r(obs.pressure_cb[n])
            kob = jnp.clip(jnp.floor(r(obs.rko[n]) + r(.45) + r(.5)).astype(jnp.int32), 1, nz)
            profile = False if _profiles is None else _profiles.nlevels[n]>1
            surface = (kob == 1)&~jnp.asarray(profile)
            accepted = ((profile|(~(jnp.abs(innovation) > r(9e4)))) & (rb >= r(slab_j - search))
                        & (rb <= r(slab_j + search)))
            slope = (r(1) - r(2)) / (psurf - r(50))
            rinfac = jnp.minimum(jnp.maximum(slope * pob + r(2) - slope * r(50), r(1)), r(2))
            rinfac = jnp.where(profile,r(2),rinfac)
            radius = jnp.where(surface, rindx * r(o.sfcfacr), rindx * rinfac)
            # Surface pressure=50 cb is irrelevant to the surface branch.
            mini = jnp.trunc(ra - radius - r(.99)).astype(jnp.int32)
            maxi = jnp.trunc(ra + r(.99) + radius).astype(jnp.int32)
            mini = jnp.maximum(jnp.where(surface, 2, 1), mini)
            maxi = jnp.minimum(ide - 1, maxi)
            accepted = (accepted & (ra >= -radius / r(3)) & (ra <= r(ide) + radius / r(3))
                        & (rb >= -radius / r(3)) & (rb <= r(o.jde) + radius / r(3)))
            rx, ry = r(slab_j) - rb, r(ii) - ra
            rsq = rx * rx + ry * ry
            # Preserve WRF's different multiply order in the two branches.
            ris = jnp.where(surface, rindx * rindx * r(o.sfcfacr) * r(o.sfcfacr),
                             rindx * rinfac * rindx * rinfac)
            if o.sfc_scheme_horiz == 1:
                d = jnp.sqrt(rsq) + rindx * dcon * jnp.abs(psurf - r(.001) * pbase[0])
                dsq = d * d
                sfc_h = (ris - dsq) / (ris + dsq)
            else:
                scratch = jnp.abs(psurf - r(.001) * pbase[0]) * dcon
                pdfac = r(1) - jnp.minimum(r(1), scratch)
                sfc_h = (ris - rsq) / (ris + rsq) * pdfac
            horizontal = jnp.maximum(r(0), jnp.where(surface, sfc_h, (ris - rsq) / (ris + rsq)))

            if o.sfc_scheme_vert == 1:
                komax_sfc = jnp.where(kpbl > 25, 3, jnp.maximum(3, kpbl))
                sfc_v = r(komax_sfc - kk + 1) / r(komax_sfc)
                sfc_v = jnp.where(kk <= komax_sfc, sfc_v, r(0))
            else:
                dz_ramp = r(1) / jnp.maximum(r(1), vih2 - vih1)
                sfc_v = jnp.maximum(r(0), jnp.minimum(r(1), r(1) + (vih1 - z + terrain) * dz_ramp))
                # WRF exits the level loop at the first zero, even on a malformed column.
                sfc_v = jnp.where(jnp.cumprod((sfc_v > r(0)).astype(jnp.int32), axis=0), sfc_v, r(0))
            # Highest k satisfying p>=pob+7.5, then first k>=3 with p<=pob-7.5.
            komin = jnp.maximum(1, jnp.max(jnp.where(pind >= pob + r(7.5), kk, 0), axis=0))
            komax = jnp.min(jnp.where((kk >= 3) & (pind <= pob - r(7.5)), kk, nz), axis=0)
            kobs = jnp.min(jnp.where((kk >= komin) & (kk <= komax) & (pob > pind), kk, nz + 1), axis=0)
            kobs = jnp.where(kobs == nz + 1, 1, kobs)
            straddles = (kpbl <= komax) & (kpbl >= komin)
            above = kobs > obs.pbl_level[n]
            komin = jnp.where(straddles & above, jnp.maximum(kobs, komin), komin)
            komax = jnp.where(straddles & ~above, jnp.minimum(kpbl, komax), komax)
            upper_v = jnp.maximum(r(0), r(1) - jnp.abs(pind - pob) / r(7.5))
            upper_v = jnp.where((kk >= komin) & (kk <= komax) & (o.nudge_pbl | (komin >= kpbl)), upper_v, r(0))
            vertical = jnp.where(surface, jnp.where(o.nudge_pbl, sfc_v, r(0)), upper_v)
            error = innovation
            if _profiles is not None:
                from gpuwrf.assimilation.obs_profiles import interpolate_profile

                pe,pv,qm = interpolate_profile(
                    _profiles.pressure[n],_profiles.innovation[n],_profiles.values[n],
                    _profiles.q_model[n],_profiles.nlevels[n],pind,kk,kpbl,
                    max_gap=_maxsnd_gap,nudge_pbl=o.nudge_pbl)
                slope_p = (r(1)-r(2))/(r(.001)*pbase[0]-r(50))
                rfac_p = jnp.minimum(jnp.maximum(slope_p*pind+r(2)-slope_p*r(50),r(1)),r(2))
                ris_p = rindx*rfac_p*rindx*rfac_p
                hp = jnp.maximum(r(0),(ris_p-rsq)/(ris_p+rsq))
                horizontal = jnp.where(profile,hp,horizontal)
                vertical = jnp.where(profile,pv,vertical)
                error = jnp.where(profile,pe,innovation)
                if _scale_negative_qv and o.ivar==4:
                    qob = _profiles.q_model[n,0]
                    qcur = jnp.maximum(_qv,r(0))
                    in_range = accepted & (ii>=mini)&(ii<=maxi)
                    eligible = in_range & (sfc_v>r(0)) & (qcur<qob) & o.nudge_pbl
                    candidates = jnp.where(eligible,qcur,jnp.abs(innovation))
                    # Surface ERFIVR is mutated across the original i-outer/k-inner
                    # traversal. The upper-air single-level branch scales a local
                    # RESERF(k), so it has no cross-cell cumulative minimum.
                    cumulative = jax.lax.associative_scan(jnp.minimum,candidates.T.reshape(-1))
                    cumulative = cumulative.reshape(nx,nz).T
                    sfc_error = -jnp.minimum(jnp.abs(innovation),cumulative)
                    one_error = jnp.where((innovation<r(0))&(qcur<qob),
                                          -jnp.minimum(jnp.abs(innovation),qcur),innovation)
                    single_error = jnp.where(surface & (innovation<r(0)),sfc_error,one_error)
                    profile_error = jnp.where((pe<r(0))&(_qv<qm),-jnp.minimum(jnp.abs(pe),jnp.abs(_qv)),pe)
                    error = jnp.where(profile,profile_error,single_error)
            t = time_weight(r(xtime), r(o.dtmin), r(o.twindo),
                            jnp.where(surface, r(o.sfcfact), r(1)), r(obs.time_hours[n]))
            mask = accepted & (ii >= mini) & (ii <= maxi)
            h = jnp.where(mask, horizontal, r(0))
            # WRF uses a different multiplication order for surface vs upper air.
            w = jnp.where(surface, t * vertical * h, t * h * vertical)
            w2 = jnp.where(surface, t * t * h * h * vertical * vertical * error,
                            t * t * h * h * error * vertical * vertical)
            # Avoid missing/non-applicable observation operands contributing NaN*0.
            return wt + jnp.where(mask, w, r(0)), wt2err + jnp.where(mask, w2, r(0))

        wt, wt2err = jax.lax.fori_loop(_start, len(obs.rio) if _end is None else _end,
                                     station, (jnp.zeros_like(aten), jnp.zeros_like(aten)))
        return jnp.where(wt == r(0), r(0), wt2err / jnp.where(wt == r(0), r(1), wt))

    refresh = (ktau % o.ionf == 0) | (restart & (ktau == ktaur))
    weights = jax.lax.cond(refresh, recompute, lambda _: saved_weights, operand=None)
    if o.ivar < 3:
        mass = mu / (msfy if o.ivar == 1 else msfx)
    else:
        mass = c1h[:, None] * mu + c2h[:, None]
    increment = r(o.coefficient) * mass * weights * r(tfaci) * r(o.enabled)
    # WRF root i_s=2, i_e=ide-1; SAVWT includes the first column too.
    result = jnp.where(ii >= 2, aten + increment, aten)
    if _j is not None:
        interior = (slab_j > 2) & (slab_j < o.jde - (1 if o.ivar < 3 else 2))
        result = jnp.where(interior, result, aten)
        weights = jnp.where(interior, weights, saved_weights)
    return result, weights
