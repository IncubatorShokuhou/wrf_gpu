"""Resident root-domain WRF observation driver (in4dob -> errob -> nudob).

The immutable input holds initialization-time report events. Only the rolling
frame is mutable: VAROBS/RKO and the four SAVWT fields survive cadence holds
and restart. No timestep reads files or transfers observation data to the host.
"""
from dataclasses import dataclass, fields, replace
from typing import Any

import jax
import jax.numpy as jnp

from gpuwrf.assimilation.obs_errors import errob
from gpuwrf.assimilation.obs_nudging import (
    SingleLevelNudgingOptions, SingleLevelObservations, dynamic_init_factor,
    regime_vif_heights, single_level_nudob,
)
from gpuwrf.assimilation.obs_profiles import ProfileData


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ObservationInputs:
    times: Any
    arrivals: Any
    deactivated: Any
    rio: Any
    rjo: Any
    rko: Any
    variants: Any
    variant_steps: Any
    lengths: Any
    options: tuple
    max_reports: int
    fingerprint: Any

    def __post_init__(self):
        # A checksum is data, not executable configuration. Keeping its text
        # in the static aux would compile a new forecast for every OBS file,
        # even when all shapes/options are unchanged. Host constructors may
        # pass hex text; traced reconstruction already supplies the byte leaf.
        if isinstance(self.fingerprint,str):
            object.__setattr__(self,'fingerprint',jnp.asarray(
                list(bytes.fromhex(self.fingerprint)),jnp.uint8))

    def tree_flatten(self):
        return (*tuple(getattr(self, f.name) for f in fields(self)[:9]),self.fingerprint), (
            self.options, self.max_reports)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children[:9], *aux, children[9])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ObservationCarry:
    varobs: Any
    rko: Any
    frame_start: Any
    variants: Any
    read_step: Any
    count: Any
    savwt_u: Any
    savwt_v: Any
    savwt_t: Any
    savwt_q: Any
    fingerprint: Any  # SHA256 bytes: restart refuses a changed observation stream

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def replace(self, **updates):
        return replace(self, **updates)

    @property
    def weights(self):
        return self.savwt_u, self.savwt_v, self.savwt_t, self.savwt_q


def initial_observation_carry(inputs, shape, dtype=jnp.float32):
    m, levels = inputs.max_reports, inputs.rko.shape[1]
    zero = jnp.zeros(shape, dtype)
    return ObservationCarry(
        jnp.full((m, levels, 6), -888888, dtype), jnp.ones((m, levels), dtype),
        jnp.asarray(-m, jnp.int32), jnp.full((m,), -1, jnp.int32),
        jnp.asarray(-1, jnp.int32), jnp.asarray(0, jnp.int32),
        zero, zero, zero, zero,
        jnp.asarray(inputs.fingerprint, jnp.uint8))


def _frame(carry, inputs, ktau, xtime):
    r = lambda x: jnp.asarray(x, carry.varobs.dtype)  # noqa: E731
    o = dict(inputs.options)
    # Literal REAL XHOUR=MAX(XTIME/60.,0.) from in4dob.
    hour = jnp.maximum(xtime/jax.lax.optimization_barrier(r(60)), r(0))
    lo = jnp.searchsorted(inputs.times, hour-r(o['obs_twindo']), side='left')
    hi = jnp.searchsorted(inputs.arrivals, ktau, side='right')
    count = jnp.maximum(hi-lo, 0)
    source = lo+jnp.arange(inputs.max_reports,dtype=jnp.int32)
    index = jnp.minimum(source, len(inputs.times)-1)
    valid = (source<hi) & (inputs.deactivated[index]>ktau)
    variant = jnp.maximum(jnp.sum(inputs.variant_steps[index]<=ktau, axis=1,dtype=jnp.int32)-1, 0)
    old = source-carry.frame_start
    reuse = (old>=0) & (old<inputs.max_reports)
    old = jnp.clip(old, 0, inputs.max_reports-1)
    unchanged = reuse & (carry.variants[old]==variant)
    varobs = jnp.where(unchanged[:,None,None], carry.varobs[old], inputs.variants[index,variant])
    rko = jnp.where(reuse[:,None], carry.rko[old], inputs.rko[index])
    lengths = jnp.where(valid, inputs.lengths[index], 0)
    return lo, count, index, variant, varobs, rko, lengths


def observation_driver(carry, inputs, *, ktau, xtime, dt_s, dx,
                       u, v, tb, qv, pbase, pp, z, uratx, vratx, tratx,
                       kpbl, regime, pblh, terrain, mut, muu, muv,
                       c1h, c2h, msfux, msfuy, msfvx, msfvy, msftx, msfty):
    """Return updated carry and (RU,RV,T,QV) coupled source increments.

    Mass-model fields use (k,j,i); wind arrays include their final C-grid face.
    SAVWT uses the mass-grid extent, as the original root driver stops at ide-1
    and jde-1. The unused final wind face receives zero source.
    """
    o = dict(inputs.options)
    dtype = tb.dtype
    r = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    nz, ny, nx = tb.shape

    def refresh(carry):
        lo, count, index, variant, varobs, rko, lengths = _frame(carry, inputs, ktau, xtime)
        m, levels = rko.shape
        valid = jnp.arange(levels)[None,:]<lengths[:,None]
        # Safe finite padding is discarded after ERROB. It cannot contribute to
        # NUDOB, and avoids speculative log/divide operands in padded entries.
        dummy = r([0,0,300,0,100,0])
        vv = jnp.where(valid[...,None], varobs, dummy).reshape(-1,6)
        rk = jnp.where(valid, rko, r(1)).reshape(-1)
        ri = jnp.broadcast_to(inputs.rio[index,None], (m,levels)).reshape(-1)
        rj = jnp.broadcast_to(inputs.rjo[index,None], (m,levels)).reshape(-1)
        vv, rk, err = errob(vv, rk, ri, rj, u=u, v=v, tb=tb, qv=qv,
                           pbase=pbase, pp=pp, z=z, uratx=uratx, vratx=vratx,
                           tratx=tratx, kpbl=kpbl, wind=bool(o['obs_nudge_wind']),
                           temperature=bool(o['obs_nudge_temp']), moisture=bool(o['obs_nudge_mois']))
        vv, rk, err = vv.reshape(m,levels,6), rk.reshape(m,levels), err.reshape(m,levels,10)
        varobs = jnp.where(valid[...,None], vv, varobs)
        rko = jnp.where(valid, rk, rko)
        weights = []
        for ivar, old in enumerate(carry.weights,1):
            name = ('wind','wind','temp','mois')[ivar-1]
            field = ('uv','uv','t','q')[ivar-1]
            if not o[f'obs_nudge_{name}']:
                weights.append(old)
                continue
            vif = [o[f'obs_nudgez{part}r{reg}_{field}']
                   for reg in (1,2,4) for part in ('full','ramp')]
            h1,h2 = regime_vif_heights(r(vif), regime, pblh, o['obs_nudgezmax'],
                                      o['obs_nudgezfullmin'], o['obs_nudgezrampmin'])
            errors = jnp.where(valid,err[...,ivar-1],r(99999))
            profile = ProfileData(varobs[...,4],errors,varobs[...,ivar-1],err[...,9],lengths)
            obs = SingleLevelObservations(inputs.rio[index], inputs.rjo[index], rko[:,0],
                    inputs.times[index], varobs[:,0,4], errors[:,0],
                    err[:,0,6 if ivar==1 else 7 if ivar==2 else 5],err[:,0,4])
            opt = SingleLevelNudgingOptions(ivar,3,ny+1,dx,float(o['obs_rinxy']),
                    float(o['obs_twindo']),float(dt_s)/60.,1.,
                    nudge_pbl=not o[f'obs_no_pbl_nudge_{field}'],
                    sfcfact=float(o['obs_sfcfact']),sfcfacr=float(o['obs_sfcfacr']),
                    dpsmx=float(o['obs_dpsmx']),sfc_scheme_horiz=int(o['obs_sfc_scheme_horiz']),
                    sfc_scheme_vert=int(o['obs_sfc_scheme_vert']))
            def slab(j, saved, pb, pert, kk, a, b, ht, zz, qq):
                _, wt = single_level_nudob(opt,obs,jnp.zeros_like(saved),saved,
                    ktau=0,xtime=xtime,mu=jnp.ones(nx,dtype),c1h=c1h,c2h=c2h,
                    msfx=jnp.ones(nx,dtype),msfy=jnp.ones(nx,dtype),pbase=pb,pp=pert,
                    kpbl=kk,vih1=a,vih2=b,terrain=ht,z=zz,_profiles=profile,_qv=qq,
                    _scale_negative_qv=int(o['obs_scl_neg_qv_innov']),
                    _maxsnd_gap=float(o['obs_max_sndng_gap']),_end=count,_j=j)
                return wt
            wt = jax.vmap(slab)(jnp.arange(1,ny+1),old.transpose(1,0,2),
                    pbase.transpose(1,0,2),pp.transpose(1,0,2),kpbl,h1,h2,terrain,
                    z.transpose(1,0,2),qv.transpose(1,0,2)).transpose(1,0,2)
            weights.append(jnp.where(count>0,wt,old))
        return carry.replace(varobs=varobs,rko=rko,frame_start=lo,variants=variant,
                read_step=ktau,count=count,savwt_u=weights[0],savwt_v=weights[1],
                savwt_t=weights[2],savwt_q=weights[3])

    active = xtime<=r(o['fdda_end'])
    carry = jax.lax.cond(active & (ktau % int(o['obs_ionf'])==0), refresh, lambda c:c, carry)
    factor = dynamic_init_factor(xtime,r(o['fdda_end']),r(o['obs_dtramp'])) if o['obs_idynin'] else r(1)
    sources = []
    for ivar, wt in enumerate(carry.weights,1):
        name = ('wind','wind','temp','mois')[ivar-1]
        if ivar<3:
            mass = (muu[:ny,:nx]/msfuy[:ny,:nx] if ivar==1 else muv[:ny,:nx]/msfvx[:ny,:nx])
        else:
            mass = c1h[:,None,None]*mut+c2h[:,None,None]
        source = r(o[f'obs_coef_{name}'])*mass*wt*factor*r(o[f'obs_nudge_{name}'])
        source = jnp.where(active & (carry.count>0) & (jnp.arange(nx)>=1),source,r(0))
        if ivar==1:source=jnp.pad(source,((0,0),(0,0),(0,1)))
        if ivar==2:source=jnp.pad(source,((0,0),(0,1),(0,0)))
        sources.append(source)
    return carry, tuple(sources)
