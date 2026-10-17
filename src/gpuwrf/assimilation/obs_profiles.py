"""WRF multi-level sounding interpolation at the nudob boundary.

Profiles are consecutive, descending-pressure levels in original file order.
Missing variables are compacted in that order before a fixed-depth binary
search; no (sounding-level x model-grid) intermediate is allocated.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class ProfileData(NamedTuple):
    pressure: object     # (reports,max_levels), cb
    innovation: object   # ERRF(ivar), same shape
    values: object       # VAROBS(ivar), used by WRF validity checks
    q_model: object      # ERRF(10), model QV at observation levels
    nlevels: object      # (reports,), actual lengths (padding is not data)


def interpolate_profile(pressure,innovation,values,q_model,nlevels,pijk,kk,kpbl,
                        *,max_gap,nudge_pbl):
    """Original nudob's multi-level search, log-pressure interpolation and mask."""
    dtype=pijk.dtype
    r=lambda x:jnp.asarray(x,dtype)  # noqa: E731
    length=len(pressure)
    levels=jnp.arange(length,dtype=jnp.int32)
    good=(levels<nlevels)&(values>r(-800000))&(pressure>r(-800000))
    # High endpoint search excludes the sounding's first raw level (NN=2).
    indices=jnp.nonzero(good&(levels>=1),size=length,fill_value=length)[0]
    safe=jnp.minimum(indices,length-1)
    keys=jnp.where(indices<length,-pressure[safe],r(1e9))
    lo=jnp.zeros(pijk.shape,jnp.int32)
    hi=jnp.full(pijk.shape,length,jnp.int32)
    for _ in range((length+1).bit_length()):
        mid=(lo+hi)//2
        move=(mid<length)&(keys[jnp.minimum(mid,length-1)] < -pijk)
        lo=jnp.where(move,mid+1,lo)
        hi=jnp.where(move,hi,mid)
    ih=indices[jnp.minimum(lo,length-1)]
    ihs=jnp.minimum(ih,length-1)
    preceding=jnp.concatenate((jnp.asarray([-1],jnp.int32),
                 jax.lax.associative_scan(jnp.maximum,jnp.where(good,levels,-1))[:-1]))
    il=preceding[ihs]
    ils=jnp.maximum(il,0)
    ph,pl=pressure[ihs],pressure[ils]
    gap_hi,gap_lo=jnp.abs(ph-pijk),jnp.abs(pl-pijk)
    valid=((lo<length)&(ih<length)&(il>=0)&(ph<=pijk)&(pl>=pijk)
           &(gap_hi<=r(max_gap))&(gap_lo<=r(max_gap))&(gap_hi+gap_lo<=r(max_gap))
           &(pijk<=pressure[0])&(pijk>pressure[jnp.maximum(nlevels-1,0)])
           &(kk<=pijk.shape[0]-2)&(nudge_pbl|(kk>kpbl)))
    # WRF jumps over invalid brackets; safe operands avoid evaluating log/0 there.
    phs,pls,ps=(jnp.where(valid,x,r(1)) for x in (ph,pl,pijk))
    same=jnp.abs(phs-pls)<r(.000001)
    den=jnp.where(same,r(1),jnp.log(phs/pls))
    fraction=jnp.where(same,r(0),jnp.log(ps/pls)/den)
    error=innovation[ils]+(innovation[ihs]-innovation[ils])*fraction
    qm=q_model[ils]+(q_model[ihs]-q_model[ils])*fraction
    return jnp.where(valid,error,r(0)),jnp.where(valid,r(1),r(0)),qm
