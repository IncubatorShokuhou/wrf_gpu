"""Literal MSKF REAL arithmetic, under component qualification only.

The complete MSKF driver and CMT are not admitted by this module. In
particular, calm wind is not repaired: the upstream timescale singularity
must propagate into the full port and be reported by the census/guards.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def wrf_nint_i4(value):
    """gfortran WRF NINT(REAL32) -> INTEGER4, without a floating clamp.

    The measured WRF compiler calls lroundf (INTEGER8 result), then narrows
    to INTEGER4. IEEE invalid conversions yield the INTEGER8 indefinite
    value, whose low 32 bits are zero. XLA's saturating float-to-int cast
    instead returns INT_MAX for positive infinity, silently removing the
    MSKF calm-column singularity when TIMEC is later capped at 86400.

    Work directly on the REAL32 significand/exponent and retain the low
    INTEGER4 bits. The same integer shift arithmetic naturally yields zero
    for Inf/NaN and out-of-INTEGER8 values, with no finite-value replacement
    of wind, timescale or tendency. The invalid result is compiler-specific;
    the pristine process fixture, rather than a mathematical NINT claim,
    governs this compatibility operation.
    """
    real=jnp.asarray(value,jnp.float32)
    bits=jax.lax.bitcast_convert_type(real,jnp.uint32)
    exponent=((bits >> jnp.uint32(23)) & jnp.uint32(255)).astype(jnp.int32)-127
    significand=(bits & jnp.uint32(0x7fffff)) | jnp.uint32(0x800000)
    drop=23-exponent
    half=jnp.left_shift(jnp.uint32(1),(drop-1).astype(jnp.uint32))
    rounded=jnp.right_shift(significand+half,drop.astype(jnp.uint32))
    integer=jnp.left_shift(significand,(exponent-23).astype(jnp.uint32))
    magnitude=jnp.where(exponent>=23,integer,rounded)
    signed=jnp.where((bits >> jnp.uint32(31)) != 0,jnp.uint32(0)-magnitude,magnitude)
    return jax.lax.bitcast_convert_type(signed,jnp.int32)


def mskf_time_scale(ulcl,vlcl,u500,v500,*,dx,dt,chmin,cloud_height,scale_fac,
                    surface_pressure,surface_temperature,surface_vapor,hfx,pblh,
                    source_layer_height,zlcl,ust,zol,abe):
    """Original module_cu_mskf.F:5564–5619, including source REAL guards.

    Inputs are scalar WRF caller values. This routine deliberately leaves
    TADVEC infinite and TIMEC zero at exactly zero VCONV, matching the
    measured original compiler. No postconversion finite substitute is used.
    """
    r=lambda x:jnp.asarray(x,jnp.float32)
    ulcl,vlcl,u500,v500=map(r,(ulcl,vlcl,u500,v500))
    dx,dt,chmin,cloud_height,scale_fac=map(r,(dx,dt,chmin,cloud_height,scale_fac))
    ps,ts,q0,hfx,pblh,source,zlcl,ust,zol,abe=map(r,(
        surface_pressure,surface_temperature,surface_vapor,hfx,pblh,
        source_layer_height,zlcl,ust,zol,abe))
    vconv=r(.5)*(jnp.sqrt(ulcl*ulcl+vlcl*vlcl)+jnp.sqrt(u500*u500+v500*v500))
    tadvec=dx/vconv
    timec=jnp.maximum(chmin,cloud_height)
    timec=timec*scale_fac
    density=ps/r(287.0)/ts
    wst=hfx/density/r(1004.5)
    wst=jnp.maximum(r(0.0),wst)
    wst=r(9.81)*pblh*wst
    thetav=(r(1e5)/ps)**(r(287.0)/r(1004.5))
    thetav=ts*thetav
    thetav=thetav*(r(1.0)+q0*r(.622))
    wst=wst/thetav
    wst=wst**r(.3333)
    sclvel=wst**3
    zlcl_kay=jnp.maximum(zlcl,source)
    sclvel=sclvel/pblh
    sclvel=sclvel*zlcl_kay
    sclvel=sclvel**r(.333)
    def unstable(_):
        frc=r(3.8)*ust*ust
        frc=frc+r(.22)*sclvel*sclvel
        zz=r(-1.0)*zol
        z=zz**(r(2.0)/r(3.0))
        z=z*(r(1.9)*ust*ust)
        return frc+z
    frc=jax.lax.cond(zol<=0,unstable,lambda _:r(3.8)*ust*ust,None)
    sclvel=jnp.sqrt(frc)
    # These two original guards are distinct from an added calm-wind floor.
    sclvel=jnp.where(sclvel<r(.1),r(.1),sclvel)
    abe=jnp.where(abe<=0,r(1.0),abe)
    timec=timec/((r(.03)*sclvel*abe)**r(.3333))
    timec=jnp.maximum(tadvec,timec)
    nic=wrf_nint_i4(timec/dt)
    timec=nic.astype(jnp.float32)*dt
    timec=jnp.minimum(timec,r(86400.0))
    return dict(vconv=vconv,tadvec=tadvec,nic=nic,timec=timec,abe=abe,sclvel=sclvel)
