"""Pristine WRF swint_opt=1 REAL surface interpolation.

Parameters refresh at radiation calls, then interpolation runs on EVERY own
step before the land surface, including radiation calls. Atmospheric heating
and radiation-profile fluxes remain held. The caller supplies actual current
and midpoint cosines, original SWDOWN/direct, and current surface albedo.
"""
from typing import NamedTuple
import numpy as np


class SolarInterpolationState(NamedTuple):
    swddir_ref: object
    bb: object
    Bx: object
    swdown_ref: object
    gg: object
    Gx: object
    coszen_ref: object


class SolarSurfaceFluxes(NamedTuple):
    swdown: object
    swddir: object
    swddni: object
    swddif: object
    gsw: object


class SolarCarry(NamedTuple):
    """Seven WRF interpolation parameters and five last surface fluxes.

    All twelve leaves are REAL surface fields, 48 logical bytes per cell.
    Keeping them apart from radiation diagnostics preserves held SWDNB/SWUPB.
    """
    parameters: SolarInterpolationState
    fluxes: SolarSurfaceFluxes


def _backend(xp):
    if xp is None:
        import jax.numpy as jnp
        return jnp
    return xp


def _operations(xp):
    if xp is np:
        barrier=lambda x:x
    else:
        from jax import lax
        barrier=lax.optimization_barrier
    # All inputs/constants are REAL; barriers preserve separate source ops.
    def add(a,b): return barrier(a+b)
    def sub(a,b): return barrier(a-b)
    def mul(a,b): return barrier(a*b)
    def div(a,b): return barrier(a/b)
    def power(a,b): return barrier(xp.power(a,b))
    def log(a): return barrier(xp.log(a))
    return add,sub,mul,div,power,log


def seed_interpolation(shape,*,xp=None):
    xp=_backend(xp)
    zero=xp.zeros(shape,dtype=xp.float32)
    return SolarInterpolationState(*(zero for _ in SolarInterpolationState._fields))


def seed_solar(state,*,xp=None):
    """WRF pre-solve t0: parameters and Solar surface outputs start at zero."""
    xp=_backend(xp)
    parameters=seed_interpolation(state.t_skin.shape,xp=xp)
    zero=xp.zeros(state.t_skin.shape,dtype=xp.float32)
    return SolarCarry(parameters,SolarSurfaceFluxes(*(zero for _ in SolarSurfaceFluxes._fields)))


def reconstruct_swdown(gsw,albedo,*,xp=None):
    """Original radiation_driver.F REAL GSW/(1-ALBEDO) before fitting.

    GSW arrives from the same RRTMG surface down-minus-up result. This differs
    in REAL rounding from simply taking the profile SWDNB. The source has no
    denominator floor; ALBEDO==1 is allowed to expose invalid caller operands.
    """
    xp=_backend(xp)
    _,sub,_,div,_,_=_operations(xp)
    gsw,albedo=(xp.asarray(x,xp.float32) for x in (gsw,albedo))
    return div(gsw,sub(xp.float32(1),albedo))


def update_parameters(state,coszen,coszen_loc,swddir,swdown,*,first_step=False,xp=None):
    """Original update_swinterp_parameters, called on radiation steps only.

    ``coszen`` is the radiation midpoint, ``coszen_loc`` is call-time solar
    geometry. The two clocks must not be interchanged. First-step initialization
    zeros Bx/bb/Gx/gg, as in module_radiation_driver.F:1709, before the update.
    The explicit night branch avoids evaluating inactive fractional powers;
    it does not alter any active solar operand or repair invalid references.
    """
    xp=_backend(xp)
    F=xp.float32
    add,sub,mul,div,power,log=_operations(xp)
    coszen,coszen_loc,swddir,swdown=(xp.asarray(x,F) for x in (coszen,coszen_loc,swddir,swdown))
    old=SolarInterpolationState(*(xp.asarray(x,F) for x in state))
    Bx=xp.where(first_step,F(0),old.Bx)
    Gx=xp.where(first_step,F(0),old.Gx)
    active=(coszen>F(1e-4))&(coszen_loc>F(1e-4))
    c=xp.where(active,coszen,F(1))
    loc=xp.where(active,coszen_loc,F(1))

    def parameters(flux,reference,amplitude):
        first=amplitude<=F(0)
        guess=mul(div(loc,c),flux)
        flux0=xp.where(first,guess,reference)
        cos0=xp.where(first,loc,old.coszen_ref)
        cos0=xp.where(active,cos0,F(1))
        ratio=div(c,cos0)
        numerator=log(div(xp.maximum(F(1),flux),xp.maximum(F(1),flux0)))
        smaller=log(xp.minimum(sub(F(1),F(1e-4)),ratio))
        greater=log(xp.maximum(add(F(1),F(1e-4)),ratio))
        exponent=xp.where(ratio<F(1),div(numerator,smaller),
                         xp.where(ratio>F(1),div(numerator,greater),F(0)))
        exponent=xp.maximum(F(-.5),xp.minimum(F(2.5),exponent))
        factor=div(flux,power(c,exponent))
        return xp.where(active,exponent,F(0)),xp.where(active,factor,F(0))

    bb,Bx=parameters(swddir,old.swddir_ref,Bx)
    gg,Gx=parameters(swdown,old.swdown_ref,Gx)
    # Refs save the new radiation call even when it was night/threshold.
    return SolarInterpolationState(swddir,bb,Bx,swdown,gg,Gx,coszen)


def interpolate_surface(state,coszen_loc,albedo,*,xp=None):
    """Original interp_sw_radiation, with source endpoint fallback and GSW.

    Endpoint slopes -.5,2.5 and exactly0 use the linear cosine ratio, NOT the
    power law. Diffuse is SWDOWN-SWDDIR with no invented nonnegative clamp.
    ALBEDO is the surface-driver entry albedo, not the held radiation albedo.
    """
    xp=_backend(xp)
    F=xp.float32
    _,sub,mul,div,power,_=_operations(xp)
    s=SolarInterpolationState(*(xp.asarray(x,F) for x in state))
    loc,albedo=(xp.asarray(x,F) for x in (coszen_loc,albedo))
    active=(s.coszen_ref>F(1e-4))&(loc>F(1e-4))
    c=xp.where(active,loc,F(1))
    refcos=xp.where(active,s.coszen_ref,F(1))

    def flux(reference,exponent,amplitude):
        linear=(exponent==F(-.5))|(exponent==F(2.5))|(exponent==F(0))
        value=xp.where(linear,mul(div(c,refcos),reference),mul(amplitude,power(c,exponent)))
        return xp.where(active,value,F(0))

    direct=flux(s.swddir_ref,s.bb,s.Bx)
    down=flux(s.swdown_ref,s.gg,s.Gx)
    diffuse=xp.where(active,sub(down,direct),F(0))
    normal=xp.where(active,div(direct,c),F(0))
    net=xp.where(active,mul(down,sub(F(1),albedo)),F(0))
    return SolarSurfaceFluxes(down,direct,normal,diffuse,net)


def solar_step(state,coszen,coszen_loc,swddir,swdown,albedo,*,run_radiation,first_step=False,xp=None):
    """Driver cadence: conditional parameter update, unconditional interp."""
    xp=_backend(xp)
    def refresh(s):
        return update_parameters(s,coszen,coszen_loc,swddir,swdown,first_step=first_step,xp=xp)
    if isinstance(run_radiation,(bool,np.bool_)):
        updated=refresh(state) if run_radiation else state
    elif xp is np:
        raise TypeError("NumPy draft requires a scalar radiation predicate")
    else:
        from jax import lax
        updated=lax.cond(run_radiation,refresh,lambda s:s,state)
    return updated,interpolate_surface(updated,coszen_loc,albedo,xp=xp)


def history_fields(group):
    """Held WRF surface outputs; accept carry or a direct operator result."""
    if group is None:
        return {}
    fluxes=group.fluxes if isinstance(group,SolarCarry) else group
    if not isinstance(fluxes,SolarSurfaceFluxes):
        raise TypeError("Solar history requires SolarCarry or SolarSurfaceFluxes")
    return {name.upper():value for name,value in zip(fluxes._fields,fluxes)}


def validate_restart_solar(group,active,version,shape=None):
    """E78: active Solar must resume all exact REAL parameters/last fluxes."""
    if group is None and not active:
        return
    if not active:
        raise ValueError("restart Solar carry and selection disagree (E78)")
    if version!=1 or not isinstance(group,SolarCarry):
        raise ValueError("restart Solar interpolation schema is missing (E78)")
    if not isinstance(group.parameters,SolarInterpolationState) or not isinstance(group.fluxes,SolarSurfaceFluxes):
        raise ValueError("restart Solar interpolation groups are incomplete (E78)")
    leaves=(*group.parameters,*group.fluxes)
    if any(value is None or getattr(value,"dtype",None)!=np.dtype("float32") for value in leaves):
        raise ValueError("restart Solar interpolation lacks REAL leaves (E78)")
    shapes={value.shape for value in leaves}
    if len(shapes)!=1 or len(group.parameters.coszen_ref.shape)!=2:
        raise ValueError("restart Solar interpolation has inconsistent surface shapes (E78)")
    if shape is not None and group.parameters.coszen_ref.shape!=tuple(shape):
        raise ValueError("restart Solar interpolation differs from the surface grid (E78)")
