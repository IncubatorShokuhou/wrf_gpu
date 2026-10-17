"""Literal FP32 WRF mp=7 column, including independent graupel and hail.

Provenance: phys/module_mp_gsfcgce_4ice_nuwrf.F, pristine WRF 4.7.1.
No mp=97 code or equations are substituted. The five sedimentation passes
retain original order and adaptive Courant substeps. Independent saturation
levels are vectorized, with original branch ownership and REAL arithmetic.
"""
from jax import lax
import jax.numpy as jnp
from .support import C, F, shift_up
from .equations import sgmap, vqrqi, saticel_s


def _fall_species(q, rho, dz, z, topo, dt, velocity, _physical_nz=None):
    """WRF fall_flux one precipitating category; k=kte is not sedimented."""
    nz = q.shape[0]
    physical_nz = nz if _physical_nz is None else _physical_nz
    kindex = jnp.arange(nz)
    distance = _distance(z,topo)
    zero = F(0.)

    def condition(state):
        return state[-1]

    def body(state):
        q, precip, sediment, elapsed, delta, _ = state
        active = (q > C['cmin']) & (kindex < physical_nz-1)
        low = jnp.min(jnp.where(active, kindex, nz-1))
        high = jnp.max(jnp.where(active, kindex, -1))
        speed = jnp.where(active, velocity(q), zero)
        # Fortran rain checks positive speed; cloud ice checks >1e-20.
        # All five terminal-speed equations are nonnegative.
        courant = jnp.where(active & (speed > zero), F(.9)*distance/speed, delta)
        delta = jnp.minimum(delta, jnp.min(courant))
        proceed = high >= low
        elapsed = elapsed + jnp.where(proceed, delta, zero)
        last = elapsed >= dt
        delta = jnp.where(proceed & last, dt + delta - elapsed, delta)

        def downward(offset, carry):
            q, sediment, fluxin = carry
            k = nz - 1 - offset
            within = proceed & (k <= high) & (k >= low)
            fluxout = _point(rho,k) * _point(speed,k) * _point(q,k)
            flux = (fluxin - fluxout) / _point(rho,k) / _point(dz,k)
            changed = jnp.maximum(zero, _point(q,k) + delta * flux)
            q = jnp.where(within & (kindex == k), changed, q)
            fluxin = jnp.where(within, fluxout, fluxin)
            sediment = jnp.where(within & (kindex == k), sediment + fluxin, sediment)
            return q, sediment, fluxin

        q, sediment, fluxin = lax.fori_loop(0, nz, downward, (q, sediment, zero))
        precip = precip + jnp.where(proceed & (low == 0), fluxin * delta, zero)
        below = jnp.maximum(low - 1, 0)
        value = _point(q,below) + delta * fluxin / _point(rho,below) / _point(dz,below)
        value = jnp.maximum(zero, value)
        q = jnp.where(proceed & (low > 0) & (kindex == below), value, q)
        return q, precip, sediment, elapsed, delta, proceed & ~last

    init = (q, zero, jnp.zeros_like(q), zero, dt, jnp.asarray(True))
    q, precip, sediment, _, _, _ = lax.while_loop(condition, body, init)
    return q, precip, sediment


def column(theta, qv, qc, qr, qi, qs, qg, qh, rho, exner, pressure, dz,
           z, w, dt, dx=3000., xland=1., topo=0., itimestep=1,
           diagnostics=None, _physical_nz=None):
    """Advance one bottom-up column; all units SI, mixing ratios kg/kg.

    Returns a mapping with eight updated prognostics, step precipitation mm,
    SR and all original six categorical radii/latent budget diagnostics plus
    five 3D precipitation fluxes. Accumulated latent-budget input is optional
    on a first step and required for continuation by the caller.
    """
    theta,qv,qc,qr,qi,qs,qg,qh,rho,exner,pressure,dz,z,w = (
        jnp.asarray(value,jnp.float32) for value in
        (theta,qv,qc,qr,qi,qs,qg,qh,rho,exner,pressure,dz,z,w))
    dt,dx,xland,topo=map(F,(dt,dx,xland,topo))
    sediment = lambda *a: _fall_species(*a, _physical_nz=_physical_nz)
    fv=jnp.sqrt(F(1.29)/rho)
    r00=rho*F(.001)
    tair=theta*exner
    tairc=tair-C['t0']
    k=jnp.arange(theta.shape[0])+1
    nz=theta.shape[0] if _physical_nz is None else _physical_nz

    qr, rain, rsed = sediment(qr,rho,dz,z,topo,dt,
        lambda q: vqrqi(1,r00,fv,q,qc,tair)*F(.01))

    def snow_speed(q):
        neighbors=(k<nz-2)&(tairc>=-5)
        qh2=jnp.where(neighbors,shift_up(qh,'qh_original'),qh)
        qg2=jnp.where(neighbors,shift_up(qg,'qg_original'),qg)
        ftns0=sgmap(1,q,qg,qg2,qh,qh2,r00,tairc)
        fros0=sgmap(3,q,qg,qg2,qh,qh2,r00,tairc)
        ftns=ftns0**C['bsq'];fros=fros0**C['bsq']
        return jnp.maximum((C['vsc']*fv)*(r00*q)**C['bsq']/ftns/fros,F(0))*F(.01)
    qs,snow,ssed=sediment(qs,rho,dz,z,topo,dt,snow_speed)

    def graupel_speed(q):
        neighbors=(k<nz-2)&(tairc>=-5)
        qh2=jnp.where(neighbors,shift_up(qh,'qh_original'),qh)
        qg2=jnp.where(neighbors,shift_up(q,'qg_current'),q)
        ftng0=sgmap(2,qs,q,qg2,qh,qh2,r00,tairc)
        ftng=ftng0**C['bgq']
        speed=jnp.maximum((C['vgc']*fv)*(r00*q)**C['bgq']/ftng,F(0))*F(.01)
        ftng2=ftng0**C['bgq2']
        speed2=jnp.maximum((C['vgc2']*fv)*(r00*q)**C['bgq2']/ftng2,F(0))*F(.01)
        return jnp.where(q>C['qrog2'],speed2,speed)
    qg,graupel,gsed=sediment(qg,rho,dz,z,topo,dt,graupel_speed)
    qh,hail,hsed=sediment(qh,rho,dz,z,topo,dt,
        lambda q:jnp.maximum((C['vhc']/jnp.sqrt(r00))*(q*r00)**C['bhq'],F(0))*F(.01))
    qi,ice,ised=sediment(qi,rho,dz,z,topo,dt,
        lambda q:jnp.where((rho*F(1000)*q)>=F(1.e-6),vqrqi(2,r00,fv,q,qc,tair),F(0))*F(.01))
    _prepare_saticel(qg,qh)
    outputs=saticel_s(dt,dx,theta,qv,qc,qr,qi,qs,qg,qh,rho,exner,pressure,w,xland,itimestep,diagnostics,_physical_nz)
    result=dict(outputs)
    # module_microphysics_driver.F:2413-2415, Registry PHYS_TOT history.
    result["phys_tot"] = (outputs["physc"] + outputs["physe"] + outputs["physd"]
                          + outputs["physs"] + outputs["physm"] + outputs["physf"])
    result.update(rainnc=rain+snow+graupel+ice+hail, snownc=snow,
                  graupelnc=graupel,hailnc=hail,rain_other=rain+ice,
                  preci3d=ised,precs3d=ssed,precg3d=gsed,prech3d=hsed,precr3d=rsed)
    result['sr']=jnp.where(result['rainnc']>F(0),(snow+graupel+ice+hail)/result['rainnc'],F(0))
    return result


def _point(value,index):
    return value[index]

def _distance(z,topo):
    return z-jnp.concatenate((jnp.asarray(topo).reshape(1),z[:-1]))

def _prepare_saticel(qg,qh):
    pass
