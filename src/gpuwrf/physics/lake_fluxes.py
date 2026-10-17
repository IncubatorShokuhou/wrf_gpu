"""WRF module_sf_lake ShalLakeFluxes and surface-layer DOUBLE arithmetic."""
from __future__ import annotations

from typing import NamedTuple
import jax
from jax import numpy as jnp

from .lake_thermodynamics import lake_monin_obukhov_initial

TFRZ, CPICE, CPLIQ, HFUS = 273.16, 2117.27, 4188.0, 333700.0
HVAP, HSUB, GRAV, VKC = 2501000.0, 2834700.0, 9.80616, 0.4
SB, RAIR, CPAIR = 5.67e-8, 287.0423, 1004.64


def real(value):
    """Unsuffixed Fortran REAL literal promoted into an explicit r8 island."""
    return jnp.asarray(value, jnp.float32).astype(jnp.float64)


TKWAT, TKICE, TKAIR = 0.6000000238418579, 2.2899999618530273, 0.023000000044703484


def at(a, index):
    return jnp.take_along_axis(a, jnp.asarray(index, jnp.int32)[..., None], axis=-1)[..., 0]


def set_at(a, index, value):
    """Independent column scatter with a last-axis vertical coordinate."""
    return jnp.where(jnp.arange(a.shape[-1]) == index[..., None], value[..., None], a)


def qsat(t, p):
    # Coefficients in the pristine routine have unsuffixed REAL initializers.
    coeff = [
        [6.11213476,.444007856,.143064234e-1,.264461437e-3,.305903558e-5,.196237241e-7,.892344772e-10,-.373208410e-12,.209339997e-15],
        [.444017302,.286064092e-1,.794683137e-3,.121211669e-4,.103354611e-6,.404125005e-9,-.788037859e-12,-.114596802e-13,.381294516e-16],
        [6.11123516,.503109514,.188369801e-1,.420547422e-3,.614396778e-5,.602780717e-7,.387940929e-9,.149436277e-11,.262655803e-14],
        [.503277922,.377289173e-1,.126801703e-2,.249468427e-4,.313703411e-6,.257180651e-8,.133268878e-10,.394116744e-13,.498070196e-16],
    ]
    td = jnp.minimum(jnp.maximum(t-TFRZ, -75.0), 100.0)
    def poly(values):
        c = real(values)
        result = c[-1]
        for i in range(7,-1,-1):
            result = c[i]+td*result
        return result
    es = jnp.where(td >= 0, poly(coeff[0]), poly(coeff[2]))*100.0
    esdt = jnp.where(td >= 0, poly(coeff[1]), poly(coeff[3]))*100.0
    vp = 1.0/(p-real(.378)*es)
    vp1 = real(.622)*vp
    vp2 = vp1*vp
    return es, esdt, es*vp1, esdt*vp2*p


def stability(zeta, momentum=False):
    chik2 = jnp.sqrt(1.0-16.0*zeta)
    if momentum:
        chik = jnp.sqrt(chik2)
        return 2.0*jnp.log((1.0+chik)*.5)+jnp.log((1.0+chik2)*.5)-2.0*jnp.arctan(chik)+jnp.pi*.5
    return 2.0*jnp.log((1.0+chik2)*.5)


def friction(height, roughness, obu, um):
    """WRF caller has identical humidity/temperature heights and roughness."""
    zeta = height/obu
    def profile(h, momentum=False):
        zt = h/obu
        transition = 1.574 if momentum else .465
        # Stability functions are only defined on the unstable branch.
        negative = jnp.where(zt < 0, zt, -0.01)
        z0_unstable = jnp.where(obu < 0, roughness/obu, -0.01)
        unstable = jnp.log(h/roughness)-stability(negative,momentum)+stability(z0_unstable,momentum)
        extreme = jnp.log(-transition*jnp.where(obu < 0, obu, -1.0)/roughness)-stability(-transition,momentum)+stability(z0_unstable,momentum)
        if momentum:
            extreme = extreme+1.14*((-negative)**.333-transition**.333)
        else:
            extreme = extreme+.8*(transition**(-.333)-(-negative)**(-.333))
        stable = jnp.log(h/roughness)+5.0*zt-5.0*roughness/obu
        very_stable = jnp.log(jnp.where(obu > 0, obu, 1.0)/roughness)+5.0-5.0*roughness/obu+(5.0*jnp.log(jnp.where(zt > 0,zt,1.0))+zt-1.0)
        return VKC/jnp.where(zt < -transition, extreme, jnp.where(zt < 0, unstable, jnp.where(zt <= 1,stable,very_stable)))
    ustar = um*profile(height,True)
    temp1 = profile(height)
    temp12m = profile(2.0+roughness)
    return ustar,temp1,temp12m


class LakeForcing(NamedTuple):
    temperature: jax.Array
    pbot: jax.Array
    psfc: jax.Array
    height: jax.Array
    specific_humidity: jax.Array
    u: jax.Array
    v: jax.Array
    longwave: jax.Array
    precipitation: jax.Array
    absorbed_solar: jax.Array
    latitude: jax.Array
    dt: jax.Array


class LakeFluxes(NamedTuple):
    ground_temperature: jax.Array
    sensible: jax.Array
    latent: jax.Array
    ground: jax.Array
    gnet: jax.Array
    evaporation: jax.Array
    t2: jax.Array
    q2: jax.Array
    ws: jax.Array
    ks: jax.Array


def surface_fluxes(s, f):
    t, p, height, q = f.temperature, f.pbot, f.height, f.specific_humidity
    ground = s.t_grnd2d
    forc_th = t*(f.psfc/p)**(RAIR/CPAIR)
    vp = q*p/(real(.622)+real(.378)*q)
    rho = (p-real(.378)*vp)/(RAIR*t)
    snl = s.snl2d.astype(jnp.int32)
    top = snl+5  # WRF snl+1 shifted by4 in snow/soil arrays
    snow = snl < 0
    beta = jnp.where(snow,1.0,.4)
    dzsur = jnp.where(snow,at(s.dz3d,top)/2.0,s.dz_lake3d[...,0]/2.0)
    _,_,qs,qst = qsat(ground,p)
    thm, thv = t+.0098*height, forc_th*(1.0+.61*q)
    roughness = jnp.where(ground >= TFRZ,.001,jnp.where(snl == 0,.005,.0024))
    ur = jnp.maximum(1.0,jnp.sqrt(f.u*f.u+f.v*f.v))
    dth,dqh = thm-ground,q-qs
    um,obu = lake_monin_obukhov_initial(ur,thv,dth*(1.0+.61*q)+.61*forc_th*dqh,height,roughness)
    # niters=3; a single lake pft cannot reach three sign reversals before
    # the final pass, so WRF's filter never drops it before an executed pass.
    for _ in range(3):
        ustar,temp1,temp12m = friction(height,roughness,obu,um)
        before = ground
        open_water = (ground > TFRZ)&(s.t_lake3d[...,0] > TFRZ)&~snow
        bw = (at(s.h2osoi_ice3d,top)+at(s.h2osoi_liq3d,top))/at(s.dz3d,top)
        snow_tk = TKAIR+(7.75e-5*bw+1.105e-6*bw*bw)*(TKICE-TKAIR)
        tksur = jnp.where(open_water,s.savedtke12d,jnp.where(~snow,TKICE,snow_tk))
        tsur = jnp.where(snow,at(s.t_soisno3d,top),s.t_lake3d[...,0])
        rah = 1.0/(temp1*ustar)
        raw = rah
        htvp = jnp.where(ground > TFRZ,HVAP,HSUB)
        st = .97*SB*before*before*before
        ax = beta*f.absorbed_solar+.97*f.longwave+3.0*st*before+rho*CPAIR/rah*thm-htvp*rho/raw*(qs-qst*before-q)+tksur*tsur/dzsur
        bx = 4.0*st+rho*CPAIR/rah+htvp*rho/raw*qst+tksur/dzsur
        ground = ax/bx
        sensible = rho*CPAIR*(ground-thm)/rah
        evap = rho*(qs+qst*(ground-before)-q)/raw
        _,_,qs,qst = qsat(ground,p)
        dth,dqh = thm-ground,q-qs
        tvstar = temp1*dth*(1.0+.61*q)+.61*forc_th*temp1*dqh
        zeta = height*VKC*GRAV*tvstar/(ustar**2*thv)
        stable = zeta >= 0
        zeta = jnp.where(stable,jnp.minimum(2.,jnp.maximum(zeta,.01)),jnp.maximum(-100.,jnp.minimum(zeta,-.01)))
        wc = (-GRAV*ustar*jnp.where(stable,-1.,tvstar)*1000.0/thv)**.333
        um = jnp.where(stable,jnp.maximum(ur,.1),jnp.sqrt(ur*ur+wc*wc))
        obu = height/zeta
    freeze_surface = ((s.h2osno2d > .5)|(s.t_lake3d[...,0] <= TFRZ))&(ground > TFRZ)
    convection = ((s.t_lake3d[...,0] > ground)&(ground > 277.))|((s.t_lake3d[...,0] < ground)&(s.t_lake3d[...,0] > TFRZ)&(ground < 277.))
    corrected = freeze_surface|convection
    original = ground
    ground = jnp.where(freeze_surface,TFRZ,jnp.where(convection,s.t_lake3d[...,0],ground))
    sensible = jnp.where(corrected,rho*CPAIR*(ground-thm)/rah,sensible)
    evap = jnp.where(corrected,rho*(qs+qst*(ground-original)-q)/raw,evap)
    htvp = jnp.where(ground > TFRZ,HVAP,HSUB)
    lwout = (1.0-.97)*f.longwave+.97*SB*ground**4
    latent = htvp*evap
    soil_ground = f.absorbed_solar+f.longwave-lwout-sensible-latent
    gnet = beta*f.absorbed_solar+f.longwave-(lwout+sensible+latent)
    t2 = thm+temp1*dth*(1.0/temp12m-1.0/temp1)
    q2 = q+temp1*dqh*(1.0/temp12m-1.0/temp1)
    u2 = jnp.maximum(.1,ustar/VKC*jnp.log(2.0/roughness))
    return LakeFluxes(ground,sensible,latent,soil_ground,gnet,evap,t2,q2,1.2e-3*u2,6.6*jnp.sqrt(jnp.abs(jnp.sin(f.latitude)))*u2**(-1.84))
