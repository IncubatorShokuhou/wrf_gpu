"""Additional WRF Noah-MP soil stress / surface evaporation resistance choices."""
import jax.numpy as jnp


def bats_infiltration(qinsur,smc,smcmax,dzs,fcr1):
    """SOILWATER:7399-7413, original depth-weighted BATS surface runoff."""
    r=lambda value:jnp.asarray(value,jnp.float32)
    qinsur,fcr1=r(qinsur),r(fcr1)
    total=jnp.zeros_like(qinsur);depth=jnp.zeros_like(qinsur)
    active=jnp.ones_like(qinsur,dtype=bool)
    for k in range(4):
        depth=jnp.where(active,depth+r(dzs[k]),depth)
        total=jnp.where(active,total+r(smc[k])/r(smcmax[k])*r(dzs[k]),total)
        active=active&(depth<2.)
    fraction=jnp.maximum(.01,total/depth)**4.0
    runoff=qinsur*((1.-fcr1)*fraction+fcr1)
    runoff=jnp.where(qinsur>0.,runoff,jnp.zeros_like(qinsur))
    infiltration=jnp.where(qinsur>0.,qinsur-runoff,jnp.zeros_like(qinsur))
    return infiltration,runoff


def supercooled_water_koren(tkelv,smc,sh2o,bexp,smcmax,psisat):
    """Original FRH2O:5814-5946; ten masked Newton iterations and Flerchinger fallback."""
    tkelv,smc,sh2o,bexp,smcmax,psisat=(jnp.asarray(value,jnp.float32)
        for value in (tkelv,smc,sh2o,bexp,smcmax,psisat))
    tfrz=jnp.asarray(273.16,jnp.float32)
    cold=tkelv <= tfrz-jnp.asarray(.001,jnp.float32)
    # The scalar WRF cold branch does not evaluate log at/above freezing.
    temperature=jnp.where(cold,tkelv,jnp.asarray(270.,jnp.float32))
    bx=jnp.minimum(bexp,5.5)
    swl=smc-sh2o
    swl=jnp.where(swl>smc-.02,smc-.02,swl)
    swl=jnp.where(swl<0.,0.,swl)
    done=jnp.zeros_like(cold)
    for _ in range(10):
        df=jnp.log((psisat*9.80616/333600.)*((1.+8.*swl)**2.0)*(smcmax/(smc-swl))**bx)-jnp.log(-(temperature-tfrz)/temperature)
        denom=2.*8./(1.+8.*swl)+bx/(smc-swl)
        swlk=swl-df/denom
        swlk=jnp.where(swlk>smc-.02,smc-.02,swlk)
        swlk=jnp.where(swlk<0.,0.,swlk)
        delta=jnp.abs(swlk-swl)
        swl=jnp.where(done,swl,swlk)
        done=done|(delta<=.005)
    fk=((333600./(9.80616*(-psisat)))*((temperature-tfrz)/temperature))**(-1./bx)*smcmax
    fk=jnp.where(fk<.02,.02,fk)
    free=jnp.where(done,smc-swl,jnp.minimum(fk,smc))
    return jnp.where(cold,free,smc)


def soil_moisture_stress(sh2o,smcwlt,smcref,smcmax,psisat,bexp,*,opt_btr):
    """ENERGY:2152-2164, returning the clipped REAL GX for one rooted layer."""
    sh2o,smcwlt,smcref,smcmax,psisat,bexp=(jnp.asarray(value,jnp.float32)
        for value in (sh2o,smcwlt,smcref,smcmax,psisat,bexp))
    if opt_btr==1:
        gx=(sh2o-smcwlt)/(smcref-smcwlt)
    elif opt_btr in (2,3):
        psi=jnp.maximum(-150.,-psisat*(jnp.maximum(.01,sh2o)/smcmax)**(-bexp))
        if opt_btr==2:
            gx=(1.-psi/(-150.))/(1.+psisat/(-150.))
        else:
            gx=1.-jnp.exp(-5.8*jnp.log((-150.)/psi))
    else:
        raise ValueError("Noah-MP opt_btr supports 1 (Noah), 2 (CLM), 3 (SSiB)")
    return jnp.minimum(1.,jnp.maximum(0.,gx))


def evaporation_resistance(sh2o,smcmax,smcwlt,bexp,zsoil1,rsurf_exp,fsno,*,opt_rsf,rsurf_snow=None):
    """ENERGY:2175-2196; caller applies the shared dry/urban overrides afterwards."""
    sh2o,smcmax,smcwlt,bexp,zsoil1,rsurf_exp,fsno=(jnp.asarray(value,jnp.float32)
        for value in (sh2o,smcmax,smcwlt,bexp,zsoil1,rsurf_exp,fsno))
    if opt_rsf in (2,3):
        bevap=jnp.maximum(0.,sh2o/smcmax)
        coefficient=4.225 if opt_rsf==2 else 6.
        return fsno*1.+(1.-fsno)*jnp.exp(8.25-coefficient*bevap)
    if opt_rsf not in (1,4):
        raise ValueError("Noah-MP opt_rsf supports 1, 2, 3 or 4")
    denominator=jnp.asarray(2.71828,jnp.float32)-jnp.asarray(1.,jnp.float32)
    length=(-zsoil1)*(jnp.exp((1.-jnp.minimum(1.,sh2o/smcmax))**rsurf_exp)-1.)/denominator
    diffusivity=2.2e-5*smcmax*smcmax*(1.-smcwlt/smcmax)**(2.+3./bexp)
    rsurf=length/diffusivity
    if opt_rsf==4:
        if rsurf_snow is None:
            raise ValueError("opt_rsf=4 requires RSURF_SNOW from MPTABLE.TBL")
        rsurf_snow=jnp.asarray(rsurf_snow,jnp.float32)
        rsurf=1./(fsno*(1./rsurf_snow)+(1.-fsno)*(1./jnp.maximum(rsurf,.001)))
    return rsurf
