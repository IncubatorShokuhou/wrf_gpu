"""Additional pristine Noah-MP canopy resistance and exchange choices."""
import jax.numpy as jnp


def jarvis_resistance(par,temperature,btran,eah,pressure,fveg,rsmin,rsmax,rgl,hs,topt):
    """CANRES + CALHUM:5141-5255, including WRF's -999.99 PSN sentinel."""
    par,temperature,btran,eah,pressure,fveg,rsmin,rsmax,rgl,hs,topt=(jnp.asarray(v,jnp.float32)
        for v in (par,temperature,btran,eah,pressure,fveg,rsmin,rsmax,rgl,hs,topt))
    r=lambda v:jnp.asarray(v,jnp.float32)
    scaled=par/jnp.maximum(fveg,1.e-6)
    q2=.622*eah/(pressure-.378*eah)
    q2=q2/(1.-q2)
    es=.611*jnp.exp((r(2.501e6)/r(461.))*(r(1.)/r(273.15)-1./temperature))
    pressure_kpa=pressure*.001
    qsat=.622*es/(pressure_kpa-es)
    qsat=qsat*1000.
    qsat=qsat/1000.
    ff=2.*scaled/rgl
    rcs=jnp.maximum((ff+rsmin/rsmax)/(1.+ff),.0001)
    rct=jnp.maximum(1.-.0016*(topt-temperature)**2.0,.0001)
    rcq=jnp.maximum(1./(1.+hs*jnp.maximum(0.,qsat-q2)),.01)
    return rsmin/(rcs*rct*rcq*btran),jnp.full_like(par,-999.99)


def sfcdif2(it,z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar2,ustar,czil):
    """SFCDIF2:4747-4948; one stability sweep, preserving the caller's WSTAR2."""
    z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar2,ustar,czil=(jnp.asarray(v,jnp.float32)
        for v in (z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar2,ustar,czil))
    r=lambda v:jnp.asarray(v,jnp.float32)
    wwst2=r(1.2)*r(1.2)
    vkrm=r(.4)
    beta=r(1.)/r(270.)
    btg=beta*r(9.80616)
    elfc=vkrm*btg
    wold=r(.15);wnew=r(1.)-wold
    pihf=r(3.14159265)/r(2.)
    power=r(2.)/r(3.)
    zilfc=-czil*vkrm*258.2
    zu=z0
    rdz=1./zlm
    cxch=.001*rdz
    dthv=thlm-thz0
    du2=jnp.maximum(speed*speed,.0001)
    btgh=btg*1000.
    def wstar(akhs):
        arg=btgh*akhs*dthv
        return jnp.where(arg!=0.,wwst2*jnp.abs(arg)**power,jnp.zeros_like(arg))
    if it==1:
        wstar2=wstar(akhs)
        ustar=jnp.maximum(jnp.sqrt(akms*jnp.sqrt(du2+wstar2)),.07)
        rlmo=elfc*akhs*dthv/ustar**3
    zt=jnp.maximum(1.e-6,jnp.exp(zilfc*jnp.sqrt(ustar*z0))*z0)
    zslu=zlm+zu;zslt=zlm+zt
    rlogu=jnp.log(zslu/zu);rlogt=jnp.log(zslt/zt)
    zetalt=jnp.maximum(zslt*rlmo,-5.)
    rlmo=zetalt/zslt
    zetalu=zslu*rlmo;zetau=zu*rlmo;zetat=zt*rlmo
    unstable=rlmo<0.
    def fourth(z):return jnp.sqrt(jnp.sqrt(1.-16.*jnp.where(unstable,z,0.)))
    xlu,xlt,xu,xt=(fourth(z) for z in (zetalu,zetalt,zetau,zetat))
    def pspmu(x):return -2.*jnp.log((x+1.)*.5)-jnp.log((x*x+1.)*.5)+2.*jnp.arctan(x)-pihf
    def psphu(x):return -2.*jnp.log((x*x+1.)*.5)
    simm_u=pspmu(xlu)-pspmu(xu)+rlogu
    simh_u=psphu(xlt)-psphu(xt)+rlogt
    zetalu=jnp.minimum(zetalu,1.);zetalt=jnp.minimum(zetalt,1.)
    zetau=jnp.minimum(zetau,1./(zslu/zu));zetat=jnp.minimum(zetat,1./(zslt/zt))
    simm=jnp.where(unstable,simm_u,5.*zetalu-5.*zetau+rlogu)
    simh=jnp.where(unstable,simh_u,5.*zetalt-5.*zetat+rlogt)
    ustar=jnp.maximum(jnp.sqrt(akms*jnp.sqrt(du2+wstar2)),.07)
    zt=jnp.maximum(1.e-6,jnp.exp(zilfc*jnp.sqrt(ustar*z0))*z0)
    zslt=zlm+zt
    ustark=ustar*vkrm
    simm=jnp.where(simm<1.e-6,1.e-6,simm)
    akms=jnp.maximum(ustark/simm,cxch)
    simh=jnp.where(simh<1.e-6,1.e-6,simh)
    akhs=jnp.maximum(ustark/simh,cxch)
    wstar2=wstar(akhs)
    rlmn=elfc*akhs*dthv/ustar**3
    rlmo=rlmo*wold+rlmn*wnew
    return akms,akhs,rlmo,wstar2,ustar
