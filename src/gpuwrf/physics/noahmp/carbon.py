"""Pristine Noah-MP CARBON/CO2FLUX for DVEG 2/5/6 (not the crop models)."""
from gpuwrf.physics.noahmp.land_columns import layer_row
from typing import NamedTuple
import jax.numpy as jnp


class CarbonState(NamedTuple):
    lfmass: object
    rtmass: object
    stmass: object
    wood: object
    stblcp: object
    fastcp: object


class CarbonParameters(NamedTuple):
    sla: object
    folnmx: object
    arm: object
    rmf25: object
    rmr25: object
    rms25: object
    wdpool: object
    wrrat: object
    ltovrc: object
    tdlef: object
    dilefw: object
    dilefc: object
    fragr: object
    tmin: object
    mrp: object


class CarbonStatic(NamedTuple):
    parameters: CarbonParameters
    initial: CarbonState


def root_zone_water(smc,smcmax,dzs,zsoil,nroot,no_veg):
    """CARBON:10008-10011; sequential root-depth weighting in WRF order."""
    smc,smcmax,dzs,zsoil=(jnp.asarray(v,jnp.float32) for v in (smc,smcmax,dzs,zsoil))
    nroot=jnp.asarray(nroot,jnp.int32)
    from gpuwrf.physics.noahmp.land_columns import pick_layer
    depth=-pick_layer(jnp.broadcast_to(zsoil,smc.shape),jnp.clip(nroot-1,0,3))
    depth=jnp.where(no_veg,1.,depth)
    total=jnp.zeros_like(layer_row(smc,0))
    for k in range(4):
        total=jnp.where(nroot>k,total+layer_row(smc,k)/layer_row(smcmax,k)*layer_row(dzs,k)/depth,total)
    return total


def carbon_flux(state,p,lai,sai,psn,btran,foln,tv,stc1,wroot,igs,evergreen,no_veg,dt):
    """CO2FLUX:10030-10298, with CARBON's early nonvegetation return."""
    cast=lambda v:jnp.asarray(v,jnp.float32)
    state=CarbonState(*(cast(v) for v in state));p=CarbonParameters(*(cast(v) for v in p))
    lai,sai,psn,btran,foln,tv,stc1,wroot,igs,dt=(cast(v) for v in (lai,sai,psn,btran,foln,tv,stc1,wroot,igs,dt))
    lfmass,rtmass,stmass,wood,stblcp,fastcp=state
    lapm=jnp.where(no_veg,1.,p.sla/1000.)
    wstres=1.-btran
    sapm=cast(3.)*cast(.001)
    lfmsmn=.05/lapm;stmsmn=cast(.05)/sapm
    rf=jnp.where(igs==0.,cast(.5),cast(1.))
    fnf=jnp.minimum(foln/jnp.maximum(1.e-6,p.folnmx),1.)
    tf=p.arm**((tv-298.16)/10.)
    resp=p.rmf25*tf*fnf*lai*rf*(1.-wstres)
    rsleaf=jnp.minimum((lfmass-lfmsmn)/dt,resp*12.e-6)
    rsroot=p.rmr25*(rtmass*.001)*tf*rf*12.e-6
    rsstem=p.rms25*((stmass-stmsmn)*.001)*tf*rf*12.e-6
    rswood=3.e-10*jnp.exp(.08*(tv-298.16))*wood*p.wdpool
    carbfx=psn*12.e-6
    leafpt=jnp.exp(.01*(1.-jnp.exp(.75*lai))*lai)
    leafpt=jnp.where(evergreen,jnp.exp(.01*(1.-jnp.exp(.5*lai))*lai),leafpt)
    nonlef=1.-leafpt
    stempt=lai/10.*leafpt
    leafpt=leafpt-stempt
    wood_safe=jnp.where(wood>1.e-6,wood,1.)
    woodf=jnp.where(wood>1.e-6,(1.-jnp.exp(-.9*(p.wrrat*rtmass/wood_safe))/.9)*p.wdpool,p.wdpool)
    rootpt=nonlef*(1.-woodf);woodpt=nonlef*woodf
    lftovr=p.ltovrc*5.e-7*lfmass;sttovr=p.ltovrc*5.e-7*stmass
    rttovr=2.e-8*rtmass;wdtovr=9.5e-10*wood
    sc=jnp.exp(-.3*jnp.maximum(0.,tv-p.tdlef))*(lfmass/120.)
    sd=jnp.exp((wstres-1.)*100.)
    dielf=lfmass*1.e-6*(p.dilefw*sd+p.dilefc*sc)
    diest=stmass*1.e-6*(p.dilefw*sd+p.dilefc*sc)
    grleaf=jnp.maximum(0.,p.fragr*(leafpt*carbfx-rsleaf))
    grstem=jnp.maximum(0.,p.fragr*(stempt*carbfx-rsstem))
    grroot=jnp.maximum(0.,p.fragr*(rootpt*carbfx-rsroot))
    grwood=jnp.maximum(0.,p.fragr*(woodpt*carbfx-rswood))
    addlf=jnp.maximum(0.,leafpt*carbfx-grleaf-rsleaf)
    addst=jnp.maximum(0.,stempt*carbfx-grstem-rsstem)
    addlf=jnp.where(tv<p.tmin,0.,addlf);addst=jnp.where(tv<p.tmin,0.,addst)
    lfdel=(lfmass-lfmsmn)/dt;stdel=(stmass-stmsmn)/dt
    dielf=jnp.minimum(dielf,lfdel+addlf-lftovr)
    diest=jnp.minimum(diest,stdel+addst-sttovr)
    nppl=jnp.maximum(addlf,-lfdel);npps=jnp.maximum(addst,-stdel)
    nppr=rootpt*carbfx-rsroot-grroot;nppw=woodpt*carbfx-rswood-grwood
    lfmass=lfmass+(nppl-lftovr-dielf)*dt
    stmass=stmass+(npps-sttovr-diest)*dt
    rtmass=rtmass+(nppr-rttovr)*dt
    rttovr=jnp.where(rtmass<0.,nppr,rttovr)
    rtmass=jnp.where(rtmass<0.,0.,rtmass)
    wood=(wood+(nppw-wdtovr)*dt)*p.wdpool
    fastcp=fastcp+(rttovr+lftovr+sttovr+wdtovr+dielf+diest)*dt
    fst=2.**((stc1-283.16)/10.)
    fsw=wroot/(.20+wroot)*.23/(.23+wroot)
    rssoil=fsw*fst*p.mrp*jnp.maximum(0.,fastcp*.001)*12.e-6
    stablc=.1*rssoil
    fastcp=fastcp-(rssoil+stablc)*dt
    stblcp=stblcp+stablc*dt
    gpp=carbfx;npp=nppl+nppw+nppr+npps
    autors=rsroot+rswood+rsleaf+rsstem+grleaf+grroot+grwood+grstem
    heters=.9*rssoil
    nee=(autors+heters-gpp)*44./12.
    totsc=fastcp+stblcp;totlb=lfmass+rtmass+stmass+wood
    lai=jnp.maximum(lfmass*lapm,.05);sai=jnp.maximum(stmass*sapm,.05)
    clear=lambda v:jnp.where(no_veg,0.,v)
    updated=CarbonState(*(clear(v) for v in (lfmass,rtmass,stmass,wood,stblcp,fastcp)))
    diagnostics=dict(zip(("GPP","NPP","NEE","AUTORS","HETERS","TOTSC","TOTLB"),
                         (clear(v) for v in (gpp,npp,nee,autors,heters,totsc,totlb))))
    return updated,clear(lai),clear(sai),diagnostics


def carbon_column(pools,p,lai,sai,psn,btran,foln,tv,stc1,smc,smcmax,dzs,zsoil,nroot,veg,igs,dt):
    """One native phase: root weighting plus carbon; only public WRF diagnostics leave it."""
    no_veg=(veg==13)|(veg==15)|(veg==16)|(veg==17)
    wroot=root_zone_water(smc,smcmax,dzs,zsoil,nroot,no_veg)
    pools,lai,sai,diag=carbon_flux(pools,p,lai,sai,psn,btran,foln,tv,stc1,wroot,igs,veg==2,no_veg,dt)
    return pools,lai,sai,{key:diag[key] for key in ("GPP","NPP","NEE")}
