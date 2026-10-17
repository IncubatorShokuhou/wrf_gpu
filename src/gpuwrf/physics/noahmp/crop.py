"""Liu crop: pristine CARBON_CROP/CO2FLUX_CROP/GROWING_GDD (WRF 4.7.1)."""
from typing import NamedTuple
import jax.numpy as jnp
from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters,root_zone_water,carbon_column


class CropCarbonState(NamedTuple):
    lfmass: object
    rtmass: object
    stmass: object
    wood: object
    stblcp: object
    fastcp: object
    grain: object
    gdd: object
    pgs: object


class CropParameters(NamedTuple):
    pltd: object
    hsd: object
    gddtbase: object
    gddtcut: object
    gdds1: object
    gdds2: object
    gdds3: object
    gdds4: object
    gdds5: object
    q10mr: object
    foln_mx: object
    lefreez: object
    fra_gr: object
    lfmr25: object
    stmr25: object
    rtmr25: object
    grainmr25: object
    bio2lai: object
    mrp: object
    dile_fc: object
    dile_fw: object
    lf_ovrc: object
    st_ovrc: object
    rt_ovrc: object
    lfpt: object
    stpt: object
    rtpt: object
    grainpt: object
    lfct: object
    stct: object
    rtct: object


class CropStaticParameters(NamedTuple):
    natural: CarbonParameters
    crop: CropParameters
    croptype: object


def growing_gdd(gdd,t2m,dt,julian,p):
    """GROWING_GDD:10705-10798, including calendar precedence at harvest."""
    gdd,t2m,dt,julian=(jnp.asarray(v,jnp.float32) for v in (gdd,t2m,dt,julian))
    tc=t2m-273.15
    ipa=jnp.where(julian<p.pltd,jnp.int32(0),jnp.int32(1))
    iha=jnp.where(julian>=p.hsd,jnp.int32(0),jnp.int32(1))
    tdiff=jnp.where(tc<p.gddtbase,0.,jnp.where(tc>=p.gddtcut,p.gddtcut-p.gddtbase,tc-p.gddtbase))
    gdd=(gdd+tdiff*dt/86400.)*ipa*iha
    pgs=jnp.where(gdd>0.,jnp.int32(2),jnp.int32(1))
    for stage,threshold in enumerate((p.gdds1,p.gdds2,p.gdds3,p.gdds4,p.gdds5),3):
        pgs=jnp.where(gdd>=threshold,jnp.int32(stage),pgs)
    pgs=jnp.where(julian>=p.hsd,jnp.int32(8),pgs)
    pgs=jnp.where(julian<p.pltd,jnp.int32(1),pgs)
    return gdd,ipa,iha,pgs


def crop_flux(state,p,lai,sai,psn,btran,foln,tv,stc1,wroot,t2m,julian,dt):
    """Original CO2FLUX_CROP, after GROWING_GDD. Valid crop land only."""
    cast=lambda v:jnp.asarray(v,jnp.float32)
    p=CropParameters(*(cast(v) for v in p))
    lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,gdd=(cast(v) for v in state[:8])
    lai,sai,psn,btran,foln,tv,stc1,wroot,dt=(cast(v) for v in (lai,sai,psn,btran,foln,tv,stc1,wroot,dt))
    gdd,ipa,iha,pgs=growing_gdd(gdd,t2m,dt,julian,p)
    from gpuwrf.physics.noahmp.land_columns import pick_layer
    pick=lambda a:pick_layer(jnp.broadcast_to(a,(8,)+pgs.shape),pgs-1)
    sapm=cast(3.)*cast(.001)
    lfmsmn=cast(.05)/cast(.035);stmsmn=cast(.05)/sapm
    carbfx=psn*12.e-6;cbhydrafx=psn*30.e-6
    wstres=1.-btran
    fnf=jnp.minimum(foln/jnp.maximum(1.e-6,p.foln_mx),1.)
    tf=p.q10mr**((tv-298.16)/10.)
    resp=p.lfmr25*tf*fnf*lai*(1.-wstres)
    rsleaf=jnp.minimum((lfmass-lfmsmn)/dt,resp*30.e-6)
    rsroot=p.rtmr25*(rtmass*1.e-3)*tf*30.e-6
    rsstem=p.stmr25*(stmass*1.e-3)*tf*30.e-6
    rsgrain=p.grainmr25*(grain*1.e-3)*tf*30.e-6
    lfpt,stpt,rtpt,grainpt=(pick(a) for a in (p.lfpt,p.stpt,p.rtpt,p.grainpt))
    grleaf=jnp.maximum(0.,p.fra_gr*(lfpt*cbhydrafx-rsleaf))
    grstem=jnp.maximum(0.,p.fra_gr*(stpt*cbhydrafx-rsstem))
    grroot=jnp.maximum(0.,p.fra_gr*(rtpt*cbhydrafx-rsroot))
    grgrain=jnp.maximum(0.,p.fra_gr*(grainpt*cbhydrafx-rsgrain))
    lftovr=pick(p.lf_ovrc)*1.e-6*lfmass
    rttovr=pick(p.rt_ovrc)*1.e-6*rtmass
    sttovr=pick(p.st_ovrc)*1.e-6*stmass
    sc=jnp.exp(-.3*jnp.maximum(0.,tv-p.lefreez))*(lfmass/120.)
    sd=jnp.exp((wstres-1.)*100.)
    dielf=lfmass*1.e-6*(pick(p.dile_fw)*sd+pick(p.dile_fc)*sc)
    addlf=lfpt*cbhydrafx-grleaf-rsleaf
    addst=stpt*cbhydrafx-grstem-rsstem
    lfdel=(lfmass-lfmsmn)/dt;stdel=(stmass-stmsmn)/dt
    lftovr=jnp.minimum(lftovr,lfdel+addlf)
    sttovr=jnp.minimum(sttovr,stdel+addst)
    dielf=jnp.minimum(dielf,lfdel+addlf-lftovr)
    nppl=addlf;npps=addst
    nppr=rtpt*cbhydrafx-rsroot-grroot
    nppg=grainpt*cbhydrafx-rsgrain-grgrain
    lfmass=lfmass+(nppl-lftovr-dielf)*dt
    stmass=stmass+(npps-sttovr)*dt
    rtmass=rtmass+(nppr-rttovr)*dt
    grain=grain+nppg*dt
    gpp=cbhydrafx*.4
    lfconvert=lfmass*(pick(p.lfct)*dt/3600.)
    stconvert=stmass*(pick(p.stct)*dt/3600.)
    rtconvert=rtmass*(pick(p.rtct)*dt/3600.)
    lfmass=lfmass-lfconvert;stmass=stmass-stconvert;rtmass=rtmass-rtconvert
    grain=grain+stconvert+rtconvert+lfconvert
    rttovr=jnp.where(rtmass<0.,nppr,rttovr);rtmass=jnp.where(rtmass<0.,0.,rtmass)
    grain=jnp.where(grain<0.,0.,grain)
    fastcp=fastcp+(rttovr+lftovr+sttovr+dielf)*dt*.4
    fst=2.**((stc1-283.16)/10.)
    fsw=wroot/(.20+wroot)*.23/(.23+wroot)
    rssoil=fsw*fst*p.mrp*jnp.maximum(0.,fastcp*1.e-3)*30.e-6
    stablc=.1*rssoil
    fastcp=fastcp-(rssoil+stablc)*dt*.4
    stblcp=stblcp+stablc*dt*.4
    npp=(nppl+npps+nppr+nppg)*.4
    autors=(rsroot+rsgrain+rsleaf+rsstem+grleaf+grroot+grgrain+grstem)*.4
    heters=.9*rssoil*.4
    nee=(autors+heters-gpp)*44./12.
    totsc=fastcp+stblcp;totlb=(lfmass+rtmass+stmass+grain)*.4
    lai=jnp.maximum(lfmass*p.bio2lai,.05);sai=jnp.maximum(stmass*sapm,.05)
    harvest=(pgs==8)&((grain>0.)|(lfmass>0.)|(stmass>0.)|(rtmass>0.))
    lai=jnp.where(harvest,.05,lai);sai=jnp.where(harvest,.05,sai)
    lfmass=jnp.where(harvest,lfmsmn,lfmass);stmass=jnp.where(harvest,stmsmn,stmass)
    rtmass=jnp.where(harvest,0.,rtmass);grain=jnp.where(harvest,0.,grain)
    state=CropCarbonState(lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,gdd,pgs)
    diag=dict(zip(("GPP","NPP","NEE","AUTORS","HETERS","TOTSC","TOTLB"),(gpp,npp,nee,autors,heters,totsc,totlb)))
    return state,lai,sai,diag


def crop_column(pools,p,lai,sai,psn,btran,foln,tv,stc1,smc,smcmax,dzs,zsoil,nroot,veg,igs,dt,t2m,julian,*,dveg):
    """Natural/crop SFLX dispatch in one optional native carbon phase."""
    active=p.croptype>0
    # Inactive operands remain valid; only the selected model's results survive.
    wroot=root_zone_water(smc,smcmax,dzs,zsoil,nroot,(veg==13)|(veg==15)|(veg==16)|(veg==17))
    crop,lai_c,sai_c,diag_c=crop_flux(pools,p.crop,lai,sai,psn,btran,foln,tv,stc1,wroot,t2m,julian,dt)
    natural=CarbonState(*pools[:6])
    if dveg in (2,5,6):
        natural,lai,sai,diag=carbon_column(natural,p.natural,lai,sai,psn,btran,foln,tv,stc1,smc,smcmax,dzs,zsoil,nroot,veg,igs,dt)
    else:diag={key:jnp.zeros_like(lai) for key in ("GPP","NPP","NEE")}
    select=lambda a,b:jnp.where(active,a,b)
    result=CropCarbonState(*(select(a,b) for a,b in zip(crop[:6],natural)),
        *(select(a,b) for a,b in zip(crop[6:],pools[6:])))
    return result,select(lai_c,lai),select(sai_c,sai),{key:select(diag_c[key],diag[key]) for key in diag}
