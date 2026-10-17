"""WRF equilibrium water table option, without prognostic lateral groundwater."""
from gpuwrf.physics.noahmp.land_columns import layer_row,iterate_six
from functools import partial
from typing import NamedTuple
from jax import lax
import jax.numpy as jnp
from gpuwrf.physics.noahmp.land_columns import pick_layer,stack_layers,in_column_kernel
from gpuwrf._x64_config import configure_jax_x64
configure_jax_x64()


class GroundwaterState(NamedTuple):
    zwt: object
    wa: object
    wt: object


def groundwater_step(state,sh2o,sice,zsoil,wcnd,fcrmax,bexp,smcmax,psisat,dksat,timean,dt):
    """GROUNDWATER:9582-9770, including its sole explicit DOUBLE S_NODE island."""
    sh2o,sice,zsoil,wcnd,fcrmax,bexp,smcmax,psisat,dksat,timean,dt=(jnp.asarray(v,jnp.float32)
        for v in (sh2o,sice,zsoil,wcnd,fcrmax,bexp,smcmax,psisat,dksat,timean,dt))
    zwt,wa,wt=(jnp.asarray(v,jnp.float32) for v in state)
    zero=jnp.zeros_like(zwt)
    if in_column_kernel():
        dzmm=stack_layers((-layer_row(zsoil,0)*1000.,*(1000.*(layer_row(zsoil,k-1)-layer_row(zsoil,k)) for k in range(1,4))))
        znode=stack_layers((-layer_row(zsoil,0)/2.,*(-layer_row(zsoil,k-1)+.5*(layer_row(zsoil,k-1)-layer_row(zsoil,k)) for k in range(1,4))))
    else:
        dzmm=jnp.concatenate((-zsoil[:1]*1000.,1000.*(zsoil[:-1]-zsoil[1:])),axis=0)
        znode=jnp.concatenate((-zsoil[:1]/2.,-zsoil[:-1]+.5*(zsoil[:-1]-zsoil[1:])),axis=0)
    smc=sh2o+sice;mliq=sh2o*dzmm
    epore=jnp.maximum(.01,smcmax-sice);hk=1000.*wcnd
    iwt=jnp.full_like(zwt,3,dtype=jnp.int32);done=jnp.zeros_like(zwt,dtype=bool)
    for iz in range(1,4):
        found=(zwt<=-layer_row(zsoil,iz))&~done
        iwt=jnp.where(found,iz-1,iwt);done=done|found
    def pick(value,index=iwt):
        return pick_layer(jnp.broadcast_to(value,sh2o.shape),index)
    fff=pick(bexp)/3.
    rsbmx=pick(hk)*1000.*jnp.exp(jnp.asarray(3.,jnp.float32))
    # TIMEAN can be a baked table scalar: Triton neg_p does not materialize
    # typed literals. REAL -0 - TIMEAN is exact negation, including signed zero.
    qdis=(1.-fcrmax)*rsbmx*jnp.exp(jnp.asarray(-0.,jnp.float32)-timean)*jnp.exp(-fff*zwt)
    # SMC/SMCMAX and MIN are REAL4 before assignment to DOUBLE. The source's
    # REAL(0.01,KIND=8) promotes the already-rounded REAL4 literal.
    s_node=jnp.minimum(1.,pick(smc)/pick(smcmax)).astype(jnp.float64)
    s_node=jnp.maximum(s_node,jnp.asarray(.01,jnp.float32).astype(jnp.float64))
    smpfz=((-pick(psisat)*1000.).astype(jnp.float64)*s_node**(-pick(bexp)).astype(jnp.float64)).astype(jnp.float32)
    smpfz=jnp.maximum(-120000.,.8*smpfz)
    ka=2.*(pick(hk)*pick(dksat)*1000.)/(pick(hk)+pick(dksat)*1000.)
    wh_zwt=-zwt*1000.;wh=smpfz-pick(znode)*1000.
    qin=-ka*(wh_zwt-wh)/((zwt-pick(znode))*1000.)
    qin=jnp.maximum(-10./dt,jnp.minimum(10./dt,qin))
    wt=wt+(qin-qdis)*dt
    deep=iwt==3
    wa_deep=wa+(qin-qdis)*dt
    wt=jnp.where(deep,wa_deep,wt)
    zwt_deep=(-layer_row(zsoil,3)+25.)-wa_deep/1000./.2
    last=layer_row(mliq,3)-qin*dt
    last=last+jnp.maximum(0.,wa_deep-5000.)
    wa=jnp.where(deep,jnp.minimum(wa_deep,5000.),wa)
    if in_column_kernel():mliq=stack_layers((*[layer_row(mliq,k) for k in range(3)],jnp.where(deep,last,layer_row(mliq,3))))
    else:mliq=jnp.concatenate((mliq[:3],jnp.where(deep,last,layer_row(mliq,3))[None,...]),axis=0)
    # For a shallower water table, saturated storage includes the resolved layers.
    ws=zero
    for iz in range(4):ws=ws+jnp.where(iz>=iwt+2,layer_row(epore,iz)*layer_row(dzmm,iz),0.)
    next_layer=jnp.minimum(iwt+1,3)
    zwt_shallow=-pick(zsoil,next_layer)-(wt-jnp.asarray(.2,jnp.float32)*1000.*25.-ws)/pick(epore,next_layer)/1000.
    zwt=jnp.where(deep,zwt_deep,zwt_shallow)
    wtsub=zero
    for iz in range(4):wtsub=wtsub+layer_row(hk,iz)*layer_row(dzmm,iz)
    layers=[jnp.where(deep,layer_row(mliq,iz),layer_row(mliq,iz)-qdis*dt*layer_row(hk,iz)*layer_row(dzmm,iz)/wtsub) for iz in range(4)]
    zwt=jnp.maximum(1.5,zwt)
    for iz in range(3):
        xs=jnp.where(layers[iz]<0.,.01-layers[iz],0.)
        layers[iz]=layers[iz]+xs;layers[iz+1]=layers[iz+1]-xs
    xs=jnp.where(layers[3]<.01,.01-layers[3],0.)
    layers[3]=layers[3]+xs;wa=wa-xs;wt=wt-xs
    return GroundwaterState(zwt,wa,wt),stack_layers(layers)/dzmm,qin,qdis


def equilibrium_water_table(sh2o,smcmax,psisat,bexp,zsoil,dzs):
    """ZWTEQ:7562-7610, retaining its 100-level sequential deficit search."""
    sh2o,smcmax,psisat,bexp,zsoil,dzs=(jnp.asarray(v,jnp.float32) for v in (sh2o,smcmax,psisat,bexp,zsoil,dzs))
    wd1=jnp.zeros_like(layer_row(sh2o,0))
    for k in range(4):wd1=wd1+(layer_row(smcmax,0)-layer_row(sh2o,k))*layer_row(dzs,k)
    dzfine=3.*(-layer_row(zsoil,3))/100.
    initial=-3.*layer_row(zsoil,3)-.001
    initial=jnp.broadcast_to(initial,wd1.shape)
    def body(k,state):
        wd2,zwt,done=state
        zfine=jnp.asarray(k+1,jnp.float32)*dzfine
        temp=1.+(initial-zfine)/layer_row(psisat,0)
        updated=wd2+layer_row(smcmax,0)*(1.-temp**(-1./layer_row(bexp,0)))*dzfine
        found=(jnp.abs(updated-wd1)<=.01)&~done
        return jnp.where(done,wd2,updated),jnp.where(found,zfine,zwt),done|found
    return lax.fori_loop(0,100,body,(jnp.zeros_like(wd1),initial,jnp.zeros_like(wd1,dtype=bool)))[1]


def _equilibrium_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcwlt,dksat,dwsat,psisat,dt,fsatmx,timean,
                           urban=None,*,opt_inf=1):
    from gpuwrf.physics.noahmp.water_hydro import _srt,_sstep
    fsatmx,timean,dt=(jnp.asarray(v,jnp.float32) for v in (fsatmx,timean,dt))
    epore=jnp.maximum(1.e-4,smcmax-sice)
    # WRF's layer loop accumulates RSAT in order.
    rsat=jnp.zeros_like(qinsur)
    for k in range(4):rsat=rsat+jnp.maximum(0.,layer_row(sh2o,k)-layer_row(epore,k))*layer_row(dzs,k)
    sh2o=jnp.minimum(epore,sh2o)
    ice=jnp.minimum(1.,sice/smcmax)
    exp4=jnp.exp(jnp.asarray(-4.,jnp.float32))
    fcr=jnp.maximum(0.,jnp.exp(-4.*(1.-ice))-exp4)/(1.-exp4)
    sicemax=jnp.max(sice,axis=0);fcrmax=jnp.max(fcr,axis=0)
    zwt=equilibrium_water_table(sh2o,smcmax,psisat,bexp,zsoil,dzs)
    runsub=(1.-fcrmax)*4.*jnp.exp(jnp.asarray(-0.,jnp.float32)-timean)*jnp.exp(-2.*zwt)
    if urban is not None:
        from gpuwrf.physics.noahmp.land_columns import set_layer
        fcr=set_layer(fcr,0,jnp.where(urban,.95,layer_row(fcr,0)))
    fsat=fsatmx*jnp.exp(-.5*jnp.asarray(2.,jnp.float32)*zwt)
    runsrf=jnp.where(qinsur>0.,qinsur*((1.-layer_row(fcr,0))*fsat+layer_row(fcr,0)),jnp.zeros_like(qinsur))
    pddum=jnp.where(qinsur>0.,qinsur-runsrf,jnp.zeros_like(qinsur))
    niter=jnp.where(pddum*dt>layer_row(dzs,0)*layer_row(smcmax,0),jnp.asarray(6.,jnp.float32),jnp.asarray(3.,jnp.float32))
    fine=dt/niter
    runsrf_save=jnp.zeros_like(runsrf)
    wcnd_last=jnp.zeros_like(sh2o)
    def fine_step(k,values):
        sh2o,smc,rsat,runsrf_save,wcnd_last=values
        rhs,ai,bi,ci,_,wcnd=_srt(zsoil,fine,pddum,etrani,qseva,sh2o,smc,fcr,fcrmax,0.,bexp,smcmax,dksat,dwsat,
                                opt_inf=opt_inf,sicemax=sicemax,opt_run=2)
        newsh,newsm,wplus=_sstep(fine,dzs,sice,sh2o,smc,ai,bi,ci,rhs,smcmax,literal=True)
        active=k<niter
        sh2o=jnp.where(active[None,...],newsh,sh2o);smc=jnp.where(active[None,...],newsm,smc)
        rsat=rsat+jnp.where(active,wplus,0.)
        runsrf_save=runsrf_save+jnp.where(active,runsrf,0.)
        wcnd_last=jnp.where(active[None,...],wcnd,wcnd_last)
        return sh2o,smc,rsat,runsrf_save,wcnd_last
    sh2o,smc,rsat,runsrf_save,wcnd_last=iterate_six(fine_step,(sh2o,smc,rsat,runsrf_save,wcnd_last))
    runsrf=(runsrf_save/niter)*1000.+rsat*1000./dt
    wtsub=jnp.zeros_like(qinsur)
    for k in range(4):wtsub=wtsub+layer_row(wcnd_last,k)*layer_row(dzs,k)
    layers=[]
    for k in range(4):
        mh2o=runsub*dt*(layer_row(wcnd_last,k)*layer_row(dzs,k))/wtsub
        layers.append((layer_row(sh2o,k)-mh2o/(layer_row(dzs,k)*1000.))*layer_row(dzs,k)*1000.)
    for k in range(3):
        xs=jnp.where(layers[k]<0.,.01-layers[k],0.)
        layers[k]=layers[k]+xs;layers[k+1]=layers[k+1]-xs
    xs=jnp.where(layers[3]<.01,.01-layers[3],0.)
    layers[3]=layers[3]+xs;runsub=runsub-xs/dt
    sh2o=stack_layers(layers)/(dzs*1000.)
    return sh2o,smc,runsrf,jnp.zeros_like(runsrf),runsub,zwt


def equilibrium_soilwater(*args,opt_inf=1):
    from gpuwrf.kernels.phys_noahmp_columns import column_call,columns_enabled
    fn=partial(_equilibrium_soilwater,opt_inf=opt_inf)
    if columns_enabled():
        from gpuwrf.physics.noahmp.land_columns import layered_column_call
        return layered_column_call(fn,args,grid_shape=jnp.shape(args[0]),name="b_core_noah_equilibrium_water")
    return fn(*args)


def _prognostic_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcwlt,dksat,dwsat,psisat,dt,fsatmx,timean,
                          state,urban=None,*,opt_inf=1):
    """SOILWATER opt1 followed by WATER's GROUNDWATER call, in one column phase."""
    from gpuwrf.physics.noahmp.water_hydro import _srt,_sstep
    fsatmx,timean,dt=(jnp.asarray(v,jnp.float32) for v in (fsatmx,timean,dt))
    epore=jnp.maximum(1.e-4,smcmax-sice)
    rsat=jnp.zeros_like(qinsur)
    for k in range(4):rsat=rsat+jnp.maximum(0.,layer_row(sh2o,k)-layer_row(epore,k))*layer_row(dzs,k)
    sh2o=jnp.minimum(epore,sh2o)
    ice=jnp.minimum(1.,sice/smcmax);exp4=jnp.exp(jnp.asarray(-4.,jnp.float32))
    fcr=jnp.maximum(0.,jnp.exp(-4.*(1.-ice))-exp4)/(1.-exp4)
    sicemax=jnp.max(sice,axis=0);fcrmax=jnp.max(fcr,axis=0)
    if urban is not None:
        from gpuwrf.physics.noahmp.land_columns import set_layer
        fcr=set_layer(fcr,0,jnp.where(urban,.95,layer_row(fcr,0)))
    fff=layer_row(bexp,0)/3.
    fsat=fsatmx*jnp.exp(-.5*fff*state.zwt)
    runsrf=jnp.where(qinsur>0.,qinsur*((1.-layer_row(fcr,0))*fsat+layer_row(fcr,0)),jnp.zeros_like(qinsur))
    pddum=jnp.where(qinsur>0.,qinsur-runsrf,jnp.zeros_like(qinsur))
    niter=jnp.where(pddum*dt>layer_row(dzs,0)*layer_row(smcmax,0),jnp.asarray(6.,jnp.float32),jnp.asarray(3.,jnp.float32))
    fine=dt/niter;runsrf_save=jnp.zeros_like(runsrf);wcnd_last=jnp.zeros_like(sh2o)
    def fine_step(k,values):
        sh2o,smc,rsat,runsrf_save,wcnd_last=values
        rhs,ai,bi,ci,_,wcnd=_srt(zsoil,fine,pddum,etrani,qseva,sh2o,smc,fcr,fcrmax,0.,bexp,smcmax,dksat,dwsat,
                                opt_inf=opt_inf,sicemax=sicemax,opt_run=1)
        newsh,newsm,wplus=_sstep(fine,dzs,sice,sh2o,smc,ai,bi,ci,rhs,smcmax,literal=True)
        active=k<niter
        sh2o=jnp.where(active[None,...],newsh,sh2o);smc=jnp.where(active[None,...],newsm,smc)
        rsat=rsat+jnp.where(active,wplus,0.)
        runsrf_save=runsrf_save+jnp.where(active,runsrf,0.)
        wcnd_last=jnp.where(active[None,...],wcnd,wcnd_last)
        return sh2o,smc,rsat,runsrf_save,wcnd_last
    sh2o,smc,rsat,runsrf_save,wcnd_last=iterate_six(fine_step,(sh2o,smc,rsat,runsrf_save,wcnd_last))
    runsrf=(runsrf_save/niter)*1000.+rsat*1000./dt
    # SOILWATER skips its final floor for run1; GROUNDWATER performs it later.
    state,sh2o,qin,qdis=groundwater_step(state,sh2o,sice,zsoil,wcnd_last,fcrmax,bexp,smcmax,psisat,dksat,timean,dt)
    return sh2o,sh2o+sice,runsrf,jnp.zeros_like(runsrf),qdis,state,qin


def prognostic_soilwater(*args,opt_inf=1):
    from gpuwrf.kernels.phys_noahmp_columns import column_call,columns_enabled
    fn=partial(_prognostic_soilwater,opt_inf=opt_inf)
    if columns_enabled():
        from gpuwrf.physics.noahmp.land_columns import layered_column_call
        return layered_column_call(fn,args,grid_shape=jnp.shape(args[0]),name="b_core_noah_prognostic_water")
    return fn(*args)
