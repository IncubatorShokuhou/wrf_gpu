"""VIC, Xinanjiang and dynamic VIC runoff from pristine Noah-MP."""
from gpuwrf.physics.noahmp.land_columns import layer_row,iterate_six
from functools import partial
from typing import NamedTuple
from jax import lax
import jax.numpy as jnp
from gpuwrf.physics.noahmp.land_columns import stack_layers


class RunoffParameters(NamedTuple):
    bvic: object
    axaj: object
    bxaj: object
    xxaj: object
    bdvic: object
    bbvic: object
    gdvic: object


class HydrologyParameters(NamedTuple):
    runoff: object
    tile: object
    tile_fraction: object
    dx: object


def vic_surface(p,smc,smcmax,zsoil,qinsur,dt):
    """COMPUTE_VIC_SURFRUNOFF:7990-8049; cumulative layer depths are literal."""
    zero=jnp.zeros_like(qinsur);top=zero;maximum=zero
    for k in range(2):
        top=top+layer_row(smc,k)*(-1.)*layer_row(zsoil,k)
        maximum=maximum+layer_row(smcmax,k)*(-1.)*layer_row(zsoil,k)
    ex=p.bvic/(1.+p.bvic)
    asat=1.-jnp.maximum(0.,1.-top/maximum)**ex
    asat=jnp.minimum(1.,jnp.maximum(0.,asat))
    imax=(1.+p.bvic)*maximum
    i0=imax*(1.-(1.-asat)**(1./p.bvic))
    over=(i0+qinsur*dt)>imax
    basis=1.-((i0+qinsur*dt)/imax)
    basis=jnp.where(over,1.,basis)  # unused fractional power on the other arm
    runoff=qinsur*dt-maximum+top+jnp.where(over,zero,maximum*basis**(1.+p.bvic))
    runoff=jnp.where(qinsur==0.,zero,jnp.where(imax==0.,qinsur*dt,runoff))
    runoff=runoff/dt
    runoff=jnp.where(runoff<0.,zero,runoff);runoff=jnp.where(runoff>qinsur,qinsur,runoff)
    return qinsur-runoff,runoff


def xaj_surface(p,smc,smcmax,smcref,zsoil,fcr,qinsur,dt):
    """COMPUTE_XAJ_SURFRUNOFF:8053-8127, tension/free water accumulation."""
    zero=jnp.zeros_like(qinsur);wm=zero;wmmax=zero;sm=zero;smmax=zero
    for k in range(2):
        wet=(layer_row(smc,k)-layer_row(smcref,k))>0.
        sm=sm+jnp.where(wet,(layer_row(smc,k)-layer_row(smcref,k))*(-1.)*layer_row(zsoil,k),zero)
        wm=wm+jnp.where(wet,layer_row(smcref,k)*(-1.)*layer_row(zsoil,k),layer_row(smc,k)*(-1.)*layer_row(zsoil,k))
        wmmax=wmmax+layer_row(smcref,k)*(-1.)*layer_row(zsoil,k)
        smmax=smmax+(layer_row(smcmax,k)-layer_row(smcref,k))*(-1.)*layer_row(zsoil,k)
    wm=jnp.minimum(wm,wmmax);sm=jnp.minimum(sm,smmax)
    irunoff=layer_row(fcr,0)*qinsur*dt
    ratio=wm/wmmax
    left=(1.-layer_row(fcr,0))*qinsur*dt*(.5-p.axaj)**(1.-p.bxaj)*ratio**p.bxaj
    right=(1.-layer_row(fcr,0))*qinsur*dt*(1.-(.5+p.axaj)**(1.-p.bxaj)*(1.-ratio)**p.bxaj)
    prunoff=jnp.where(ratio<=(.5-p.axaj),left,right)
    runoff=jnp.where(qinsur==0.,zero,prunoff*(1.-(1.-sm/smmax)**p.xxaj)+irunoff)
    runoff=runoff/dt;runoff=jnp.maximum(0.,runoff);runoff=jnp.minimum(qinsur,runoff)
    return qinsur-runoff,runoff


def _rr1(p,i0,imax,yd):
    depth=jnp.minimum(i0+yd,imax)
    r1=yd-((imax/(p.bdvic+1.))*((1.-i0/imax)**(p.bdvic+1.)-(1.-depth/imax)**(p.bdvic+1.)))
    return jnp.where(r1<0.,0.,r1)


def dynamic_infiltration(p,smc,sice,zsoil,smcmax,bexp,dksat,dwsat,qinsur,dt,facc,*,opt_infdv):
    """Original INFLMAX=0 PHILIP/GREEN_AMPT/SMITH_PARLANGE callees."""
    from gpuwrf.physics.noahmp.water_hydro import _wdfcnd2
    wdf,wcnd=_wdfcnd2(layer_row(smc,0),layer_row(sice,0),layer_row(bexp,0),layer_row(smcmax,0),layer_row(dksat,0),layer_row(dwsat,0))
    if opt_infdv==1:
        sp=jnp.sqrt(2.*jnp.maximum(0.,layer_row(smcmax,0)-layer_row(smc,0))*(layer_row(dwsat,0)-wdf))
        ap=jnp.minimum(wcnd,(jnp.float32(2.)/jnp.float32(3.))*layer_row(dksat,0))
        ap=jnp.maximum(ap,(jnp.float32(1.)/jnp.float32(3.))*layer_row(dksat,0))
        fsur=.5*sp*(dt**(-.5))+ap
    else:
        jj=p.gdvic*jnp.maximum(0.,layer_row(smcmax,0)-layer_row(smc,0))*(-1.)*layer_row(zsoil,0)
        if opt_infdv==2:fsur=layer_row(dksat,0)+((jj/facc)*(layer_row(dksat,0)-wcnd))
        elif opt_infdv==3:
            safe=jnp.where(jj==0.,1.,jj)
            fsur=layer_row(dksat,0)+(.82*(layer_row(dksat,0)-wcnd)/(jnp.exp(.82*facc/safe)-1.))
            fsur=jnp.where(jj==0.,wcnd,fsur)
        else:raise ValueError('Dynamic VIC requires opt_infdv=1/2/3; WRF leaves FSUR unassigned for 0')
    fsur=jnp.where(layer_row(dksat,0)<qinsur,jnp.minimum(qinsur,fsur),qinsur)
    return fsur,facc+fsur


def dynamic_vic(p,smc,sh2o,sice,zsoil,smcmax,bexp,dksat,dwsat,qinsur,dt,facc,*,opt_infdv):
    """DYNAMIC_VIC:8131-8408 with compiler-zero YD at its unassigned reads.

    At :8261/:8347 YD has never been assigned. Zero makes RR1(YD)=0,
    so TEMPR1+FMAX*DT<=DP on both low-capacity arms. Their alternative
    IITERATION30/3 branches are unreachable under the binding init-zero policy.
    Source sensitivity and this specialization are disclosed in the oracle packet.
    """
    zero=jnp.zeros_like(qinsur);top=zero;maximum=zero
    for k in range(2):
        top=top+layer_row(smc,k)*(-1.)*layer_row(zsoil,k)
        maximum=maximum+layer_row(smcmax,k)*(-1.)*layer_row(zsoil,k)
    top=jnp.where(top>maximum,maximum,top)
    dp=qinsur*dt;imax=maximum*(p.bdvic+1.)
    i0=imax*(1.-(1.-(top/maximum)**(1./(1.+p.bdvic))))
    fsur,facc=dynamic_infiltration(p,smc,sice,zsoil,smcmax,bexp,dksat,dwsat,qinsur,dt,facc,opt_infdv=opt_infdv)
    fmax=(p.bbvic+1.)*fsur
    upper=(dp+i0)>imax;high=(fmax*dt)>=dp
    safe=jnp.where(fmax==0.,1.,fmax)
    def complex_update(r1):
        return r1+((fsur*dt)*(1.-(1.-((dp-r1)/(safe*dt))**(p.bbvic+1.))))
    r1_end=_rr1(p,i0,imax,imax-i0)
    temp1=imax-i0-r1_end-((fsur*dt)*(1.-(1.-((dp-r1_end)/(safe*dt))**(p.bbvic+1.))))
    fill=upper&jnp.where(high,temp1<=0.,(imax-i0-(fmax*dt))<=0.)
    saturated=(top>=maximum)&(i0>=imax)
    solve=(dp>0.)&~saturated&~fill
    initial=jnp.where(upper,zero,dp/2.)
    error=jnp.float32(1.388889e-7)*dt
    def body(k,state):
        yd,done=state
        r1=_rr1(p,i0,imax,yd)
        updated=jnp.where(high,complex_update(r1),r1+(fsur*dt))
        change=~done
        finished=(jnp.abs(updated-yd)<=error)|(k==19)
        return jnp.where(change,updated,yd),done|finished
    yd,_=lax.fori_loop(0,20,body,(initial,~solve))
    yd=jnp.where(yd<=0.,zero,yd);yd=jnp.where(yd>=dp,dp,yd)
    runoff=(_rr1(p,i0,imax,yd)+(dp-yd))/dt
    runoff=jnp.where(fill,(dp-(maximum-top))/dt,runoff)
    runoff=jnp.where(saturated,dp/dt,runoff)
    runoff=jnp.where(dp<=0.,zero,runoff)
    runoff=jnp.minimum(runoff,qinsur);runoff=jnp.maximum(runoff,0.)
    return qinsur-runoff,runoff,facc


def _advanced_soilwater(qinsur,qseva,etrani,sh2o,smc,sice,zsoil,dzs,bexp,smcmax,smcref,smcwlt,dksat,dwsat,kdt,frzx,slope,dt,p,urban,
                        *,opt_run,opt_inf,opt_infdv,opt_tdrn,drain_layer_opt):
    """SOILWATER:7234-7556, one optional native phase for runoff/tile branches."""
    from gpuwrf.physics.noahmp.water_hydro import _infil,_srt,_sstep
    zero=jnp.zeros_like(qinsur);rsat=zero
    epore=jnp.maximum(1.e-4,smcmax-sice)
    for k in range(4):rsat=rsat+jnp.maximum(0.,layer_row(sh2o,k)-layer_row(epore,k))*layer_row(dzs,k)
    sh2o=jnp.minimum(epore,sh2o)
    ice=jnp.minimum(1.,sice/smcmax)
    exp4=jnp.exp(jnp.asarray(-4.,jnp.float32))
    fcr=jnp.maximum(0.,jnp.exp(-4.*(1.-ice))-exp4)/(1.-exp4)
    sicemax=jnp.max(sice,axis=0);fcrmax=jnp.max(fcr,axis=0)
    if urban is not None:
        from gpuwrf.physics.noahmp.land_columns import set_layer
        fcr=set_layer(fcr,0,jnp.where(urban,.95,layer_row(fcr,0)))
    def infiltration(smc,sh2o,fine,facc):
        if opt_run==6:
            inf,ro=vic_surface(p.runoff,smc,smcmax,zsoil,qinsur,fine)
        elif opt_run==7:
            inf,ro=xaj_surface(p.runoff,smc,smcmax,smcref,zsoil,fcr,qinsur,fine)
        elif opt_run==8:
            return dynamic_vic(p.runoff,smc,sh2o,sice,zsoil,smcmax,bexp,dksat,dwsat,qinsur,fine,facc,opt_infdv=opt_infdv)
        else:inf,ro=_infil(qinsur,fine,zsoil,sh2o,sice,sicemax,bexp,smcmax,smcwlt,dksat,dwsat,kdt,frzx,literal=True)
        return inf,ro,facc
    facc=jnp.full_like(qinsur,1.e-6)
    inf,ro,_=infiltration(smc,sh2o,dt,facc)
    inf=jnp.where(qinsur>0.,inf,zero) if opt_run==3 else inf
    niter=jnp.where(inf*dt>layer_row(dzs,0)*layer_row(smcmax,0),jnp.float32(6.),jnp.float32(3.))
    fine=dt/niter;drain_save=zero;runoff_save=zero;wcnd_last=jnp.zeros_like(sh2o)
    def fine_step(k,values):
        sh2o,smc,rsat,drain_save,runoff_save,wcnd_last,facc=values
        inf,ro,newfacc=infiltration(smc,sh2o,fine,facc)
        inf=jnp.where(qinsur>0.,inf,zero);ro=jnp.where(qinsur>0.,ro,zero)
        facc=jnp.where((k<niter)&(qinsur>0.),newfacc,facc)
        rhs,ai,bi,ci,drain,wcnd=_srt(zsoil,fine,inf,etrani,qseva,sh2o,smc,fcr,fcrmax,slope,bexp,smcmax,dksat,dwsat,
            opt_inf=opt_inf,sicemax=sicemax,opt_run=opt_run)
        newsh,newsm,wplus=_sstep(fine,dzs,sice,sh2o,smc,ai,bi,ci,rhs,smcmax,literal=True)
        active=k<niter
        sh2o=jnp.where(active[None,...],newsh,sh2o);smc=jnp.where(active[None,...],newsm,smc)
        rsat=rsat+jnp.where(active,wplus,zero)
        drain_save=drain_save+jnp.where(active,drain,zero)
        runoff_save=runoff_save+jnp.where(active,ro,zero)
        wcnd_last=jnp.where(active[None,...],wcnd,wcnd_last)
        return sh2o,smc,rsat,drain_save,runoff_save,wcnd_last,facc
    sh2o,smc,rsat,drain_save,runoff_save,wcnd_last,facc=iterate_six(fine_step,(sh2o,smc,rsat,drain_save,runoff_save,wcnd_last,facc))
    qdrain=(drain_save/niter)*1000.
    runsrf=(runoff_save/niter)*1000.+rsat*1000./dt
    if opt_tdrn:
        from gpuwrf.physics.noahmp.tile_drain import simple_tile,hooghoudt_tile
        if opt_tdrn==1:
            newsh,newsm,qtl=simple_tile(p.tile,sh2o,smc,sice,zsoil,smcref,dt,drain_layer_opt=drain_layer_opt)
            zwt=jnp.full_like(qinsur,2.5);active=p.tile_fraction>.3
        else:
            newsh,newsm,qtl,zwt=hooghoudt_tile(p.tile,wcnd_last,sh2o,smc,sice,zsoil,smcref,smcwlt,dt,p.dx)
            active=p.tile_fraction>.1
        sh2o=jnp.where(active[None,...],newsh,sh2o);smc=jnp.where(active[None,...],newsm,smc)
        qtl=jnp.where(active,qtl,zero);zwt=jnp.where(active,zwt,jnp.float32(2.5))
    layers=[layer_row(sh2o,k)*layer_row(dzs,k)*1000. for k in range(4)]
    for k in range(3):
        xs=jnp.where(layers[k]<0.,.01-layers[k],zero)
        layers[k]=layers[k]+xs;layers[k+1]=layers[k+1]-xs
    xs=jnp.where(layers[3]<.01,.01-layers[3],zero)
    layers[3]=layers[3]+xs
    result=(stack_layers(layers)/(dzs*1000.),smc,runsrf,qdrain,zero-xs/dt)
    return (*result,qtl,zwt) if opt_tdrn else result


def advanced_soilwater(*args,opt_run,opt_inf,opt_infdv=0,opt_tdrn=0,drain_layer_opt=0):
    from gpuwrf.kernels.phys_noahmp_columns import column_call,columns_enabled
    fn=partial(_advanced_soilwater,opt_run=opt_run,opt_inf=opt_inf,opt_infdv=opt_infdv,opt_tdrn=opt_tdrn,drain_layer_opt=drain_layer_opt)
    if columns_enabled():
        from gpuwrf.physics.noahmp.land_columns import layered_column_call
        return layered_column_call(fn,args,grid_shape=jnp.shape(args[0]),name='b_core_noah_runoff_tile')
    return fn(*args)
