"""Scientific nonchem CAMMGMP public column, preserving the WRF REAL4 seam.

Full public literal and native CPU column gates: proofs/s2_cammg.
The retained WRF DOUBLE island contains macrophysics, activation and MG. Unused
CAM-history-only diagnostics are omitted; every public CAMMGMP return is retained.
"""
import numpy as np
import jax.numpy as jnp

from . import macro,micro,aerosol
from .cloud import polysvp
from .constants import CONSTANTS
from . import _native
from ._support import _zeros,_literal_array,_get


def _returns(module,name,values):return dict(zip(module.OUTPUTS[name],values,strict=True))


def step(fields,dt,*,itimestep=1,cu=0,shcu=0,accum_mode=1.e9,aitken_mode=3.e8,coarse_mode=2.e5,
         globals,xp=jnp):
    """One public WRF CAMMGMP column; input/output fields are bottom-up REAL4."""
    f={k:xp.asarray(v,dtype=xp.float32) for k,v in fields.items()}
    native=_native.active.get()
    nz=_native.levels.get() if native else f['t_phy'].shape[0]-1
    d=lambda x:xp.asarray(x,dtype=xp.float64)
    r=lambda x:xp.asarray(x,dtype=xp.float32)
    first=itimestep==1
    g={**CONSTANTS,**globals,'pcols':1,'pver':nz,'pverp':nz+1,'pcnst':12,
       'ixcldliq':2,'ixcldice':3,'ixnumliq':4,'ixnumice':5,
       'ptend_top_level':1,'ptend_bot_level':nz,'fillvalue':np.float64(1.e36),'liu_in':True}
    gravity=g['gravit'];cpair=g['cpair'];eps=g['epsqs'];dtime=d(r(dt))
    flip=lambda value,n:_native.flip(value,n) if native else value[:n][::-1]
    z=lambda name:d(flip(f[name],nz))[None,:]
    one=lambda name:d(f[name].reshape(()))
    component=lambda value,i:_get(value,(slice(None),slice(None),i+1))
    # WRF evaluates the differences in REAL before assigning CAM DOUBLE.
    if native:
        from gpuwrf.kernels.phys_thompson_sedimentation import _gather
        k=xp.arange(f['p8w'].shape[0],dtype=xp.int32)
        delta=f['p8w']-_gather.bind(f['p8w'],xp.minimum(k+1,f['p8w'].shape[0]-1))
    else:delta=f['p8w'][:-1]-f['p8w'][1:]
    pdel=d(flip(delta,nz))[None,:]
    zm=d(flip(f['z_sea_level']-f['ht'].reshape(()),nz))[None,:]
    pmid=z('p_hyd');t=z('t_phy')
    static=cpair*z('th')*z('pi_phy')+gravity*zm+one('ht')*gravity
    q=_zeros((1,nz,12),'f64',xp)
    multiplier=d(1)/(d(1)+z('qv_curr'))
    for i,name in enumerate(['qv_curr','qc_curr','qi_curr','nc3d','ni3d']):
        value=z(name)*multiplier
        threshold=d(1.e-11 if i<3 else 5.e2 if i==3 else .1)
        minimum=g['qmin'][i]
        value=xp.where((xp.abs(value)<threshold)&(value<minimum),minimum,value)
        value=xp.where(value<d(0),minimum,value)
        if xp is np or getattr(xp,'_literal_numpy',False):q[:,:,i]=value
        elif native:
            category=xp.arange(q.shape[-1],dtype=xp.int32)
            q=xp.where(category==i,value[...,None],q)
        else:q=q.at[:,:,i].set(value)
    qi_mac=z('qi_curr')*multiplier
    state_q=q;state_s=static
    zero=_zeros((1,nz),'f64',xp)
    zeroq=_zeros((1,nz,12),'f64',xp)
    old_cloud=xp.where(first,zero,z('cldfra_old_mp'))
    old_total=xp.where(first,zero,z('cldfra_mp_all'))
    dp_icwmr=z('icwmrdp3d') if cu==7 else zero
    sh_icwmr=z('icwmrsh3d') if shcu==2 else zero
    sh_frac=z('shfrc3d') if shcu==2 else zero
    cmfmc=d(flip(f['cmfmc3d'],nz+1))[None,:] if cu==7 or shcu==2 else _zeros((1,nz+1),'f64',xp)
    cmfmc2=d(flip(f['cmfmc2_3d'],nz+1))[None,:] if shcu==2 else _zeros((1,nz+1),'f64',xp)
    if native:
        k=xp.arange(cmfmc.shape[-1],dtype=xp.int32)
        diff=_gather.bind((cmfmc-cmfmc2)[0],xp.minimum(k+1,cmfmc.shape[-1]-1))[None,:]
    else:diff=cmfmc[:,1:]-cmfmc2[:,1:]
    dp_frac=xp.maximum(d(0),xp.minimum(d(.10)*xp.log(d(1)+d(500)*diff),d(.60)))
    m=_returns(macro,'macrophysics_simple',macro.macrophysics_simple(
        first,12,1,nz,2,4,1,dtime,t,pmid,state_q,dp_icwmr,sh_icwmr,dp_frac,sh_frac,
        zero,zero,zero,qi_mac,one('xland'),one('snowh'),static,q,z('rh_old_mp'),old_total,
        zero,zero,'',xp.bool_(False),_zeros((12,),'bool',xp),zero,zeroq,
        '',xp.bool_(False),_zeros((12,),'bool',xp),zero,zeroq,_globals=g,xp=xp))
    ast=m['ast'];all_cloud=xp.minimum(ast+xp.minimum(d(.8),dp_frac+sh_frac),d(1))
    conv_cloud=xp.maximum(xp.minimum(d(.8),dp_frac+sh_frac),d(0))
    q=m['state1_q'];t=m['state1_t'];static=m['state1_s']
    # Prescribed modal mass/number, exactly as the nonchem WRF driver packs it.
    for mode,number in [(1,accum_mode),(2,aitken_mode),(3,coarse_mode)]:
        n=mode-1;number=d(r(number));lnum,mass_slots,density_index=g['_modal_plan'][n]
        mass=_get(g['specdens_amode'],(density_index+1,))*number/_get(g['voltonumb_amode'],(mode,))
        if xp is np or getattr(xp,'_literal_numpy',False):q=q.copy();q[:,:,lnum]=number
        elif native:q=xp.where(xp.arange(q.shape[-1])==lnum,number,q)
        else:q=q.at[:,:,lnum].set(number)
        for lmass in mass_slots:
            amount=mass*d(.5) if mode==3 else mass
            if xp is np or getattr(xp,'_literal_numpy',False):q[:,:,lmass]=amount
            elif native:q=xp.where(xp.arange(q.shape[-1])==lmass,amount,q)
            else:q=q.at[:,:,lmass].set(amount)
    dgvals=_literal_array([np.float64(.1e-6),np.float64(np.float32(.02e-6)),np.float64(np.float32(1.e-6))],'f64',xp)
    dgnum=xp.broadcast_to(dgvals,_native.storage((1,nz,3)) if native else (1,nz,3))
    interface=lambda name:xp.where(first,_zeros((1,nz+1),'f64',xp),d(flip(f[name],nz+1))[None,:])
    cflx=_zeros((1,12),'f64',xp)
    if xp is np or getattr(xp,'_literal_numpy',False):cflx[:,0]=one('qfx')
    elif native:cflx=xp.where(xp.arange(cflx.shape[-1])==0,one('qfx'),cflx)
    else:cflx=cflx.at[:,0].set(one('qfx'))
    a=_returns(aerosol,'microp_aero_ts',aerosol.microp_aero_ts(
        1,1,dtime,t,zero,component(q,0),component(q,1),component(q,2),component(q,3),component(q,4),pmid,pdel,ast,ast,ast,
        old_cloud,d(flip(f['p8w'],nz+1))[None,:],d(1)/pdel,zm,zero,q,cflx,m['ptend_loc_q'],dgnum,dgnum,
        interface('kvh3d'),interface('tke_pbl'),interface('turbtype3d'),interface('smaw3d'),
        zero,zero,zero,zero,zero,_zeros((1,nz,4),'f64',xp),_zeros((1,nz,4),'f64',xp),
        _zeros((1,nz,12),'f64',xp)+d(1.e-38),_globals=g,xp=xp))
    kwargs=dict(sub_column=False,lchnk=1,ncol=1,deltatin=dtime,tn=t,qn=component(q,0),qc=component(q,1),qi=component(q,2),nc=component(q,3),ni=component(q,4),
                p=pmid,pdel=pdel,cldn=ast,liqcldf=ast,icecldf=ast,cldo=old_cloud,naai=a['naai'],npccnin=a['npccn'],rndst=a['rndst'],nacon=a['nacon'])
    for name in micro.ARGUMENTS['mmicro_pcond']:
        if name not in kwargs:
            shape=tuple(eval(ext,{},dict(pcols=1,pver=nz,pverp=nz+1)) for ext in micro.VARIABLES['mmicro_pcond'][name]['dims'])
            kwargs[name]=_zeros(shape,'f64',xp)
    b=_returns(micro,'mmicro_pcond',micro.mmicro_pcond(**kwargs,_globals=g,xp=xp))
    rates=a['qaertend']
    for i,name in enumerate(['qvlat','qctend','qitend','nctend','nitend']):
        if xp is np or getattr(xp,'_literal_numpy',False):rates=rates.copy();rates[:,:,i]=b[name]
        elif native:rates=xp.where(xp.arange(rates.shape[-1])==i,b[name][...,None],rates)
        else:rates=rates.at[:,:,i].set(b[name])
    flags=_literal_array([True]*5+[False]*7,'bool',xp)
    all_rates=macro.physics_ptend_sum(True,flags,b['tlat'],rates,m['ptend_all_ls'],m['ptend_all_lq'],m['ptend_all_s'],m['ptend_all_q'],12,_globals=g,xp=xp)
    update=macro.physics_update(1,dtime,q,rates,static,b['tlat'],'microp',flags,True,12,_globals=g,xp=xp)
    updated1_q=update[0]
    update=macro.physics_update(1,dtime,state_q,all_rates[3],state_s,all_rates[2],'cldwat',all_rates[1],all_rates[0],12,_globals=g,xp=xp)
    finalq,finals=update[0],update[2]
    result={k:v for k,v in f.items()}
    result['lradius']=xp.full_like(f['lradius'],r(10))
    result['iradius']=xp.full_like(f['iradius'],r(70))
    for name in ['cldfra_old_mp','cldfrai','cldfral','cldfra_mp','cldfra_mp_all','cldfra_conv']:
        result[name]=xp.where(first,xp.zeros_like(f[name]),f[name])
    def put(name,value):
        out=r(flip(value[0],nz))
        if xp is np or getattr(xp,'_literal_numpy',False):result[name]=result[name].copy();result[name][:nz]=out
        elif native:result[name]=xp.where(xp.arange(out.shape[0])<nz,out,result[name])
        else:result[name]=result[name].at[:nz].set(out)
    qv=r(component(finalq,0)/(d(1)-component(finalq,0)));mf=d(1)+d(qv)
    put('qv_curr',qv)
    for name,value in [('qc_curr',component(finalq,1)*mf),('qi_curr',component(finalq,2)*mf),('qr_curr',b['qrout']*mf),
                       ('qs_curr',b['qsout']*mf),('nc3d',component(finalq,3)*mf),('ni3d',component(finalq,4)*mf),
                       ('nr3d',b['nrout']*z('alt')*mf),('ns3d',b['nsout']*z('alt')*mf)]:put(name,value)
    put('qndrop',d(flip(result['nc3d'],nz))[None,:])
    put('th',(finals-gravity*z('z_sea_level'))/(cpair*z('pi_phy')))
    put('wsedl3d',b['vtrmc']);put('cldfra_old_mp',ast);put('cldfra_mp',ast)
    put('cldfra_mp_all',all_cloud);put('cldfra_conv',conv_cloud)
    put('cldfral',m['alst']);put('cldfrai',m['aist']);put('lcd_old_mp',m['alst'])
    actual_t=(finals-gravity*z('z_sea_level'))/cpair
    # polysvp is scalar in the reference; native maps the same operation over levels.
    if native:
        # The same original scalar formula evaluated in parallel on real
        # levels. Neutral padding only prevents unused ghost arithmetic.
        valid=xp.arange(actual_t.shape[-1])<nz
        tt=xp.where(valid,actual_t[0],d(273.15))
        exponent=(d(-7.90298)*(d(373.16)/tt-d(1))+d(5.02808)*xp.log10(d(373.16)/tt)
                  -d(1.3816e-7)*(xp.power(d(10),d(11.344)*(d(1)-tt/d(373.16)))-d(1))
                  +d(8.1328e-3)*(xp.power(d(10),d(-3.49149)*(d(373.16)/tt-d(1)))-d(1))
                  +xp.log10(d(1013.246)))
        es=(xp.power(d(10),exponent)*d(100))[None,:]
    else:es=xp.stack([polysvp(actual_t[0,k],0,_globals=g,xp=xp) for k in range(nz)])[None,:]
    qs=xp.maximum(d(1.e-30),eps*es/(pmid-(d(1)-eps)*es))
    put('rh_old_mp',xp.maximum(d(0),component(finalq,0)/qs))
    liquid=xp.minimum(component(updated1_q,1)/xp.maximum(d(.0001),ast),d(.005))
    ice=xp.minimum(component(updated1_q,2)/xp.maximum(d(.0001),ast),d(.005))
    put('lradius',xp.where((ast>d(.01))&(liquid>d(5.e-5)),b['effc'],d(10)))
    put('iradius',xp.where((ast>d(.01))&(ice>d(1.e-6)),b['effi'],d(25)))
    result['rainncv']=r(b['prect']*dtime*d(1.e3));result['snowncv']=r(b['preci']*dtime*d(1.e3))
    result['rainnc']=r(f['rainnc']+result['rainncv']);result['snownc']=r(f['snownc']+result['snowncv'])
    result['sr']=r(d(result['snowncv'])/(d(result['rainncv'])+d(1.e-12)))
    return result,m['_fatal']+a['_fatal']
