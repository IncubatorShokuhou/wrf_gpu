"""Experimental CPU-oracle-qualified KF-CuP CPS ensemble for cu=10.

One Pallas program owns a column, its ordered perturbation ensemble and
parcel/closure recurrences. The resident phys_cuplus_driver attaches operational
cadence and held diagnostics. GPU and full-forecast qualification remain pending.
WRF chemistry activation is excluded, exactly as in the oracle configuration.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels.phys_kfcup_column import _cup_column, _get, _set, _where, _roll, _cond
from gpuwrf.physics import cumulus_kfcup_tables as tables
from gpuwrf.physics.cumulus_kfcup_support import find_cp, find_rs, cup_adjust_mass_flux


_TEND = ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN')
_PARCEL_PROFILE = _TEND + ('DTDT','QLG','QIG','QLIQ','QICE','WUP','UMF','UER','UDR','DMF','DER','DDR',
                          'FCVT_QC_TO_QI','FCVT_QC_TO_PR','FCVT_QI_TO_PR')
_PARCEL_SCALAR = ('RAINCV','NCA','CUTOP','CUBOT','UPDFRA','WLCL','WULCL')
_OUTPUT_PROFILE = _TEND + ('CLDFRA_CUP','QC_IC_CUP','QC_IU_CUP','WUP_CUP','MFUP_CUP','MFUP_ENT_CUP',
                           'MFDN_CUP','MFDN_ENT_CUP','UPDFRA_CUP',
                           'FCVT_QC_TO_QI','FCVT_QC_TO_PR','FCVT_QI_TO_PR')
_OUTPUT_SCALAR = ('RAINCV','PRATEC','NCA','CUTOP','CUBOT','ISHALL','CUPFLAG','ACTIVEFRAC',
                  'WSTAR','TAUCLOUD','TACTIVE','TSTAR','LNINT','TCLOUD_CUP','WLCL','WULCL')


def _surface(qv, th, rho, z, hfx, qfxin, land, pblh, kpbl, ht, kx):
    """Live cupSlopeSigma equations; indices retain WRF's one-based form."""
    # ep_1 is a WRF REAL derived parameter, rounded at each source operation.
    ep1 = jnp.float32(jnp.float32(461.6)/jnp.float32(287.0)-jnp.float32(1.0))
    qfxmin = jnp.float32(1.0)/jnp.float32(2.5e6)
    qfx = _where(jnp.abs(qfxin) < qfxmin, jnp.copysign(qfxmin,qfxin),qfxin)
    middle = jnp.maximum(jnp.int32(1),kpbl//2)
    above = jnp.minimum(kpbl+2,kx)
    flux = (1.0+ep1*_get(qv,1))*hfx/_get(rho,1)/1004.5 + ep1*_get(th,1)*qfx/_get(rho,1)
    thv = _get(th,middle)*(1.0+ep1*_get(qv,middle))
    wstar = (9.81*pblh*flux/thv)**jnp.float32(jnp.float32(1.0)/jnp.float32(3.0))
    slope_sfc = hfx/(2.5e6*qfx)*2.5e6/1004.5
    qdiff = _get(qv,above)-_get(qv,middle)
    qdiff = _where(jnp.abs(qdiff)<5e-4,jnp.copysign(jnp.float32(5e-4),qdiff),qdiff)
    slope_ez = (_get(th,above)*(1.0+ep1*_get(qv,above))-thv)/qdiff
    height = (_get(z,middle)-ht)/pblh
    sigma_ez = flux/wstar*(2.0+jnp.float32(8.2e-4)*height**jnp.float32(-1.8))
    sigma_sfc = (qfx/_get(rho,1))/wstar*(2.3+jnp.float32(1.1e-2)*height**jnp.float32(-1.6))
    return slope_sfc,slope_ez,sigma_sfc,sigma_ez,wstar


def _distribution(sfc,ez,ss,se,num_bins,th_bin_size,r_bin_size):
    sfc = jnp.copysign(jnp.maximum(jnp.abs(sfc),jnp.float32(1e-15)),sfc)
    ez = jnp.minimum(2000.0,jnp.maximum(-2000.0,ez))
    ez = _where((ez<10.0)&(ez>0),10.0,_where((ez<0)&(ez>-10.0),-10.0,ez))
    ss = jnp.copysign(jnp.maximum(jnp.abs(ss),jnp.float32(1e-15)),ss)
    se = jnp.copysign(jnp.maximum(jnp.abs(se),jnp.float32(1e-15)),se)
    delta = ez-sfc
    delta = jnp.copysign(jnp.maximum(jnp.abs(delta),jnp.float32(1e-15)),delta)
    count=num_bins*num_bins
    size=1 << (count-1).bit_length()
    index=jnp.arange(size,dtype=jnp.int32)
    # j outer, i inner, exactly the original Fortran ensemble traversal.
    theta=(index%num_bins-num_bins//2).astype(jnp.float32)*th_bin_size
    moisture=(index//num_bins-num_bins//2).astype(jnp.float32)*r_bin_size
    c=ez*(theta-sfc*moisture)/delta
    m=(theta-ez*moisture)/delta
    bigcheck=jnp.sqrt(jnp.float32(jnp.finfo(jnp.float32).max))
    def square(value):
        return _where(jnp.abs(value)>bigcheck,jnp.copysign(bigcheck,value),value*value)
    raw=jnp.exp(-0.5*(square(m/ss)+square(c/se)))
    total=jax.lax.fori_loop(0,count,lambda k,acc:acc+_get(raw,k),jnp.float32(0.0))
    freq=_cond(total!=0,lambda x:x/total,lambda x:x,raw)
    return theta,moisture,freq


def _empty(n):
    out={key:jnp.zeros(n,jnp.float32) for key in _OUTPUT_PROFILE}
    out.update({key:jnp.float32(0.0) for key in _OUTPUT_SCALAR})
    out.update(NCA=jnp.float32(-100.0),CUTOP=jnp.float32(1.0),CUBOT=jnp.float32(n),
               ISHALL=jnp.int32(2),CUPFLAG=jnp.int32(0))
    return out


def _cps_column(profiles,ctrl,kx,the0,ttab,qstab,alu,*,num_bins,th_bin_size,r_bin_size,
                min_deep_freq,min_shallow_freq,warm_rain,f_qi,f_qs):
    t,qv,p,dz,rho,w0,u,v,th,z,zw,pi=profiles
    dt,dx,pblh,kpbl_real,hfx,qfx,land,ht=ctrl
    kpbl=kpbl_real.astype(jnp.int32)
    n=t.shape[0]
    levels=jnp.arange(n,dtype=jnp.int32)
    lev=(levels>=1)&(levels<=kx)
    enabled=(kpbl>2)&(hfx>=_where(land==1,100.0,1.0))

    def parcel(theta,moisture,freq,cupflag):
        return _cup_column(t,qv,p,dz,rho,w0,u,v,dt,dx,kx,the0,ttab,qstab,alu,
                           warm_rain,f_qi,f_qs,jnp.float32(0.0),cupflag,
                           theta,moisture,freq,pblh,zw)

    def fallback(_):
        result=parcel(jnp.float32(0.0),jnp.float32(0.0),jnp.float32(.01),jnp.array(False))
        out=_empty(n)
        out.update({key:result[key] for key in _TEND+('RAINCV','NCA','CUTOP','CUBOT')})
        out['PRATEC']=result['RAINCV']/dt
        out['RTHCUTEN']=result['DTDT']/_where(lev,pi,1.0)
        # WRF leaves CuP SHALL/CUP diagnostics at the surface helper's SHALL=2.
        return out

    def ensemble(_):
        sfc,ez,ss,se,wstar=_surface(qv,th,rho,z,hfx,qfx,land,pblh,kpbl,ht,kx)
        theta,moisture,jfd=_distribution(sfc,ez,ss,se,num_bins,th_bin_size,r_bin_size)
        center=num_bins//2*num_bins+num_bins//2
        minimum=min_shallow_freq*_get(jfd,center)
        keys=_PARCEL_PROFILE+_PARCEL_SCALAR
        empty_parcel={key:jnp.zeros(n,jnp.float32) for key in _PARCEL_PROFILE}
        empty_parcel.update({key:jnp.float32(0.0) for key in _PARCEL_SCALAR})
        initial=dict(deep=empty_parcel,shallow=empty_parcel,deep_freq=jnp.float32(0.0),
                     shallow_freq=jnp.float32(0.0),biggest=jnp.float32(-999.0),
                     cloud_freq=jnp.zeros(n,jnp.float32))

        def one(index,state):
            freq=_get(jfd,index)
            skip=(freq<minimum)|((state['deep_freq']>min_deep_freq)&(freq<state['biggest']))

            def run(st):
                result=parcel(_get(theta,index),_get(moisture,index),freq,jnp.array(True))
                deep=result['ISHALL']==0
                shallow=result['ISHALL']==1
                save=deep&(freq>st['biggest'])
                d={key:_where(save,result[key],st['deep'][key]) for key in keys}
                total=dict(st['shallow'])
                inside=lev&(levels>=result['CUBOT'])&(levels<=result['CUTOP'])
                for key in _TEND+('DTDT','UMF','UER','UDR','UPDFRA'):
                    total[key]=_where(shallow,total[key]+result[key],total[key])
                for key in ('QLG','QIG','WUP','WLCL','WULCL','RAINCV'):
                    total[key]=_where(shallow,total[key]+result[key]*freq,total[key])
                total['CUBOT']=_where(shallow,total['CUBOT']+_get(z,result['CUBOT'].astype(jnp.int32))*freq,total['CUBOT'])
                total['CUTOP']=_where(shallow,total['CUTOP']+_get(z,result['CUTOP'].astype(jnp.int32))*freq,total['CUTOP'])
                total['NCA']=_where(shallow,result['NCA'],total['NCA'])
                for key in ('QLIQ','QICE'):
                    total[key]=_where(shallow&inside,total[key]+result[key]*freq,total[key])
                for key,condensate in (('FCVT_QC_TO_QI','QLIQ'),('FCVT_QC_TO_PR','QLIQ'),('FCVT_QI_TO_PR','QICE')):
                    total[key]=_where(shallow&inside,total[key]+result[key]*result[condensate]*freq,total[key])
                return dict(deep=d,shallow=total,
                            deep_freq=_where(deep,st['deep_freq']+freq,st['deep_freq']),
                            shallow_freq=_where(shallow,st['shallow_freq']+freq,st['shallow_freq']),
                            biggest=_where(save,freq,st['biggest']),
                            cloud_freq=_where(shallow&inside,st['cloud_freq']+freq,st['cloud_freq']))
            return _cond(skip,lambda st:st,run,state)

        accumulated=jax.lax.fori_loop(0,num_bins*num_bins,one,initial)
        deep=accumulated['deep_freq']>min_deep_freq
        shallow=(~deep)&(accumulated['shallow_freq']>0)
        ishall=_where(deep,jnp.int32(0),_where(shallow,jnp.int32(1),jnp.int32(2)))
        sfreq=accumulated['shallow_freq']

        def finish_shallow(_):
            total=dict(accumulated['shallow'])
            for key in ('QLG','QIG','WUP','WLCL','WULCL','RAINCV'):
                total[key]=total[key]/sfreq
            def find_index(height):
                first=jnp.min(_where(lev&(height<=zw),levels,kx+1))
                return _where(first<=kx,first-1,jnp.int32(-1)).astype(jnp.float32)
            total['CUBOT']=find_index(total['CUBOT']/sfreq)
            total['CUTOP']=find_index(total['CUTOP']/sfreq)
            weights=accumulated['cloud_freq']
            for key,condensate in (('FCVT_QC_TO_QI','QLIQ'),('FCVT_QC_TO_PR','QLIQ'),('FCVT_QI_TO_PR','QICE')):
                total[key]=_where(weights>0,total[key]/jnp.maximum(jnp.float32(1e-20),total[condensate]),total[key])
            for key in ('QLIQ','QICE'):
                total[key]=_where(weights>0,total[key]/_where(weights>0,weights,1.0),total[key])
            return total

        selected=_cond(deep,lambda _:accumulated['deep'],
                  lambda _:_cond(shallow,finish_shallow,lambda __:empty_parcel,None),None)
        out=_empty(n)
        out.update({key:selected[key] for key in _TEND+('RAINCV','NCA','CUTOP','CUBOT','WLCL','WULCL')})
        out.update(ISHALL=ishall,CUPFLAG=jnp.int32(1),WSTAR=wstar,
                   ACTIVEFRAC=_where(deep,1.0,_where(shallow,sfreq,0.0)))
        out['PRATEC']=out['RAINCV']/dt
        out['RTHCUTEN']=selected['DTDT']/_where(lev,pi,1.0)
        out['NCA']=_where(ishall==2,jnp.float32(-1.0),out['NCA'])
        out['CUBOT']=_where(ishall==2,jnp.float32(1.0),out['CUBOT'])
        out['CUTOP']=_where(ishall==2,jnp.float32(1.0),out['CUTOP'])
        bottom=out['CUBOT'].astype(jnp.int32)
        top=out['CUTOP'].astype(jnp.int32)
        inside=lev&(levels>=bottom)&(levels<=top)
        flux=cup_adjust_mass_flux(selected['UMF'],selected['UER'],selected['UDR'],
                                  selected['DMF'],selected['DER'],selected['DDR'],cloud_top=top,ishall=ishall,
                                  shift_previous=lambda f:_where(levels>0,_roll(f,1),0.0))
        active=ishall<2
        out['MFUP_CUP']=_where(active&lev&(levels>=2)&(levels<=top),_roll(flux['umf'],1),0.0)
        out['MFUP_ENT_CUP']=_where(active&lev&(levels<=top),flux['uer'],0.0)
        out['MFDN_CUP']=_where(deep&lev&(levels>=2)&(levels<=top),_roll(flux['dmf'],1),0.0)
        out['MFDN_ENT_CUP']=_where(deep&lev&(levels<=top),flux['der'],0.0)
        area_bottom=jnp.min(_where(lev&(levels<bottom)&((flux['umf']>0)|(flux['uer']>0)),levels,bottom))
        out['UPDFRA_CUP']=_where(active&lev&(levels>=area_bottom)&(levels<=top),selected['UPDFRA'],0.0)
        out['QC_IU_CUP']=_where(active&lev&(levels<=top),selected['QLIQ'],0.0)
        out['QC_IC_CUP']=_where(shallow&inside,jnp.float32(1e-3),out['QC_IU_CUP'])
        out['WUP_CUP']=selected['WUP']
        out['TCLOUD_CUP']=_where(active,selected['NCA'],0.0)
        for key in ('FCVT_QC_TO_QI','FCVT_QC_TO_PR','FCVT_QI_TO_PR'):
            out[key]=_where(active&lev&(levels<=top),selected[key],0.0)
        return _cloud_fraction(out,selected,z,qv,t,p,wstar,pblh,inside,lev,kx)

    return _cond(enabled,ensemble,fallback,None)


def _cloud_fraction(out,parcel,z,qv,t,p,wstar,pblh,inside,lev,kx):
    """Literal cupCloudFraction, using the physical rather than padded top."""
    def deep(_):
        cf=jnp.maximum(0.0,jnp.minimum(0.1*jnp.log(1.0+675.0*out['MFUP_CUP']),1.0))
        result=dict(out)
        result.update(CLDFRA_CUP=_where(inside,cf,0.0),TSTAR=pblh/wstar)
        return result

    def shallow(_):
        bottom=out['CUBOT'].astype(jnp.int32)
        top=out['CUTOP'].astype(jnp.int32)
        # Only saturation deficits up to cloud_top+1 are observed by WRF.
        satdef=jnp.maximum(find_rs(t,p)-qv,jnp.float32(1e-20))
        def integrate(k,carry):
            qc,sat=carry
            kp1=jnp.minimum(k+1,kx-1)
            dz=_get(z,kp1)-_get(z,k)
            return (0.5*(_get(parcel['QLG'],k)+_get(parcel['QLG'],kp1))*dz+qc,
                    0.5*(_get(satdef,k)+_get(satdef,kp1))*dz+sat)
        intqc,intsat=jax.lax.fori_loop(bottom,top+1,integrate,(jnp.float32(0.0),jnp.float32(0.0)))
        tb=_get(t,bottom)
        cp=find_cp(_get(qv,bottom))
        rs=find_rs(tb,_get(p,bottom))
        lv=3.15e6-2370.0*tb
        gamma=(lv**2)*rs/(cp*jnp.float32(461.6)*tb**2)
        lnint=jnp.maximum(jnp.log(1.0+(1.0+gamma)*intqc/intsat),1.0)
        tstar=pblh/wstar
        tau=jnp.minimum(tstar*lnint,1800.0)
        tactive=_cond(out['WLCL']>0,
                            lambda _:_get(z,top)/out['WLCL'],
                            lambda _:_get(z,top)/wstar,None)
        cf=jnp.minimum(1.0,jnp.maximum(out['ACTIVEFRAC']*tau/tactive,0.01))
        result=dict(out)
        result.update(CLDFRA_CUP=_where(inside,cf,0.0),TAUCLOUD=tau,TACTIVE=tactive,TSTAR=tstar,LNINT=lnint)
        return result

    return _cond(out['ISHALL']==0,deep,
                lambda _:_cond(out['ISHALL']==1,shallow,lambda __:out,None),None)


def kfcup_fresh_column(T,QV,P,DZ,RHO,W0AVG,U,V,TH,Z,ZW,PI,*,dt,dx,pblh,kpbl,hfx,qfx,
                       land,terrain,num_bins=21,th_bin_size=.1,r_bin_size=1e-4,
                       min_deep_freq=.1,min_shallow_freq=.001,warm_rain=False,
                       f_qi=True,f_qs=True,interpret=False):
    """Fresh-cloud CPS boundary, after WRF's W0AVG update/cadence/NCA test.

    Arrays are bottom-up mass vectors, ZW contains the corresponding bottom
    interfaces, and kpbl is one based. Unsupported aerosol/chemistry inputs are
    deliberately absent. Undefined cloud-time diagnostics have zero internal
    placeholders and are meaningful only in their original assigned regime.
    """
    kx=T.shape[0]
    size=1 << (kx+2).bit_length()
    inputs=tuple(jnp.pad(jnp.asarray(a,jnp.float32),(1,size-kx-1))
                 for a in (T,QV,P,DZ,RHO,W0AVG,U,V,TH,Z,ZW,PI))
    controls=jnp.asarray([dt,dx,pblh,kpbl,hfx,qfx,land,terrain],jnp.float32)

    def kernel(*refs):
        profiles=tuple(ref[...] for ref in refs[:12])
        ctrl,the0,ttab,qstab,alu=refs[12:17]
        with jax.enable_x64(False):
            out=_cps_column(profiles,ctrl[...],kx,the0,ttab,qstab,alu,
                            num_bins=num_bins,th_bin_size=jnp.float32(th_bin_size),
                            r_bin_size=jnp.float32(r_bin_size),min_deep_freq=jnp.float32(min_deep_freq),
                            min_shallow_freq=jnp.float32(min_shallow_freq),warm_rain=warm_rain,f_qi=f_qi,f_qs=f_qs)
        for key,ref in zip(_OUTPUT_PROFILE+_OUTPUT_SCALAR,refs[17:]):
            ref[...]=out[key]

    shapes=tuple(jax.ShapeDtypeStruct((size,),jnp.float32) for _ in _OUTPUT_PROFILE)
    shapes+=tuple(jax.ShapeDtypeStruct((),jnp.int32 if key in ('ISHALL','CUPFLAG') else jnp.float32)
                  for key in _OUTPUT_SCALAR)
    result=pl.pallas_call(kernel,out_shape=shapes,grid=(),interpret=interpret,
                          compiler_params=plt.CompilerParams(num_warps=4),name='kfcup_cps_fp32')(
        *inputs,controls,jnp.asarray(tables.THE0K_R4),jnp.asarray(tables.TTAB_R4),
        jnp.asarray(tables.QSTAB_R4),jnp.asarray(tables.ALU))
    np_=len(_OUTPUT_PROFILE)
    out={key:value[1:kx+1] for key,value in zip(_OUTPUT_PROFILE,result[:np_])}
    out.update(zip(_OUTPUT_SCALAR,result[np_:]))
    return out
