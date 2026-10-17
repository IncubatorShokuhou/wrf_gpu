"""Chou-Suarez Goddard option-5 LW core, explicit WRF DOUBLE island.

Top-down column inputs use the lwrad argument conventions. Wrapper outputs
round once to REAL4. WRF hardwires high=False, trace=True, overcast=True.
This port retains those compile-time options and the source's use of aerosol
ssa/asymmetry in the cloud diffuse-transmission expression (10577-10578).
"""
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from .ra_goddard_common import _host_tables


def R(x):
    """Fortran default REAL literal promoted into a REAL(fp_kind) expression."""
    return float(np.float32(x))


def _weighted(x, weights):
    return lax.fori_loop(1, x.shape[-1], lambda k,a:a+x[...,k]*weights[k], x[...,0]*weights[0])


def _powers(first, count, power):
    out = jnp.ones(first.shape+(count,), jnp.float64).at[...,0].set(first)
    def step(k,out):
        x = out[...,k-1]
        if power == 8:
            y=x*x; y=y*y; y=y*y
        elif power == 5:
            y=x*x; y=y*y; y=y*x
        elif power == 12:
            y=x*x*x; y=y*y; y=y*y
        else:
            raise ValueError(power)
        return out.at[...,k].set(y)
    return lax.fori_loop(1,count,step,out)


def _layer_exponents(ib, dp, pa, dt, water, continuum, co2, n2o, ch4, cfcs, tables):
    """lwrad's h2oexps/conexps/co2exps/trace-gas helpers, all REAL8."""
    b = jnp.minimum(ib,8)
    x=water*(pa/R(500))**tables['pm'][b]*(1+(tables['aw'][b]+tables['bw'][b]*dt)*dt)
    water_exp=jnp.ones((dp.shape[0],6),jnp.float64).at[:,0].set(jnp.exp(-x*tables['xkw'][b]))
    def hstep(k,exp):
        prev=exp[:,k-1]
        mw=tables['mw'][b]
        x=jnp.where(mw==9,prev*prev*prev,prev*prev)
        cutoff=jnp.where(mw==6,R(1e-4),jnp.where(mw==8,R(1e-3),jnp.where(mw==9,R(1e-4),R(2e-2))))
        x=jnp.where(x<cutoff,0.,x)
        x=jnp.where((mw==6)|(mw==9),x*x*x,jnp.where(mw==8,(x*x)*(x*x),((x*x)*(x*x))*((x*x)*(x*x))))
        return exp.at[:,k].set(x)
    water_exp=lax.fori_loop(1,6,hstep,water_exp)
    x=jnp.exp(-continuum*tables['xke'][b])
    con_exp=jnp.stack((x,x*x,(x*x)*(x*x)),axis=-1)
    c0=co2*(pa/R(300))**R(.5)*(1+(R(.0182)+R(1.07e-4)*dt)*dt)
    c1=co2*(pa/R(30))**R(.85)*(1+(R(.0042)+R(2e-5)*dt)*dt)
    co_exp=jnp.stack((_powers(jnp.exp(-c0*R(2.656e-5)),6,8),_powers(jnp.exp(-c1*R(2.656e-3)),6,8)),axis=-1)
    n0=n2o*(1+(R(1.9297e-3)+R(4.375e-6)*dt)*dt)
    n1=n2o*(pa/R(500))**R(.48)*(1+(R(1.3804e-3)+R(7.4838e-6)*dt)*dt)
    first=jnp.exp(-jnp.where(ib==5,n0*R(6.31582e-2),n1*R(5.35779e-2)))
    n_exp=_powers(first,4,8)
    x=first*first*first; x1=x*x; x2=x1*x1
    n_exp=n_exp.at[:,1].set(jnp.where(ib==5,x*x1*x2,n_exp[:,1]))
    c0=ch4*(1+(R(1.7007e-2)+R(1.5826e-4)*dt)*dt)
    c1=ch4*(pa/R(500))**R(.65)*(1+(R(5.959e-4)-R(2.2931e-6)*dt)*dt)
    ch_exp=_powers(jnp.exp(-jnp.where(ib==5,c0*R(5.80708e-3),c1*R(6.29247e-2))),4,12)
    a=jnp.where(ib==3,R(3.5775e-2),R(3.4268e-2))
    bminor=jnp.where(ib==3,R(4.0447e-4),R(3.7401e-4))
    minor=_powers(jnp.exp(-co2*(1+(a+bminor*dt)*dt)*R(1.922e-7)),6,5)
    # cfcexps: each triplet is a,b,fk in bands 4 and (5 or 6).
    coeff=jnp.asarray(np.asarray([
        [[1.2661e-3,3.5594e-6,1.89736e1],[8.1937e-4,4.6781e-6,1.01487e1]],
        [[8.7737e-4,-5.8844e-6,1.58104e1],[8.62e-4,-4.225e-6,3.70107e1]],
        [[9.6513e-4,1.3128e-5,6.18536],[ -3.0001e-5,5.2501e-7,3.27912e1]],
    ],np.float32).astype(np.float64))[:,jnp.where(ib==3,0,1),:]
    cfc_exp=jnp.exp(-cfcs*(1+(coeff[:,0]+coeff[:,1]*dt[:,None])*dt[:,None])*coeff[:,2])
    # b10exps replaces the ordinary exponentials for the narrow 540-620 band.
    def narrow(_):
        x=water*(pa/R(500))*(1+(R(.0149)+R(6.2e-5)*dt)*dt)
        h=_powers(jnp.exp(-x*R(.10624)),6,8)
        x=co2*(pa/R(300))**R(.5)*(1+(R(.0179)+R(1.02e-4)*dt)*dt)
        c=co_exp.at[:,:,0].set(_powers(jnp.exp(-x*R(2.656e-5)),6,8))
        con=con_exp.at[:,0].set(jnp.exp(-continuum*R(1.04995e2)))
        x=n2o*(1+(R(1.4476e-3)+R(3.6656e-6)*dt)*dt)
        nfirst=jnp.exp(-x*R(.25238))
        x=nfirst*nfirst; x1=x*x; x1=x1*x1; x2=x1*x1; x3=x2*x2
        n=n_exp.at[:,0].set(nfirst).at[:,1].set(x*x1*x2*x3)
        return h,con,c,n
    water_exp,con_exp,co_exp,n_exp=lax.cond(ib==9,narrow,lambda _:(water_exp,con_exp,co_exp,n_exp),operand=None)
    return water_exp,con_exp,co_exp,n_exp,ch_exp,minor,cfc_exp


def _ozone_table(k1,k2,amount,pressure,temperature,tables):
    x1=amount[k2]-amount[k1]
    we=(jnp.log10(x1)-R(-6))/R(.3)
    active=we>=R(-8)
    # Original tablup branches before these divisions. Safe inactive operands
    # retain that behavior when XLA predicates a batched conditional.
    safe_amount=jnp.where(active,x1,1.)
    x2=jnp.where(active,(pressure[k2]-pressure[k1])/safe_amount,1.)
    x3=jnp.where(active,(temperature[k2]-temperature[k1])/safe_amount,0.)
    we=jnp.where(active,we,0.)
    pe=(jnp.log10(x2)-R(-2))/R(.2)
    we=jnp.minimum(we,20.)
    pe=jnp.clip(pe,0.,25.)
    iw=jnp.clip(jnp.trunc(we+1).astype(jnp.int32),2,20)-1
    ip=jnp.clip(jnp.trunc(pe+1).astype(jnp.int32),1,25)-1
    fw=we-iw.astype(jnp.float64); fp=pe-ip.astype(jnp.float64)
    c1,c2,c3=tables['o1'],tables['o2'],tables['o3']
    pa=c1[ip,iw-1]*(1-fp)+c1[ip+1,iw-1]*fp
    pb=c1[ip,iw]*(1-fp)+c1[ip+1,iw]*fp
    pc=c1[ip,iw+1]*(1-fp)+c1[ip+1,iw+1]*fp
    ax=(-pa*(1-fw)+pc*(1+fw))*fw*R(.5)+pb*(1-fw*fw)
    ba=c2[ip,iw]*(1-fp)+c2[ip+1,iw]*fp
    bb=c2[ip,iw+1]*(1-fp)+c2[ip+1,iw+1]*fp
    ca=c3[ip,iw]*(1-fp)+c3[ip+1,iw]*fp
    cb=c3[ip,iw+1]*(1-fp)+c3[ip+1,iw+1]*fp
    t1=ba*(1-fw)+bb*fw; t2=ca*(1-fw)+cb*fw
    return ax+(t1+t2*x3)*x3,active


def solve_lw_column(pl,ta,wa,oa,tb,ts,emiss,fcld,taucl,ssaal,asyal,taual):
    """Pristine lwrad default options; outputs net/down/up in REAL4 W/m2.

    Inputs pressure mb, temperatures K, specific humidity/ozone kg/kg, band
    optics (...,10). Upwelling is negative, matching the original lwrad.
    """
    pl,ta,wa,oa,tb,ts,emiss,fcld,taucl,ssaal,asyal,taual=[jnp.asarray(v,jnp.float64) for v in (pl,ta,wa,oa,tb,ts,emiss,fcld,taucl,ssaal,asyal,taual)]
    tables={key[3:]:jnp.asarray(value) for key,value in _host_tables().items() if key.startswith('lw_')}
    n=ta.shape[0]
    dp=pl[1:]-pl[:-1]; pa=R(.5)*(pl[:-1]+pl[1:]); dt=ta-R(250)
    water=jnp.maximum(R(1.02)*wa*dp,1e-30)
    ozone=jnp.maximum(R(476)*oa*dp,1e-30)
    co2=jnp.maximum(R(789)*R(401.3e-6)*dp,1e-30)
    n2o=R(789)*R(.32e-6)*dp; ch4=R(789)*R(1.79e-6)*dp
    cfcs=R(789)*jnp.asarray(np.asarray([268e-12,503e-12,105e-12],np.float32).astype(np.float64))*dp[:,None]
    x=pa*R(.001618)*wa*wa*dp
    continuum=x*jnp.exp(R(1800)/ta-R(6.081))+R(1e-10)
    zeros=jnp.zeros((n+1,),jnp.float64)
    def column(k,carry):
        a,p,t=carry
        return a.at[k+1].set(a[k]+ozone[k]),p.at[k+1].set(p[k]+pa[k]*ozone[k]),t.at[k+1].set(t[k]+dt[k]*ozone[k])
    oamount,opress,otemp=lax.fori_loop(0,n,column,(zeros,zeros,zeros))
    # K weights are source REAL literals assigned to DOUBLE, in original order.
    wco=jnp.asarray(np.asarray([[.1395,.1407,.1549,.1357,.0182,.0220],
                                [.0766,.1372,.1189,.0335,.0169,.0059]],np.float32).astype(np.float64))
    wn=jnp.asarray(np.asarray([[.940414,.059586,0,0],[.561961,.138707,.240670,.058662]],np.float32).astype(np.float64))
    wch=jnp.asarray(np.asarray([.610650,.280212,.107349,.001789],np.float32).astype(np.float64))
    wm=jnp.asarray(np.asarray([[.12159,.24359,.24981,.26427,.07807,.04267],
                              [.06869,.14795,.19512,.33446,.17199,.08179]],np.float32).astype(np.float64))
    wh10=jnp.asarray(np.asarray([.3153,.4604,.1326,.0798,.0119,0],np.float32).astype(np.float64))
    wc10=jnp.asarray(np.asarray([.2673,.2201,.2106,.2409,.0196,.0415],np.float32).astype(np.float64))
    def band(ib,totals):
        total,down,up=totals
        cb=tables['cb'][:,ib]
        def planck(t):
            return t*(t*(t*(t*(t*cb[5]+cb[4])+cb[3])+cb[2])+cb[1])+cb[0]
        bl=jnp.concatenate((jnp.zeros((1,),jnp.float64),planck(ta),jnp.reshape(planck(ts)*emiss[ib],(1,))))
        dbl=bl[:-1]-bl[1:]
        middle=(bl[1:n]*dp[1:]+bl[2:n+1]*dp[:-1])/(dp[:-1]+dp[1:])
        top=bl[1]+(bl[1]-bl[2])*dp[0]/(dp[0]+dp[1])
        level=jnp.concatenate((jnp.reshape(top,(1,)),middle,jnp.reshape(planck(tb),(1,))))
        gg=asyal[:,ib]; ww=ssaal[:,ib]
        ff=R(.5)+(R(.3739)+(R(.0076)+R(.1185)*gg)*gg)*gg
        tx=jnp.maximum(0.,R(1.66)*(taucl[:,ib]*(1-ww*ff)))
        cloud=jnp.where((taucl[:,ib]>R(.0001))&(fcld>R(.01)),jnp.where(tx<R(80),jnp.exp(-tx),0.),1.)
        aerosol=jnp.where(taual[:,ib]>R(.01),jnp.exp(-R(1.66)*taual[:,ib]*(1-ww*ff)),1.)
        exps=_layer_exponents(ib,dp,pa,dt,water,continuum,co2,n2o,ch4,cfcs,tables)
        def upper(k1,carry):
            flxd,flxu,rflx,transfc=carry
            product=tuple(jnp.ones(x.shape[1:],jnp.float64) for x in exps)
            def lower(k2,c):
                fd,fu,rf,prod,fclr,tranal,trans=c
                layer=k2-1
                prod=tuple(p*x[layer] for p,x in zip(prod,exps))
                h,con,co,nn,ch,minor,cfc=prod
                b=jnp.minimum(ib,8)
                htrans=_weighted(h,tables['fkw'][:,b])
                htrans=jnp.where((ib>=3)&(ib<=6),htrans*con[0],htrans)
                band3=_weighted(h,tables['gkw'][:,0])*con[0]+_weighted(h,tables['gkw'][:,1])*con[1]+_weighted(h,tables['gkw'][:,2])*con[2]
                tran=jnp.where(ib==2,band3,htrans)
                # co2kdis continues its accumulator across both subbands.
                co_sum=_weighted(jnp.concatenate((co[:,0],co[:,1])),wco.reshape(12))
                tran=jnp.where(ib==2,tran*co_sum,tran)
                oz,oz_active=_ozone_table(k1,k2,oamount,opress,otemp,tables)
                # The original low-amount ELSE overwrites, rather than
                # multiplies, the incoming water-vapor transmittance (:11695).
                oz_result=jnp.where(oz_active,jnp.clip(oz*tran,1e-7,.9999999),R(.9999999))
                tran=jnp.where(ib==4,oz_result,tran)
                tran=jnp.where((ib==5)|(ib==6),tran*_weighted(nn,wn[jnp.where(ib==5,0,1)]),tran)
                tran=jnp.where((ib==5)|(ib==6),tran*jnp.where(ib==5,ch[0],_weighted(ch,wch)),tran)
                tran=jnp.where((ib==3)|(ib==4),tran*_weighted(minor,wm[jnp.where(ib==3,0,1)]),tran)
                tran=jnp.where((ib==3)|(ib==4),tran*cfc[0],tran)
                tran=jnp.where((ib==3)|(ib==5),tran*cfc[1],tran)
                tran=jnp.where((ib==3)|(ib==5),tran*cfc[2],tran)
                narrow=_weighted(h,wh10)*con[0]*_weighted(co[:,0],wc10)*(R(.970831)*nn[0]+R(.029169)*nn[1]-1)
                tran=jnp.where(ib==9,narrow,tran)
                tranal=tranal*aerosol[layer]; tran=tran*tranal
                fclr=fclr*cloud[layer]
                def normal(args):
                    fd,fu,rf=args
                    def adjacent(args):
                        fd,fu=args
                        yy=jnp.clip(tran,.001,.999)
                        xx=(level[k1]-level[k2])/jnp.log(yy)
                        bu=(level[k1]-level[k2]*yy)/(1-yy)+xx
                        bd=(level[k2]-level[k1]*yy)/(1-yy)-xx
                        fu=fu.at[k1].set(fu[k1]-bu+(bu-bl[k2+1])*tran*fclr)
                        fd=fd.at[k2].set(fd[k2]+bd-(bd-bl[k1])*tran*fclr)
                        return fd,fu
                    def distant(args):
                        fd,fu=args
                        fu=fu.at[k1].set(fu[k1]+tran*dbl[k2]*fclr)
                        fd=fd.at[k2].set(fd[k2]+tran*dbl[k1]*fclr)
                        return fd,fu
                    fd,fu=lax.cond(k2==k1+1,adjacent,distant,(fd,fu))
                    return fd,fu,rf
                def correction(args):
                    fd,fu,rf=args
                    rf=rf.at[k1].set(rf[k1]+tran*fclr*dbl[k2])
                    rf=rf.at[k2].set(rf[k2]+tran*fclr*dbl[k1])
                    return fd,fu,rf
                fd,fu,rf=lax.cond(ib==9,correction,normal,(fd,fu,rf))
                return fd,fu,rf,prod,fclr,tranal,tran*fclr
            fd,fu,rf,_,_,_,trans=lax.fori_loop(k1+1,n+1,lower,(flxd,flxu,rflx,product,jnp.float64(1),jnp.float64(1),jnp.float64(1)))
            return fd,fu,rf,transfc.at[k1].set(trans)
        fd,fu,rf,trans=lax.fori_loop(0,n,upper,(zeros,zeros,zeros,jnp.ones((n+1,),jnp.float64)))
        def surface(args):
            fd,fu=args
            fu=fu.at[n].set(-bl[n+1])
            return fd,fu-fd[n]*trans*(1-emiss[ib])
        fd,fu=lax.cond(ib!=9,surface,lambda x:x,(fd,fu))
        total=total+fd+fu; down=down+fd; up=up+fu
        rf=rf.at[0].set(0.)
        total=total+jnp.where(ib==9,rf,0.)
        down=down+jnp.where((ib==9)&(rf>=0),rf,0.)
        up=up+jnp.where((ib==9)&(rf<0),rf,0.)
        return total,down,up
    return tuple(x.astype(jnp.float32) for x in lax.fori_loop(0,10,band,(zeros,zeros,zeros)))
