"""WRF 4.7.1 module_sf_noahmp_glacier.F:301-1606, REAL column energy.

Lists hold the seven snow/ice layers in registers inside the Pallas caller.
``xp=np`` supplies a literal binary32 debug path without compiler contraction.
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp

GRAV, SB, VKC, TFRZ = map(np.float32, (9.80616, 5.67e-8, .40, 273.16))
HSUB, HFUS, CWAT, CICE, CPAIR, RAIR = map(np.float32, (2.844e6, .3336e6, 4.188e6, 2.094e6, 1004.64, 287.04))
DENH2O, DENICE, Z0 = map(np.float32, (1000., 917., .002))


def at(values, index, xp=jnp):
    """Dynamic register-list selection; no physical index is clamped."""
    result = xp.zeros_like(values[0])
    for k, value in enumerate(values):
        result = xp.where(index == k, value, result)
    return result


def put(values, index, value, mask=True, xp=jnp):
    for k in range(len(values)):
        values[k] = xp.where(mask & (index == k), value, values[k])


def cube(x):
    return (x*x)*x


def fourth(x):
    square=x*x
    return square*square


def esat(t, xp=jnp):
    # WRF ESAT uses its own polynomial coefficients, not a generic saturation formula.
    water = (6.107799961, .4436518521, .01428945805, .0002650648471,
             .000003031240396, .00000002034080948, .00000000006136820929)
    ice = (6.109177956, .5034698970, .01886013408, .0004176223716,
           .000005824720280, .00000004838803174, .0000000001838826904)
    # Separate derivative coefficients are the literal WRF constants (:1142-1159).
    dw = (.4438099984, .02857002636, .0007938054040, .00001215215065,
          .0000001036561403, .0000000003532421810, -.0000000000007090244804)
    di = (.5030305237, .03773255020, .001267995369, .00002477563108,
          .0000003005693132, .000000002158542548, .000000000007131097725)
    def poly(coeff):
        result = xp.full_like(t, np.float32(coeff[-1]))
        for c in reversed(coeff[:-1]):
            result = np.float32(c) + t * result
        return np.float32(100.) * result
    return poly(water), poly(ice), poly(dw), poly(di)


def atmosphere(v, xp=jnp):
    q, p, t = v['q2'], v['sfcprs'], v['sfctmp']
    e = q*p/(np.float32(.622)+np.float32(.378)*q)
    rho = (p-np.float32(.378)*e)/(RAIR*t)
    sw = xp.where(v['cosz'] <= 0., 0., v['soldn'])
    return e, rho, [sw*np.float32(.7)*np.float32(.5)]*2, [sw*np.float32(.3)*np.float32(.5)]*2, sw


def thermal(v, dz, xp=jnp):
    n = v['isnow']; df, hc = [], []
    for k in range(3):
        active = k >= n+3
        depth = xp.where(active, dz[k], 1.)
        icevol = xp.minimum(1., v['snice'][k]/(depth*DENICE))
        porosity = 1.-icevol
        liqvol = xp.minimum(porosity, v['snliq'][k]/(depth*DENH2O))
        bulk = (v['snice'][k]+v['snliq'][k])/depth
        df.append(xp.where(active, np.float32(3.2217e-6)*(bulk*bulk), 0.))
        hc.append(xp.where(active, CICE*icevol+CWAT*liqvol, 0.))
    for k in range(4):
        zmid = np.float32(.5)*dz[k+3]
        for j in range(k):
            zmid = zmid + dz[j+3]
        hc.append(np.float32(1.e6)*(np.float32(.8194)+np.float32(.1309)*zmid))
        df.append(np.float32(.32333)+np.float32(.10073)*zmid)
    fact = [v['dt']/(xp.where(k >= n+3, hc[k]*dz[k], 1.)) for k in range(7)]
    df[3] = xp.where(n == 0, (df[3]*dz[3]+np.float32(.35)*v['snowh'])/(v['snowh']+dz[3]),
                    (df[3]*dz[3]+df[2]*dz[2])/(dz[2]+dz[3]))
    return df, hc, fact


def radiation(v, direct, diffuse, opt_alb, xp=jnp):
    delta = np.float32(1.e-6)*v['dt']
    arg = np.float32(5.e3)*(np.float32(1.)/TFRZ-np.float32(1.)/v['tg'])
    age = xp.exp(arg)+xp.exp(xp.minimum(0.,np.float32(10.)*arg))+np.float32(.3)
    fresh = xp.maximum(0.,v['sneqv']-v['sneqvo'])
    tauss = xp.maximum(0.,(v['tauss']+delta*age)*(1.-fresh))
    v['tauss'] = xp.where((v['sneqv'] <= 0.) | (v['sneqv'] > 800.), 0., tauss)
    fage = v['tauss']/(v['tauss']+1.)
    if opt_alb == 1:
        cf = (np.float32(1.)+np.float32(.5))/(np.float32(1.)+np.float32(4.)*v['cosz'])-np.float32(.5)
        fzen = xp.maximum(cf,0.)
        ai = [np.float32(.95)*(1.-np.float32(.2)*fage), np.float32(.65)*(1.-np.float32(.5)*fage)]
        ad = [a+np.float32(.4)*fzen*(1.-a) for a in ai]
    else:
        alb = np.float32(.55)+(v['albold']-np.float32(.55))*xp.exp(-np.float32(.01)*v['dt']/np.float32(3600.))
        alb = xp.where(v['qsnow'] > 0., alb+xp.minimum(v['qsnow']*v['dt'],1.)*(np.float32(.84)-alb),alb)
        v['albold'] = xp.where(v['cosz'] > 0.,alb,v['albold'])
        ai, ad = [alb]*2, [alb]*2
    fsno = xp.where(v['sneqv'] > 0.,np.float32(1.),np.float32(0.))
    sag, fsr = xp.zeros_like(v['tg']), xp.zeros_like(v['tg'])
    for k, icealb in enumerate((.8,.55)):
        a = np.float32(icealb)*(1.-fsno)+xp.where(v['cosz'] > 0.,ad[k],0.)*fsno
        b = np.float32(icealb)*(1.-fsno)+xp.where(v['cosz'] > 0.,ai[k],0.)*fsno
        sag = sag + (direct[k]*(1.-a)+diffuse[k]*(1.-b))
        fsr = fsr + (direct[k]*a+diffuse[k]*b)
    v.update(sag=sag, fsa=sag, fsr=fsr)


def sfcdif(iteration, z, zpd, q, t, h, rho, ur, carry, xp=jnp):
    moz, signs, fm, fh, fm2, fh2, fv = carry
    old = moz
    lm = xp.log((z-zpd)/Z0); l2 = xp.log((np.float32(2.)+Z0)/Z0)
    if iteration == 1:
        fv = xp.zeros_like(t); moz = xp.zeros_like(t); moz2 = xp.zeros_like(t)
    else:
        tv = (np.float32(1.)+np.float32(.61)*q)*t
        temp = VKC*(GRAV/tv)*h/(rho*CPAIR)
        temp = xp.where(xp.abs(temp) <= np.float32(1.e-6),np.float32(1.e-6),temp)
        mol = -np.float32(1.)*cube(fv)/temp
        moz = xp.minimum((z-zpd)/mol,1.); moz2 = xp.minimum((np.float32(2.)+Z0)/mol,1.)
    signs = xp.where(old*moz < 0.,signs+1,signs)
    reset = signs >= 2
    moz = xp.where(reset,0.,moz); moz2 = xp.where(reset,0.,moz2)
    fm = xp.where(reset,0.,fm); fh = xp.where(reset,0.,fh)
    fm2 = xp.where(reset,0.,fm2); fh2 = xp.where(reset,0.,fh2)
    def correction(a):
        # Inactive unstable expression is made defined without changing active operands.
        u = (1.-np.float32(16.)*xp.minimum(a,0.))**np.float32(.25)
        temp2 = xp.log((1.+u*u)/2.); temp3 = xp.log((1.+u)/2.)
        m = 2.*temp3+temp2-2.*xp.arctan(u)+np.float32(1.5707963)
        return xp.where(moz < 0.,m,-5.*a), xp.where(moz < 0.,2*temp2,-5.*a)
    mn, hn = correction(moz); m2n, h2n = correction(moz2)
    if iteration == 1:
        fm,fh,fm2,fh2 = mn,hn,m2n,h2n
    else:
        fm,fh,fm2,fh2 = [.5*(a+b) for a,b in zip((fm,fh,fm2,fh2),(mn,hn,m2n,h2n))]
    fm=xp.minimum(fm,.9*lm); fh=xp.minimum(fh,.9*lm)
    fm2=xp.minimum(fm2,.9*l2); fh2=xp.minimum(fh2,.9*l2)
    def floor(a):
        return xp.where(xp.abs(a) <= np.float32(1.e-6),np.float32(1.e-6),a)
    cmfm, chfh = floor(lm-fm), floor(lm-fh)
    cm = VKC*VKC/(cmfm*cmfm); ch = VKC*VKC/(cmfm*chfh)
    fv = ur*xp.sqrt(cm)
    return cm,ch,(moz,signs,fm,fh,fm2,fh2,fv)


def surface_flux(v, dz, df, eair, rho, opt_gla, opt_stc, xp=jnp):
    tg = v['tg']; zero = xp.zeros_like(tg)
    ur=xp.maximum(xp.sqrt(v['uu']*v['uu']+v['vv']*v['vv']),1.)
    carry = (zero,xp.zeros_like(v['isnow']),zero,zero,zero,zero,xp.full_like(tg,.1))
    h=zero; cir=np.float32(.98)*SB; gamma=CPAIR*v['sfcprs']/(np.float32(.622)*HSUB)
    cgh=2.*at(df,v['isnow']+3,xp)/at(dz,v['isnow']+3,xp)
    stc0=at(v['stc'],v['isnow']+3,xp)
    for iteration in range(1,6):
        cm,ch,carry=sfcdif(iteration,v['snowh']+v['zlvl'],v['snowh'],v['q2'],v['sfctmp'],h,rho,ur,carry,xp)
        rah=xp.maximum(1.,1./(ch*ur))
        t=xp.minimum(50.,xp.maximum(-50.,tg-TFRZ)); ew,ei,dw,di=esat(t,xp)
        est=xp.where(t>0.,ew,ei); dest=xp.where(t>0.,dw,di)
        csh=rho*CPAIR/rah
        cev=rho*CPAIR/gamma/(1.+rah)
        if opt_gla == 2:
            cev=xp.where(v['snowh']>0.,cev,0.)
        ir=cir*fourth(tg)-np.float32(.98)*v['lwdn']
        sh=csh*(tg-v['sfctmp']); ev=cev*(est-eair); gh=cgh*(tg-stc0)
        b=v['sag']-ir-sh-ev-gh; a=4.*cir*cube(tg)+csh+cev*dest+cgh
        dtg=b/a
        ir=ir+4.*cir*cube(tg)*dtg; sh=sh+csh*dtg; ev=ev+cev*dest*dtg; gh=gh+cgh*dtg
        tg=tg+dtg; h=csh*(tg-v['sfctmp'])
        t=xp.minimum(50.,xp.maximum(-50.,tg-TFRZ)); ew,ei,_,_=esat(t,xp)
        est=xp.where(t>0.,ew,ei)
        qsfc=np.float32(.622)*est/(v['sfcprs']-np.float32(.378)*est)
    if opt_gla == 1 and opt_stc in (1,3):
        ice=xp.maximum(xp.maximum(v['smc'][0]-v['sh2o'][0],v['smc'][1]-v['sh2o'][1]),
                       xp.maximum(v['smc'][2]-v['sh2o'][2],v['smc'][3]-v['sh2o'][3]))
        cap=((ice>0.)|(v['snowh']>0.)) & (tg>TFRZ)
        tg=xp.where(cap,TFRZ,tg)
        _,ei,_,_=esat(xp.minimum(50.,xp.maximum(-50.,tg-TFRZ)),xp)
        qsfc=xp.where(cap,np.float32(.622)*ei/(v['sfcprs']-np.float32(.378)*ei),qsfc)
        ir=xp.where(cap,cir*fourth(tg)-np.float32(.98)*v['lwdn'],ir)
        sh=xp.where(cap,csh*(tg-v['sfctmp']),sh); ev=xp.where(cap,cev*(ei-eair),ev)
        gh=xp.where(cap,v['sag']-(ir+sh+ev),gh)
    eh=carry[-1]*VKC/(xp.log((2.+Z0)/Z0)-carry[-2])
    t2=xp.where(eh<np.float32(1.e-5),tg,tg-sh/(rho*CPAIR)*1./eh)
    q2=xp.where(eh<np.float32(1.e-5),qsfc,qsfc-ev/(HSUB*rho)*(1./eh+1.))
    v.update(tg=tg,qsfc=qsfc,cm=cm,ch=1./rah,fira=ir,fsh=sh,fgev=ev,ssoil=gh,t2m=t2,q2e=q2,ch2b=eh,
             emissi=xp.full_like(tg,.98),trad=(((v['lwdn']+ir)-(1.-np.float32(.98))*v['lwdn'])/(np.float32(.98)*SB))**np.float32(.25))


def soil_temperature(v, df, hc, opt_tbot, opt_stc, xp=jnp):
    top=v['isnow']+3; t=v['stc']; z=v['zsnso']; dt=v['dt']
    zero=xp.zeros_like(v['tg']); dd=[zero]*7; grad=[zero]*7; denom=[zero]*7; flux=[zero]*7
    for k in range(7):
        active=k>=top; first=k==top
        prev=zero if k==0 else z[k-1]
        denom[k]=xp.where(first,-z[k]*hc[k],(prev-z[k])*hc[k])
        if k<6:
            temp=xp.where(first,-z[k+1],prev-z[k+1])
            temp=xp.where(active,temp,1.)
            dd[k]=2./temp; grad[k]=2.*(t[k]-t[k+1])/temp
            old=zero if k==0 else df[k-1]*grad[k-1]
            flux[k]=xp.where(first,df[k]*grad[k]-v['ssoil'],df[k]*grad[k]-old)
        else:
            bot=zero
            if opt_tbot==2:
                grad[k]=(t[k]-v['tbot'])/(.5*(z[k-1]+z[k])-(-8.-v['snowh']))
                bot=-df[k]*grad[k]
            flux[k]=-bot-df[k-1]*grad[k-1]
    a=[]; b=[]; c=[]; d=[]
    for k in range(7):
        den=xp.where(k>=top,denom[k],1.)
        ai=zero if k==0 else -df[k-1]*dd[k-1]/den
        ai=xp.where(k==top,0.,ai)
        ci=-df[k]*dd[k]/den if k<6 else zero
        bi=-(ai+ci)
        if opt_stc==2:
            boundary=xp.where(k==top,.5*z[k]*z[k]*hc[k],1.)
            bi=xp.where(k==top,-ci+df[k]/boundary,bi)
        a.append(ai*dt);b.append(1.+bi*dt);c.append(ci*dt);d.append(flux[k]/(-den)*dt)
    p=[zero]*7; delta=[zero]*7
    for k in range(7):
        pp=zero if k==0 else p[k-1]; pd=zero if k==0 else delta[k-1]
        p[k]=xp.where(k==top,-c[k]/b[k],-c[k]*(1./(b[k]+a[k]*pp)))
        delta[k]=xp.where(k==top,d[k]/b[k],(d[k]-a[k]*pd)*(1./(b[k]+a[k]*pp)))
        p[k]=xp.where(k>=top,p[k],zero);delta[k]=xp.where(k>=top,delta[k],zero)
    p[6]=delta[6]
    for k in range(5,-1,-1):
        p[k]=p[k]*p[k+1]+delta[k]
    v['stc']=[xp.where(k>=top,t[k]+p[k],t[k]) for k in range(7)]
