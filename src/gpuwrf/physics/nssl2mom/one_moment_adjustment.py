"""Original ipconc0 homogeneous freezing and two-iteration saturation solve.

WRF4.7.1 module_mp_nssl_2mom.F24060..24438; no alternative physics closure.
Inputs/outputs use the existing canonical stack and per-point gs temporaries.
"""
import jax.numpy as jnp

from .indices import LC, LI, LV
from .satfun import ltemq_index,tabqis,tabqvs


def adjust(qx,cx,thetap,qwvp,temg,an,g,C,prec,dtp,dtpinv):
    R=prec.R;D=jnp.float64
    r=lambda x:jnp.asarray(x,R)
    W,mn,mx=jnp.where,jnp.minimum,jnp.maximum
    qx,cx=dict(qx),dict(cx)
    zero=jnp.zeros_like(temg)
    theta0,qv0,pi0,pk=g['theta0'],g['qv0'],g['pi0'],g['pk']
    felvcp,felscp,felfcp=g['felvcp'],g['felscp'],g['felfcp']
    # Original24080: ibfc1 and ipconc0 allow the homogeneous-freeze branch.
    on=(temg<r(235.15)) & (qx[LC]>0.)
    frac=mx(r(.25),mn(r(1.),((r(235.15)+r(1.))-temg)/r(4.))).astype(D)
    qtmp=(frac*qx[LC].astype(D)).astype(R)
    qx[LI]=W(on,qx[LI]+qtmp,qx[LI])
    # Preserve the original reassignment of qtmp using the previous level's
    # pre-GS cloud. Driver2979 sets t9's halo to zero; GS14004 copies an(lc).
    previous=jnp.concatenate([jnp.zeros_like(an[LC][...,:1]),an[LC][...,:-1]],axis=-1)
    from_previous=previous>qx[LC]
    qtmp=W(from_previous,(frac*previous.astype(D)).astype(R),qtmp)
    ctmp=W(from_previous,cx[LC]*qx[LC]*g['rho0']/W(from_previous,qtmp,r(1.)),zero)
    cx[LC]=W(on & ~from_previous,r(C.cwccn),cx[LC])
    thetap=W(on,thetap+g['fcc3']*qtmp,thetap)
    qx[LC]=W(on,qx[LC]-qtmp,qx[LC])
    cx[LC]=W(on,cx[LC]-ctmp,cx[LC])

    # Original24211: these are WRF's own saturation-solver input floors.
    qx[LV]=mx(r(0.),qwvp+qv0)
    qx[LC]=mx(r(0.),qx[LC]);qx[LI]=mx(r(0.),qx[LI])
    temg=(thetap+theta0)*(r(g['pinit'])+g['p2'])

    def saturation(temp):
        l=ltemq_index(temp,C,R)
        tab=tabqvs(l,C,R)
        qvs=C.rdorv*C.esbolton*tab/(g['pres']-C.esbolton*tab)
        qis=g['pqs']*tabqis(l,C,R)
        qss=qvs
        cold=temp<C.tfr
        ice=qx[LI]>C.qxmin[LI]
        qss=W(cold & (qx[LC]<=C.qxmin[LC]) & ice,qis,qss)
        mixed=cold & (qx[LC]>C.qxmin[LC]) & ice
        total=qx[LC]+qx[LI]
        mix=(qx[LC]*qvs+qx[LI]*qis)/W(mixed,total,r(1.))
        return qvs,qis,W(mixed,mix,qss)

    qvs,qis,qss=saturation(temg)
    for _ in range(2):
        dqcw,dqci=zero,zero
        dqwv=qx[LV]-qss
        sub=dqwv<0.
        enough=qx[LC]>-dqwv
        dqcw=W(sub,W(enough,dqwv,-qx[LC]),dqcw)
        dqwv=W(sub,W(enough,zero,dqwv+qx[LC]),dqwv)
        enough=qx[LI]>-dqwv
        dqci=W(sub,W(enough,dqwv,-qx[LI]),dqci)
        dqwv=W(sub,W(enough,zero,dqwv+qx[LI]),dqwv)
        qwvp=W(sub,qwvp-(dqcw+dqci),qwvp)
        qitmp=qx[LI]
        fcci=W(qitmp>=C.qxmin[LI],qx[LI]/W(qitmp>=C.qxmin[LI],qitmp,r(1.)),r(1.))
        qx[LC]=W(sub,qx[LC]+dqcw,qx[LC])
        qx[LI]=W(sub,qx[LI]+dqci*fcci,qx[LI])
        thetap=W(sub,thetap+r(1.)/pi0*(felvcp*dqcw+felscp*dqci),thetap)

        cond=dqwv>=0.
        qitmp=qx[LI]
        cold=(temg<C.tfr)&(temg>r(235.15))
        fracl=W(cold,mx(mn(r(1.),(temg-r(233.15))/r(20.)),r(0.)),r(1.))
        fracl=W(temg<=r(235.15),r(0.),fracl)
        fraci=r(1.)-fracl
        gamss=(felvcp*fracl+felscp*fraci)/pi0
        dqvcnd=dqwv/(r(1.)+g['fcqv1']*qss/(temg-C.cbw)**2)
        pure_ice=(temg<C.tfr)&(qx[LC]==0.)&(qitmp>C.qxmin[LI])
        ice_cond=dqwv/(r(1.)+g['fcqv2']*qss/(temg-C.cbi)**2)
        dqvcnd=W(pure_ice,ice_cond,dqvcnd)
        mixed=(temg<C.tfr)&(qx[LC]>0.)&(qitmp>C.qxmin[LI])
        cdw=C.caw*pi0*g['tfrcbw']/(temg-C.cbw)**2
        cdi=C.cai*pi0*g['tfrcbi']/(temg-C.cbi)**2
        denom1=qx[LC]+qitmp
        denom2=r(1.)+gamss*(qx[LC]*qvs*cdw+qitmp*qis*cdi)/W(mixed,denom1,r(1.))
        dqvcnd=W(mixed,dqwv/denom2,dqvcnd)
        fcci=W(qitmp>C.qxmin[LI],qx[LI]/W(qitmp>C.qxmin[LI],qitmp,r(1.)),r(1.))
        dqcw=dqvcnd*fracl;dqci=dqvcnd*fraci
        thetap=W(cond,thetap+(felvcp*dqcw+felscp*dqci)/pi0,thetap)
        qwvp=W(cond,qwvp-dqvcnd,qwvp)
        qx[LC]=W(cond,qx[LC]+dqcw,qx[LC])
        qx[LI]=W(cond,qx[LI]+dqci*fcci,qx[LI])
        temg=(thetap+theta0)*pk
        qx[LV]=mx(qwvp+qv0,r(0.))
        qx[LC]=mx(r(0.),qx[LC])
        # Original24405 only floors qitmp, not the stored qx(li).
        qvs,qis,qss=saturation(temg)
    return qx,cx,thetap,qwvp,temg
