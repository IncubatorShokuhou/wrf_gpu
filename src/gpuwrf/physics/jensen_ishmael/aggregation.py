"""Original aggregation3855..4341/col1, specialized to its seven active pairs.

Only categories3/4/5 enter these routines. Unused 7x7 energy-transfer scratch
does not reach any returned variable and is omitted. Pair traversal and the
original sums remain ordered; collection tables are explicit array operands.
"""
import jax.numpy as jnp
from .math import _real,gamma


def aggregation(fields,coltab,coltabn,*,xp=jnp,observer=None):
    f=fields;_,r=_real(f['qdum1'],xp);W=xp.where
    if not getattr(coltab,'_jensen_table_ref',False):coltab=xp.asarray(coltab)
    if not getattr(coltabn,'_jensen_table_ref',False):coltabn=xp.asarray(coltabn)
    q=tuple(f['qdum'+str(i)] for i in (1,2,3))
    n=tuple(f['ndum'+str(i)]*f['rhoairdum'] for i in (1,2,3))
    d=tuple(f['ddum'+str(i)] for i in (1,2,3))
    # Original table dimensions60; all three habits have the same table bounds.
    lower=xp.log(r(1.e-6));upper=xp.log(r(1.e-2))
    spacing=r(59.)/(upper-lower)
    minimum=tuple(xp.maximum(r(1.e-10),a) for a in d)
    # The literal-reference backend models gfortran's observed vector-libm
    # boundary. The compiled port uses its native log implementation.
    native_pair=getattr(xp,'original_pair_log',None)
    logs=((*native_pair(*minimum[:2]),xp.log(minimum[2])) if native_pair is not None
          else tuple(xp.log(a) for a in minimum))
    coordinate=tuple(xp.maximum(r(1.0001),xp.minimum(r(.9999)*r(60.),
                       spacing*(a-lower)+r(1.))) for a in logs)
    index=tuple(xp.trunc(a).astype('int32')-1 for a in coordinate)
    w2=tuple(a-(i+1).astype(a.dtype) for a,i in zip(coordinate,index))
    w1=tuple(r(1.)-a for a in w2)
    denfac=xp.sqrt(r(1.)/f['rhoairdum'])
    temperature=f['tempdum']-r(273.15)

    def efficiency(density,phi):
        rhoeff=W(density<=r(400.),r(1.),r(-.001923)*density+r(1.76916))
        rhoeff=xp.minimum(xp.maximum(rhoeff,r(0.)),r(1.))
        phieff=W(phi<=r(.03),r(1.),W(phi<r(.5),r(.0001)*xp.power(phi+r(.07),r(-4.)),r(0.)))
        phieff=W(phieff<=r(.001),r(0.),phieff)
        phieff=xp.minimum(xp.maximum(phieff,r(.01)),r(1.))
        return phieff*rhoeff

    phi1=xp.minimum(f['phidum1'],r(1.)/f['phidum1'])
    phi2=xp.minimum(f['phidum2'],r(1.)/f['phidum2'])
    eff12=efficiency(xp.maximum(f['rhodum1'],f['rhodum2']),xp.maximum(phi1,phi2))
    eff1=efficiency(f['rhodum1'],phi1);eff2=efficiency(f['rhodum2'],phi2)

    def collect(mx,my,pair,factor):
        # Original col1 specialization: mz=5, icf=5, cr=1, NZ=1.
        eff=xp.minimum(r(.2),xp.power(r(10.),r(.035)*temperature-r(.7)))*factor
        eff=W(xp.abs(temperature+r(14.))<=r(2.),r(1.4)*factor,eff)
        ix,iy=index[mx],index[my]
        def interpolate(table):
            return (w1[mx]*w1[my]*table[ix,iy,pair-1]
                    +w2[mx]*w1[my]*table[ix+1,iy,pair-1]
                    +w1[mx]*w2[my]*table[ix,iy+1,pair-1]
                    +w2[mx]*w2[my]*table[ix+1,iy+1,pair-1])
        prefactor=f['dtdum']*r(1.)*r(.785)*eff*denfac/f['rhoairdum']*n[mx]*n[my]
        mass=xp.minimum(prefactor*interpolate(coltab),q[mx])
        number=xp.minimum(prefactor*interpolate(coltabn),n[mx])
        weight=xp.minimum(xp.maximum(r(0.),r(100.)*xp.abs(mass)/xp.maximum(r(1.e-20),q[mx])),r(1.))
        delta=number*(r(1.)-weight)+weight*n[mx]*mass/xp.maximum(r(1.e-20),q[mx])
        if observer is not None:
            observer(pair,dict(eff=eff,efdum=factor,tmax=temperature,denfac=denfac,dn0=f['rhoairdum'],
                              enmx=n[mx],enmy=n[my],w1mx=w1[mx],w2mx=w2[mx],w1my=w1[my],w2my=w2[my],
                              colamt=mass,colamtn=number,wght=weight,deltan=delta,rmx=q[mx]))
        return mass,delta

    a,an=collect(0,1,23,eff12);b,bn=collect(1,0,24,eff12)
    c,cn=collect(0,2,25,eff1);d,dn=collect(1,2,29,eff2)
    e,en=collect(0,0,22,eff1);f4,f4n=collect(1,1,28,eff2)
    _,gn=collect(2,2,35,r(1.))
    sink1=r(1.)*a+r(1.)*c+r(1.)*e
    sink2=r(1.)*b+r(1.)*d+r(1.)*f4
    active1=(sink1>q[0])&(q[0]>r(1.e-8));active2=(sink2>q[1])&(q[1]>r(1.e-8))
    ratio1=q[0]/W(active1,sink1,r(1.));ratio2=q[1]/W(active2,sink2,r(1.))
    a=W(active1,a*ratio1,a);c=W(active1,c*ratio1,c);e=W(active1,e*ratio1,e)
    b=W(active2,b*ratio2,b);d=W(active2,d*ratio2,d);f4=W(active2,f4*ratio2,f4)
    q1=r(-1.)*a-r(1.)*c-r(1.)*e
    q2=r(-1.)*b-r(1.)*d-r(1.)*f4
    q3=r(1.)*a+r(1.)*b+r(1.)*c+r(1.)*d+r(1.)*e+r(1.)*f4
    n1=r(-1.)*an-r(1.)*cn-r(1.)*en
    n2=r(-1.)*bn-r(1.)*dn-r(1.)*f4n
    # Original4310 omits the planar/columnar + aggregate number transfers
    # (CN/DN) from aggregate number gain, while retaining their mass gain.
    n3=r(.5)*an+r(.5)*bn+r(.5)*en+r(.5)*f4n-r(.5)*gn
    q1=xp.minimum(q1,r(0.));q2=xp.minimum(q2,r(0.));q3=xp.maximum(q3,r(0.))
    cfmas=r(3.14159)/r(6.)*r(50.)*r(.2)
    diameter=xp.power((q[2]+q3)*f['rhoairdum']/(xp.maximum(n[2]+n3,r(1.25e-7))*cfmas
                     *gamma(r(7.),xp=xp)/gamma(r(4.),xp=xp)),r(1.)/r(3.))
    return dict(qagg1=q1,qagg2=q2,qagg3=q3,nagg1=n1/f['rhoairdum'],
                nagg2=n2/f['rhoairdum'],nagg3=n3/f['rhoairdum'],ddum3=diameter)
