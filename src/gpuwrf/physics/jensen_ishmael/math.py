"""Literal WRF4.7.1 Jensen math; component qualification is required before use.

Polysvp3427, get_igr3820, GAMMA3583, gammln3552 and wet_growth_check3404.
REAL follows the argument dtype; gammln preserves its original REAL*8 island.
The NumPy backend executes literal separate operations for source-oracle tests.
"""
import jax.numpy as jnp
import numpy as np

IGR=(.910547,.81807,.6874,.60127,1.59767,2.32423,2.08818,1.61921,1.15865,.863071,
     .617586,.453917,.351975,.28794,.269298,.28794,.333623,.418883,.56992,.796458,
     1.14325,1.64103,1.90138,1.82653,1.61921,1.47436,1.32463,1.25556,1.22239,1.206,
     1.11522,1.10751,1.10738,1.11484,1.12234,1.12221,1.14529,1.16884,1.20104,1.22573,
     1.25094,1.27666,1.31183,1.3388,1.35704,1.37553,1.38479,1.39411,1.40349,1.41294,1.42245,
     1.43202,1.44166,1.45137,1.46114,1.47097,1.48087,1.50105,1.50087,1.51098)


def _real(value,xp):
    a=xp.asarray(value)
    return a,lambda v:xp.asarray(v,dtype=a.dtype)


def polysvp(temperature,kind,*,xp=jnp):
    t,r=_real(temperature,xp)
    coeff=((6.11147274,.503160820,.188439774e-1,.420895665e-3,.615021634e-5,
            .602588177e-7,.385852041e-9,.146898966e-11,.252751365e-14) if kind==1 else
           (6.11239921,.443987641,.142986287e-1,.264847430e-3,.302950461e-5,
            .206739458e-7,.640689451e-10,-.952447341e-13,-.976195544e-15))
    dt=xp.maximum(r(-80.),t-r(273.16))
    result=r(coeff[-1])
    for value in reversed(coeff[:-1]):
        result=r(value)+dt*result
    return result*r(100.)


def get_igr(temperature,*,data=IGR,xp=jnp):
    t,r=_real(temperature,xp)
    if not getattr(data,'_jensen_table_ref',False):data=xp.asarray(data,dtype=t.dtype)
    delta=t-r(273.15);absolute=xp.abs(delta)
    integer=xp.trunc(delta).astype(np.int32)
    first=xp.maximum(-integer,1)-1
    second=xp.minimum(-integer+1,60)-1
    # Safe inactive indices do not change original values on the selected arm.
    first=xp.minimum(first,59);second=xp.maximum(second,0)
    dum=(xp.abs(integer.astype(t.dtype))+r(1.))-absolute
    middle=dum*data[first]+(r(1.)-dum)*data[second]
    dum=r(1.)-absolute
    warm=dum*r(1.)+(r(1.)-dum)*data[0]
    dum=r(60.)-absolute
    cold=dum*data[58]+(r(1.)-dum)*data[59]
    result=xp.where(delta<r(-60.),data[59],r(1.))
    result=xp.where((delta>=r(-60.))&(delta<r(-59.)),cold,result)
    result=xp.where((delta>r(-1.))&(delta<=r(0.)),warm,result)
    return xp.where((delta>=r(-59.))&(delta<=r(-1.)),middle,result)


def gamma(argument,*,xp=jnp):
    """Original Cody REAL gamma, including its original single-precision limits."""
    x,r=_real(argument,xp);W=xp.where
    pi=r(3.14159265);one=r(1.);zero=r(0.);half=r(.5)
    eps=r(1.19e-7);xmin=r(1.18e-38);xbig=r(35.040);xinf=r(3.4e38)
    negative=x<=zero
    ay=zero-x;integer=xp.trunc(ay);fraction=ay-integer
    singular=negative&(fraction==zero)
    parity=negative&(integer!=xp.trunc(integer*half)*r(2.))
    fact=r(-3.14159265)/xp.sin(pi*W(negative&~singular,fraction,half))
    y=W(negative,ay+one,x)
    terminal=singular|(y<xmin)|(y>xbig)
    rational=(y>=eps)&(y<r(12.))
    yr=W(rational,y,one)
    n=W(yr<one,0,xp.trunc(yr).astype(np.int32)-1)
    reduced=W(yr<one,yr+one,yr-n.astype(x.dtype))
    z=W(yr<one,yr,reduced-one)
    p=(-1.71618513886549492533811,2.47656508055759199108314e1,-3.79804256470945635097577e2,
       6.29331155312818442661052e2,8.66966202790413211295064e2,-3.14512729688483675254357e4,
       -3.61444134186911729807069e4,6.64561438202405440627855e4)
    q=(-3.08402300119738975254353e1,3.15350626979604161529144e2,-1.01515636749021914166146e3,
       -3.10777167157231109440444e3,2.25381184209801510330112e4,4.75584627752788110767815e3,
       -1.34659959864969306392456e5,-1.15132259675553483497211e5)
    numerator=zero;denominator=one
    for a,b in zip(p,q):
        numerator=(numerator+r(a))*z
        denominator=denominator*z+r(b)
    result=numerator/denominator+one
    result=W(yr<reduced,result/yr,result)
    for i in range(10):  # Original positive rational interval has N<=10.
        active=i<n
        result=W(active,result*reduced,result)
        reduced=W(active,reduced+one,reduced)
    large_y=W((y>=r(12.))&(y<=xbig),y,r(12.))
    square=large_y*large_y
    coefficients=(-1.910444077728e-3,8.4171387781295e-4,-5.952379913043012e-4,
                  7.93650793500350248e-4,-2.777777777777681622553e-3,8.333333333333333331554247e-2)
    total=r(5.7083835261e-3)
    for c in coefficients:
        total=total/square+r(c)
    total=total/large_y-large_y+r(.9189385332046727417803297)
    total=total+(large_y-half)*xp.log(large_y)
    result=W(y>=r(12.),xp.exp(total),result)
    result=W(y<eps,one/W(y>=xmin,y,one),result)
    result=W(parity,-result,result)
    result=W(negative&~singular,fact/result,result)
    # Original GOTO900 skips reflection/parity on overflow and singularities.
    return W(terminal,xinf,result)


def gammln(argument,*,xp=jnp):
    a,r=_real(argument,xp)
    x=a.astype(np.float64);y=x
    tmp=x+np.float64(5.5)
    tmp=(x+np.float64(.5))*xp.log(tmp)-tmp
    ser=np.float64(1.000000000190015)
    for c in (76.18009172947146,-86.50532032941677,24.01409824083091,
              -1.231739572450155,.1208650973866179e-2,-.5395239384953e-5):
        y=y+np.float64(1.)
        ser=ser+np.float64(c)/y
    return (tmp+xp.log(np.float64(2.5066282746310005)*ser/x)).astype(a.dtype)


def capacitance_gamma(ani,dsdum,nu,alphstr,i_gammnu,*,xp=jnp):
    ani,r=_real(ani,xp);W=xp.where
    planar=dsdum<=r(1.)
    a1=W(planar,r(.6369427),r(.5714285))
    a2=W(planar,r(.57)*a1,r(.75)*a1)
    b1=W(planar,r(0.),r(-1.));b2=W(planar,r(.95),r(-.18))
    c1=W(planar,a1*xp.power(alphstr,b1),a1*xp.power(alphstr,b1+r(1.)))
    c2=W(planar,a2*xp.power(alphstr,b2),a2*xp.power(alphstr,b2+r(1.)))
    d1=W(planar,b1*(dsdum-r(1.))+r(1.),b1*(dsdum-r(1.))+dsdum)
    d2=W(planar,b2*(dsdum-r(1.))+r(1.),b2*(dsdum-r(1.))+dsdum)
    g1=gamma(nu+d1,xp=xp);g2=gamma(nu+d2,xp=xp)
    result=c1*xp.power(ani,d1)*g1*i_gammnu+c2*xp.power(ani,d2)*g2*i_gammnu
    return W((dsdum==r(1.))&(nu==r(1.)),ani*gamma(nu+r(1.),xp=xp)*i_gammnu,result)


def wet_growth_check(nu,temp,rhodum,xxlv,xxlf,qvdum,dv,kt,qs0,fvdum,fhdum,rimedum,rni,nidum,dgflag,*,xp=jnp):
    temp,r=_real(temp,xp)
    dum=nidum*nu*r(2.)*rni
    wetg=r(2.)*r(3.14159265)*dum*(kt*fhdum*(r(273.15)-temp)+rhodum*xxlv*dv*fvdum*(qs0-qvdum))/(xxlf+r(4218.)*(temp-r(273.15)))
    return xp.where(rimedum/rhodum>wetg,False,dgflag)
