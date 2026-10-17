"""Literal NTU special functions, module_mp_ntu.F:607–767.

GAMLN and the aerosol inversion/error-function helpers have original DOUBLE
PRECISION work. The other routines follow the input REAL kind. A native
caller enables x64 while tracing these explicitly typed original islands.
"""
import numpy as np


def polysvp(t, phase, *, xp=np):
    r=np.dtype(t.dtype).type
    ice=(6.11147274,.503160820,.188439774e-1,.420895665e-3,.615021634e-5,
         .602588177e-7,.385852041e-9,.146898966e-11,.252751365e-14)
    water=(6.11239921,.443987641,.142986287e-1,.264847430e-3,.302950461e-5,
           .206739458e-7,.640689451e-10,-.952447341e-13,-.976195544e-15)
    dt=xp.maximum(r(-80),t-r(273.16))
    def evaluate(coeff):
        result=r(coeff[-1])
        for c in reversed(coeff[:-1]):
            result=r(c)+dt*result
        return result*r(100)
    if isinstance(phase,(int,np.integer)):
        return evaluate(ice if phase==1 else water)
    return xp.where(phase==1,evaluate(ice),evaluate(water))


def gamln(xx, *, xp=np):
    """Original Numerical Recipes coefficients and association; F:641–672."""
    x=xp.asarray(xx,dtype=np.float64)
    y=x
    tmp=x+np.float64(5.5)
    tmp=(x+np.float64(.5))*xp.log(tmp)-tmp
    ser=np.float64(1.000000000190015)
    for c in (76.18009172947146,-86.50532032941677,24.01409824083091,
              -1.231739572450155,.1208650973866179e-2,-.5395239384953e-5):
        y=y+np.float64(1)
        ser=ser+np.float64(c)/y
    return xp.asarray(tmp+xp.log(np.float64(2.5066282746310005)*ser/x),dtype=xx.dtype)


def gammp(a,x,*,xp=np):
    """Original series/continued fraction with converged lanes held unchanged."""
    r=np.dtype(a.dtype).type
    gln=gamln(a,xp=xp)
    ap=a
    summ=r(1)/a
    de=summ
    active=(x>r(0)) & (x<a+r(1))
    # Bounded native fori loop is replaced by the kernel entry's loop adapter.
    def series(n,state):
        ap,summ,de,active=state
        ap1=ap+r(1);de1=de*x/ap1;summ1=summ+de1
        return (xp.where(active,ap1,ap),xp.where(active,summ1,summ),
                xp.where(active,de1,de),active & (xp.abs(de1)>=xp.abs(summ1)*r(3.e-7)))
    ap,summ,de,active=_loop(1,501,series,(ap,summ,de,active),xp=xp)
    ser=xp.where(x<=r(0),r(0),summ*xp.exp(-x+a*xp.log(x)-gln))
    b=x+r(1)-a;c=xp.full_like(x,r(1)/r(1.e-30));d=r(1)/b;h=d
    active=x>=a+r(1)
    def fraction(i,state):
        b,c,d,h,active=state
        ri=xp.asarray(i,dtype=a.dtype)
        an=-ri*(ri-a);b1=b+r(2);d1=an*d+b1
        d1=xp.where(xp.abs(d1)<r(1.e-30),r(1.e-30),d1)
        c1=b1+an/c;c1=xp.where(xp.abs(c1)<r(1.e-30),r(1.e-30),c1)
        d1=r(1)/d1;de=d1*c1;h1=h*de
        return (xp.where(active,b1,b),xp.where(active,c1,c),xp.where(active,d1,d),
                xp.where(active,h1,h),active & (xp.abs(de-r(1))>=r(3.e-7)))
    b,c,d,h,active=_loop(1,501,fraction,(b,c,d,h,active),xp=xp)
    cf=xp.exp(-x+a*xp.log(x)-gln)*h
    return xp.where(x<a+r(1),ser,r(1)-cf)


def _loop(start,stop,body,state,*,xp):
    if xp is np or getattr(xp,"literal",False):
        for i in range(start,stop):
            state=body(np.int32(i),state)
            if not np.any(state[-1]):
                break
        return state
    import jax
    import jax.numpy as jnp
    from .support import any_true
    def condition(carry):
        i,state=carry
        return (i<stop) & any_true(state[-1],xp=jnp)
    def advance(carry):
        i,state=carry
        return i+np.int32(1),body(i,state)
    return jax.lax.while_loop(condition,advance,(np.int32(start),state))[1]


def gamin(p,xmax,*,xp=np):
    return gammp(p,xmax,xp=xp)*xp.exp(gamln(p,xp=xp))


def erf(x,*,xp=np):
    if getattr(xp,'literal',False):return xp.erf(x)
    if xp is np:
        from scipy.special import erf as scipy_erf
        return scipy_erf(x)
    from . import _native
    if _native.active.get():
        from gpuwrf.kernels._ntu_math import erf as resident_erf
        return resident_erf(x)
    import jax
    return jax.lax.erf(x)


def dlnx(rx,xmode,sigma,n,*,xp=np):
    return dlnx2(xp.asarray(rx,dtype=np.float64),xmode,sigma,n,xp=xp)


def dlnx2(drx,xmode,sigma,n,*,xp=np):
    mode=xp.asarray(xmode,dtype=np.float64)
    std=xp.asarray(sigma,dtype=np.float64)
    return (xp.log(drx/mode)-xp.power(std,np.float32(2))*xp.asarray(n,dtype=np.float64))/(xp.sqrt(np.float64(2))*std)


def dpdf(x,*,xp=np):
    return xp.exp(-x*x)/xp.sqrt(np.float64(np.arccos(-1.)))


def find_rc0(xafrc,cmode,cstdv,wmas,tblxa,tblrc,*,xp=np):
    """Original DOUBLE Newton inversion, FIND_RC0/F:318–447.

    The three aerosol modes are statically unrolled; the bounded Newton loop
    is resident in the caller's column kernel. Output alone is WRF REAL.
    """
    r=np.dtype(cmode.dtype).type
    d=lambda x:xp.asarray(x,dtype=np.float64)
    w=[d(wmas[...,i]) for i in range(3)]
    std=[d(cstdv[...,i]) for i in range(3)]
    mode=[xp.log(d(cmode[...,i]))+d(r(3)*cstdv[...,i]*cstdv[...,i]) for i in range(3)]
    sqrt2=xp.sqrt(np.float64(2))
    def yequ(x):
        result=np.float64(0)
        for i in range(3):
            dx=(x-mode[i])/std[i]/sqrt2
            result=result+w[i]*(np.float64(1)+erf(dx,xp=xp))*np.float64(.5)
        return result-xafrc
    def dyequ(x):
        result=np.float64(0)
        for i in range(3):
            dx=(x-mode[i])/std[i]/sqrt2
            result=result+w[i]*dpdf(dx,xp=xp)/std[i]/sqrt2
        return result
    guess=xp.broadcast_to(tblrc[0],xafrc.shape)
    for i in range(1,25):
        guess=xp.where(xafrc>=tblxa[i],tblrc[i],guess)
    x1ma=np.float64(1)-xafrc
    ul=(np.float32(1)-erf(np.float64(5)/sqrt2,xp=xp))*np.float32(.5)
    low=xafrc<np.float64(1.e-10)
    high=x1ma<ul
    # NaN fractions (dcn/tcn = 0/0 on cloud-free levels, also in WRF) make y0
    # NaN, so the bounded Newton loop can only run all 50 trips and return NaN.
    # Such levels skip the loop and take that NaN directly (t-ntucompile);
    # every finite level follows the identical iteration.
    undefined=xp.isnan(xafrc)
    active=~(low|high|undefined)
    value=guess
    y0=yequ(value)
    result=xp.exp(value)
    def iterate(i,state):
        value,y0,result,active=state
        dy=dyequ(value)
        dx=-y0/dy
        dx=xp.where(xp.abs(dx)>np.float64(3),dx/xp.abs(dx/np.float64(.4)),dx)
        converged=xp.abs(dx)<np.float64(r(1.e-5))
        selected=xp.minimum(r(99),xp.maximum(xp.exp(value+dx),r(1.e-9)))
        result=xp.where(active&converged,selected,result)
        next_y=yequ(value+dx)
        crossing=y0*next_y<np.float32(0)
        dx=xp.where(crossing,dx*np.float64(.5),dx)
        next_y=xp.where(crossing,yequ(value+dx),next_y)
        remaining=active&~converged
        value=xp.where(remaining,value+dx,value)
        y0=xp.where(remaining,next_y,y0)
        return value,y0,result,remaining
    value,y0,result,active=_loop(0,50,iterate,(value,y0,result,active),xp=xp)
    result=xp.where(active,xp.exp(value),result)
    result=xp.where(undefined,xafrc,result)
    result=xp.where(low,r(1.e-9),xp.where(high,r(99),result))
    return xp.asarray(result,dtype=cmode.dtype)
