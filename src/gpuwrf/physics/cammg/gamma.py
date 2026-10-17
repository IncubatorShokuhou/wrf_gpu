"""Original CAM MG Cody/Stoltz GAMMA (WRF cldwat2m_micro), DOUBLE arithmetic.

The original single-precision-range limits inside this r8 routine are retained.
This is not a substitution of scipy/libdevice gamma for the WRF algorithm.
"""
import numpy as np
import jax.numpy as jnp

from ._support import _cond,_loop,_d,_int

_P=(-1.71618513886549492533811e0,2.47656508055759199108314e1,
    -3.79804256470945635097577e2,6.29331155312818442661052e2,
    8.66966202790413211295064e2,-3.14512729688483675254357e4,
    -3.61444134186911729807069e4,6.64561438202405440627855e4)
_Q=(-3.08402300119738975254353e1,3.15350626979604161529144e2,
    -1.01515636749021914166146e3,-3.10777167157231109440444e3,
    2.25381184209801510330112e4,4.75584627752788110767815e3,
    -1.34659959864969306392456e5,-1.15132259675553483497211e5)
_C=(-1.910444077728e-3,8.4171387781295e-4,-5.952379913043012e-4,
    7.93650793500350248e-4,-2.777777777777681622553e-3,
    8.333333333333333331554247e-2,5.7083835261e-3)


def gamma(x,*,xp=jnp,_globals=None):
    x=xp.asarray(x,dtype=xp.float64)
    one=_d(1.);half=_d(.5);zero=_d(0.);xinf=_d(3.4e38)
    pi=_d(3.1415926535897932384626434)
    y=xp.where(x<=zero,0-x,x)
    y1=xp.trunc(y);fraction=y-y1
    parity=(x<=zero)&(y1!=xp.trunc(y1*half)*_d(2.))
    # The inactive singular denominator is replaced solely inside a masked
    # branch; the returned singular result remains the original XINF.
    fact=xp.where(x<=zero,-pi/xp.sin(pi*xp.where(fraction!=zero,fraction,one)),one)
    y=xp.where(x<=zero,y+one,y)

    def positive(y):
        def tiny(y):return xp.where(y>=_d(1.18e-38),one/y,xinf)
        def regular(y):
            y1=y
            n=xp.where(y<one,_int(0),_int(y)-1)
            y=xp.where(y<one,y+one,y-n.astype(xp.float64))
            z=xp.where(y1<one,y1,y-one)
            num=zero;den=one
            for p,q in zip(_P,_Q,strict=True):
                num=(num+_d(p))*z
                den=den*z+_d(q)
            result=num/den+one
            result=xp.where(y1<y,result/y1,result)
            def multiply(i,carry):
                result,yy=carry
                return result*yy,yy+one
            result,_=_loop(1,n,1,multiply,(result,y),xp)
            return result
        def large(y):
            ysq=y*y;total=_d(_C[6])
            for c in _C[:6]:total=total/ysq+_d(c)
            total=total/y-y+_d(.9189385332046727417803297)
            total=total+(y-half)*xp.log(y)
            return xp.where(y<=_d(35.040),xp.exp(total),xinf)
        return _cond(y<_d(1.19e-7),tiny,lambda y:_cond(y<_d(12.),regular,large,y,xp),y,xp)
    result=positive(y)
    result=xp.where(parity,0-result,result)
    result=xp.where(fact!=one,fact/result,result)
    return xp.where((x<=zero)&(fraction==zero),xinf,result)
