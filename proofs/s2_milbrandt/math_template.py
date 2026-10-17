"""Literal WRF MY2 helper functions (REAL except gamma's explicit DOUBLE island)."""
import numpy as np
import jax.numpy as jnp
from .rounding import real


def _gamma(xx):
    # WRF uses four (not six) terms in gamma, but six in gammln.
    R = np.asarray(xx).dtype.type
    x = np.float64(xx)
    y = x
    tmp = x + np.float64(5.5)
    tmp = (x + np.float64(.5)) * np.log(tmp) - tmp
    ser = np.float64(1.000000000190015)
    for coeff in (76.18009172947146, -86.50532032941677,
                  24.01409824083091, -1.231739572450155):
        y = y + np.float64(1.)
        ser = ser + np.float64(coeff)/y
    return R(np.exp(tmp + np.log(np.float64(2.5066282746310005)*ser/x)))


def _dim(a, b):
    return jnp.maximum(a-b, jnp.asarray(0., jnp.asarray(a).dtype))


def _dm_x(de, q, inv_n, inv_c, inv_d):
    return jnp.exp(inv_d*jnp.log(de*q*inv_n*inv_c))


_ilamda_x = _dm_x


def _des_of_ds(d, maximum, eds, fds):
    return jnp.minimum(maximum, eds*jnp.exp(fds*jnp.log(d)))


def _n_cooper(trpl, t):
    r = lambda x: real(x, t.dtype)
    return r(5.)*jnp.exp(r(.304)*(trpl-jnp.maximum(r(233.),t)))


def _nos_thompson(trpl, t):
    r = lambda x: real(x, t.dtype)
    return jnp.minimum(r(2.e8),r(2.e6)*jnp.exp(-r(.12)*jnp.minimum(-r(.001),t-trpl)))


def _polysvp(t, kind):
    r = lambda x: real(x, t.dtype)
    coeffs = ((6.11239921,.443987641,.0142986287,.000264847430,.00000302950461,
               .0000000206739458,.0000000000640689451,-.0000000000000952447341,
               -.000000000000000976195544),
              (6.11147274,.503160820,.0188439774,.000420895665,.00000615021634,
               .0000000602588177,.000000000385852041,.00000000000146898966,
               .00000000000000252751365))[kind]
    dt=jnp.maximum(r(-80.),t-r(273.16))
    result=r(coeffs[-1])
    for c in coeffs[-2::-1]:
        result=r(c)+dt*result
    return result*r(100.)


def _qsat(t, pres, kind):
    tmp=_polysvp(t,kind)
    return real(.622,t.dtype)*tmp/(pres-tmp)


def _nccnfnc(win,tin,pin,ccntype=2):
    if ccntype != 2:
        raise ValueError('WRF mp9 driver fixes CCNtype=2')
    r=lambda v:real(v,tin.dtype)
    x=jnp.log10(win*r(100.)); x2=x*x; x3=x2*x; x4=x2*x2
    t=tin-r(273.15); t2=t*t; t3=t2*t; t4=t2*t2
    p=pin*r(.01); p2=p*p
    a=r(0.); b=r(0.)
    c=-r(2.112e-9)*t4+r(3.9836e-8)*t3+r(2.3703e-6)*t2-r(1.4542e-4)*t-r(.0698)
    d=-r(4.210e-8)*t4+r(5.5745e-7)*t3+r(1.8460e-5)*t2+r(9.6078e-4)*t+r(.7120)
    e=r(1.434e-7)*t4-r(1.6455e-6)*t3-r(4.3334e-5)*t2-r(7.6720e-3)*t+r(1.0056)
    f=r(1.340e-6)*p2-r(3.5114e-3)*p+r(1.9453)
    g=r(4.226e-3)*x4-r(5.6012e-3)*x3-r(8.7846e-2)*x2+r(2.7435e-2)*x+r(.9932)
    h=r(5.811e-9)*t4+r(1.5589e-7)*t3-r(3.8623e-5)*t2+r(1.4471e-3)*t+r(.1496)
    y=a*x4+b*x3+c*x2+d*x+e+(f*g*h)
    return r(10.)**jnp.maximum(r(0.),y)*r(1.e6)


def _sxfnc(win,tin,pin,qsw,qsi,ccntype=2,wrt=1):
    if ccntype != 2:
        raise ValueError('WRF mp9 driver fixes CCNtype=2')
    r=lambda v:real(v,tin.dtype)
    x=jnp.log10(jnp.maximum(win,r(1.e-20))*r(100.)); x2=x*x; x3=x2*x; x4=x2*x2
    t=tin; t2=t*t; t3=t2*t
    p=pin*r(.01); p2=p*p
    a=r(3.80e-5)*t2+r(1.65e-4)*t+r(.0988)
    b=-r(7.38e-5)*t2-r(2.53e-3)*t-r(.323)
    c=r(8.39e-5)*t2+r(3.96e-3)*t+r(.350)
    d=-r(1.88e-6)*t2-r(1.33e-3)*t-r(.0373)
    f=-r(1.9761e-6)*p2+r(4.1473e-3)*p-r(1.771)
    g=r(.1539)*x4-r(.5575)*x3+r(.9262)*x2-r(.3498)*x-r(.1293)
    h=-r(8.035e-9)*(t2*t2)+r(3.162e-7)*t3+r(1.029e-5)*t2-r(5.931e-4)*t+r(.0562)
    pcorr=f*g*h
    t2corr=r(.98888)-r(5.0525e-4)*t-r(1.7598e-5)*t2-r(8.3308e-8)*t3
    sw=(a*x3+b*x2+c*x+d)+pcorr
    sw=r(1.)+r(.01)*sw
    qv=qsw*sw
    si=qv/qsi
    si=si*t2corr
    return jnp.where(win<=r(0.),r(1.),sw if wrt==1 else si)
