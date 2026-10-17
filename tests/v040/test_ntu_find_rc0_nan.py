"""NTU FIND_RC0: NaN fractions skip the bounded Newton loop (t-ntucompile).

dcn/tcn = 0/0 on cloud-free levels gives a NaN fraction (also in WRF); its Newton
residual is NaN, so the 50-trip loop can only return NaN. The port now returns
that NaN without iterating; every finite level must iterate exactly as before.
"""
import numpy as np

from gpuwrf.physics.ntu import math as ntumath
from gpuwrf.physics.ntu.tables import selected_tables


# Verbatim pre-change FIND_RC0 port (git 9bc2d6b6a7:src/gpuwrf/physics/ntu/math.py).
from gpuwrf.physics.ntu.math import erf, dpdf, _loop  # noqa: E402


def _old_find_rc0(xafrc,cmode,cstdv,wmas,tblxa,tblrc,*,xp=np):
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
    active=~(low|high)
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
    value,y0,result,active=_counted_loop(0,50,iterate,(value,y0,result,active),xp=xp)
    result=xp.where(active,xp.exp(value),result)
    result=xp.where(low,r(1.e-9),xp.where(high,r(99),result))
    return xp.asarray(result,dtype=cmode.dtype)


_TRIPS = [0]


def _counted_loop(start, stop, body, state, *, xp):
    def b(i, s):
        _TRIPS[0] += 1
        return body(i, s)
    return _loop(start, stop, b, state, xp=xp)


def _reference(xafrc, *args):
    _TRIPS[0] = 0
    out = _old_find_rc0(xafrc, *args, xp=np)
    return out, _TRIPS[0]


def _run(xafrc, *args):
    trips = [0]
    loop = ntumath._loop

    def counted(start, stop, body, state, *, xp):
        def b(i, s):
            trips[0] += 1
            return body(i, s)
        return loop(start, stop, b, state, xp=xp)

    ntumath._loop = counted
    try:
        return ntumath.find_rc0(xafrc, *args, xp=np), trips[0]
    finally:
        ntumath._loop = loop


def test_nan_fractions_skip_newton_and_finite_levels_are_bitwise():
    c = selected_tables(np.float32, 2)
    rng = np.random.default_rng(5)
    xafrc = rng.uniform(0., 1., 44)
    xafrc[[0, 3, 7, 20, 21, 40]] = np.nan
    xafrc[[5]] = 0.
    xafrc[[9]] = 1.
    args = (c['cnmod'][:, 0].astype(np.float32), c['cnstd'][:, 0].astype(np.float32),
            c['wmas'][:, 0].astype(np.float32), c['tblxf'][:, 0], c['tblrc'])
    shape = lambda a: np.broadcast_to(a, (44,) + np.shape(a)) if np.ndim(a) == 1 and a.size == 3 else a
    args = tuple(shape(a) for a in args[:3]) + args[3:]
    new, trips = _run(xafrc, *args)
    old, old_trips = _reference(xafrc, *args)
    finite = ~np.isnan(xafrc)
    assert new.dtype == old.dtype == np.float32
    assert new[finite].tobytes() == old[finite].tobytes()
    assert np.isnan(new[~finite]).all() and np.isnan(old[~finite]).all()
    assert old_trips == 50 and trips < 50
    only_nan, nan_trips = _run(np.full(44, np.nan), *args)
    assert nan_trips <= 1 and np.isnan(only_nan).all()  # NumPy literal _loop is do-while (no-op pass)
