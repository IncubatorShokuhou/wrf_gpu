"""Literal WRF MY2 helper functions (REAL except gamma's explicit DOUBLE island)."""
import numpy as np
import jax.numpy as jnp
from .rounding import _op, _pow, _exp, _log, _log10
from .rounding import real

def _gamma(xx):
    R = np.asarray(xx).dtype.type
    x = np.float64(xx)
    y = x
    tmp = x + np.float64(5.5)
    tmp = (x + np.float64(0.5)) * np.log(tmp) - tmp
    ser = np.float64(1.000000000190015)
    for coeff in (76.18009172947146, -86.50532032941678, 24.01409824083091, -1.231739572450155):
        y = y + np.float64(1.0)
        ser = ser + np.float64(coeff) / y
    return R(np.exp(tmp + np.log(np.float64(2.5066282746310007) * ser / x)))

def _dim(a, b):
    return jnp.maximum(_op('sub', a, b), jnp.asarray(0.0, jnp.asarray(a).dtype))

def _dm_x(de, q, inv_n, inv_c, inv_d):
    return _exp(_op('mul', inv_d, _log(_op('mul', _op('mul', _op('mul', de, q), inv_n), inv_c))))
_ilamda_x = _dm_x

def _des_of_ds(d, maximum, eds, fds):
    return jnp.minimum(maximum, _op('mul', eds, _exp(_op('mul', fds, _log(d)))))

def _n_cooper(trpl, t):
    r = lambda x: real(x, t.dtype)
    return _op('mul', r(5.0), _exp(_op('mul', r(0.304), _op('sub', trpl, jnp.maximum(r(233.0), t)))))

def _nos_thompson(trpl, t):
    r = lambda x: real(x, t.dtype)
    return jnp.minimum(r(200000000.0), _op('mul', r(2000000.0), _exp(_op('mul', -r(0.12), jnp.minimum(-r(0.001), _op('sub', t, trpl))))))

def _polysvp(t, kind):
    r = lambda x: real(x, t.dtype)
    coeffs = ((6.11239921, 0.443987641, 0.0142986287, 0.00026484743, 3.02950461e-06, 2.06739458e-08, 6.40689451e-11, -9.52447341e-14, -9.76195544e-16), (6.11147274, 0.50316082, 0.0188439774, 0.000420895665, 6.15021634e-06, 6.02588177e-08, 3.85852041e-10, 1.46898966e-12, 2.52751365e-15))[kind]
    dt = jnp.maximum(r(-80.0), _op('sub', t, r(273.16)))
    result = r(coeffs[-1])
    for c in coeffs[-2::-1]:
        result = _op('add', r(c), _op('mul', dt, result))
    return _op('mul', result, r(100.0))

def _qsat(t, pres, kind):
    tmp = _polysvp(t, kind)
    return _op('div', _op('mul', real(0.622, t.dtype), tmp), _op('sub', pres, tmp))

def _nccnfnc(win, tin, pin, ccntype=2):
    if ccntype != 2:
        raise ValueError('WRF mp9 driver fixes CCNtype=2')
    r = lambda v: real(v, tin.dtype)
    x = _log10(_op('mul', win, r(100.0)))
    x2 = _op('mul', x, x)
    x3 = _op('mul', x2, x)
    x4 = _op('mul', x2, x2)
    t = _op('sub', tin, r(273.15))
    t2 = _op('mul', t, t)
    t3 = _op('mul', t2, t)
    t4 = _op('mul', t2, t2)
    p = _op('mul', pin, r(0.01))
    p2 = _op('mul', p, p)
    a = r(0.0)
    b = r(0.0)
    c = _op('sub', _op('sub', _op('add', _op('add', _op('mul', -r(2.112e-09), t4), _op('mul', r(3.9836e-08), t3)), _op('mul', r(2.3703e-06), t2)), _op('mul', r(0.00014542), t)), r(0.0698))
    d = _op('add', _op('add', _op('add', _op('add', _op('mul', -r(4.21e-08), t4), _op('mul', r(5.5745e-07), t3)), _op('mul', r(1.846e-05), t2)), _op('mul', r(0.00096078), t)), r(0.712))
    e = _op('add', _op('sub', _op('sub', _op('sub', _op('mul', r(1.434e-07), t4), _op('mul', r(1.6455e-06), t3)), _op('mul', r(4.3334e-05), t2)), _op('mul', r(0.007672), t)), r(1.0056))
    f = _op('add', _op('sub', _op('mul', r(1.34e-06), p2), _op('mul', r(0.0035114), p)), r(1.9453))
    g = _op('add', _op('add', _op('sub', _op('sub', _op('mul', r(0.004226), x4), _op('mul', r(0.0056012), x3)), _op('mul', r(0.087846), x2)), _op('mul', r(0.027435), x)), r(0.9932))
    h = _op('add', _op('add', _op('sub', _op('add', _op('mul', r(5.811e-09), t4), _op('mul', r(1.5589e-07), t3)), _op('mul', r(3.8623e-05), t2)), _op('mul', r(0.0014471), t)), r(0.1496))
    y = _op('add', _op('add', _op('add', _op('add', _op('add', _op('mul', a, x4), _op('mul', b, x3)), _op('mul', c, x2)), _op('mul', d, x)), e), _op('mul', _op('mul', f, g), h))
    return _op('mul', _pow(r(10.0), jnp.maximum(r(0.0), y)), r(1000000.0))

def _sxfnc(win, tin, pin, qsw, qsi, ccntype=2, wrt=1):
    if ccntype != 2:
        raise ValueError('WRF mp9 driver fixes CCNtype=2')
    r = lambda v: real(v, tin.dtype)
    x = _log10(_op('mul', jnp.maximum(win, r(1e-20)), r(100.0)))
    x2 = _op('mul', x, x)
    x3 = _op('mul', x2, x)
    x4 = _op('mul', x2, x2)
    t = tin
    t2 = _op('mul', t, t)
    t3 = _op('mul', t2, t)
    p = _op('mul', pin, r(0.01))
    p2 = _op('mul', p, p)
    a = _op('add', _op('add', _op('mul', r(3.8e-05), t2), _op('mul', r(0.000165), t)), r(0.0988))
    b = _op('sub', _op('sub', _op('mul', -r(7.38e-05), t2), _op('mul', r(0.00253), t)), r(0.323))
    c = _op('add', _op('add', _op('mul', r(8.39e-05), t2), _op('mul', r(0.00396), t)), r(0.35))
    d = _op('sub', _op('sub', _op('mul', -r(1.88e-06), t2), _op('mul', r(0.00133), t)), r(0.0373))
    f = _op('sub', _op('add', _op('mul', -r(1.9761e-06), p2), _op('mul', r(0.0041473), p)), r(1.771))
    g = _op('sub', _op('sub', _op('add', _op('sub', _op('mul', r(0.1539), x4), _op('mul', r(0.5575), x3)), _op('mul', r(0.9262), x2)), _op('mul', r(0.3498), x)), r(0.1293))
    h = _op('add', _op('sub', _op('add', _op('add', _op('mul', -r(8.035e-09), _op('mul', t2, t2)), _op('mul', r(3.162e-07), t3)), _op('mul', r(1.029e-05), t2)), _op('mul', r(0.0005931), t)), r(0.0562))
    pcorr = _op('mul', _op('mul', f, g), h)
    t2corr = _op('sub', _op('sub', _op('sub', r(0.98888), _op('mul', r(0.00050525), t)), _op('mul', r(1.7598e-05), t2)), _op('mul', r(8.3308e-08), t3))
    sw = _op('add', _op('add', _op('add', _op('add', _op('mul', a, x3), _op('mul', b, x2)), _op('mul', c, x)), d), pcorr)
    sw = _op('add', r(1.0), _op('mul', r(0.01), sw))
    qv = _op('mul', qsw, sw)
    si = _op('div', qv, qsi)
    si = _op('mul', si, t2corr)
    return jnp.where(win <= r(0.0), r(1.0), sw if wrt == 1 else si)
