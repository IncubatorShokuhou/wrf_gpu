"""Literal P3 operations used by the generated level-local process block."""
from functools import wraps
from contextvars import ContextVar

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma as _jax_gamma

from . import lookup as lut
from .properties import bulk_rime_density, compute_mu_3moment

RATE_NAMES = set("qccol qwgrth qidep qrcol qinuc nccol nrcol ninuc qisub qimlt nimlt nisub nislf qchetc qcheti qrhetc qrheti nchetc ncheti nrhetc nrheti nrshdr qcshd qrmul nimul ncshdc rhorime_c eii_fact epsi dumm0 dumm3 log_wetgrowth".split())
CAT_NAMES = set("qitot qirim nitot birim zitot diam_ice rimefraction rimevolume diag_effi diag_vmi diag_di diag_rhoi".split())
INT_NAMES = set("dumi dumj dumii dumjj dumzz dumic dumiic dumjjc iice_dest tmpint1".split())
NATIVE_REAL = ContextVar('p3_native_real', default=False)
MP53_PREDICATE_RESHAPE = ContextVar('p3_mp53_predicate_reshape', default=False)
GAMMA_FACTORIALS = ContextVar('p3_gamma_factorials', default=None)
DERF_TABLES = ContextVar('p3_derf_tables', default=None)


def _rounded_real(operation, *values):
    """Preserve WRF's REAL operator/function-result rounding in native code."""
    if NATIVE_REAL.get():
        from jax.experimental.pallas import triton as plt
        values = jnp.broadcast_arrays(*values)
        bits = 64 if values[0].dtype == jnp.float64 else 32
        register = 'd' if bits == 64 else 'f'
        asm = f'mov.b{bits} $0, $1;' if operation == 'identity' else f'{operation}.rn.f{bits} $0, $1, $2;'
        return plt.elementwise_inline_asm(
            asm, args=values, constraints=f'={register},'+','.join(register for _ in values), pack=1,
            result_shape_dtypes=(jax.ShapeDtypeStruct(values[0].shape, values[0].dtype),))[0]
    if operation == 'identity':
        result = values[0]
    elif operation == 'mul':
        result = values[0]*values[1]
    elif operation == 'add':
        result = values[0]+values[1]
    elif operation == 'sub':
        result = values[0]-values[1]
    elif operation == 'div':
        result = values[0]/values[1]
    else:
        raise ValueError(operation)
    if operation == 'mul':
        # A reversible IEEE neighbor round-trip makes the product observable
        # to LLVM's integer nextafter lowering. Unlike an HLO-only barrier it
        # prevents contraction with the following add on FMA-capable CPUs.
        # Zero/max-finite/Inf/NaN retain their original values and bit signs.
        up = jnp.nextafter(result, jnp.full_like(result, jnp.inf))
        back = jnp.nextafter(up, jnp.full_like(result, -jnp.inf))
        result = jnp.where((result != 0) & jnp.isfinite(up), back, result)
    return jax.lax.optimization_barrier(result)


def _any(value, axis=None):
    # Installed Triton supports integer max rather than reduce_or.
    return jnp.max(value.astype(jnp.int32), axis=axis) != 0


def _real_binary(operation, left, right):
    # These wrappers are generated only for the REAL source/sink updates,
    # where a contracted cancellation changes small-mass cleanup/freezing.
    return _rounded_real(operation, left, right)


def _r(x):
    return jnp.asarray(x, dtype=jnp.float32)


def _d(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _int(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _with_double(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        # Only DBLE/DEXP and declared tmpdbl values become DOUBLE: all REAL
        # literals/operands below explicitly retain REAL4 dtype.
        with jax.enable_x64(True):
            return fn(*args, **kwargs)
    return call


def _initial(field, name, ncat):
    extra = (ncat, ncat) if name in {"qicol", "nicol"} else ((ncat,) if name in RATE_NAMES | CAT_NAMES else ())
    dtype = jnp.bool_ if name.startswith("log_") else jnp.int32 if name in INT_NAMES else jnp.float64 if name.startswith("tmpdbl") else jnp.float32
    return jnp.zeros(field.shape + extra, dtype=dtype)


def _get(value, indices):
    if NATIVE_REAL.get():
        # Triton's register tensors have no general slice primitive. With
        # only one/two categories, select/reduce is the same finite value.
        for index in reversed(indices):
            pick = jnp.arange(value.shape[-1], dtype=jnp.int32) == _int(index)[..., None]
            value = jnp.sum(jnp.where(pick, value, jnp.zeros_like(value)), axis=-1)
        return value
    if all(isinstance(i, int) for i in indices):
        return value[(...,) + indices]
    assert len(indices) == 1
    return jnp.take_along_axis(value, _int(indices[0])[..., None], axis=-1)[..., 0]


def _expand(value, dimensions):
    value = _int(value) if isinstance(value, int) else jnp.asarray(value)
    if MP53_PREDICATE_RESHAPE.get() and value.dtype == jnp.bool_:
        # A singleton reshape preserves every predicate bit. Triton 3.6's
        # layout-removal pass can move expand_dims of a reused predicate
        # below its earlier users, producing invalid SSA for MP53.
        return value.reshape(value.shape + (1,) * dimensions)
    return value[(...,) + (None,) * dimensions]


def _put(v, name, indices, value, mask):
    old = v[name]
    value = jnp.asarray(value, dtype=old.dtype)
    if NATIVE_REAL.get() and value.ndim == 0:
        shape = old.shape if indices is None else old.shape[:-len(indices)]
        value = jnp.broadcast_to(value, shape)
    if jnp.issubdtype(old.dtype, jnp.floating):
        value = _rounded_real('identity', value)
    if indices is None:
        while mask.ndim < old.ndim:
            mask = _expand(mask, 1)
        v[name] = jnp.where(mask, value, old)
    elif all(isinstance(i, int) for i in indices):
        part = _get(old, indices)
        select = jnp.ones(old.shape[-len(indices):], dtype=jnp.bool_)
        for axis, index in enumerate(indices):
            shape = [1] * len(indices)
            shape[axis] = old.shape[old.ndim-len(indices)+axis]
            select = select & (jnp.arange(shape[axis]).reshape(shape) == index)
        dimensions = (None,) * len(indices)
        v[name] = jnp.where(select, jnp.where(mask, value, part)[(...,)+dimensions], old)
    else:
        assert len(indices) == 1
        select = jnp.arange(old.shape[-1], dtype=jnp.int32) == _int(indices[0])[..., None]
        v[name] = jnp.where(_expand(mask, 1) & select, value[..., None], old)


def _max(*args):
    result = args[0]
    for value in args[1:]:
        result = jnp.maximum(result, value)
    return result


def _min(*args):
    result = args[0]
    for value in args[1:]:
        result = jnp.minimum(result, value)
    return result


def _sum(x):
    return jnp.sum(x, axis=-1)


def gamma(x):
    """REAL gamma on P3's positive 1..22 range, without log/exp loss.

    Cody/Stoltz rational coefficients also used by pristine WRF Morrison's
    GAMMA. Range reduction is REAL4 throughout; integer arguments use their
    correctly rounded factorial. No extra DOUBLE island is introduced.
    """
    import math
    p = (-1.7161851388654949, 24.76565080557592, -379.80425647094564,
         629.3311553128184, 866.9662027904132, -31451.272968848367,
         -36144.41341869117, 66456.14382024054)
    q = (-30.8402300119739, 315.35062697960416, -1015.1563674902191,
         -3107.771671572311, 22538.11842098015, 4755.846277527881,
         -134659.9598649693, -115132.25967555348)
    x = _r(x)
    n = _int(jnp.floor(x)) - 1
    y = x - _r(n)
    z = y - _r(1)
    numerator, denominator = jnp.zeros_like(x), jnp.ones_like(x)
    for a,b in zip(p,q):
        numerator = (numerator + _r(a)) * z
        denominator = denominator * z + _r(b)
    result = numerator / denominator + _r(1)
    for k in range(22):
        result = jnp.where(k < n, result * (y + _r(k)), result)
    integers = GAMMA_FACTORIALS.get()
    if integers is None:
        integers = _r(tuple(float(math.factorial(k)) for k in range(35)))
    exact_integer = (x == jnp.floor(x)) & (x >= 1) & (x <= 35)
    integer = integers[jnp.clip(_int(x) - 1, 0, 34)]
    return jnp.where(exact_integer, integer, result)


def polysvp1(t, ice):
    t = _r(t)
    wi = (ice == 1) & (t < _r(273.15))
    dt = t - _r(273.15)
    coeff = (6.11147274, .503160820, .188439774e-1, .420895665e-3,
             .615021634e-5, .602588177e-7, .385852041e-9, .146898966e-11, .252751365e-14)
    water = (6.11239921, .443987641, .142986287e-1, .264847430e-3,
             .302950461e-5, .206739458e-7, .640689451e-10, -.952447341e-13, -.976195544e-15)
    def horner(c):
        # WRF's REAL polynomial is cancellation-sensitive near 200 K.
        # Contracting these pairs into FMAs changes qs by ~0.15% and flips
        # Cooper nucleation at the 5% ice-supersaturation threshold.
        result = _rounded_real('add', _rounded_real('mul', _r(c[-1]), dt), _r(c[-2]))
        for a in reversed(c[:-2]):
            result = _rounded_real('add', _r(a), _rounded_real('mul', dt, result))
        return _rounded_real('mul', result, _r(100))
    pi = _r(10) ** (_r(-9.09718) * (_r(273.16) / t - 1) - _r(3.56654) * jnp.log10(_r(273.16) / t)
                     + _r(.876793) * (1 - t / _r(273.16)) + jnp.log10(_r(6.1071))) * _r(100)
    pw = _r(10) ** (_r(-7.90298) * (_r(373.16) / t - 1) + _r(5.02808) * jnp.log10(_r(373.16) / t)
                     - _r(1.3816e-7) * (_r(10) ** (_r(11.344) * (1 - t / _r(373.16))) - 1)
                     + _r(8.1328e-3) * (_r(10) ** (_r(-3.49149) * (_r(373.16) / t - 1)) - 1)
                     + jnp.log10(_r(1013.246))) * _r(100)
    return jnp.where(wi, jnp.where(t >= _r(195.8), horner(coeff), pi),
                     jnp.where(t >= _r(202), horner(water), pw))


def qv_sat(t, p, ice):
    e = polysvp1(t, ice)
    # qv_sat returns a REAL value before the caller divides by it. Without
    # this boundary XLA rewrites qv/(a/b) as (qv*b)/a, removing the WRF
    # function-result rounding and flipping Cooper's 5% activation test.
    return _rounded_real('identity', _r(.622) * e / jnp.maximum(_r(1.e-3), p - e))


def g_of_mu(mu):
    return ((_r(6) + mu) * (_r(5) + mu) * (_r(4) + mu)) / ((_r(3) + mu) * (_r(2) + mu) * (_r(1) + mu))


def derf(x):
    from .constants import DERF_A, DERF_B
    w = jnp.abs(x)
    t = w * w
    k = _int(t)
    ta = t - _r(k)
    ka = jnp.clip(k, 0, 4) * 13
    kb = 13 * (jnp.clip(_int(w), 2, 6) - 2)
    tb = w - _r(_int(w))
    coefficients = DERF_TABLES.get()
    a, b = (_r(DERF_A), _r(DERF_B)) if coefficients is None else coefficients
    ya, yb = a[ka], b[kb]
    for i in range(1, 13):
        ya = ya * ta + a[ka + i]
        yb = yb * tb + b[kb + i]
    ya = ya * w
    yb = yb * yb; yb = yb * yb; yb = yb * yb; yb = 1 - yb * yb
    y = jnp.where(w < _r(2.2), ya, jnp.where(w < _r(6.9), yb, _r(1)))
    return jnp.where(x < 0, -y, y)


def _cloud(qc, nc, rho, invfrac):
    from .constants import CONSTANTS as c
    active = qc * invfrac >= _r(1.e-14)
    q = jnp.where(active, qc * invfrac, _r(1))
    n = jnp.maximum(nc * invfrac, _r(1.e-16))
    # module_mp_p3.F:6659-6660, assignment and operator REAL rounding.
    # A compiled FMA here changes mu/lambda, then the stiff Bigg sink limiter
    # amplifies its one-ULP residual into cloud/ice number differences (P345).
    mu = _rounded_real('mul', n, _r(1.e-6))
    mu = _rounded_real('mul', mu, rho)
    mu = _rounded_real('mul', _r(.0005714), mu)
    mu = _rounded_real('add', mu, _r(.2714))
    mu = _rounded_real('mul', mu, mu)
    mu = _rounded_real('sub', _rounded_real('div', _r(1), mu), _r(1))
    mu = jnp.clip(mu, _r(2), _r(15))
    lam = (_r(c['cons1']) * n * (mu + 3) * (mu + 2) * (mu + 1) / q) ** _r(c['thrd'])
    lo, hi = (mu + 1) * _r(2.5e4), (mu + 1) * _r(1.e6)
    limited = (lam < lo) | (lam > hi)
    lam = jnp.clip(lam, lo, hi)
    nn = _r(6) * lam ** 3 * q / (_r(c['pi']) * _r(1000) * (mu + 3) * (mu + 2) * (mu + 1))
    n = jnp.where(limited, nn, n)
    return (jnp.where(active, n / invfrac, nc), jnp.where(active, mu, _r(0)), _r(0),
            jnp.where(active, lam, _r(0)), lo, hi,
            jnp.where(active, n * (mu + 1) / lam, _r(0)), jnp.where(active, n / gamma(mu + 1), _r(0)))


def _rain(qr, nr, invfrac):
    from .constants import CONSTANTS as c
    active = qr * invfrac >= _r(1.e-14)
    q = jnp.where(active, qr * invfrac, _r(1))
    n = jnp.maximum(nr * invfrac, _r(1.e-16))
    mu = _r(0)
    lam = (_r(c['cons1']) * n * (mu + 3) * (mu + 2) * (mu + 1) / q) ** _r(c['thrd'])
    lo, hi = (mu + 1) * _r(c['inv_drmax']), (mu + 1) * _r(1.e5)
    limited = (lam < lo) | (lam > hi)
    lam = jnp.clip(lam, lo, hi)
    nn = jnp.exp(_r(3) * jnp.log(lam) + jnp.log(q) + jnp.log(gamma(mu + 1)) - jnp.log(gamma(mu + 4))) / _r(c['cons1'])
    n = jnp.where(limited, nn, n)
    return (jnp.where(active, n / invfrac, nr), mu, jnp.where(active, lam, _r(0)),
            jnp.where(active, n / gamma(mu + 1), _r(0)),
            jnp.where(active, jnp.log10(n) + (mu + 1) * jnp.log10(lam) - jnp.log10(gamma(mu + 1)), _r(0)))


def _call_get_cloud_dsd2(q, n, mu, rho, nu, dnu, lam, lo, hi, cd, cd1, frac, **_):
    return _cloud(q, n, rho, frac)


def _call_get_rain_dsd2(q, n, mu, lam, cd, logn, frac, **_):
    return _rain(q, n, frac)


def _call_calc_bulkrhorime(q, r, b, rho, **_):
    return bulk_rime_density(q, r, b)


def _call_impose_max_total_ni(n, cap, invrho, **_):
    total = _sum(n)
    ratio = cap * invrho / jnp.where(total >= _r(1.e-20), total, _r(1))
    return (jnp.where(_expand(total >= _r(1.e-20), 1), n * jnp.minimum(ratio, 1)[..., None], n),)


def _call_find_lookuptable_indices_1a(i, jj, ii, a, b, c, size, rims, dens, q, n, r, rho, **_):
    x = lut.ice_coordinates(q, n, r, rho)
    return x.mass_index, x.density_index, x.rime_index, x.mass, x.rime, x.density


def _call_find_lookuptable_indices_1b(j, a, size, q, n, **_):
    x = lut.rain_coordinates(q, n)
    return x.index, x.value


def _call_find_lookuptable_indices_1c(i, a, size, q, mu, **_):
    x = lut.mu_coordinates(q, mu)
    return x.index, x.value


def _call_find_lookuptable_indices_2(i, ii, jj, ic, iic, jjc, a, b, c, ac, bc, cc,
                                  size, rims, dens, q1, q2, n1, n2, r1, r2, v1, v2, **_):
    x, y = lut.ice_pair_coordinates(q1, n1, r1, v1, q2, n2, r2, v2)
    return (*x[:3], *y[:3], *x[3:], *y[3:])


def _call_find_lookuptable_indices_3(i, j, a, ai, aj, inv, mu, lam, **_):
    from .constants import CONSTANTS as c
    size = (mu + 1) / lam
    small = size <= _r(195.e-6)
    inv = jnp.where(small, _r(.1), _r(c['thrd']) * _r(.1))
    ai = jnp.where(small, (size * _r(1.e6) + 5) * inv, (size * _r(1.e6) - 195) * inv + 20)
    ai = jnp.where(small, jnp.clip(ai, _r(1), _r(20)), jnp.clip(ai, _r(20), _r(300)))
    i = jnp.where(small, jnp.clip(_int(ai), 1, 20), jnp.clip(_int(ai), 20, 299))
    aj = jnp.clip(mu + 1, _r(1), _r(10))
    j = jnp.clip(_int(aj), 1, 9)
    return i, j, size, ai, aj, inv


def _coords(jj, ii, i, a, b, c):
    return lut.IceCoordinates(_int(i), _int(ii), _int(jj), a, b, c)


def _call_access_lookup_table(jj, ii, i, channel, a, b, c, out, *, tables, **_):
    return (lut.ice_lookup(tables['ice'], _coords(jj, ii, i, a, b, c), int(channel)),)


def _call_access_lookup_table_3mom(z, jj, ii, i, channel, a, b, c, d, out, *, tables, **_):
    return (lut.ice_lookup(tables['ice'], _coords(jj, ii, i, a, b, c), int(channel), lut.Coordinate(_int(z), d)),)


def _call_access_lookup_table_coll(jj, ii, j, i, channel, a, d, b, c, out, *, tables, **_):
    return (lut.rain_ice_lookup(tables['rain'], _coords(jj, ii, i, a, b, c), lut.Coordinate(_int(j), d), int(channel)),)


def _call_access_lookup_table_coll_3mom(z, jj, ii, j, i, channel, a, d, b, c, e, out, *, tables, **_):
    return (lut.rain_ice_lookup(tables['rain'], _coords(jj, ii, i, a, b, c), lut.Coordinate(_int(j), d),
                               int(channel), lut.Coordinate(_int(z), e)),)


def _call_access_lookup_table_colli(jjc, iic, ic, jj, ii, i, channel, ac, bc, cc, a, b, c, out, *, tables, **_):
    return (lut.ice_ice_lookup(tables['number' if channel == 1 else 'mass'], _coords(jj, ii, i, a, b, c), _coords(jjc, iic, ic, ac, bc, cc)),)


def _call_icecat_destination(q, d, new, delta, out, **_):
    active = q >= _r(1.e-14)
    diff = jnp.where(active, jnp.abs(d - new[..., None]), _r(9.e9))
    if NATIVE_REAL.get():
        # The surrounding WRF DOUBLE islands enable x64; Triton requires the
        # reduction's index itself to be INTEGER4, not a later conversion.
        nearest = jax.lax.argmin(diff, diff.ndim-1, jnp.int32) + 1
        empty = jax.lax.argmax((~active).astype(jnp.int32), active.ndim-1, jnp.int32) + 1
    else:
        nearest = jnp.argmin(diff, axis=-1).astype(jnp.int32) + 1
        empty = jnp.argmax(~active, axis=-1).astype(jnp.int32) + 1
    all_active = jnp.min(active.astype(jnp.int32), axis=-1) != 0
    dest = jnp.where(all_active | (jnp.min(diff, axis=-1) < delta), nearest, empty)
    return (jnp.where(_sum(q) < _r(1.e-14), 1, dest),)


__all__ = [name for name in globals() if not name.startswith('__')]
