"""WRF radiation_driver cal_cldfra1 (icloud=1), WRF REAL arithmetic."""
from __future__ import annotations

import jax.numpy as jnp


def wrf_resolved_cloud_fraction(qv, qc, qi, qs, temperature, pressure, *, mp_physics=8,return_flags=False):
    """Resolved clouds for standard QC/QI/QS schemes and QC-only WRF options.

    Literal module_radiation_driver.F:3886–4111. Ferrier needs its separate
    F_ICE_PHY partition and is not a caller of this standard-species route.
    """
    f = jnp.float32
    qv, qc, qi, qs, t, p = (jnp.asarray(a, f) for a in (qv, qc, qi, qs, temperature, pressure))
    if int(mp_physics) == 0:
        zero=jnp.zeros_like(qv)
        return (zero,jnp.ones_like(qv,dtype=jnp.int32)) if return_flags else zero
    if int(mp_physics) in (5, 85, 95):
        raise NotImplementedError("Ferrier cal_cldfra1 requires F_ICE_PHY")
    tc = t - f(273.15)
    esw = f(1000.0) * f(0.61078) * jnp.exp(f(17.2693882) * tc / (t - f(35.86)))
    esi = f(1000.0) * f(0.61078) * jnp.exp(f(21.8745584) * tc / (t - f(7.66)))
    ep2 = f(287.0) / f(461.6)
    qvsw = ep2 * esw / (p - esw)
    qvsi = ep2 * esi / (p - esi)
    if int(mp_physics) in (1, 3):
        qcld = qc
        weight = jnp.where(t > f(273.15), f(0.0), f(1.0))
    else:
        ice = qi if int(mp_physics) in (50, 51) else qi + qs
        qcld = qi + qc if int(mp_physics) in (50, 51) else qi + qc + qs
        weight = ice / jnp.where(qcld < f(1.e-12), f(1.0), qcld)
    clear = qcld < f(1.e-12)
    weight = jnp.where(clear, f(0.0), weight)
    qvs = (f(1.0) - weight) * qvsw + weight * qvsi
    rh = qv / qvs
    subsat = jnp.maximum(f(1.e-10), f(1.0) * qvs - qv)
    denom = subsat ** f(0.49)
    arg = jnp.maximum(f(-6.9), -f(100.0) * qcld / denom)
    fraction = (jnp.maximum(f(1.e-10), rh) / f(1.0)) ** f(0.25) * (f(1.0) - jnp.exp(arg))
    fraction = jnp.where(fraction < f(0.01), f(0.0), fraction)
    fraction=jnp.where(clear, f(0.0), jnp.where(rh >= f(1.0), f(1.0), fraction))
    if return_flags:
        flag=jnp.where(clear,jnp.int32(1),jnp.where(rh>=f(1.),jnp.int32(2),jnp.int32(3)))
        return fraction,flag
    return fraction


def wrf_cal_cldfra2(qc, qi, *, has_qi=True):
    """WRF radiation_driver cal_cldfra2 (icloud=2), module_radiation_driver.F:3801–3874."""
    f = jnp.float32
    qc = jnp.asarray(qc, f)
    cond = (qc + jnp.asarray(qi, f) if has_qi else qc) > f(1.0e-6)
    return jnp.where(cond, f(1.0), f(0.0))


# Literal-arithmetic proofs swap in the host libm's log10f (glibc 2.43: correctly rounded);
# XLA's log10 differs from it by <= 1 ulp. Read at call time.
_LOG10 = jnp.log10


def _rslf(p, t):
    """module_mp_thompson.F:5383 RSLF, REAL."""
    f = jnp.float32
    c = [f(v) for v in (.611583699E03, .444606896E02, .143177157E01, .264224321E-1, .299291081E-3,
                        .203154182E-5, .702620698E-8, .379534310E-11, -.321582393E-13)]
    x = jnp.maximum(f(-80.), t - f(273.16))
    esl = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * (c[7] + x * c[8])))))))
    esl = jnp.minimum(esl, p * f(0.15))
    return f(.622) * esl / (p - esl)


def _rsif(p, t):
    """module_mp_thompson.F:5418 RSIF, REAL."""
    f = jnp.float32
    c = [f(v) for v in (.609868993E03, .499320233E02, .184672631E01, .402737184E-1, .565392987E-3,
                        .521693933E-5, .307839583E-7, .105785160E-9, .161444444E-12)]
    x = jnp.maximum(f(-80.), t - f(273.16))
    esi = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * (c[7] + x * c[8])))))))
    esi = jnp.minimum(esi, p * f(0.15))
    return f(.622) * esi / jnp.maximum(f(1.e-4), p - esi)


def wrf_gridkm(dx, dy):
    """radiation_driver.F:1053–1058 gridkm without dxkm (first_rk_step_part1 passes dx, dy)."""
    import numpy as np
    dx, dy = np.float32(dx), np.float32(dy)
    km = np.float32(0.001)
    return np.sqrt(dx * km * dx * km + dy * km * dy * km, dtype=np.float32)


def _scan_levels(fn, init, *xs):
    """Ascending sequential loop over the trailing level axis (WRF DO k=kts,kte order)."""
    import jax
    moved = tuple(jnp.moveaxis(x, -1, 0) for x in xs)
    carry, ys = jax.lax.scan(fn, init, moved)
    return carry, jax.tree.map(lambda y: jnp.moveaxis(y, 0, -1), ys)


def _layer_runs(cloudy, level, lo, hi, min_top, nz):
    """Maximal runs of ``cloudy`` inside [lo, hi] visited by find_cloudLayers' DO WHILE scan.

    The downward scan starts at ``hi`` and stops above ``min_top - 1``; each run's
    bottom is clipped at ``lo`` (the k2 loop's ``k2 == k_m12C``/``kbot`` exit).
    Returns per-level run bottom/top (1-based) and membership of a visited run.
    """
    import jax
    inside = cloudy & (level >= lo[..., None]) & (level <= hi[..., None])
    gap = jnp.where(inside, jnp.int32(0), level)
    bottom = jax.lax.cummax(gap, axis=gap.ndim - 1) + 1
    above = jnp.where(inside, jnp.int32(nz + 1), level)
    top = jax.lax.cummin(above, axis=above.ndim - 1, reverse=True)
    # first non-cloudy level strictly above k, minus one
    top = jnp.concatenate([top[..., 1:], jnp.full(top[..., :1].shape, nz + 1, jnp.int32)], axis=-1) - 1
    member = inside & (top >= min_top[..., None])
    return bottom, top, member


def wrf_cal_cldfra3(qv, qc, qi, qs, dz, pressure, temperature, xland, gridkm, *, return_diagnostics=False):
    """WRF 4.7.1 radiation_driver icloud=3 (G. Thompson) cloud fraction, WRF REAL arithmetic.

    Literal module_radiation_driver.F:4128–4594 (cal_cldfra3, find_cloudLayers,
    adjust_cloudIce/H2O/Final) as called at :1470–1500: modify_qvapor=.false.,
    max_relh=1.5, entrmnt=0.5, gridkm from dx/dy (:1053–1058). Columns are
    ``(..., nz)`` bottom→top, ``xland`` is ``(...)``. Returns the cloud fraction
    and the radiation-only qc/qi (WRF restores model qc/qi after radiation,
    :3262–3280; qs is INTENT(IN)). The fatal k_m12C > kte-3 profile check
    (:4307) cannot raise inside a compiled step and is not emulated.
    """
    import numpy as np
    f = jnp.float32
    qv, qc, qi, qs, dz, p, t = (jnp.asarray(a, f) for a in (qv, qc, qi, qs, dz, pressure, temperature))
    xland = jnp.asarray(xland, f)
    gridkm = f(gridkm)
    nz = qv.shape[-1]
    level = jnp.arange(1, nz + 1, dtype=jnp.int32)
    # cal_cldfra3 first loop (:4157–4183)
    qvsw = _rslf(p, t)
    qvsi = _rsif(p, t)
    tc = t - f(273.15)
    qvs = jnp.where(tc >= f(-12.0), qvsw, jnp.where(
        tc < f(-35.0), qvsi, qvsw - (qvsw - qvsi) * (f(-12.0) - tc) / f(23.0)))
    rh = jnp.maximum(f(0.01), qv / qvs)
    rhoa = p / (f(287.0) * t)
    # second loop (:4192–4245)
    delz = jnp.maximum(f(100.), dz)
    root = jnp.sqrt(f(1.) / (f(50.0) + gridkm * gridkm * delz * f(0.01)))
    rh00l = f(0.77) + jnp.minimum(f(0.22), root)
    rh00o = f(0.85) + jnp.minimum(f(0.14), root)
    ocean = (xland - f(1.5) > f(0.))[..., None]
    tcm = jnp.maximum(f(-80.0), t - f(273.15))
    rh00 = jnp.where(tcm < f(-12.0), rh00l, jnp.where(ocean, rh00o, rh00l))
    warm = f(1.0) - jnp.sqrt((f(1.001) - jnp.minimum(rh, f(1.0))) / (f(1.001) - rh00))
    cold00 = rh00 + (f(1.45) - rh00) * (f(-12.0) - tcm) / f(73.0)
    cold = f(1.0) - jnp.sqrt((f(1.46) - jnp.minimum(rh, f(1.45))) / (f(1.46) - cold00))
    rh_cf = jnp.where(tcm >= f(25.0), f(0.0), jnp.maximum(f(0.), jnp.where(tcm >= f(-12.0), warm, cold)))
    rh_cf = jnp.where(rh_cf > f(0.), jnp.maximum(f(0.01), jnp.minimum(rh_cf, f(0.99))), rh_cf)
    qcqi = qc + qi
    small = (qcqi > f(1.e-10)) & (qcqi < f(1.e-6))
    small_cf = jnp.minimum(f(0.99), f(0.1) * (f(11.0) + _LOG10(jnp.where(small, qcqi, f(1.e-8)))))
    full = (qc > f(1.e-6)) | (qi >= f(1.e-7)) | ((qs > f(1.e-6)) & (t < f(273.)))
    cfr = jnp.where(full, f(1.0), jnp.where(small, small_cf, rh_cf))
    cfr = jnp.where((cfr > f(0.0)) & (p < f(7000.0)), f(0.0), cfr)

    # find_cloudLayers (:4282–4466)
    expo = np.float32(287.05) / np.float32(1004.)
    theta = t * ((f(100000.0) / p) ** f(expo))
    zero = jnp.int32(0)
    m12 = (t - f(273.16) > f(-12.0)) & (p > f(10100.0))
    k_m12c = jnp.max(jnp.where(m12, level, zero), axis=-1)
    k_m12c = jnp.where(k_m12c <= 1, jnp.int32(1), k_m12c)
    k_p150 = jnp.max(jnp.where(p > f(14999.0), level, zero), axis=-1)
    k_p150 = jnp.where(nz - k_p150 < 3, jnp.int32(nz - 3), k_p150)

    def _shift(a, n):
        return jnp.concatenate([a[..., n:], jnp.zeros(a[..., :n].shape, a.dtype)], axis=-1)

    delz3 = f(0.5) * dz + _shift(dz, 1) + f(0.5) * _shift(dz, 2)
    slope = (_shift(theta, 2) - theta) / delz3
    flat = slope < f(10.) / f(1500.)
    # DO k = k_p150-2, kts, -1 ... EXIT
    hit1 = (flat | (p > f(70000.))) & (level <= (k_p150 - 2)[..., None])
    found1 = jnp.max(jnp.where(hit1, level, zero), axis=-1)
    k1 = jnp.where(jnp.any(hit1, axis=-1), found1, jnp.minimum(k_p150 - 2, 0))
    k_tropo = jnp.maximum(3, jnp.minimum(k1 + 2, nz - 1))
    # DO k = kte-3, k_p150-2, -1 ... EXIT (only when k_tropo > k_p150)
    hit2 = flat & (p > f(9500.)) & (level <= nz - 3) & (level >= (k_p150 - 2)[..., None])
    found2 = jnp.max(jnp.where(hit2, level, zero), axis=-1)
    trips2 = jnp.maximum(0, (nz - 3) - (k_p150 - 2) + 1)
    k2 = jnp.where(jnp.any(hit2, axis=-1), found2, (nz - 3) - trips2)
    k_tropo = jnp.where(k_tropo > k_p150, jnp.maximum(k_p150 - 1, jnp.minimum(k2 + 2, nz - 1)), k_tropo)
    fractional = (cfr > f(0.0)) & (cfr < f(1.0))
    cfr = jnp.where(fractional & (level > k_tropo[..., None]), f(0.), cfr)
    # well-mixed layer below LCL: DO k = kbot, k_m12C ... EXIT
    stable = (theta - jnp.concatenate([theta[..., :1], theta[..., :-1]], axis=-1)) > f(0.010E-3) * dz
    hit3 = stable & (level >= 2) & (level <= k_m12c[..., None])
    found3 = jnp.min(jnp.where(hit3, level, jnp.int32(nz + 1)), axis=-1)
    k3 = jnp.where(jnp.any(hit3, axis=-1), found3, jnp.maximum(k_m12c + 1, 2))
    kbot = jnp.maximum(2, k3 - 2)
    fractional = (cfr > f(0.0)) & (cfr < f(1.0))
    cfr = jnp.where(fractional & (level <= kbot[..., None]), jnp.maximum(f(0.01), f(0.5) * cfr), cfr)
    lowest = jnp.min(jnp.where((cfr > f(0.0)) & (level <= k_tropo[..., None]), level, jnp.int32(nz + 1)), axis=-1)
    kbot = jnp.minimum(kbot, lowest)

    cloudy = cfr >= f(0.01)
    ib, it, imember = _layer_runs(cloudy, level, k_m12c + 1, k_tropo, k_m12c + 2, nz)
    wb, wt, wmember = _layer_runs(cloudy, level, kbot + 1, k_m12c + 2, kbot + 1, nz)

    def _gather(a, idx):
        return jnp.take_along_axis(a, jnp.clip(idx - 1, 0, nz - 1), axis=-1)

    ice0 = jnp.maximum(f(0.0), _gather(qvs, ib) - _gather(qvs, it))
    h2o0 = jnp.maximum(f(0.0), _gather(qvs, wb) - _gather(qvs, wt))
    is_ib, is_wb = level == ib, level == wb
    half_dz = f(0.5) * dz

    def body(carry, x):
        itdz, imax, idz, wtdz, wmax, wdz = carry
        dzk, hdzk, qik, qsk, qck, i0, w0, ibk, wbk = x
        itdz = jnp.where(ibk, f(0.), itdz) + dzk
        imax = jnp.maximum(f(1.E-6), jnp.where(ibk, i0, imax) - (qik + qsk))
        idz = jnp.where(ibk, f(0.0) + hdzk, idz + dzk)
        wtdz = jnp.where(wbk, f(0.), wtdz) + dzk
        wmax = jnp.maximum(f(1.E-6), jnp.where(wbk, w0, wmax) - qck)
        wdz = jnp.where(wbk, f(0.0) + hdzk, wdz + dzk)
        out = (itdz, imax, idz, wtdz, wmax, wdz)
        return out, out

    shape = qv.shape[:-1]
    init = tuple(jnp.zeros(shape, f) for _ in range(6))
    _, (itdz, imax, idz, wtdz, wmax, wdz) = _scan_levels(
        body, init, dz, half_dz, qi, qs, qc, ice0, h2o0, is_ib, is_wb)
    partial = (cfr > f(0.0)) & (cfr < f(1.0))
    ent = f(1.) - f(0.5)

    def _add(member, b, top, tdz, mx, thisdz, q, tmin):
        multi = member & (top - b + 1 >= 2)
        single = member & (top - b + 1 == 1)
        mxf = jnp.minimum(f(1.E-4), _gather(mx, top))
        this = mxf * thisdz / _gather(tdz, top)
        amount = jnp.maximum(f(1.E-6), this * ent)
        q = jnp.where(multi & partial & (t >= f(tmin)), q + cfr * cfr * amount, q)
        return jnp.where(single & partial, q + f(0.05) * qvs * cfr, q)

    qi = _add(imember, ib, it, itdz, imax, idz, qi, 203.16)
    qc = _add(wmember, wb, wt, wtdz, wmax, wdz, qc, 258.16)

    # adjust_cloudFinal (:4557–4594)
    def path(carry, x):
        lwp, iwp = carry
        qck, qik, rk, dzk, pk = x
        lwp = jnp.where(pk, lwp + qck * rk * dzk, lwp)
        iwp = jnp.where(pk, iwp + qik * rk * dzk, iwp)
        return (lwp, iwp), None

    (lwp, iwp), _ = _scan_levels(path, (jnp.zeros(shape, f), jnp.zeros(shape, f)), qc, qi, rhoa, dz, partial)
    qc = jnp.where(partial & (lwp > f(1.0))[..., None], qc * (f(1.0) / lwp)[..., None], qc)
    qi = jnp.where(partial & (iwp > f(1.0))[..., None], qi * (f(1.0) / iwp)[..., None], qi)
    if return_diagnostics:
        # Branch census for oracle coverage (E154); not used by the model.
        return cfr, qc, qi, dict(
            full=full, small=small & ~full, rh=~full & ~small, k_m12c=k_m12c, k_tropo=k_tropo, kbot=kbot,
            ice_multi=imember & (it - ib + 1 >= 2), ice_single=imember & (it == ib),
            h2o_multi=wmember & (wt - wb + 1 >= 2), h2o_single=wmember & (wt == wb),
            lwp_scaled=lwp > f(1.0), iwp_scaled=iwp > f(1.0), top_cleared=(p < f(7000.0)))
    return cfr, qc, qi
