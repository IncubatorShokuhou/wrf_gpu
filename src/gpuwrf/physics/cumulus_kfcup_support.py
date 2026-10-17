"""WRF KF-CuP cloud-feedback components, for component qualification only.

Literal REAL32 ports of ``cup_jfd`` and ``cupCloudFraction`` from pristine
WRF 4.7.1 ``phys/module_cu_kfcup.F``, plus its radiation-driver mediation.
This is NOT a cu_physics=10 driver:
the CuP parcel ensemble, convection and radiation mediation remain unported.
No production module imports this file, and cu=10 remains fail-closed.

Arrays use bottom-up, zero-based mass levels. Cloud bounds are inclusive.
``ishall`` is static: 0 deep, 1 shallow, 2 absent. The cloud routine returns
only outputs assigned by WRF in the selected branch. In particular WRF
never assigns cldfraTend, and only shallow convection defines taucloud,
tactive, lnterms and lnint; exposing invented zero diagnostics would hide
an undefined-output dependency at the caller boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _real(value):
    return jnp.asarray(value, dtype=jnp.float32)


def find_cp(mixing_ratio):
    """WRF findCp: moist heat capacity (J kg-1 K-1), REAL32."""
    return _real(1004.67) * (_real(1.0) + _real(0.84) * _real(mixing_ratio))


def find_rs(temperature, pressure):
    """WRF findRs: saturation mixing ratio over water (kg/kg), K and Pa.

    The original comment says hPa, but its 610.78 coefficient and the CPS
    caller p1d=Pcps require Pa. Keep that caller convention (E11).
    """
    t, p = _real(temperature), _real(pressure)
    es = _real(610.78) * jnp.exp(_real(17.67) * (t - _real(273.16)) / (t - _real(29.66)))
    return _real(0.622) * es / (p - es)


def find_rsi(temperature, pressure):
    """Literal WRF findRsi, including its historical hPa-sized ice es.

    This standalone helper is NOT used by WRF's current cloud-fraction
    calculation (the ice-term code is commented out). Do not silently fix
    its units while claiming a line-faithful port.
    """
    t, p = _real(temperature), _real(pressure)
    esi = _real(10.0) ** (
        -_real(9.09718) * (_real(273.15) / t - _real(1.0))
        - _real(3.56654) * jnp.log10(_real(273.15) / t)
        + _real(0.876793) * (_real(1.0) - t / _real(273.15))
        + jnp.log10(_real(6.1071))
    )
    return _real(0.622) * esi / (p - esi)


def cup_slope_sigma(temperature, theta, vapor, density, height, *, surface_heat,
                    surface_moisture, land, pblh, kpbl, terrain):
    """Live arithmetic of WRF cupSlopeSigma:4614–4912, REAL32.

    ``kpbl`` retains WRF's one-based index. The original computes a surface
    stability classification and then never uses it: the final CuP selection
    depends on land, heat flux and kpbl only. Those dead calculations do not
    influence any output. wstar is undefined in WRF on the disabled branch;
    its returned zero is an internal placeholder that callers must not use.
    """
    t, th, qv, rho, z = map(_real, (temperature, theta, vapor, density, height))
    hfx, qfxin, land, pblh, terrain = map(_real, (surface_heat, surface_moisture, land, pblh, terrain))
    kpbl = jnp.asarray(kpbl, jnp.int32)
    enabled = (kpbl > 2) & (hfx >= jnp.where(land == 1, _real(100.0), _real(1.0)))
    def active(_):
        ep1 = _real(_real(461.6) / _real(287.0) - _real(1.0))
        cp, lv = _real(1004.5), _real(2.5e6)
        qfx_floor = _real(1.0) / lv
        qfx = jnp.where(jnp.abs(qfxin) < qfx_floor, jnp.copysign(qfx_floor, qfxin), qfxin)
        middle = jnp.maximum(1, kpbl // 2) - 1
        above = jnp.minimum(kpbl + 2, t.shape[0]) - 1
        flux = (_real(1.0) + ep1*qv[0])*hfx/rho[0]/cp + ep1*th[0]*qfx/rho[0]
        thv = th[middle]*(_real(1.0) + ep1*qv[middle])
        wstar = (_real(9.81)*pblh*flux/thv)**(_real(1.0)/_real(3.0))
        slope_sfc = hfx/(lv*qfx)*lv/cp
        qdiff = qv[above] - qv[middle]
        qdiff = jnp.where(jnp.abs(qdiff) < _real(5.0e-4), jnp.copysign(_real(5.0e-4), qdiff), qdiff)
        slope_ez = (th[above]*(_real(1.0) + ep1*qv[above]) - thv)/qdiff
        normalized_z = (z[middle] - terrain)/pblh
        sigma_ez = flux/wstar*(_real(2.0) + _real(8.2e-4)*normalized_z**_real(-1.8))
        sigma_sfc = (qfx/rho[0])/wstar*(_real(2.3) + _real(1.1e-2)*normalized_z**_real(-1.6))
        result = dict(slope_sfc=slope_sfc, slope_ez=slope_ez, sigma_sfc=sigma_sfc, sigma_ez=sigma_ez, wstar=wstar)
        return dict(cupflag=jnp.array(True), **result)

    def inactive(_):
        return dict(cupflag=jnp.array(False), slope_sfc=_real(0.0), slope_ez=_real(0.0),
                    sigma_sfc=_real(0.0), sigma_ez=_real(0.0), wstar=_real(0.0))

    return jax.lax.cond(enabled, active, inactive, None)


def cup_adjust_mass_flux(umf, uer, udr, dmf, der, ddr, *, cloud_top, ishall,
                         shift_previous=None):
    """Literal adjust_mfentdet_kfcup with zero-based inclusive cloud_top.

    Each layer's adjustment uses its original adjacent mass flux, so the
    independent layer arithmetic can run in parallel. Source max operations
    are retained, including the small-flux branch's ordered recomputation.
    """
    umf, uer, udr, dmf, der, ddr = map(_real, (umf, uer, udr, dmf, der, ddr))
    levels = jnp.arange(umf.shape[0])
    inside = levels <= cloud_top

    def adjust(flux, entrain, detrain):
        flux = jnp.where(levels < cloud_top, flux, _real(0.0))
        entrain = jnp.where(inside, entrain, _real(0.0))
        detrain = jnp.where(inside, detrain, _real(0.0))
        previous = (jnp.concatenate((jnp.zeros(1, jnp.float32), flux[:-1]))
                    if shift_previous is None else shift_previous(flux))
        delta = flux - previous
        residual = delta - (entrain - detrain)
        positive = residual > 0
        small_entrain = entrain < detrain*_real(0.05)
        small_detrain = detrain <= entrain*_real(0.05)
        less_detrain = jnp.maximum(_real(0.0), detrain-residual)
        more_entrain = entrain + (delta - (entrain-less_detrain))
        less_entrain = jnp.maximum(_real(0.0), entrain+residual)
        more_detrain = detrain - (delta - (less_entrain-detrain))
        e = jnp.where(positive, jnp.where(small_entrain, more_entrain, entrain+residual),
                      jnp.where(small_detrain, less_entrain, entrain))
        d = jnp.where(positive, jnp.where(small_entrain, less_detrain, detrain),
                      jnp.where(small_detrain, more_detrain, detrain-residual))
        return flux, jnp.where(inside, e, _real(0.0)), jnp.where(inside, d, _real(0.0))

    up = adjust(umf, uer, udr)
    down = adjust(dmf, der, ddr)
    active = jnp.asarray(ishall) < 2
    deep = jnp.asarray(ishall) == 0
    up = tuple(jnp.where(active, new, old) for new,old in zip(up,(umf,uer,udr)))
    down = tuple(jnp.where(deep, new, old) for new,old in zip(down,(dmf,der,ddr)))
    return dict(umf=up[0],uer=up[1],udr=up[2],dmf=down[0],der=down[1],ddr=down[2])


def cup_jfd(slope_sfc, slope_ez, sigma_sfc, sigma_ez, num_bins, th_bin_size, r_bin_size):
    """CuP joint frequency distribution; theta axis first, moisture second.

    ``num_bins`` must be static. Modified slope/sigma arguments are returned
    because WRF declares them INOUT. The normalizing sum follows WRF's
    j-outer/i-inner order rather than an unordered parallel reduction.
    """
    if not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError("num_bins must be a positive static integer")
    sfc, ez, ss, se = map(_real, (slope_sfc, slope_ez, sigma_sfc, sigma_ez))
    tiny = _real(1.0e-15)
    sfc = jnp.copysign(jnp.maximum(jnp.abs(sfc), tiny), sfc)
    ez = jnp.where(ez > _real(2000.0), _real(2000.0), ez)
    ez = jnp.where(ez < _real(-2000.0), _real(-2000.0), ez)
    ez = jnp.where((ez < _real(10.0)) & (ez > 0), _real(10.0), ez)
    ez = jnp.where((ez < 0) & (ez > _real(-10.0)), _real(-10.0), ez)
    ss = jnp.copysign(jnp.maximum(jnp.abs(ss), tiny), ss)
    se = jnp.copysign(jnp.maximum(jnp.abs(se), tiny), se)
    delta = ez - sfc
    dslope = jnp.copysign(jnp.maximum(jnp.abs(delta), tiny), delta)
    bins = jnp.arange(num_bins, dtype=jnp.int32) - num_bins // 2
    theta = _real(th_bin_size) * bins.astype(jnp.float32)
    moisture = _real(r_bin_size) * bins.astype(jnp.float32)
    c = ez * (theta[:, None] - sfc * moisture[None, :]) / dslope
    m = (theta[:, None] - ez * moisture[None, :]) / dslope
    bigcheck = jnp.sqrt(_real(jnp.finfo(jnp.float32).max))

    def overflow_term(value):
        # Preserve WRF's unusual overflow branch: signed bigcheck, NOT its
        # square. This is a source guard, not an added physics clamp.
        return jnp.where(jnp.abs(value) > bigcheck, jnp.copysign(bigcheck, value), value * value)

    raw = jnp.exp(_real(-0.5) * (overflow_term(m / ss) + overflow_term(c / se)))
    # Flatten in Fortran traversal order, retaining one scalar reduction
    # carry rather than a bins-by-bins history or repeated JFD allocations.
    ordered = raw.T.reshape(-1)
    total = jax.lax.fori_loop(0, num_bins * num_bins, lambda k, acc: acc + ordered[k], _real(0.0))
    jfd = jax.lax.cond(total != 0, lambda a: a / total, lambda a: a, raw)
    return {"slope_sfc": sfc, "slope_ez": ez, "sigma_sfc": ss, "sigma_ez": se,
            "th_perturb": theta, "r_perturb": moisture, "jfd": jfd}


def cup_cloud_fraction(qlg, qig, qv, temperature, height, pressure, *, cloud_bottom,
                       cloud_top, ishall, wstar, wparcel, pblh, active_frac, mfup):
    """Return the outputs WRF defines in this cloud regime (REAL32).

    ``qig`` is accepted to preserve the WRF input boundary; WRF's active
    shallow expression integrates liquid alone. This function intentionally
    does not add ice to that expression. ``dt`` is absent because its only
    use in the original subroutine is commented out.
    """
    if ishall not in (0, 1, 2):
        raise ValueError("WRF KF-CuP ishall must be 0 (deep), 1 (shallow) or 2 (absent)")
    liquid, ice, vapor, t, z, p, mass_flux = map(_real, (qlg, qig, qv, temperature, height, pressure, mfup))
    if liquid.ndim != 1 or liquid.shape[0] < 3 or any(a.shape != liquid.shape for a in (ice, vapor, t, z, p, mass_flux)):
        raise ValueError("KF-CuP requires matching mass-level vectors with at least 3 levels")
    n = liquid.shape[0]
    levels = jnp.arange(n)
    bottom, top = jnp.asarray(cloud_bottom, jnp.int32), jnp.asarray(cloud_top, jnp.int32)
    inside = (levels >= bottom) & (levels <= top)
    zero = jnp.zeros_like(liquid)
    if ishall == 2:
        return {"cldfra": zero}
    wstar, wparcel, pblh, active_frac = map(_real, (wstar, wparcel, pblh, active_frac))
    tstar = pblh / wstar
    if ishall == 0:
        fraction = jnp.maximum(_real(0.0), jnp.minimum(_real(0.1) * jnp.log(_real(1.0) + _real(675.0) * mass_flux), _real(1.0)))
        return {"cldfra": jnp.where(inside, fraction, zero), "tstar": tstar}

    satdef = jnp.maximum(find_rs(t, p) - vapor, _real(1.0e-20))

    def integrate(k, carry):
        intqc, intsat = carry
        # Literal kp1=min(k+1,kte-1), including the topmost-layer behavior.
        kp1 = jnp.minimum(k + 1, n - 2)
        dz = z[kp1] - z[k]
        intqc = _real(0.5) * (liquid[k] + liquid[kp1]) * dz + intqc
        intsat = _real(0.5) * (satdef[k] + satdef[kp1]) * dz + intsat
        return intqc, intsat

    intqc, intsat = jax.lax.fori_loop(bottom, top + 1, integrate, (_real(0.0), _real(0.0)))
    cp = find_cp(vapor[bottom])
    rs = find_rs(t[bottom], p[bottom])
    lv = _real(3.15e6) - _real(2370.0) * t[bottom]
    gamma = (lv ** 2) * rs / (cp * _real(461.6) * t[bottom] ** 2)
    lnint = jnp.maximum(jnp.log(_real(1.0) + (_real(1.0) + gamma) * intqc / intsat), _real(1.0))
    taucloud = jnp.minimum(tstar * lnint, _real(1800.0))
    tactive = jax.lax.cond(wparcel > 0, lambda _: z[top] / wparcel, lambda _: z[top] / wstar, None)
    fraction = jnp.minimum(jnp.maximum(active_frac * taucloud / tactive, _real(0.01)), _real(1.0))
    return {"cldfra": jnp.where(inside, fraction, zero), "tstar": tstar,
            "taucloud": taucloud, "tactive": tactive, "lnint": lnint, "lnterms": zero}


def cup_radiation_feedback(cldfra, qc, cldfra_cup, cldfra1_flag, *, ishall,
                           cloud_bottom, cloud_top, icloud, shallowcu_forced_ra=False):
    """Literal CuP mediation in module_radiation_driver.F:1510-1565.

    Bottom-up mass-level vectors; cloud bounds are inclusive and zero-based.
    ``ishall`` retains WRF's REAL shall argument, including comparisons with
    its integer truncation. This boundary is tested separately from cloud
    fraction production: it is not wired into operational radiation yet.
    Returns all three arrays the original block may modify. No time history
    or extra whole-domain scratch is required beyond those output arrays.
    """
    fraction, water, cup_fraction = map(_real, (cldfra, qc, cldfra_cup))
    flag = jnp.asarray(cldfra1_flag, jnp.int32)
    if fraction.ndim != 1 or any(a.shape != fraction.shape for a in (water, cup_fraction, flag)):
        raise ValueError("CuP radiation requires matching mass-level vectors")
    shall = _real(ishall)
    bottom, top = _real(cloud_bottom), _real(cloud_top)
    levels = jnp.arange(fraction.size, dtype=jnp.int32)
    # WRF's first test uses INT(CUBOT/CUTOP), its condensate test uses REAL.
    integer_inside = ((levels >= (bottom + _real(1)).astype(jnp.int32) - 1)
                      & (levels <= (top + _real(1)).astype(jnp.int32) - 1))
    real_inside = (levels >= bottom) & (levels <= top)
    special = jnp.asarray(icloud) == 1
    clear = special & (flag == 1)
    overcast = special & (flag == 2) & (shall > _real(1))
    fraction = jnp.where(clear, _real(0), jnp.where(overcast, _real(1), fraction))
    compute = (~special) | ((flag == 1) & (shall <= _real(1))) | (flag == 3)
    use_cup = compute & (shall.astype(jnp.int32) <= 1) & integer_inside
    fraction = jnp.where(use_cup, cup_fraction, fraction)
    cup_fraction = jnp.where(compute & ~use_cup & (shall > _real(1)), _real(0), cup_fraction)
    condensate = (shall <= _real(1)) & real_inside
    water = jnp.where(condensate, jnp.maximum(cup_fraction * _real(1.0e-3), water), water)
    fraction = jnp.where(condensate & jnp.asarray(shallowcu_forced_ra),
                         jnp.maximum(_real(0.36), fraction), fraction)
    return {"cldfra": fraction, "qc": water, "cldfra_cup": cup_fraction}
