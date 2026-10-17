"""Thompson mp=8 online 10 cm reflectivity at the microphysics exit.

The boundary is ``calc_refl10cm`` in pristine WRF ``module_mp_thompson.F``;
inputs have trailing bottom-to-top mass levels, kg/kg and number/kg. In
particular ``ng`` is the private MP-exit number, never an entry reconstruction.
The default-REAL parts remain float32; distribution slopes, wet-particle
scattering and the serial 50-bin integral are WRF DOUBLE islands. The GPU
implementation keeps a single column in registers, with no bins-by-domain
intermediate. CPU execution uses the same equations under vmap for testing.
"""
from __future__ import annotations

import cmath
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


class ThompsonRadarInputs(NamedTuple):
    qv: jax.Array
    qr: jax.Array
    nr: jax.Array
    qs: jax.Array
    qg: jax.Array
    ng: jax.Array
    temperature: jax.Array
    pressure: jax.Array


def final_graupel_number(qg, ng, rho):
    """Private ng1d final balance, WRF :4059–4077, for diagnostics only.

    ``rho`` retains its pre-final-temperature lifetime. This does not write the
    prognostic State: the pre-existing default Thompson graph remains intact.
    """
    real, double = jnp.float32, jnp.float64
    qg, ng, rho = [jnp.asarray(x, real) for x in (qg, ng, rho)]
    ng = jnp.maximum(jnp.asarray(1e-6, real) / rho, ng)
    q = jnp.maximum(qg, jnp.asarray(1e-12, real))
    slope = ((jnp.asarray(F(AMG * F(6)), real) * ng / q)
             ** jnp.asarray(THIRD, real)).astype(double)
    numerator = F(F(F(3) + F(0)) + F(.672))
    diameter = (jnp.asarray(numerator, double) / slope).astype(real)
    diameter = jnp.maximum(jnp.asarray(50e-6, real), jnp.minimum(jnp.asarray(25.4e-3, real), diameter))
    slope = (jnp.asarray(numerator, real) / diameter).astype(double)
    balanced = (jnp.asarray(F(F(1) / F(6)), real) * q).astype(double) * slope ** 3 / jnp.asarray(AMG, double)
    return jnp.where(qg > jnp.asarray(1e-12, real), balanced.astype(real), jnp.asarray(0., real))


F = np.float32
PI = F(3.1415926536)
AMR = F(PI * F(1000) / F(6))
AMG = F(PI * F(400) / F(6))
AMS = F(.069)
THIRD = F(F(1) / F(3))
SA = (5.065339, -.062659, -3.032362, .029469, -.000285, .31255,
      .000204, .003199, 0., -.015952)
SB = (.476221, -.015896, .165977, .007468, -.000141, .060366,
      .000079, .000594, 0., -.003577)


def _radar_constants():
    """WRF radar_init constants, including its default REAL CMPLX rounding."""
    wavelength = float(F(.10))  # DOUBLE PARAMETER initialized by REAL literal.
    epss = 78.54 * (1 + .004579 * 25 + .0000119 * 625 + .000000028 * 15625)
    alpha = -16.8129 / 273.16 + .0609265
    ls = .00033836 * math.exp(2513.98 / 273.16) * float(F(.01))
    a = (ls / wavelength) ** (1 - alpha)
    denom = 1 + 2 * a * math.sin(alpha * math.pi * .5) + a * a
    er = 5.27137 + (epss - 5.27137) * (a * math.sin(alpha * math.pi * .5) + 1) / denom
    ei = ((epss - 5.27137) * a * math.cos(alpha * math.pi * .5) / denom
          + wavelength * float(F(1.25664)) / float(F(1.88496)))
    mw = complex(np.sqrt(np.complex64(complex(er, -ei))))
    freq = 2.99e8 / wavelength * 1e-9
    beta = (float(F(.0207)) / 273.16 * math.exp(335 / 273.16)
            / (math.exp(335 / 273.16) - 1) ** 2 + float(F(1.16e-11)) * freq ** 2
            + math.exp(-float(F(10.02))))
    theta = 300 / 273.16 - 1
    alfa = (.00504 + .0062 * theta) * math.exp(-22.1 * theta)
    mi = cmath.sqrt(complex(float(F(3.1884)), -float(F(alfa / freq + beta * freq))))
    ew, eice = mw * mw, mi * mi
    kw = abs((ew - 1) / (ew + 2)) ** 2
    edges = [1e-4] + [math.exp(n / 50 * math.log(.02 / 1e-4) + math.log(1e-4))
                       for n in range(1, 50)] + [.02]
    weights = [0.] * 51
    for n in range(0, 49, 2):
        for offset, weight in enumerate((1 / 3, 4 / 3, 1 / 3)):
            weights[n + offset] += weight
    bins = [(math.sqrt(edges[n] * edges[n + 1]),
             weights[n] * (edges[n + 1] - edges[n])) for n in range(50)]
    return (ew.real, ew.imag), (eice.real, eice.imag), kw, tuple(bins)


EW, EI, KW, BINS = _radar_constants()


def _mul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _div(a, b):
    norm = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / norm,
            (a[1] * b[0] - a[0] * b[1]) / norm)


def _mix_eps(matrix, inclusion, fraction):
    """Default spheroidal Maxwell-Garnett, represented as two DOUBLE reals.

    Returning permittivity cancels the source's sqrt-followed-by-square. No
    complex device type or model-sized scattering table is required.
    """
    delta = (inclusion[0] - matrix[0], inclusion[1] - matrix[1])
    ratio = _div(inclusion, matrix)
    logarithm = (.5 * jnp.log(ratio[0] * ratio[0] + ratio[1] * ratio[1]),
                 jnp.arctan2(ratio[1], ratio[0]))
    part = _mul(_div(inclusion, delta), logarithm)
    beta = _mul(_div((2 * matrix[0], 2 * matrix[1]), delta), (part[0] - 1., part[1]))
    term = _mul(beta, inclusion)
    numerator = ((1 - fraction) * matrix[0] + fraction * term[0],
                 (1 - fraction) * matrix[1] + fraction * term[1])
    denominator = (1 - fraction + fraction * beta[0], fraction * beta[1])
    return _div(numerator, denominator)


def _wet_particle_z(diameter, melt_fraction):
    """WRF rayleigh_soak_wetgraupel divided by K_w*PI5/lamda4."""
    double = jnp.float64
    d = jnp.asarray(diameter, double)
    fm = melt_fraction
    mass = jnp.asarray(AMS, double) * d ** 2
    a_geo = jnp.asarray(F(F(1) / AMS) ** F(.5), double)
    dry_d = a_geo * jnp.sqrt(mass)
    volume = jnp.asarray(math.pi / 6, double) * dry_d ** 3
    density = jnp.minimum(jnp.maximum(mass / volume, 10.), 900.)
    volume = mass / density
    mra = .9 + (1. - .9) * fm
    xw = mass * fm
    limit = (900. - density) / (mra * 900. - density + 900. * density / 1000.)
    volume = jnp.where((mra <= 1. - density / 1000.) | (fm <= limit),
                       volume * (1. - mra * fm), (mass - xw) / 900. + xw / 1000.)
    vi = (mass - xw) / (volume * 900.)
    vw = xw / (1000. * volume)
    va = 1. - vi - vw
    iw = _mix_eps(tuple(jnp.asarray(x, double) for x in EW),
                  tuple(jnp.asarray(x, double) for x in EI), vi / jnp.maximum(vi + vw, 1e-10))
    mixed = _mix_eps(iw, (jnp.asarray(1., double), jnp.asarray(0., double)), va)
    quotient = _div((mixed[0] - 1., mixed[1]), (mixed[0] + 2., mixed[1]))
    large_d = (6. / math.pi * volume) ** jnp.asarray(THIRD, double)
    return (quotient[0] ** 2 + quotient[1] ** 2) * large_d ** 6 / KW


def _snow_moment(tc, m2, power):
    p = jnp.asarray(power, jnp.float32)
    def polynomial(coefficients):
        c = tuple(jnp.asarray(F(x)) for x in coefficients)
        return (c[0] + c[1] * tc + c[2] * p + c[3] * tc * p + c[4] * tc * tc
                + c[5] * p * p + c[6] * tc * tc * p + c[7] * tc * p * p
                + c[8] * tc * tc * tc + c[9] * p * p * p)
    return jnp.asarray(10., jnp.float32) ** polynomial(SA) * m2 ** polynomial(SB)


def _column_reflectivity(qv, qr, nr, qs, qg, ng, temp, pres, *, nz, ke_diag, gather, read_bin=None):
    real, double = jnp.float32, jnp.float64
    qv, qr, nr, qs, qg, ng, temp, pres = [jnp.asarray(x, real) for x in (qv, qr, nr, qs, qg, ng, temp, pres)]
    k = jnp.arange(qv.shape[-1], dtype=jnp.int32)
    valid = k < nz
    rho = jnp.asarray(.622, real) * pres / (jnp.asarray(287.04, real) * temp
                                            * (jnp.maximum(jnp.asarray(1e-10, real), qv) + jnp.asarray(.622, real)))
    lr, ls, lg = valid & (qr > 1e-12), valid & (qs > 1e-6), valid & (qg > 1e-6)
    rr, rs, rg = [jnp.where(active, q * rho, jnp.asarray(1e-12, real))
                  for active, q in ((lr, qr), (ls, qs), (lg, qg))]
    nr = jnp.where(lr, jnp.maximum(jnp.asarray(1e-6, real), nr * rho), jnp.asarray(1e-12, real))
    ng = jnp.where(lg, jnp.maximum(jnp.asarray(1e-6, real), ng * rho), jnp.asarray(1e-6, real))
    # Fortran's REAL RHS is evaluated before assignment to DOUBLE lamr/lamg.
    lamr = ((jnp.asarray(F(AMR * F(6)), real) * nr / rr) ** jnp.asarray(THIRD, real)).astype(double)
    lamg = ((jnp.asarray(F(AMG * F(6)), real) * ng / rg) ** jnp.asarray(THIRD, real)).astype(double)
    ir, ig = 1. / lamr, 1. / lamg
    n0r, n0g = nr.astype(double) * lamr, ng.astype(double) * lamg
    m2 = rs * jnp.asarray(F(F(1) / AMS), real)
    tc = jnp.minimum(jnp.asarray(-.1, real), temp - jnp.asarray(273.15, real))
    m3, m4 = _snow_moment(tc, m2, 3), _snow_moment(tc, m2, 4)
    zr = jnp.where(lr, n0r * 720. * ir ** 7, 1e-22).astype(real)
    factor_s = F(F(F(F(F(F(.176) / F(.93)) * F(F(6) / PI)) * F(F(6) / PI)) * F(AMS / F(900))) * F(AMS / F(900)))
    factor_g = F(F(F(F(F(F(.176) / F(.93)) * F(F(6) / PI)) * F(F(6) / PI)) * F(AMG / F(900))) * F(AMG / F(900)))
    zs = jnp.where(ls, jnp.asarray(factor_s) * m4, jnp.asarray(1e-22, real))
    zg = jnp.where(lg, jnp.asarray(factor_g, real).astype(double) * n0g * 720. * ig ** 7, 1e-22).astype(real)
    above_ice = gather(ls | lg, jnp.minimum(k + 1, qv.shape[-1] - 1))
    trigger = (k + 1 < nz) & (temp > jnp.asarray(273.15, real)) & lr & above_ice
    k0 = jnp.max(jnp.where(trigger, k + 1, 0))
    rs0 = gather(rs, jnp.broadcast_to(k0, k.shape))
    ls0 = gather(ls, jnp.broadcast_to(k0, k.shape))
    wet = ls & ls0 & (k < k0) & (k < ke_diag)

    def wet_integral():
        if ke_diag == 1:
            # k0 and rs0 above still inspect the complete physical column.
            # Only the diagnosed bottom layer needs DOUBLE particle scattering.
            def bottom(values):
                return jnp.sum(jnp.where(k == 0, values, jnp.asarray(0., real)))
            work_rs, work_rs0, work_m2, work_m3 = map(bottom, (rs, rs0, m2, m3))
        else:
            work_rs, work_rs0, work_m2, work_m3 = rs, rs0, m2, m3
        # The source ratio is REAL division, promoted by 1.0d0 subtraction.
        fm = jnp.maximum(.05, jnp.minimum(1. - (work_rs / work_rs0).astype(double), .99))
        ratio = work_m2 * (jnp.asarray(1., real) / work_m3)
        scale = work_m2 * ratio * ratio * ratio
        slope1, slope2 = ratio * jnp.asarray(20.78, real), ratio * jnp.asarray(3.29, real)
        bins = jnp.asarray(BINS, double)
        def bin_body(index, eta):
            d, weight = (bins[index, 0], bins[index, 1]) if read_bin is None else read_bin(index)
            f = scale.astype(double) * (jnp.asarray(F(490.6), double) * jnp.exp(-slope1.astype(double) * d)
                                       + jnp.asarray(F(17.46), double) * (ratio.astype(double) * d) ** jnp.asarray(F(.6357), double)
                                       * jnp.exp(-slope2.astype(double) * d))
            return eta + f * _wet_particle_z(d, fm) * weight
        eta = lax.fori_loop(0, 50, bin_body, jnp.zeros_like(work_rs, dtype=double))
        return jnp.where(wet, eta.astype(real), zs)

    zs = lax.cond(jnp.max(wet.astype(jnp.int32)) != 0, wet_integral, lambda: zs)
    # WRF adds the three REAL contributions before the DOUBLE scale/log10.
    z = (10. * jnp.log10((zr + zs + zg).astype(double) * 1e18)).astype(real)
    return jnp.where(k < ke_diag, jnp.maximum(jnp.asarray(-35., real), z), jnp.asarray(-35., real))


def thompson_reflectivity(qv, qr, nr, qs, qg, ng, temperature, pressure, *, ke_diag=None, interpret=None):
    """WRF mp=8 ``REFL_10CM``, with optional bottom-only diagnostics.

    ``ke_diag`` is the number of diagnosed levels (1 or the full physical
    column in operational WRF). All levels above it remain -35 dBZ. Full-column
    melting detection is retained even when only the bottom level is required.
    The caller must supply finite MP-exit fields and the transient ``ng``.
    """
    arrays = tuple(jnp.asarray(x, jnp.float32) for x in (qv, qr, nr, qs, qg, ng, temperature, pressure))
    shape = arrays[0].shape
    if not shape or any(a.shape != shape for a in arrays):
        raise ValueError("Thompson radar inputs must have identical (..., vertical) shapes")
    nz = shape[-1]
    ke_diag = nz if ke_diag is None else int(ke_diag)
    if not 1 <= ke_diag <= nz:
        raise ValueError("ke_diag must be within the physical column")
    if interpret is None and jax.default_backend() == "cpu":
        flat = tuple(a.reshape(-1, nz) for a in arrays)
        fn = lambda *column: _column_reflectivity(*column, nz=nz, ke_diag=ke_diag,
                                                  gather=lambda a, i: a[i])
        return jax.vmap(fn)(*flat).reshape(shape)

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    from gpuwrf.kernels.phys_thompson_sedimentation import _gather, _load, _store
    ncol, width = math.prod(shape[:-1]), 1 << (nz - 1).bit_length()
    def kernel(*refs):
        col = pl.program_id(0)
        k = jnp.arange(width, dtype=jnp.int32)
        defaults = (0., 0., 0., 0., 0., 0., 280., 100000.)
        column = [_load(ref, (col, k), mask=k < nz, other=default)
                  for ref, default in zip(refs[:-2], defaults)]
        out = _column_reflectivity(*column, nz=nz, ke_diag=ke_diag, gather=_gather.bind,
                                   read_bin=lambda n: (_load(refs[-2], (n, 0)), _load(refs[-2], (n, 1))))
        _store(refs[-1], (col, k), out, mask=k < nz)
    out = pl.pallas_call(kernel, grid=(ncol,), out_shape=jax.ShapeDtypeStruct((ncol, nz), jnp.float32),
                         interpret=bool(interpret), compiler_params=plt.CompilerParams(num_warps=2),
                         name="thompson_calc_refl10cm")(*(a.reshape(ncol, nz) for a in arrays), jnp.asarray(BINS, jnp.float64))
    return out.reshape(shape)
