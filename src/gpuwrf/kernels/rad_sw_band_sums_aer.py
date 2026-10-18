"""WRF aer_opt=1 variant of the fused RRTMG SW band kernel (kernels/rad_sw_band_sums.py).

Same launch, adding-method passes and cloud optics as the released kernel; two WRF ``spcvmc_sw`` terms differ:

* clear-sky optics include the aerosol (module_ra_rrtmg_sw.F:8555-8558) and its delta scaling is no longer the
  g = 0 identity (:8603-8609): ``ztauc = ztaur + ztaug + ptaua``, ``zomcc = ztaur + ptaua*pomga``,
  ``zgcc = pasya*pomga*ptaua / zomcc``, ``zomcc /= ztauc``, then ``zf = zgcc**2``, ``zwf = zomcc*zf``,
  ``ztauc *= 1 - zwf``, ``zomcc = (zomcc - zwf)/(1 - zwf)``, ``zgcc = (zgcc - zf)/(1 - zf)``;
* the emitted direct flux is WRF's ``ztdbt_nodel`` product (:8577-8600, :8737): the UNSCALED clear
  (gas + Rayleigh + aerosol) and clear + unscaled-cloud optical depths, while the adding method keeps the
  delta-scaled transmittance.  (Without aerosol and clouds the two coincide.)

``aer`` is ``(..., nlay, nbands, 3)`` = ztaua, zomga, zasya per layer and band (g-point independent).  Kept in a
separate file: the released kernel's source lines enter the Triton TTIR of the default program (E109).
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.kernels.rad_sw_band_sums import F32, TX, WARPS, _c, _lookup, _reftra, _ssa_upper


def _kernel(tau_gas, tau_ray, cloud, incloud, cfac, gmask, sflux, colf, band_ref, aer,
            down, up, direct, clear_down, clear_up, su, sud, scu, scud, *, nlay, ng, ngin, nbands_loop, mp_re):
    rr = pl.program_id(0) * TX + jnp.arange(TX)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    gin = jnp.minimum(points, ngin - 1)
    zero, one, eps, tiny = _c(0.0), _c(1.0), _c(1.0e-12), _c(1.0e-10)
    mu = colf[rows, 0]
    alb = colf[rows, 1] + jnp.zeros((TX, ng), F32)
    outs = (down, up, direct, clear_down, clear_up)

    if nbands_loop is None:
        def sunlit():
            _solve_band(band_ref[0], lambda j: (tau_gas[rows, j, gin], tau_ray[rows, j, gin]), None)
    else:
        def sunlit():
            def per_band(b, carry):
                _solve_band(b, lambda j: (tau_gas[rows, j, b, gin], tau_ray[rows, j, b, gin]), b)
                return carry
            jax.lax.fori_loop(0, nbands_loop, per_band, 0)

    def dark():
        def clear(lev, carry):
            for out in outs:
                if nbands_loop is None:
                    out[rr, lev] = jnp.zeros((TX,), F32)
                else:
                    for b in range(nbands_loop):
                        out[rr, b, lev] = jnp.zeros((TX,), F32)
            return carry
        jax.lax.fori_loop(0, nlay + 1, clear, 0)

    def put(out, lev, value, slot):
        if slot is None:
            out[rr, lev] = value
        else:
            out[rr, slot, lev] = value

    def _solve_band(band, taus, slot):
        top = jnp.where(points < ngin, (mu * colf[rows, 2]) * sflux[rows, band, gin], zero)
        m = gmask[band, points] + jnp.zeros((TX, ng), F32)
        mu_b = mu + jnp.zeros((TX, ng), F32)
        mu_d = jnp.maximum(mu_b, _c(1.0e-6))
        if not mp_re:
            lx0, ix0, sx0 = cfac[0, band], cfac[1, band], cfac[2, band]
            ld0, id0, sd0 = cfac[3, band], cfac[4, band], cfac[5, band]
            lo0, io0, so0 = cfac[6, band], cfac[7, band], cfac[8, band]
            la0, ia0, sa0 = cfac[9, band], cfac[10, band], cfac[11, band]

        def cloud_coefficients(j):
            if not mp_re:
                return (lx0, ix0, sx0), (ld0, id0, sd0), (lo0, io0, so0), (la0, ia0, sa0)

            def coeff(slot_, radius, ice):
                start, step, length = (5., 3., 46) if ice else (2.5, 1., 58)
                idx = jnp.clip(jnp.floor((radius - _c(start)) / _c(step)).astype(jnp.int32), 0, length - 2)
                frac = (radius - (_c(start) + idx.astype(F32) * _c(step))) / _c(step)
                return cfac[slot_, band, idx] + frac * (cfac[slot_, band, idx + 1] - cfac[slot_, band, idx])
            rl, ri, rs = incloud[rr, j, 3], incloud[rr, j, 4], incloud[rr, j, 5]
            lx, lssa, lasy = (coeff(n, rl, False) for n in (0, 1, 2))
            ix, issa, iasy, ifwd = (coeff(n, ri, True) for n in (3, 4, 5, 6))
            sx, sssa, sasy, sfwd = (coeff(n, rs, True) for n in (3, 4, 5, 6))

            def scale_coeff(ssa, asym, forward):
                den = jnp.maximum(one - forward * ssa, eps)
                om = jnp.clip(ssa * (one - forward) / den, zero, _c(_ssa_upper()))
                asy = jnp.clip((asym - forward) / jnp.maximum(one - forward, eps), _c(-.999999), _c(.999999))
                return den[:, None], om[:, None], asy[:, None]
            ld, lo, la = scale_coeff(lssa, lasy, lasy * lasy)
            id_, io, ia = scale_coeff(issa, iasy, jnp.minimum(iasy, ifwd + _c(.5) / issa))
            sd, so, sa = scale_coeff(sssa, sasy, jnp.minimum(sasy, sfwd + _c(.5) / sssa))
            return (lx[:, None], ix[:, None], sx[:, None]), (ld, id_, sd), (lo, io, so), (la, ia, sa)

        def layer(k):
            j = nlay - 1 - k            # inputs are bottom-up layers; the solve is top-down
            (lx, ix, sx), (ld, id_, sd), (lo, io, so), (la, ia, sa) = cloud_coefficients(j)
            tg, tr = taus(j)
            c = cloud[rows, j, band, gin]
            ta, wa, ga = aer[rows, j, band, 0], aer[rows, j, band, 1], aer[rows, j, band, 2]
            # WRF clear-sky optics incl. aerosol + delta scaling (literal operation order)
            tco = (tg + tr) + ta
            sc = tr + ta * wa
            gc = ((ga * wa) * ta) / jnp.maximum(sc, tiny)
            oc = sc / jnp.maximum(tco, tiny)
            f = gc * gc
            wf = oc * f
            tcl = (one - wf) * tco
            ocl = (oc - wf) / jnp.maximum(one - wf, eps)
            acl = (gc - f) / jnp.maximum(one - f, eps)
            # cloud (cldprmc in-cloud paths x McICA amount); unscaled tl0/ti0/ts0 for WRF's direct output
            tl0, ti0, ts0 = (incloud[rows, j, 0] * lx) * c, (incloud[rows, j, 1] * ix) * c, (incloud[rows, j, 2] * sx) * c
            tl, ti, ts = ld * tl0, id_ * ti0, sd * ts0
            sl, si, ss = tl * lo, ti * io, ts * so
            tcd = tl + ti + ts
            scd = sl + si + ss
            ocd = jnp.clip(scd / jnp.maximum(tcd, tiny), zero, _c(_ssa_upper()))
            ocd = jnp.where(c > zero, ocd, one)
            acd = jnp.where(scd > tiny, (sl * la + si * ia + ss * sa) / jnp.maximum(scd, tiny), zero)
            stot = tcl * ocl + tcd * ocd
            ttot = jnp.maximum(tcl + tcd, tiny)
            otot = jnp.clip(stot / jnp.maximum(ttot, tiny), zero, _c(_ssa_upper()))
            atot = jnp.where(stot > tiny, (tcl * ocl * acl + tcd * ocd * acd) / jnp.maximum(stot, tiny), zero)
            tcl = jnp.maximum(tcl, tiny) * m
            ocl = ocl * m
            acl = acl * m
            ttot = ttot * m
            otot = otot * m
            atot = atot * m
            cl = jnp.clip(c, zero, one)
            cloud_active = m * (cl > _c(1.0e-12)).astype(F32)
            rc, rdc, tc, tdc = _reftra(tcl, ocl, acl, mu_b, m)
            rk, rdk, tk, tdk = _reftra(ttot, otot, atot, mu_b, cloud_active)
            w = one - cl
            r = w * rc + cl * rk
            rd = w * rdc + cl * rdk
            t = w * tc + cl * tk
            td = w * tdc + cl * tdk
            bcl = _lookup(tcl / mu_d)
            bcd = _lookup(ttot / mu_d)
            b = jnp.where(m > zero, (w * bcl + cl * bcd) * m, one)
            bc = jnp.where(m > zero, bcl, one)
            # ztdbt_nodel: unscaled clear and clear + unscaled cloud (zdbtmc / zdbtmo of the first layer loop)
            tno = tco * m
            nbcl = _lookup(tno / mu_d)
            nbcd = _lookup((tno + (tl0 + ti0 + ts0) * m) / mu_d)
            bn = jnp.where(m > zero, (w * nbcl + cl * nbcd) * m, one)
            return (r, rd, t, td, b, bn), (rc, rdc, tc, tdc, bc)

        su[rows, nlay, points] = alb
        sud[rows, nlay, points] = alb
        scu[rows, nlay, points] = alb
        scud[rows, nlay, points] = alb

        def reflect(i, carry):
            u, ud, cu, cud = carry
            k = nlay - 1 - i
            (r, rd, t, td, b, _bn), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - ud * rd, eps)
            nu = r + td * ((t - b) * ud + b * u) * refl
            nud = rd + td * td * ud * refl
            reflc = one / jnp.maximum(one - cud * rdc, eps)
            ncu = rc + tdc * ((tc - bc) * cud + bc * cu) * reflc
            ncud = rdc + tdc * tdc * cud * reflc
            su[rows, k, points] = nu
            sud[rows, k, points] = nud
            scu[rows, k, points] = ncu
            scud[rows, k, points] = ncud
            return nu, nud, ncu, ncud

        jax.lax.fori_loop(0, nlay, reflect, (alb, alb, alb, alb))

        def emit(d, tr_, df, u, ud):
            refl = one / jnp.maximum(one - df * ud, eps)
            fup = (d * u + (tr_ - d) * ud) * refl
            fdn = d + (tr_ - d + d * u * df) * refl
            return jnp.sum(fdn * top, axis=1), jnp.sum(fup * top, axis=1)

        def store(k, d, tr_, df, dc, trc, dfc, dn):
            lev = nlay - k              # output is bottom-up interfaces
            fdn, fup = emit(d, tr_, df, su[rows, k, points], sud[rows, k, points])
            cdn, cup = emit(dc, trc, dfc, scu[rows, k, points], scud[rows, k, points])
            put(down, lev, fdn, slot)
            put(up, lev, fup, slot)
            put(direct, lev, jnp.sum(dn * top, axis=1), slot)
            put(clear_down, lev, cdn, slot)
            put(clear_up, lev, cup, slot)

        def transmit(k, carry):
            d, tr_, df, dc, trc, dfc, dn = carry
            store(k, d, tr_, df, dc, trc, dfc, dn)
            (r, rd, t, td, b, bn), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - rd * df, eps)
            ntr = d * t + td * ((tr_ - d) + d * r * df) * refl
            ndf = rd + td * td * df * refl
            reflc = one / jnp.maximum(one - rdc * dfc, eps)
            ntrc = dc * tc + tdc * ((trc - dc) + dc * rc * dfc) * reflc
            ndfc = rdc + tdc * tdc * dfc * reflc
            return d * b, ntr, ndf, dc * bc, ntrc, ndfc, dn * bn

        ones = jnp.ones((TX, ng), F32)
        zeros = jnp.zeros((TX, ng), F32)
        final = jax.lax.fori_loop(0, nlay, transmit, (ones, ones, zeros, ones, ones, zeros, ones))
        store(nlay, *final)

    jax.lax.cond(jnp.max(mu) > zero, sunlit, dark)


def _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
            coszen, surface_albedo, source_scale, band, aer, *, all_bands, interpret):
    tau_suffix = 3 if all_bands else 2
    leading, nlay, ngin = tau_gas.shape[:-tau_suffix], tau_gas.shape[-tau_suffix], tau_gas.shape[-1]
    ncol = math.prod(leading) if leading else 1
    npad = ((ncol + TX - 1) // TX) * TX
    nbands = cloud_amount.shape[-2]
    ng = 1 << (ngin - 1).bit_length()

    def pack(value, suffix):
        a = jnp.broadcast_to(value, leading + suffix).reshape((ncol,) + suffix).astype(F32)
        return a if npad == ncol else jnp.pad(a, ((0, npad - ncol),) + ((0, 0),) * len(suffix))

    tau_shape = (nlay, nbands, ngin) if all_bands else (nlay, ngin)
    colf = jnp.stack([coszen, surface_albedo, source_scale], axis=-1)
    gmask = jnp.pad(jnp.asarray(gpoint_mask, F32), ((0, 0), (0, ng - ngin)))
    args = [pack(tau_gas, tau_shape), pack(tau_ray, tau_shape), pack(cloud_amount, (nlay, nbands, ngin)),
            pack(incloud, (nlay, incloud.shape[-1])), jnp.asarray(cloud_coeffs, F32), gmask, pack(sfluxzen, (nbands, ngin)),
            pack(colf, (3,)), jnp.reshape(jnp.asarray(band, jnp.int32), (1,)), pack(aer, (nlay, nbands, 3))]
    flux = jax.ShapeDtypeStruct((npad, nbands, nlay + 1) if all_bands else (npad, nlay + 1), F32)
    scratch = jax.ShapeDtypeStruct((npad, nlay + 1, ng), F32)
    kernel = partial(_kernel, nlay=nlay, ng=ng, ngin=ngin, nbands_loop=nbands if all_bands else None,
                     mp_re=incloud.shape[-1] == 6)
    result = pl.pallas_call(kernel, grid=(npad // TX,),
                            out_shape=[flux] * 5 + [scratch] * 4, interpret=interpret,
                            compiler_params=pt.CompilerParams(num_warps=WARPS),
                            name="rrtmg_sw_allband_sums_aer" if all_bands else "rrtmg_sw_band_sums_aer")(*args)
    fields = result[:5]
    if all_bands:
        def band_sum(x):
            acc = x[:, 0]
            for i in range(1, nbands):
                acc = acc + x[:, i]
            return acc
        fields = tuple(band_sum(x) for x in fields)
    return tuple(x[:ncol].reshape(leading + (nlay + 1,)) for x in fields)


def sw_band_flux_sums_aer(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                          coszen, surface_albedo, source_scale, band, aer, *, interpret=False):
    """:func:`rad_sw_band_sums.sw_band_flux_sums` with WRF aerosol optics ``aer (..., nlay, nbands, 3)``."""
    return _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                   coszen, surface_albedo, source_scale, band, aer, all_bands=False, interpret=interpret)


def sw_allband_flux_sums_aer(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                             coszen, surface_albedo, source_scale, aer, *, interpret=False):
    """:func:`rad_sw_band_sums.sw_allband_flux_sums` with WRF aerosol optics ``aer (..., nlay, nbands, 3)``."""
    return _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                   coszen, surface_albedo, source_scale, 0, aer, all_bands=True, interpret=interpret)
