"""Offline Thompson mp=8 radar operator, translated from pristine WRF.

Inputs are arrays (..., vertical), bottom to top. Mixing ratios are kg/kg,
number concentrations #/kg, temperature K, pressure Pa. S-band uses WRF's
Rayleigh operator including wet snow; C/X use full Mie sphere scattering with
the same particle size distributions. No model timestep code is imported.
"""
from __future__ import annotations

import numpy as np

from .radar_scattering import dielectric, dry_table, mie_backscatter, wet_snow_particle

F = np.float32
PI = F(3.1415926536)
AMR = F(PI*F(1000)/F(6))
AMG = F(PI*F(400)/F(6))
AMS = F(.069)
SA = np.array([5.065339, -.062659, -3.032362, .029469, -.000285, .31255, .000204, .003199, 0, -.015952], dtype=F)
SB = np.array([.476221, -.015896, .165977, .007468, -.000141, .060366, .000079, .000594, 0, -.003577], dtype=F)


def _snow_moment(temp, m2, power):
    tc = np.minimum(F(-.1), temp-F(273.15))
    p = F(power)
    def polynomial(c):
        return (c[0]+c[1]*tc+c[2]*p+c[3]*tc*p+c[4]*tc*tc+c[5]*p*p
                +c[6]*tc*tc*p+c[7]*tc*p*p+c[8]*tc*tc*tc+c[9]*p*p*p)
    return F(10)**polynomial(SA)*m2**polynomial(SB)


def _inputs(qv, qr, nr, qs, qg, ng, temperature, pressure):
    arrays = [np.asarray(a, dtype=F) for a in (qv, qr, nr, qs, qg, ng, temperature, pressure)]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim < 1 or arrays[0].shape[-1] < 1:
        raise ValueError("Thompson inputs must have identical shapes with a vertical axis last")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Thompson inputs contain NaN/Inf")
    qv, qr, nr, qs, qg, ng, temp, pres = arrays
    if np.any(temp <= 0) or np.any(pres <= 0):
        raise ValueError("temperature/pressure must be positive")
    if np.any((qr > F(1e-12)) & (nr <= 0)) or np.any((qg > F(1e-6)) & (ng <= 0)):
        raise ValueError("active rain/graupel requires positive number concentration")
    # WRF deliberately floors vapor and ignores sub-threshold/negative species.
    rho = F(.622)*pres/(F(287.04)*temp*(np.maximum(F(1e-10), qv)+F(.622)))
    lr, ls, lg = qr > F(1e-12), qs > F(1e-6), qg > F(1e-6)
    rr, rs, rg = [np.where(active, q*rho, F(1e-12)) for q, active in ((qr, lr), (qs, ls), (qg, lg))]
    nr = np.maximum(F(1e-6), nr*rho)
    ng = np.maximum(F(1e-6), ng*rho)
    # WRF uses REAL product before DOUBLE exponentiation; 1/3 is REAL.
    lamr = (F(AMR*F(6))*nr/rr).astype(float)**float(F(1)/F(3))
    lamg = (F(AMG*F(6))*ng/rg).astype(float)**float(F(1)/F(3))
    m2 = rs*F(F(1)/AMS)
    m3 = _snow_moment(temp, m2, 3)
    m4 = _snow_moment(temp, m2, 4)
    # WRF scans from model top down for the first warm/raining layer with ice above.
    melt_start = np.zeros(temp.shape[:-1], dtype=int)
    if temp.shape[-1] > 1:
        trigger = (temp[..., :-1] > F(273.15)) & lr[..., :-1] & (ls[..., 1:] | lg[..., 1:])
        melt_start = np.max(np.where(trigger, np.arange(1, temp.shape[-1]), 0), axis=-1)
    at_start = np.take_along_axis(rs, melt_start[..., None], axis=-1)
    snow_at_start = np.take_along_axis(ls, melt_start[..., None], axis=-1)
    wet = ls & snow_at_start & (np.arange(temp.shape[-1]) < melt_start[..., None])
    melt = np.clip(1-rs.astype(float)/at_start, .05, .99)
    return lr, ls, lg, rr, rs, rg, nr, ng, lamr, lamg, m2, m3, m4, wet, melt


def _wet_snow(m2, m3, melt, wavelength, *, mie=False, chunk=256):
    """WRF 50-bin midpoint/Simpson integration with bounded workspace."""
    edge = np.geomspace(1e-4, .02, 51)
    d = np.sqrt(edge[:-1]*edge[1:])
    width = np.diff(edge)
    weights = np.zeros(51)
    for n in range(0, 49, 2):
        weights[n:n+3] += [1/3, 4/3, 1/3]
    out = np.empty(m2.size)
    mw, _ = dielectric(wavelength)
    kw = abs((mw*mw-1)/(mw*mw+2))**2
    for start in range(0, m2.size, chunk):
        sl = slice(start, start+chunk)
        inv = F(1)/m3[sl]
        ratio = m2[sl]*inv
        scale = m2[sl]*ratio*ratio*ratio
        slope1, slope2 = ratio*F(20.78), ratio*F(3.29)
        large_d, index = wet_snow_particle(d[None], melt[sl, None], wavelength)
        if mie:
            # Convert radar cross section to equivalent water reflectivity (m^6/m^3).
            particle_z = wavelength**4/(np.pi**5*kw)*mie_backscatter(large_d, index, wavelength)
        else:
            # WRF lamda4/PI5 cancels in its Rayleigh CBACK integration.
            particle_z = abs((index**2-1)/(index**2+2))**2*large_d**6/kw
        psd = scale[:, None]*(float(F(490.6))*np.exp(-slope1[:, None]*d)
                             +float(F(17.46))*(ratio[:, None]*d)**float(F(.6357))*np.exp(-slope2[:, None]*d))
        out[sl] = np.sum(psd*particle_z*width*weights[:50], axis=1)
    return out.astype(F)


def thompson_reflectivity(qv, qr, nr, qs, qg, ng, temperature, pressure, *, band="S"):
    """dBZ, including the WRF writer floor -35 dBZ.

``ng`` is mandatory, including for mp=8: WRF evolves it locally during the
microphysics call and standard mp=8 history omits it. Supplying zero only works
when no graupel exceeds the WRF radar threshold. Hail-aware mp=38 is unsupported.
"""
    if band not in ("S", "C", "X"):
        raise ValueError("radar band must be S, C, or X")
    a = _inputs(qv, qr, nr, qs, qg, ng, temperature, pressure)
    lr, ls, lg, rr, rs, rg, nr, ng, lamr, lamg, m2, m3, m4, wet, melt = a
    if band == "S":
        zr = np.where(lr, nr.astype(float)*lamr*720/lamr**7, 1e-22).astype(F)
        zg = np.where(lg, F(F(.176)/F(.93))*F(F(6)/PI)**2*F(AMG/F(900))**2
                      *ng.astype(float)*lamg*720/lamg**7, 1e-22).astype(F)
        zs = np.where(ls, F(F(.176)/F(.93))*F(F(6)/PI)*F(F(6)/PI)
                      *F(AMS/F(900))*F(AMS/F(900))*m4, F(1e-22)).astype(F)
        if wet.any():
            zs[wet] = _wet_snow(m2[wet], m3[wet], melt[wet], .1)
        total = (zr+zs+zg).astype(F).astype(float)
    else:
        wavelength = {"C": .056, "X": .032}[band]
        mw, _ = dielectric(wavelength)
        factor = wavelength**4/(np.pi**5*abs((mw*mw-1)/(mw*mw+2))**2)
        total = np.zeros(rr.shape)
        for species, mask, slope, number in (("rain", lr, lamr, nr), ("graupel", lg, lamg, ng), ("snow", ls & ~wet, None, None)):
            d, widths, sigma = dry_table(species, wavelength)
            inds = np.flatnonzero(mask)
            for start in range(0, inds.size, 256):
                ind = inds[start:start+256]
                if species == "snow":
                    ratio = (m2.ravel()[ind]/m3.ravel()[ind]).astype(float)[:, None]
                    scale = m2.ravel()[ind, None]*ratio**3
                    psd = scale*(float(F(490.6))*np.exp(-ratio*float(F(20.78))*d)
                                 +float(F(17.46))*(ratio*d)**float(F(.6357))*np.exp(-ratio*float(F(3.29))*d))
                else:
                    lam = slope.ravel()[ind, None]
                    psd = number.ravel()[ind, None]*lam*np.exp(-lam*d)
                total.ravel()[ind] += factor*np.sum(psd*widths*sigma, axis=1)
        if wet.any():
            total[wet] += _wet_snow(m2[wet], m3[wet], melt[wet], wavelength, mie=True)
    result = np.maximum(-35., 10*np.log10(np.maximum(total*1e18, 10**(-3.5))))
    if not np.isfinite(result).all():
        raise ValueError("non-finite radar result")
    return result.astype(F)
