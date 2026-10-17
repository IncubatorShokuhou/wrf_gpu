"""Spherical particle scattering; no JAX or forecast dependencies.

Wet-particle geometry and nested Maxwell-Garnett mixing follow the default
air/water/icewater/spheroidal branch of WRF module_mp_radar.F (Blahak/Thompson).
Mie backscatter is evaluated by the optional, independently validated miepython.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


def dielectric(wavelength):
    """WRF's Ray (1972) water / Maetzler (1998) ice indices at 0 Celsius.

Preserve default-REAL rounding in WRF's CMPLX and constants. Negative imaginary
index denotes absorption. Water formula valid for wavelengths 0.001–1 m.
"""
    w = float(wavelength)
    if not np.isfinite(w) or not 0.001 <= w <= 1.0:
        raise ValueError("wavelength must be finite in [0.001, 1] metres")
    f32 = np.float32
    epsinf = 5.27137
    epss = 78.54 * (1 + .004579*25 + .0000119*625 + .000000028*15625)
    alpha = -16.8129/273.16 + .0609265
    ls = .00033836*np.exp(2513.98/273.16)*float(f32(.01))
    a = (ls/w)**(1-alpha)
    denom = 1 + 2*a*np.sin(alpha*np.pi*.5) + a*a
    er = epsinf + (epss-epsinf)*(a*np.sin(alpha*np.pi*.5)+1)/denom
    ei = (epss-epsinf)*a*np.cos(alpha*np.pi*.5)/denom + w*float(f32(1.25664))/float(f32(1.88496))
    mw = complex(np.sqrt(np.complex64(complex(er, -ei))))
    freq = 2.99e8/w*1e-9
    beta = float(f32(.0207))/273.16 * np.exp(335/273.16)/(np.exp(335/273.16)-1)**2 + float(f32(1.16e-11))*freq**2 + np.exp(-float(f32(10.02)))
    theta = 300/273.16 - 1
    alfa = (.00504 + .0062*theta)*np.exp(-22.1*theta)
    mi = np.sqrt(complex(float(f32(3.1884)), -float(f32(alfa/freq + beta*freq))))
    return mw, mi


def _mix(matrix, inclusion, fraction):
    """WRF spheroidal Maxwell-Garnett, with one nonzero inclusion species."""
    e1, e2 = np.asarray(matrix)**2, np.asarray(inclusion)**2
    delta = e2-e1
    # Continuous equal-index limit, useful for a completely melted particle.
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(abs(delta) < 1e-14, 1., 2*e1/delta*(e2/delta*np.log(e2/e1)-1))
    return np.sqrt(((1-fraction)*e1 + fraction*beta*e2)/(1-fraction+fraction*beta))


def wet_snow_particle(diameter, melt_fraction, wavelength=.1):
    """Return wet-snow sphere diameter (m) and effective complex index.

Snow mass=.069 D**2 kg; ice 900, liquid 1000 kg/m³; 90% exterior
melting, increasing linearly to 100% at full melt. This is an effective
homogeneous sphere, not a coated-sphere solution or an aggregate-shape model.
"""
    d, fm = np.broadcast_arrays(np.asarray(diameter, float), np.asarray(melt_fraction, float))
    if not np.isfinite(d).all() or not np.isfinite(fm).all() or np.any(d <= 0) or np.any((fm < 0) | (fm > 1)):
        raise ValueError("particle diameter >0 and melt fraction in [0,1] must be finite")
    am = float(np.float32(.069))
    mass = am*d**2
    # WRF a_geo=REAL((1/am)**(1/2)), b_geo=REAL(1/2).
    a_geo = float(np.float32(np.float32(1)/np.float32(.069))**np.float32(.5))
    dry_d = a_geo*np.sqrt(mass)
    density = np.clip(mass/(np.pi/6*dry_d**3), 10., 900.)
    dry_volume = mass/density
    mra = .9 + .1*fm
    limit = (900-density)/(mra*900-density+900*density/1000)
    volume = np.where((mra <= 1-density/1000) | (fm <= limit), dry_volume*(1-mra*fm), mass*(1-fm)/900 + mass*fm/1000)
    vi = mass*(1-fm)/(volume*900)
    vw = mass*fm/(volume*1000)
    va = 1-vi-vw
    mw, mi = dielectric(wavelength)
    mixed_iw = _mix(mw, mi, vi/np.maximum(vi+vw, 1e-10))
    mixed = _mix(mixed_iw, 1.+0j, va)
    # Fortran literal 1./3. is default REAL even inside DOUBLE expression.
    large_d = (6/np.pi*volume)**float(np.float32(1)/np.float32(3))
    return large_d, mixed


def mie_backscatter(diameter, refractive_index, wavelength):
    """Monostatic radar cross-section in m² (Qback × geometric area).

miepython uses the Wiscombe/Bohren-Huffman full Mie series. Its Qback includes
the 4*pi radar convention; no additional 4*pi factor belongs here.
"""
    try:
        import miepython
    except ImportError as exc:
        raise ImportError("C/X radar requires: pip install 'gpuwrf[radar]'") from exc
    d, m = np.broadcast_arrays(np.asarray(diameter, float), np.asarray(refractive_index, complex))
    if not np.isfinite(d).all() or not np.isfinite(m).all() or np.any(d <= 0):
        raise ValueError("Mie particle diameter/index must be finite, diameter >0")
    dielectric(wavelength)  # validate wavelength even for empty inputs
    _, _, back, _ = miepython.efficiencies_mx(m.ravel(), (np.pi*d/wavelength).ravel())
    result = np.asarray(back).reshape(d.shape)*np.pi*d*d/4
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("non-finite/negative Mie backscatter")
    return result


@lru_cache(maxsize=8)
def dry_table(species, wavelength, bins=256):
    """Small cached immutable scattering table; no model-sized cache entries."""
    limits = {"rain": (1e-7, .02), "snow": (1e-7, .05), "graupel": (1e-7, .05)}
    if species not in limits:
        raise ValueError("unknown hydrometeor")
    edges = np.geomspace(*limits[species], bins+1)
    d = np.sqrt(edges[:-1]*edges[1:])
    mw, mi = dielectric(wavelength)
    if species == "rain":
        sphere_d, m = d, mw
    elif species == "snow":
        # Preserve the mass law at small sizes where a raw geometric sphere
        # would imply density > solid ice; use the WRF volume convention.
        sphere_d, m = wet_snow_particle(d, np.zeros_like(d), wavelength)
    else:
        sphere_d, rho = d, np.full_like(d, 400.)
        # Air inclusions in an ice matrix, using the same default mixing rule.
        m = _mix(mi, 1.+0j, 1-rho/900)
    sigma = mie_backscatter(sphere_d, m, wavelength)
    for a in (d, edges, sigma):
        a.setflags(write=False)
    return d, np.diff(edges), sigma
