"""WRF-Python/RIP CALCDBZ fallback (not scheme-specific Thompson parity)."""
import numpy as np


def rip_reflectivity(pressure, temperature, vapor, rain, snow, graupel, *,
                     variable_intercept=False, liquid_skin=False, snow_present=True):
    """Equivalent reflectivity dBZ, floor -30; inputs Pa/K/kg kg-1.

Matches NCAR wrf-python fortran/wrf_user_dbz.f90. Constant exponential
intercepts 8e6/2e7/4e6 m-4 and particle densities 1000/100/400 kg m-3.
Optional variable intercepts follow Thompson et al. (2004), *not* mp=8's
double-moment 2008 operator. Negative species are zeroed as in CALCDBZ.
"""
    arrays = [np.asarray(v, float) for v in (pressure, temperature, vapor, rain, snow, graupel)]
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("RIP inputs must have identical shapes and no NaN/Inf")
    p, t = arrays[:2]
    if np.any(p <= 0) or np.any(t <= 0):
        raise ValueError("pressure/temperature must be positive")
    qv, qr, qs, qg = [np.maximum(v, 0) for v in arrays[2:]]
    if not snow_present:
        qs = np.where(t < 273.15, qr, qs)
        qr = np.where(t < 273.15, 0, qr)
    rho = p/(287*t*(.622+qv)/(.622*(1+qv)))
    nr, ns, ng = 8e6, 2e7, 4e6
    if variable_intercept:
        ns = np.minimum(2e8, 2e6*np.exp(-.12*np.minimum(-.001, t-273.15)))
        ng = np.where(qg > 1e-15, np.clip(2.38*(np.pi*400/np.maximum(rho*qg,1e-300))**.92,1e4,5e7),5e7)
        nr = np.where(qr > 1e-15, (1e10-8e6)*.5*np.tanh((.0001-qr)/.000025)+(1e10+8e6)*.5,1e10)
    factor = lambda density: 720e18*(1/(np.pi*density))**1.75
    dielectric = np.where((t > 273.15) & liquid_skin, 1., .224)
    z = (factor(1000)*(rho*qr)**1.75/np.asarray(nr)**.75
         +factor(100)*(.1)**2*dielectric*(rho*qs)**1.75/np.asarray(ns)**.75
         +factor(400)*(.4)**2*dielectric*(rho*qg)**1.75/np.asarray(ng)**.75)
    result = 10*np.log10(np.maximum(z, .001))
    if not np.isfinite(result).all():
        raise ValueError("non-finite RIP reflectivity")
    return result
