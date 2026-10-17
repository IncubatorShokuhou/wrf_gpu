"""Host-only REAL32 plans for WRF's stochastic spectral generator.

This is SETUP_RAND_PERTURB arithmetic, not a new random-number generator.
Plans are built once outside tracing/timestepping. No JAX arrays are cached.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np


def initialize_spectrum(nx, ny, nz, *, variable, dx, dy, dt,
                        stddev=.5, lengthscale=150000., timescale=21600.,
                        backscatter=1e-6, decorrelation=10800., exponent=-1.83,
                        kmin=1, lmin=1, kmax=1000000, lmax=1000000, vertical=0):
    """Return WRF amplitude/alpha/vertical operands as host float32 arrays.

    Variable codes follow WRF: W streamfunction, T potential temperature,
    P SPPT, R generic random perturbation, S/Q/O SPP conv/PBL/LSM. WRF's
    domain extent includes its final staggered point. WRF's setup accepts an
    integer timestep, including the caller's integer truncation convention.
    """
    sizes = (nx, ny, nz, dt, kmin, lmin, kmax, lmax, vertical)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in sizes):
        raise ValueError("grid sizes, timestep, wave bounds and vertical option must be integers")
    if min(nx, ny) < 12 or nz < 1 or dt < 1 or vertical not in (0, 1):
        raise ValueError("WRF stochastic setup needs nx/ny>=12, nz/dt>=1 and vertical=0/1")
    if variable not in {"W", "T", "P", "R", "S", "Q", "O"}:
        raise ValueError("unknown WRF stochastic variable code")
    if any(not math.isfinite(v) for v in (dx, dy, stddev, lengthscale, timescale, backscatter, decorrelation, exponent)):
        raise ValueError("stochastic setup parameters must be finite")
    if min(dx, dy, timescale, decorrelation) <= 0 or min(stddev, lengthscale, backscatter) < 0:
        raise ValueError("invalid stochastic setup scale")
    f = np.float32
    with np.errstate(over="ignore", under="ignore"):
        dx, dy, stddev, lengthscale = map(f, (dx, dy, stddev, lengthscale))
        timescale, backscatter, decorrelation, exponent = map(f, (timescale, backscatter, decorrelation, exponent))
    if (not np.isfinite([dx, dy, stddev, lengthscale, timescale, backscatter, decorrelation, exponent]).all()
            or min(dx, dy, timescale, decorrelation) <= 0):
        raise ValueError("stochastic setup scales are not representable as finite positive REAL32")
    rx, ry, step, pi = f(nx)*dx, f(ny)*dy, f(dt), f(3.141592653589793)
    upper_k, upper_l = min(min(nx//2, ny//2)-5, kmax), min(min(nx//2, ny//2)-5, lmax)
    alpha = step/decorrelation
    sigma2 = f(1)/(f(12)*alpha)
    gaussian = variable in {"P", "R", "S", "Q", "O"}
    if gaussian:
        kappa = lengthscale*lengthscale
        # The 1-phi cancellation amplifies a one-ulp vector-exp error. Host
        # libm's correctly rounded scalar value reproduces the WRF expf here.
        phi = f(math.exp(float(-step/timescale)))
        alpha = f(1)-phi
    zchi = np.zeros((ny, nx), dtype="f4")
    gamma = f(0)
    # Source accumulation order: j-wave outer, i-wave inner, REAL each add.
    for ik in range(ny+1):
        for il in range(nx+1):
            x, y = f(il), f(ik)
            rho2 = y/ry*y/ry + x/rx*x/rx
            rho = np.sqrt(rho2)
            inside = ((rho < (f(upper_k)+f(.5))/rx and rho >= (f(kmin)-f(.5))/rx)
                      or (rho < (f(upper_l)+f(.5))/ry and rho >= (f(lmin)-f(.5))/ry))
            if inside and (ik > 0 or il > 0):
                if variable in {"W", "T"}:
                    zchi[ik, il] = rho2**(exponent/f(2))
                    gamma += rho2**(exponent + f(1) if variable == "W" else exponent)
                else:
                    zchi[ik, il] = np.exp(-f(2)*(pi*pi)*kappa*rho2)
                    gamma += np.exp(-f(4)*(pi*pi)*kappa*rho2)
    gamma *= f(4)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("WRF spectral band has no finite nonzero normalization")
    if variable == "W":
        normalization = np.sqrt(alpha*backscatter/(step*sigma2*gamma))/(f(2)*pi)
    elif variable == "T":
        normalization = np.sqrt(f(300)*alpha*backscatter/(step*f(1006)*sigma2*gamma))
    else:
        normalization = stddev*np.sqrt((f(1)-phi*phi)/(f(2)*gamma))
    amplitude = np.zeros_like(zchi)
    for j in range(ny):
        for i in range(nx):
            upper = j <= ny//2
            jj = j if upper else ny-j
            ii = i if (i <= nx//2 if upper else i < nx//2) else nx-i
            amplitude[j, i] = normalization*zchi[jj, ii]
    vc = np.ones((nz, ny, nx), dtype="f4")
    vs = np.zeros_like(vc)
    if vertical:
        maxrho = np.sqrt(f(ny*ny)/(dy*dy) + f(nx*nx)/(dx*dx))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    left = i < nx//2
                    wave = i+2 if left else nx-i+1
                    rho = np.sqrt(f((j+2)**2)/(dy*dy) + f(wave*wave)/(dx*dx))
                    eps = ((maxrho-rho)/maxrho)*(f(k+1)/f(32))*pi
                    vc[k,j,i] = np.cos(eps*f(wave))
                    vs[k,j,i] = np.sin(eps*f(wave)) if left else -np.sin(eps*f(wave))
    result = {"amplitude": amplitude, "alpha": f(alpha), "vertical_cosine": vc,
              "vertical_sine": vs, "vertical_amplitude": np.ones(nz, dtype="f4")}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("stochastic setup produced NaN/Inf")
    return result
