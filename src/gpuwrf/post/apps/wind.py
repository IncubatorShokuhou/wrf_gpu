"""Sampled rotor wind and pitch-regulated power-curve approximations.

REWS is the area-weighted cube mean of speed (DTU/IEC 61400-12-1).
These instantaneous WRF samples are not IEC 10-minute measurements.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from ._numeric import finite


def interpolate_profile(height_agl, value, targets):
    """Linear interpolation on last axis; refuse extrapolation per column."""
    z = finite("height_agl", height_agl)
    v = finite("profile", value)
    h = finite("targets", targets)
    if z.shape != v.shape or z.ndim < 1 or z.shape[-1] < 2:
        raise ValueError("height and profile must have matching (..., level) shapes")
    if np.any(np.diff(z, axis=-1) <= 0):
        raise ValueError("profile heights must increase strictly")
    out = []
    for target in np.atleast_1d(h):
        if np.any(target < z[..., 0]) or np.any(target > z[..., -1]):
            raise ValueError("requested height lies outside the resolved profile; no extrapolation")
        i = np.clip(np.sum(z < target, axis=-1) - 1, 0, z.shape[-1] - 2)
        get = lambda a, j: np.take_along_axis(a, j[..., None], axis=-1)[..., 0]
        z0, z1 = get(z, i), get(z, i + 1)
        v0, v1 = get(v, i), get(v, i + 1)
        out.append(v0 + (v1 - v0) * (target - z0) / (z1 - z0))
    return np.stack(out, axis=-1)


def rotor_weights(sample_heights, hub_height, rotor_diameter):
    """Exact disk-segment areas with boundaries midway between sample heights."""
    hub = float(finite("hub_height", hub_height, minimum=0))
    radius = float(finite("rotor_diameter", rotor_diameter, minimum=1e-6)) / 2
    h = finite("sample_heights", sample_heights)
    if h.ndim != 1 or h.size < 3 or np.any(np.diff(h) <= 0):
        raise ValueError("REWS requires >=3 strictly increasing sample heights")
    if hub <= radius or h[0] < hub - radius or h[-1] > hub + radius:
        raise ValueError("rotor must be above ground and sample heights within disk")
    edges = np.r_[-radius, (h[:-1] + h[1:]) / 2 - hub, radius]
    primitive = edges * np.sqrt(np.maximum(0, radius**2 - edges**2)) + radius**2 * np.arcsin(edges / radius)
    return np.diff(primitive) / (np.pi * radius**2)


def rotor_wind(height_agl, u, v, *, hub_height=100.0, rotor_diameter=100.0,
               samples=9, veer=True):
    """Return hub speed and REWS; optionally project onto hub wind direction."""
    if isinstance(samples, bool) or int(samples) != samples or samples < 3:
        raise ValueError("samples must be an integer >=3")
    hub = float(finite("hub_height", hub_height, minimum=0))
    diameter = float(finite("rotor_diameter", rotor_diameter, minimum=1e-6))
    heights = np.linspace(hub - diameter / 2, hub + diameter / 2, int(samples))
    weights = rotor_weights(heights, hub, diameter)
    uh = interpolate_profile(height_agl, u, [hub])[..., 0]
    vh = interpolate_profile(height_agl, v, [hub])[..., 0]
    speed = np.hypot(uh, vh)
    us = interpolate_profile(height_agl, u, heights)
    vs = interpolate_profile(height_agl, v, heights)
    if veer:
        # The hub direction is undefined when calm: use speed-only in that case.
        along = np.divide(us * uh[..., None] + vs * vh[..., None], speed[..., None],
                          out=np.hypot(us, vs), where=speed[..., None] > 0)
        # IEC veer extension retains the signed rotor-normal component.
        sampled = along
    else:
        sampled = np.hypot(us, vs)
    return speed, np.cbrt(np.sum(weights * sampled**3, axis=-1))


def read_power_curve(path: str | Path):
    """CSV columns speed_ms,power_w; strictly sorted, finite, nonnegative."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.dtype.names != ("speed_ms", "power_w") or table.ndim != 1 or table.size < 2:
        raise ValueError("power curve needs >=2 rows with CSV header speed_ms,power_w")
    speed = finite("curve speed", table["speed_ms"], minimum=0)
    power = finite("curve power", table["power_w"], minimum=0)
    if np.any(np.diff(speed) <= 0):
        raise ValueError("power curve speeds must increase strictly")
    return speed, power


def wind_power(speed_ms, density_kgm3, curve_speed, curve_power, *, reference_density=1.225):
    """Equivalent-speed density correction: v_ref=v*(rho/rho_ref)^(1/3).

    Linear power interpolation, zero outside supplied curve (including cutout).
    This is the pitch-regulated turbine approximation; no wake/availability loss.
    """
    v = finite("speed_ms", speed_ms, minimum=0)
    rho = finite("density_kgm3", density_kgm3, minimum=1e-12)
    ref = float(finite("reference_density", reference_density, minimum=1e-12))
    cs = finite("curve_speed", curve_speed, minimum=0)
    cp = finite("curve_power", curve_power, minimum=0)
    if cs.ndim != 1 or cs.size < 2 or cs.shape != cp.shape or np.any(np.diff(cs) <= 0):
        raise ValueError("power curve must be 1D, matching, and strictly sorted")
    return np.interp(v * np.cbrt(rho / ref), cs, cp, left=0.0, right=0.0)


def sampled_energy_wh(power_w, seconds):
    """Cumulative trapezoidal Wh along axis 0; irregular intervals supported."""
    p = finite("power_w", power_w, minimum=0)
    t = finite("seconds", seconds)
    if p.ndim < 1 or t.ndim != 1 or p.shape[0] != t.size or t.size == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("energy needs strictly increasing timestamps matching power axis 0")
    dt = np.diff(t).reshape((-1,) + (1,) * (p.ndim - 1))
    return np.concatenate([np.zeros_like(p[:1]), np.cumsum((p[1:] + p[:-1]) / 2 * dt / 3600, axis=0)])
