"""WRF ``toposhad_init``/``toposhad`` single-patch reference operator.

The nested input is the interpolated, time-dependent parent HGT_SHAD, not
parent terrain. WRF preserves that height in the outer two mass-grid rows.
The boundary height starts with init-time parent interpolation, then WRF's
BXS supplies the child's own preceding height. Current-parent injection is
not the WRF caller. The pure JAX loop is the CPU reference; the GPU ray loop
must run inside the native kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def toposhad_solar_geometry(latitude_deg, longitude_deg, declination_rad,
                           xtime_minutes, gmt_hours, radt_minutes):
    """WRF REAL toposhad clock, preserving its assignment rounding."""
    from gpuwrf.backends import pallas as pl
    if pl.cuda_native():
        from gpuwrf.kernels.topographic_shadow import topographic_solar_geometry_native
        return topographic_solar_geometry_native(latitude_deg, longitude_deg, declination_rad,
                                                 xtime_minutes, gmt_hours, radt_minutes)
    F = jnp.float32
    barrier = jax.lax.optimization_barrier
    def mul(a, b): return barrier(a * b)
    def add(a, b): return barrier(a + b)
    def sub(a, b): return barrier(a - b)
    def div(a, b): return barrier(a / barrier(b))
    degrad = div(F(jnp.pi), F(180))
    lat = mul(jnp.asarray(latitude_deg, F), degrad)
    xt24 = jnp.fmod(add(F(xtime_minutes), mul(F(radt_minutes), F(.5))), F(1440))
    tloc = add(add(F(gmt_hours), div(xt24, F(60))), div(jnp.asarray(longitude_deg, F), F(15)))
    hrang = mul(mul(F(15), sub(tloc, F(12))), degrad)
    declin = F(declination_rad)
    csza = add(mul(jnp.sin(lat), jnp.sin(declin)),
               mul(mul(jnp.cos(lat), jnp.cos(declin)), jnp.cos(hrang)))
    return csza, hrang


def toposhad_solar_azimuth(latitude_deg, coszen, declination_rad, hour_angle_rad, sina, cosa):
    """Pristine topographic ray direction, rotated onto the WRF grid."""
    F = jnp.float32
    pi = F(jnp.pi)
    lat = jnp.asarray(latitude_deg, F) * (pi / F(180))
    csza, declin, hrang = F(coszen), F(declination_rad), F(hour_angle_rad)
    argu = (csza * jnp.sin(lat) - jnp.sin(declin)) / (jnp.sin(jnp.arccos(csza)) * jnp.cos(lat))
    azi = jnp.copysign(jnp.arccos(jnp.clip(argu, F(-1), F(1))), jnp.sin(hrang)) + pi
    rotation = jnp.where(jnp.asarray(cosa) >= 0, jnp.arcsin(jnp.asarray(sina, F)),
                         pi - jnp.arcsin(jnp.asarray(sina, F)))
    return azi + rotation


def toposhad_solar_geometry_from_seconds(latitude_deg, longitude_deg, declination_rad,
                                         midpoint_lead_seconds, base_utc_minutes):
    """Convert the operational caller's clock inside the rounded shadow kernel.

    Converting /60 in XLA before the call would reintroduce div.full error
    (e.g. 1800 seconds becomes 30.000001907 minutes). The lead already
    includes the caller's radiation midpoint, so radt is zero here.
    """
    from gpuwrf.backends import pallas as pl
    if pl.cuda_native():
        from gpuwrf.kernels.topographic_shadow import topographic_solar_geometry_native
        return topographic_solar_geometry_native(latitude_deg, longitude_deg, declination_rad,
            midpoint_lead_seconds, base_utc_minutes, 0., caller_units=True)
    denominator = jax.lax.optimization_barrier(jnp.float32(60))
    return toposhad_solar_geometry(latitude_deg, longitude_deg, declination_rad,
        jnp.asarray(midpoint_lead_seconds, jnp.float32) / denominator,
        jnp.asarray(base_utc_minutes, jnp.float32) / denominator, 0.)


def topographic_shadow(
    terrain_height_m,
    *,
    latitude_deg,
    longitude_deg,
    sina,
    cosa,
    declination_rad,
    xtime_minutes,
    gmt_hours,
    radt_minutes: float,
    dx_m: float,
    dy_m: float,
    shadow_length_m: float,
    parent_shadow_height_m=None,
    terrain_halo_m=None,
    geometry_override=None,
):
    """Return ``(SHADOWMASK, HGT_SHAD)`` on mass points, in WRF REAL.

    ``terrain_halo_m`` is the WRF two-point halo, shape ``(ny+4,nx+4)``.
    The edge-repeated fallback is an explicit single-patch physical-boundary
    convention; callers reproducing a different halo must supply it. The
    parent input uses child mass-point registration and is read only on the
    outer two rows. WRF's Fortran INT is applied in ONE-based coordinates.
    Solar phase is the toposhad formula, which does not add calc_coszen's EOT.
    """
    terrain = jnp.asarray(terrain_height_m, jnp.float32)
    ny, nx = terrain.shape
    if min(ny, nx) < 1:
        raise ValueError("toposhad needs a nonempty mass grid")
    F = jnp.float32
    yy, xx = jnp.indices((ny, nx), dtype=jnp.int32)
    x, y = (xx + 1).astype(terrain.dtype), (yy + 1).astype(terrain.dtype)
    border = (xx < 2) | (xx >= nx - 2) | (yy < 2) | (yy >= ny - 2)
    height = terrain - F(.001)
    imported = jnp.zeros_like(border)
    local = terrain
    if parent_shadow_height_m is not None:
        parent = jnp.asarray(parent_shadow_height_m, terrain.dtype)
        if parent.shape != terrain.shape:
            raise ValueError("parent HGT_SHAD must be interpolated onto child mass points")
        height = jnp.where(border, parent, height)
        imported = border & (parent > terrain)
        local = jnp.where(imported, parent, terrain)
    halo = (jnp.pad(terrain, 2, mode="edge") if terrain_halo_m is None
            else jnp.asarray(terrain_halo_m, terrain.dtype))
    if halo.shape != (ny + 4, nx + 4):
        raise ValueError("toposhad terrain halo must have two points on each side")
    halo = halo.at[2:-2, 2:-2].set(local)
    pi = F(jnp.pi)
    if geometry_override is None:
        csza, hrang = toposhad_solar_geometry(latitude_deg, longitude_deg, declination_rad,
                                             xtime_minutes, gmt_hours, radt_minutes)
    else:
        csza, hrang = (jnp.asarray(a, F) for a in geometry_override)
    daylight = csza >= F(.01)
    azi = toposhad_solar_azimuth(latitude_deg, csza, declination_rad, hrang, sina, cosa)
    north = (azi > F(1.75) * pi) | (azi < F(.25) * pi)
    east = (~north) & (azi < F(.75) * pi)
    south = (~north) & (~east) & (azi < F(1.25) * pi)
    along_y = north | south
    direction = jnp.where(north | east, F(1), F(-1))
    tangent = jnp.where(along_y, jnp.tan(azi), -jnp.tan(pi / F(2) + azi))
    # WRF exits a ray at its first point outside the patch plus two-point halo.
    def scan(step, carry):
        mask, ht, active = carry
        delta = F(step) * direction
        ri = x + jnp.where(along_y, delta * tangent, delta)
        rj = y + jnp.where(along_y, delta, delta * tangent)
        i1, j1 = jnp.trunc(ri).astype(jnp.int32), jnp.trunc(rj).astype(jnp.int32)
        i2 = i1 + jnp.where(along_y, 1, 0)
        j2 = j1 + jnp.where(along_y, 0, 1)
        valid = (i1 > -2) & (i2 < nx + 3) & (j1 > -2) & (j2 < ny + 3)
        active = active & valid
        weight = jnp.where(along_y, ri - i1.astype(terrain.dtype), rj - j1.astype(terrain.dtype))
        h1 = halo[jnp.clip(j1 + 1, 0, ny + 3), jnp.clip(i1 + 1, 0, nx + 3)]
        h2 = halo[jnp.clip(j2 + 1, 0, ny + 3), jnp.clip(i2 + 1, 0, nx + 3)]
        distance = jnp.sqrt((F(dx_m) * (ri - x)) ** 2 + (F(dy_m) * (rj - y)) ** 2)
        elevation = jnp.arctan((weight * h2 + (F(1) - weight) * h1 - local) / distance)
        hit = active & (jnp.sin(elevation) >= csza)
        projected = local + distance * (jnp.tan(elevation) - jnp.tan(jnp.arcsin(csza)))
        return mask | hit, jnp.where(hit, jnp.maximum(ht, projected), ht), active

    steps = int(float(shadow_length_m) / float(dx_m) + 1)
    mask, height, _ = jax.lax.fori_loop(1, steps + 1, scan, (imported, height, daylight))
    return (jnp.where(daylight, mask, False).astype(jnp.int32),
            jnp.where(daylight, height, local - F(.001)))


def topographic_shadow_from_geometry(terrain_height_m, *, latitude_deg, coszen,
                                    declination_rad, hour_angle_rad, sina, cosa,
                                    dx_m, dy_m, shadow_length_m, previous_height_m=None):
    """Dispatch the same per-cell operator to CUDA or the validated CPU reference.

    A non-None previous height denotes a nest's BXS (its OWN preceding height,
    originally initialized from its parent). None selects the root init rule.
    Geometry is the separate toposhad clock, not RRTMG's EOT-bearing geometry.
    """
    from gpuwrf.backends import pallas as pl
    if pl.cuda_native():
        from gpuwrf.kernels.topographic_shadow import topographic_shadow_native
        azi = toposhad_solar_azimuth(latitude_deg, coszen, declination_rad, hour_angle_rad, sina, cosa)
        terrain = jnp.asarray(terrain_height_m, jnp.float32)
        old = jnp.zeros_like(terrain) if previous_height_m is None else jnp.asarray(previous_height_m, jnp.float32)
        return topographic_shadow_native(terrain, old, coszen, azi,
            dx_m=dx_m, dy_m=dy_m, shadow_length_m=shadow_length_m,
            nested=previous_height_m is not None)
    return topographic_shadow(terrain_height_m, latitude_deg=latitude_deg, longitude_deg=latitude_deg,
        sina=sina, cosa=cosa, declination_rad=declination_rad, xtime_minutes=0., gmt_hours=0.,
        radt_minutes=0., dx_m=dx_m, dy_m=dy_m, shadow_length_m=shadow_length_m,
        parent_shadow_height_m=previous_height_m, geometry_override=(coszen, hour_angle_rad))


__all__ = ["topographic_shadow", "topographic_shadow_from_geometry",
           "toposhad_solar_geometry", "toposhad_solar_azimuth",
           "toposhad_solar_geometry_from_seconds"]
