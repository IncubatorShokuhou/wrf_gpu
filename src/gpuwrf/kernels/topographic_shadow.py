"""WRF topographic ray search: independent cells, loop inside one Pallas kernel."""
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas import triton as pt
from gpuwrf.backends import pallas as pl


def _rn(op, a, b, *, interpret):
    if interpret or not pl.cuda_native():
        value = {"add": lambda: a + b, "sub": lambda: a - b,
                 "mul": lambda: a * b,
                 "div": lambda: a / jax.lax.optimization_barrier(b)}[op]()
        return jax.lax.optimization_barrier(value)
    return pt.elementwise_inline_asm(op + ".rn.f32 $0, $1, $2;", args=[a, b],
        constraints="=f,f,f", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]


def _solar_kernel(latitude, longitude, clock, coszen_out, hrang_out,
                  *, size, block, interpret, caller_units):
    lane = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = lane < size
    F = jnp.float32
    def rn(op, a, b):
        a, b = (jnp.broadcast_to(v, (block,)) for v in (a, b))
        return _rn(op, a, b, interpret=interpret)
    declin, minute, gmt, radt = (jnp.broadcast_to(clock[i], (block,)) for i in range(4))
    if caller_units:
        minute = rn("div", minute, F(60))  # midpoint lead seconds -> XTIME minutes
        gmt = rn("div", gmt, F(60))  # base UTC minutes -> GMT hours
    lat = pt.load(latitude.at[lane], mask=valid, other=F(0))
    lon = pt.load(longitude.at[lane], mask=valid, other=F(0))
    # WRF module_model_constants REAL parameter, folded once by its compiler.
    degrad = F(jnp.pi / 180.)
    xxlat = rn("mul", lat, degrad)
    xt24 = jnp.fmod(rn("add", minute, rn("mul", radt, F(.5))), F(1440))
    # XLA's div.full.f32 changes 1125/60 to 18.750001907... . A barrier
    # cannot impose IEEE rounding on division (E84); explicitly use div.rn.
    tloc = rn("add", rn("add", gmt, rn("div", xt24, F(60))), rn("div", lon, F(15)))
    hrang = rn("mul", rn("mul", F(15), rn("sub", tloc, F(12))), degrad)
    csza = rn("add", rn("mul", jnp.sin(xxlat), jnp.sin(declin)),
              rn("mul", rn("mul", jnp.cos(xxlat), jnp.cos(declin)), jnp.cos(hrang)))
    pt.store(coszen_out.at[lane], csza, mask=valid)
    pt.store(hrang_out.at[lane], hrang, mask=valid)


def topographic_solar_geometry_native(latitude_deg, longitude_deg, declination_rad,
                                      xtime_minutes, gmt_hours, radt_minutes, *, interpret=False,
                                      caller_units=False):
    """WRF REAL shadow clock in one kernel with explicitly rounded arithmetic."""
    lat, lon = (jnp.asarray(v, jnp.float32) for v in (latitude_deg, longitude_deg))
    if lat.shape != lon.shape:
        raise ValueError("toposhad latitude and longitude must share the mass-grid shape")
    clock = jnp.stack([jnp.asarray(v, jnp.float32) for v in
                       (declination_rad, xtime_minutes, gmt_hours, radt_minutes)])
    block = 128
    kernel = partial(_solar_kernel, size=lat.size, block=block, interpret=bool(interpret),
                     caller_units=bool(caller_units))
    out = jax.ShapeDtypeStruct((lat.size,), jnp.float32)
    result = pl.pallas_call(kernel, out_shape=(out, out), grid=((lat.size + block - 1) // block,),
        name="wrf_toposhad_solar", interpret=bool(interpret),
        compiler_params=pt.CompilerParams(num_warps=4))(lat.reshape(-1), lon.reshape(-1), clock)
    return tuple(v.reshape(lat.shape) for v in result)


def _kernel(terrain, previous, csza_ref, azi_ref, constants, mask_out, height_out,
            *, ny, nx, steps, nested, block, interpret):
    lane = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = lane < nx * ny
    row, col = lane // nx, lane % nx
    F = jnp.float32
    rn = partial(_rn, interpret=interpret)
    one = jnp.ones((block,), jnp.float32)
    dx, dy, pi = (jnp.broadcast_to(constants[i], (block,)) for i in range(3))
    load = lambda ref, idx, mask=valid: pt.load(ref.at[idx], mask=mask, other=F(0))
    h = load(terrain, lane)
    old = load(previous, lane)
    border = (col < 2) | (col >= nx - 2) | (row < 2) | (row >= ny - 2)
    imported = valid & border & (old > h) if nested else jnp.zeros((block,), bool)
    h0 = jnp.where(imported, old, h)
    height = jnp.where(border, old, rn("sub", h, one * F(.001))) if nested else rn("sub", h, one * F(.001))
    csza, azi = load(csza_ref, lane), load(azi_ref, lane)
    daylight = csza >= F(.01)
    north = (azi > F(1.75) * pi) | (azi < F(.25) * pi)
    east = (~north) & (azi < F(.75) * pi)
    south = (~north) & (~east) & (azi < F(1.25) * pi)
    along_y = north | south
    direction = jnp.where(north | east, one, -one)
    tangent = jnp.where(along_y, jnp.tan(azi), -jnp.tan(rn("add", pi / F(2), azi)))
    sun_tan = jnp.tan(jnp.arcsin(csza))
    x, y = (col + 1).astype(jnp.float32), (row + 1).astype(jnp.float32)

    def local_at(i, j, active):
        # Physical two-point halos repeat raw terrain; only in-domain boundary
        # mass points get the WRF toposhad_init height raise.
        ii, jj = jnp.clip(i, 1, nx) - 1, jnp.clip(j, 1, ny) - 1
        index = jj * nx + ii
        base = load(terrain, index, active)
        if nested:
            value = load(previous, index, active)
            inside = (i >= 1) & (i <= nx) & (j >= 1) & (j <= ny)
            edge = (ii < 2) | (ii >= nx - 2) | (jj < 2) | (jj >= ny - 2)
            base = jnp.where(inside & edge & (value > base), value, base)
        return base

    def keep(carry):
        step, _, _, active = carry
        return (step <= steps) & (jnp.max(active.astype(jnp.int32)) > 0)

    def scan(carry):
        step, mask, ht, active = carry
        delta = step.astype(jnp.float32) * direction
        ri = rn("add", x, jnp.where(along_y, rn("mul", delta, tangent), delta))
        rj = rn("add", y, jnp.where(along_y, delta, rn("mul", delta, tangent)))
        i1, j1 = jnp.trunc(ri).astype(jnp.int32), jnp.trunc(rj).astype(jnp.int32)
        i2 = i1 + along_y.astype(jnp.int32)
        j2 = j1 + (~along_y).astype(jnp.int32)
        active = active & (i1 > -2) & (i2 < nx + 3) & (j1 > -2) & (j2 < ny + 3)
        w = jnp.where(along_y, rn("sub", ri, i1.astype(jnp.float32)), rn("sub", rj, j1.astype(jnp.float32)))
        h1, h2 = local_at(i1, j1, active), local_at(i2, j2, active)
        ddx, ddy = rn("mul", dx, rn("sub", ri, x)), rn("mul", dy, rn("sub", rj, y))
        distance = jnp.sqrt(rn("add", rn("mul", ddx, ddx), rn("mul", ddy, ddy)))
        sample = rn("add", rn("mul", w, h2), rn("mul", rn("sub", one, w), h1))
        elevation = jnp.arctan(rn("div", rn("sub", sample, h0), distance))
        hit = active & (jnp.sin(elevation) >= csza)
        projected = rn("add", h0, rn("mul", distance, rn("sub", jnp.tan(elevation), sun_tan)))
        return step + 1, mask | hit, jnp.where(hit, jnp.maximum(ht, projected), ht), active

    _, mask, height, _ = jax.lax.while_loop(keep, scan,
        (jnp.int32(1), imported, height, valid & daylight))
    pt.store(mask_out.at[lane], jnp.where(daylight, mask, False).astype(jnp.int32), mask=valid)
    pt.store(height_out.at[lane], jnp.where(daylight, height, rn("sub", h0, one * F(.001))), mask=valid)


def topographic_shadow_native(terrain_height_m, previous_height_m, coszen, solar_azimuth_rad,
                              *, dx_m, dy_m, shadow_length_m, nested, interpret=False):
    """Return mask/height; one program lane per mass point, FP32 and 128 lanes."""
    terrain = jnp.asarray(terrain_height_m, jnp.float32)
    ny, nx = terrain.shape
    arrays = [jnp.asarray(v, jnp.float32).reshape(-1) for v in
              (terrain, previous_height_m, coszen, solar_azimuth_rad)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("topographic shadow native inputs must share the mass-grid shape")
    constants = jnp.asarray([dx_m, dy_m, jnp.pi], jnp.float32)
    kernel = partial(_kernel, ny=ny, nx=nx, steps=int(float(shadow_length_m) / float(dx_m) + 1),
                     nested=bool(nested), block=128, interpret=bool(interpret))
    result = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct((ny * nx,), jnp.int32),
                                              jax.ShapeDtypeStruct((ny * nx,), jnp.float32)),
        grid=((ny * nx + 127) // 128,), name="wrf_topographic_shadow", interpret=bool(interpret),
        compiler_params=pt.CompilerParams(num_warps=4))(*arrays, constants)
    return tuple(x.reshape(ny, nx) for x in result)


__all__ = ["topographic_shadow_native", "topographic_solar_geometry_native"]
