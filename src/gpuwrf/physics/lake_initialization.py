"""Standalone WRF ``lakeini`` port (EM_CORE=1, ten water/soil layers).

The operational timestep is implemented separately in ``lake_step``.
Arrays have column dimensions followed by a layer axis. Snow/soil slots run
from WRF -4 through 10, interfaces -5 through 10, water/soil properties 1..10.
Caller/carry are REAL32. Explicit DOUBLE arithmetic mirrors WRF's r8 islands,
including assignment back to REAL32 between thermal-property calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import numpy as jnp


@dataclass(frozen=True)
class LakeInitConfig:
    """Static WRF flags; ISWATER is the caller's land-use water category."""

    iswater: int = 17
    lakeflag: int = 1
    lake_depth_flag: int = 1
    use_lakedepth: int = 1
    lakedepth_default: float = 50.0
    lake_min_elev: float = 5.0
    xice_threshold: float = 0.5


class LakeInitInputs(NamedTuple):
    ivgtyp: jax.Array
    isltyp: jax.Array
    ht: jax.Array
    snow: jax.Array
    tsk: jax.Array
    lakemask: jax.Array
    xice: jax.Array
    xland: jax.Array
    lake_depth: jax.Array


class LakeColumnState(NamedTuple):
    lakedepth2d: jax.Array
    savedtke12d: jax.Array
    snowdp2d: jax.Array
    h2osno2d: jax.Array
    snl2d: jax.Array
    t_grnd2d: jax.Array
    t_lake3d: jax.Array
    lake_icefrac3d: jax.Array
    z_lake3d: jax.Array
    dz_lake3d: jax.Array
    t_soisno3d: jax.Array
    h2osoi_ice3d: jax.Array
    h2osoi_liq3d: jax.Array
    h2osoi_vol3d: jax.Array
    z3d: jax.Array
    dz3d: jax.Array
    zi3d: jax.Array
    watsat3d: jax.Array
    csol3d: jax.Array
    tkmg3d: jax.Array
    tkdry3d: jax.Array
    tksatu3d: jax.Array


class LakeInitialization(NamedTuple):
    columns: LakeColumnState
    ivgtyp: jax.Array
    isltyp: jax.Array
    xland: jax.Array
    xice: jax.Array
    lakemask: jax.Array
    lake: jax.Array
    lake_depth_flag: jax.Array


def empty_lake_columns(shape=(), *, fill=0.0):
    """Allocate a REAL32 seed, including WRF's non-lake/unused snow slots.

    WRF does not assign every output at non-lake points. Pass a caller's prior
    payload to ``lake_initial_state`` to preserve those slots; the default
    zero seed represents a fresh domain allocation, not a physical estimate.
    """
    sizes = (0,) * 6 + (10,) * 4 + (15,) * 6 + (16,) + (10,) * 5
    return LakeColumnState(*(jnp.full(tuple(shape) + ((n,) if n else ()), fill, jnp.float32)
                             for n in sizes))


def lake_initial_state(inputs: LakeInitInputs, *, config=LakeInitConfig(),
                       previous: LakeInitialization | None = None, restart=False):
    """Initialize WRF lake columns, preserving its literal assignment order.

    ``config`` and ``restart`` must be static when jitting. Restart returns
    the supplied payload unchanged, exactly as lakeini's early RETURN.
    Missing requested depth data is refused before initializing a domain.
    Input category validity and finite data belong to the caller's ingestion
    checks; this component adds no masks/clamps to WRF's physical equations.
    """
    if restart:
        if previous is None:
            raise ValueError("lakeini restart needs the existing lake payload")
        return previous
    if config.use_lakedepth == 1 and config.lake_depth_flag == 0:
        raise ValueError("WRF lakeini needs lake-depth information; rerun WPS or set use_lakedepth=0")
    with jax.enable_x64():
        return _initialize(inputs, config, previous)


def _initialize(inp, cfg, previous):
    f = lambda x: jnp.asarray(x, jnp.float32)
    d = lambda x: jnp.asarray(x, jnp.float64)
    snow, skin, height = f(inp.snow), f(inp.tsk), f(inp.ht)
    mask, xice, xland = f(inp.lakemask), f(inp.xice), f(inp.xland)
    veg, soil = jnp.asarray(inp.ivgtyp, jnp.int32), jnp.asarray(inp.isltyp, jnp.int32)
    old = empty_lake_columns(skin.shape) if previous is None else previous.columns
    # Classification and sea-ice conversion precede physical initialization.
    if cfg.lakeflag == 0:
        convert_ice = (height >= f(cfg.lake_min_elev)) & (xice > f(cfg.xice_threshold))
        veg = jnp.where(convert_ice, cfg.iswater, veg)
        xland = jnp.where(convert_ice, f(2.0), xland)
        xice = jnp.where(convert_ice, f(0.0), xice)
        lake = (veg == cfg.iswater) & (height >= f(cfg.lake_min_elev))
        mask = lake.astype(jnp.float32)
    else:
        lake = mask == f(1.0)
        convert_ice = lake & (xice > f(cfg.xice_threshold))
        veg = jnp.where(convert_ice, cfg.iswater, veg)
        xland = jnp.where(convert_ice, f(2.0), xland)
        xice = jnp.where(convert_ice, f(0.0), xice)
    # WRF sets these non-lake outputs to DEFVAL; others retain caller seeds.
    s = old._asdict()
    s["snowdp2d"], s["h2osno2d"] = snow * f(0.005), snow
    defval = f(-999.0)
    for name in ("lakedepth2d", "snl2d", "h2osoi_liq3d", "h2osoi_ice3d",
                 "t_soisno3d", "z3d", "dz3d", "t_lake3d", "lake_icefrac3d",
                 "z_lake3d", "dz_lake3d"):
        s[name] = jnp.full_like(s[name], defval)
    def put(name, value):
        value = f(value)
        select = lake.reshape(lake.shape + (1,) * (value.ndim - lake.ndim))
        s[name] = jnp.where(select, value, s[name])

    default = f(cfg.lakedepth_default)
    # SPVAL's initializer is an unsuffixed REAL literal even though the
    # parameter is declared r8 in WRF. Preserve the REAL32 representation.
    fallback = jnp.where(default > 0, default, f(1.e36))
    if cfg.lake_depth_flag == 1 and cfg.use_lakedepth != 0:
        input_depth = f(inp.lake_depth)
        depth = jnp.where(input_depth > 0, input_depth, fallback)
    else:
        depth = jnp.broadcast_to(fallback, skin.shape)
    standard = depth == f(1.e36)
    put("lakedepth2d", jnp.where(standard, f(1.0), depth))
    # Actual ten-layer source uses 1m standard depth, fixes first layer at
    # 0.1m, then scales remaining layers. Do not substitute commented 50m grid.
    zlak = d([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    ratio = d(jnp.where(standard, f(1.0), depth)) / (zlak[-1] + d(0.5) * d(0.1))
    dl = jnp.broadcast_to(f(d(0.1) * ratio[..., None]), skin.shape + (10,))
    dl = dl.at[..., 0].set(f(0.1))
    zl = f(zlak * ratio[..., None] + d(dl[..., :1]) * (d(1.0) - ratio[..., None]))
    zl = zl.at[..., 0].set(f(zlak[0]))
    # The sentinel branch copies all unscaled DOUBLE coordinates directly.
    zl = jnp.where(standard[..., None], f(zlak), zl)
    put("z_lake3d", zl)
    put("dz_lake3d", dl)

    # DATA statements initialize DOUBLE tables from default REAL literals.
    sand = d(f([92.,80.,66.,20.,5.,43.,60.,10.,32.,51.,6.,22.,39.7,0.,100.,54.,17.,100.,92.]))
    clay = d(f([3.,5.,10.,15.,5.,18.,27.,33.,33.,41.,47.,58.,14.7,0.,0.,8.5,54.,0.,3.]))
    isl = jnp.where(soil == 14, soil + 1, soil) - 1
    sa, cl = sand[isl], clay[isl]
    porosity = f(d(0.489) - d(0.00126) * sa)
    bulk = (d(1.0) - d(porosity)) * d(2.7e3)
    mineral = (d(8.80) * sa + d(2.92) * cl) / (sa + cl)
    tkmg = f(mineral ** (d(1.0) - d(porosity)))
    props = {"watsat3d": porosity,
             "tkmg3d": tkmg,
             "tksatu3d": f(d(tkmg) * d(0.57) ** d(porosity)),
             "tkdry3d": f((d(0.135) * bulk + d(64.7)) / (d(2.7e3) - d(0.947) * bulk)),
             "csol3d": f((d(2.128) * sa + d(2.385) * cl) / (sa + cl) * d(1.e6))}
    for name, value in props.items():
        put(name, jnp.broadcast_to(value[..., None], skin.shape + (10,)))

    # Lake temperatures are REAL operations; do not use DOUBLE here. WRF
    # overwrites the temporary soil temperature profile below with bottom TL.
    temp = jnp.where(zl <= f(50.0), skin[..., None] + (f(277.0) - skin[..., None]) / f(50.0) * zl, f(277.0))
    temp = temp.at[..., 0].set(skin)
    put("t_lake3d", temp)
    put("t_grnd2d", jnp.full_like(skin, 277.0))
    put("lake_icefrac3d", (d(temp) < d(273.16)).astype(jnp.float32))
    put("savedtke12d", jnp.full_like(skin, 0.6))

    levels = jnp.arange(1, 11, dtype=jnp.float64)
    zsoil = d(0.025) * (jnp.exp(d(0.5) * (levels - d(0.5))) - d(1.0))
    dsoil = jnp.concatenate((d(0.5) * (zsoil[:1] + zsoil[1:2]),
                             d(0.5) * (zsoil[2:] - zsoil[:-2]), zsoil[-1:] - zsoil[-2:-1]))
    zisoil = jnp.concatenate((d([0.0]), d(0.5) * (zsoil[:-1] + zsoil[1:]),
                              zsoil[-1:] + d(0.5) * dsoil[-1:]))
    # Snow-layer thickness assignment is a mixture of DOUBLE and REAL32
    # expressions. Keep every branch and round before later consumers.
    sd = s["snowdp2d"]
    zeros = jnp.zeros(skin.shape + (5,), jnp.float32)
    snl = jnp.zeros_like(skin)
    dsnow = zeros
    branches = (
        (0.01, 0.03, -1, (0., 0., 0., 0., sd)),
        (0.03, 0.04, -2, (0., 0., 0., f(d(sd)/d(2.0)), f(d(sd)/d(2.0)))),
        (0.04, 0.07, -2, (0., 0., 0., f(0.02), sd-f(0.02))),
        (0.07, 0.12, -3, (0., 0., f(0.02), f((d(sd)-d(0.02))/d(2.0)), f((d(sd)-d(0.02))/d(2.0)))),
        (0.12, 0.18, -3, (0., 0., f(0.02), f(0.05), sd-f(0.02)-f(0.05))),
        (0.18, 0.29, -4, (0., f(0.02), f(0.05), f(d(sd-f(0.02)-f(0.05))/d(2.0)), f(d(sd-f(0.02)-f(0.05))/d(2.0)))),
        (0.29, 0.41, -4, (0., f(0.02), f(0.05), f(0.11), sd-f(0.02)-f(0.05)-f(0.11))),
        (0.41, 0.64, -5, (f(0.02), f(0.05), f(0.11), f(d(sd-f(0.02)-f(0.05)-f(0.11))/d(2.0)), f(d(sd-f(0.02)-f(0.05)-f(0.11))/d(2.0)))),
        (0.64, float("inf"), -5, (f(0.02), f(0.05), f(0.11), f(0.23), sd-f(0.02)-f(0.05)-f(0.11)-f(0.23))),
    )
    for n, (lower, upper, count, thickness) in enumerate(branches):
        selected = ((d(sd) >= d(lower)) if n == 0 else (d(sd) > d(lower))) & (d(sd) <= d(upper))
        values = jnp.stack([jnp.broadcast_to(f(v), skin.shape) for v in thickness], axis=-1)
        dsnow = jnp.where(selected[..., None], values, dsnow)
        snl = jnp.where(selected, f(count), snl)
    put("snl2d", snl)
    put("dz3d", jnp.concatenate((dsnow, jnp.broadcast_to(f(dsoil), skin.shape + (10,))), axis=-1))
    # Build snow geometry downward from interface k=0. Each stored interface
    # is rounded to REAL before the next step, as in the source DO loop.
    zsnow, zisnow = zeros, jnp.zeros(skin.shape + (6,), jnp.float32)
    for idx in range(4, -1, -1):
        active = f(idx - 4) > snl
        center = f(d(zisnow[..., idx+1]) - d(0.5) * d(dsnow[..., idx]))
        interface = zisnow[..., idx+1] - dsnow[..., idx]
        zsnow = zsnow.at[..., idx].set(jnp.where(active, center, f(0.0)))
        zisnow = zisnow.at[..., idx].set(jnp.where(active, interface, f(0.0)))
    put("z3d", jnp.concatenate((zsnow, jnp.broadcast_to(f(zsoil), skin.shape + (10,))), axis=-1))
    put("zi3d", jnp.concatenate((zisnow[..., :5], jnp.broadcast_to(f(zisoil), skin.shape + (11,))), axis=-1))
    active_snow = jnp.arange(-4, 1) > snl[..., None]
    tsoisnow = jnp.where(active_snow, f(250.0), defval)
    put("t_soisno3d", jnp.concatenate((tsoisnow, jnp.broadcast_to(temp[..., -1:], skin.shape + (10,))), axis=-1))
    volume = jnp.concatenate((zeros, jnp.broadcast_to(porosity[..., None], skin.shape + (10,))), axis=-1)
    put("h2osoi_vol3d", volume)
    frozen_soil = d(temp[..., -1:]) <= d(273.16)
    water = d(f(dsoil)) * d(1000.0) * d(porosity[..., None])
    soil_ice = d(f(dsoil)) * d(917.0) * d(porosity[..., None])
    snow_ice = f(d(dsnow) * d(250.0))
    put("h2osoi_ice3d", jnp.concatenate((snow_ice, jnp.where(frozen_soil, f(soil_ice), f(0.0))), axis=-1))
    put("h2osoi_liq3d", jnp.concatenate((zeros, jnp.where(frozen_soil, f(0.0), f(water))), axis=-1))
    # lake_depth_flag is shared across columns; WRF clears it only upon
    # encountering a lake when use_lakedepth=0, not for an all-land tile.
    flag = jnp.asarray(cfg.lake_depth_flag, jnp.int32)
    if cfg.use_lakedepth == 0 and cfg.lake_depth_flag == 1:
        flag = jnp.where(jnp.any(lake), jnp.int32(0), flag)
    return LakeInitialization(LakeColumnState(**s), veg, soil, xland, xice, mask, lake, flag)
