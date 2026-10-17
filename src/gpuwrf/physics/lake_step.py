"""Full WRF Lake/LakeMain column timestep with the REAL32 caller seam."""
from typing import NamedTuple
import jax
from jax import numpy as jnp

from .lake_fluxes import LakeForcing,surface_fluxes,TFRZ,HVAP,HSUB,real
from .lake_thermal import thermal_step
from .lake_hydrology import hydrology_step


class LakeAtmosphere(NamedTuple):
    temperature: jax.Array
    pbot: jax.Array
    psfc: jax.Array
    dz: jax.Array
    qv: jax.Array
    u: jax.Array
    v: jax.Array
    glw: jax.Array
    emissivity: jax.Array
    rainbl: jax.Array
    dt: jax.Array
    swdown: jax.Array
    albedo: jax.Array
    latitude: jax.Array


class LakeSurface(NamedTuple):
    hfx: jax.Array
    lh: jax.Array
    grdflx: jax.Array
    tsk: jax.Array
    qfx: jax.Array
    t2: jax.Array
    th2: jax.Array
    q2: jax.Array
    albedo: jax.Array


class LakeStepResult(NamedTuple):
    columns: object
    surface: LakeSurface
    energy_residual: jax.Array


def lake_step(columns,atmosphere:LakeAtmosphere):
    """Lake water/soil/snow timestep; batch dimensions precede vertical axes.

    The caller selects lake columns before this function. All saved payload
    and surface outputs are REAL32. WRF's r8 internals stay inside this call;
    energy_residual is the pre-correction DOUBLE diagnostic in W/m².
    """
    with jax.enable_x64():
        a = jax.tree.map(lambda v:jnp.asarray(v,jnp.float32),atmosphere)
        f = LakeForcing(a.temperature.astype(jnp.float64),a.pbot.astype(jnp.float64),a.psfc.astype(jnp.float64),
                         (jnp.float32(.5)*a.dz).astype(jnp.float64),
                         (a.qv/(jnp.float32(1.)+a.qv)).astype(jnp.float64),
                         a.u.astype(jnp.float64),a.v.astype(jnp.float64),(a.glw*a.emissivity).astype(jnp.float64),
                         (a.rainbl/a.dt).astype(jnp.float64),(a.swdown*(jnp.float32(1.)-a.albedo)).astype(jnp.float64),
                         a.latitude.astype(jnp.float64)*jnp.pi/180,a.dt.astype(jnp.float64))
        s = jax.tree.map(lambda v:jnp.asarray(v,jnp.float64),columns)
        flux = surface_fluxes(s,f)
        s,flux,fracold,snomelt,imelt,residual = thermal_step(s,f,flux)
        s = hydrology_step(s,f,flux,fracold,snomelt,imelt)
        f32 = lambda x:jnp.asarray(x,jnp.float32)
        t2 = f32(flux.t2)
        rcp = jnp.float32(287.)/(jnp.float32(7.)*jnp.float32(287.)/jnp.float32(2.))
        th2 = t2*(jnp.float32(1.e5)/a.psfc)**rcp
        albedo = real(.6)*s.lake_icefrac3d[...,0]+(1.-s.lake_icefrac3d[...,0])*real(.08)
        qfx = flux.latent/jnp.where(flux.ground_temperature >= TFRZ,HVAP,HSUB)
        # Lake writes GRDFLX from eflx_gnet (surface interface), not the
        # total soil-ground flux that includes penetrating solar radiation.
        surface = LakeSurface(f32(flux.sensible),f32(flux.latent),f32(flux.gnet),f32(flux.ground_temperature),
                               f32(qfx),t2,th2,f32(flux.q2),f32(albedo))
        return LakeStepResult(jax.tree.map(f32,s),surface,residual)
