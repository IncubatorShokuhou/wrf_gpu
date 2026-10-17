"""Original WRF SPP operators for MYNN-EDMF and its surface layer."""
import jax.numpy as jnp


def diffusion(dfm,dfh,pattern,zw):
    f = dfm.dtype.type
    decay = jnp.maximum(jnp.exp(-jnp.maximum(zw-f(8000),f(0))/f(2000)),f(.001))
    return dfm+dfm*pattern*f(1.5)*decay,dfh+dfh*pattern*f(1.5)*decay


def total_water(qw,pattern):
    return qw+qw*qw.dtype.type(.5)*pattern


def entrainment(ent,pattern):
    return ent*(ent.dtype.type(1)-pattern)


def momentum_roughness(znt, pattern):
    """module_sf_mynn.F:665/715: local ZNTstoch, not prognostic ZNT."""
    return jnp.maximum(znt+znt*pattern, znt.dtype.type(1.e-6))


def thermal_roughness(land, water_unclipped, pattern):
    """Zilitinkevich 1995 and Fairall 2003, with source-ordered bounds."""
    f = land.dtype.type
    land = jnp.maximum(land+land*f(.5)*pattern, f(.0001))
    water = water_unclipped+water_unclipped*f(.5)*pattern
    return land, jnp.maximum(jnp.minimum(water, f(1.e-4)), f(2.e-9))
