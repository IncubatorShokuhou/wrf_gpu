"""WRF first_rk_step_part2 stochastic forcing at native C-grid extents.

The source uses the mass at (i,j), including for U/V; it does not average
mass onto the faces. The outer mass ghost is the retained edge halo.
"""
import jax.numpy as jnp

from gpuwrf.physics.stochastic_driver import field


def scalar_sources(theta, vapor, carry, mu, mub, metrics):
    nz = theta.shape[0]
    t = field(carry, "T", levels=nz)
    if t is not None:
        c1 = jnp.asarray(metrics.c1h, theta.dtype)[:, None, None]
        c2 = jnp.asarray(metrics.c2h, theta.dtype)[:, None, None]
        mass = c1*jnp.asarray(mu,theta.dtype)+(c1*jnp.asarray(mub,theta.dtype)+c2)
        theta = theta+jnp.asarray(t,theta.dtype)*mass
    p = field(carry, "P", levels=nz)
    if p is not None:
        factor = theta.dtype.type(1)+jnp.asarray(p,theta.dtype)
        theta, vapor = theta*factor, vapor*factor
    return theta, vapor


def momentum_sources(u, v, carry, mu, mub, metrics, nz):
    ny, nx = mu.shape
    if carry.patterns[0] is not None:
        dtype = jnp.result_type(mu, jnp.float32) if u is None else u.dtype
        c1 = jnp.asarray(metrics.c1h,dtype)[:,None,None]
        c2 = jnp.asarray(metrics.c2h,dtype)[:,None,None]
        # WRF arrays have a mass halo at the high U/V face, as does the
        # dycore's nonperiodic native mass coupling (flux_advection.py).
        # module_initialize_real.F:3847/3863 explicitly fills MUB/MU_2 this way.
        mass = c1*jnp.asarray(mu,dtype)+(c1*jnp.asarray(mub,dtype)+c2)
        umass = jnp.pad(mass,((0,0),(0,0),(0,1)),mode="edge")
        vmass = jnp.pad(mass,((0,0),(0,1),(0,0)),mode="edge")
        up = field(carry,"W",levels=nz,mass_only=False)[:,:ny,:nx+1]
        vp = field(carry,"W",levels=nz,mass_only=False,component=1)[:,:ny+1,:nx]
        du, dv = up*umass, vp*vmass
        u = du if u is None else u+du
        v = dv if v is None else v+dv
    p = field(carry,"P",levels=nz,mass_only=False)
    if p is not None:
        if u is not None:
            u = u*(u.dtype.type(1)+jnp.asarray(p[:,:ny,:nx+1],u.dtype))
        if v is not None:
            v = v*(v.dtype.type(1)+jnp.asarray(p[:,:ny+1,:nx],v.dtype))
    return u,v


def vapor_source(vapor, carry):
    p = field(carry,"P",levels=vapor.shape[0])
    return vapor if p is None else vapor*(vapor.dtype.type(1)+jnp.asarray(p,vapor.dtype))
