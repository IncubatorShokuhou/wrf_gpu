"""Deterministic old-OSAS draws: Threefry2x32-20, key=(950040, domain).

The counter is (one-based own step, global row-major column). Keys are uint32[2]
restart leaves; the existing restart clock supplies the step, so no hidden RNG
state evolves. This fixed algorithm is independent of JAX's configurable RNG.
Uniform draws use 53 bits, in [0,1), as WRF's DOUBLE XKT2.

WRF draws from the system clock; gpuwrf is deterministic by design.
"""
import jax.numpy as jnp

DEFAULT_SEED = 950040


def initial_osas_key(domain=1):
    if int(domain) < 1 or int(domain) > 0xffffffff:
        raise ValueError('OSAS domain must be a positive uint32 identifier')
    return jnp.asarray([DEFAULT_SEED,int(domain)],jnp.uint32)


def threefry2x32(key,step,column):
    """Random123 Threefry2x32, 20 rounds; all integer arithmetic wraps uint32."""
    key=jnp.asarray(key,jnp.uint32)
    ks=(key[0],key[1],key[0]^key[1]^jnp.uint32(0x1BD11BDA))
    x=jnp.asarray(step,jnp.uint32)+ks[0]
    y=jnp.asarray(column,jnp.uint32)+ks[1]
    rotations=(13,15,26,6,17,29,16,24)
    for round_index in range(20):
        rotation=jnp.uint32(rotations[round_index%8])
        x=x+y
        y=((y<<rotation)|(y>>(jnp.uint32(32)-rotation)))^x
        if (round_index+1)%4==0:
            injection=(round_index+1)//4
            x=x+ks[injection%3]
            y=y+ks[(injection+1)%3]+jnp.uint32(injection)
    return x,y


def osas_draws(key,step,column):
    x,y=threefry2x32(key,step,column)
    return (x.astype(jnp.float64)*jnp.asarray(2.**-32,jnp.float64)
            +(y>>jnp.uint32(11)).astype(jnp.float64)*jnp.asarray(2.**-53,jnp.float64))
