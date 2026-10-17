"""WRF sqrt(N*Z) wrapper representation with gradual REAL4 underflow.

Native CUDA uses explicit non-FTZ REAL instructions. The CPU backend emulates
the REAL4 intermediate in a binary64 container (like other CPU RN emulators),
including rounding to the subnormal 2^-149 lattice before division/sqrt. No
additional DOUBLE is traced in the native column or stored in State.
"""
import jax
import jax.numpy as jnp

from .process_support import _r, _rounded_real, NATIVE_REAL


def _cpu_product(a, b):
    # Binary64 represents the exact product of two binary32 significands.
    # Keep the rounded subnormal in the container until the next operation:
    # casting it to binary32 here would enable CPU DAZ/FTZ on that operand.
    with jax.enable_x64(True):
        exact = a.astype(jnp.float64)*b.astype(jnp.float64)
        normal = jax.lax.reduce_precision(exact, exponent_bits=11, mantissa_bits=23)
        normal = jnp.where(jnp.abs(normal)>jnp.float64(jnp.finfo(jnp.float32).max),
                           jnp.copysign(jnp.float64(jnp.inf),normal),normal)
        subnormal = jnp.rint(exact*jnp.float64(2.**149))*jnp.float64(2.**-149)
        return jnp.where(jnp.abs(exact)<jnp.float64(2.**-126), subnormal, normal)


def _cpu_cast(value):
    # Return subnormal REAL bits without a hardware narrowing conversion that
    # would flush them. Any overflow/NaN remains visible in the normal path.
    with jax.enable_x64(True):
        normal=value.astype(jnp.float32)
        magnitude=jnp.rint(jnp.abs(value)*jnp.float64(2.**149)).astype(jnp.uint32)
        bits=magnitude | (jnp.signbit(value).astype(jnp.uint32)<<jnp.uint32(31))
        subnormal=jax.lax.bitcast_convert_type(bits,jnp.float32)
        return jnp.where(jnp.abs(value)<jnp.float64(2.**-126),subnormal,normal)


def decode(qzi, number):
    """module_mp_p3.F:830-834, square rounded BEFORE division."""
    qzi, number = _r(qzi), _r(number)
    active = number >= _r(1.e-14)
    divisor = jnp.where(active, number, _r(1))
    if NATIVE_REAL.get():
        value = _rounded_real('div', _rounded_real('mul', qzi, qzi), divisor)
    else:
        with jax.enable_x64(True):
            value = _cpu_cast(_cpu_product(qzi,qzi)/divisor.astype(jnp.float64))
    return jnp.where(active,value,_r(0))


def encode(z, number):
    """module_mp_p3.F:916-920, product rounded BEFORE square root."""
    z, number = _r(z), _r(number)
    active = (number >= _r(1.e-14)) & (z >= _r(1.e-30))
    a,b = jnp.where(active,z,_r(0)),jnp.where(active,number,_r(0))
    if NATIVE_REAL.get():
        from jax.experimental.pallas import triton as plt
        product = _rounded_real('mul',a,b)
        value = plt.elementwise_inline_asm('sqrt.rn.f32 $0, $1;',args=(product,),
            constraints='=f,f',pack=1,
            result_shape_dtypes=(jax.ShapeDtypeStruct(product.shape,jnp.float32),))[0]
    else:
        with jax.enable_x64(True):
            value = _cpu_cast(jnp.sqrt(_cpu_product(a,b)))
    return jnp.where(active,value,_r(0))
