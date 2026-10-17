"""Shared Davis roughness formula, literal WRF module_sf_mynn.F:1399–1407."""
import jax.numpy as jnp


def davis(ust):
    zw = jnp.minimum((ust / 1.06) ** 0.3, 1.0)
    zn1 = 0.011 * ust * ust / 9.81 + 1.59e-5
    zn2 = 10.0 * jnp.exp(-9.5 * ust ** (-.3333)) + 0.11 * 1.5e-5 / jnp.maximum(ust, .01)
    return jnp.clip((1.0 - zw) * zn1 + zw * zn2, 1.27e-7, 2.85e-3)
