"""Deliberately delete nest-start inheritance; only a genuine assertion kills it."""
def pytest_configure(config):
    import jax.numpy as jnp
    import gpuwrf.nesting.scalar_initialization as initialization
    original = initialization.interp_sint_full

    def deleted(*args, **kwargs):
        return jnp.zeros_like(original(*args, **kwargs))

    initialization.interp_sint_full = deleted
