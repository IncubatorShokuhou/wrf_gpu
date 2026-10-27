"""Explicit pytest plugin: delete cross-call HGT_SHAD retention, not the test."""
def pytest_collection_modifyitems(session, config, items):
    from gpuwrf.runtime import operational_mode as op
    import jax.numpy as jnp
    original = op._shadow_held_kwargs
    def reset_height(namelist, diagnostics):
        result = original(namelist, diagnostics)
        if 'previous_shadow_height_m' in result:
            result['previous_shadow_height_m'] = jnp.zeros_like(result['previous_shadow_height_m'])
        return result
    op._shadow_held_kwargs = reset_height
