"""Genuine output-deletion mutant on the production FDDA driver."""
def pytest_configure(config):
    from gpuwrf.assimilation import fdda_grid as fg
    import jax.numpy as jnp
    original = fg.fdda_driver_tendencies
    def delete_history(*a, **kw):
        result = original(*a, **kw)
        if isinstance(result, fg.GridFddaHistoryCarry):
            result = result.replace(hfx_fdda=jnp.zeros_like(result.hfx_fdda))
        return result
    fg.fdda_driver_tendencies = delete_history
