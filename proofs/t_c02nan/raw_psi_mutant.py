"""Deletion control: restore the incorrect raw neutral PSI for resistance users."""
def pytest_configure(config):
    from gpuwrf.coupling import scan_adapters as adapters
    import jax.numpy as jnp
    original = adapters._pbl_surface_forcing
    def raw_psi(state, grid, *, surface_forcing=None, use_resistances=False, ysu_surface=None):
        result = original(state, grid, surface_forcing=surface_forcing,
                          use_resistances=use_resistances, ysu_surface=ysu_surface)
        if (surface_forcing is not None and use_resistances) or ysu_surface is not None:
            result = dict(result, psim=jnp.zeros_like(result['psim']), psih=jnp.zeros_like(result['psih']))
        return result
    adapters._pbl_surface_forcing = raw_psi
