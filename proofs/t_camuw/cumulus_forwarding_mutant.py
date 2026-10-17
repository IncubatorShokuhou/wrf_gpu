"""pytest plugin: erase the actual adapter's opt-in TP/KP diagnostics."""


def pytest_configure(config):
    import jax.numpy as jnp
    from gpuwrf.coupling import scan_adapters as adapters
    original = adapters.camuw_pbl_adapter

    def dropped_diagnostics(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("return_cumulus_inputs", False):
            tp, kp = result[-1]
            return result[:-1] + ((jnp.zeros_like(tp), jnp.zeros_like(kp)),)
        return result

    adapters.camuw_pbl_adapter = dropped_diagnostics
