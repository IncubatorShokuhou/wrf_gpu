"""Delete the production history overlay for both output consumers."""
def pytest_configure(config):
    from gpuwrf.coupling import fdda_rk
    fdda_rk.fdda_history = lambda carry, diagnostics=None: diagnostics
