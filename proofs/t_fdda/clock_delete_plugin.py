"""Restore State-only daily FDDA routing; require a genuine clock assertion."""
def pytest_configure(config):
    from gpuwrf.coupling import fdda_rk
    fdda_rk.grid_fdda_active = lambda namelist: False
