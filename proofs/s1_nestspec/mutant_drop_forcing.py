"""Pytest plugin: restore the missing extra-species force-down loop (E39/E202)."""


def pytest_configure(config):
    from gpuwrf.nesting import boundary_construction
    boundary_construction.EXTRA_SCALAR_BOUNDARY_SPECIES = ()
