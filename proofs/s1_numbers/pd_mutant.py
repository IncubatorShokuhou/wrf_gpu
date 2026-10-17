"""Pytest plugin deleting only the repaired families' forced-native PD route."""
def pytest_collection_modifyitems(session, config, items):
    from gpuwrf.runtime import operational_mode as op
    original = op.advect_moisture_scalars
    def legacy(*args, **kwargs):
        kwargs['force_native_pd'] = False
        return original(*args, **kwargs)
    op.advect_moisture_scalars = legacy
