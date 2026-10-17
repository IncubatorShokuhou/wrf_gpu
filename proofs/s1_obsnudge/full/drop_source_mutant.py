"""Pytest plugin: delete the actual production observation source merge."""
def pytest_sessionstart(session):
    from gpuwrf.runtime import operational_mode as om
    original=om._source_leaf_dry_tendencies
    def drop(*args,**kwargs):
        kwargs['obs_tendf']=None
        return original(*args,**kwargs)
    om._source_leaf_dry_tendencies=drop
