"""External pytest plugin: genuine scientific mutants, no source edits.

APP_MUTANT=wind|solar|visibility|fwi python -m pytest -p mutant ...
A kill requires an assertion failure and zero setup/collection errors.
"""
import os


def pytest_collection_modifyitems(session,config,items):
    name=os.environ["APP_MUTANT"]
    if name=="wind":
        import gpuwrf.post.apps.wind as m
        original=m.rotor_weights
        def equal_area(*a,**k):
            import numpy as np
            weights=original(*a,**k)
            return np.full_like(weights,1/weights.size)
        m.rotor_weights=equal_area
    elif name=="solar":
        import gpuwrf.post.apps.solar as m
        original=m.pvwatts_dc
        def no_temperature(poa,temp,**kwargs):
            return original(poa,25,**kwargs)
        m.pvwatts_dc=no_temperature
    elif name=="visibility":
        import gpuwrf.post.apps.hazards as m
        original=m.visibility
        def no_ice(t,p,qv,qc,qr,qi,qs):
            return original(t,p,qv,qc,qr,0,qs)
        m.visibility=no_ice
    elif name=="fwi":
        import gpuwrf.post.apps.fire as m
        m._FFMC_MOISTURE=147.2
    else:
        raise ValueError(name)
    # Tests import function names at collection; rebind to the changed function.
    for item in items:
        mod=item.module
        for key in ("rotor_weights","pvwatts_dc","visibility","daily_fwi"):
            if hasattr(m,key) and hasattr(mod,key):
                setattr(mod,key,getattr(m,key))
