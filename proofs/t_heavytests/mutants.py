"""Deliberate physical/wiring regressions for the reduced CPU harnesses."""
import os
import pytest


@pytest.fixture(autouse=True)
def scientific_mutant(monkeypatch):
    name = os.environ.get('GPUWRF_HEAVY_MUTANT')
    if name == 'sms':
        import numpy as np
        from gpuwrf.dynamics import les3d_smagorinsky as sms
        original = sms.les3d_smagorinsky_memory
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            return result._replace(ru_tendf=result.ru_tendf + np.float32(1e8))
        monkeypatch.setattr(sms, 'les3d_smagorinsky_memory', corrupt)
    elif name == 'kf':
        from gpuwrf.coupling import kf_rk
        # Remove the WRF RK coupling branch despite the enabled namelist/env.
        monkeypatch.setattr(kf_rk, 'kf_tend_rk_enabled', lambda: False)
    else:
        raise ValueError(f'Unknown deliberate mutant: {name!r}')
