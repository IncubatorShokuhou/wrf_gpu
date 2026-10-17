"""Delete the prognostic TKE update in-process; require a numeric assertion kill."""
import pytest


class DeleteTkeUpdate:
    def pytest_sessionstart(self, session):
        from gpuwrf.runtime import les3d_km3
        les3d_km3.les_tke_stage_update = lambda state, origin, **kw: state


raise SystemExit(pytest.main([
    'tests/dynamics/test_les3d_smagorinsky.py::test_km2_tke_reaches_every_operational_rk_stage[True]',
    '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/o2-km2/pytest', '-q',
], plugins=[DeleteTkeUpdate()]))
