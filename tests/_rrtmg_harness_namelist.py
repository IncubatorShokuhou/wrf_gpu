"""Run RRTMG under the WRF namelist of the tier-1 harness fixtures (cloud overlap cldovrlp=1).

The tier-1 RRTMG fixtures (gpuwrf.validation.tier1_rrtmg, rrtmg_intermediate_oracles) are real WRF
RRTMG_SWRAD/RRTMG_LWRAD calls from scripts/wrf_rrtmg_harness.f90 with cldovrlp=1 (:818) and
icld_lw=1 (:387): McICA RANDOM overlap. The product selects the overlap from the namelist
(io.namelist_check.apply_cldovrlp: cldovrlp=1 pins GPUWRF_RRTMG_MAXRAND=0 before radiation is
imported); the release default MAXRAND=1 implements WRF's Registry default cldovrlp=2, which has
its own WRF oracles (tests/v025/b_rad/test_mcica_maxrand*.py). In one test process the import-time
flag is already resolved, so pin the module constant the product consults and drop every compiled
trace on entry and exit (E161: traces capture it).
"""
from __future__ import annotations

import jax
import pytest


@pytest.fixture
def harness_cldovrlp1(monkeypatch):
    from gpuwrf.kernels import rad_mcica
    jax.clear_caches()
    monkeypatch.setattr(rad_mcica, "_MAXRAND", False)
    yield
    jax.clear_caches()
