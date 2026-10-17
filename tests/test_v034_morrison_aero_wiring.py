"""v0.3.4 O1: mp_physics=40 (Morrison-aerosol, aercu_opt=0) operational wiring.

CPU-only. Checks that mp=40 is genuinely its own scheme on the operational
path (not mp=10, not fail-closed), runs on a REAL (fp32) carry, writes the
WRF qnc scalar (Nc) exactly as WRF's wrapper does, and that aercu_opt>0 is
refused instead of being silently ignored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.coupling.scan_adapters import (  # noqa: E402
    MP_SCAN_ADAPTERS,
    P0_PA,
    R_D_OVER_CP,
    morrison_adapter,
    morrison_aero_adapter,
)
from gpuwrf.io.namelist_check import (  # noqa: E402
    UnsupportedSchemeError,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.physics import morrison_constants as C  # noqa: E402
from gpuwrf.runtime.operational_mode import _resolve_operational_suite  # noqa: E402
from tests.test_v013_operational_smoke import _base_state, _grid, _namelist  # noqa: E402


def _real_state(state):
    """The release carry is WRF REAL: cast every float64 leaf to float32."""

    def cast(x):
        return x.astype(jnp.float32) if getattr(x, "dtype", None) == jnp.float64 else x

    return jax.tree_util.tree_map(cast, state)


def test_mp40_is_scan_wired_and_dispatched():
    assert MP_SCAN_ADAPTERS[40] is morrison_aero_adapter
    support = classify_scheme("mp_physics", 40)
    # v0.4.0 o2-rootbdy: the root Ns/Ng/Nc take WRF flow_dep_bdy (v0.3.4 o1-nlbind had refused
    # mp=40 for the missing record) -> IMPLEMENTED, caveats in the note.
    assert support.status is SupportStatus.IMPLEMENTED
    assert "Ns/Ng/Nc" in support.reason
    # Pristine physics_init excludes mp40 at supported aercu_opt=0.
    assert "aercu_opt=0" in support.reason
    grid = _grid()
    _resolve_operational_suite(_namelist(grid, mp_physics=40, bl_pbl_physics=0,
                                         sf_sfclay_physics=0, cu_physics=0))


def test_mp40_adapter_real_carry_writes_constant_nc():
    grid = _grid()
    state = _real_state(_base_state(grid))
    out = morrison_aero_adapter(state, 20.0, grid)
    for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng", "Nc"):
        a = getattr(out, name)
        assert a.dtype == jnp.float32, name
        assert np.all(np.isfinite(np.asarray(a))), name
    assert not np.allclose(np.asarray(out.qv), np.asarray(state.qv))
    # WRF aercu_opt=0: NC(i,k,j) = NDCNST*1e6/RHO everywhere (aero.F l.4997,
    # l.1142) with RHO = P/(R*T) of the ENTRY temperature T = TH*PII.
    pii = (np.maximum(np.asarray(state.p, np.float64), 1.0) / P0_PA) ** R_D_OVER_CP
    # WRF phy_prep supplies DRY theta, while State carries theta_m.
    theta_dry = np.asarray(state.theta, np.float64) / (
        1. + (461.6 / 287.) * np.asarray(state.qv, np.float64))
    rho = np.asarray(state.p, np.float64) / (C.R * theta_dry * pii)
    np.testing.assert_allclose(np.asarray(out.Nc, np.float64), C.NDCNST * 1.0e6 / rho, rtol=2e-6)
    # seeded Nc input is dead (wrapper placeholder nc1d=0): the input is replaced.
    assert not np.allclose(np.asarray(out.Nc), np.asarray(state.Nc))


def test_mp40_differs_from_mp10_on_same_state():
    """Deletion sensitivity: routing mp=40 through the mp=10 kernel is caught."""
    grid = _grid()
    state = _base_state(grid)
    a = morrison_aero_adapter(state, 20.0, grid)
    b = morrison_adapter(state, 20.0, grid)
    diff = max(float(np.max(np.abs(np.asarray(getattr(a, n)) - np.asarray(getattr(b, n)))))
               / max(float(np.max(np.abs(np.asarray(getattr(b, n))))), 1e-30)
               for n in ("qi", "qs", "Ni", "Ns"))
    assert diff > 1.0e-3, diff


@pytest.mark.parametrize("aercu_opt", [1, 2])
def test_aercu_opt_positive_is_refused(aercu_opt):
    cfg = {"physics": {"mp_physics": [40], "cu_physics": [0], "aercu_opt": aercu_opt}}
    with pytest.raises(UnsupportedSchemeError, match="aercu_opt"):
        validate_operational_namelist(cfg)


def test_aercu_opt_zero_mp40_is_accepted():
    # v0.4.0 (o2-rootbdy): the namelist/aercu_opt=0 gate accepts mp=40 and so does the
    # operational CLI gate (root Ns/Ng/Nc take WRF flow_dep_bdy); nested runs are refused.
    from gpuwrf.io.namelist_check import (NotOperationallyWiredError, refuse_single_domain_mp_nested,
                                          validate_namelist)

    validate_namelist({"physics": {"mp_physics": [40], "cu_physics": [0], "aercu_opt": 0}})
    validate_operational_namelist({"physics": {"mp_physics": [40]}})
    refuse_single_domain_mp_nested({"physics": {"mp_physics": [40]}}, 1)
    with pytest.raises(NotOperationallyWiredError, match="single domains only"):
        refuse_single_domain_mp_nested({"physics": {"mp_physics": [40, 40]}}, 2)
