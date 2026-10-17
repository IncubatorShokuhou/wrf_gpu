"""Original WRF icloud_bl selection and cal_cldfra1 gates for CAM-UW."""
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics.cloud_fraction import wrf_resolved_cloud_fraction
from gpuwrf.runtime import operational_mode as om
from test_v022_camuw_pbl import _grid, _namelist, _state


ORACLE = Path(__file__).resolve().parents[1] / "proofs/t_camuw/cloud_oracle.json"


def test_cloud_oracle_is_packaged_with_independent_active_regimes():
    data = json.loads(ORACLE.read_text())
    assert "verbatim original WRF" in data["reference"]
    assert len(data["source_sha256"]) == len(data["subroutine_sha256"]) == 64
    assert data["rows"] == 520 and len(data["cases"]) == 40
    assert {c["branch"] for c in data["cases"]} == {1, 2, 3}
    assert {c["input"][0] for c in data["cases"]} == {1, 3, 8, 50, 51}


@pytest.mark.parametrize("mp", [1, 3, 8, 50, 51])
def test_resolved_cloud_fraction_matches_original_wrf(mp):
    cases = [c for c in json.loads(ORACLE.read_text())["cases"] if c["input"][0] == mp]
    a = np.asarray([c["input"][1:] for c in cases], np.float32)
    t, p, qv, qc, qi, qs = a.T
    got = np.asarray(jax.jit(lambda *args: wrf_resolved_cloud_fraction(*args, mp_physics=mp))(
        qv, qc, qi, qs, t, p))
    expected = np.asarray([c["fraction"] for c in cases], np.float32)
    bound = np.asarray([c["input_ulp_bound"] for c in cases])
    assert np.isfinite(got).all() and got.dtype == np.float32
    # Independently measured response of the pristine routine to one ULP
    # in each input bounds CPU contraction/intrinsic implementation differences.
    assert np.all(np.abs(got.astype(np.float64) - expected) <= bound), (got, expected, bound)
    assert {c["branch"] for c in cases} == {1, 2, 3}
    assert np.any((expected > 0) & (expected < 1))


def test_non_mynn_resolved_clouds_survive_radiation_input_assembly(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SGS_CLOUD", "1")
    grid = _grid(nz=4)
    state = _state(grid).replace(cldfra_bl=jnp.zeros((4, 3, 3)), qc_bl=jnp.zeros((4, 3, 3)),
                                 qi_bl=jnp.zeros((4, 3, 3)))
    # Supersaturated cloudy columns: original cal_cldfra1 branch 2 is exactly1.
    state = state.replace(qv=jnp.full_like(state.qv, .03), qc=jnp.full_like(state.qc, 1e-4))
    clear_sw, clear_lw, *_ = pc._rrtmg_column_inputs(
        state, grid, time_utc="2023-01-15T04:00:00Z", lead_seconds=3600., use_mp_re=0)
    sw, lw, *_ = pc._rrtmg_column_inputs(
        state, grid, time_utc="2023-01-15T04:00:00Z", lead_seconds=3600., use_mp_re=0,
        use_mynn_sgs_cloud=False, cloud_mp_physics=8)
    np.testing.assert_array_equal(clear_sw.cloud_fraction, 0.)
    np.testing.assert_array_equal(clear_lw.cloud_fraction, 0.)
    np.testing.assert_array_equal(sw.cloud_fraction, 1.)
    np.testing.assert_array_equal(lw.cloud_fraction, 1.)
    np.testing.assert_array_equal(sw.qc, clear_sw.qc)
    np.testing.assert_array_equal(lw.qc, clear_lw.qc)


def test_operational_rrtmg_driver_selects_original_wrf_cloud_policy(monkeypatch):
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=9, use_noahmp=False, mp_physics=8)
    carry = om._initial_carry_for_run(_state(grid), nml)
    calls = []

    def radiation(*args, **kwargs):
        calls.append(kwargs)
        return carry.rthraten, carry.radiation_diagnostics

    monkeypatch.setattr(om, "rrtmg_theta_tendency", radiation)
    om._refresh_rrtmg_driver(carry, nml, 3600., True, None)
    assert calls[-1]["use_mynn_sgs_cloud"] is False
    assert calls[-1]["cloud_mp_physics"] == 8
    om._refresh_rrtmg_driver(carry, dataclasses.replace(nml, bl_pbl_physics=5), 3600., True, None)
    assert "use_mynn_sgs_cloud" not in calls[-1]  # release call signature unchanged
