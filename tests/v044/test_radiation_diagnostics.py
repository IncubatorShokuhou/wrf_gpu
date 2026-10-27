"""V044 opt-in radiation history: WRF Registry, pristine RE01 flux/rate savepoints, E78/E200."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).parents[1]),
               str(Path(__file__).parents[1] / "v025/fid_q2")]
from test_v022_camuw_pbl import _grid, _namelist, _state
from rrtmg_mp_re_fixture import load, HEATING_BOUND
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.diagnostics import radiation as rd
from gpuwrf.io.namelist_binding import bound_options, namelist_honour_report
from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState
from gpuwrf.runtime import operational_mode as om


def test_rad_diag_binding_static_key_and_refusals():
    raw = {"gpuwrf": {"rad_diag": 1}, "physics": {"mp_physics": [8],
           "ra_sw_physics": [4], "ra_lw_physics": [4], "cu_physics": [0]}}
    assert bound_options(raw, "d03")["rad_diag"] == 1
    assert not namelist_honour_report(raw, ("d01", "d02", "d03"))[0]
    off = _namelist(_grid())
    on = replace(off, rad_diag=1)
    assert jax.tree.structure(off) == jax.tree.structure(replace(off, rad_diag=0))
    assert jax.tree.structure(off) != jax.tree.structure(on)
    assert jax.tree.unflatten(*reversed(jax.tree.flatten(on))).rad_diag == 1
    for value in (-1, 2, 1.5):
        with pytest.raises(ValueError, match="rad_diag"):
            replace(off, rad_diag=value)
    with pytest.raises(ValueError, match="RRTMG"):
        replace(on, ra_lw_physics=1)
    raw["physics"]["ra_lw_physics"] = [1]
    assert any("RRTMG" in e for e in namelist_honour_report(raw, ("d01",))[0])


def test_registry_metadata():
    text = Path("data/wrf_pristine/WRF/Registry/Registry.EM_COMMON").read_text()
    for name in ("QI_BL", *rd.HISTORY_NAMES[:4]):
        spec = WRFOUT_VARIABLE_SPECS[name]
        row = next(row for row in text.splitlines() if f'"{name}"' in row)
        assert f'"{spec.description}"' in row
        assert f'"{spec.units}"' in row
    for name in rd.HISTORY_NAMES[4:]:
        assert "gpuwrf diagnostic, no WRF equivalent" in WRFOUT_VARIABLE_SPECS[name].description


def test_optional_initial_carry_and_radiation_hold(monkeypatch):
    grid = _grid()
    off = _namelist(grid, use_noahmp=False)
    state = _state(grid)
    assert om._initial_carry_for_run(state, off).radiation_diagnostics.rad_diag is None
    on = replace(off, rad_diag=1)
    carry = om._initial_carry_for_run(state, on)
    assert len(rd.history_fields(carry.radiation_diagnostics)) == 8
    assert all(a.dtype == jnp.float32 for a in carry.radiation_diagnostics.rad_diag)
    calls = []

    def radiation(*args, **kwargs):
        calls.append(kwargs)
        held = rd.RadiationHistory(*(a + i + 1 for i, a in enumerate(carry.radiation_diagnostics.rad_diag)))
        return carry.rthraten, carry.radiation_diagnostics._replace(rad_diag=held)

    monkeypatch.setattr(om, "rrtmg_theta_tendency", radiation)
    updated = om._refresh_rrtmg_driver(carry, on, 3600., True, None)
    assert calls[-1]["with_rad_diag"] is True
    held = om._refresh_rrtmg_driver(updated, on, 3618., False, None)
    assert len(calls) == 1
    for a, b in zip(updated.radiation_diagnostics.rad_diag, held.radiation_diagnostics.rad_diag):
        np.testing.assert_array_equal(a, b)


def test_restart_refuses_old_missing_inactive_and_nonfinite_diagnostics():
    state = _state(_grid())
    held = rd.initial(state)
    diag = SimpleNamespace(rad_diag=held)
    rd.validate_restart(diag, {"rad_diag": 1}, 1, state.theta.shape)
    rd.validate_restart(None, {}, None)
    for d, config, version in ((None, {"rad_diag": 1}, None),
                               (diag, {}, 1), (diag, {"rad_diag": 1}, None)):
        with pytest.raises(ValueError, match="E78"):
            rd.validate_restart(d, config, version, state.theta.shape)
    for bad in (np.nan, np.inf, -np.inf):
        for i in range(8):
            values = list(held)
            values[i] = np.full(state.theta.shape, bad, np.float32)
            with pytest.raises(ValueError, match="nonfinite"):
                rd.validate_restart(SimpleNamespace(rad_diag=rd.RadiationHistory(*values)),
                                    {"rad_diag": 1}, 1, state.theta.shape)


def test_rate_export_against_pristine_rrtmg_all_and_clear_savepoints():
    """Independently saved WRF TTEN/PI3D and clear fluxes on real day/night columns.

    This checks the changed export/divergence, reusing the existing solver oracle
    for radiation itself. Frozen RE01 heating bounds (1e-4 + 5%) are unchanged.
    """
    fx = load()
    nz = fx["in_t"].shape[-1]
    # All 4 regimes are present in this real winter fixture; no new synthetic
    # hydrometeors or physics self-comparison is used as WRF evidence.
    assert np.any(fx["col_coszen"] > 0) and np.any(fx["col_coszen"] <= 0)
    assert np.any(fx["in_qi_rad"] > 0) and np.any(fx["in_qc_rad"] > 0)
    c = lambda name: jnp.asarray(fx[name][:, None, :], jnp.float64)
    common = dict(T=c("in_t"), p=c("in_p_hyd"), qv=c("in_qv"), qc=c("in_qc_rad"),
                  qi=c("in_qi_rad"), qs=c("in_qs"), qg=c("in_qg"), cloud_fraction=c("in_cldfra"),
                  dz=c("in_dz8w")[..., :nz], rho=c("in_rho"), pressure_interfaces=c("in_p_hyd_w"),
                  temperature_interfaces=c("in_t8w"))
    lw_state = RRTMGLWColumnState(**common, surface_temperature=jnp.asarray(fx["in_tsk"][:, None]),
                                 surface_emissivity=jnp.asarray(fx["in_emiss"][:, None]),
                                 top_pressure_pa=float(fx["in_p_top"]))
    sw_state = RRTMGSWColumnState(**common, surface_albedo=jnp.asarray(fx["in_albedo"][:, None]),
                                 coszen=jnp.asarray(fx["col_coszen"][:, None]))
    pi = jnp.asarray(np.moveaxis(fx["in_pi"][:, None, :], -1, 0), jnp.float64)
    state = SimpleNamespace(theta=jnp.zeros_like(pi), p=pc.P0_PA * pi ** (1 / pc.R_D_OVER_CP))
    lw = SimpleNamespace(heating_rate=c("A_hlw") * jnp.moveaxis(pi, 0, -1),
                         clear_flux_down=c("A_lwdnc"), clear_flux_up=c("A_lwupc"))
    sw = SimpleNamespace(heating_rate=c("A_hsw") * jnp.moveaxis(pi, 0, -1),
                         clear_flux_down=c("A_swdnc"), clear_flux_up=c("A_swupc"))
    result = pc._rrtmg_history_from_results(state, sw_state, lw_state, sw, lw, with_clear_sky=True)
    for index, (name, actual) in enumerate(zip(("hlw", "hsw", "hlwc", "hswc"), result[:4])):
        expected = fx[f"A_{name}"]
        actual = np.moveaxis(np.asarray(actual), 0, -1)[:, 0, :]
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        assert np.all(np.abs(actual - expected) <= HEATING_BOUND[0] + HEATING_BOUND[1] * np.abs(expected)), name
        # Export is an algebraic operation, with no solver uncertainty. All-sky
        # rates must survive the axis/Exner round trip exactly in WRF REAL.
        if index < 2:
            np.testing.assert_array_equal(actual, expected)
        else:
            # Clear fluxes were serialized in REAL. Bound cancellation using
            # 16 eps * sum of the four interface flux magnitudes, g/(cp dp pi),
            # plus 1e-4 relative for retained/WRF constant and rounding order.
            # This bound is defined from reference operands, never candidate
            # differences; it rejects replacing each active clear rate by zero.
            down, up = (fx[f"A_{prefix}{suffix}"] for prefix in
                        (("lwdn", "lwup") if name == "hlwc" else ("swdn", "swup")) for suffix in ("c",))
            scale = np.abs(down[:, 1:nz+1]) + np.abs(down[:, :nz]) + np.abs(up[:, 1:nz+1]) + np.abs(up[:, :nz])
            dp = fx["in_p_hyd_w"][:, :-1] - fx["in_p_hyd_w"][:, 1:]
            bound = 16 * np.finfo(np.float32).eps * scale * 9.81 / (1004 * dp * fx["in_pi"]) + 1.e-4 * np.abs(expected)
            assert np.all(np.abs(actual - expected) <= bound), name
            assert np.any(np.abs(expected) > bound), f"zero-{name} mutant would survive"
    for field, key in zip(result[4:], ("in_qc_rad", "in_qi_rad", "in_qs", "in_qg")):
        np.testing.assert_array_equal(np.moveaxis(np.asarray(field), 0, -1)[:, 0], fx[key].astype(np.float32))
