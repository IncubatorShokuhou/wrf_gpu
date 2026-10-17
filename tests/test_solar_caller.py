"""Solar caller regression seams; original Fortran numerics live in the oracle.

The existing radiation and land solvers are stubbed here to expose argument
lifetimes. These tests make no coupled trajectory or GPU fidelity claim.
"""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as radiation
from gpuwrf.diagnostics import solar
from gpuwrf.io.namelist_binding import bind_namelist_options, namelist_honour_report
from gpuwrf.io.namelist_check import validate_operational_namelist
from gpuwrf.io.scheme_catalog import classify_control, SupportStatus
from gpuwrf.runtime import operational_mode as om
from test_solar_restart import case


class HeldRadiation(NamedTuple):
    surface_albedo: object
    coszen: object
    swdown: object
    swnorm: object
    glw: object


def other_source_group(group):
    fixture = json.loads((Path(__file__).resolve().parents[1] / "proofs/u2_diag/solar_oracle.json").read_text())
    for record in reversed([values for case in fixture["cases"] for values in case["expected"]]):
        values = np.asarray(record, np.float32).reshape(12, 4, 4)
        if values[9].max() > 0 and not np.array_equal(values[7], group.fluxes.swdown):
            return solar.SolarCarry(solar.SolarInterpolationState(*map(jnp.asarray, values[:7])),
                                    solar.SolarSurfaceFluxes(*map(jnp.asarray, values[7:])))
    raise AssertionError("independent oracle needs two distinct active surface snapshots")


@pytest.mark.parametrize("run_radiation,first", [(False, False), (True, False), (True, True)])
def test_actual_physics_caller_solar_lifetimes(case, monkeypatch, run_radiation, first):
    grid, nl, carry = case
    before = carry.solar_interpolation
    fresh = other_source_group(before)
    full = lambda value: jnp.full(carry.state.t_skin.shape, value, jnp.float32)
    diag = HeldRadiation(full(.2), full(.45), full(999.), full(888.), full(321.))
    land = SimpleNamespace(albedo=full(.3), emiss=full(.95))
    # A land model only runs behind a surface layer (module_surface_driver.F:1490 returns for
    # sf=0); the Noah-MP step is replaced below, so sf=5 adds no extra surface call here.
    nl = replace(nl, time_utc=datetime(2023, 1, 15), mp_physics=0, bl_pbl_physics=0,
                 sf_sfclay_physics=5, cu_physics=0, gwd_opt=0, rad_rk_tendf=0,
                 radiation_static=SimpleNamespace(xlat_deg=full(47.), xlong_deg=full(8.)))
    carry = carry.replace(noahmp_land=land, radiation_diagnostics=diag,
                          rthraten=jnp.full_like(carry.state.theta, .001))
    calls = {key: [] for key in ("geometry", "radiation", "fit", "interpolate", "land")}
    monkeypatch.setattr(radiation, "_SOLAR_JULIAN_WRF", True)
    from gpuwrf.kernels import dyn_carry_fp32
    monkeypatch.setattr(dyn_carry_fp32, "real_all_enabled", lambda: False)

    def current_coszen(c, n, lead, clock):
        calls["geometry"].append((lead, clock))
        return full(.6)
    monkeypatch.setattr(om, "_solar_current_coszen", current_coszen)

    def solve(state, grid, **kwargs):
        calls["radiation"].append(kwargs)
        assert kwargs["_with_solar"] is True
        return carry.rthraten, diag, radiation.RRTMGSurfaceSolarInputs(full(123.), full(77.), full(256.))
    monkeypatch.setattr(om, "rrtmg_theta_tendency", solve)

    def fit(parameters, midpoint, current, direct, total, *, first_step):
        calls["fit"].append(first_step)
        np.testing.assert_array_equal(midpoint, full(.45))
        np.testing.assert_array_equal(current, full(.6))
        np.testing.assert_array_equal(direct, full(123.))
        # Independent literal WRF REAL reconstruction, distinct from SWDNB.
        expected = np.float32(256.) / (np.float32(1.) - np.float32(.2))
        np.testing.assert_array_equal(total, full(expected))
        return fresh.parameters
    monkeypatch.setattr(solar, "update_parameters", fit)

    def interpolate(parameters, current, albedo):
        calls["interpolate"].append(parameters)
        np.testing.assert_array_equal(current, full(.6))
        np.testing.assert_array_equal(albedo, full(.3))  # current land, not radiation-call .2
        return fresh.fluxes
    monkeypatch.setattr(solar, "interpolate_surface", interpolate)

    def land_step(state, old, static, dt, **kwargs):
        forcing = kwargs["radiation"]
        calls["land"].append(forcing)
        np.testing.assert_array_equal(forcing.soldn, fresh.fluxes.swdown)
        np.testing.assert_array_equal(forcing.lwdn, diag.glw)
        np.testing.assert_array_equal(forcing.cosz, diag.coszen)  # held midpoint
        return state, old
    monkeypatch.setattr(om, "noahmp_surface_step", land_step)
    result = om._physics_step_forcing(carry, nl, 43218., run_radiation=run_radiation,
                                      first_timestep=first)
    assert len(calls["geometry"]) == len(calls["interpolate"]) == len(calls["land"]) == 1
    assert len(calls["radiation"]) == len(calls["fit"]) == int(run_radiation)
    if run_radiation:
        assert calls["fit"] == [first]
        assert calls["radiation"][0]["lead_seconds"] == 43218.
        assert calls["radiation"][0]["solar_lead_seconds"] == 43518.  # nominal600, not33*18
    expected = fresh.parameters if run_radiation else before.parameters
    for actual, wanted in zip(result.carry.solar_interpolation.parameters, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(result.carry.solar_interpolation.fluxes, fresh.fluxes, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(result.carry.radiation_diagnostics.swdown, diag.swdown)
    np.testing.assert_array_equal(result.carry.rthraten, carry.rthraten)


@pytest.mark.parametrize("ozone", [False, True])
def test_held_refresh_preserves_solar_and_optional_ozone(case, monkeypatch, ozone):
    _, nl, carry = case
    monkeypatch.setattr(om, "rrtmg_theta_tendency", lambda *args, **kwargs: pytest.fail("held step solved radiation"))
    if ozone:
        carry = carry.replace(o3rad=jnp.zeros_like(carry.state.qv))
    result = om._refresh_rrtmg_driver(carry, nl, 18., False, None)
    for actual, expected in zip(result.solar_interpolation.parameters, carry.solar_interpolation.parameters, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(result.solar_interpolation.fluxes, carry.solar_interpolation.fluxes, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert result.radiation_diagnostics is carry.radiation_diagnostics
    assert result.o3rad is carry.o3rad


def test_solar_binding_static_marker_and_catalog(case):
    _, nl, _ = case
    off = replace(nl, swint_opt=0)
    raw = {"physics": {"swint_opt": 1, "ra_sw_physics": 4, "ra_lw_physics": 4, "sf_surface_physics": 4}}
    import jax
    for domain in ("d01", "d02"):
        active = bind_namelist_options(off, raw, domain)
        assert active.swint_opt == 1
        leaves, tree = jax.tree.flatten(active)
        assert jax.tree.unflatten(tree, leaves).swint_opt == 1
        assert tree != jax.tree.structure(off)
    assert classify_control("swint_opt", 1).status is SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified; GPU/coupled-forecast qualification pending" in classify_control("swint_opt", 1).reason
    assert not namelist_honour_report(raw, ("d01", "d02"))[0]
    assert namelist_honour_report(raw, ("d01",), replay=True)[0]
    validate_operational_namelist(raw)


@pytest.mark.parametrize("key,value", [("ra_sw_physics", 3), ("ra_lw_physics", 1), ("sf_surface_physics", 3),
                                       ("solar_opt", 1), ("topo_shading", 1), ("slope_rad", 1),
                                       ("sf_urban_physics", 1), ("sf_lake_physics", 1)])
def test_solar_selection_refused_by_both_host_gates(key, value):
    raw = {"physics": {"swint_opt": 1, "ra_sw_physics": 4, "ra_lw_physics": 4,
                       "sf_surface_physics": 4, key: value}}
    assert namelist_honour_report(raw, ("d01",))[0]
    with pytest.raises(ValueError, match="Solar|swint"):
        validate_operational_namelist(raw)


@pytest.mark.parametrize("damage", ["land", "radiation", "date", "coordinates", "legacy_geometry"])
def test_runtime_refuses_unqualified_solar_caller(case, monkeypatch, damage):
    _, nl, carry = case
    monkeypatch.setattr(radiation, "_SOLAR_JULIAN_WRF", True)
    nl = replace(nl, time_utc=datetime(2023, 1, 15), radiation_static=object())
    carry = carry.replace(noahmp_land=object(), radiation_diagnostics=object())
    if damage in ("land", "radiation"):
        field = {"land": "noahmp_land", "radiation": "radiation_diagnostics"}[damage]
        carry = carry.replace(**{field: None})
    elif damage == "date":
        nl = replace(nl, time_utc=None)
    elif damage == "coordinates":
        nl = replace(nl, radiation_static=None)
    else:
        monkeypatch.setattr(radiation, "_SOLAR_JULIAN_WRF", False)
    with pytest.raises(ValueError, match="swint_opt=1"):
        om._validate_solar_runtime(nl, carry)


def test_solar_seed_precedes_pipeline_land_attachment(case):
    _, nl, carry = case
    seeded = om._initial_carry_for_run(carry.state, replace(nl, cu_physics=0))
    assert seeded.noahmp_land is None
    leaves = jax.tree.leaves(seeded.solar_interpolation)
    assert len(leaves) == 12
    for leaf in leaves:
        assert leaf.dtype == np.dtype("float32") and leaf.shape == (4, 4)
        np.testing.assert_array_equal(leaf, 0.)
    # Initialization can seed before the loader attaches land. Actual physics
    # must still refuse an incomplete bundle before consuming those zeros.
    with pytest.raises(ValueError, match="active Noah-MP"):
        om._validate_solar_runtime(nl, seeded)
