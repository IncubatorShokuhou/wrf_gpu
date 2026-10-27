"""Frozen pristine-WRF toposhad trajectory on real WN3 0227 d03 (critic F2).

The fixture holds verbatim pristine REAL answers (proofs/w44_topo/
freeze_shadow_regression.py): four calls of a 30-minute cold-zero trajectory,
sunrise, sunset, the first night call and the next sunrise, the last two with
a held nest-edge height above terrain. No port output was used to build it.
Each production layer is checked against it: toposhad clock, CPU reference,
nested coupler, CUDA dispatch wiring and the Pallas kernel (interpret mode).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.backends import pallas as pallas_backend
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.kernels import topographic_shadow as kernel
from gpuwrf.physics.topographic_shading import (
    topographic_shadow,
    toposhad_solar_azimuth,
    toposhad_solar_geometry,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "shadow_d03_regression.npz"
META = json.loads(FIXTURE.with_suffix(".json").read_text())
DATA = np.load(FIXTURE)
CALLS = [row["key"] for row in META["rows"]]
HELD = [row["key"] for row in META["rows"] if row["edge_above_terrain"] > 0]
MINUTE = {row["key"]: row["minute"] for row in META["rows"]}


def _operands(key):
    return dict(old=DATA[key + "_old"], csza=DATA[key + "_csza"], hrang=DATA[key + "_hrang"],
                declin=np.float32(DATA[key + "_declin"]))


def _check(key, mask, height):
    """Exact mask, frozen height bound; rejects nonfinite candidates (E200)."""
    mask, height = np.asarray(mask), np.asarray(height, np.float64)
    assert mask.shape == height.shape == DATA["HGT"].shape
    assert np.isfinite(height).all(), f"{key}: nonfinite HGT_SHAD"
    misses = int(np.count_nonzero(mask != DATA[key + "_mask"]))
    assert misses == 0, f"{key}: {misses} SHADOWMASK misses"
    want = DATA[key + "_height"].astype(np.float64)
    tol = META["height_tolerance"]
    error = np.abs(height - want)
    failed = int(np.count_nonzero(~(error <= tol["atol"] + tol["rtol"] * np.abs(want))))
    assert failed == 0, f"{key}: {failed} HGT_SHAD failures, max {np.nanmax(error)} m"


def test_fixture_is_the_frozen_pristine_trajectory():
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == META["fixture_sha256"]
    assert CALLS == ["sunrise", "sunset", "night", "held_next_sunrise"]
    assert HELD == ["night", "held_next_sunrise"]
    assert META["mask_tolerance"] == "exact"
    assert int(DATA["night_mask"].sum()) == 0 < int(DATA["sunrise_mask"].sum())


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_comparator_rejects_nonfinite_candidate(bad):
    height = DATA["sunset_height"].copy()
    height[40, 50] = bad
    with pytest.raises(AssertionError, match="nonfinite"):
        _check("sunset", DATA["sunset_mask"], height)


@pytest.mark.parametrize("key", CALLS)
def test_toposhad_clock_matches_pristine_without_eot(key):
    o = _operands(key)
    csza, hrang = toposhad_solar_geometry(DATA["XLAT"], DATA["XLONG"], o["declin"],
                                          MINUTE[key], 0., META["radt_minutes"])
    np.testing.assert_allclose(np.asarray(csza), o["csza"], rtol=0, atol=META["clock_atol"])
    np.testing.assert_allclose(np.asarray(hrang), o["hrang"], rtol=0, atol=META["clock_atol"])


def _reference(key, geometry=None):
    o = _operands(key)
    if geometry is None:
        geometry = (o["csza"], o["hrang"])
    return topographic_shadow(
        DATA["HGT"], latitude_deg=DATA["XLAT"], longitude_deg=DATA["XLONG"],
        sina=DATA["SINALPHA"], cosa=DATA["COSALPHA"], declination_rad=o["declin"],
        xtime_minutes=0., gmt_hours=0., radt_minutes=0., dx_m=META["dx_m"], dy_m=META["dy_m"],
        shadow_length_m=META["shadow_length_m"], parent_shadow_height_m=o["old"],
        geometry_override=geometry)


@pytest.mark.parametrize("key", CALLS)
def test_cpu_reference_operator_matches_pristine(key):
    _check(key, *_reference(key))


@pytest.mark.parametrize("key", CALLS)
def test_own_clock_and_operator_chain_matches_pristine(key):
    o = _operands(key)
    geometry = toposhad_solar_geometry(DATA["XLAT"], DATA["XLONG"], o["declin"],
                                       MINUTE[key], 0., META["radt_minutes"])
    _check(key, *_reference(key, geometry))


@pytest.mark.parametrize("key", CALLS)
def test_pallas_kernel_interpret_matches_pristine(key):
    o = _operands(key)
    azimuth = toposhad_solar_azimuth(DATA["XLAT"], o["csza"], o["declin"], o["hrang"],
                                     DATA["SINALPHA"], DATA["COSALPHA"])
    mask, height = kernel.topographic_shadow_native(
        DATA["HGT"], o["old"], o["csza"], azimuth, dx_m=META["dx_m"], dy_m=META["dy_m"],
        shadow_length_m=META["shadow_length_m"], nested=True, interpret=True)
    _check(key, mask, height)


def _coupler(key):
    o = _operands(key)
    static = SimpleNamespace(
        xlat_deg=jnp.asarray(DATA["XLAT"]), terrain_height_m=jnp.asarray(DATA["HGT"]),
        sina=jnp.asarray(DATA["SINALPHA"]), cosa=jnp.asarray(DATA["COSALPHA"]),
        slope_rad=jnp.zeros_like(DATA["HGT"]), slope_azimuth_rad=jnp.zeros_like(DATA["HGT"]))
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=META["dx_m"], dy_m=META["dy_m"]))
    shade = pc.SolarGeometry(jnp.asarray(o["csza"]), o["declin"], jnp.asarray(o["hrang"]))
    eot = pc.SolarGeometry(jnp.asarray(o["csza"]) + 1, o["declin"], jnp.asarray(o["hrang"]) + 1)
    topography = pc._rrtmg_topography_state(
        static, grid, eot, slope_rad=1, topo_shading=1,
        shadow_length_m=float(META["shadow_length_m"]), shadow_geometry=shade,
        previous_shadow_height_m=jnp.asarray(o["old"]), nested_shadow=True)
    assert topography.shadow_height_m.dtype == jnp.float32
    return topography.shadow_mask, topography.shadow_height_m


@pytest.mark.parametrize("key", CALLS)
def test_nested_coupler_uses_own_held_height(key):
    _check(key, *_coupler(key))


@pytest.mark.parametrize("key", HELD)
def test_cuda_dispatch_wiring_reaches_kernel_with_held_height(key, monkeypatch):
    calls = []
    native = kernel.topographic_shadow_native

    def interpreted(*args, **kwargs):
        calls.append(kwargs["nested"])
        return native(*args, **kwargs, interpret=True)

    monkeypatch.setattr(pallas_backend, "cuda_native", lambda: True)
    monkeypatch.setattr(kernel, "topographic_shadow_native", interpreted)
    _check(key, *_coupler(key))
    assert calls == [True]
