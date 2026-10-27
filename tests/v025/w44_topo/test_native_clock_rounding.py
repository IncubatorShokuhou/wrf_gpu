"""Window-A sunset operands: pristine WRF REAL, not a GPU regression oracle."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.backends import pallas as pl
from gpuwrf.kernels import topographic_shadow as kernel
from gpuwrf.physics.topographic_shading import toposhad_solar_geometry, toposhad_solar_geometry_from_seconds


# TP19 fixture f15fa10e..., d02_4_0, minute1110, first five failed cells.
# WRF oracle source4f0177189e..., built REAL -O2 -ffp-contract=off.
LAT = np.array([27.732891082763672, 27.760498046875, 27.787506103515625,
                27.787784576416016, 27.81479263305664], np.float32)
LON = np.array([-17.788970947265625, -17.728271484375, -17.728607177734375,
                -17.69805908203125, -17.698394775390625], np.float32)
DECLIN = np.float32(-0.14298494160175323)
COSZEN = np.array([0.03345738351345062, 0.03244917839765549, 0.03237041085958481,
                  0.031905487179756165, 0.0318269208073616], np.float32)
HRANG = np.array([1.4566694498062134, 1.4577291011810303, 1.4577231407165527,
                 1.458256483078003, 1.4582504034042358], np.float32)


def test_native_clock_pristine_sunset_operands():
    fn = jax.jit(lambda lat, lon: kernel.topographic_solar_geometry_native(
        lat, lon, DECLIN, np.float32(1110), np.float32(0), np.float32(30), interpret=True))
    csza, hrang = (np.asarray(v) for v in fn(LAT, LON))
    assert np.isfinite(csza).all() and np.isfinite(hrang).all()
    # RN-division failure is 6e-7 here. This pins the actual clock separately
    # from the unchanged .01m+2e-5*height scientific operator tolerance.
    np.testing.assert_allclose(csza, COSZEN, rtol=0, atol=3.e-8)
    np.testing.assert_array_equal(hrang, HRANG)


@pytest.mark.parametrize("lead_seconds,utc_minutes", [(1125 * 60, 0), (0, 1125)])
def test_native_clock_caller_units_pristine_sunset_operands(lead_seconds, utc_minutes):
    fn = jax.jit(lambda lat, lon: kernel.topographic_solar_geometry_native(
        lat, lon, DECLIN, np.float32(lead_seconds), np.float32(utc_minutes), np.float32(0),
        interpret=True, caller_units=True))
    csza, hrang = (np.asarray(v) for v in fn(LAT, LON))
    np.testing.assert_allclose(csza, COSZEN, rtol=0, atol=3.e-8)
    np.testing.assert_array_equal(hrang, HRANG)


def test_cuda_clock_caller_unit_dispatch(monkeypatch):
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return "native-coszen", "native-hrang"
    monkeypatch.setattr(pl, "cuda_native", lambda: True)
    monkeypatch.setattr(kernel, "topographic_solar_geometry_native", capture)
    assert toposhad_solar_geometry_from_seconds(LAT, LON, DECLIN, 28800., 450.) == ("native-coszen", "native-hrang")
    assert calls[0][0][3:] == (28800., 450., 0.)
    assert calls[0][1] == {"caller_units": True}


def test_cuda_clock_dispatch(monkeypatch):
    calls = []
    def capture(*args):
        calls.append(args)
        return "native-coszen", "native-hrang"
    monkeypatch.setattr(pl, "cuda_native", lambda: True)
    monkeypatch.setattr(kernel, "topographic_solar_geometry_native", capture)
    assert toposhad_solar_geometry(LAT, LON, DECLIN, 1110., 0., 30.) == ("native-coszen", "native-hrang")
    assert len(calls) == 1 and calls[0][3:] == (1110., 0., 30.)


@pytest.mark.parametrize("caller_units", [False, True])
def test_native_clock_cuda_lowering_uses_ieee_division(monkeypatch, caller_units):
    from jax._src.pallas.triton import lowering
    monkeypatch.setattr(pl, "cuda_native", lambda: True)
    shape = jax.ShapeDtypeStruct((5,), jnp.float32)
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    fn = lambda *args: kernel.topographic_solar_geometry_native(*args, caller_units=caller_units)
    jp = jax.make_jaxpr(fn)(
        shape, shape, scalar, scalar, scalar, scalar).jaxpr
    params = next(e.params for e in jp.eqns if e.primitive.name == "pallas_call")
    ir = str(lowering.lower_jaxpr_to_triton_module(
        params["jaxpr"], params["grid_mapping"], "cuda", 120).module)
    assert ir.count("div.rn.f32") == (4 if caller_units else 2)
    assert "div.full.f32" not in ir and "f64" not in ir
    assert "wrf_toposhad_solar" in ir


def test_mini_builder_with_real_gpu_caps(tmp_path):
    """Do not use smoke_caps: their unused 'new' key hid the window-A crash."""
    path = Path(__file__).resolve().parents[3] / "proofs/w44_topo/forecast_driver.py"
    spec = importlib.util.spec_from_file_location("topo_profile_builder", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    profile = dict(input_dir=str(tmp_path), hours=3, native_manifest=str(tmp_path / "manifest.json"),
                   phases=["native", "old"])
    caps = dict(native=50, old=120, old_post=30, e41=55, checker=0, e41_new=0)
    work = driver.build_work(dict(old_bundle=str(tmp_path / "bundle")), profile,
                             tmp_path / "tree", tmp_path / "tools", tmp_path / "output", caps, "16G")
    assert set(work) == {"native", "old"}
    assert not (tmp_path / "output/new_cache").exists()
    for command, env in work.values():
        assert env["JAX_PLATFORMS"] == "cuda"
        assert command[:3] == ["systemd-run", "--user", "--scope"]
