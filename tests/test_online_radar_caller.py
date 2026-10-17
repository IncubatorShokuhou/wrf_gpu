"""Radar caller cadence/boundaries and actual nested publication.

Numerical expectations come from the independently compiled pristine-WRF
fixtures, never from a second invocation of the port. The caller runs its
actual radar implementation under JIT; no radar numerical path is mocked.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.diagnostics.online import radar_from_context
from gpuwrf.diagnostics.thompson_radar import ThompsonRadarInputs
from test_initial_history_wrf_t0 import _writer, _wrfinput
from test_m7_netcdf_writer import synthetic_case, writer_authority


PROOFS = Path(__file__).resolve().parents[1] / "proofs/u2_diag"


@dataclass(frozen=True)
class OracleCase:
    context: ThompsonRadarInputs
    expected: np.ndarray
    previous: np.ndarray
    tolerance: float


def _oracle_case(name, *, ny, nx, repeats=1, field):
    path = PROOFS / f"{name}.npz"
    manifest = json.loads(path.with_name(f"{name}_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(path) as data:
        inputs = np.tile(data["inputs"], (repeats, 1, 1))
        expected = np.tile(data[field], (repeats, 1))
    nz = inputs.shape[-1]
    inputs = inputs.reshape(ny, nx, 10, nz)
    context = ThompsonRadarInputs(*(jnp.asarray(inputs[:, :, index, :], jnp.float32)
                                    for index in (0, 2, 3, 4, 5, 6, 8, 9)))
    expected = np.moveaxis(expected.reshape(ny, nx, nz), -1, 0)
    # Every held value differs from the pristine field; wrong cadence or a
    # missing boundary mask cannot accidentally pass as a dry/no-echo sample.
    previous = (np.arange(expected.size, dtype=np.float32).reshape(expected.shape)
                / np.float32(32) + np.float32(100))
    assert np.isfinite(expected).all() and expected.max() < previous.min()
    return OracleCase(context, expected, previous, manifest["tolerance"]["max_absolute_dbz"])


@pytest.fixture(scope="module")
def full_case():
    return _oracle_case("radar_oracle", ny=5, nx=8, field="dbz")


@pytest.fixture(scope="module")
def bottom_case():
    # Repeating all eight independent bottom-active cases yields a genuine
    # interior and boundary ring for the specified-boundary caller test.
    return _oracle_case("radar_bottom_oracle", ny=4, nx=4, repeats=2, field="bottom")


def _caller(case, *, nwp, spec_zone=0, dt=18.):
    namelist = SimpleNamespace(mp_physics=8, nwp_diagnostics=int(nwp), do_radar_ref=1,
                              dt_s=dt, diagnostic_history_interval=60.)
    return jax.jit(lambda previous, step: radar_from_context(
        previous, case.context, namelist, step, spec_zone))


@pytest.mark.parametrize("nwp", [0, 1])
def test_history_alarm_calls_full_pristine_radar(full_case, nwp):
    actual = np.asarray(_caller(full_case, nwp=nwp)(jnp.asarray(full_case.previous), jnp.int32(200)))
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, full_case.expected, rtol=0, atol=full_case.tolerance)
    # A bottom-only accidental call would erase the independently active upper
    # levels; the fixture explicitly contains upper echoes above the writer floor.
    assert np.any(full_case.expected[1:] > -35.)


def test_nondivisible_history_alarm_calls_full_pristine_radar(full_case):
    # WRF's54s step crosses the one-hour alarm at3618s (step67), not3564s.
    caller = _caller(full_case, nwp=0, dt=54.)
    held = np.asarray(caller(jnp.asarray(full_case.previous), jnp.int32(66)))
    np.testing.assert_array_equal(held, full_case.previous)
    actual = np.asarray(caller(jnp.asarray(full_case.previous), jnp.int32(67)))
    np.testing.assert_allclose(actual, full_case.expected, rtol=0, atol=full_case.tolerance)


def test_ordinary_nwp_calls_pristine_bottom_only(bottom_case):
    caller = _caller(bottom_case, nwp=1)
    for step in (199, 201):
        actual = np.asarray(caller(jnp.asarray(bottom_case.previous), jnp.int32(step)))
        np.testing.assert_allclose(actual, bottom_case.expected, rtol=0, atol=bottom_case.tolerance)
        np.testing.assert_array_equal(actual[1:], np.float32(-35.))
    assert np.any(bottom_case.expected[0] > 0.)


def test_ordinary_without_nwp_preserves_held_volume(full_case):
    caller = _caller(full_case, nwp=0)
    for step in (1, 199, 201, 403):
        actual = np.asarray(caller(jnp.asarray(full_case.previous), jnp.int32(step)))
        assert actual.tobytes() == full_case.previous.tobytes()


@pytest.mark.parametrize("mode", ["full", "bottom"])
def test_specified_boundary_retains_prior_radar(full_case, bottom_case, mode):
    case, step = (full_case, 200) if mode == "full" else (bottom_case, 199)
    actual = np.asarray(_caller(case, nwp=1, spec_zone=1)(jnp.asarray(case.previous), jnp.int32(step)))
    inside = np.zeros(case.previous.shape[-2:], dtype=bool)
    inside[1:-1, 1:-1] = True
    np.testing.assert_array_equal(actual[:, ~inside], case.previous[:, ~inside])
    np.testing.assert_allclose(actual[:, inside], case.expected[:, inside], rtol=0, atol=case.tolerance)


@pytest.mark.parametrize("own_step", [0, 200])
def test_actual_nested_writer_radar_without_nwp_publishes_exact_fields(tmp_path, full_case, own_step):
    # Use the real nested callback, payload builder and NetCDF writer. The
    # standard helper builds authenticated metadata and isolated WRF input;
    # extend its geometry to fit the original44-level reflectivity fixture.
    writer, _ = _writer(tmp_path, lambda *a, **kwargs: {})
    nz, ny, nx = full_case.expected.shape
    state, grid, namelist = synthetic_case(nx=nx, ny=ny, nz=nz)
    namelist = dict(namelist, physics={"mp_physics": 8, "do_radar_ref": 1},
                    time_control={"nwp_diagnostics": 0, "history_interval": 60.})
    writer.bundles["d01"] = SimpleNamespace(namelist=namelist, grid=grid)
    writer.domain_authorities["d01"] = writer_authority(grid, "d01")
    _wrfinput(tmp_path / "wrfinput_d01", ny, nx)
    radar = full_case.expected.copy()
    output = writer("d01", own_step, SimpleNamespace(state=state, nwp_diagnostics=None,
                                                     radar_reflectivity=radar))
    schema = json.loads((PROOFS / "cpu_reference_manifest.json").read_text())["frames"][0]["fields"]
    with Dataset(output["wrfout"]) as ds:
        assert "REFD" not in ds.variables  # Separate AFWA product is disabled.
        for name, dimensions in (
                ("REFL_10CM", ("Time", "bottom_top", "south_north", "west_east")),
                ("REFD_MAX", ("Time", "south_north", "west_east"))):
            variable = ds[name]
            assert variable.dimensions == dimensions
            assert np.dtype(variable.dtype) == np.dtype(schema[name]["dtype"]) == np.dtype("float32")
            assert variable.units == schema[name]["units"]
        actual = np.asarray(ds["REFL_10CM"][0])
        assert actual.tobytes() == radar.tobytes()
        np.testing.assert_array_equal(ds["REFD_MAX"][0], np.zeros((ny, nx), dtype=np.float32))
        assert radar.max() > 0.  # Zero REFD_MAX must not be a reflectivity alias.
