"""P3 foundation parity against compiled pristine WRF, not operational qualification."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.p3 import lookup as lut
from gpuwrf.physics.p3.properties import bulk_rime_density, compute_mu_3moment
from gpuwrf.physics.p3.tables import load_tables
from gpuwrf.physics.p3.qzi import decode, encode

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "proofs/s2_p3/fixtures"


def fixture(kind="2mom"):
    with np.load(FIXTURES / f"{kind}.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def check(actual, expected, *, rtol=4.e-6, atol=1.e-30):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all(), "nonfinite P3 candidate"
    assert np.isfinite(expected).all(), "nonfinite P3 WRF oracle"
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('kind', ['2mom','3mom'])
@pytest.mark.parametrize('compiled', [False,True])
def test_qzi_original_underflow_and_wrapper_thresholds(kind, compiled):
    data=fixture(kind)
    def run(x):
        return jnp.stack((decode(x[:,15],x[:,17]),encode(x[:,16],x[:,17])),axis=-1)
    actual=np.asarray((jax.jit(run) if compiled else run)(jnp.asarray(data['inputs'])))
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    assert actual.tobytes()==data['qzi'].tobytes()


def test_qzi_ieee_subnormal_output_and_overflow_remain_visible():
    # Exact powers of two: (2^-63)^2 / 2^10 = 2^-136, bit13 of REAL's
    # subnormal mantissa. These are independent IEEE/source constraints.
    value=np.asarray(jax.jit(decode)(jnp.float32(2.**-63),jnp.float32(2.**10)))
    assert int(value.view(np.uint32))==1<<13
    assert np.isinf(decode(jnp.float32(2.**64),jnp.float32(1)))
    assert np.isinf(encode(jnp.float32(2.**127),jnp.float32(2)))


def ice_coord(a):
    return lut.IceCoordinates(*(jnp.asarray(a[:, k], dtype=jnp.int32 if k < 3 else jnp.float32)
                                for k in range(6)))


def scalar_coord(a):
    return lut.Coordinate(jnp.asarray(a[:, 0], dtype=jnp.int32), jnp.asarray(a[:, 1]))


def packed(c):
    return jnp.stack(tuple(v.astype(jnp.float32) for v in c), axis=-1)


@pytest.mark.parametrize("compiled", [False, True])
def test_coordinates_original_wrf(compiled):
    data = fixture()
    x = jnp.asarray(data["inputs"])
    def run(x):
        return (packed(lut.ice_coordinates(x[:, 0], x[:, 1], x[:, 2], x[:, 3])),
                packed(lut.rain_coordinates(x[:, 4], x[:, 5])),
                packed(lut.mu_coordinates(x[:, 0], x[:, 6])))
    results = (jax.jit(run) if compiled else run)(x)
    for result, name, count in zip(results, ("ice_coords", "rain_coords", "mu_coords"), (3, 1, 1)):
        np.testing.assert_array_equal(np.asarray(result)[:, :count], data[name][:, :count])
        check(result, data[name], rtol=8.e-6, atol=2.e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_two_category_coordinates_original_wrf(compiled):
    data = fixture("ice_ice")
    x = jnp.asarray(data["inputs"])
    def run(x):
        a, b = lut.ice_pair_coordinates(x[:, 0], x[:, 1], x[:, 2], x[:, 10],
                                        x[:, 7], x[:, 8], x[:, 9], x[:, 11])
        return jnp.concatenate((packed(a), packed(b)), axis=-1)
    result = (jax.jit(run) if compiled else run)(x)
    for start in (0, 6):
        np.testing.assert_array_equal(np.asarray(result)[:, start:start+3], data["pair_coords"][:, start:start+3])
    check(result, data["pair_coords"], rtol=8.e-6, atol=2.e-6)


@pytest.mark.parametrize("kind", ["2mom", "3mom"])
@pytest.mark.parametrize("compiled", [False, True])
def test_all_ice_and_rain_channels_original_wrf(kind, compiled):
    data = fixture(kind)
    table = load_tables(kind)
    ice = ice_coord(data["ice_coords"])
    rain = scalar_coord(data["rain_coords"])
    mu = scalar_coord(data["mu_coords"]) if kind == "3mom" else None
    def run(itab, coll, ice, rain, mu):
        return (jnp.stack([lut.ice_lookup(itab, ice, k, mu) for k in range(1, itab.shape[-1]+1)], -1),
                jnp.stack([lut.rain_ice_lookup(coll, ice, rain, k, mu) for k in (1, 2)], -1))
    result = (jax.jit(run) if compiled else run)(jnp.asarray(table["ice"]), jnp.asarray(table["rain"]), ice, rain, mu)
    check(result[0], data["lookup"])
    check(result[1], data["collection"])


@pytest.mark.parametrize("compiled", [False, True])
def test_both_ice_ice_channels_original_wrf(compiled):
    data = fixture("ice_ice")
    table = load_tables("ice_ice")
    collector = ice_coord(data["pair_coords"][:, :6])
    collectee = ice_coord(data["pair_coords"][:, 6:])
    fn = jax.jit(lut.ice_ice_lookup) if compiled else lut.ice_ice_lookup
    for k, name in enumerate(("number", "mass")):
        check(fn(jnp.asarray(table[name]), collector, collectee), data["lookup"][:, k])


@pytest.mark.parametrize("compiled", [False, True])
def test_particle_properties_original_wrf(compiled):
    data = fixture()
    x = jnp.asarray(data["inputs"])
    def run(x):
        return (jnp.stack(bulk_rime_density(x[:, 0], x[:, 2], x[:, 10]), -1),
                compute_mu_3moment(x[:, 12], x[:, 13], x[:, 14]))
    rime, mu = (jax.jit(run) if compiled else run)(x)
    check(rime, data["bulk"], rtol=0, atol=0)
    check(mu, data["mu"], atol=2.e-5)


def test_invalid_three_moment_input_is_visible():
    assert np.isnan(compute_mu_3moment(1., 0., 1.))
    assert np.isnan(compute_mu_3moment(1., 1.e-21, 1.))
    for bad in (np.nan, np.inf, -np.inf):
        for index in range(4):
            args = [1., 1.e-5, 1.e-10, 20.]
            args[index] = bad
            assert np.isnan(compute_mu_3moment(*args)), (index, bad)


def test_wrong_table_variant_refused():
    c = ice_coord(fixture()["ice_coords"])
    with pytest.raises(ValueError, match="table rank"):
        lut.ice_lookup(load_tables("3mom")["ice"], c, 1)
    for channel in (0, 15, 1.5):
        with pytest.raises(ValueError, match="process channel"):
            lut.ice_lookup(load_tables("2mom")["ice"], c, channel)


def test_inactive_and_endpoint_lookup_coordinates():
    # Independent WRF endpoint constraints, including the upper cell, not an
    # expectation built by calling the implementation a second time.
    c = lut.ice_coordinates(jnp.array([1.e-20, 1.e-1]), jnp.array([1.e7, 1.e-5]),
                            jnp.array([0., 1.e-1]), jnp.array([0., 1000.]))
    np.testing.assert_array_equal(c.mass_index, [1, 49])
    np.testing.assert_array_equal(c.mass, [1, 50])
    np.testing.assert_array_equal(c.rime_index, [1, 3])
    np.testing.assert_array_equal(c.density_index, [1, 4])
    np.testing.assert_array_equal(c.density, [1, 5])
    np.testing.assert_array_equal(packed(lut.rain_coordinates(jnp.array([0., 1.e-16]), 0.)),
                                  np.array([[1., 1.], [1., 1.]], dtype=np.float32))
    np.testing.assert_array_equal(packed(lut.mu_coordinates(0., 20.)), [1., 1.])


def test_host_only_tables_and_two_jits():
    arrays = load_tables("2mom")
    assert all(type(v) is np.ndarray and not v.flags.writeable for v in arrays.values())
    with pytest.raises(TypeError):
        arrays["new"] = np.zeros(1)
    with pytest.raises(ValueError):
        arrays["ice"].flat[0] = 0.
    with pytest.raises(ValueError, match="Unknown P3"):
        load_tables("bogus")
    c = ice_coord(fixture()["ice_coords"])
    # Loader first used while tracing must not cache a tracer (E206).
    load_tables.cache_clear()
    for channel in (1, 2):
        fn = jax.jit(lambda c: lut.ice_lookup(load_tables("2mom")["ice"], c, channel))
        check(fn(c), fixture()["lookup"][:, channel-1])
    assert all(type(v) is np.ndarray for v in load_tables("2mom").values())


def test_no_double_compute():
    with jax.enable_x64(True):
        c = ice_coord(fixture()["ice_coords"])
        t = jnp.asarray(load_tables("2mom")["ice"])
        text = str(jax.make_jaxpr(lambda t, c: lut.ice_lookup(t, c, 1))(t, c))
        for kind in ("2mom", "3mom"):
            data = fixture(kind)
            table = load_tables(kind)
            rain = scalar_coord(data["rain_coords"])
            mu = scalar_coord(data["mu_coords"]) if kind == "3mom" else None
            text += str(jax.make_jaxpr(lambda t, c, rain, mu:
                lut.rain_ice_lookup(t, c, rain, 1, mu))(jnp.asarray(table["rain"]), c, rain, mu))
            if mu is not None:
                text += str(jax.make_jaxpr(lambda t, c, mu: lut.ice_lookup(t, c, 1, mu))(
                    jnp.asarray(table["ice"]), c, mu))
        pair = fixture("ice_ice")["pair_coords"]
        text += str(jax.make_jaxpr(lut.ice_ice_lookup)(jnp.asarray(load_tables("ice_ice")["number"]),
                                                    ice_coord(pair[:, :6]), ice_coord(pair[:, 6:])))
        x = jnp.asarray(fixture()["inputs"])
        text += str(jax.make_jaxpr(lambda x: (lut.ice_pair_coordinates(
            x[:, 0], x[:, 1], x[:, 2], x[:, 10], x[:, 7], x[:, 8], x[:, 9], x[:, 11]),
            bulk_rime_density(x[:, 0], x[:, 2], x[:, 10]),
            compute_mu_3moment(x[:, 12], x[:, 13], x[:, 14])))(x))
        assert "f64" not in text


def test_provenance_census_and_inventory():
    manifest = json.loads((ROOT / "proofs/s2_p3/manifest.json").read_text())
    assert manifest["modifications"].startswith("private visibility only")
    assert manifest["census"]["positive_qi"] == 128
    assert manifest["census"]["positive_qi2"] == 128
    assert manifest["census"]["positive_qir"] > 100
    assert manifest["census"]["positive_qir2"] > 90
    assert manifest["census"]["collector_collectee_threshold_gap"] > 0
    for filename, item in manifest["files"].items():
        path = ROOT / "src/gpuwrf/physics/p3/data" / Path(filename).name if filename.startswith("assets/") else ROOT / "proofs/s2_p3" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    pristine = ROOT / "data/wrf_pristine/WRF/phys/module_mp_p3.F"
    if pristine.is_file():
        assert hashlib.sha256(pristine.read_bytes()).hexdigest() == manifest["source_sha256"]


def test_tables_in_clean_archive(tmp_path):
    archive = tmp_path / "p3.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), "HEAD",
                    "src/gpuwrf/physics/p3", "proofs/s2_p3"], cwd=ROOT, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "clean", filter="data")
    module_file = tmp_path / "clean/src/gpuwrf/physics/p3/tables.py"
    spec = importlib.util.spec_from_file_location("p3_archive_tables", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for kind in ("2mom", "3mom", "ice_ice", "warm"):
        assert all(np.isfinite(a).all() for a in module.load_tables(kind).values())
    archived = tmp_path / "clean/proofs/s2_p3"
    for name in json.loads((archived / "manifest.json").read_text())["files"]:
        if name.startswith("fixtures/"):
            assert (archived / name).is_file(), name
    for scheme in (50, 51, 52, 53):
        assert (archived / f"fixtures/columns_mp{scheme}.npz").is_file()
    import tomllib
    package_data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    assert "data/*.npz" in package_data["gpuwrf.physics.p3"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_comparison_rejects_nonfinite(bad):
    for a, b in ((bad, 0.), (0., bad)):
        with pytest.raises(AssertionError, match="nonfinite"):
            check(np.array([a], dtype=np.float32), np.array([b], dtype=np.float32))
