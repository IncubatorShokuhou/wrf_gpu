"""FLG leaf parity to unchanged WRF, not operational scheme qualification."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import ra_flg as flg

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "s2_flg_leaves_v1.npz"


def test_fixture_provenance_and_census(data):
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "proofs/s2_flggfdl/manifest.json").read_text())
    for relative, digest in manifest["fixtures_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    assert manifest["leaf_tolerance_preregistered"] == {"ulps":16,"abs":1.e-12,"finite_required":True,"dtype":"float32"}
    assert data["gas"].shape == (40,44,121)
    assert np.count_nonzero(data["in_fields"][...,7]) > 0  # rain
    assert np.count_nonzero(data["in_fields"][...,8]) > 0  # ice
    assert np.count_nonzero(data["in_fields"][...,9]) > 0  # snow
    assert np.count_nonzero(data["graupel"][...,0,:]) > 0  # explicit active graupel


@pytest.fixture(scope="module")
def data():
    with np.load(FIXTURE, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def check(candidate, reference):
    candidate = np.asarray(candidate)
    assert candidate.dtype == reference.dtype == np.float32
    assert np.isfinite(candidate).all() and np.isfinite(reference).all()
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    tolerance = 16. * np.spacing(np.abs(reference)).astype(np.float64) + 1.e-12
    assert np.all(error <= tolerance), f"WRF leaf parity: {np.count_nonzero(error > tolerance)} outside; max ratio {np.max(error / tolerance)}"


def profiles(data):
    f = data["in_fields"]
    return tuple(jnp.asarray(v) for v in (f[:, ::-1, 12] / 100., f[:, ::-1, 4], f[:, ::-1, 5], data["ozone"][:, ::-1, 4]))


def test_geometric_optics(data):
    p, t, _, _ = profiles(data)
    check(jax.jit(flg.layer_thickness)(p, t), data["dz"])
    check(jax.jit(flg.rayleigh_pressure)(p), data["trp"])


@pytest.mark.parametrize("profile", range(1, 6))
def test_prescribed_ozone(data, profile):
    p = jnp.asarray(data["in_fields"][:, :, 12] / 100.)
    check(jax.jit(lambda p: flg.ozone_profile(p, profile))(p), data["ozone"][..., profile - 1])


@pytest.mark.parametrize("band", range(1, 19))
def test_all_gas_quadratures(data, band):
    p, t, q, o3 = profiles(data)
    t = jnp.clip(t, 180., 345.) if band > 7 else t
    check(jax.jit(lambda p,t,q: flg.continuum(p,t,q,band))(p,t,q), data["continuum"][..., band - 1])
    run = jax.jit(lambda p,t,q,o3,ig: flg.gas_optics(p,t,q,o3,band,ig))
    offset = sum(flg.G_POINTS[:band - 1])
    for ig in range(flg.G_POINTS[band - 1]):
        depth, weight, t = run(p,t,q,o3,jnp.asarray(ig,np.int32))
        check(depth, data["gas"][..., offset + ig])
        check(weight, data["weights"][offset + ig])


@pytest.mark.parametrize("band", range(1, 19))
def test_scattering_and_active_graupel(data, band):
    f = data["in_fields"]
    rr = (f[:, -2::-1, 7] * f[:, -2::-1, 3]) * np.float32(1000.)
    gg = (f[:, -2::-1, 10] * f[:, -2::-1, 3]) * np.float32(1000.)
    gg[:, :5] = [0., 1.e-6, 1.e-5, 1.e-4, .01]
    dz = jnp.asarray(data["dz"])
    for name, content, graupel in (("rain", rr, False), ("graupel", gg, True)):
        tau, omega, moments = jax.jit(lambda c,d: flg.precipitation_optics(c,d,band,graupel=graupel))(jnp.asarray(content),dz)
        expected = data[name][..., band - 1]
        check(tau, expected[..., 0]); check(omega, expected[..., 1]); check(moments, expected[..., 2:])
    tau, omega, moments = jax.jit(lambda p: flg.rayleigh_optics(p,band,jnp.float32(.6)))(jnp.asarray(data["trp"]))
    expected = data["rayleigh"][..., band - 1]
    check(tau, expected[..., 0]); check(omega, expected[..., 1]); check(moments, expected[..., 2:])


@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
def test_comparison_rejects_nonfinite(bad):
    with pytest.raises(AssertionError):
        check(np.array([bad],np.float32),np.zeros(1,np.float32))
    with pytest.raises(AssertionError):
        check(np.zeros(1,np.float32),np.array([bad],np.float32))
