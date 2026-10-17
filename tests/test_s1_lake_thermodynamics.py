"""Original-WRF DOUBLE helper parity; no GPU or coupled-forecast claim."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from gpuwrf.physics.lake_thermodynamics import (
    lake_combine, lake_monin_obukhov_initial, lake_tridiagonal,
)

FIXTURE = Path(__file__).resolve().parents[1] / "proofs/s1_lake/fixtures"


def compare(actual, expected):
    assert np.isfinite(expected).all()
    for value in jax.tree.leaves(actual):
        assert np.asarray(value).dtype == np.float64
        assert np.isfinite(value).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


def test_combo_pristine_double():
    with np.load(FIXTURE / "helpers.npz", allow_pickle=False) as f, jax.enable_x64():
        data = f["COMBO"]
        compare(jax.jit(lake_combine)(*data[:, :8].T), data[:, 8:].T)
        # The oracle actually takes all three enthalpy branches.
        assert data[0, -1] < 273.16
        assert data[1, -1] == 273.16
        assert data[2, -1] > 273.16


def test_monin_pristine_double():
    with np.load(FIXTURE / "helpers.npz", allow_pickle=False) as f, jax.enable_x64():
        data = f["MONIN"]
        compare(jax.jit(lake_monin_obukhov_initial)(*data[:, :5].T), data[:, 5:].T)
        assert (data[:, -1] < 0).any() and (data[:, -1] > 0).any()


def test_tridiagonal_pristine_double():
    with np.load(FIXTURE / "helpers.npz", allow_pickle=False) as f, jax.enable_x64():
        actual = jax.jit(lake_tridiagonal)(f["TRI_A"],f["TRI_B"],f["TRI_C"],f["TRI_R"],
                                          jtop=f["TRI_TOP"],previous=f["TRI_SEED"],enabled=f["TRI_FILTER"])
        compare(actual, f["TRI_U"])
        np.testing.assert_array_equal(actual[2], f["TRI_SEED"][2])
        np.testing.assert_array_equal(actual[1,:5], f["TRI_SEED"][1,:5])


def test_helper_provenance():
    m = json.loads((FIXTURE / "helpers.json").read_text())
    proof = FIXTURE.parent
    for name, key in (("helper_driver.F90", "driver_sha256"), ("wrf_stubs.F90", "stubs_sha256")):
        assert hashlib.sha256((proof / name).read_bytes()).hexdigest() == m[key]
    assert hashlib.sha256((FIXTURE / "helpers.npz").read_bytes()).hexdigest() == m["fixture_sha256"]
    assert m["source_sha256"] == json.loads((FIXTURE / "manifest.json").read_text())["source_sha256"]
