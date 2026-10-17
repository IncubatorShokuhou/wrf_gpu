"""Original-WRF operator qualification; mp38 forecast admission is separate."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import thompson_hail as hail


FIXTURE = Path(__file__).resolve().parents[3] / "proofs/v040/mpmisc/thompson38/hail_operators.json"


def assert_original_operators():
    ref = {name: np.asarray(values, np.float32) for name, values in json.loads(FIXTURE.read_text())["values"].items()}
    inputs = tuple(jnp.asarray(ref[n]) for n in ("QG_INPUT", "NG_INPUT", "QB_INPUT", "RHO", "VISCO"))
    with jax.disable_jit():
        d = hail.entry_distribution(*inputs[:4])
        vt, vtn = hail.fall_speeds(d, inputs[3], inputs[4])
        qten, nten, bten, precip = hail.sediment_hail(d.rg, d.ng, d.rb, inputs[3], jnp.asarray(ref["DZ"]), vt, vtn, 90.)
    names = ("QG", "NG_IN", "QB", "RG", "NG", "RB", "IDX", "ILAMG", "VTG", "VTNG", "QGTEN", "NGTEN", "QBTEN", "PRECIP")
    outputs = (d.qg, d.Ng, d.qvolg, d.rg, d.ng, d.rb, d.index, d.ilamg.astype(jnp.float32), vt, vtn, qten, nten, bten, precip)
    for name, actual in zip(names, outputs):
        actual = np.asarray(actual)
        expected = (ref[name] - 1 if name == "IDX" else ref[name]).reshape(actual.shape)
        assert np.isfinite(actual).all(), name
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    assert d.lamg.dtype == jnp.float64  # Original explicit DOUBLE island.
    assert d.rg.dtype == d.rb.dtype == qten.dtype == bten.dtype == jnp.float32


def test_original_hail_operators_literal_bytes():
    assert_original_operators()


def test_original_reference_detects_volume_flux_units_mutant(monkeypatch):
    original = hail.sediment_hail

    def mass_units_for_volume(*args, **kwargs):
        qten, nten, _bten, precip = original(*args, **kwargs)
        return qten, nten, qten, precip

    monkeypatch.setattr(hail, "sediment_hail", mass_units_for_volume)
    with pytest.raises(AssertionError, match="QBTEN"):
        assert_original_operators()
