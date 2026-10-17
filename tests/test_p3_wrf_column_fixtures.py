"""Integrity/activity of original WRF enablers; no JAX p3_main parity claim."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "proofs/s2_p3"


@pytest.mark.parametrize("scheme", [50, 51, 52, 53])
def test_original_wrapper_fixtures_finite_and_active(scheme):
    manifest = json.loads((PROOF / "columns_manifest.json").read_text())
    assert manifest["source_sha256"] == manifest["staged_sha256"]
    assert "no JAX" in manifest["qualification"]
    assert "-fcheck=all" in manifest["flags"]
    name = f"columns_mp{scheme}.npz"
    path = PROOF / "fixtures" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"]
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype == np.float32
            assert np.isfinite(data[key]).all(), (scheme, key)
        assert data["qi"].shape == (8, 12, 4)
        assert (data["qi"][:, :, 0] > 0).sum() == 72
        assert (data["qir"][:, :, 0] > 0).sum() == 60
        assert np.any(data["qi"][:, :, 1:] != data["qi"][:, :, :1])
        assert np.any(data["rainnc"][:, 1:] > 0)
        if scheme == 52:
            assert (data["qi2"][:, :, 0] > 0).sum() == 72
            assert np.any(data["qi2"][:, :, 1:] != data["qi2"][:, :, :1])
        else:
            np.testing.assert_array_equal(data["qi2"], 0.)
        if scheme == 50:
            np.testing.assert_array_equal(data["nc"], 0.)
        else:
            assert np.any(data["nc"][:, :, 1:] != data["nc"][:, :, :1])
        if scheme == 53:
            assert np.any(data["qzi"][:, :, 1:] != data["qzi"][:, :, :1])
        else:
            np.testing.assert_array_equal(data["qzi"], 0.)


def test_original_wrapper_source_and_registry_provenance():
    manifest = json.loads((PROOF / "columns_manifest.json").read_text())
    pristine = ROOT / "data/wrf_pristine/WRF"
    for name, key in (("phys/module_mp_p3.F", "source_sha256"),
                      ("phys/module_microphysics_driver.F", "caller_source_sha256"),
                      ("Registry/Registry.EM_COMMON", "registry_sha256")):
        path = pristine / name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]
