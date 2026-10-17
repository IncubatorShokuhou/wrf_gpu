"""Every DFI asset the tests read is TRACKED: load them from a clean ``git archive HEAD`` export."""

import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "proofs/o2_dfi/dfcoef_oracle.json",
    "proofs/o2_dfi/dfcoef_driver.F90",
    "proofs/o2_dfi/build_dfcoef_oracle.py",
    "proofs/o2_dfi/build_start_em_fixture.py",
    "tests/fixtures/o2_dfi_start_em_crop.npz",
    "tests/fixtures/o2_dfi_swiss_tdfi.namelist",
)


def test_dfi_assets_load_from_a_clean_git_archive(tmp_path):
    try:
        packed = subprocess.check_output(["git", "-C", str(ROOT), "archive", "HEAD", *ASSETS])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - not a checkout
        pytest.skip(f"no git checkout: {exc}")
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        archive.extractall(tmp_path, filter="data")
    oracle = json.loads((tmp_path / ASSETS[0]).read_text())
    assert len(oracle["cases"]) == 64
    with np.load(tmp_path / ASSETS[4]) as crop:
        assert crop["P"].shape == (2, 44, 6, 6)
    assert "dfi_opt" in (tmp_path / ASSETS[5]).read_text()
