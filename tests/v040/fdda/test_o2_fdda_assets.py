"""The o2-fdda oracle savepoints are tracked (``*.npz`` is gitignored) and match their MANIFEST."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SAVEPOINTS = Path("proofs/o2_fdda/savepoints")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def test_savepoints_tracked_and_match_manifest():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    manifest = json.loads((ROOT / SAVEPOINTS / "MANIFEST.json").read_text())
    tracked = set(_git("ls-files", str(SAVEPOINTS)).decode().split())
    for name, meta in manifest["cases"].items():
        assert str(SAVEPOINTS / name) in tracked, name
        assert hashlib.sha256((ROOT / SAVEPOINTS / name).read_bytes()).hexdigest() == meta["sha256"], name


def test_savepoints_load_from_clean_git_archive(tmp_path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    blob = _git("archive", "--format=tar", "HEAD", str(SAVEPOINTS))
    with tarfile.open(fileobj=BytesIO(blob)) as tar:
        tar.extractall(tmp_path, filter="data")
    manifest = json.loads((tmp_path / SAVEPOINTS / "MANIFEST.json").read_text())
    for name in manifest["cases"]:
        with np.load(tmp_path / SAVEPOINTS / name) as data:
            assert "call0_unc_rth" in data and data["calls"].shape[1] == 4
