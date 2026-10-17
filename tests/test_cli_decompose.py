"""``gpuwrf run --decompose`` admission (pre-JAX; lane o2-multigpu): opt-in only, native single d01 only."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from gpuwrf.cli import main

REPO = Path(__file__).resolve().parents[1]
SWISS = REPO / "examples" / "switzerland_d01"


@pytest.fixture(autouse=True)
def _no_decompose_env(monkeypatch):
    monkeypatch.delenv("GPUWRF_DECOMPOSE", raising=False)
    yield
    os.environ.pop("GPUWRF_DECOMPOSE", None)  # the CLI sets it in-process on success


def _run(tmp_path, *extra):
    return main(["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path / "out"), "--domain", "d01",
                 "--hours", "1", *extra])


@pytest.mark.parametrize("spec", ["1x1", "2", "2x1,foo=3", "0x2", "2x1,transport=mpi"])
def test_malformed_decompose_is_refused_pre_jax(tmp_path, capsys, spec):
    assert _run(tmp_path, "--dry-run", "--decompose", spec) == 2
    assert "--decompose" in capsys.readouterr().err
    assert "GPUWRF_DECOMPOSE" not in os.environ


def test_decompose_needs_the_native_single_domain_path(tmp_path, capsys):
    case = tmp_path / "replay_case"
    case.mkdir()
    for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01"):
        shutil.copy(SWISS / name, case / name)
    for stamp in ("2023-01-15_00:00:00", "2023-01-15_01:00:00"):  # >= 2 history files -> CPU-WRF replay mode
        (case / f"wrfout_d01_{stamp}").write_bytes(b"")
    rc = main(["run", "--input-dir", str(case), "--output-dir", str(tmp_path / "out"), "--domain", "d01", "--hours", "1",
               "--dry-run", "--decompose", "2x1"])
    assert rc == 2
    assert "--decompose supports only a single native d01 domain" in capsys.readouterr().err


def test_stray_environment_variable_is_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GPUWRF_DECOMPOSE", "2x1")
    assert _run(tmp_path, "--dry-run") == 2
    assert "GPUWRF_DECOMPOSE is set" in capsys.readouterr().err


def test_valid_decompose_passes_admission_and_is_recorded(tmp_path, capsys):
    rc = _run(tmp_path, "--dry-run", "--decompose", "2x1,halo=20")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert os.environ.get("GPUWRF_DECOMPOSE") == "2x1,halo=20"
    assert '"decompose": "2x1,halo=20"' in out or json.loads(out[out.index("{"):]).get("decompose") == "2x1,halo=20"


def test_without_the_flag_nothing_is_set(tmp_path, capsys):
    assert _run(tmp_path, "--dry-run") == 0
    assert "GPUWRF_DECOMPOSE" not in os.environ
