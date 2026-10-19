"""scripts/wn3_score.py gate: every non-finite value fails, in the reference (CPU-WRF) as well as the candidate (GPU),
for every compared numeric field, bounded by the frozen D6 manifest or not (E200; cr-w42-scorer GATE4).

The comparator scores finite pairs only, so a NaN/±Inf on the reference side used to drop out silently. The real gate
cases carry none ([M] 0 over 374 fields x all leads, 0227 24 h and 0420 72 h), so the exemption list is empty; an
exemption, if ever added, must be listed in the summary, never silently applied.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
TIMES = ["2026-02-28_00:00:00", "2026-02-28_01:00:00"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


score = _load(REPO / "scripts" / "wn3_score.py", "wn3_score_finite_gate_test")


def _frame(path: Path, time: str, fields: dict[str, np.ndarray]) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in time]
        for name, values in fields.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = values


def _score(tmp_path: Path, side: str | None, field: str = "AUX", value: float = np.nan, whole: bool = False) -> dict:
    """Comparator report for 2 frames; `value` is put into `field` on `side` in frame 1 (one cell, or every cell)."""
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    cpu.mkdir(parents=True)
    gpu.mkdir(parents=True)
    base = {"T2": np.full((3, 4), 290.0, "f4") + np.arange(12, dtype="f4").reshape(3, 4) * 0.01,
            "AUX": np.arange(12, dtype="f4").reshape(3, 4)}
    for i, t in enumerate(TIMES):
        sides = {"cpu": {k: v.copy() for k, v in base.items()}, "gpu": {k: v.copy() for k, v in base.items()}}
        if side is not None and i == 1:
            if whole:
                sides[side][field][:] = value
            else:
                sides[side][field][1, 2] = value
        _frame(cpu / f"wrfout_d01_{t}", t, sides["cpu"])
        _frame(gpu / f"wrfout_d01_{t}", t, sides["gpu"])
    compare = _load(REPO / "scripts" / "compare_wrfout_grid.py", "compare_wrfout_grid_finite_gate_probe")
    out = tmp_path / "d01.json"
    compare.main(["--cpu-dir", str(cpu), "--gpu-dir", str(gpu), "--domain", "d01", "--init", "2026-02-28T00:00:00Z",
                  "--min-lead", "0", "--max-lead", "1", "--tolerance-json", str(MANIFEST), "--no-spatial-splits",
                  "--out-json", str(out), "--out-md", str(out.with_suffix(".md"))])
    report = json.loads(out.read_text())
    assert "T2" in json.loads(MANIFEST.read_text())["fields"] and "AUX" not in json.loads(MANIFEST.read_text())["fields"]
    return report


def test_finite_control_passes(tmp_path):
    report = _score(tmp_path, None)
    assert score.report_gate_failures(report, expected_frames=2) == []
    assert score.reference_nonfinite_exempted(report) == []


@pytest.mark.parametrize("field", ["AUX", "T2"])  # AUX has no D6 bound, T2 is bounded
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["cpu", "gpu"])
def test_single_nonfinite_value_fails_on_either_side(tmp_path, side, value, field):
    report = _score(tmp_path, side, field, value)
    failures = score.report_gate_failures(report, expected_frames=2)
    named = [f for f in failures if f["kind"] == f"nonfinite_{side}"]
    assert named == [{"kind": f"nonfinite_{side}", "field": field, "lead_h": 1, "count": 1}], failures
    assert report["summaries"]["verdict"] in ("PASS", "FAIL")  # the comparator itself still reports


@pytest.mark.parametrize("side", ["cpu", "gpu"])
def test_whole_nonfinite_frame_fails(tmp_path, side):
    report = _score(tmp_path, side, "AUX", np.nan, whole=True)
    failures = score.report_gate_failures(report, expected_frames=2)
    assert {"kind": f"nonfinite_{side}", "field": "AUX", "lead_h": 1, "count": 12} in failures, failures


def test_reference_exemption_is_listed_never_silent(tmp_path, monkeypatch):
    report = _score(tmp_path, "cpu", "AUX", np.inf)
    monkeypatch.setattr(score, "NONFINITE_REFERENCE_EXEMPT", {"AUX": "test-only documented reason"})
    assert not [f for f in score.report_gate_failures(report, expected_frames=2) if f["kind"] == "nonfinite_cpu"]
    assert score.reference_nonfinite_exempted(report) == [
        {"field": "AUX", "lead_h": 1, "count": 1, "reason": "test-only documented reason"}]
    gpu_report = _score(tmp_path / "g", "gpu", "AUX", np.inf)  # the candidate side is never exempt
    assert [f["kind"] for f in score.report_gate_failures(gpu_report, expected_frames=2)] == ["nonfinite_gpu"]
