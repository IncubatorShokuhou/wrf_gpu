"""w42-scorer: bounded-memory scoring (compare_wrfout_grid batches + compact split percentiles, wn3_score domains).

- Spatial-split percentiles cut out of the overall accumulator's chunks equal the former per-label |diff| copies
  exactly (with unpaired cells, and without), so a field keeps one float32 |diff| per value instead of five.
- The field-batch planner keeps every pass within its budget (a lone oversize field gets its own pass), in order.
- wn3_score runs the domain compares one after another for > 24 h or > 3 domains (9-nest) and on request, in parallel
  otherwise; --domains selects the domain list.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


compare = _load(REPO / "scripts" / "compare_wrfout_grid.py", "compare_wrfout_grid_bounded_test")
score = _load(REPO / "scripts" / "wn3_score.py", "wn3_score_bounded_test")


def _split_masks(rng, ny, nx):
    hgt = rng.uniform(-50.0, 2500.0, (ny, nx))
    land = (rng.uniform(size=(ny, nx)) > 0.4).astype(float)
    lat, lon = np.meshgrid(np.linspace(27.5, 28.5, ny), np.linspace(-17.0, -16.0, nx), indexing="ij")
    return compare.split_masks_for_grid(hgt, land, lat, lon, boundary_width=2)


@pytest.mark.parametrize("with_nan", [False, True])
def test_split_percentiles_from_overall_chunks_equal_own_copies(with_nan):
    rng = np.random.default_rng(11)
    shape = (5, 9, 13)
    splits = _split_masks(rng, *shape[1:])
    old = {s: {lab: compare.StatsAccumulator(element_abs_tolerance=0.3) for lab in m} for s, m in splits.items()}
    overall = compare.StatsAccumulator(element_abs_tolerance=0.3, record_valid=True)
    new = {s: {lab: compare.StatsAccumulator(element_abs_tolerance=0.3, keep_abs=False) for lab in m}
           for s, m in splits.items()}
    for lead in range(6):
        cpu = rng.standard_normal(shape).astype("f4") + 280.0
        gpu = cpu + rng.standard_normal(shape).astype("f4") * (0.1 + lead)
        if with_nan:
            cpu[rng.uniform(size=shape) < 0.07] = np.nan
            gpu[rng.uniform(size=shape) < 0.05] = np.nan
            if lead == 3:
                cpu[:, :4, :] = np.nan  # some labels have no pairs in this lead
        overall.update(gpu, cpu)
        for s, masks in splits.items():
            for lab, mask in masks.items():
                old[s][lab].update(gpu, cpu, mask2d=mask)
                new[s][lab].update(gpu, cpu, mask2d=mask)
    assert all(not acc.abs_chunks for labels in new.values() for acc in labels.values())
    for s, masks in splits.items():
        for lab, mask in masks.items():
            got = new[s][lab].finish(abs_values=overall.subset_abs(mask))
            want = old[s][lab].finish()
            assert got == want, (s, lab)
            assert np.array_equal(overall.subset_abs(mask), np.concatenate(old[s][lab].abs_chunks))


def test_a_field_keeps_one_float32_abs_copy(monkeypatch):
    """VariableComparison keeps exactly one float32 |diff| per paired value (overall chunks); the split accumulators
    keep none (they were 4 extra full copies: 2 GB per 3-D d02 field at 72 h)."""
    rng = np.random.default_rng(3)
    ny, nx, nz = 9, 13, 4
    masks = {"mass": _split_masks(rng, ny, nx)}
    meta = {"cpu": {"dims": ["bottom_top", "south_north", "west_east"], "shape": [nz, ny, nx]}}
    comparison = compare.VariableComparison("T", meta, None, masks)
    frames = 5

    class Var:
        dimensions = ("Time", "bottom_top", "south_north", "west_east")
        dtype = np.dtype("f4")

        def __init__(self, values):
            self.values = values
            self.shape = (1, *values.shape)

        def __getitem__(self, key):
            return self.values

    class Ds:
        def __init__(self, values):
            self.variables = {"T": Var(values)}

    monkeypatch.setattr(compare, "var_metadata", lambda ds, name: {"dims": ["bottom_top", "south_north", "west_east"],
                                                                     "shape": [nz, ny, nx], "dtype": "float32"})
    for lead in range(frames):
        cpu = rng.standard_normal((nz, ny, nx)).astype("f4")
        comparison.add({"lead_h": lead, "valid_time_utc": str(lead)}, Ds(cpu), Ds(cpu + 0.5))
    kept = sum(c.nbytes for c in comparison.overall.abs_chunks)
    split_kept = sum(c.nbytes for labels in comparison.split_accs.values() for acc in labels.values() for c in acc.abs_chunks)
    assert kept == 4 * nz * ny * nx * frames and split_kept == 0
    assert set(comparison.split_accs) == {"land_ocean", "elevation", "quadrant", "boundary"}
    report = comparison.finish()
    assert report["spatial_splits"]["land_ocean"] and report["compared_lead_count"] == frames


def _entry(i, ny, nx, nz=None):
    shape = [nz, ny, nx] if nz else [ny, nx]
    return (i, f"F{i}", {"cpu": {"shape": shape}, "gpu": {"shape": shape}})


def test_field_batches_respect_the_budget_and_order():
    entries = [_entry(1, 117, 267, 44), _entry(2, 117, 267), (3, "Times", None), _entry(4, 117, 267, 44),
               _entry(5, 117, 267), _entry(6, 117, 267), _entry(7, 117, 267, 44)]
    frames = 73
    budget = int(1.5 * compare.field_state_bytes(entries[0][2], frames))
    batches = compare.plan_field_batches(entries, frames, budget)
    assert [e for batch in batches for e in batch] == entries, "every field once, in the original order"
    for batch in batches:
        used = sum(compare.field_state_bytes(e[2], frames) for e in batch)
        assert used <= budget or len(batch) == 1, batch
    assert len(batches) == 3
    lone = compare.plan_field_batches(entries[:1], frames, 1)
    assert lone == [entries[:1]], "a field above the budget still gets its own pass"


def _frame(path: Path, time: str, t2: np.ndarray) -> None:
    ny, nx = t2.shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in time]
        yy, xx = np.meshgrid(np.arange(ny, dtype="f4"), np.arange(nx, dtype="f4"), indexing="ij")
        for name, values in {"HGT": yy * 100.0 + xx * 50.0, "LANDMASK": (xx > nx / 2).astype("f4"),
                             "XLAT": 28.0 + yy * 0.1, "XLONG": -17.0 + xx * 0.1, "T2": t2}.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = values


def _case(root: Path, hours: int, domains=("d01", "d02", "d03")) -> tuple[Path, Path]:
    rng = np.random.default_rng(5)
    cpu, gpu = root / "cpu", root / "gpu"
    cpu.mkdir()
    gpu.mkdir()
    t0 = datetime(2026, 2, 28)
    for domain in domains:
        for h in range(hours + 1):
            t = (t0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")
            base = rng.standard_normal((4, 5)).astype("f4") + 290.0
            _frame(cpu / f"wrfout_{domain}_{t}", t, base)
            _frame(gpu / f"wrfout_{domain}_{t}", t, base + 0.01)
    return cpu, gpu


@pytest.mark.parametrize("hours, flags, domains, sequential", [
    (25, [], 3, True), (2, [], 3, False), (2, ["--sequential-domains"], 3, True),
    (2, [], 4, True), (2, ["--no-sequential-domains"], 4, False)])
def test_wn3_score_domain_scheduling(tmp_path, monkeypatch, hours, flags, domains, sequential):
    names = [f"d{i:02d}" for i in range(1, domains + 1)]
    cpu, gpu = _case(tmp_path, hours, names)
    out = tmp_path / "out"
    earlier_running: list[int] = []
    procs: list[subprocess.Popen] = []
    real = subprocess.Popen

    class Spy(real):
        def __init__(self, *args, **kwargs):
            earlier_running.append(sum(p.returncode is None for p in procs))
            super().__init__(*args, **kwargs)
            procs.append(self)

    monkeypatch.setattr(score.subprocess, "Popen", Spy)
    twin = tmp_path / "twin_thresholds.json"  # the real file lives under the guarded ALISIOS roots
    twin.write_text("{}\n")
    monkeypatch.setattr(score, "TWIN_THRESHOLDS", twin)
    argv = ["wn3_score.py", "--cpu-dir", str(cpu), "--gpu-dir", str(gpu), "--out", str(out), "--hours", str(hours)]
    if domains != 3:
        argv += ["--domains", *names]
    monkeypatch.setattr(sys, "argv", argv + flags)
    with pytest.raises(SystemExit):  # synthetic files miss the mandatory fields and WRF globals: the gate fails
        score.main()
    assert len(procs) == domains and all(p.returncode == 0 for p in procs)
    if sequential:
        assert earlier_running == [0] * domains, "each domain compare must finish before the next starts"
    else:
        assert earlier_running == list(range(domains)), "parallel mode starts all compares before waiting"
    resources = json.loads((out / "resources.json").read_text())
    assert resources["domain_mode"] == ("sequential" if sequential else "parallel")
    assert sorted(resources["domain_compares"]) == names
    assert all(u["peak_rss_mib"] > 0 and u["rc"] == 0 for u in resources["domain_compares"].values())
    summary = json.loads((out / "summary.json").read_text())
    assert sorted(summary["domains"]) == names
