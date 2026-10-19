"""scripts/wn3_fast_compare.py on top of compare_wrfout_grid.py.

Regression for review-writer 04:16Z: the comparator must open every file through the context-manager protocol, so a
`Dataset` substitute that is only a context manager keeps working; the wrapper's report must equal the direct
comparator's report (modulo timing) with spatial splits ON. w42-scorer: handles are bounded to one frame pair at a time
(the former held-handle wrapper OOMed 72 h scores) and the report does not depend on how fields are batched.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]
COMPARE = REPO / "scripts" / "compare_wrfout_grid.py"
FAST = REPO / "scripts" / "wn3_fast_compare.py"
TIMES = ["2026-02-28_00:00:00", "2026-02-28_01:00:00", "2026-02-28_02:00:00"]


def _frame(path: Path, time: str, t2: np.ndarray) -> None:
    ny, nx = t2.shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in time]
        yy, xx = np.meshgrid(np.arange(ny, dtype="f4"), np.arange(nx, dtype="f4"), indexing="ij")
        statics = {"HGT": yy * 100.0 + xx * 50.0, "LANDMASK": (xx > nx / 2).astype("f4"),
                   "XLAT": 28.0 + yy * 0.1, "XLONG": -17.0 + xx * 0.1, "T2": t2}
        for name, values in statics.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = values


def _pairs(root: Path, perturb: float = 0.1, seed: int = 3, nan_cells: bool = False) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    cpu, gpu = root / "cpu", root / "gpu"
    cpu.mkdir(parents=True)
    gpu.mkdir(parents=True)
    for i, t in enumerate(TIMES):
        base = rng.standard_normal((8, 10)).astype("f4") + 290.0 + i
        other = base + perturb * rng.standard_normal((8, 10)).astype("f4")
        if nan_cells:  # unpaired cells in different places per frame and side
            base[i, 2 + i] = np.nan
            other[7 - i, 9 - i] = np.nan
        _frame(cpu / f"wrfout_d01_{t}", t, base)
        _frame(gpu / f"wrfout_d01_{t}", t, other)
    return cpu, gpu


def _args(cpu: Path, gpu: Path, out: Path) -> list[str]:
    return ["--cpu-dir", str(cpu), "--gpu-dir", str(gpu), "--domain", "d01", "--init", "2026-02-28T00:00:00Z",
            "--min-lead", "0", "--max-lead", "2", "--out-json", str(out), "--out-md", str(out.with_suffix(".md"))]


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in ("generated_utc", "elapsed_seconds")}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(script: Path, args: list[str]) -> dict:
    proc = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(Path(args[args.index("--out-json") + 1]).read_text())


def test_wn3_fast_compare_report_equals_direct_comparator(tmp_path):
    cpu, gpu = _pairs(tmp_path)
    direct = _run(COMPARE, _args(cpu, gpu, tmp_path / "direct.json"))
    fast = _run(FAST, _args(cpu, gpu, tmp_path / "fast.json"))
    assert json.dumps(_strip(direct), sort_keys=True) == json.dumps(_strip(fast), sort_keys=True)
    assert fast["pairing"]["paired_file_count"] == 3 and fast["spatial_splits"]["mask_counts"], "spatial splits must run"
    assert fast["field_summaries"]["T2"]["compared_lead_count"] == 3


def test_wn3_fast_compare_sees_a_mutation(tmp_path):
    cpu, gpu = _pairs(tmp_path, perturb=0.0)
    clean = _run(FAST, _args(cpu, gpu, tmp_path / "clean.json"))
    with netCDF4.Dataset(sorted(gpu.glob("wrfout_d01_*"))[2], "a") as ds:
        ds["T2"][0, 3, 4] += 7.0
    mutated = _run(FAST, _args(cpu, gpu, tmp_path / "mut.json"))
    assert clean["field_summaries"]["T2"]["overall"]["max_abs"] == 0.0
    assert mutated["field_summaries"]["T2"]["overall"]["max_abs"] == pytest.approx(7.0, abs=1e-3)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_handles_are_bounded_to_one_frame_pair(tmp_path, monkeypatch):
    """w42-scorer: at most one CPU+GPU frame pair is open at any time (held handles OOMed 72 h scores), whatever the
    number of passes; every file is opened once per pass (inventory + field batches) plus the spatial-mask read."""
    cpu, gpu = _pairs(tmp_path)
    compare = _load(COMPARE, "compare_wrfout_grid_bound_probe")
    real = netCDF4.Dataset
    live: list[str] = []
    peak, opened = [0], []

    class Counting:
        def __init__(self, path, mode="r"):
            self.path, self.ds = str(path), real(path, mode)
            opened.append(self.path)
            live.append(self.path)
            peak[0] = max(peak[0], len(live))

        def __enter__(self):
            return self.ds

        def __exit__(self, *exc):
            live.remove(self.path)
            self.ds.close()
            return False

    monkeypatch.setattr(compare, "Dataset", Counting)
    try:
        # Tiny fields fit any budget, so force the worst case (one pass per field) with an explicit planner.
        monkeypatch.setattr(compare, "plan_field_batches", lambda entries, n, budget: [[e] for e in entries])
        compare.main(_args(cpu, gpu, tmp_path / "bounded.json"))
    finally:
        sys.modules.pop("compare_wrfout_grid_bound_probe", None)
    assert peak[0] == 2, f"at most one frame pair may be open at once, saw {peak[0]}"
    assert not live, "every open must be closed"
    report = json.loads((tmp_path / "bounded.json").read_text())
    passes = 1 + len(report["field_summaries"])  # inventory pass + one pass per field
    first_cpu = str(sorted(cpu.glob("wrfout_d01_*"))[0])
    for path in sorted(str(p) for p in [*cpu.glob("wrfout_d01_*"), *gpu.glob("wrfout_d01_*")]):
        assert opened.count(path) == passes + (path == first_cpu), path


def test_wrapper_does_not_hold_handles(tmp_path, monkeypatch):
    """The wn3_fast_compare entry point must not re-introduce held handles (the 72 h OOM): with every netCDF open
    counted, at most one frame pair is open at once and nothing stays open after the run."""
    cpu, gpu = _pairs(tmp_path)
    real = netCDF4.Dataset
    live: list[int] = []
    peak = [0]

    class Counting:
        def __init__(self, path, mode="r", *args, **kwargs):
            self._ds = real(path, mode, *args, **kwargs)
            live.append(id(self))
            peak[0] = max(peak[0], len(live))

        def __getattr__(self, name):
            return getattr(self._ds, name)

        def close(self):
            if id(self) in live:
                live.remove(id(self))
            self._ds.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    monkeypatch.setattr(netCDF4, "Dataset", Counting)  # before the wrapper imports the comparator
    fast = _load(FAST, "wn3_fast_compare_bound_probe")
    try:
        monkeypatch.setattr(fast.compare, "plan_field_batches", lambda entries, n, budget: [[e] for e in entries])
        fast.compare.main(_args(cpu, gpu, tmp_path / "wrapped.json"))
        assert peak[0] <= 2 and not live, f"peak {peak[0]} open, {len(live)} still open after the run"
    finally:
        sys.modules.pop("wn3_fast_compare_bound_probe", None)
        sys.modules.pop("compare_wrfout_grid", None)


def test_report_does_not_depend_on_the_field_batching(tmp_path, monkeypatch):
    """One field per pass vs. every field in one pass vs. the real planner: identical reports (modulo timing)."""
    cpu, gpu = _pairs(tmp_path, nan_cells=True)
    compare = _load(COMPARE, "compare_wrfout_grid_batch_probe")
    planners = {"one": lambda entries, n, budget: [[e] for e in entries],
                "all": lambda entries, n, budget: [list(entries)], "real": compare.plan_field_batches}
    reports = {}
    try:
        for label, planner in planners.items():
            monkeypatch.setattr(compare, "plan_field_batches", planner)
            compare.main(_args(cpu, gpu, tmp_path / f"{label}.json"))
            reports[label] = json.dumps(_strip(json.loads((tmp_path / f"{label}.json").read_text())), sort_keys=True)
    finally:
        sys.modules.pop("compare_wrfout_grid_batch_probe", None)
    assert reports["one"] == reports["all"] == reports["real"]


def test_comparator_uses_the_context_manager_protocol(tmp_path, monkeypatch):
    """Any context-manager Dataset substitute works: __enter__ supplies the dataset, __exit__ runs once per open."""
    cpu, gpu = _pairs(tmp_path)
    compare = _load(COMPARE, "compare_wrfout_grid_cm_probe")
    entered, exited = [], []

    class Holder:
        def __init__(self, path, mode="r"):
            self.path, self.ds = str(path), netCDF4.Dataset(path, mode)

        def __enter__(self):
            entered.append(self.path)
            return self.ds

        def __exit__(self, *exc):
            exited.append(self.path)
            self.ds.close()
            return False

    monkeypatch.setattr(compare, "Dataset", Holder)
    try:
        compare.main(_args(cpu, gpu, tmp_path / "cm.json"))
    finally:
        sys.modules.pop("compare_wrfout_grid_cm_probe", None)
    assert entered and sorted(entered) == sorted(exited), "every entered holder must be exited exactly once"
    report = json.loads((tmp_path / "cm.json").read_text())
    assert report["field_summaries"]["T2"]["compared_lead_count"] == 3


def test_build_report_for_namespace_consumers_with_the_lock_dataset(tmp_path):
    """Frozen proof scorers call build_report(argparse.Namespace(...)) without --field-batch-mib and substitute the
    process-wide gpuwrf.io.netcdf_lock.Dataset (an RLock-held context manager): same report as the CLI."""
    import argparse
    lock = pytest.importorskip("gpuwrf.io.netcdf_lock")
    cpu, gpu = _pairs(tmp_path, nan_cells=True)
    cli = _run(COMPARE, _args(cpu, gpu, tmp_path / "cli.json"))
    compare = _load(COMPARE, "compare_wrfout_grid_namespace_probe")
    try:
        compare.Dataset = lock.Dataset
        report = compare.build_report(argparse.Namespace(
            cpu_dir=cpu, gpu_dir=gpu, domain="d01", init="2026-02-28T00:00:00Z", min_lead=0, max_lead=2, vars=None,
            tolerance_json=None, boundary_width=5, no_spatial_splits=False, progress=0))
    finally:
        sys.modules.pop("compare_wrfout_grid_namespace_probe", None)
    as_json = json.loads(json.dumps(report, sort_keys=True, default=compare.json_default))
    assert json.dumps(_strip(as_json), sort_keys=True) == json.dumps(_strip(cli), sort_keys=True)
