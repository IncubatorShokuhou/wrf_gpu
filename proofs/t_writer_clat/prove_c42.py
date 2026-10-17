"""Publish existing input-feedback CLAT and compare with original c42 CPU files.

No CPU history value is supplied to the writer, and no model is advanced.
The producer's already-qualified all-frame result is reused after source checks.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/v025"), str(ROOT / "src"), str(ROOT / "tests")]
import cpu_guard
import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io import wrfout_writer as writer
from gpuwrf.coupling.boundary_feedback import build_state_feedback_weights
from gpuwrf.nesting.initial_feedback import _input_point_overrides
from gpuwrf.integration.daily_pipeline import _load_static_latlon_writer_diagnostics
from test_m7_netcdf_writer import synthetic_case, writer_authority

BASE = "f737666b666271d4fc492a431f3754b70d531b9c"
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "cpu"
    args.scratch.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "proofs/t_writer_clat/C42_CLAT_PRODUCER.json"
    receipt = json.loads(receipt_path.read_text())
    # The old producer and every operator it calls are unchanged; reuse VP162.
    for path in ("src/gpuwrf/nesting/initial_feedback.py", "src/gpuwrf/coupling/boundary_feedback.py",
                 "src/gpuwrf/coupling/feedback_registry.py"):
        original = subprocess.check_output(["git", "show", f"{receipt['source_commit']}:{path}"], cwd=ROOT)
        assert original == (ROOT / path).read_bytes(), path
    run = Gen2Run(args.case / "input")
    sources = {name: run.grid(name) for name in ("d01", "d02")}
    child = sources["d02"]
    weights = build_state_feedback_weights(parent_grid_ratio=child.parent_grid_ratio,
        i_parent_start=child.i_parent_start, j_parent_start=child.j_parent_start,
        parent_grid=sources["d01"], child_grid=child, spec_zone=1, smooth_option=2)
    overrides = _input_point_overrides(run, "d01", "d02", weights)
    parent_clat = np.asarray(overrides["CLAT"])
    assert parent_clat.dtype == np.float32
    assert sha(parent_clat.tobytes()) == receipt["producer_CLAT_sha256"]
    manifests = {row["file"]: row for row in json.loads((args.case / "CPU_OUTPUTS.json").read_text())}
    report = {"kind": "CPU host publication of existing producer, not forecast requalification",
              "producer_receipt_sha256": sha(receipt_path.read_bytes()), "producer_sha256": sha(parent_clat.tobytes()),
              "producer_sources_byte_identical": True, "parent_all25_producer_checks_reused": receipt["all_original_frames_BYTE"],
              "domains": {}, "input_only_writer_sources": True}
    for name in ("d01", "d02"):
        source = sources[name]
        grid = source.as_grid_spec()
        diag, _ = _load_static_latlon_writer_diagnostics(run, name, grid=grid)
        if name == "d01":
            diag.update(overrides)
        else:
            # The child has no up-fed descendant: retain its original input CLAT.
            with Dataset(run.path / "wrfinput_d02") as ds:
                diag["CLAT"] = np.asarray(ds["CLAT"][0])
        path = args.scratch / f"wrfout_{name}"
        authority = writer.bind_wrfout_domain_authority(name, source, grid)
        payload = writer.prepare_wrfout_payload({}, grid, None, path, domain=name,
            domain_authority=authority, run_start=datetime(2026, 2, 28), valid_time=datetime(2026, 2, 28),
            lead_hours=0, diagnostics=diag, variable_subset=("CLAT", "XLAT", "XLONG"), full_variable_set=True)
        writer.write_prepared_wrfout(payload, expected_domain=name, expected_domain_authority=authority,
                                    variable_subset=("CLAT", "XLAT", "XLONG"))
        with Dataset(path) as ds:
            actual = np.asarray(ds["CLAT"][0])
            assert ds["CLAT"].dtype == np.dtype(np.float32)
            for field in ("XLAT", "XLONG"):
                assert np.asarray(ds[field][0]).tobytes() == np.asarray(diag[field]).tobytes(), field
        rows = []
        files = sorted((args.case / "cpu").glob(f"wrfout_{name}_*"))
        assert len(files) == 25
        for file in files:
            with Dataset(file) as ds:
                expected = np.asarray(ds["CLAT"][0])
                assert actual.dtype == expected.dtype == np.float32
                assert actual.tobytes() == expected.tobytes(), file.name
            h = hashlib.sha256()
            with file.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
            assert h.hexdigest() == manifests[file.name]["sha256"], file.name
            rows.append({"file": file.name, "file_sha256": h.hexdigest(), "CLAT_BYTE": True,
                         "CLAT_sha256": sha(actual.tobytes()), "max_abs": 0.0})
        report["domains"][name] = {"frames": rows, "written_CLAT_sha256": sha(actual.tobytes()),
                                   "shape": list(actual.shape), "written_dtype": str(actual.dtype)}
    # Only prepare_wrfout_payload changed. Execute its unchanged parent body as
    # the timing/control arm, sharing the exact same builders and inputs.
    path = "src/gpuwrf/io/wrfout_writer.py"
    tree = ast.parse(subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_wrfout_payload")
    namespace = dict(vars(writer))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<original-parent-writer>", "exec"), namespace)
    old = namespace["prepare_wrfout_payload"]
    state, grid, nl = synthetic_case(nx=120, ny=70, nz=44)
    kwargs = dict(domain="d01", domain_authority=writer_authority(grid, "d01"),
        run_start=datetime(2026, 2, 28), valid_time=datetime(2026, 2, 28, 1), lead_hours=1,
        diagnostics={"CLAT": parent_clat})
    def prepare(fn): return fn(state, grid, nl, args.scratch / "timing.nc", **kwargs)
    before, after = prepare(old), prepare(writer.prepare_wrfout_payload)
    assert after.fields["CLAT"] is parent_clat, "producer retained without array allocation/copy"
    assert set(before.fields) == set(after.fields)
    assert all(before.fields[n].tobytes() == after.fields[n].tobytes() for n in before.fields if n != "CLAT")
    report["other_full_fields_BYTE"] = len(before.fields)-1
    del before, after
    costs = {}
    for label, fn in (("parent", old), ("fixed", writer.prepare_wrfout_payload)):
        times = []
        for _ in range(9):
            started = time.perf_counter(); result = prepare(fn)
            times.append(time.perf_counter()-started); del result
        tracemalloc.start(); result = prepare(fn)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); del result
        costs[label] = {"seconds": times, "median_seconds": statistics.median(times), "peak_traced_bytes": peak}
    report["host_cost_120x70x44"] = costs
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
