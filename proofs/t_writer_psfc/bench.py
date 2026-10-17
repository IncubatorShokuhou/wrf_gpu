"""Host writer comparison, including changed fallback allocation and I/O work.

Not a forecast/GPU benchmark. Parent source is loaded unchanged from git.
Run in an immutable checkout, CPU only, capped RAM slot, no persistent cache.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_m7_netcdf_writer import synthetic_case, authenticated_source_grid
from gpuwrf.io import wrfout_writer as candidate
from datetime import datetime
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_commit = "b9852764fbca3d3c9cb22d3bfc5d703432bcbdbe"
    source = subprocess.check_output(["git", "show", f"{parent_commit}:src/gpuwrf/io/wrfout_writer.py"], cwd=ROOT)
    parent = types.ModuleType("_writer_psfc_parent")
    parent.__file__ = candidate.__file__
    sys.modules[parent.__name__] = parent
    exec(compile(source, parent.__file__, "exec"), parent.__dict__)
    state, grid, nl = synthetic_case(nx=267, ny=117, nz=44)
    del state.psfc
    grid.metrics = SimpleNamespace(c1h=np.linspace(.1, 1, 44, dtype=np.float32),
        c2h=np.linspace(50, 100, 44, dtype=np.float32),
        dnw=np.full(44, -1/44, np.float32), p_top=np.float32(5000))
    state.qs = np.full_like(state.qv, .0001)
    state.qg = np.full_like(state.qv, .0002)
    authority = candidate.bind_wrfout_domain_authority("d01", authenticated_source_grid(grid, "d01"), grid)
    kwargs = dict(domain="d01", domain_authority=authority,
        run_start=datetime(2026, 10, 15), valid_time=datetime(2026, 10, 15, 1), lead_hours=1.)
    # Authority is frozen data; construct the parent's own dataclass type.
    parent_authority = parent.WrfoutDomainAuthority(**vars(authority))
    def run(module, subset):
        return module.prepare_wrfout_payload(state, grid, nl, args.output.with_suffix(".nc"),
            **dict(kwargs, domain_authority=parent_authority if module is parent else authority),
            variable_subset=("PSFC",) if subset else None)
    report = {"kind": "host-only controlled writer timing; not forecast/GPU fidelity", "parent": parent_commit,
              "shape": list(state.qv.shape), "repeats": 7, "cache": "NONE", "arms": {}}
    old = run(parent, False)
    new = run(candidate, False)
    assert set(old.fields) == set(new.fields)
    assert all(old.fields[name].tobytes() == new.fields[name].tobytes() for name in old.fields)
    report["no_hail_full_fields_byte_equal"] = len(old.fields)
    del old, new
    for optional in (False, True):
        state.qh = np.full_like(state.qv, .001) if optional else None
        for label, module, subset in (("parent_full", parent, False), ("candidate_full", candidate, False),
                                      ("parent_subset", parent, True), ("candidate_subset", candidate, True)):
            del_payload = run(module, subset)
            del del_payload
            times = []
            for _ in range(7):
                started = time.perf_counter()
                payload = run(module, subset)
                times.append(time.perf_counter() - started)
                del payload
            gc.collect()
            tracemalloc.start()
            payload = run(module, subset)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del payload
            report["arms"][f"{label}_{'hail' if optional else 'no_hail'}"] = {
                "seconds": times, "median_seconds": statistics.median(times), "python_numpy_peak_bytes": peak}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
