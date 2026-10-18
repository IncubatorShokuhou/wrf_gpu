"""Production per-domain wrfout writer on the synthetic case, without ALISIOS references.

Copy of tests/test_initial_history_wrf_t0.py `_wrfinput`/`_writer` (that module stats a WN3 path
at import, which tests/conftest.py's ALISIOS guard turns into a module-level skip)."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.integration.daily_pipeline import _merge_output_diagnostics, finite_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_m7_netcdf_writer import synthetic_case, writer_authority  # noqa: E402

INPUT = {"PSFC": 95000.0, "T2": 290.5, "TSK": 291.25, "U10": 3.5, "V10": -1.25, "Q2": 0.0075, "PBLH": 123.0}


def _wrfinput(path: Path, ny: int, nx: int) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        for name, value in INPUT.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = np.full((ny, nx), value, np.float32)


def _writer(tmp_path: Path, resolve) -> tuple[object, object]:
    state, grid, namelist = synthetic_case()
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer._surface_diagnostics_for_output = resolve
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer._finite_summary = finite_summary
    writer.writer_diagnostics = {}
    writer.writer_static_latlon_metadata = {}
    writer._async_writer = None
    writer._output_pipeline = None
    writer._variable_subset = None
    writer._full_variable_set = False
    writer.written = {"d01": []}
    writer.census_io_ledger = None
    writer._census_persist_info = {}
    writer.domain_authorities = {"d01": writer_authority(grid, "d01")}
    writer.output_dir = tmp_path / "out"
    writer.output_dir.mkdir()
    writer.input_dir = tmp_path
    writer.run_start = datetime(2026, 2, 28, 0, tzinfo=timezone.utc)
    writer.bundles = {"d01": SimpleNamespace(namelist=namelist, grid=grid)}
    writer.dt_by_domain = {"d01": 18.0}
    _wrfinput(tmp_path / "wrfinput_d01", grid.ny, grid.nx)
    return writer, state

