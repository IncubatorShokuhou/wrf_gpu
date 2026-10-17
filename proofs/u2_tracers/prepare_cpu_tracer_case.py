"""Stage an original CPU-WRF 54-second Swiss tracer case; do not launch it.

Usage: python proofs/u2_tracers/prepare_cpu_tracer_case.py OUTPATH
OUTPATH must not exist. Shared reference files are never modified. The WRF
executable and writable Thompson tables are copied; other tables and boundary
inputs are read-only links. NetCDF access uses the production NetCDF lock.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys


CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
TABLES = Path(os.environ.get(
    "TRACER_WRF_TABLE_DIR", "<USER_HOME>/p1_cpu_20260923/p0g2_r1/run"
))
TABLE_NAMES = ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat")
# Scalar names are not uppercased by WRF's ordinary state-variable machinery:
# Registry.EM -> scalar_indices.inc tracer_dname_table -> input_wrf.F ->
# ext_ncd_read_field's case-sensitive NF_INQ_VARID. Preserve quoted spelling.
TRACER_NAMES = tuple(f"tr17_{i}" for i in range(1, 9))


def locked_dataset():
    """Load the canonical host-only IO gate without gpuwrf's JAX bootstrap."""
    name = "gpuwrf.io.netcdf_lock"
    if name not in sys.modules:
        source = Path(__file__).resolve().parents[2] / "src/gpuwrf/io/netcdf_lock.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].Dataset


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_original_executable(path: Path) -> None:
    """Refuse known E104 dumps and embedded absolute dump/savepoint paths."""
    forbidden = (b"em_quarter_ss_center_dump", b"em_grav2d_x_touchdown_dump")
    absolute_dump = re.compile(rb"/(?:home|mnt|tmp)/[\w./-]*(?:dump|savepoint)[\w./-]*")
    tail = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            data = tail + block
            if any(marker in data for marker in forbidden) or absolute_dump.search(data):
                raise RuntimeError("instrumented WRF executable refused (E104 dump path)")
            tail = data[-4096:]


def edit(text: str, group: str, values: dict) -> str:
    match = re.search(rf"&{group}\b(.*?)/", text, flags=re.S | re.I)
    body = match.group(1) if match else "\n"
    for key, value in values.items():
        line = f" {key} = {value},"
        pattern = rf"^\s*{key}\s*=.*$"
        if re.search(pattern, body, flags=re.M | re.I):
            body = re.sub(pattern, line, body, flags=re.M | re.I)
        else:
            body += line + "\n"
    replacement = f"&{group}{body}/"
    return (text[:match.start()] + replacement + text[match.end():]
            if match else text + "\n" + replacement + "\n")


def prepare(root: Path) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing existing output path: {root}")
    executable = CPU / "wrf.exe"
    check_original_executable(executable)
    original_hash = file_hash(executable)
    source_input_hash = file_hash(CPU / "wrfinput_d01")
    source_boundary_hash = file_hash(CPU / "wrfbdy_d01")
    table_sources = {
        name: CPU / name if (CPU / name).is_file() else TABLES / name
        for name in TABLE_NAMES
    }
    for source in table_sources.values():
        if not source.is_file():
            raise FileNotFoundError(f"required writable Thompson table missing: {source}")
    Dataset = locked_dataset()
    # Check domain authority before creating any case files. RD11 is the
    # independent original CPU-WRF case; no alternate/synthetic case fallback.
    with Dataset(CPU / "wrfinput_d01") as source:
        dims = {name: len(source.dimensions[name]) for name in
                ("Time", "bottom_top", "south_north", "west_east")}
    if dims != {"Time": 1, "bottom_top": 44, "south_north": 42, "west_east": 42}:
        raise ValueError(f"unexpected RD11 mass dimensions: {dims}")
    root.mkdir(parents=True)
    for path in CPU.iterdir():
        if path.name.startswith(("wrfout", "wrfrst", "rsl.", "namelist", "wrf.exe", "wrfinput", "wrffdda")):
            continue
        if path.name in TABLE_NAMES:
            continue
        if path.is_file():
            (root / path.name).symlink_to(path.resolve())
    for name, source in table_sources.items():
        shutil.copyfile(source, root / name)
    shutil.copyfile(executable, root / "wrf.exe")
    (root / "wrf.exe").chmod(0o755)
    shutil.copyfile(CPU / "wrfinput_d01", root / "wrfinput_d01")
    with Dataset(root / "wrfinput_d01", "a") as dataset:
        dims_xyz = ("Time", "bottom_top", "south_north", "west_east")
        # Keep WRF's NC_INT FieldType encoding. A bare Python int becomes
        # NC_INT64 in NETCDF4 and is not the original io_netcdf ABI.
        field_type = dataset.variables["T"].getncattr("FieldType")
        if int(field_type) != 104:
            raise ValueError("original WRF REAL FieldType must be 104")
        # Literal real.exe initialization: ids=jds=1, ide=jde=43,
        # integer (ide+ids)/2=22; one-based 18:26 -> Python [17:26).
        for i, name in enumerate(TRACER_NAMES):
            variable = dataset.variables.get(name)
            if variable is None:
                variable = dataset.createVariable(name, "f4", dims_xyz)
            if variable.dimensions != dims_xyz or variable.dtype != "float32":
                raise ValueError(f"invalid existing tracer layout: {name}")
            variable[:] = 0.0
            if i < 4:
                variable[0, 0, 17:26, 17:26] = 1.0
            variable.setncatts({
                "FieldType": field_type, "MemoryOrder": "XYZ", "stagger": "",
                "units": "Dimensionless", "description": name,
                "coordinates": "XLONG XLAT XTIME",
            })
    text = (CPU / "namelist.input").read_text()
    text = edit(text, "time_control", {
        "run_days": 0, "run_hours": 0, "run_minutes": 0, "run_seconds": 54,
        "end_year": 2023, "end_month": 1, "end_day": 15, "end_hour": 0,
        "end_minute": 0, "end_second": 54, "history_interval": 0,
        "history_interval_s": 18, "history_interval_m": 0, "frames_per_outfile": 1,
        "restart": ".false.",
    })
    text = edit(text, "domains", {"time_step": 18})
    text = edit(text, "dynamics", {"tracer_opt": 2, "tracer_adv_opt": 1})
    text = edit(text, "physics", {"tracer_pblmix": 1})
    (root / "namelist.input").write_text(text)
    if file_hash(root / "wrf.exe") != original_hash:
        raise RuntimeError("copied executable is not byte-identical to original")
    if file_hash(CPU / "wrfinput_d01") != source_input_hash:
        raise RuntimeError("shared CPU-WRF input changed while staging")
    receipt = {
        "schema": "u2-tracers-original-cpu-case-v1",
        "source_case": str(CPU), "cpu_source": str(executable.resolve()),
        "executable_sha256": original_hash, "instrumented_dump_paths": False,
        "source_input_sha256": source_input_hash,
        "input_sha256": file_hash(root / "wrfinput_d01"),
        "boundary_sha256": source_boundary_hash,
        "namelist_sha256": file_hash(root / "namelist.input"),
        "steps": 3, "dt_s": 18, "history_interval_s": 18,
        "tracer_opt": 2, "tracer_adv_opt": 1, "tracer_pblmix": 1,
        "mass_shape": [44, 42, 42], "tracer_names": TRACER_NAMES,
        "io_name_reference": [
            "pristine Registry/Registry.EM:28-35",
            "pristine inc/scalar_indices.inc:19650-19748",
            "pristine share/input_wrf.F:1698-1750",
            "pristine external/io_netcdf/wrf_io.F90:2852-2880",
        ],
        "seed": {"species": TRACER_NAMES[:4], "value": 1,
                 "layer": 0, "x0": 17, "x1": 26, "y0": 17, "y1": 26,
                 "reference": "pristine dyn_em/module_initialize_real.F:4423-4438"},
        "thompson_table_sha256": {name: file_hash(root / name) for name in TABLE_NAMES},
        "mpi_launcher": "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/mpirun",
        "suggested_mpi_ranks": 4, "ran_wrf": False,
    }
    (root / "PREPARATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    result = prepare(Path(sys.argv[1]).resolve())
    print(json.dumps(result, indent=2))
