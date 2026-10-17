"""Launch the original WPS/real programs in an owned directory, then the CLI."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .forcing import MAX_DOWNLOAD_BYTES, ecmwf_report, field_report, inventory_grib, retrieve_gfs, sha256
from .region import namelists, plan_region, profiles


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("quickstart", help="plan or prepare a region with original WPS/real.exe",
                                  description="Plan a region or prepare it with original WPS/real.exe. See docs/onboarding.md. "
                                              "Planning/preparation use the CPU; --run starts the public forecast CLI.",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--resolution-km", type=float, default=9)
    parser.add_argument("--date", help="cycle YYYY-MM-DDTHH, UTC")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--vram-gib", type=int, choices=(12, 16, 24, 32), default=12)
    parser.add_argument("--profiles", action="store_true", help="print inferred memory profiles only")
    parser.add_argument("--case-dir", type=Path, help="fresh directory for prepared inputs and logs")
    parser.add_argument("--geog-dir", type=Path, default=os.environ.get("WPS_GEOG"))
    parser.add_argument("--wps-root", type=Path, default=os.environ.get("WPS_ROOT"))
    parser.add_argument("--wrf-root", type=Path, default=os.environ.get("GPUWRF_WRF_ROOT"))
    parser.add_argument("--source", choices=("gfs", "ifs", "aifs"), default="gfs")
    parser.add_argument("--audit-grib", type=Path, help="audit local GFS/IFS/AIFS fields at the initial time")
    parser.add_argument("--check-source", action="store_true", help="check live ECMWF IFS/AIFS open-data field availability")
    parser.add_argument("--prepare", action="store_true", help="download and run WPS/real; default only prints plan")
    parser.add_argument("--run", action="store_true", help="prepare then invoke gpuwrf run on the GPU")
    parser.add_argument("--max-download-mb", type=int, default=2000, help="total GFS transfer cap, at most 2000 MB")
    parser.add_argument("--stage-timeout", type=int, default=1800, help="seconds for each CPU preprocessor")
    parser.set_defaults(func=command)


def _exe(root: Path, name: str) -> Path:
    for candidate in (root / f"{name}.exe", root / "bin" / f"{name}.exe",
                      root / "bin" / name, root / "main" / f"{name}.exe"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ValueError(f"missing executable {name}.exe under {root}")


def _link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing upstream WRF/WPS asset: {source}")
    target.symlink_to(source.resolve())


def _stage(argv: list[str], directory: Path, name: str, timeout: int) -> None:
    print(f"quickstart: {name} (timeout {timeout}s)", file=sys.stderr, flush=True)
    # WPS and real are CPU programs; never inherit a CUDA platform setting here.
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
    with (directory / f"{name}.stage.log").open("w") as log:
        try:
            subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"{name} timed out; see {log.name}") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{name} exited {exc.returncode}; see {log.name}") from exc


def _metgrid_levels(directory: Path, plan: dict) -> int:
    from gpuwrf.io import netcdf_lock
    levels = set()
    for lead in plan["boundary_hours"]:
        from datetime import timedelta
        valid = datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S") + timedelta(hours=lead)
        path = directory / f"met_em.d01.{valid:%Y-%m-%d_%H:%M:%S}.nc"
        with netcdf_lock.Dataset(path) as ds:
            levels.add(len(ds.dimensions["num_metgrid_levels"]))
            if len(ds.dimensions["num_st_layers"]) != 4 or len(ds.dimensions["num_sm_layers"]) != 4:
                raise ValueError(f"metgrid must provide 4 temperature AND moisture soil layers: {path}")
    if len(levels) != 1:
        raise ValueError("metgrid vertical level count changes across forcing times")
    return levels.pop()


def check_real(directory: Path, plan: dict) -> dict:
    """Independent original-WRF output checks; rejects nonfinite input operands."""
    from gpuwrf.io import netcdf_lock
    import numpy as np
    records = []
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"real.exe did not produce {name}; inspect rsl.error.0000/real.stage.log")
        with netcdf_lock.Dataset(path) as ds:
            times = [b"".join(row).decode() for row in ds.variables["Times"][:]]
            for varname, var in ds.variables.items():
                if var.dtype.kind not in "f":
                    continue
                # Bound the scan to one time slice rather than materializing full wrfbdy.
                for index in range(var.shape[0]) if var.ndim else [None]:
                    values = var[index] if index is not None else var[...]
                    if np.ma.is_masked(values) or not np.isfinite(values).all():
                        raise ValueError(f"nonfinite/missing real.exe field {name}:{varname}")
            if name == "wrfinput_d01":
                if len(ds.dimensions["west_east"]) != plan["nx"] or len(ds.dimensions["south_north"]) != plan["ny"]:
                    raise ValueError("real.exe geometry does not match region plan")
                west, south, east, north = plan["bbox"]
                lat, lon = ds.variables["XLAT"][0], ds.variables["XLONG"][0]
                if not (float(lon.min()) <= west and float(lon.max()) >= east and
                        float(lat.min()) <= south and float(lat.max()) >= north):
                    raise ValueError("actual WRF grid does not cover the requested bbox")
                if times != [plan["start"]]:
                    raise ValueError("wrfinput time differs from requested cycle")
            else:
                expected = [datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S")]
                from datetime import timedelta
                expected = [(expected[0] + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")
                            for h in plan["boundary_hours"][:-1]]
                if times != expected:
                    raise ValueError(f"wrfbdy interval records mismatch: expected {expected}, found {times}")
            records.append(dict(path=str(path), sha256=sha256(path), times=times,
                                variables=len(ds.variables)))
    return dict(status="PASS", files=records)


def prepare(plan: dict, args) -> dict:
    if args.case_dir is None or args.geog_dir is None or args.wps_root is None or args.wrf_root is None:
        raise ValueError("--prepare requires --case-dir, --geog-dir, --wps-root and --wrf-root (or environment defaults)")
    if args.source != "gfs":
        raise ValueError("IFS/AIFS preparation is not qualified: use --audit-grib to report missing soil/surface fields; "
                         "no replacement fields or forcing-source mixing is performed. See docs/onboarding.md")
    if not 1 <= args.max_download_mb <= MAX_DOWNLOAD_BYTES // 1_000_000:
        raise ValueError("max-download-mb must be 1–2000")
    if not 1 <= args.stage_timeout <= 3600:
        raise ValueError("stage-timeout must be 1–3600 seconds")
    directory = args.case_dir.resolve()
    wps, wrf, geog = (path.resolve() for path in (args.wps_root, args.wrf_root, args.geog_dir))
    if not geog.is_dir():
        raise ValueError(f"missing geography directory: {geog}")
    executables = {name: _exe(wps if name != "real" else wrf, name)
                   for name in ("geogrid", "ungrib", "metgrid", "real")}
    # Refuse before any network/overwrite if a directory is already in use.
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan=plan, status="PREPARING", source="gfs", executables={k: str(v) for k, v in executables.items()})
    manifest_path = directory / "onboarding.json"
    try:
        wps_text, wrf_text = namelists(plan, geog)
        (directory / "namelist.wps").write_text(wps_text)
        (directory / "namelist.input").write_text(wrf_text)
        for name in ("geogrid", "metgrid"):
            (directory / name).mkdir()
            _link(wps / name / f"{name.upper()}.TBL.ARW", directory / name / f"{name.upper()}.TBL")
        _link(wps / "ungrib/Variable_Tables/Vtable.GFS", directory / "Vtable")
        for table in (wrf / "run").iterdir():
            if table.is_file() and table.name not in ("namelist.input", "real.exe", "wrf.exe", "ndown.exe", "tc.exe"):
                _link(table, directory / table.name)
        manifest["forcing"] = retrieve_gfs(plan, directory / "grib", args.max_download_mb * 1_000_000)
        for index, record in enumerate(manifest["forcing"]):
            letters = "".join(chr(65 + (index // 26**power) % 26) for power in (2, 1, 0))
            _link(Path(record["path"]), directory / f"GRIBFILE.{letters}")
        for name in ("geogrid", "ungrib", "metgrid"):
            _stage([str(executables[name])], directory, name, args.stage_timeout)
        levels = _metgrid_levels(directory, plan)
        (directory / "namelist.input").write_text(namelists(plan, geog, levels)[1])
        _stage([str(executables["real"])], directory, "real", args.stage_timeout)
        manifest["real_check"] = check_real(directory, plan)
        manifest["num_metgrid_levels"] = levels
        argv = [sys.executable, "-m", "gpuwrf.cli", "run", "--input-dir", str(directory),
                "--output-dir", str(directory / "forecast"), "--domain", "d01", "--hours", str(plan["hours"]),
                "--scratch-dir", str(directory / "scratch")]
        manifest["forecast_argv"] = argv
        _stage(argv + ["--dry-run"], directory, "gpuwrf-dry-run", min(args.stage_timeout, 120))
        manifest["status"] = "PREPARED_CPU_ONLY"
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def command(args) -> int:
    try:
        if args.profiles:
            print(json.dumps(profiles(), indent=2))
            return 0
        if args.bbox is None or args.date is None:
            raise ValueError("--bbox and --date are required (unless --profiles)")
        plan = plan_region(args.bbox, args.resolution_km, args.date, args.hours, args.vram_gib)
        plan["source"] = args.source
        if args.audit_grib:
            report = field_report(inventory_grib(args.audit_grib), datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S"))
            print(json.dumps(dict(source=args.source, field_report=report), indent=2))
            return 0 if report["status"] == "PASS" else 2
        if args.check_source or ((args.prepare or args.run) and args.source != "gfs"):
            if args.source == "gfs":
                raise ValueError("--check-source is for ifs/aifs; GFS fields are checked at every downloaded time")
            print(json.dumps(ecmwf_report(plan, args.source), indent=2))
            return 2
        result = prepare(plan, args) if args.prepare or args.run else plan
        if args.run:
            # Keep the original CLI in this process: the official lock fd stays
            # open, and there is no GPU child alongside a parent CUDA pool.
            # All preprocessing above uses only NumPy/netCDF and CPU children;
            # no JAX backend has been initialized by onboarding.
            from gpuwrf.cli import main as run_main
            os.environ["GPUWRF_WRF_ROOT"] = str(args.wrf_root.resolve())
            print("quickstart: CPU inputs prepared; starting gpuwrf run", file=sys.stderr, flush=True)
            return run_main(result["forecast_argv"][3:])
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"gpuwrf quickstart: {exc}", file=sys.stderr)
        return 2
