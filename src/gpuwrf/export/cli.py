"""Deferred export implementation: optional dependencies stay out of run."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def register(subparsers):
    parser = subparsers.add_parser("export", help="offline CF NetCDF/Zarr or GRIB2 surface/isobaric export")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("netcdf", "zarr", "grib2"), default="netcdf")
    parser.add_argument("--variables", nargs="+", required=True, help="WRF names: T2 PSFC Q2 U10 V10 HGT T QVAPOR U V PH")
    parser.add_argument("--raw", action="store_true", help="crop arbitrary WRF variables preserving native dimensions/attrs; NetCDF/Zarr only, no CF claim")
    parser.add_argument("--pressure-levels", nargs="+", type=float, help="hPa, required for profile fields; no extrapolation")
    parser.add_argument("--region", nargs=4, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    parser.add_argument("--chunk", type=int, default=128, help="Zarr horizontal chunk edge; time/pressure chunk=1")
    parser.add_argument("--grib-grid", nargs=6, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH", "NX", "NY"), help="explicit regular lat/lon GRIB2 target; linear interpolation, no extrapolation")
    parser.set_defaults(func=command)


def command(args):
    try:
        from .core import load, raw_subset
        from .writers import write
        if args.output.resolve() in [p.resolve() for p in args.input]:
            raise ValueError("output must differ from input")
        if args.grib_grid is not None and args.format != "grib2":
            raise ValueError("--grib-grid requires --format grib2")
        if args.raw:
            if args.format == "grib2" or args.pressure_levels:
                raise ValueError("--raw requires NetCDF/Zarr and no pressure levels")
            result = raw_subset(args.input, args.variables, args.region)
        else:
            result = load(args.input, args.variables, args.pressure_levels, args.region)
        write(result, args.output, args.format, args.chunk, args.grib_grid)
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        print(f"gpuwrf export: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "format": args.format, "sizes": dict(result.sizes),
                      "variables": [n for n in result.data_vars if n != "crs"]}, sort_keys=True))
    return 0
