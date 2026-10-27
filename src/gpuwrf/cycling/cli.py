"""CLI registration stays backend-dark."""
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser("cycle", help="cycle same-grid land/snow or opt-in atmospheric state")
    parser.add_argument("--source", required=True, type=Path, help="previous wrfout or NetCDF wrfrst, at the new init time")
    parser.add_argument("--wrfinput", required=True, type=Path, help="new real.exe wrfinput template (preserved)")
    parser.add_argument("--output", required=True, type=Path, help="new cycled wrfinput; must not exist")
    parser.add_argument("--atmosphere", action="store_true", help="also cycle atmospheric state; requires new boundary output")
    parser.add_argument("--native-atmosphere", action="store_true", help="WRF native cycling: keep the source's resolved grid/base fields (preferred for nests)")
    parser.add_argument("--wrfbdy", type=Path, help="new real.exe wrfbdy_d01 template (preserved)")
    parser.add_argument("--boundary-output", type=Path, help="new wrfbdy_d01 with adjusted first interval; must not exist")
    parser.add_argument("--nest", nargs=3, action="append", default=[], type=Path,
                        metavar=("SOURCE", "WRFINPUT", "OUTPUT"), help="additional same-grid domain, in domain order")
    parser.set_defaults(func=command)


def command(args):
    import json
    import sys
    from .core import cycle
    try:
        report = cycle(args.source, args.wrfinput, args.output, atmosphere=args.atmosphere,
                       wrfbdy=args.wrfbdy, boundary_output=args.boundary_output, nests=args.nest,
                       native_atmosphere=args.native_atmosphere)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"gpuwrf cycle: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0
