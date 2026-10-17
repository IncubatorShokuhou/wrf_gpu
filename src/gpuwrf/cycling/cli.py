"""CLI registration stays backend-dark."""
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser("cycle", help="cycle same-grid land/snow state into a new wrfinput")
    parser.add_argument("--source", required=True, type=Path, help="previous wrfout or NetCDF wrfrst, at the new init time")
    parser.add_argument("--wrfinput", required=True, type=Path, help="new real.exe wrfinput template (preserved)")
    parser.add_argument("--output", required=True, type=Path, help="new cycled wrfinput; must not exist")
    parser.set_defaults(func=command)


def command(args):
    import json
    import sys
    from .core import cycle
    try:
        report = cycle(args.source, args.wrfinput, args.output)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"gpuwrf cycle: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0
