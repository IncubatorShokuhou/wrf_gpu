"""Command registration stays independent of NumPy, NetCDF and JAX."""
from pathlib import Path
import json
import sys


def register(subparsers):
    parser = subparsers.add_parser("perturb", help="prepare coherent wind IC/BC ensemble perturbations (offline)")
    parser.add_argument("--case", type=Path, required=True, help="WRF input case directory")
    parser.add_argument("--output", type=Path, required=True, help="new member case directory")
    parser.add_argument("--members", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wind-std", type=float, required=True,
                        help="sample standard deviation of a spatially uniform fractional U/V multiplier")
    parser.add_argument("--ic-only", action="store_true", help="explicitly leave lateral boundaries unchanged")
    parser.set_defaults(func=command)


def command(args):
    from .inputs import prepare
    try:
        result = prepare(args.case, args.output, members=args.members, seed=args.seed,
                         wind_std=args.wind_std, ic_only=args.ic_only)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"gpuwrf perturb: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0
