"""Backend-dark registration for the offline field override tool."""
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser("override", help="edit checked private copies of per-nest WRF inputs")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; must not exist")
    parser.add_argument("--spec", required=True, type=Path, help="JSON/YAML overrides and explicit check waivers")
    parser.add_argument("--surface-physics", type=int, default=4, help="WRF sf_surface_physics (default Noah-MP=4)")
    parser.add_argument("--force-no-effect", action="store_true", help="allow EMISS/ALBBCK/ZNT storage edits despite Noah-MP recomputation")
    parser.set_defaults(func=command)


def command(args):
    import json
    import sys
    from gpuwrf.io.field_override import load_spec, override_inputs
    from gpuwrf.io.wrfinput_consistency import WrfinputConsistencyError
    try:
        spec, base = load_spec(args.spec)
        report = override_inputs(args.input_dir, args.output_dir, spec, spec_base=base,
            surface_physics=args.surface_physics, force_no_effect=args.force_no_effect)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        if isinstance(exc, WrfinputConsistencyError):
            print(json.dumps({"error": str(exc), "validation": exc.report}, sort_keys=True), file=sys.stderr)
        else:
            print(f"gpuwrf override: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0
