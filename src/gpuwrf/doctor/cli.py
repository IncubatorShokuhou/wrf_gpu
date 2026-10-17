"""CLI handlers. All three commands are read-only except explicit JSON output."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def register(subparsers):
    doctor = subparsers.add_parser("doctor", help="inspect native case inputs, support, CFL and VRAM (offline)")
    doctor.add_argument("case_dir", type=Path)
    doctor.add_argument("--namelist", type=Path)
    doctor.add_argument("--budget", nargs="+", type=Path, metavar="WRFOUT", help="also report offline mass/moisture storage changes from one domain")
    doctor.set_defaults(func=command)
    manifest = subparsers.add_parser("manifest", help="hash native case inputs and record source/environment (offline)")
    manifest.add_argument("case_dir", type=Path)
    manifest.add_argument("--namelist", type=Path)
    manifest.add_argument("--output", type=Path, help="write a new JSON file; refuses to overwrite existing files")
    manifest.set_defaults(func=command)
    reproduce = subparsers.add_parser("reproduce", help="verify a manifest; dry-run only, never launches a forecast")
    reproduce.add_argument("manifest", type=Path)
    reproduce.add_argument("--case-dir", type=Path, help="verify relocated case-relative inputs")
    reproduce.set_defaults(func=command)


def command(args):
    try:
        if args.command == "doctor":
            from .case import inspect_case
            result = inspect_case(args.case_dir, args.namelist)
            if args.budget:
                from .budget import offline_budget
                result["budget"] = offline_budget(args.budget)
        elif args.command == "manifest":
            from .repro import create_manifest
            result = create_manifest(args.case_dir, args.namelist)
        else:
            from .repro import reproduce
            result = reproduce(args.manifest, args.case_dir)
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if args.command == "manifest" and args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        print(serialized)
        return 1 if result.get("status") == "FAIL" else 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"gpuwrf {args.command}: {exc}", file=sys.stderr)
        return 2
